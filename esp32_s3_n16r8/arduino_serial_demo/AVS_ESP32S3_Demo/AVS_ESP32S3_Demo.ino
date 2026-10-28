#include <Arduino.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "avs_pipeline.h"
#include "w_standard_data.h"

namespace {
constexpr size_t kWindowFrames = 3u * AVS_SAMPLE_RATE / AVS_HOP_LENGTH;
constexpr size_t kMaxBlockSamples = 512;
constexpr unsigned kNnlsIterations = 512;
constexpr float kNnlsTolerance = 1.0e-6f;
constexpr size_t kMaxBenchFrames = 256;
constexpr size_t kBenchWarmupFrames = 8;
constexpr double kTwoPi = 6.2831853071795864769;
constexpr size_t kStageCount = 6;

enum bench_stage_t : size_t {
    kStagePreprocess = 0,
    kStageFft,
    kStageNnls,
    kStageEventSum,
    kStageEnvelopeUpdate,
    kStagePeakDetection,
};

struct stage_memory_profile_t {
    uint32_t internal_start_free = 0;
    uint32_t internal_last_free = 0;
    uint32_t internal_min_free = UINT32_MAX;
    uint32_t internal_peak_extra = 0;
    uint32_t internal_largest_start = 0;
    uint32_t internal_largest_last = 0;
    uint32_t psram_start_free = 0;
    uint32_t psram_last_free = 0;
    uint32_t psram_min_free = UINT32_MAX;
    uint32_t psram_peak_extra = 0;
    uint32_t psram_largest_start = 0;
    uint32_t psram_largest_last = 0;
    uint32_t iram_start_free = 0;
    uint32_t iram_last_free = 0;
    uint32_t iram_min_free = UINT32_MAX;
    uint32_t iram_peak_extra = 0;
    uint32_t iram_largest_start = 0;
    uint32_t iram_largest_last = 0;
    uint32_t stack_hwm_start = 0;
    uint32_t stack_hwm_min_after = UINT32_MAX;
    uint32_t stack_new_peak = 0;
};

struct bench_stage_profile_t {
    uint32_t elapsed_us[kMaxBenchFrames]{};
    uint32_t cycles[kMaxBenchFrames]{};
    stage_memory_profile_t memory{};
};

struct stage_measurement_t {
    int64_t start_us = 0;
    esp_cpu_cycle_count_t start_cycles = 0;
    uint32_t internal_free_before = 0;
    uint32_t internal_largest_before = 0;
    uint32_t psram_free_before = 0;
    uint32_t psram_largest_before = 0;
    uint32_t iram_free_before = 0;
    uint32_t iram_largest_before = 0;
    uint32_t stack_hwm_before = 0;
    bool local_heap_monitor_started = false;
};

constexpr const char *kStageNames[kStageCount] = {
    "preprocess", "fft_spectrum", "nnls", "event_sum",
    "envelope_update", "peak_detection_3s"
};

static_assert(AVS_W_ROWS == AVS_NUM_BINS, "Dictionary row count mismatch");
static_assert(AVS_W_COLS == AVS_NUM_COMPONENTS, "Dictionary column count mismatch");

avs_fft_workspace_t g_fft{};
avs_spectrum_t g_spectrum{};
avs_nnls_workspace_t g_nnls{};
float g_window[AVS_N_FFT]{};
float g_frame[AVS_N_FFT]{};
float g_activation[AVS_NUM_COMPONENTS]{};
float g_envelope[kWindowFrames]{};
uint32_t g_bench_latency_us[kMaxBenchFrames]{};
uint32_t g_bench_latency_cycles[kMaxBenchFrames]{};
bench_stage_profile_t g_bench_stages[kStageCount]{};
uint8_t g_rx_bytes[kMaxBlockSamples * sizeof(float)]{};
size_t g_envelope_count = 0;
size_t g_peak_indices[kWindowFrames]{};
size_t g_frame_fill = 0;
uint64_t g_samples_seen = 0;
uint64_t g_next_report_sample = 0;
uint64_t g_expected_samples = 0;
uint32_t g_frames_processed = 0;
size_t g_profile_frame_count = 0;
size_t g_profile_peak_count = 0;
bool g_ready = false;
bool g_stream_active = false;
bool g_profile_stream_active = false;

void begin_stage_measurement(stage_measurement_t &measurement) {
    constexpr uint32_t internal_caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    constexpr uint32_t psram_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    constexpr uint32_t iram_caps = MALLOC_CAP_EXEC;
    measurement.internal_free_before = (uint32_t)heap_caps_get_free_size(internal_caps);
    measurement.internal_largest_before = (uint32_t)heap_caps_get_largest_free_block(internal_caps);
    measurement.psram_free_before = (uint32_t)heap_caps_get_free_size(psram_caps);
    measurement.psram_largest_before = (uint32_t)heap_caps_get_largest_free_block(psram_caps);
    measurement.iram_free_before = (uint32_t)heap_caps_get_free_size(iram_caps);
    measurement.iram_largest_before = (uint32_t)heap_caps_get_largest_free_block(iram_caps);
    measurement.stack_hwm_before =
        (uint32_t)uxTaskGetStackHighWaterMark(nullptr) * sizeof(StackType_t);
    measurement.local_heap_monitor_started =
        heap_caps_monitor_local_minimum_free_size_start() == ESP_OK;
    measurement.start_cycles = esp_cpu_get_cycle_count();
    measurement.start_us = esp_timer_get_time();
}

void end_stage_measurement(bench_stage_t stage, size_t sample_index,
                           const stage_measurement_t &measurement) {
    const esp_cpu_cycle_count_t end_cycles = esp_cpu_get_cycle_count();
    const int64_t end_us = esp_timer_get_time();
    bench_stage_profile_t &profile = g_bench_stages[(size_t)stage];
    profile.cycles[sample_index] = (uint32_t)(end_cycles - measurement.start_cycles);
    profile.elapsed_us[sample_index] = (uint32_t)(end_us - measurement.start_us);

    constexpr uint32_t internal_caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    constexpr uint32_t psram_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    constexpr uint32_t iram_caps = MALLOC_CAP_EXEC;
    const uint32_t internal_after = (uint32_t)heap_caps_get_free_size(internal_caps);
    const uint32_t psram_after = (uint32_t)heap_caps_get_free_size(psram_caps);
    const uint32_t iram_after = (uint32_t)heap_caps_get_free_size(iram_caps);
    const uint32_t stack_hwm_after =
        (uint32_t)uxTaskGetStackHighWaterMark(nullptr) * sizeof(StackType_t);
    const uint32_t internal_min = measurement.local_heap_monitor_started
        ? (uint32_t)heap_caps_get_minimum_free_size(internal_caps) : internal_after;
    const uint32_t psram_min = measurement.local_heap_monitor_started
        ? (uint32_t)heap_caps_get_minimum_free_size(psram_caps) : psram_after;
    const uint32_t iram_min = measurement.local_heap_monitor_started
        ? (uint32_t)heap_caps_get_minimum_free_size(iram_caps) : iram_after;
    if (measurement.local_heap_monitor_started) {
        heap_caps_monitor_local_minimum_free_size_stop();
    }

    stage_memory_profile_t &memory = profile.memory;
    if (sample_index == 0) {
        memory.internal_start_free = measurement.internal_free_before;
        memory.internal_largest_start = measurement.internal_largest_before;
        memory.psram_start_free = measurement.psram_free_before;
        memory.psram_largest_start = measurement.psram_largest_before;
        memory.iram_start_free = measurement.iram_free_before;
        memory.iram_largest_start = measurement.iram_largest_before;
        memory.stack_hwm_start = measurement.stack_hwm_before;
    }
    memory.internal_last_free = internal_after;
    memory.internal_largest_last = (uint32_t)heap_caps_get_largest_free_block(internal_caps);
    memory.internal_min_free = std::min(memory.internal_min_free, internal_min);
    memory.internal_peak_extra = std::max(memory.internal_peak_extra,
        measurement.internal_free_before > internal_min
            ? measurement.internal_free_before - internal_min : 0u);
    memory.psram_last_free = psram_after;
    memory.psram_largest_last = (uint32_t)heap_caps_get_largest_free_block(psram_caps);
    memory.psram_min_free = std::min(memory.psram_min_free, psram_min);
    memory.psram_peak_extra = std::max(memory.psram_peak_extra,
        measurement.psram_free_before > psram_min
            ? measurement.psram_free_before - psram_min : 0u);
    memory.iram_last_free = iram_after;
    memory.iram_largest_last = (uint32_t)heap_caps_get_largest_free_block(iram_caps);
    memory.iram_min_free = std::min(memory.iram_min_free, iram_min);
    memory.iram_peak_extra = std::max(memory.iram_peak_extra,
        measurement.iram_free_before > iram_min
            ? measurement.iram_free_before - iram_min : 0u);
    memory.stack_hwm_min_after = std::min(memory.stack_hwm_min_after, stack_hwm_after);
    memory.stack_new_peak = std::max(memory.stack_new_peak,
        measurement.stack_hwm_before > stack_hwm_after
            ? measurement.stack_hwm_before - stack_hwm_after : 0u);
}

avs_status_t run_dsp_frame(float *frame, float *event_sum, int profile_index = -1) {
    stage_measurement_t measurement{};
    if (profile_index >= 0) begin_stage_measurement(measurement);
    float mean = 0.0f;
    for (size_t i = 0; i < AVS_N_FFT; ++i) mean += frame[i];
    mean /= AVS_N_FFT;
    for (size_t i = 0; i < AVS_N_FFT; ++i) frame[i] -= mean;
    if (profile_index >= 0) {
        end_stage_measurement(kStagePreprocess, (size_t)profile_index, measurement);
        begin_stage_measurement(measurement);
    }

    const avs_status_t fft_status = avs_compute_spectrum(frame, g_window, &g_fft, &g_spectrum);
    if (profile_index >= 0) {
        end_stage_measurement(kStageFft, (size_t)profile_index, measurement);
    }
    if (fft_status != AVS_OK) return fft_status;
    if (profile_index >= 0) begin_stage_measurement(measurement);
    const avs_status_t nnls_status = avs_solve_nnls(
        g_spectrum.log_magnitude, AVS_W_STANDARD, &g_nnls, g_activation,
        kNnlsIterations, kNnlsTolerance);
    if (profile_index >= 0) {
        end_stage_measurement(kStageNnls, (size_t)profile_index, measurement);
    }
    if (nnls_status != AVS_OK) return nnls_status;
    if (profile_index >= 0) begin_stage_measurement(measurement);
    *event_sum = avs_sum_event_activations(g_activation, AVS_NUM_COMPONENTS);
    if (profile_index >= 0) {
        end_stage_measurement(kStageEventSum, (size_t)profile_index, measurement);
    }
    return AVS_OK;
}

void update_event_envelope(float event_sum) {
    if (g_envelope_count < kWindowFrames) {
        g_envelope[g_envelope_count++] = event_sum;
    } else {
        memmove(g_envelope, g_envelope + 1,
                (kWindowFrames - 1) * sizeof(g_envelope[0]));
        g_envelope[kWindowFrames - 1] = event_sum;
    }
}

size_t detect_event_peaks() {
    return avs_detect_peaks(g_envelope, g_envelope_count, AVS_PEAK_PROMINENCE,
                            AVS_PEAK_DISTANCE, g_peak_indices, kWindowFrames);
}

void profile_envelope_update(size_t sample_index, float event_sum);
size_t profile_peak_detection(size_t sample_index);
void record_total_frame(size_t sample_index);

void reset_stream() {
    memset(g_frame, 0, sizeof(g_frame));
    memset(g_envelope, 0, sizeof(g_envelope));
    g_envelope_count = 0;
    g_frame_fill = 0;
    g_samples_seen = 0;
    g_next_report_sample = 0;
    g_frames_processed = 0;
}

void process_frame() {
    float he = 0.0f;
    const int profile_index = g_profile_stream_active &&
        g_profile_frame_count < kMaxBenchFrames ? (int)g_profile_frame_count : -1;
    const avs_status_t status = run_dsp_frame(g_frame, &he, profile_index);
    if (status != AVS_OK) {
        Serial.printf("WARN NNLS status=%d\r\n", (int)status);
        return;
    }
    ++g_frames_processed;
    const size_t frame_index = g_frames_processed - 1;
    const double event_time =
        (double)(AVS_N_FFT / 2 + frame_index * AVS_HOP_LENGTH) / AVS_SAMPLE_RATE;
    if (!g_profile_stream_active) {
        Serial.printf("H_EVENT t=%.6f sum=%.8f\r\n", event_time, (double)he);
    }
    if (profile_index >= 0) {
        profile_envelope_update((size_t)profile_index, he);
        record_total_frame((size_t)profile_index);
        ++g_profile_frame_count;
    } else {
        update_event_envelope(he);
    }

    if (g_envelope_count == kWindowFrames &&
        (g_next_report_sample == 0 || g_samples_seen >= g_next_report_sample)) {
        size_t peaks;
        if (g_profile_stream_active && g_profile_peak_count < kMaxBenchFrames) {
            peaks = profile_peak_detection(g_profile_peak_count++);
        } else {
            peaks = detect_event_peaks();
        }
        if (peaks == AVS_PEAKS_OVERFLOW) {
            Serial.println(F("ERROR PEAK_BUFFER_OVERFLOW"));
        } else if (!g_profile_stream_active) {
            Serial.printf("STATUS t=%.2f count_3s=%u feed_rate_parts_s=%.3f H_event_sum=%.8f peak_times_s=",
                (double)g_samples_seen / AVS_SAMPLE_RATE,
                (unsigned)peaks, (double)peaks / 3.0, (double)he);
            const size_t first_frame = g_frames_processed - g_envelope_count;
            for (size_t i = 0; i < peaks; ++i) {
                const size_t peak_frame = first_frame + g_peak_indices[i];
                const double peak_time =
                    (double)(AVS_N_FFT / 2 + peak_frame * AVS_HOP_LENGTH) / AVS_SAMPLE_RATE;
                if (i != 0) Serial.print(',');
                Serial.printf("%.6f", peak_time);
            }
            if (peaks == 0) Serial.print('-');
            Serial.println();
        }
        g_next_report_sample = g_samples_seen + AVS_SAMPLE_RATE;
    }
}

void fill_benchmark_frame(size_t frame_number) {
    const double frame_offset =
        (double)(frame_number * AVS_HOP_LENGTH) / AVS_SAMPLE_RATE;
    for (size_t i = 0; i < AVS_N_FFT; ++i) {
        const double t = frame_offset + (double)i / AVS_SAMPLE_RATE;
        g_frame[i] = (float)(0.35 * sin(kTwoPi * 440.0 * t) +
                             0.12 * sin(kTwoPi * 1320.0 * t));
    }
}

void reset_benchmark_profiles() {
    memset(g_bench_latency_us, 0, sizeof(g_bench_latency_us));
    memset(g_bench_latency_cycles, 0, sizeof(g_bench_latency_cycles));
    memset(g_bench_stages, 0, sizeof(g_bench_stages));
    g_profile_frame_count = 0;
    g_profile_peak_count = 0;
    for (size_t i = 0; i < kStageCount; ++i) {
        g_bench_stages[i].memory.internal_min_free = UINT32_MAX;
        g_bench_stages[i].memory.psram_min_free = UINT32_MAX;
        g_bench_stages[i].memory.iram_min_free = UINT32_MAX;
        g_bench_stages[i].memory.stack_hwm_min_after = UINT32_MAX;
    }
}

void profile_envelope_update(size_t sample_index, float event_sum) {
    stage_measurement_t measurement{};
    begin_stage_measurement(measurement);
    update_event_envelope(event_sum);
    end_stage_measurement(kStageEnvelopeUpdate, sample_index, measurement);
}

size_t profile_peak_detection(size_t sample_index) {
    stage_measurement_t measurement{};
    begin_stage_measurement(measurement);
    const size_t peaks = detect_event_peaks();
    end_stage_measurement(kStagePeakDetection, sample_index, measurement);
    return peaks;
}

void summarize_stage(bench_stage_t stage, size_t sample_count) {
    bench_stage_profile_t &profile = g_bench_stages[(size_t)stage];
    if (sample_count == 0) {
        Serial.printf("BENCH_STAGE name=%s samples=0 status=NOT_MEASURED\r\n",
            kStageNames[(size_t)stage]);
        return;
    }
    uint64_t sum_us = 0;
    uint64_t sum_cycles = 0;
    for (size_t i = 0; i < sample_count; ++i) {
        sum_us += profile.elapsed_us[i];
        sum_cycles += profile.cycles[i];
    }
    std::sort(profile.elapsed_us, profile.elapsed_us + sample_count);
    std::sort(profile.cycles, profile.cycles + sample_count);
    const size_t p95_index = ((sample_count * 95 + 99) / 100) - 1;
    const double average_us = (double)sum_us / sample_count;
    const double average_cycles = (double)sum_cycles / sample_count;
    Serial.printf("BENCH_STAGE name=%s samples=%u min_us=%u median_us=%u average_us=%.2f p95_us=%u max_us=%u average_cycles=%.0f p95_cycles=%u max_cycles=%u\r\n",
        kStageNames[(size_t)stage], (unsigned)sample_count,
        (unsigned)profile.elapsed_us[0],
        (unsigned)profile.elapsed_us[(sample_count - 1) / 2], average_us,
        (unsigned)profile.elapsed_us[p95_index],
        (unsigned)profile.elapsed_us[sample_count - 1], average_cycles,
        (unsigned)profile.cycles[p95_index], (unsigned)profile.cycles[sample_count - 1]);

    const stage_memory_profile_t &m = profile.memory;
    Serial.printf("BENCH_STAGE_MEM name=%s internal_free_start=%u internal_min_free=%u internal_free_end=%u internal_peak_extra=%u internal_largest_start=%u internal_largest_end=%u psram_free_start=%u psram_min_free=%u psram_free_end=%u psram_peak_extra=%u psram_largest_start=%u psram_largest_end=%u iram_free_start=%u iram_min_free=%u iram_free_end=%u iram_peak_extra=%u iram_largest_start=%u iram_largest_end=%u stack_hwm_start=%u stack_hwm_min_end=%u stack_new_peak=%u flash_dynamic=0\r\n",
        kStageNames[(size_t)stage],
        (unsigned)m.internal_start_free, (unsigned)m.internal_min_free,
        (unsigned)m.internal_last_free, (unsigned)m.internal_peak_extra,
        (unsigned)m.internal_largest_start, (unsigned)m.internal_largest_last,
        (unsigned)m.psram_start_free, (unsigned)m.psram_min_free,
        (unsigned)m.psram_last_free, (unsigned)m.psram_peak_extra,
        (unsigned)m.psram_largest_start, (unsigned)m.psram_largest_last,
        (unsigned)m.iram_start_free, (unsigned)m.iram_min_free,
        (unsigned)m.iram_last_free, (unsigned)m.iram_peak_extra,
        (unsigned)m.iram_largest_start, (unsigned)m.iram_largest_last,
        (unsigned)m.stack_hwm_start, (unsigned)m.stack_hwm_min_after,
        (unsigned)m.stack_new_peak);
}

void record_total_frame(size_t sample_index) {
    uint32_t frame_us = 0;
    uint32_t frame_cycles = 0;
    for (size_t stage = kStagePreprocess; stage <= kStageEnvelopeUpdate; ++stage) {
        frame_us += g_bench_stages[stage].elapsed_us[sample_index];
        frame_cycles += g_bench_stages[stage].cycles[sample_index];
    }
    g_bench_latency_us[sample_index] = frame_us;
    g_bench_latency_cycles[sample_index] = frame_cycles;
}

void print_benchmark_results(size_t frame_count, size_t peak_count,
                             size_t warmup_frames, const char *input_kind) {
    if (frame_count == 0) {
        Serial.println(F("BENCH_ERROR NO_PROFILED_FRAMES"));
        return;
    }
    uint64_t total_us = 0;
    uint64_t total_cycles = 0;
    for (size_t i = 0; i < frame_count; ++i) {
        total_us += g_bench_latency_us[i];
        total_cycles += g_bench_latency_cycles[i];
    }
    std::sort(g_bench_latency_us, g_bench_latency_us + frame_count);
    std::sort(g_bench_latency_cycles, g_bench_latency_cycles + frame_count);
    const size_t p95_index = ((frame_count * 95 + 99) / 100) - 1;
    const double average_frame_us = (double)total_us / frame_count;
    const double average_frame_cycles = (double)total_cycles / frame_count;
    const double deadline_us = 1000000.0 * AVS_HOP_LENGTH / AVS_SAMPLE_RATE;
    const double frame_rate = (double)AVS_SAMPLE_RATE / AVS_HOP_LENGTH;
    uint64_t peak_sum_us = 0;
    for (size_t i = 0; i < peak_count; ++i) {
        peak_sum_us += g_bench_stages[kStagePeakDetection].elapsed_us[i];
    }
    const double peak_average_us = peak_count ? (double)peak_sum_us / peak_count : 0.0;
    const double compute_per_audio_second_us = average_frame_us * frame_rate + peak_average_us;
    const double real_time_capacity = 1000000.0 / compute_per_audio_second_us;

    const uint32_t internal_caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    const uint32_t psram_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    const uint32_t iram_caps = MALLOC_CAP_EXEC;
    Serial.printf("BENCH_CHIP model=%s revision=%u cores=%u cpu_mhz=%u measured_core=%u\r\n",
        ESP.getChipModel(), (unsigned)ESP.getChipRevision(), (unsigned)ESP.getChipCores(),
        (unsigned)ESP.getCpuFreqMHz(), (unsigned)xPortGetCoreID());
    Serial.printf("BENCH_FRAME fft_samples=%u hop_samples=%u sample_rate=%u deadline_us=%.2f measured_frames=%u warmup_frames=%u input_kind=%s\r\n",
        (unsigned)AVS_N_FFT, (unsigned)AVS_HOP_LENGTH, (unsigned)AVS_SAMPLE_RATE,
        deadline_us, (unsigned)frame_count, (unsigned)warmup_frames, input_kind);
    Serial.printf("BENCH_TOTAL_FRAME average_us=%.2f p95_us=%u average_cycles=%.0f p95_cycles=%u avg_frame_capacity_x=%.3f\r\n",
        average_frame_us, (unsigned)g_bench_latency_us[p95_index],
        average_frame_cycles, (unsigned)g_bench_latency_cycles[p95_index],
        deadline_us / average_frame_us);
    Serial.printf("BENCH_REALTIME frame_rate_hz=%.3f peak_runs_measured=%u peak_average_us=%.2f total_compute_us_per_audio_second=%.2f capacity_x=%.3f meets_realtime=%s\r\n",
        frame_rate, (unsigned)peak_count, peak_average_us, compute_per_audio_second_us,
        real_time_capacity, real_time_capacity >= 1.0 ? "YES" : "NO");

    for (size_t stage = 0; stage < kStageCount; ++stage) {
        const size_t stage_samples = stage == kStagePeakDetection ? peak_count : frame_count;
        summarize_stage((bench_stage_t)stage, stage_samples);
    }
    Serial.printf("BENCH_MEMORY_GLOBAL internal_total=%u internal_free=%u internal_min_free_since_boot=%u psram_total=%u psram_free=%u iram_exec_total=%u iram_exec_free=%u\r\n",
        (unsigned)heap_caps_get_total_size(internal_caps),
        (unsigned)heap_caps_get_free_size(internal_caps),
        (unsigned)heap_caps_get_minimum_free_size(internal_caps),
        (unsigned)heap_caps_get_total_size(psram_caps),
        (unsigned)heap_caps_get_free_size(psram_caps),
        (unsigned)heap_caps_get_total_size(iram_caps),
        (unsigned)heap_caps_get_free_size(iram_caps));
    Serial.printf("BENCH_FLASH_GLOBAL chip_size=%u chip_speed_hz=%u sketch_size=%u app_free_space=%u stage_dynamic_alloc=0\r\n",
        (unsigned)ESP.getFlashChipSize(), (unsigned)ESP.getFlashChipSpeed(),
        (unsigned)ESP.getSketchSize(), (unsigned)ESP.getFreeSketchSpace());
    Serial.println(F("BENCH_DONE"));
}

void run_hardware_benchmark(size_t frame_count) {
    reset_benchmark_profiles();
    g_envelope_count = 0;
    float event_sum = 0.0f;
    for (size_t i = 0; i < kBenchWarmupFrames; ++i) {
        fill_benchmark_frame(i);
        if (run_dsp_frame(g_frame, &event_sum) != AVS_OK) {
            Serial.println(F("BENCH_ERROR DSP_WARMUP"));
            return;
        }
        update_event_envelope(event_sum);
    }

    for (size_t i = 0; i < frame_count; ++i) {
        fill_benchmark_frame(i + kBenchWarmupFrames);
        const avs_status_t status = run_dsp_frame(g_frame, &event_sum, (int)i);
        if (status != AVS_OK) {
            Serial.printf("BENCH_ERROR DSP_STATUS=%d\r\n", (int)status);
            return;
        }
        profile_envelope_update(i, event_sum);
        record_total_frame(i);
    }

    // Peak detection is performed once per second on the current 3-second window.
    g_envelope_count = kWindowFrames;
    for (size_t i = 0; i < kWindowFrames; ++i) {
        const float baseline = 0.05f + 0.01f * (float)(i % 7);
        g_envelope[i] = (i % 24 == 12) ? baseline + 0.4f : baseline;
    }
    for (size_t i = 0; i < frame_count; ++i) profile_peak_detection(i);

    print_benchmark_results(frame_count, frame_count, kBenchWarmupFrames, "synthetic");
}

void feed_sample(float sample) {
    g_frame[g_frame_fill++] = sample;
    ++g_samples_seen;
    if (g_frame_fill == AVS_N_FFT) {
        process_frame();
        memmove(g_frame, g_frame + AVS_HOP_LENGTH,
                (AVS_N_FFT - AVS_HOP_LENGTH) * sizeof(g_frame[0]));
        g_frame_fill = AVS_N_FFT - AVS_HOP_LENGTH;
    }
}

bool read_exact(uint8_t *dst, size_t length) {
    size_t received = 0;
    while (received < length) {
        const size_t n = Serial.readBytes(dst + received, length - received);
        if (n == 0) return false;
        received += n;
    }
    return true;
}

void handle_command(String command) {
    command.trim();
    if (command == "PING") {
        Serial.println(F("PONG"));
        return;
    }

    if (command.startsWith("BENCH_WAV ")) {
        const uint64_t samples = strtoull(command.c_str() + 10, nullptr, 10);
        const size_t expected_frames = samples < AVS_N_FFT ? 0 :
            1 + (size_t)(samples - AVS_N_FFT) / AVS_HOP_LENGTH;
        if (!g_ready || g_stream_active || expected_frames == 0 ||
            expected_frames > kMaxBenchFrames) {
            Serial.printf("BENCH_ERROR WAV_LENGTH_OR_BOARD_STATE max_frames=%u\r\n",
                (unsigned)kMaxBenchFrames);
            return;
        }
        reset_stream();
        reset_benchmark_profiles();
        g_expected_samples = samples;
        g_profile_stream_active = true;
        g_stream_active = true;
        Serial.printf("ACK_BENCH_START samples=%llu expected_frames=%u\r\n",
            (unsigned long long)g_expected_samples, (unsigned)expected_frames);
        return;
    }

    if (command == "BENCH" || command.startsWith("BENCH ")) {
        if (!g_ready || g_stream_active) {
            Serial.println(F("BENCH_ERROR BOARD_NOT_IDLE"));
            return;
        }
        const long requested = command == "BENCH" ? 100 : command.substring(6).toInt();
        if (requested < 1 || requested > (long)kMaxBenchFrames) {
            Serial.printf("BENCH_ERROR FRAME_COUNT_RANGE max=%u\r\n",
                (unsigned)kMaxBenchFrames);
            return;
        }
        run_hardware_benchmark((size_t)requested);
        return;
    }

    if (command.startsWith("START ")) {
        const uint64_t samples = strtoull(command.c_str() + 6, nullptr, 10);
        if (!g_ready || samples == 0) {
            Serial.println(F("ERROR START_INVALID"));
            return;
        }
        reset_stream();
        g_expected_samples = samples;
        g_stream_active = true;
        Serial.printf("ACK_START %llu\r\n", (unsigned long long)g_expected_samples);
        g_profile_stream_active = false;
        return;
    }

    if (command.startsWith("DATA ")) {
        if (!g_stream_active) {
            Serial.println(F("ERROR NO_ACTIVE_STREAM"));
            return;
        }
        const size_t count = (size_t)command.substring(5).toInt();
        if (count == 0 || count > kMaxBlockSamples ||
            g_samples_seen + count > g_expected_samples) {
            Serial.println(F("ERROR DATA_BLOCK_SIZE"));
            return;
        }
        const size_t byte_count = count * sizeof(float);
        if (!read_exact(g_rx_bytes, byte_count)) {
            Serial.println(F("ERROR DATA_TIMEOUT"));
            g_stream_active = false;
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            float sample;
            memcpy(&sample, g_rx_bytes + i * sizeof(float), sizeof(sample));
            if (!std::isfinite(sample)) {
                Serial.println(F("ERROR NONFINITE_SAMPLE"));
                g_stream_active = false;
                return;
            }
            feed_sample(sample);
        }
        Serial.printf("ACK %llu\r\n", (unsigned long long)g_samples_seen);
        return;
    }

    if (command == "END") {
        if (!g_stream_active || g_samples_seen != g_expected_samples) {
            Serial.printf("ERROR END_COUNT expected=%llu received=%llu\r\n",
                (unsigned long long)g_expected_samples,
                (unsigned long long)g_samples_seen);
            g_stream_active = false;
            g_profile_stream_active = false;
            return;
        }
        size_t final_count = 0;
        if (g_envelope_count == kWindowFrames) {
            final_count = detect_event_peaks();
        }
        Serial.printf("RESULT samples=%llu frames=%lu duration=%.3fs final_count_3s=%u final_feed_rate=%.3f\r\n",
            (unsigned long long)g_samples_seen,
            (unsigned long)g_frames_processed,
            (double)g_samples_seen / AVS_SAMPLE_RATE,
            (unsigned)final_count,
            (float)final_count / 3.0f);
        Serial.println(F("DONE"));
        g_stream_active = false;
        if (g_profile_stream_active) {
            print_benchmark_results(g_profile_frame_count, g_profile_peak_count,
                                    0, "wav_segment");
            g_profile_stream_active = false;
        }
        return;
    }

    Serial.println(F("ERROR UNKNOWN_COMMAND"));
}
}

void setup() {
    Serial.begin(2000000);
    Serial.setTimeout(10000);
    delay(1200);
    Serial.println();
    Serial.println(F("Acoustic Virtual Sensor - WAV serial receiver"));
    if (avs_fft_init() != AVS_OK || avs_nnls_init(&g_nnls, AVS_W_STANDARD) != AVS_OK) {
        Serial.println(F("ERROR DSP_OR_DICTIONARY_INIT"));
        return;
    }
    avs_make_hann(g_window, AVS_N_FFT);
    g_ready = true;
    Serial.printf("AVS_READY sample_rate=%u format=float32le max_block=%u\r\n",
                  AVS_SAMPLE_RATE, (unsigned)kMaxBlockSamples);
}

void loop() {
    if (!g_ready || !Serial.available()) return;
    String command = Serial.readStringUntil('\n');
    handle_command(command);
}
