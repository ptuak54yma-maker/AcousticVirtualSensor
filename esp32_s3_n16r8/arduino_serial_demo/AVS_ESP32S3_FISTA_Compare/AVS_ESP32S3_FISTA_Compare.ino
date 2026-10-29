#include <Arduino.h>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "avs_pipeline.h"
#include "w_standard_data.h"

namespace {
constexpr size_t kMaxFrames = 1200;
constexpr size_t kMaxBlockSamples = 512;
constexpr unsigned kFistaIterations = 256;
constexpr size_t kStageCount = 9;
constexpr double kTwoPi = 6.2831853071795864769;

enum stage_t : size_t {
    kFrameInputCopy = 0,
    kPreprocess,
    kWindowing,
    kFft,
    kSpectrum,
    kNnls,
    kEventSum,
    kEnvelopeWrite,
    kPeakDetection,
};

constexpr const char *kStageNames[kStageCount] = {
    "frame_input_copy", "preprocess", "windowing", "fft", "spectrum",
    "nnls", "event_sum", "envelope_write", "peak_detection_full_file"
};

#if defined(__OPTIMIZE__)
constexpr const char *kOptimization = "ON";
#else
constexpr const char *kOptimization = "OFF";
#endif

struct stage_timer_t {
    int64_t us;
    esp_cpu_cycle_count_t cycles;
};

avs_fft_workspace_t g_fft{};
avs_spectrum_t g_spectrum{};
float g_window[AVS_N_FFT]{};
float g_input_frame[AVS_N_FFT]{};
float g_work_frame[AVS_N_FFT]{};
float g_activation[AVS_NUM_COMPONENTS]{};
float g_column_scale[AVS_NUM_COMPONENTS]{};
float g_gram[AVS_NUM_COMPONENTS * AVS_NUM_COMPONENTS]{};
float g_rhs[AVS_NUM_COMPONENTS]{};
float g_x0[AVS_NUM_COMPONENTS]{};
float g_x1[AVS_NUM_COMPONENTS]{};
float g_momentum[AVS_NUM_COMPONENTS]{};
float g_event_scores[kMaxFrames]{};
float g_projected_gradient[kMaxFrames]{};
float *g_spectrum_results = nullptr;
float *g_activation_results = nullptr;
uint32_t g_stage_us[kStageCount][kMaxFrames]{};
uint32_t g_stage_cycles[kStageCount][kMaxFrames]{};
uint32_t g_frame_total_us[kMaxFrames]{};
uint32_t g_frame_total_cycles[kMaxFrames]{};
uint32_t g_sort_scratch[kMaxFrames]{};
size_t g_peak_indices[kMaxFrames]{};
uint8_t g_rx_bytes[kMaxBlockSamples * sizeof(float)]{};
float g_lipschitz = 0.0f;
float g_step_size = 0.0f;
size_t g_frame_fill = 0;
size_t g_frame_count = 0;
uint64_t g_expected_samples = 0;
uint64_t g_samples_received = 0;
bool g_stream_active = false;
bool g_ready = false;

stage_timer_t timer_start() {
    return {esp_timer_get_time(), esp_cpu_get_cycle_count()};
}

void timer_stop(stage_t stage, size_t frame, stage_timer_t start) {
    g_stage_us[stage][frame] =
        static_cast<uint32_t>(esp_timer_get_time() - start.us);
    g_stage_cycles[stage][frame] = static_cast<uint32_t>(
        esp_cpu_get_cycle_count() - start.cycles);
}

bool init_normalized_fista() {
    std::fill(g_gram, g_gram + AVS_NUM_COMPONENTS * AVS_NUM_COMPONENTS, 0.0f);
    for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) {
        float norm_squared = 0.0f;
        for (size_t f = 0; f < AVS_NUM_BINS; ++f) {
            const float w = AVS_W_STANDARD[f * AVS_NUM_COMPONENTS + j];
            if (!std::isfinite(w) || w < 0.0f) return false;
            norm_squared += w * w;
        }
        if (!(norm_squared > 0.0f) || !std::isfinite(norm_squared)) return false;
        g_column_scale[j] = 1.0f / std::sqrt(norm_squared);
    }

    for (size_t i = 0; i < AVS_NUM_COMPONENTS; ++i) {
        for (size_t j = i; j < AVS_NUM_COMPONENTS; ++j) {
            float dot = 0.0f;
            for (size_t f = 0; f < AVS_NUM_BINS; ++f) {
                const float wi = AVS_W_STANDARD[f * AVS_NUM_COMPONENTS + i] *
                                 g_column_scale[i];
                const float wj = AVS_W_STANDARD[f * AVS_NUM_COMPONENTS + j] *
                                 g_column_scale[j];
                dot += wi * wj;
            }
            g_gram[i * AVS_NUM_COMPONENTS + j] = dot;
            g_gram[j * AVS_NUM_COMPONENTS + i] = dot;
        }
    }

    // The normalized Gram matrix is symmetric and nonnegative. Its maximum
    // row sum safely bounds its largest eigenvalue for the FISTA step size.
    for (size_t i = 0; i < AVS_NUM_COMPONENTS; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) {
            row_sum += g_gram[i * AVS_NUM_COMPONENTS + j];
        }
        g_lipschitz = std::max(g_lipschitz, row_sum);
    }
    if (!(g_lipschitz > 0.0f) || !std::isfinite(g_lipschitz)) return false;
    g_step_size = 1.0f / g_lipschitz;
    return true;
}

bool solve_fista(const float *v, float *h_out, float *relative_pg) {
    for (size_t f = 0; f < AVS_NUM_BINS; ++f) {
        if (!std::isfinite(v[f]) || v[f] < 0.0f) return false;
    }
    std::fill(g_rhs, g_rhs + AVS_NUM_COMPONENTS, 0.0f);
    for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) {
        float sum = 0.0f;
        for (size_t f = 0; f < AVS_NUM_BINS; ++f) {
            const float wn = AVS_W_STANDARD[f * AVS_NUM_COMPONENTS + j] *
                             g_column_scale[j];
            sum += wn * v[f];
        }
        g_rhs[j] = sum;
    }

    std::fill(g_x0, g_x0 + AVS_NUM_COMPONENTS, 0.0f);
    std::fill(g_x1, g_x1 + AVS_NUM_COMPONENTS, 0.0f);
    std::fill(g_momentum, g_momentum + AVS_NUM_COMPONENTS, 0.0f);
    float *x = g_x0;
    float *x_next = g_x1;
    float t = 1.0f;

    for (unsigned iteration = 0; iteration < kFistaIterations; ++iteration) {
        const float t_next = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
        const float beta = (t - 1.0f) / t_next;
        for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) {
            float gy = 0.0f;
            for (size_t k = 0; k < AVS_NUM_COMPONENTS; ++k) {
                gy += g_gram[j * AVS_NUM_COMPONENTS + k] * g_momentum[k];
            }
            x_next[j] = std::max(0.0f,
                g_momentum[j] - g_step_size * (gy - g_rhs[j]));
        }
        for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) {
            g_momentum[j] = x_next[j] + beta * (x_next[j] - x[j]);
        }
        std::swap(x, x_next);
        t = t_next;
    }

    float max_rhs = 0.0f;
    float max_delta = 0.0f;
    for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) {
        float gx = 0.0f;
        for (size_t k = 0; k < AVS_NUM_COMPONENTS; ++k) {
            gx += g_gram[j * AVS_NUM_COMPONENTS + k] * x[k];
        }
        const float projected = std::max(0.0f,
            x[j] - g_step_size * (gx - g_rhs[j]));
        max_delta = std::max(max_delta, std::fabs(x[j] - projected));
        max_rhs = std::max(max_rhs, std::fabs(g_rhs[j]));
        if (!std::isfinite(x[j])) return false;
        // Convert from normalized-column variables back to original H units.
        h_out[j] = x[j] * g_column_scale[j];
    }
    *relative_pg = (max_delta / g_step_size) / std::max(1.0f, max_rhs);
    return std::isfinite(*relative_pg);
}

bool process_frame() {
    if (g_frame_count >= kMaxFrames) return false;
    const size_t frame_index = g_frame_count;
    uint32_t frame_us = 0;
    uint32_t frame_cycles = 0;
    stage_timer_t tm = timer_start();
    std::memcpy(g_work_frame, g_input_frame, sizeof(g_work_frame));
    timer_stop(kFrameInputCopy, frame_index, tm);

    tm = timer_start();
    float mean = 0.0f;
    for (float sample : g_work_frame) mean += sample;
    mean /= AVS_N_FFT;
    for (float &sample : g_work_frame) sample -= mean;
    timer_stop(kPreprocess, frame_index, tm);

    tm = timer_start();
    const avs_status_t prep_status =
        avs_prepare_fft_input(g_work_frame, g_window, &g_fft);
    timer_stop(kWindowing, frame_index, tm);
    if (prep_status != AVS_OK) return false;

    tm = timer_start();
    const avs_status_t fft_status = avs_execute_fft(&g_fft);
    timer_stop(kFft, frame_index, tm);
    if (fft_status != AVS_OK) return false;

    tm = timer_start();
    const avs_status_t spectrum_status = avs_compute_log_spectrum(&g_fft, &g_spectrum);
    timer_stop(kSpectrum, frame_index, tm);
    if (spectrum_status != AVS_OK) return false;
    tm = timer_start();
    if (!solve_fista(g_spectrum.log_magnitude, g_activation,
                     &g_projected_gradient[frame_index])) return false;
    timer_stop(kNnls, frame_index, tm);

    tm = timer_start();
    float score = 0.0f;
    for (size_t j = 0; j < AVS_EVENT_COMPONENTS; ++j) score += g_activation[j];
    timer_stop(kEventSum, frame_index, tm);

    tm = timer_start();
    g_event_scores[frame_index] = score;
    timer_stop(kEnvelopeWrite, frame_index, tm);
    // Diagnostic output copies happen after the timed DSP stages.
    std::memcpy(g_spectrum_results + frame_index * AVS_NUM_BINS,
                g_spectrum.log_magnitude, AVS_NUM_BINS * sizeof(float));
    std::memcpy(g_activation_results + frame_index * AVS_NUM_COMPONENTS,
                g_activation, AVS_NUM_COMPONENTS * sizeof(float));

    for (size_t s = kFrameInputCopy; s <= kEnvelopeWrite; ++s) {
        frame_us += g_stage_us[s][frame_index];
        frame_cycles += g_stage_cycles[s][frame_index];
    }
    g_frame_total_us[frame_index] = frame_us;
    g_frame_total_cycles[frame_index] = frame_cycles;
    ++g_frame_count;
    return true;
}

bool feed_sample(float sample) {
    g_input_frame[g_frame_fill++] = sample;
    ++g_samples_received;
    if (g_frame_fill != AVS_N_FFT) return true;
    if (!process_frame()) return false;
    std::memmove(g_input_frame, g_input_frame + AVS_HOP_LENGTH,
                 (AVS_N_FFT - AVS_HOP_LENGTH) * sizeof(g_input_frame[0]));
    g_frame_fill = AVS_N_FFT - AVS_HOP_LENGTH;
    return true;
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

uint32_t percentile95(uint32_t *values, size_t count) {
    std::copy(values, values + count, g_sort_scratch);
    std::sort(g_sort_scratch, g_sort_scratch + count);
    const size_t index = ((count * 95 + 99) / 100) - 1;
    return g_sort_scratch[index];
}

void report_results() {
    if (g_frame_count == 0) {
        Serial.println(F("BENCH_ERROR NO_FRAMES"));
        return;
    }
    const stage_timer_t peak_timer = timer_start();
    const size_t peak_count = avs_detect_peaks(
        g_event_scores, g_frame_count, AVS_PEAK_PROMINENCE,
        AVS_PEAK_DISTANCE, g_peak_indices, kMaxFrames);
    g_stage_us[kPeakDetection][0] =
        static_cast<uint32_t>(esp_timer_get_time() - peak_timer.us);
    g_stage_cycles[kPeakDetection][0] = static_cast<uint32_t>(
        esp_cpu_get_cycle_count() - peak_timer.cycles);

    for (size_t i = 0; i < g_frame_count; ++i) {
        const double time_s = (double)(AVS_N_FFT / 2 + i * AVS_HOP_LENGTH) /
                              AVS_SAMPLE_RATE;
        Serial.printf("H_EVENT frame=%u time_s=%.6f score=%.9g projected_gradient=%.9g\r\n",
            (unsigned)i, time_s, (double)g_event_scores[i],
            (double)g_projected_gradient[i]);
    }
    Serial.printf("BENCH_CHIP model=%s revision=%u cores=%u cpu_mhz=%u measured_core=%u\r\n",
        ESP.getChipModel(), (unsigned)ESP.getChipRevision(), (unsigned)ESP.getChipCores(),
        (unsigned)ESP.getCpuFreqMHz(), (unsigned)xPortGetCoreID());
    Serial.printf("BENCH_CONFIG algorithm=FISTA normalized_columns=YES precision=float32 iterations=%u step_size=%.9g lipschitz_bound=%.9g optimization=%s serial_logging_in_timed_region=NO allocation_in_timed_region=NO\r\n",
        kFistaIterations, (double)g_step_size, (double)g_lipschitz, kOptimization);
    Serial.printf("BENCH_FRAME samples=%llu frames=%u sample_rate=%u hop=%u deadline_us=%.2f\r\n",
        (unsigned long long)g_samples_received, (unsigned)g_frame_count,
        AVS_SAMPLE_RATE, AVS_HOP_LENGTH,
        1000000.0 * AVS_HOP_LENGTH / AVS_SAMPLE_RATE);

    uint64_t total_us = 0;
    uint64_t total_cycles = 0;
    double pg_sum = 0.0;
    double pg_max = 0.0;
    double event_sum = 0.0;
    float event_min = INFINITY;
    float event_max = -INFINITY;
    for (size_t i = 0; i < g_frame_count; ++i) {
        total_us += g_frame_total_us[i];
        total_cycles += g_frame_total_cycles[i];
        pg_sum += g_projected_gradient[i];
        pg_max = std::max(pg_max, (double)g_projected_gradient[i]);
        event_sum += g_event_scores[i];
        event_min = std::min(event_min, g_event_scores[i]);
        event_max = std::max(event_max, g_event_scores[i]);
    }
    const double average_us = (double)total_us / g_frame_count;
    const uint32_t p95_us = percentile95(g_frame_total_us, g_frame_count);
    const double frame_rate = (double)AVS_SAMPLE_RATE / AVS_HOP_LENGTH;
    const double peak_us_per_audio_second =
        (double)g_stage_us[kPeakDetection][0] /
        ((double)g_samples_received / AVS_SAMPLE_RATE);
    const double compute_us_per_audio_second = average_us * frame_rate +
                                               peak_us_per_audio_second;
    Serial.printf("BENCH_TOTAL_FRAME average_us=%.2f p95_us=%u average_cycles=%.0f\r\n",
        average_us, (unsigned)p95_us, (double)total_cycles / g_frame_count);
    Serial.printf("BENCH_REALTIME capacity_x=%.4f meets_realtime=%s\r\n",
        1000000.0 / compute_us_per_audio_second,
        compute_us_per_audio_second <= 1000000.0 ? "YES" : "NO");
    for (size_t stage = 0; stage < kStageCount; ++stage) {
        const size_t count = stage == kPeakDetection ? 1 : g_frame_count;
        uint64_t us_sum = 0;
        uint64_t cycles_sum = 0;
        uint32_t max_us = 0;
        for (size_t i = 0; i < count; ++i) {
            us_sum += g_stage_us[stage][i];
            cycles_sum += g_stage_cycles[stage][i];
            max_us = std::max(max_us, g_stage_us[stage][i]);
        }
        const uint32_t p95_us_stage = percentile95(g_stage_us[stage], count);
        Serial.printf("BENCH_STAGE name=%s samples=%u average_us=%.2f p95_us=%u max_us=%u average_cycles=%.0f\r\n",
            kStageNames[stage], (unsigned)count, (double)us_sum / count,
            (unsigned)p95_us_stage, (unsigned)max_us, (double)cycles_sum / count);
    }
    Serial.printf("BENCH_NNLS_DIAGNOSTICS samples=%u avg_relative_projected_gradient=%.9g max_relative_projected_gradient=%.9g\r\n",
        (unsigned)g_frame_count, pg_sum / g_frame_count, pg_max);
    Serial.printf("BENCH_EVENT_SCORE min=%.9g average=%.9g max=%.9g\r\n",
        (double)event_min, event_sum / g_frame_count, (double)event_max);
    Serial.printf("RESULT duration_s=%.6f peak_count=%u peak_frames=",
        (double)g_samples_received / AVS_SAMPLE_RATE, (unsigned)peak_count);
    for (size_t i = 0; i < peak_count; ++i) {
        if (i) Serial.print(',');
        Serial.print((unsigned)g_peak_indices[i]);
    }
    Serial.println();
    const size_t spectrum_bytes = g_frame_count * AVS_NUM_BINS * sizeof(float);
    const size_t activation_bytes =
        g_frame_count * AVS_NUM_COMPONENTS * sizeof(float);
    Serial.printf("FISTA_DATA_BEGIN frames=%u bins=%u components=%u format=float32le spectrum_bytes=%u activation_bytes=%u\r\n",
        (unsigned)g_frame_count, AVS_NUM_BINS, AVS_NUM_COMPONENTS,
        (unsigned)spectrum_bytes, (unsigned)activation_bytes);
    Serial.write(reinterpret_cast<const uint8_t *>(g_spectrum_results), spectrum_bytes);
    Serial.write(reinterpret_cast<const uint8_t *>(g_activation_results), activation_bytes);
    Serial.println();
    Serial.println(F("FISTA_COMPARE_DONE"));
}

void clear_run() {
    std::memset(g_input_frame, 0, sizeof(g_input_frame));
    std::memset(g_stage_us, 0, sizeof(g_stage_us));
    std::memset(g_stage_cycles, 0, sizeof(g_stage_cycles));
    std::memset(g_frame_total_us, 0, sizeof(g_frame_total_us));
    std::memset(g_frame_total_cycles, 0, sizeof(g_frame_total_cycles));
    std::memset(g_event_scores, 0, sizeof(g_event_scores));
    std::memset(g_projected_gradient, 0, sizeof(g_projected_gradient));
    g_frame_fill = 0;
    g_frame_count = 0;
    g_samples_received = 0;
}

void handle_command(String command) {
    command.trim();
    if (command == "PING") {
        Serial.println(F("PONG"));
        return;
    }
    if (command.startsWith("BENCH_WAV ")) {
        const uint64_t samples = strtoull(command.c_str() + 10, nullptr, 10);
        const uint64_t frames = samples < AVS_N_FFT ? 0 :
            1 + (samples - AVS_N_FFT) / AVS_HOP_LENGTH;
        if (!g_ready || g_stream_active || frames == 0 || frames > kMaxFrames) {
            Serial.printf("BENCH_ERROR WAV_LENGTH_OR_BOARD_STATE max_frames=%u\r\n",
                (unsigned)kMaxFrames);
            return;
        }
        clear_run();
        g_expected_samples = samples;
        g_stream_active = true;
        Serial.printf("ACK_BENCH_START samples=%llu expected_frames=%llu\r\n",
            (unsigned long long)samples, (unsigned long long)frames);
        return;
    }
    if (command.startsWith("DATA ")) {
        if (!g_stream_active) {
            Serial.println(F("ERROR NO_ACTIVE_STREAM"));
            return;
        }
        const size_t count = (size_t)command.substring(5).toInt();
        if (count == 0 || count > kMaxBlockSamples ||
            g_samples_received + count > g_expected_samples) {
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
            std::memcpy(&sample, g_rx_bytes + i * sizeof(float), sizeof(sample));
            if (!std::isfinite(sample) || !feed_sample(sample)) {
                Serial.println(F("ERROR FRAME_PROCESSING"));
                g_stream_active = false;
                return;
            }
        }
        Serial.printf("ACK %llu\r\n", (unsigned long long)g_samples_received);
        return;
    }
    if (command == "END") {
        if (!g_stream_active) {
            Serial.println(F("ERROR END_COUNT expected=0 received=0"));
            return;
        }
        if (g_samples_received != g_expected_samples) {
            Serial.printf("ERROR END_COUNT expected=%llu received=%llu\r\n",
                (unsigned long long)g_expected_samples,
                (unsigned long long)g_samples_received);
            g_stream_active = false;
            return;
        }
        g_stream_active = false;
        report_results();
        return;
    }
    Serial.println(F("ERROR UNKNOWN_COMMAND"));
}
}

void setup() {
    Serial.begin(2000000);
    Serial.setTimeout(10000);
    delay(1200);
    constexpr uint32_t compare_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    g_spectrum_results = static_cast<float *>(heap_caps_calloc(
        kMaxFrames * AVS_NUM_BINS, sizeof(float), compare_caps));
    g_activation_results = static_cast<float *>(heap_caps_calloc(
        kMaxFrames * AVS_NUM_COMPONENTS, sizeof(float), compare_caps));
    if (!g_spectrum_results || !g_activation_results) {
        Serial.println(F("ERROR COMPARE_PSRAM_ALLOC"));
        return;
    }
    if (avs_fft_init() != AVS_OK || !init_normalized_fista()) {
        Serial.println(F("ERROR DSP_OR_FISTA_INIT"));
        return;
    }
    avs_make_hann(g_window, AVS_N_FFT);
    g_ready = true;
    Serial.printf("AVS_READY algorithm=FISTA normalized_columns=YES precision=float32 iterations=%u sample_rate=%u\r\n",
        kFistaIterations, AVS_SAMPLE_RATE);
}

void loop() {
    if (!g_ready || !Serial.available()) return;
    handle_command(Serial.readStringUntil('\n'));
}
