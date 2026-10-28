#include "avs_pipeline.h"
#include "w_standard_data.h"

#include "driver/i2s_std.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {
constexpr char kTag[] = "AVS";
constexpr size_t kReadSamples = 256;
constexpr size_t kWindowFrames = 3u * AVS_SAMPLE_RATE / AVS_HOP_LENGTH;
constexpr unsigned kNnlsIterations = 512;
constexpr float kNnlsTolerance = 1.0e-6f;

static_assert(AVS_W_ROWS == AVS_NUM_BINS, "Dictionary row count mismatch");
static_assert(AVS_W_COLS == AVS_NUM_COMPONENTS, "Dictionary column count mismatch");

i2s_chan_handle_t g_rx = nullptr;
avs_fft_workspace_t g_fft{};
avs_spectrum_t g_spectrum{};
avs_nnls_workspace_t g_nnls{};
float g_window[AVS_N_FFT]{};
float g_frame[AVS_N_FFT]{};
float g_activation[AVS_NUM_COMPONENTS]{};
float g_envelope[kWindowFrames]{};
size_t g_envelope_count = 0;
size_t g_peak_indices[kWindowFrames]{};
size_t g_frame_fill = 0;
uint64_t g_samples_seen = 0;
uint64_t g_last_report_sample = 0;

void check(esp_err_t err, const char *what) {
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "%s failed: %s", what, esp_err_to_name(err));
        abort();
    }
}

void init_i2s() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(
        static_cast<i2s_port_t>(CONFIG_AVS_I2S_PORT), I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = kReadSamples;
    check(i2s_new_channel(&chan_cfg, nullptr, &g_rx), "i2s_new_channel");

    i2s_std_config_t std_cfg = {};
    std_cfg.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AVS_SAMPLE_RATE);
    std_cfg.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
        I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
    std_cfg.gpio_cfg.mclk = I2S_GPIO_UNUSED;
    std_cfg.gpio_cfg.bclk = static_cast<gpio_num_t>(CONFIG_AVS_I2S_BCLK_GPIO);
    std_cfg.gpio_cfg.ws = static_cast<gpio_num_t>(CONFIG_AVS_I2S_WS_GPIO);
    std_cfg.gpio_cfg.dout = I2S_GPIO_UNUSED;
    std_cfg.gpio_cfg.din = static_cast<gpio_num_t>(CONFIG_AVS_I2S_DIN_GPIO);
    std_cfg.gpio_cfg.invert_flags.mclk_inv = false;
    std_cfg.gpio_cfg.invert_flags.bclk_inv = false;
    std_cfg.gpio_cfg.invert_flags.ws_inv = false;
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    check(i2s_channel_init_std_mode(g_rx, &std_cfg), "i2s_channel_init_std_mode");
    check(i2s_channel_enable(g_rx), "i2s_channel_enable");
}

void process_frame() {
    float mean = 0.0f;
    for (float x : g_frame) mean += x;
    mean /= AVS_N_FFT;
    for (float &x : g_frame) x -= mean;

    if (avs_compute_spectrum(g_frame, g_window, &g_fft, &g_spectrum) != AVS_OK) {
        ESP_LOGE(kTag, "FFT failed");
        return;
    }
    const avs_status_t status = avs_solve_nnls(
        g_spectrum.log_magnitude, AVS_W_STANDARD, &g_nnls, g_activation,
        kNnlsIterations, kNnlsTolerance);
    if (status != AVS_OK) {
        ESP_LOGW(kTag, "NNLS frame skipped (status=%d)", static_cast<int>(status));
        return;
    }

    const float he = avs_sum_event_activations(g_activation, AVS_NUM_COMPONENTS);
    if (g_envelope_count < kWindowFrames) {
        g_envelope[g_envelope_count++] = he;
    } else {
        std::memmove(g_envelope, g_envelope + 1,
                     (kWindowFrames - 1) * sizeof(g_envelope[0]));
        g_envelope[kWindowFrames - 1] = he;
    }

    if (g_envelope_count == kWindowFrames &&
        g_samples_seen - g_last_report_sample >= AVS_SAMPLE_RATE) {
        const size_t peaks = avs_detect_peaks(
            g_envelope, g_envelope_count, AVS_PEAK_PROMINENCE,
            AVS_PEAK_DISTANCE, g_peak_indices, kWindowFrames);
        if (peaks == AVS_PEAKS_OVERFLOW) {
            ESP_LOGE(kTag, "Peak buffer overflow");
        } else {
            const float feed_rate = static_cast<float>(peaks) / 3.0f;
            ESP_LOGI(kTag, "t=%.2fs count_3s=%u feed_rate=%.2f parts/s H_event=%.5f",
                     static_cast<double>(g_samples_seen) / AVS_SAMPLE_RATE,
                     static_cast<unsigned>(peaks), feed_rate, he);
        }
        g_last_report_sample = g_samples_seen;
    }
}

void accept_pcm_word(int32_t raw) {
    // Common I2S MEMS microphones put signed 24-bit PCM in bits 31..8.
    // Adjust this conversion if the selected microphone sends a different format.
    const int32_t pcm24 = raw >> 8;
    const float sample = static_cast<float>(pcm24) / 8388608.0f;
    g_frame[g_frame_fill++] = sample;
    ++g_samples_seen;
    if (g_frame_fill == AVS_N_FFT) {
        process_frame();
        std::memmove(g_frame, g_frame + AVS_HOP_LENGTH,
                     (AVS_N_FFT - AVS_HOP_LENGTH) * sizeof(g_frame[0]));
        g_frame_fill = AVS_N_FFT - AVS_HOP_LENGTH;
    }
}

void run_serial_demo() {
    ESP_LOGW(kTag, "DEMO MODE: synthetic audio only; no microphone/I2S hardware is used");
    constexpr uint64_t demo_samples = 5ull * AVS_SAMPLE_RATE;
    constexpr float kTwoPi = 6.28318530718f;
    for (uint64_t n = 0; n < demo_samples; ++n) {
        const float t = static_cast<float>(n) / AVS_SAMPLE_RATE;
        const float phase = std::fmod(t, 0.75f);
        const float burst = phase < 0.07f
            ? 0.45f * std::sin(kTwoPi * 1800.0f * t) * (1.0f - phase / 0.07f)
            : 0.0f;
        const float sample = 0.04f * std::sin(kTwoPi * 440.0f * t) + burst;
        g_frame[g_frame_fill++] = sample;
        ++g_samples_seen;
        if (g_frame_fill == AVS_N_FFT) {
            process_frame();
            std::memmove(g_frame, g_frame + AVS_HOP_LENGTH,
                         (AVS_N_FFT - AVS_HOP_LENGTH) * sizeof(g_frame[0]));
            g_frame_fill = AVS_N_FFT - AVS_HOP_LENGTH;
        }
    }
    ESP_LOGI(kTag, "DEMO_DONE: synthetic audio processing finished (%u seconds)",
             static_cast<unsigned>(demo_samples / AVS_SAMPLE_RATE));
    ESP_LOGI(kTag, "This demonstrates firmware/serial output only; connect an I2S mic for real sensing.");
}
}

extern "C" void app_main(void) {
    ESP_LOGI(kTag, "Acoustic Virtual Sensor: ESP32-S3, %u Hz, NFFT=%u, hop=%u",
             AVS_SAMPLE_RATE, AVS_N_FFT, AVS_HOP_LENGTH);
    if (avs_fft_init() != AVS_OK ||
        avs_nnls_init(&g_nnls, AVS_W_STANDARD) != AVS_OK) {
        ESP_LOGE(kTag, "DSP or frozen dictionary initialization failed");
        abort();
    }
    avs_make_hann(g_window, AVS_N_FFT);

#if defined(CONFIG_AVS_DEMO_MODE) && CONFIG_AVS_DEMO_MODE
    run_serial_demo();
    for (;;) vTaskDelay(portMAX_DELAY);
#else
    ESP_LOGI(kTag, "I2S GPIO: BCLK=%d WS=%d DIN=%d; left slot, 32-bit samples",
             CONFIG_AVS_I2S_BCLK_GPIO, CONFIG_AVS_I2S_WS_GPIO,
             CONFIG_AVS_I2S_DIN_GPIO);
    init_i2s();

    int32_t samples[kReadSamples];
    ESP_LOGI(kTag, "Streaming started; waiting for 3 seconds of audio");
    for (;;) {
        size_t bytes_read = 0;
        const esp_err_t err = i2s_channel_read(
            g_rx, samples, sizeof(samples), &bytes_read, portMAX_DELAY);
        if (err != ESP_OK) {
            ESP_LOGW(kTag, "I2S read: %s", esp_err_to_name(err));
            continue;
        }
        const size_t count = bytes_read / sizeof(samples[0]);
        for (size_t i = 0; i < count; ++i) accept_pcm_word(samples[i]);
    }
#endif
}
