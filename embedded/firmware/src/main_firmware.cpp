#include "avs_pipeline.h"
#include "w_standard_data.h"

#include <cmath>
#include <cstdint>

extern "C" {
extern volatile uint32_t avs_fw_test_status;
extern volatile uint32_t avs_fw_failed_stage;
extern volatile uint32_t avs_fw_frames_processed;
extern volatile uint32_t avs_fw_peak_count;
extern volatile uint32_t avs_fw_peak_first;
extern volatile float avs_fw_last_event_sum;
extern volatile float avs_fw_fft_bin32;
extern volatile float avs_fw_last_activation_0;
}

namespace {
constexpr uint32_t kStatusRunning = 1u;
constexpr uint32_t kStatusPass = 0x600D0001u;
constexpr uint32_t kStatusFail = 0xBAD00000u;
constexpr float kPi = 3.14159265358979323846f;

avs_fft_workspace_t g_fft_workspace;
avs_spectrum_t g_spectrum;
avs_nnls_workspace_t g_nnls_workspace;
float g_window[AVS_N_FFT];
float g_frame[AVS_N_FFT];
float g_activation[AVS_NUM_COMPONENTS];
float g_synthetic_envelope[] = {0.0f, 0.0f, 0.02f, 0.0f,
                                0.0f, 0.04f, 0.0f, 0.0f};
size_t g_peak_indices[8];

constexpr uintptr_t kRccAhb1Enr = 0x40023830u;
constexpr uintptr_t kRccApb1Enr = 0x40023840u;
constexpr uintptr_t kGpiodModer = 0x40020C00u;
constexpr uintptr_t kGpiodAfrl = 0x40020C20u;
constexpr uintptr_t kUart4Base = 0x40004C00u;
constexpr uintptr_t kUartSr = 0x00u;
constexpr uintptr_t kUartDr = 0x04u;
constexpr uintptr_t kUartBrr = 0x08u;
constexpr uintptr_t kUartCr1 = 0x0Cu;

volatile uint32_t &reg32(uintptr_t address) {
    return *reinterpret_cast<volatile uint32_t *>(address);
}

void uart4_init() {
    reg32(kRccAhb1Enr) |= 1u << 3;  // GPIOD clock
    reg32(kRccApb1Enr) |= 1u << 19; // UART4 clock
    reg32(kGpiodModer) = (reg32(kGpiodModer) & ~(0xFu << 4)) | (0xAu << 4);
    reg32(kGpiodAfrl) = (reg32(kGpiodAfrl) & ~(0xFFu << 8)) |
                        (8u << 8) | (8u << 12); // PD2/PD3 AF8
    reg32(kUart4Base + kUartBrr) = 139u; // 16 MHz HSI / 115200 baud
    reg32(kUart4Base + kUartCr1) = (1u << 13) | (1u << 3); // UE | TE
}

void uart4_putc(char value) {
    uint32_t timeout = 1000000u;
    while ((reg32(kUart4Base + kUartSr) & (1u << 7)) == 0u && timeout != 0u) {
        --timeout;
    }
    if (timeout != 0u) reg32(kUart4Base + kUartDr) = static_cast<uint32_t>(value);
}

void uart4_write(const char *message) {
    while (*message) uart4_putc(*message++);
}

void uart4_write_u32(uint32_t value) {
    char digits[10];
    size_t count = 0;
    do {
        digits[count++] = static_cast<char>('0' + (value % 10u));
        value /= 10u;
    } while (value != 0u && count < sizeof(digits));
    while (count > 0u) uart4_putc(digits[--count]);
}

void uart4_write_micro(const char *label, float value) {
    uart4_write(label);
    if (!std::isfinite(value) || value < 0.0f || value > 4294.0f) {
        uart4_write("INVALID\r\n");
        return;
    }
    uart4_write_u32(static_cast<uint32_t>(value * 1000000.0f + 0.5f));
    uart4_write("\r\n");
}

void fail(uint32_t stage, const char *message) {
    avs_fw_failed_stage = stage;
    avs_fw_test_status = kStatusFail | stage;
    uart4_write(message);
    for (;;) {}
}
}

extern "C" {
volatile uint32_t avs_fw_test_status = 0u;
volatile uint32_t avs_fw_failed_stage = 0u;
volatile uint32_t avs_fw_frames_processed = 0u;
volatile uint32_t avs_fw_peak_count = 0u;
volatile uint32_t avs_fw_peak_first = 0u;
volatile float avs_fw_last_event_sum = 0.0f;
volatile float avs_fw_fft_bin32 = 0.0f;
volatile float avs_fw_last_activation_0 = 0.0f;

int main() {
    avs_fw_test_status = kStatusRunning;
    uart4_init();
    uart4_write("AVS STM32F407VG bare-metal test start\r\n");

    if (avs_fft_init() != AVS_OK) fail(1u, "FFT_INIT_FAIL\r\n");
    avs_make_hann(g_window, AVS_N_FFT);
    if (avs_nnls_init(&g_nnls_workspace, AVS_W_STANDARD) != AVS_OK) {
        fail(2u, "NNLS_INIT_FAIL\r\n");
    }
    uart4_write("FFT_AND_DICTIONARY_OK\r\n");

    // Keep the bare-metal simulator smoke workload bounded. Full WAV behavior
    // is checked by the host-side Python/C++ comparison harness.
    for (uint32_t frame_no = 0; frame_no < 1u; ++frame_no) {
        const float amplitude = 0.25f + 0.05f * static_cast<float>(frame_no);
        for (size_t n = 0; n < AVS_N_FFT; ++n) {
            const float phase = 2.0f * kPi * static_cast<float>(n) /
                                static_cast<float>(AVS_N_FFT);
            g_frame[n] = amplitude * std::sin(32.0f * phase) +
                         0.1f * std::cos(77.0f * phase);
        }
        float mean = 0.0f;
        for (float sample : g_frame) mean += sample;
        mean /= static_cast<float>(AVS_N_FFT);
        for (float &sample : g_frame) sample -= mean;

        if (avs_compute_spectrum(g_frame, g_window, &g_fft_workspace,
                                 &g_spectrum) != AVS_OK) {
            fail(3u, "FFT_FRAME_FAIL\r\n");
        }
        if (avs_solve_nnls(g_spectrum.log_magnitude, AVS_W_STANDARD,
                           &g_nnls_workspace, g_activation, 20000u,
                           1e-6f) != AVS_OK) {
            fail(4u, "NNLS_FRAME_FAIL\r\n");
        }
        avs_fw_last_event_sum = avs_sum_event_activations(
            g_activation, AVS_NUM_COMPONENTS);
        avs_fw_last_activation_0 = g_activation[0];
        avs_fw_fft_bin32 = g_spectrum.log_magnitude[32];
        ++avs_fw_frames_processed;
    }
    uart4_write("FFT_NNLS_EVENT_SUM_OK\r\n");
    uart4_write("FRAMES_PROCESSED=");
    uart4_write_u32(avs_fw_frames_processed);
    uart4_write("\r\n");
    uart4_write_micro("FFT_BIN32_MICRO=", avs_fw_fft_bin32);
    uart4_write_micro("ACTIVATION_0_MICRO=", avs_fw_last_activation_0);
    uart4_write_micro("EVENT_SUM_MICRO=", avs_fw_last_event_sum);

    avs_fw_peak_count = static_cast<uint32_t>(avs_detect_peaks(
        g_synthetic_envelope,
        sizeof(g_synthetic_envelope) / sizeof(g_synthetic_envelope[0]),
        AVS_PEAK_PROMINENCE, AVS_PEAK_DISTANCE,
        g_peak_indices, sizeof(g_peak_indices) / sizeof(g_peak_indices[0])));
    if (avs_fw_peak_count != 1u || g_peak_indices[0] != 5u) {
        fail(5u, "PEAK_TEST_FAIL\r\n");
    }
    avs_fw_peak_first = static_cast<uint32_t>(g_peak_indices[0]);
    uart4_write("PEAK_TEST_OK count=1 index=5\r\n");

    avs_fw_test_status = kStatusPass;
    uart4_write("AVS_TEST_PASS\r\n");
    for (;;) {}
}
}
