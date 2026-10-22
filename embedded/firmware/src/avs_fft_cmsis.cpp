#include "avs_pipeline.h"

#include "arm_math.h"

#include <cmath>

namespace {
arm_rfft_fast_instance_f32 g_rfft;
bool g_rfft_ready = false;
constexpr float kPi = 3.14159265358979323846f;
}

extern "C" avs_status_t avs_fft_init(void) {
    g_rfft_ready = arm_rfft_fast_init_2048_f32(&g_rfft) == ARM_MATH_SUCCESS;
    return g_rfft_ready ? AVS_OK : AVS_ERR_UNSUPPORTED;
}

extern "C" void avs_make_hann(float *window, size_t length) {
    if (!window || length == 0) return;
    for (size_t n = 0; n < length; ++n) {
        window[n] = 0.5f - 0.5f * std::cos(
            (2.0f * kPi * static_cast<float>(n)) / static_cast<float>(length));
    }
}

extern "C" avs_status_t avs_compute_spectrum(const float *frame,
                                             const float *window,
                                             avs_fft_workspace_t *workspace,
                                             avs_spectrum_t *out) {
    if (!frame || !window || !workspace || !out) return AVS_ERR_ARGUMENT;
    if (!g_rfft_ready) return AVS_ERR_UNSUPPORTED;

    for (size_t n = 0; n < AVS_N_FFT; ++n) {
        workspace->real[n] = frame[n] * window[n];
    }
    arm_rfft_fast_f32(&g_rfft, workspace->real, workspace->imag, 0u);

    out->log_magnitude[0] = std::log1p(std::fabs(workspace->imag[0]));
    for (size_t k = 1; k < AVS_N_FFT / 2; ++k) {
        const float re = workspace->imag[2 * k];
        const float im = workspace->imag[2 * k + 1];
        out->log_magnitude[k] = std::log1p(std::sqrt(re * re + im * im));
    }
    out->log_magnitude[AVS_N_FFT / 2] =
        std::log1p(std::fabs(workspace->imag[1]));
    return AVS_OK;
}
