#include "avs_pipeline.h"

#include <cmath>
#include <utility>

namespace {
constexpr float kPi = 3.14159265358979323846f;

void fft_radix2(float *real, float *imag, size_t n) {
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1u;
        for (; (j & bit) != 0u; bit >>= 1u) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
    for (size_t len = 2; len <= n; len <<= 1u) {
        const float angle = -2.0f * kPi / static_cast<float>(len);
        const float wr_step = std::cos(angle), wi_step = std::sin(angle);
        for (size_t base = 0; base < n; base += len) {
            float wr = 1.0f, wi = 0.0f;
            for (size_t j = 0; j < len / 2; ++j) {
                const size_t u = base + j, v = u + len / 2;
                const float vr = real[v] * wr - imag[v] * wi;
                const float vi = real[v] * wi + imag[v] * wr;
                const float ur = real[u], ui = imag[u];
                real[u] = ur + vr; imag[u] = ui + vi;
                real[v] = ur - vr; imag[v] = ui - vi;
                const float next_wr = wr * wr_step - wi * wi_step;
                wi = wr * wi_step + wi * wr_step;
                wr = next_wr;
            }
        }
    }
}
}

extern "C" void avs_make_hann(float *window, size_t length) {
    if (!window || length == 0) return;
    for (size_t n = 0; n < length; ++n)
        window[n] = 0.5f - 0.5f * std::cos(2.0f * kPi * n / length);
}

extern "C" avs_status_t avs_fft_init(void) { return AVS_OK; }

extern "C" avs_status_t avs_prepare_fft_input(
    const float *frame, const float *window, avs_fft_workspace_t *workspace) {
    if (!frame || !window || !workspace) return AVS_ERR_ARGUMENT;
    for (size_t n = 0; n < AVS_N_FFT; ++n) {
        workspace->real[n] = frame[n] * window[n];
        workspace->imag[n] = 0.0f;
    }
    return AVS_OK;
}

extern "C" avs_status_t avs_execute_fft(avs_fft_workspace_t *workspace) {
    if (!workspace) return AVS_ERR_ARGUMENT;
    fft_radix2(workspace->real, workspace->imag, AVS_N_FFT);
    return AVS_OK;
}

extern "C" avs_status_t avs_compute_log_spectrum(
    const avs_fft_workspace_t *workspace, avs_spectrum_t *out) {
    if (!workspace || !out) return AVS_ERR_ARGUMENT;
    for (size_t k = 0; k < AVS_NUM_BINS; ++k) {
        const float re = workspace->real[k], im = workspace->imag[k];
        out->log_magnitude[k] = std::log1p(std::sqrt(re * re + im * im));
    }
    return AVS_OK;
}

extern "C" avs_status_t avs_compute_spectrum(
    const float *frame, const float *window, avs_fft_workspace_t *workspace,
    avs_spectrum_t *out) {
    avs_status_t status = avs_prepare_fft_input(frame, window, workspace);
    if (status != AVS_OK) return status;
    status = avs_execute_fft(workspace);
    if (status != AVS_OK) return status;
    return avs_compute_log_spectrum(workspace, out);
}
