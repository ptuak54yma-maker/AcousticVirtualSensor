#include "avs_pipeline.h"

#include <cmath>
#include <utility>

namespace {
constexpr float kPi = 3.14159265358979323846f;

void fft_radix2(float *real, float *imag, size_t n) {
    // In-place bit reversal permutation.
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1U;
        for (; (j & bit) != 0U; bit >>= 1U) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    for (size_t len = 2; len <= n; len <<= 1U) {
        const float angle = -2.0f * kPi / static_cast<float>(len);
        const float wr_step = std::cos(angle);
        const float wi_step = std::sin(angle);
        for (size_t base = 0; base < n; base += len) {
            float wr = 1.0f;
            float wi = 0.0f;
            for (size_t j = 0; j < len / 2; ++j) {
                const size_t u_index = base + j;
                const size_t v_index = u_index + len / 2;
                const float vr = real[v_index] * wr - imag[v_index] * wi;
                const float vi = real[v_index] * wi + imag[v_index] * wr;
                const float ur = real[u_index];
                const float ui = imag[u_index];
                real[u_index] = ur + vr;
                imag[u_index] = ui + vi;
                real[v_index] = ur - vr;
                imag[v_index] = ui - vi;
                const float next_wr = wr * wr_step - wi * wi_step;
                const float next_wi = wr * wi_step + wi * wr_step;
                wr = next_wr;
                wi = next_wi;
            }
        }
    }
}
}

extern "C" void avs_make_hann(float *window, size_t length) {
    if (!window || length == 0) return;
    for (size_t n = 0; n < length; ++n) {
        window[n] = 0.5f - 0.5f * std::cos((2.0f * kPi * static_cast<float>(n)) /
                                          static_cast<float>(length));
    }
}

extern "C" avs_status_t avs_compute_spectrum(const float *frame,
                                               const float *window,
                                               avs_fft_workspace_t *workspace,
                                               avs_spectrum_t *out) {
    if (!frame || !window || !workspace || !out) return AVS_ERR_ARGUMENT;

    for (size_t n = 0; n < AVS_N_FFT; ++n) {
        workspace->real[n] = frame[n] * window[n];
        workspace->imag[n] = 0.0f;
    }
    fft_radix2(workspace->real, workspace->imag, AVS_N_FFT);

    for (size_t k = 0; k < AVS_NUM_BINS; ++k) {
        const float re = workspace->real[k];
        const float im = workspace->imag[k];
        const float magnitude = std::sqrt(re * re + im * im);
        out->log_magnitude[k] = std::log1p(magnitude);
    }
    return AVS_OK;
}
