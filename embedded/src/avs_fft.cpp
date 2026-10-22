#include "avs_pipeline.h"

#include <cmath>
#include <complex>
#include <vector>

namespace {
constexpr float kPi = 3.14159265358979323846f;

void fft_radix2(std::vector<std::complex<float>> &a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1U;
        for (; j & bit; bit >>= 1U) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1U) {
        const float angle = -2.0f * kPi / static_cast<float>(len);
        const std::complex<float> wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; ++j) {
                const auto u = a[i + j];
                const auto v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
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
                                                avs_spectrum_t *out) {
    if (!frame || !window || !out) return AVS_ERR_ARGUMENT;

    std::vector<std::complex<float>> fft(AVS_N_FFT);
    for (size_t n = 0; n < AVS_N_FFT; ++n) {
        fft[n] = std::complex<float>(frame[n] * window[n], 0.0f);
    }
    fft_radix2(fft);

    for (size_t k = 0; k < AVS_NUM_BINS; ++k) {
        const float mag = std::abs(fft[k]);
        out->magnitude[k] = mag;
        out->log_magnitude[k] = std::log1p(mag);
    }
    return AVS_OK;
}
