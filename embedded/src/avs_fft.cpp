#include "avs_pipeline.h"

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace {
constexpr float kPi = 3.14159265358979323846f;
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

    // Simple O(N^2) real DFT is deliberately used as an easily auditable reference
    // implementation for Stage 1. Replace with CMSIS-DSP RFFT for MCU builds.
    // This implementation is suitable for correctness smoke tests, not realtime.
    for (size_t k = 0; k < AVS_NUM_BINS; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (size_t n = 0; n < AVS_N_FFT; ++n) {
            const double angle = -2.0 * static_cast<double>(kPi) *
                                 static_cast<double>(k) * static_cast<double>(n) /
                                 static_cast<double>(AVS_N_FFT);
            const double x = static_cast<double>(frame[n]) * static_cast<double>(window[n]);
            re += x * std::cos(angle);
            im += x * std::sin(angle);
        }
        const float mag = static_cast<float>(std::sqrt(re * re + im * im));
        out->magnitude[k] = mag;
        out->log_magnitude[k] = std::log1p(mag);
    }
    return AVS_OK;
}
