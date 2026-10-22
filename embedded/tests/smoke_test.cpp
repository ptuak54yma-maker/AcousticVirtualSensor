#include "avs_pipeline.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    std::vector<float> hann(AVS_N_FFT);
    avs_make_hann(hann.data(), hann.size());

    if (std::fabs(hann.front()) > 1e-6f ||
        std::fabs(hann.back()) > 0.002f ||
        !(hann[AVS_N_FFT / 2] > 0.999f)) {
        std::fprintf(stderr, "Hann window smoke test failed\n");
        return 1;
    }

    float h[AVS_NUM_COMPONENTS] = {};
    h[0] = 1.25f;
    h[1] = 2.5f;
    h[AVS_EVENT_COMPONENTS] = 99.0f;
    const float event_sum = avs_sum_event_activations(h, AVS_NUM_COMPONENTS);
    if (std::fabs(event_sum - 3.75f) > 1e-6f) {
        std::fprintf(stderr, "Event activation aggregation failed: %g\n", event_sum);
        return 2;
    }

    const float env[] = {0.f, 0.f, 1.f, 0.f, 0.f, 2.f, 0.f};
    size_t peaks[4] = {};
    const size_t n = avs_detect_peaks(env, 7, 0.5f, 2, peaks, 4);
    if (n != 2 || peaks[0] != 2 || peaks[1] != 5) {
        std::fprintf(stderr, "Peak smoke test failed: count=%zu\n", n);
        return 3;
    }

    // Synthetic dictionary: each component owns one distinct frequency bin.
    // It tests the solver mechanics without claiming equivalence to the trained W.
    std::vector<float> w(AVS_NUM_BINS * AVS_NUM_COMPONENTS, 0.0f);
    for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) {
        w[(j * 7) * AVS_NUM_COMPONENTS + j] = 1.0f;
    }
    std::vector<float> v(AVS_NUM_BINS, 0.0f);
    float expected[AVS_NUM_COMPONENTS] = {};
    for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) {
        expected[j] = 0.1f + 0.025f * static_cast<float>(j);
        v[j * 7] = expected[j];
    }
    avs_nnls_workspace_t workspace{};
    if (avs_nnls_init(&workspace, w.data()) != AVS_OK) {
        std::fprintf(stderr, "NNLS initialization failed\n");
        return 4;
    }
    float solution[AVS_NUM_COMPONENTS] = {};
    const auto status = avs_solve_nnls(v.data(), w.data(), &workspace,
                                        solution, 200, 1e-6f);
    if (status != AVS_OK) {
        std::fprintf(stderr, "NNLS solver failed: status=%d\n", static_cast<int>(status));
        return 5;
    }
    for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) {
        if (std::fabs(solution[j] - expected[j]) > 1e-4f) {
            std::fprintf(stderr, "NNLS mismatch at %zu: got=%g expected=%g\n",
                         j, solution[j], expected[j]);
            return 6;
        }
    }

    // FFT smoke test: periodic Hann should produce a positive log-magnitude spectrum.
    std::vector<float> frame(AVS_N_FFT, 0.0f);
    for (size_t n = 0; n < AVS_N_FFT; ++n) {
        frame[n] = std::sin(2.0f * 3.14159265358979323846f * 32.0f *
                            static_cast<float>(n) / static_cast<float>(AVS_N_FFT));
    }
    avs_spectrum_t spectrum{};
    if (avs_compute_spectrum(frame.data(), hann.data(), &spectrum) != AVS_OK ||
        !(spectrum.log_magnitude[32] > 5.0f)) {
        std::fprintf(stderr, "FFT/log-magnitude smoke test failed\n");
        return 7;
    }

    std::puts("AVS Stage 1 native smoke tests passed.");
    return 0;
}
