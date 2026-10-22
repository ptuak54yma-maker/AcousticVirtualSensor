#include "avs_pipeline.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    std::vector<float> hann(AVS_N_FFT);
    avs_make_hann(hann.data(), hann.size());
    if (avs_fft_init() != AVS_OK) {
        std::fprintf(stderr, "FFT initialization failed\n");
        return 12;
    }

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
    const float plateau[] = {0.f, 1.f, 1.f, 1.f, 0.f};
    const size_t plateau_count = avs_detect_peaks(plateau, 5, 0.5f, 1, peaks, 4);
    if (plateau_count != 1 || peaks[0] != 2) {
        std::fprintf(stderr, "Plateau peak test failed: count=%zu index=%zu\n",
                     plateau_count, peaks[0]);
        return 4;
    }

    const float tied[] = {0.f, 2.f, 0.f, 2.f, 0.f};
    const size_t tied_count = avs_detect_peaks(tied, 5, 0.0f, 3, peaks, 5);
    if (tied_count != 1 || peaks[0] != 3) {
        std::fprintf(stderr, "Equal-height distance priority failed: count=%zu index=%zu\n",
                     tied_count, peaks[0]);
        return 9;
    }

    const float many_peaks[] = {0.f, 2.f, 0.f, 1.f, 0.f, 3.f, 0.f};
    const size_t overflow = avs_detect_peaks(many_peaks, 7, 0.0f, 1, peaks, 2);
    if (overflow != AVS_PEAKS_OVERFLOW) {
        std::fprintf(stderr, "Peak capacity overflow was not reported: %zu\n", overflow);
        return 10;
    }

    const float invalid_env[] = {0.f, 1.f, NAN, 1.f, 0.f};
    if (avs_detect_peaks(invalid_env, 5, 0.0f, 1, peaks, 5) != 0) {
        std::fprintf(stderr, "Non-finite envelope was not rejected\n");
        return 11;
    }

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
    avs_nnls_workspace_t nnls_workspace{};
    if (avs_nnls_init(&nnls_workspace, w.data()) != AVS_OK) {
        std::fprintf(stderr, "NNLS initialization failed\n");
        return 5;
    }
    float solution[AVS_NUM_COMPONENTS] = {};
    const auto status = avs_solve_nnls(v.data(), w.data(), &nnls_workspace,
                                        solution, 200, 1e-6f);
    if (status != AVS_OK) {
        std::fprintf(stderr, "NNLS solver failed: status=%d\n", static_cast<int>(status));
        return 6;
    }
    for (size_t j = 0; j < AVS_NUM_COMPONENTS; ++j) {
        if (std::fabs(solution[j] - expected[j]) > 1e-4f) {
            std::fprintf(stderr, "NNLS mismatch at %zu: got=%g expected=%g\n",
                         j, solution[j], expected[j]);
            return 7;
        }
    }

    std::vector<float> frame(AVS_N_FFT, 0.0f);
    for (size_t n = 0; n < AVS_N_FFT; ++n) {
        frame[n] = std::sin(2.0f * 3.14159265358979323846f * 32.0f *
                            static_cast<float>(n) / static_cast<float>(AVS_N_FFT));
    }
    avs_fft_workspace_t fft_workspace{};
    avs_spectrum_t spectrum{};
    if (avs_compute_spectrum(frame.data(), hann.data(), &fft_workspace, &spectrum) != AVS_OK ||
        !(spectrum.log_magnitude[32] > 5.0f)) {
        std::fprintf(stderr, "FFT/log-magnitude smoke test failed\n");
        return 8;
    }

    std::printf("AVS Stage 1 native smoke tests passed (NNLS workspace=%zu bytes).\n",
                sizeof(avs_nnls_workspace_t));
    return 0;
}
