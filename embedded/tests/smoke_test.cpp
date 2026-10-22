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

    std::puts("AVS Stage 1 native smoke tests passed.");
    return 0;
}
