#include "avs_pipeline.h"

#include <algorithm>
#include <cmath>
#include <vector>

extern "C" size_t avs_detect_peaks(const float *envelope,
                                  size_t length,
                                  float prominence,
                                  size_t min_distance,
                                  size_t *peak_indices,
                                  size_t max_peaks) {
    if (!envelope || !peak_indices || length < 3 || max_peaks == 0 ||
        !std::isfinite(prominence) || prominence < 0.0f || min_distance == 0) {
        return 0;
    }

    struct Candidate { size_t index; float height; float prom; };
    std::vector<Candidate> candidates;

    // SciPy selects the middle sample of a flat peak (rounded down).
    // Keep all local maxima for distance suppression first: scipy.find_peaks
    // applies distance before prominence filtering.
    size_t i = 1;
    while (i + 1 < length) {
        if (!(envelope[i] > envelope[i - 1])) {
            ++i;
            continue;
        }

        const size_t left_edge = i;
        size_t right_edge = i;
        while (right_edge + 1 < length &&
               envelope[right_edge + 1] == envelope[left_edge]) {
            ++right_edge;
        }
        if (right_edge + 1 >= length ||
            !(envelope[right_edge] > envelope[right_edge + 1])) {
            i = right_edge + 1;
            continue;
        }
        const size_t peak = (left_edge + right_edge) / 2;
        const float height = envelope[peak];

        // Prominence bases: extend to the boundary or until encountering a sample
        // strictly higher than the peak, then take the minimum on each side.
        float left_min = height;
        for (size_t j = peak; j > 0; --j) {
            const size_t p = j - 1;
            if (envelope[p] > height) break;
            left_min = std::min(left_min, envelope[p]);
        }
        float right_min = height;
        for (size_t j = peak + 1; j < length; ++j) {
            if (envelope[j] > height) break;
            right_min = std::min(right_min, envelope[j]);
        }
        const float prom = height - std::max(left_min, right_min);
        candidates.push_back({peak, height, prom});
        i = right_edge + 1;
    }

    // Consider taller peaks first, suppressing lower-priority candidates at
    // distances strictly smaller than min_distance.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.height > b.height; });

    std::vector<Candidate> distance_kept;
    for (const auto &c : candidates) {
        bool too_close = false;
        for (const auto &k : distance_kept) {
            const size_t d = (c.index > k.index) ? c.index - k.index : k.index - c.index;
            if (d < min_distance) { too_close = true; break; }
        }
        if (!too_close) distance_kept.push_back(c);
    }
    std::sort(distance_kept.begin(), distance_kept.end(),
              [](const Candidate& a, const Candidate& b) { return a.index < b.index; });

    size_t count = 0;
    for (const auto &c : distance_kept) {
        if (c.prom < prominence) continue;
        if (count >= max_peaks) break;
        peak_indices[count++] = c.index;
    }
    return count;
}
