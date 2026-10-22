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
        if (prom >= prominence) candidates.push_back({peak, height, prom});
        i = right_edge + 1;
    }

    // Match the usual SciPy distance rule: consider taller peaks first, then remove
    // any lower-priority peak whose index is closer than min_distance.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.height > b.height; });

    std::vector<size_t> kept;
    for (const auto &c : candidates) {
        bool too_close = false;
        for (const size_t k : kept) {
            const size_t d = (c.index > k) ? c.index - k : k - c.index;
            if (d < min_distance) { too_close = true; break; }
        }
        if (!too_close) kept.push_back(c.index);
    }
    std::sort(kept.begin(), kept.end());

    const size_t count = std::min(kept.size(), max_peaks);
    for (size_t p = 0; p < count; ++p) peak_indices[p] = kept[p];
    return count;
}
