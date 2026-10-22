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
    if (!envelope || !peak_indices || length < 3 || max_peaks == 0) return 0;

    struct Candidate { size_t index; float height; float prom; };
    std::vector<Candidate> candidates;

    // Candidate peaks are strict local maxima; boundaries are not peaks, as in SciPy.
    for (size_t i = 1; i + 1 < length; ++i) {
        if (!(envelope[i] > envelope[i - 1] && envelope[i] > envelope[i + 1])) continue;

        // Simplified prominence estimate: min drop to each side until a higher peak.
        // This intentionally documents an approximation: exact SciPy prominence
        // requires bases calculated using SciPy's contour-line rules.
        float left_min = envelope[i];
        for (size_t j = i; j > 0; --j) {
            const size_t p = j - 1;
            if (envelope[p] > envelope[i]) break;
            left_min = std::min(left_min, envelope[p]);
        }
        float right_min = envelope[i];
        for (size_t j = i + 1; j < length; ++j) {
            if (envelope[j] > envelope[i]) break;
            right_min = std::min(right_min, envelope[j]);
        }
        const float prom = envelope[i] - std::max(left_min, right_min);
        if (prom >= prominence) candidates.push_back({i, envelope[i], prom});
    }

    // SciPy removes the smaller peak first to enforce distance. Greedy selection in
    // descending height is close for ordinary non-tied signals; reference tests needed.
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
    for (size_t i = 0; i < count; ++i) peak_indices[i] = kept[i];
    return count;
}
