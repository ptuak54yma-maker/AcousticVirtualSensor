#include "avs_pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr size_t kSuppressed =
    static_cast<size_t>(1) << (sizeof(size_t) * 8u - 1u);
constexpr size_t kIndexMask = kSuppressed - 1u;

float peak_prominence(const float *envelope, size_t length, size_t peak) {
    const float height = envelope[peak];
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
    return height - std::max(left_min, right_min);
}
}

extern "C" size_t avs_detect_peaks(const float *envelope,
                                    size_t length,
                                    float prominence,
                                    size_t min_distance,
                                    size_t *peak_indices,
                                    size_t max_peaks) {
    if (!envelope || !peak_indices || length < 3 || max_peaks == 0 ||
        !std::isfinite(prominence) || prominence < 0.0f || min_distance == 0 ||
        length >= kSuppressed) {
        return 0;
    }
    for (size_t i = 0; i < length; ++i) {
        if (!std::isfinite(envelope[i])) return 0;
    }

    // Collect every local maximum before applying prominence. SciPy applies
    // distance filtering before prominence filtering, and the caller-provided
    // output array doubles as bounded scratch storage.
    size_t candidate_count = 0;
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
        if (right_edge + 1 < length &&
            envelope[right_edge] > envelope[right_edge + 1]) {
            if (candidate_count == max_peaks) return AVS_PEAKS_OVERFLOW;
            // SciPy picks the lower middle index for a flat peak.
            peak_indices[candidate_count++] = (left_edge + right_edge) / 2;
        }
        i = right_edge + 1;
    }

    // Highest peaks get distance priority; on a tie SciPy's distance selector
    // processes the later index first. std::sort on this caller buffer does not
    // allocate heap storage.
    std::sort(peak_indices, peak_indices + candidate_count,
              [envelope](size_t a, size_t b) {
                  if (envelope[a] != envelope[b]) return envelope[a] > envelope[b];
                  return a > b;
              });

    // Mark suppressed candidate indices in their top bit. The guard above
    // reserves that bit and makes the bound explicit for every target width.
    for (size_t p = 0; p < candidate_count; ++p) {
        const size_t peak = peak_indices[p] & kIndexMask;
        bool suppressed = false;
        for (size_t q = 0; q < p; ++q) {
            if ((peak_indices[q] & kSuppressed) != 0u) continue;
            const size_t kept = peak_indices[q] & kIndexMask;
            const size_t distance = peak > kept ? peak - kept : kept - peak;
            if (distance < min_distance) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) peak_indices[p] |= kSuppressed;
    }

    size_t output_count = 0;
    for (size_t p = 0; p < candidate_count; ++p) {
        if ((peak_indices[p] & kSuppressed) != 0u) continue;
        const size_t peak = peak_indices[p] & kIndexMask;
        if (peak_prominence(envelope, length, peak) >= prominence) {
            peak_indices[output_count++] = peak;
        }
    }
    std::sort(peak_indices, peak_indices + output_count);
    return output_count;
}
