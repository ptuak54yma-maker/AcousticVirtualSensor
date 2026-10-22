#include "avs_pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

extern "C" avs_status_t avs_solve_nnls(const float *v,
                                       const float *w_row_major,
                                       float *h_out,
                                       unsigned max_iterations,
                                       float tolerance) {
    if (!v || !w_row_major || !h_out || max_iterations == 0 ||
        !std::isfinite(tolerance) || tolerance <= 0.0f) {
        return AVS_ERR_ARGUMENT;
    }

    // Fail closed until numerical behaviour is validated. This prevents the
    // test harness from silently presenting an unverified solver as equivalent.
    std::fill(h_out, h_out + AVS_NUM_COMPONENTS, 0.0f);
    return AVS_ERR_UNSUPPORTED;
}

extern "C" float avs_sum_event_activations(const float *h, size_t components) {
    if (!h) return 0.0f;
    const size_t n = std::min(components, static_cast<size_t>(AVS_EVENT_COMPONENTS));
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (std::isfinite(h[i])) sum += h[i];
    }
    return sum;
}
