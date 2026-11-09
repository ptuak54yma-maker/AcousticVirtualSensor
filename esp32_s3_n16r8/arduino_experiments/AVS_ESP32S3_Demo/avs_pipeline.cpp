#include "avs_pipeline.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr size_t K = AVS_NUM_COMPONENTS;
constexpr size_t F = AVS_NUM_BINS;
}

extern "C" avs_status_t avs_nnls_init(avs_nnls_workspace_t *workspace,
                                      const float *w_row_major) {
    if (!workspace || !w_row_major) return AVS_ERR_ARGUMENT;
    workspace->initialized = 0u;
    std::fill(workspace->gram, workspace->gram + K * K, 0.0f);

    for (size_t f = 0; f < F; ++f) {
        for (size_t j = 0; j < K; ++j) {
            const float value = w_row_major[f * K + j];
            if (!std::isfinite(value) || value < 0.0f) return AVS_ERR_NUMERIC;
        }
    }

    // W is fixed after initialization, so retain G = W^T W for every frame.
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = i; j < K; ++j) {
            float sum = 0.0f;
            for (size_t f = 0; f < F; ++f) {
                sum += w_row_major[f * K + i] * w_row_major[f * K + j];
            }
            workspace->gram[i * K + j] = sum;
            workspace->gram[j * K + i] = sum;
        }
    }

    // G is symmetric and elementwise nonnegative, so its maximum row sum is
    // a safe upper bound for lambda_max(G), hence for the FISTA Lipschitz L.
    float lipschitz_bound = 0.0f;
    for (size_t i = 0; i < K; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < K; ++j) row_sum += workspace->gram[i * K + j];
        lipschitz_bound = std::max(lipschitz_bound, row_sum);
    }
    if (!(lipschitz_bound > 0.0f) || !std::isfinite(lipschitz_bound)) {
        return AVS_ERR_NUMERIC;
    }

    workspace->lipschitz_bound = lipschitz_bound;
    workspace->step_size = 1.0f / lipschitz_bound;
    workspace->last_iterations = 0u;
    workspace->last_relative_projected_gradient = 0.0f;
    workspace->initialized = 1u;
    return AVS_OK;
}

extern "C" avs_status_t avs_solve_nnls(const float *v,
                                       const float *w_row_major,
                                       avs_nnls_workspace_t *workspace,
                                       float *h_out,
                                       unsigned max_iterations) {
    if (!v || !w_row_major || !workspace || !h_out ||
        workspace->initialized != 1u || max_iterations == 0u ||
        !(workspace->step_size > 0.0f)) {
        return AVS_ERR_ARGUMENT;
    }
    for (size_t f = 0; f < F; ++f) {
        if (!std::isfinite(v[f]) || v[f] < 0.0f) return AVS_ERR_NUMERIC;
    }

    // b = W^T v is frame-dependent; G = W^T W was computed once above.
    std::fill(workspace->rhs, workspace->rhs + K, 0.0f);
    for (size_t j = 0; j < K; ++j) {
        float sum = 0.0f;
        for (size_t f = 0; f < F; ++f) {
            sum += w_row_major[f * K + j] * v[f];
        }
        workspace->rhs[j] = sum;
    }

    std::fill(workspace->solution, workspace->solution + K, 0.0f);
    std::fill(workspace->candidate, workspace->candidate + K, 0.0f);
    std::fill(workspace->momentum, workspace->momentum + K, 0.0f);

    float *x = workspace->solution;
    float *x_next = workspace->candidate;
    float momentum_scale = 1.0f;
    for (unsigned iteration = 0; iteration < max_iterations; ++iteration) {
        const float next_scale =
            0.5f * (1.0f + std::sqrt(1.0f + 4.0f * momentum_scale * momentum_scale));
        const float beta = (momentum_scale - 1.0f) / next_scale;

        // FISTA step: x_next = projection_+(y - (G y - b) / L).
        for (size_t j = 0; j < K; ++j) {
            float gy = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                gy += workspace->gram[j * K + k] * workspace->momentum[k];
            }
            const float projected = workspace->momentum[j] -
                workspace->step_size * (gy - workspace->rhs[j]);
            x_next[j] = std::max(0.0f, projected);
        }

        for (size_t j = 0; j < K; ++j) {
            workspace->momentum[j] = x_next[j] + beta * (x_next[j] - x[j]);
        }
        std::swap(x, x_next);
        momentum_scale = next_scale;
    }

    float max_rhs = 0.0f;
    float max_projected_delta = 0.0f;
    for (size_t j = 0; j < K; ++j) {
        float gh = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            gh += workspace->gram[j * K + k] * x[k];
        }
        const float projected = std::max(
            0.0f, x[j] - workspace->step_size * (gh - workspace->rhs[j]));
        max_projected_delta = std::max(max_projected_delta,
                                       std::fabs(x[j] - projected));
        max_rhs = std::max(max_rhs, std::fabs(workspace->rhs[j]));
        if (!std::isfinite(x[j])) return AVS_ERR_NUMERIC;
        h_out[j] = x[j];
    }

    workspace->last_iterations = max_iterations;
    workspace->last_relative_projected_gradient =
        (max_projected_delta / workspace->step_size) / std::max(1.0f, max_rhs);
    if (!std::isfinite(workspace->last_relative_projected_gradient)) {
        return AVS_ERR_NUMERIC;
    }
    return AVS_OK;
}

extern "C" float avs_sum_event_activations(const float *h, size_t components) {
    if (!h) return 0.0f;
    const size_t n = std::min(components, static_cast<size_t>(AVS_EVENT_COMPONENTS));
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += h[i];
    return sum;
}
