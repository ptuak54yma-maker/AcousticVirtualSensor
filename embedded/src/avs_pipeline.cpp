#include "avs_pipeline.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr size_t K = AVS_NUM_COMPONENTS;
constexpr size_t F = AVS_NUM_BINS;

bool cholesky_solve(const float *gram,
                    const float *rhs,
                    const size_t *active_index,
                    size_t n_active,
                    float ridge,
                    float *solution) {
    if (n_active == 0) return true;

    float lower[K * K] = {};
    float y[K] = {};

    for (size_t i = 0; i < n_active; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            float value = gram[active_index[i] * K + active_index[j]];
            if (i == j) value += ridge;
            for (size_t q = 0; q < j; ++q) {
                value -= lower[i * K + q] * lower[j * K + q];
            }
            if (i == j) {
                if (!(value > 1e-20f) || !std::isfinite(value)) return false;
                lower[i * K + j] = std::sqrt(value);
            } else {
                const float denom = lower[j * K + j];
                if (!(denom > 0.0f)) return false;
                lower[i * K + j] = value / denom;
            }
        }
    }

    for (size_t i = 0; i < n_active; ++i) {
        float value = rhs[active_index[i]];
        for (size_t j = 0; j < i; ++j) value -= lower[i * K + j] * y[j];
        y[i] = value / lower[i * K + i];
    }
    for (size_t rev = n_active; rev-- > 0;) {
        float value = y[rev];
        for (size_t j = rev + 1; j < n_active; ++j) {
            value -= lower[j * K + rev] * solution[active_index[j]];
        }
        solution[active_index[rev]] = value / lower[rev * K + rev];
    }
    return true;
}
}

extern "C" avs_status_t avs_nnls_init(avs_nnls_workspace_t *workspace,
                                      const float *w_row_major) {
    if (!workspace || !w_row_major) return AVS_ERR_ARGUMENT;
    std::fill(workspace->gram, workspace->gram + K * K, 0.0f);
    double diag_sum = 0.0;

    for (size_t f = 0; f < F; ++f) {
        for (size_t j = 0; j < K; ++j) {
            const float value = w_row_major[f * K + j];
            if (!std::isfinite(value) || value < 0.0f) {
                workspace->initialized = 0u;
                return AVS_ERR_NUMERIC;
            }
        }
    }

    for (size_t i = 0; i < K; ++i) {
        for (size_t j = i; j < K; ++j) {
            double sum = 0.0;
            for (size_t f = 0; f < F; ++f) {
                sum += static_cast<double>(w_row_major[f * K + i]) *
                       static_cast<double>(w_row_major[f * K + j]);
            }
            workspace->gram[i * K + j] = static_cast<float>(sum);
            workspace->gram[j * K + i] = static_cast<float>(sum);
        }
        diag_sum += workspace->gram[i * K + i];
    }
    workspace->diagonal_scale = static_cast<float>(diag_sum / static_cast<double>(K));
    workspace->initialized = 1u;
    return AVS_OK;
}

extern "C" avs_status_t avs_solve_nnls(const float *v,
                                       const float *w_row_major,
                                       avs_nnls_workspace_t *workspace,
                                       float *h_out,
                                       unsigned max_iterations,
                                       float tolerance) {
    if (!v || !w_row_major || !workspace || !h_out ||
        workspace->initialized != 1u || max_iterations == 0 ||
        !std::isfinite(tolerance) || tolerance <= 0.0f) {
        return AVS_ERR_ARGUMENT;
    }
    for (size_t f = 0; f < F; ++f) {
        if (!std::isfinite(v[f]) || v[f] < 0.0f) return AVS_ERR_NUMERIC;
    }

    float rhs[K] = {};
    for (size_t j = 0; j < K; ++j) {
        double sum = 0.0;
        for (size_t f = 0; f < F; ++f) {
            sum += static_cast<double>(w_row_major[f * K + j]) *
                   static_cast<double>(v[f]);
        }
        rhs[j] = static_cast<float>(sum);
        h_out[j] = 0.0f;
    }

    bool passive[K] = {};
    float gradient[K] = {};
    float candidate[K] = {};
    size_t active_index[K] = {};
    const float rhs_scale = std::max(1.0f, *std::max_element(rhs, rhs + K));
    const float grad_tol = tolerance * rhs_scale;
    const float ridge = std::max(1e-12f, workspace->diagonal_scale * 1e-8f);

    unsigned iterations = 0;
    while (iterations < max_iterations) {
        ++iterations;
        float max_gradient = grad_tol;
        size_t entering = K;

        for (size_t j = 0; j < K; ++j) {
            float prediction = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                prediction += workspace->gram[j * K + k] * h_out[k];
            }
            gradient[j] = rhs[j] - prediction;
            if (!passive[j] && gradient[j] > max_gradient) {
                max_gradient = gradient[j];
                entering = j;
            }
        }
        if (entering == K) {
            for (size_t j = 0; j < K; ++j) {
                if (!std::isfinite(h_out[j]) || h_out[j] < 0.0f) return AVS_ERR_NUMERIC;
            }
            return AVS_OK;
        }

        passive[entering] = true;

        bool inner_done = false;
        unsigned inner_iterations = 0;
        while (!inner_done && inner_iterations < K + 1 && iterations < max_iterations) {
            ++inner_iterations;
            ++iterations;
            size_t active_count = 0;
            for (size_t j = 0; j < K; ++j) {
                if (passive[j]) active_index[active_count++] = j;
                candidate[j] = 0.0f;
            }

            bool solved = cholesky_solve(workspace->gram, rhs, active_index,
                                         active_count, 0.0f, candidate);
            if (!solved) {
                std::fill(candidate, candidate + K, 0.0f);
                solved = cholesky_solve(workspace->gram, rhs, active_index,
                                         active_count, ridge, candidate);
            }
            if (!solved) return AVS_ERR_NUMERIC;

            bool all_positive = true;
            for (size_t p = 0; p < active_count; ++p) {
                if (candidate[active_index[p]] <= 0.0f) {
                    all_positive = false;
                    break;
                }
            }
            if (all_positive) {
                for (size_t p = 0; p < active_count; ++p) {
                    const size_t j = active_index[p];
                    h_out[j] = candidate[j];
                }
                inner_done = true;
            } else {
                float alpha = 1.0f;
                for (size_t p = 0; p < active_count; ++p) {
                    const size_t j = active_index[p];
                    if (candidate[j] <= 0.0f) {
                        const float denom = h_out[j] - candidate[j];
                        if (denom > 0.0f) alpha = std::min(alpha, h_out[j] / denom);
                    }
                }
                for (size_t p = 0; p < active_count; ++p) {
                    const size_t j = active_index[p];
                    h_out[j] += alpha * (candidate[j] - h_out[j]);
                    if (h_out[j] <= tolerance) {
                        h_out[j] = 0.0f;
                        passive[j] = false;
                    }
                }
            }
        }
        if (!inner_done) return AVS_ERR_NO_CONVERGENCE;
    }
    return AVS_ERR_NO_CONVERGENCE;
}

extern "C" float avs_sum_event_activations(const float *h, size_t components) {
    if (!h) return 0.0f;
    const size_t n = std::min(components, static_cast<size_t>(AVS_EVENT_COMPONENTS));
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += h[i];
    return sum;
}
