#include "avs_pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
constexpr size_t K = AVS_NUM_COMPONENTS;
constexpr size_t F = AVS_NUM_BINS;

bool cholesky_solve(const double *gram,
                    const double *rhs,
                    const size_t *active_index,
                    size_t n_active,
                    double ridge,
                    double *solution,
                    double *lower,
                    double *y) {
    if (n_active == 0) return true;

    for (size_t i = 0; i < n_active; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double value = gram[active_index[i] * K + active_index[j]];
            if (i == j) value += ridge;
            for (size_t q = 0; q < j; ++q) {
                value -= lower[i * K + q] * lower[j * K + q];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) return false;
                lower[i * K + j] = std::sqrt(value);
            } else {
                const double denom = lower[j * K + j];
                if (!(denom > 0.0)) return false;
                lower[i * K + j] = value / denom;
            }
        }
    }

    for (size_t i = 0; i < n_active; ++i) {
        double value = rhs[active_index[i]];
        for (size_t j = 0; j < i; ++j) value -= lower[i * K + j] * y[j];
        y[i] = value / lower[i * K + i];
    }
    for (size_t rev = n_active; rev-- > 0;) {
        double value = y[rev];
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
    workspace->initialized = 0u;
    std::fill(workspace->gram, workspace->gram + K * K, 0.0);

    double max_column_l1 = 0.0;
    for (size_t j = 0; j < K; ++j) {
        double column_l1 = 0.0;
        for (size_t f = 0; f < F; ++f) {
            const float value = w_row_major[f * K + j];
            if (!std::isfinite(value) || value < 0.0f) return AVS_ERR_NUMERIC;
            column_l1 += std::fabs(static_cast<double>(value));
        }
        max_column_l1 = std::max(max_column_l1, column_l1);
    }

    double minimum_diagonal = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = i; j < K; ++j) {
            double sum = 0.0;
            for (size_t f = 0; f < F; ++f) {
                sum += static_cast<double>(w_row_major[f * K + i]) *
                       static_cast<double>(w_row_major[f * K + j]);
            }
            workspace->gram[i * K + j] = sum;
            workspace->gram[j * K + i] = sum;
        }
        minimum_diagonal = std::min(minimum_diagonal,
                                    workspace->gram[i * K + i]);
    }

    // scipy.optimize.nnls uses the Lawson-Hanson active-set method with a
    // projected-gradient tolerance derived from ||W||_1 and double precision.
    // Mirror that scale instead of multiplying the caller epsilon by ||W^T v||.
    const double scipy_style_tolerance =
        10.0 * static_cast<double>(std::max(F, K)) *
        std::numeric_limits<double>::epsilon() * max_column_l1;
    workspace->projected_gradient_tolerance = scipy_style_tolerance;
    workspace->diagonal_scale = minimum_diagonal;
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

    std::fill(workspace->rhs, workspace->rhs + K, 0.0);
    std::fill(workspace->solution, workspace->solution + K, 0.0);
    std::fill(workspace->passive, workspace->passive + K,
              static_cast<uint8_t>(0));
    std::fill(h_out, h_out + K, 0.0f);
    for (size_t j = 0; j < K; ++j) {
        double sum = 0.0;
        for (size_t f = 0; f < F; ++f) {
            sum += static_cast<double>(w_row_major[f * K + j]) *
                   static_cast<double>(v[f]);
        }
        workspace->rhs[j] = sum;
    }

    const double grad_tol = std::max(
        static_cast<double>(tolerance), workspace->projected_gradient_tolerance);
    const double ridge = std::max(1e-24, workspace->diagonal_scale * 1e-12);
    unsigned iterations = 0;
    while (iterations < max_iterations) {
        ++iterations;
        double max_gradient = grad_tol;
        size_t entering = K;

        for (size_t j = 0; j < K; ++j) {
            double prediction = 0.0;
            for (size_t k = 0; k < K; ++k) {
                prediction += workspace->gram[j * K + k] *
                              workspace->solution[k];
            }
            workspace->gradient[j] = workspace->rhs[j] - prediction;
            if (!workspace->passive[j] &&
                workspace->gradient[j] > max_gradient) {
                max_gradient = workspace->gradient[j];
                entering = j;
            }
        }
        if (entering == K) {
            for (size_t j = 0; j < K; ++j) {
                if (!std::isfinite(workspace->solution[j]) ||
                    workspace->solution[j] < 0.0) {
                    return AVS_ERR_NUMERIC;
                }
                h_out[j] = static_cast<float>(workspace->solution[j]);
            }
            return AVS_OK;
        }

        workspace->passive[entering] = 1u;
        bool inner_done = false;
        unsigned inner_iterations = 0;
        while (!inner_done && inner_iterations < K + 1 &&
               iterations < max_iterations) {
            ++inner_iterations;
            ++iterations;
            size_t active_count = 0;
            for (size_t j = 0; j < K; ++j) {
                if (workspace->passive[j]) {
                    workspace->active_index[active_count++] = j;
                }
                workspace->candidate[j] = 0.0;
            }

            bool solved = cholesky_solve(workspace->gram, workspace->rhs,
                                         workspace->active_index, active_count,
                                         0.0, workspace->candidate,
                                         workspace->lower, workspace->solve_y);
            if (!solved) {
                std::fill(workspace->candidate,
                          workspace->candidate + K, 0.0);
                solved = cholesky_solve(workspace->gram, workspace->rhs,
                                        workspace->active_index, active_count,
                                        ridge, workspace->candidate,
                                        workspace->lower, workspace->solve_y);
            }
            if (!solved) return AVS_ERR_NUMERIC;

            bool all_positive = true;
            for (size_t p = 0; p < active_count; ++p) {
                if (workspace->candidate[workspace->active_index[p]] <= 0.0) {
                    all_positive = false;
                    break;
                }
            }
            if (all_positive) {
                for (size_t p = 0; p < active_count; ++p) {
                    const size_t j = workspace->active_index[p];
                    workspace->solution[j] = workspace->candidate[j];
                }
                inner_done = true;
            } else {
                double alpha = 1.0;
                for (size_t p = 0; p < active_count; ++p) {
                    const size_t j = workspace->active_index[p];
                    if (workspace->candidate[j] <= 0.0) {
                        const double denom =
                            workspace->solution[j] - workspace->candidate[j];
                        if (denom > 0.0) {
                            alpha = std::min(alpha,
                                workspace->solution[j] / denom);
                        }
                    }
                }
                for (size_t p = 0; p < active_count; ++p) {
                    const size_t j = workspace->active_index[p];
                    workspace->solution[j] += alpha *
                        (workspace->candidate[j] - workspace->solution[j]);
                    if (workspace->solution[j] <= 0.0) {
                        workspace->solution[j] = 0.0;
                        workspace->passive[j] = 0u;
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
