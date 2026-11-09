#include "avs_pipeline.h"
#include "w_standard_mel_data.h"

#include "dsps_dotprod.h"
#include "dsps_fft2r.h"
#include "esp_err.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr size_t K = AVS_NUM_COMPONENTS;
constexpr size_t F = AVS_NUM_BINS;
constexpr float kPi = 3.14159265358979323846f;

// ESP-DSP stores complex samples as Re[0], Im[0], Re[1], Im[1], ... .
// Keep this staging buffer out of the task stack and avoid per-frame allocs.
alignas(16) float g_fft_interleaved[2u * AVS_N_FFT]{};
bool g_fft_ready = false;

bool dot_product(const float *a, const float *b, int length, float *out) {
    *out = 0.0f;
    return dsps_dotprod_f32(a, b, out, length) == ESP_OK;
}
}

extern "C" void avs_make_hann(float *window, size_t length) {
    if (!window || length == 0) return;
    for (size_t n = 0; n < length; ++n) {
        window[n] = 0.5f - 0.5f * std::cos(2.0f * kPi * n / length);
    }
}

extern "C" avs_status_t avs_fft_init(void) {
    if (g_fft_ready) return AVS_OK;
    if (dsps_fft2r_init_fc32(nullptr, static_cast<int>(AVS_N_FFT)) != ESP_OK) {
        return AVS_ERR_UNSUPPORTED;
    }
    g_fft_ready = true;
    return AVS_OK;
}

extern "C" avs_status_t avs_compute_spectrum(
    const float *frame, const float *window, avs_fft_workspace_t *workspace,
    avs_spectrum_t *out) {
    if (!frame || !window || !workspace || !out) return AVS_ERR_ARGUMENT;
    if (!g_fft_ready) return AVS_ERR_UNSUPPORTED;

    for (size_t n = 0; n < AVS_N_FFT; ++n) {
        g_fft_interleaved[2u * n] = frame[n] * window[n];
        g_fft_interleaved[2u * n + 1u] = 0.0f;
    }
    if (dsps_fft2r_fc32(g_fft_interleaved, static_cast<int>(AVS_N_FFT)) != ESP_OK ||
        dsps_bit_rev_fc32(g_fft_interleaved, static_cast<int>(AVS_N_FFT)) != ESP_OK) {
        return AVS_ERR_NUMERIC;
    }

    for (size_t mel = 0; mel < AVS_NUM_BINS; ++mel) {
        float power = 0.0f;
        for (size_t i = AVS_MEL_OFFSETS[mel]; i < AVS_MEL_OFFSETS[mel + 1u]; ++i) {
            const size_t k = AVS_MEL_BIN_INDEX[i];
            const float re = g_fft_interleaved[2u * k];
            const float im = g_fft_interleaved[2u * k + 1u];
#if AVS_MEL_INPUT_POWER
            power += AVS_MEL_WEIGHTS[i] * (re * re + im * im);
#else
            power += AVS_MEL_WEIGHTS[i] * std::sqrt(re * re + im * im);
#endif
        }
#if AVS_MEL_COMPRESSOR_LOG1P
        out->features[mel] = std::log1p(power / AVS_MEL_REFERENCE);
#elif AVS_MEL_COMPRESSOR_POWER
        out->features[mel] = std::pow(power / AVS_MEL_REFERENCE, AVS_MEL_POWER_GAMMA);
#else
        out->features[mel] = power / AVS_MEL_REFERENCE;
#endif
    }
    return AVS_OK;
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

    // G = W^T W is symmetric and fixed for every audio frame.
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
        !(workspace->step_size > 0.0f)) return AVS_ERR_ARGUMENT;
    for (size_t f = 0; f < F; ++f) {
        if (!std::isfinite(v[f]) || v[f] < 0.0f) return AVS_ERR_NUMERIC;
    }

    // Strided ESP-DSP dot products exploit row-major W without transposing it.
    for (size_t j = 0; j < K; ++j) {
        float rhs = 0.0f;
        if (dsps_dotprode_f32(w_row_major + j, v, &rhs,
                              static_cast<int>(F), static_cast<int>(K), 1) != ESP_OK ||
            !std::isfinite(rhs)) return AVS_ERR_NUMERIC;
        workspace->rhs[j] = rhs;
    }

    std::fill(workspace->solution, workspace->solution + K, 0.0f);
    std::fill(workspace->candidate, workspace->candidate + K, 0.0f);
    std::fill(workspace->momentum, workspace->momentum + K, 0.0f);
    float *x = workspace->solution;
    float *x_next = workspace->candidate;
    float momentum_scale = 1.0f;

    for (unsigned iteration = 0; iteration < max_iterations; ++iteration) {
        const float next_scale = 0.5f *
            (1.0f + std::sqrt(1.0f + 4.0f * momentum_scale * momentum_scale));
        const float beta = (momentum_scale - 1.0f) / next_scale;
        for (size_t j = 0; j < K; ++j) {
            float gy = 0.0f;
            if (!dot_product(workspace->gram + j * K, workspace->momentum,
                             static_cast<int>(K), &gy)) return AVS_ERR_NUMERIC;
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
        if (!dot_product(workspace->gram + j * K, x, static_cast<int>(K), &gh)) {
            return AVS_ERR_NUMERIC;
        }
        const float projected = std::max(
            0.0f, x[j] - workspace->step_size * (gh - workspace->rhs[j]));
        max_projected_delta = std::max(max_projected_delta, std::fabs(x[j] - projected));
        max_rhs = std::max(max_rhs, std::fabs(workspace->rhs[j]));
        if (!std::isfinite(x[j])) return AVS_ERR_NUMERIC;
        h_out[j] = x[j];
    }
    workspace->last_iterations = max_iterations;
    workspace->last_relative_projected_gradient =
        (max_projected_delta / workspace->step_size) / std::max(1.0f, max_rhs);
    return std::isfinite(workspace->last_relative_projected_gradient)
        ? AVS_OK : AVS_ERR_NUMERIC;
}

extern "C" float avs_sum_event_activations(const float *h, size_t components) {
    if (!h) return 0.0f;
    const size_t n = std::min(components, static_cast<size_t>(AVS_EVENT_COMPONENTS));
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += h[i];
    return sum;
}
