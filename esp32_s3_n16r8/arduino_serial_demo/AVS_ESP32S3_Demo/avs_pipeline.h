#ifndef AVS_PIPELINE_H
#define AVS_PIPELINE_H

#include <stddef.h>
#include <stdint.h>
#include "avs_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AVS_OK = 0,
    AVS_ERR_ARGUMENT = -1,
    AVS_ERR_NUMERIC = -2,
    AVS_ERR_UNSUPPORTED = -3,
    AVS_ERR_NO_CONVERGENCE = -4
} avs_status_t;

typedef struct {
    float real[AVS_N_FFT];
    float imag[AVS_N_FFT];
} avs_fft_workspace_t;

typedef struct { float log_magnitude[AVS_NUM_BINS]; } avs_spectrum_t;

void avs_make_hann(float *window, size_t length);
avs_status_t avs_fft_init(void);
avs_status_t avs_prepare_fft_input(const float *frame, const float *window,
                                  avs_fft_workspace_t *workspace);
avs_status_t avs_execute_fft(avs_fft_workspace_t *workspace);
avs_status_t avs_compute_log_spectrum(const avs_fft_workspace_t *workspace,
                                      avs_spectrum_t *out);
avs_status_t avs_compute_spectrum(const float *frame, const float *window,
                                  avs_fft_workspace_t *workspace,
                                  avs_spectrum_t *out);

typedef struct {
    float gram[AVS_NUM_COMPONENTS * AVS_NUM_COMPONENTS];
    float rhs[AVS_NUM_COMPONENTS];
    float solution[AVS_NUM_COMPONENTS];
    float candidate[AVS_NUM_COMPONENTS];
    float momentum[AVS_NUM_COMPONENTS];
    float lipschitz_bound;
    float step_size;
    float last_relative_projected_gradient;
    uint32_t last_iterations;
    uint32_t initialized;
} avs_nnls_workspace_t;

avs_status_t avs_nnls_init(avs_nnls_workspace_t *workspace,
                           const float *w_row_major);
avs_status_t avs_solve_nnls(const float *v, const float *w_row_major,
                            avs_nnls_workspace_t *workspace, float *h_out,
                            unsigned max_iterations);
float avs_sum_event_activations(const float *h, size_t components);

#define AVS_PEAKS_OVERFLOW ((size_t)-1)
size_t avs_detect_peaks(const float *envelope, size_t length,
                        float prominence, size_t min_distance,
                        size_t *peak_indices, size_t max_peaks);

#ifdef __cplusplus
}
#endif
#endif
