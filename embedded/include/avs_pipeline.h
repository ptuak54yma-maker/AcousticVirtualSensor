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

/* Explicit FFT scratch storage: 16 KiB. Allocate statically in MCU firmware. */
typedef struct {
    float real[AVS_N_FFT];
    float imag[AVS_N_FFT];
} avs_fft_workspace_t;

typedef struct {
    float log_magnitude[AVS_NUM_BINS];
} avs_spectrum_t;

/* Periodic Hann: scipy.signal.get_window("hann", N, fftbins=True). */
void avs_make_hann(float *window, size_t length);
avs_status_t avs_fft_init(void);

/* Apply periodic Hann, unscaled real FFT, then log1p(abs(X)).
 * Does not remove DC: caller must reproduce the chosen Python entry point's
 * DC-removal policy (the current streaming engine removes the frame mean).
 */
avs_status_t avs_compute_spectrum(const float *frame,
                                  const float *window,
                                  avs_fft_workspace_t *workspace,
                                  avs_spectrum_t *out);

/* Caller-owned reusable solver workspace. It holds double-precision Gram
 * and active-set/Cholesky scratch (about 38.5 KiB at K=48).
 * Allocate it statically or
 * in persistent storage, never as an automatic local on a small MCU stack.
 * A workspace is mutable and not reentrant: use one per concurrent solver.
 */
typedef struct {
    double gram[AVS_NUM_COMPONENTS * AVS_NUM_COMPONENTS];
    double lower[AVS_NUM_COMPONENTS * AVS_NUM_COMPONENTS];
    double rhs[AVS_NUM_COMPONENTS];
    double solve_y[AVS_NUM_COMPONENTS];
    double gradient[AVS_NUM_COMPONENTS];
    double candidate[AVS_NUM_COMPONENTS];
    double solution[AVS_NUM_COMPONENTS];
    size_t active_index[AVS_NUM_COMPONENTS];
    uint8_t passive[AVS_NUM_COMPONENTS];
    double diagonal_scale;
    double projected_gradient_tolerance;
    uint32_t initialized;
} avs_nnls_workspace_t;

avs_status_t avs_nnls_init(avs_nnls_workspace_t *workspace,
                           const float *w_row_major);

avs_status_t avs_solve_nnls(const float *v,
                            const float *w_row_major,
                            avs_nnls_workspace_t *workspace,
                            float *h_out,
                            unsigned max_iterations,
                            float tolerance);

float avs_sum_event_activations(const float *h, size_t components);

/* Batch peak picking over a complete finite envelope. `peak_indices` is both
 * the scratch buffer and output, with capacity `max_peaks`; no heap is used.
 * Returns the count, 0 for invalid/empty input, or AVS_PEAKS_OVERFLOW if the
 * buffer cannot hold every local-max candidate (output is invalid on overflow).
 * The caller owns the capacity and must provision it for the maximum candidate
 * count it expects. SciPy-style plateau selection and distance priority apply.
 */
#define AVS_PEAKS_OVERFLOW ((size_t)-1)
size_t avs_detect_peaks(const float *envelope,
                        size_t length,
                        float prominence,
                        size_t min_distance,
                        size_t *peak_indices,
                        size_t max_peaks);

#ifdef __cplusplus
}
#endif

#endif
