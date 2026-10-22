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
    AVS_ERR_UNSUPPORTED = -3
} avs_status_t;

typedef struct {
    float real;
    float imag;
} avs_complex_f32_t;

typedef struct {
    float magnitude[AVS_NUM_BINS];
    float log_magnitude[AVS_NUM_BINS];
} avs_spectrum_t;

/* Creates periodic Hann coefficients: w[n] = 0.5 - 0.5*cos(2*pi*n/N). */
void avs_make_hann(float *window, size_t length);

/* Compute real FFT and log1p(abs(X)); the implementation is supplied by avs_fft.cpp.
 * Complex FFT outputs are optional for first-stage tests; log_magnitude is primary.
 */
avs_status_t avs_compute_spectrum(const float *frame,
                                  const float *window,
                                  avs_spectrum_t *out);

/* Solve one frame min ||v - W h||_2^2, h >= 0.
 * W is row-major [freq_bin][component], shape 1025 x 48.
 * This is a deterministic projected-gradient implementation for the first port;
 * it is NOT claimed numerically identical to scipy.optimize.nnls until reference
 * vectors have been compared. Returns AVS_ERR_UNSUPPORTED until enabled.
 */
avs_status_t avs_solve_nnls(const float *v,
                            const float *w_row_major,
                            float *h_out,
                            unsigned max_iterations,
                            float tolerance);

float avs_sum_event_activations(const float *h, size_t components);

/* Offline batch peak detection: implemented semantics are documented in avs_peak.h.
 * To exactly match SciPy find_peaks, compare reference vectors because edge and
 * equal-height distance tie behavior can differ.
 */
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
