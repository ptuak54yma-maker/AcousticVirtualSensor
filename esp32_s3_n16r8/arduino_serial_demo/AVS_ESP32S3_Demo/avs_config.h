#ifndef AVS_CONFIG_H
#define AVS_CONFIG_H

#define AVS_SAMPLE_RATE       44100u
#define AVS_N_FFT             2048u
#define AVS_HOP_LENGTH        1024u
#define AVS_NUM_BINS          1025u
#define AVS_NUM_COMPONENTS    48u
#define AVS_EVENT_COMPONENTS  24u
#define AVS_PEAK_DISTANCE     4u
/* Initial embedded baseline. Replace with the project's trained peak settings
 * after comparing this device's envelope against the Python reference. */
#define AVS_PEAK_PROMINENCE   0.01f

#endif
