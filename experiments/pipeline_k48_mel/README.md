# Retrained W standard: 48 components with Mel features

This candidate preserves the standard component layout (24 event, 12 bowl
noise, 12 environmental noise) while reducing the frequency feature rows from
1,025 to 64 Mel bands. The original `models/nmf/W_standard.npy` remains
unchanged; the retrained matrix is saved as
`models/nmf/W_standard_mel.npy` and as a model bundle in this folder.

The front end uses 44.1 kHz mono audio, a 2,048-sample periodic Hann window,
1,024-sample hop, magnitude spectrum, Slaney-normalized Mel filterbank from 20 Hz
to Nyquist, and `log1p` compression. The compressor reference is estimated
from the training-source WAVs and saved with the model. W columns are
L2-normalized after fitting; this preserves the factorization scale freedom and
improves the FISTA solver's numeric conditioning. Firmware uses the same
filterbank and reference embedded in a generated header.

## Train, evaluate, and export

Run these from the project root in a Python environment with `requirements.txt`:

```powershell
python experiments/pipeline_k48_mel/train_dictionary.py
python experiments/pipeline_k48_mel/compare_iterations.py
python esp32_s3_n16r8/tools/export_mel_dictionary.py
```

Training fits separate event, bowl-noise, and environmental-noise dictionaries
with 24, 12, and 12 components, then concatenates them into W `(64, 48)`. To
limit training time while including every recording, it samples up to 2,048
frames per training WAV. It writes the bundled model to `models/` under this
folder and the standalone W to `models/nmf/W_standard_mel.npy`.

The comparison applies SciPy NNLS as the Python reference and the firmware's
projected FISTA solver at 8, 16, 32, 64, 128, 256, 512, 1,024, 2,048, and
4,096 iterations. Peak
prominence and distance are tuned only on a fixed 14-recording calibration
split; results are reported separately for calibration, four held-out
validation recordings, and the labeled test recordings. Outputs include
per-file counts, mean count accuracy, MAE, selected thresholds, FISTA
activations, and binary Mel feature files usable by the C++ host evaluator.

Count accuracy uses the project's metric:
`max(0, 1 - abs(predicted - true) / true) * 100%`.

## Firmware build

The current sweep selected 512 FISTA iterations as the lowest count that
matched the Python reference's validation accuracy. On the held-out test set,
the MCU-style FISTA achieved 97.23% mean count accuracy versus 94.91% for the
SciPy Python reference; the existing linear-frequency Python baseline scored
96.60%. Validation accuracy was 98.11% for both updated solvers. Test labels
were not used to select iterations or peak thresholds.

After training and comparison, the selected FISTA count and thresholds are
already applied in `esp32_s3_n16r8/espidf_experiments/avs_config.h`. Export the W and Mel
filterbank with `python esp32_s3_n16r8/tools/export_mel_dictionary.py`, then
build from `esp32_s3_n16r8` with ESP-IDF:

```powershell
idf.py set-target esp32s3
idf.py build
```

The resulting `build/avs_esp32s3_n16r8.bin` is ready to flash when the ESP32-S3
is available. This host-side comparison does not substitute for measuring
real-time performance or checking microphone I2S alignment on the board.
