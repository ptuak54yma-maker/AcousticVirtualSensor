# Experimental NMF pipeline: K=24 with Mel front end

This folder is a separate experiment. It does not replace or modify the
existing `src/` pipeline or `models/nmf/W_standard.npy` (K=48). The candidate
dictionary has 64 Mel bands and 24 components in this order:

| Rows in H | Source | Components |
|---|---|---:|
| 0–11 | Part/event sound | 12 |
| 12–17 | Bowl noise | 6 |
| 18–23 | Environmental noise | 6 |

The front end uses the project's 44.1 kHz, 2,048 point Hann STFT and 1,024
sample hop, projects the **linear power spectrogram** through a Slaney
normalized Mel filterbank (64 bands, 20 Hz to Nyquist), then compresses it.
NMF training and inference reuse the same saved filterbank and compressor
metadata from the experiment model bundle.

## Compression choice

The default is `log1p(MelPower / reference)`. The reference is one robust global
scale estimated as the median of each training WAV's 95th percentile and saved
with the dictionary. This keeps the input finite and nonnegative, and avoids
letting arbitrary WAV amplitude units set the logarithm's operating range.
Always use that same saved reference at inference; do not refit it on test audio.

Because log is nonlinear, it weakens the additive-mixture assumption of NMF.
For the comparison, `config.py` also supports `COMPRESSOR = "power"` with
`POWER_GAMMA = 0.5` (square-root compression) and `COMPRESSOR = "linear"`.
The linear Mel-power version is the most directly consistent with additive NMF;
the square-root and log versions trade that consistency for lower dynamic range.
Train a separate dictionary for each compressor and compare reconstruction,
activation stability, and event detection on the same recordings.

## Train and run

From the project root, after installing the project's `requirements.txt`:

```powershell
python experiments/pipeline_k24_mel/train_dictionary.py
python experiments/pipeline_k24_mel/pipeline.py data/test/Bowl_3.wav
```

Training uses `data/train/phoi_*.wav`, `noisebowl_*.wav`, and
`noiseevir_*.wav`. It saves only inside this experiment folder:
`experiments/pipeline_k24_mel/models/W_k24_mel.npz`.

The inference output stores compressed Mel features, the full 24-row H matrix,
and separate event, bowl-noise, and environmental-noise activation sums. By
default, the feature bundle is written to
`experiments/pipeline_k24_mel/results/features/<wav-name>.k24_mel_features.npz`
and rerunning that WAV overwrites the same output. No model is trained or
selected by adding this code; it remains an evaluation candidate until
compared with the current pipeline.

## One-command comparison against the current pipeline

From the project root, run:

```powershell
python experiments/pipeline_k24_mel/run_comparison.py
```

If `W_k24_mel.npz` does not exist, the command trains it first. It makes a
deterministic 14/4 calibration/validation split of the 18 labeled training
recordings. It tunes P (peak prominence) and D (minimum peak distance) for
both pipelines on the 14 calibration recordings, then checks the selected
values on the four untouched validation recordings. Finally it reports the
same count metrics on `data/test/*.wav`; test labels are never used to tune
P/D. The original K=48 model and its deployed peak settings remain unchanged.
All generated comparison outputs go to the fixed
`experiments/pipeline_k24_mel/results/` folder. Each comparison clears and
recreates that folder, so the next run replaces the previous results. The
canonical K=24 model remains at `models/W_k24_mel.npz`; it is not copied into
the results folder. Do not use `--retrain` or preprocessing overrides if you
want to keep the current model unchanged.

K=24 is marked **PASS** only if its mean count accuracy is no lower than both
K=48 deployed and recalibrated settings on validation, no lower than deployed
K=48 on test, and all three source NMF fits converged. Otherwise the result is
marked **HOLD**. A held-out test regression therefore blocks promotion; it is
not hidden by selecting parameters against test labels.

To deliberately test another compressor or Mel input, pass an override. This
re-trains and replaces the canonical K=24 model, and overwrites the fixed
results folder:

```powershell
python experiments/pipeline_k24_mel/run_comparison.py --compressor power
python experiments/pipeline_k24_mel/run_comparison.py --compressor linear
python experiments/pipeline_k24_mel/run_comparison.py --mel-input magnitude
```

Available compressor choices are `log1p`, `power` (square root by default),
and `linear`; Mel input choices are `power` and `magnitude`. The fixed
`results/` folder contains:
- `peak_params_k24_mel.json`: training-calibrated new peak settings.
- `summary_metrics.csv`: held-out count MAE, MAPE and accuracy, by
  pipeline.
- `per_file_metrics.csv`: predictions and errors for every test file.
- `validation_metrics.csv`: per-file validation counts for old deployed K=48,
  recalibrated K=48, and K=24.
- `training_calibration_metrics.csv`: calibration counts used to select P/D
  for both pipelines, kept separate from validation and test metrics.
- `train_peak_parameter_comparison.csv` and
  `train_peak_parameter_summary.csv`: per-file and aggregate K=48 peak counts
  using original versus calibrated P/D on the same H_event_sum.
- `peak_params_k48_calibrated.json` and `peak_params_k24_mel.json`: selected
  P/D values and calibration MAE for each pipeline.
- `p_d_mae_matrix_k24_mel.csv` and
  `p_d_mae_matrix_k48_original_vs_calibrated.csv`: full wide-form search
  matrices; each row is P, each column is D, and each cell is calibration MAE.
- `p_d_mae_heatmap_k24_mel.png` and
  `p_d_mae_heatmap_k48_original_vs_calibrated.png`: visualizations of those
  search matrices; the K=48 heatmap marks both original and calibrated P/D.
- `pipeline_diagnostics.csv` and `validation_pipeline_diagnostics.csv`:
  per-recording H/event-envelope distributions, event H sum, event/bowl/env
  activation statistics, detected-peak prominence/height/intervals,
  reconstruction residuals, feature statistics, memory sizes, and host timing
  for STFT/Mel front end, compression, NNLS, and peak detection.
- `diagnostic_summary.csv`: means of numeric test diagnostics grouped by model.
- `promotion_gate.json`: explicit PASS/HOLD results and the accuracy/convergence
  checks behind them.
- `activations/`: full H matrices, event envelopes and detected peak
  frame indices for each split/model.
- `plots/`: old/new event-envelope and peak overlays. Test files are directly
  under `plots/`; all labeled training files are under `plots/train_calibration/`
  or `plots/validation/`. `plots/train_peak_params/` overlays original and
  calibrated K=48 peaks for each labeled training file.
- `run_metadata.json`: input data, compressor/peak settings, NMF iteration counts
  and convergence flags for the latest comparison.

The current test set has count labels but no frame-level event annotations, so
the held-out report compares count accuracy. Event precision/recall/F1 will be
added automatically for any test WAV with a matching annotation file.
The Mel and linear-frequency front ends have different units and dimensions.
Compare relative reconstruction error within each pipeline and treat raw
feature RMSE and `H_event_sum` magnitude as descriptive rather than directly
equivalent. Compare normalized event-envelope shape/statistics and count
accuracy alongside them. Timing values are host-side Python measurements,
not ESP32 execution times; use the embedded build for board timing/RAM checks.
If any source dictionary reaches `NMF_MAX_ITER`, it is recorded as not
converged and the gate remains HOLD even if count accuracy passes.
The comparison runner removes the previous contents of `results/` before it
starts writing a new comparison. The single-model runner replaces only
`results/single_inference/`, leaving the latest comparison tables and plots
intact.

## Run only the new model and peak detector

After a comparison run has tuned peak parameters,
run the script with no arguments to process all `Bowl_*.wav` files in
`data/test` independently:

```powershell
python experiments/pipeline_k24_mel/run_single.py
```

The command processes every matching Bowl file in sorted order. Each WAV gets
its own `summary.json`, `peak_results.csv`, `event_envelope.csv`,
`activations.npz`, and `peak_detection.png` under
`results/single_inference/<wav-name>/`. `single_inference/batch_summary.csv`
collects the predicted count, label count, and absolute error for all files.
Each batch run replaces the previous `single_inference/` outputs. To process
just one WAV, pass its path instead, for example:

```powershell
python experiments/pipeline_k24_mel/run_single.py data/test/Bowl_2.wav
```
