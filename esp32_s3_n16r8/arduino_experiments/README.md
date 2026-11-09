# WAV streaming test with Arduino IDE

This folder contains experimental streaming and benchmark variants for
comparison; none is the finalized model or firmware selection.

## Test the optimized Arduino experiment

Open
`AVS_ESP32S3_Optimized_Experiment/AVS_ESP32S3_Optimized_Experiment.ino`
in Arduino IDE to upload the separate experimental sketch. It keeps the same
256 FISTA iterations as `AVS_ESP32S3_FISTA_Compare`, traverses the row-major
dictionary in one pass for `W^T v`, and unrolls the 48-element Gram dot
products. `BENCH_CONFIG` identifies this kernel as
`ROW_SWEEP_DOT48_UNROLLED`. Compare it with the existing FISTA sketch using
the same board settings and WAV input. The optimized sketch retains its own
copy of the dictionary header and helper sources, so the existing sketches are
left intact.

The sketch at `AVS_ESP32S3_Demo/AVS_ESP32S3_Demo.ino` accepts a WAV stream from
the laptop over USB serial. It does not need an I2S microphone or other
peripheral. `Train_0.wav` was inspected: mono, 44,100 Hz, float WAV, 880,702
samples (about 19.97 s).

## How the test represents live capture

The laptop reads the WAV and sends consecutive blocks of up to 512 float32
samples. Each block is acknowledged before the next one is sent, preventing
UART buffer overflow. The block boundaries do not define audio frames. On the
ESP32, samples are appended continuously; it processes each 2,048-sample frame
and advances 1,024 samples for the next frame, preserving the 50% overlap used
by the streaming pipeline. The sender paces blocks against the WAV's 44.1 kHz
sample timeline by default and reports the achieved audio-rate factor.

The sender requires NumPy, SoundFile, PySerial and Matplotlib. Install them from a PowerShell
opened at `esp32_s3_n16r8`:

```powershell
python -m pip install -r arduino_experiments\requirements.txt
```

The NMF dictionary header must be generated once if it is not already present:

```powershell
python tools\export_dictionary.py --input ..\models\nmf\W_standard.npy --output arduino_experiments\AVS_ESP32S3_Demo\w_standard_data.h
```

Open `arduino_experiments\AVS_ESP32S3_Demo\AVS_ESP32S3_Demo.ino` in Arduino
IDE, choose **ESP32S3 Dev Module**, then compile and upload. Use the board's COM
port. The sketch communicates at 2,000,000 baud over a USB-to-UART bridge; with
native USB CDC the baud setting is ignored by the USB transport.

## Send `Train_0.wav`

Close Arduino IDE's Serial Monitor first because it holds the COM port. From a
PowerShell at `esp32_s3_n16r8`, run:

```powershell
python arduino_experiments\send_wav_to_esp32.py --port COM4
```

Replace `COM4` with the board's port. The script defaults to the project's
`data\train\Train_0.wav`. To send another file, add `--wav path\to\file.wav`.
The WAV must be 44,100 Hz; stereo is downmixed to mono. The script does not
globally remove DC because the firmware removes the mean of each frame, matching
the existing streaming implementation.

The script waits for the board's `AVS_READY` line, sends `START`, acknowledged
`DATA` blocks and `END`, then prints a productivity report about once per second.
`count_3s` is the number of detected peaks in the current rolling 3-second
window, and `feed_rate_parts_s` is `count_3s / 3`. After the first full window,
the firmware reruns peak detection every second. The chart window opens at the
start of the transfer and displays only the latest 3-second window once the
first report arrives. It advances once per report, drops the oldest second and
marks the peaks detected in that window. The chart is shown only as a live
window; the script does not save an image file.

If transfer/processing is slower than the WAV duration, the reported factor is
below `1.00x`; acknowledged blocks prevent sample loss, but this indicates the
board cannot keep up with real-time playback at the current settings. Use
`--fast` to disable pacing for an offline throughput run.

The serial sender owns the port during the transfer, so its terminal displays
the results. Do not open Serial Monitor at the same time. This checks the
firmware with the actual training WAV; it does not itself validate the model's
event labels or replace the Python reference comparison.

## Benchmark the ESP32-S3 hardware

After uploading the sketch, close Serial Monitor and run this from a PowerShell
at the main `AcousticVirtualSensor` folder. By default it profiles the actual
`Train_0.wav` segment from 1.00 to 4.05 seconds:

```powershell
python .\esp32_s3_n16r8\arduino_experiments\hardware_benchmark.py --port COM4
```

The firmware must be uploaded to the ESP32-S3 before running this command.
Open `esp32_s3_n16r8\arduino_experiments\AVS_ESP32S3_Demo\AVS_ESP32S3_Demo.ino`
in Arduino IDE, select **ESP32S3 Dev Module** and the board's COM port, then
upload. Close Serial Monitor before starting the Python script. Install the
host-side packages once from the project root with:

```powershell
python -m pip install -r .\esp32_s3_n16r8\arduino_experiments\requirements.txt
```

If the sketch cannot find `w_standard_data.h`, generate it once from the project
root:

```powershell
python .\esp32_s3_n16r8\tools\export_dictionary.py --input .\models\nmf\W_standard.npy --output .\esp32_s3_n16r8\arduino_experiments\AVS_ESP32S3_Demo\w_standard_data.h
```

Replace `COM4` with the port shown for the board in Windows Device Manager.
The Python tool prints the raw ESP32 records and then a stage summary table.
Use `BENCH_STAGE`/the table to compare individual stage averages and P95 values;
use `BENCH_TOTAL_FRAME` for per-frame compute time and `BENCH_REALTIME` for the
capacity verdict. `wall_capacity_x` also includes host-to-board serial transfer,
so it is not the MCU-only processing capacity.

The laptop extracts this WAV range, streams its real float32 samples to the MCU,
and the MCU measures the frame-input copy, preprocessing, windowing, FFT,
spectrum conversion, NNLS, H_event sum, rolling-envelope update, and peak
detection on those samples.
Every stage reports min/median/average/P95/max time and CPU cycles. The frame
input stage measures copying the assembled frame into a working buffer; UART
transfer/wait time is excluded. The current firmware has no envelope smoothing,
so `smoothing=NOT_IMPLEMENTED` is reported instead of a misleading timing.
NNLS now uses fixed-iteration FISTA with float32 and the initialization-time
Gram matrix. `BENCH_CONFIG` reports its iteration count, step size, and
Lipschitz bound; `BENCH_NNLS_DIAGNOSTICS` reports the relative projected
gradient after the configured iterations. `BENCH_EVENT_SCORE` reports the
resulting event-score range. Compare the `nnls` stage time with the earlier
active-set benchmark on the same WAV segment, and use the projected-gradient
and event-score diagnostics when judging the speed/accuracy trade-off.
`BENCH_CHIP` and `BENCH_CONFIG` report the measured core, CPU frequency,
compiler optimization status, and whether serial logging, transport, or dynamic
allocation is inside timed regions. Serial output happens after measurements.
Memory lines report
internal RAM, PSRAM, executable IRAM, and stack watermarks before/after each
stage, including stage-local minimum free heap and peak extra allocation. Flash
code size and remaining app partition are reported once globally because flash
is static while a stage runs. Arduino's compile output reports global/static
RAM. The benchmark also reports transfer wall time and real-time capacity.

The default clip is 3.05 seconds so it fills the 3-second peak window. To use
exactly 1-4 seconds, add `--duration-sec 3`, but that may end one frame before a
full peak window is available. The synthetic-frame check remains available with
`--synthetic --frames 100`. This measures processing on actual WAV samples but
does not capture through an I2S microphone or include microphone/ADC latency.

## Batch comparison on ESP32-S3

`AVS_ESP32S3_Experiment_Viewer` runs the entire profile sweep in C++ for each
frame while the host streams each WAV once. Profiles are legacy 1025-bin NNLS,
Mel64 NNLS, Mel128 NNLS, Mel64 FISTA at 8 through 4096 iterations, and Mel128
FISTA512. The C++ sketch emits Mel64/Mel128 features and all 48 activations for
each profile, so the host can compare features, H, event envelopes, peak
counts, and timing. Frame state is streaming; WAVs are not limited by a
fixed-size envelope array.

Generate the model header once; the generated arrays remain local and are not
checked in:

```powershell
python .\esp32_s3_n16r8\tools\export_arduino_experiment_models.py
```

Open `AVS_ESP32S3_Experiment_Viewer\AVS_ESP32S3_Experiment_Viewer.ino` in
Arduino IDE, select **ESP32S3 Dev Module** and the COM port, set **PSRAM = OPI
PSRAM** and **Partition Scheme = Huge APP**, then upload once. PSRAM is used
for per-file score buffers. Close Serial Monitor. Then install host packages
and run one batch command:

```powershell
python -m pip install -r .\esp32_s3_n16r8\arduino_experiments\requirements.txt
python .\esp32_s3_n16r8\arduino_experiments\AVS_ESP32S3_Experiment_Viewer\run_dataset_experiments.py --port COM4 --split all
```

The runner walks every `data/train/*.wav` and `data/test/*.wav` automatically;
no per-file commands or re-flashing are needed. `--split train` or `--split
test` selects a subset, and `--max-files 1` is available for a short protocol
check; `--max-seconds 2` truncates files to two seconds for a quick protocol
check (truncated runs do not produce label metrics). Output goes to
`outputs\arduino_dataset_experiments`:

- `cpp_activations.csv`: each frame's 48 H values and event envelope for all
  C++ profiles.
- `cpp_features.csv` and per-file `*_cpp_features.npz`: C++ Mel64/Mel128
  features.
- `feature_parity.csv` and `activation_parity.csv`: C++ versus matching Python
  features/H metrics, including relative error and event-envelope correlation.
- `python_cpp_count_comparison.csv` and `python_cpp_count_summary.csv`: C++ and
  Python predictions compared using the same train-selected peak parameters.
- `predictions_per_file.csv`, `count_metrics.csv`: per-file predictions and
  train-tuned peak settings, with summary Count Accuracy, MAE, and count-based
  F1 for train and test.
- `timing_summary.csv`: mean/P50/P95/P99/max compute time, percentage below the
  23.22 ms target, plus telemetry wall time for the full sweep.
- `memory_samples.csv`, `files_processed.csv`, and `dataset_comparison.png`.

Only product counts are present in the supplied label files, so Event F1 is
reported as a count-based proxy; timestamp-level event matching needs
frame/peak annotations. The legacy MCU NNLS profile uses fixed-sweep
coordinate descent, while Python's exact reference uses SciPy NNLS. Timing
separates computation from serial telemetry; the latter is excluded from the
per-profile compute latency.
