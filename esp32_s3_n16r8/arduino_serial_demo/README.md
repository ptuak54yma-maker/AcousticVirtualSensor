# WAV streaming test with Arduino IDE

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
python -m pip install -r arduino_serial_demo\requirements.txt
```

The NMF dictionary header must be generated once if it is not already present:

```powershell
python tools\export_dictionary.py --input ..\models\nmf\W_standard.npy --output arduino_serial_demo\AVS_ESP32S3_Demo\w_standard_data.h
```

Open `arduino_serial_demo\AVS_ESP32S3_Demo\AVS_ESP32S3_Demo.ino` in Arduino
IDE, choose **ESP32S3 Dev Module**, then compile and upload. Use the board's COM
port. The sketch communicates at 2,000,000 baud over a USB-to-UART bridge; with
native USB CDC the baud setting is ignored by the USB transport.

## Send `Train_0.wav`

Close Arduino IDE's Serial Monitor first because it holds the COM port. From a
PowerShell at `esp32_s3_n16r8`, run:

```powershell
python arduino_serial_demo\send_wav_to_esp32.py --port COM4
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
python .\esp32_s3_n16r8\arduino_serial_demo\hardware_benchmark.py --port COM4
```

The laptop extracts this WAV range, streams its real float32 samples to the MCU,
and the MCU measures preprocessing, FFT/spectrum, NNLS, H_event sum,
rolling-envelope update, and peak detection on those samples. Every stage
reports min/median/average/P95/max time and CPU cycles. Memory lines report
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
