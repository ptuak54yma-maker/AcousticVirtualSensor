# Acoustic Virtual Sensor — ESP32-S3-N16R8

Standalone ESP-IDF firmware folder for the ESP32-S3. By default it runs a
synthetic-audio demo and prints results to the serial console, so it can be
flashed and tried without a microphone or other external peripheral. It reuses
the frozen NMF dictionary; it does not retrain or alter that model.

## What runs on the board

Synthetic PCM by default (or I2S microphone when demo mode is disabled) →
2,048-sample frame / 1,024-sample hop → frame-mean removal → periodic Hann →
2,048-point FFT → `log1p(|X|)` → fixed-dictionary NNLS (1,025 × 48) → sum of
the first 24 event activations → 3-second event peak count → serial log with
count and parts/second. Demo mode processes five seconds of synthetic audio
then prints `DEMO_DONE` and idles.

The initial port follows the project's embedded baseline. It does not execute
the PyTorch Hybrid CRNN yet. The `.pt` model cannot be loaded by ESP-IDF as-is;
that stage needs an explicit export/conversion and frame/event parity validation
before it can replace peak counting.

## Arduino IDE option

A separate Arduino sketch is provided in
[`arduino_serial_demo/AVS_ESP32S3_Demo`](arduino_serial_demo/AVS_ESP32S3_Demo).
It receives WAV samples from a laptop over USB serial and processes them as a
continuous stream, so it needs no microphone. See that folder's README for the
sender script, board settings and upload steps. The ESP-IDF instructions below
are for the standalone I2S firmware project.

## Serial-only demo (no external hardware)

The default `CONFIG_AVS_DEMO_MODE=y` generates a repeatable tone with short
bursts in software. It exercises the firmware DSP path and prints the resulting
baseline count/feed-rate fields to the ESP-IDF serial monitor. This checks that
the firmware boots and runs; it does not validate microphone capture or
real-world detection accuracy.

## Hardware and assumptions

- ESP32-S3 module marked N16R8 (16 MB flash and 8 MB PSRAM). The pipeline's
  static working buffers use internal RAM; PSRAM is available for future model
  or logging buffers, not required for this first baseline.
- Standard-mode I2S MEMS microphone, 32-bit slot carrying signed 24-bit PCM in
  bits 31..8, left slot selected. Change `accept_pcm_word()` for another format.
- GPIO defaults are BCLK=4, WS=5, DIN=6. Set the actual board wiring with
  `idf.py menuconfig` → Acoustic Virtual Sensor board.
- The microphone must support 44.1 kHz. Confirm its clocking and GPIO wiring on
  the actual development board.

## Prepare and build

Install ESP-IDF 5.3+ using Espressif's [Windows setup](https://docs.espressif.com/projects/esp-idf/en/v5.3.5/esp32s3/get-started/windows-setup.html),
then open the **ESP-IDF PowerShell** shortcut so `idf.py` and its toolchain are
available. Connect the board directly to the laptop by USB. In Device Manager,
find the COM port that appears for the board. From this folder, export the
frozen dictionary once (Python with `numpy` required):

```powershell
python tools/export_dictionary.py
idf.py set-target esp32s3
idf.py build
idf.py -p COM5 flash monitor
```

Replace `COM5` with the board's actual port. The first build can take several
minutes. `flash monitor` both writes the firmware and opens its serial output;
exit the monitor with `Ctrl+]`. The ESP-IDF Windows installer adds a Start Menu
shortcut that opens a shell with the required environment set up, as described
in [Espressif's Windows flashing guide](https://docs.espressif.com/projects/esp-idf/en/v5.3.5/esp32s3/get-started/windows-setup.html).

If running the exporter from another working directory, pass explicit paths,
for example `--input ..\models\nmf\W_standard.npy --output main\w_standard_data.h`.
The generated header is ignored by Git because it is reproducible from the
existing `W_standard.npy`.

For microphone operation later, run `idf.py menuconfig` and disable
**Acoustic Virtual Sensor board → Run synthetic serial-only demo**, set the
three I2S pins, then build and flash again.

## Bring-up and parity work

1. Set the three I2S pins and confirm the microphone's slot/bit alignment.
2. Check that `H_event` changes when parts pass the bowl; adjust only the PCM
   conversion if the input representation differs.
3. Compare spectra, all 48 NNLS activations, and event envelopes with
   `embedded/tools/verify_against_python.py` using representative 44.1 kHz WAVs.
4. Compare event thresholds and timing with the Python baseline, then set the
   trained peak prominence/distance in `main/avs_config.h`.
5. Measure per-frame processing time on the physical board. The 1,024-sample
   hop gives 23.22 ms per frame; no real-time performance claim is made until
   measurement confirms the deadline with margin.

## Notes

- This directory includes the embedded C++ NNLS and peak detector in local
  copies so the ESP-IDF project is self-contained. The FFT is a portable radix-2
  implementation; optimize with ESP-DSP only after measuring and checking
  numeric parity.
- NNLS uses a persistent double-precision workspace (~39 KiB), allocated
  statically rather than on the FreeRTOS task stack.
- ESP32-S3-N16R8 identifies the module's flash/PSRAM configuration, but GPIOs,
  microphone wiring and available internal heap depend on the carrier board and
  ESP-IDF configuration.
