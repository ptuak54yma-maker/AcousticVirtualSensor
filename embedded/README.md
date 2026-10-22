# Embedded Stage 1 — host-side C/C++ logic validation

This directory starts Stage 1 of the Acoustic Virtual Sensor embedded port. It is a **host-native test harness**, not STM32 timing evidence and not yet MCU firmware.

## Current reference pipeline (kept from repository)

- 44,100 Hz; frame/window 2,048 samples; hop 1,024; periodic Hann window.
- Real FFT, 1,025 one-sided bins.
- `log1p(abs(X))` log-magnitude.
- Fixed dictionary `W_standard.npy`, expected shape 1,025 x 48, FP32; first 24 components are event components.
- Python reference NNLS is `scipy.optimize.nnls`; its solver is an active-set method.
- Peak reference is `scipy.signal.find_peaks` with parameters from `models/peak_detection/peak_params.json` (currently prominence 0.01, distance 4).

## Stage 1 design

The first acceptance target is **numerical comparison on the same vectors**. Audio loading / resampling is deliberately outside the C core: supply mono float WAV at 44.1 kHz with the same DC-removal policy as the Python reference.

The initial native core is organized as:
- DSP front end: Hann -> real FFT -> log magnitude.
- Fixed-dictionary non-negative activation solver interface.
- Event activation sum.
- Offline/batch peak detector used to compare against the SciPy result.

A small embedded peak state machine is **not assumed equivalent** to SciPy's batch `find_peaks`; boundary and prominence semantics must be tested separately before using it in streaming.

## Build on Windows (VS Code terminal)

Recommended: install CMake and a C++17 compiler (Visual Studio Build Tools or MinGW-w64). Then:

```powershell
cmake -S embedded -B embedded/build -DCMAKE_BUILD_TYPE=Release
cmake --build embedded/build --config Release
ctest --test-dir embedded/build -C Release --output-on-failure
```

The host target makes it easy to test logic but its latency is **not** an STM32F407 latency estimate.

## Dictionary export

The MCU build cannot load NumPy `.npy` directly. Export the already-trained `W_standard.npy` as a C header with:

```powershell
python embedded/tools/export_dictionary.py --input models/nmf/W_standard.npy --output embedded/src/w_standard_data.h
```

Do not retrain NMF. Keep the original `.npy` as the reference source.

## Known gap before claiming equivalence

SciPy NNLS and any custom embedded solver must be compared frame-by-frame for:
- activation absolute/relative error,
- reconstruction residual,
- non-negativity,
- event-envelope error,
- final peak indices / count.

No equivalence claim should be made until those tests pass against real vectors from the existing Python pipeline.
