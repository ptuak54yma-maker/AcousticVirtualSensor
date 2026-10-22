# Embedded Stage 1 — host-side C/C++ logic validation

This directory starts Stage 1 of the Acoustic Virtual Sensor embedded port. It is a **host-native test harness**, not STM32 timing evidence and not yet a complete STM32Cube firmware project.

## Reference pipeline preserved from the repository

- 44,100 Hz; frame/window 2,048 samples; hop 1,024.
- Periodic Hann window (`scipy.signal.get_window("hann", N, fftbins=True)`).
- Real FFT, 1,025 one-sided bins; no FFT normalization.
- `log1p(abs(X))` log-magnitude.
- Frozen `models/nmf/W_standard.npy`, shape 1,025 x 48, FP32; first 24 components are event components.
- Python reference solver: `scipy.optimize.nnls`.
- Peak reference: `scipy.signal.find_peaks`, with prominence and distance read from `models/peak_detection/peak_params.json` (currently 0.01 and 4 frames).

No NMF retraining is done by this tooling.

## 1. Windows prerequisites

Install:
- VS Code
- CMake 3.20+
- Visual Studio Build Tools 2022 with **Desktop development with C++**, or a working MinGW-w64 C++17 toolchain
- Python environment with the repository's existing `requirements.txt`

From a VS Code PowerShell terminal at the repository root, check:
```powershell
cmake --version
python --version
```

## 2. Compile and run smoke tests

This does not require the dictionary header to have been exported:
```powershell
cmake -S embedded -B embedded/build
cmake --build embedded/build --config Release
ctest --test-dir embedded/build -C Release --output-on-failure
```

The smoke test covers the periodic Hann window, radix-2 FFT/log magnitude, event activation summation, peak picking on a controlled signal, and NNLS mechanics on a synthetic orthogonal dictionary.

## 3. Compare the C++ pipeline with the Python reference on a WAV

Use a WAV file at exactly 44.1 kHz. The helper generates the C header from the **existing** `W_standard.npy`, builds `avs_infer`, then compares all 48 activation values and event envelope for every frame against the Python pipeline.

```powershell
python embedded/tools/verify_against_python.py data/test/your_file.wav
```

Optional strict peak comparison:
```powershell
python embedded/tools/verify_against_python.py data/test/your_file.wav --require-exact-peaks
```

The WAV path above is an example; the repository may not contain raw WAVs because dataset audio is git-ignored. Supply a local test WAV, at 44.1 kHz. The default maximum absolute activation error threshold is 0.05 and is a **starting diagnostic threshold**, not a declared result or universal accuracy criterion. Review printed maximum error/RMSE and peak-index differences before deciding acceptance criteria for the thesis.

The helper uses the streaming reference's per-frame DC subtraction. WAV input is loaded as float32, must already be at 44.1 kHz, and is not resampled by the C++ side.

## 4. Algorithm and numerical limitations

- The MCU-facing data layout for W is row-major `W[f * 48 + k]`; the Python exporter validates shape, finite values, and non-negativity before generating `w_standard_data.h`.
- The C++ NNLS is an active-set-style solver on the precomputed Gram matrix `W^T W). This keeps the iterative workspace bounded (48 x 48 FP32 = 9 KiB), but normal equations can amplify conditioning errors. It is **not assumed equivalent** to SciPy until comparison passes on representative real project WAVs.
- The C++ peak picker handles ordinary finite local maxima, prominence, and minimum distance. SciPy has precise plateau and tie-breaking semantics; edge cases need validation against Python, so the Python detector remains the reference.
- FFT uses a compact radix-2 host implementation for stage-1 testability. Replace it with CMSIS-DSP's Cortex-M4F real FFT in the MCU build, then repeat numerical tests.
- Host runtime, host memory usage, and CI results are not STM32F407 timing or RAM evidence.

## 5. Frozen dictionary export

Manual export if needed:
```powershell
python embedded/tools/export_dictionary.py --input models/nmf/W_standard.npy --output embedded/src/w_standard_data.h
```

The generated header is ignored by Git to avoid storing a second copy of the large dictionary. Do not modify or retrain the original `.npy` model for this stage.
