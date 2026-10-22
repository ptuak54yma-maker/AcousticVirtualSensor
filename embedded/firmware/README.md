# STM32F407VG bare-metal simulation target

This standalone target keeps the missing CubeIDE `test1` project untouched. It
assumes the STM32F4 Discovery-compatible STM32F407VG part requested for this
simulation: Cortex-M4F, 1 MiB Flash, 128 KiB main SRAM, and 64 KiB CCM. The
firmware linker uses only the 1 MiB Flash and 128 KiB contiguous SRAM regions;
CCM is declared but unused. No ADC, microphone, DMA, I2S, or clock-tree setup is
assumed.

## Build

The Windows build script fetches pinned upstream CMSIS-DSP and CMSIS Core
revisions into the ignored `_deps/` directory, then builds the firmware ELF with
Arm GNU Toolchain 12.2, Ninja, and CMake:

```powershell
powershell -ExecutionPolicy Bypass -File embedded/firmware/build_stm32f407.ps1
```

Pinned upstream sources:

- CMSIS-DSP `v1.16.2`, commit `d5717e454fec0337bef114a21f1d2d01d74f2701`
- CMSIS Core `v6.2.0`, commit `6f0a58d01aa9bd2feba212097f9afe7acd991d52`

The real FFT uses `arm_rfft_fast_init_2048_f32` and `arm_rfft_fast_f32`; CMSIS's
packed DC/Nyquist representation is unpacked into the existing 1,025-bin
log-magnitude interface. Compiler flags select Cortex-M4, Thumb, FPv4-SP-D16,
and the hard-float ABI. The ELF and linker map report Flash/RAM usage.

The NNLS workspace is caller-owned, persistent, double precision, and 39,240
bytes (38.3 KiB). The firmware stores it in BSS alongside the FFT workspace, frame,
window, and spectrum buffers; it is not placed on the stack. The stack
reservation is 8 KiB. One mutable NNLS workspace is required for every
concurrent solve. Peak detection uses the caller's bounded output buffer as
scratch, reports `AVS_PEAKS_OVERFLOW` when candidate capacity is insufficient,
and does not allocate heap memory.

Firmware test state is exported as `avs_fw_test_status`,
`avs_fw_failed_stage`, `avs_fw_frames_processed`, `avs_fw_peak_count`, and
floating-point result symbols in the ELF. UART4 emits stage messages at 115200
baud when the simulator model supports the configured UART pins.

## Renode smoke test

After building, run `powershell -ExecutionPolicy Bypass -File
embedded/firmware/sim/run_renode.ps1`. It runs the one-frame FFT/NNLS/event
check and deterministic peak check on Renode's STM32F4 Discovery model, then
checks for `AVS_TEST_PASS` in `sim/uart.log`. The model provides a compatible
Cortex-M4 and UART4 peripheral; it is not a substitute for testing on the exact
physical F407VG board.

For the full repeatable workflow, input definitions, captured UART values, and
host WAV comparison results, see [REPRODUCING.md](REPRODUCING.md).
