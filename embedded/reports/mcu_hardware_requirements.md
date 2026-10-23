# MCU hardware requirements — Acoustic Virtual Sensor

Generated from `embedded/include/avs_config.h`.

## Scope and measured pipeline facts

This sizing covers the currently ported embedded chain: **frame input → FFT/log spectrum → fixed-dictionary NNLS → H_event_sum → peak detection**. The embedded firmware does not yet include a WAV reader, microphone/ADC/I2S capture path, or the project's CRNN inference. Add those resources separately before treating this as a complete product specification.

| Item | Value |
|---|---:|
| Audio sample rate | 44,100 samples/s, mono |
| FFT frame / hop | 2,048 / 1,024 samples |
| Frames per second | 43.07 |
| Time available per frame for real-time operation | 23.22 ms |
| Spectrum bins / dictionary components | 1,025 / 48 |
| Frozen float32 dictionary storage | 192.2 KiB in Flash (approximately) |
| Current FFT + frame + window + spectrum + NNLS + H buffers | 74.3 KiB estimated |

The frame deadline is calculated from the configured hop. A real-time implementation must finish each frame, including input and scheduling overhead, before the next frame is due. Execution time has **not** been measured on a physical MCU.

## Current firmware build footprint

ELF size tool: text=228.8 KiB, data=0.1 KiB, bss=82.7 KiB. Flash load estimate (text+data): 228.9 KiB; RAM linker allocation (data+bss): 82.8 KiB. The current bss figure includes the linker's 8 KiB stack reservation.

ELF examined: `E:\HUST\Master\Luận văn\AcousticVirtualSensor\embedded\firmware\out\avs_stm32f407.elf`. A generated ELF size is a build measurement, not proof that streaming audio fits in the same RAM budget; DMA/audio buffers and the final application stack must be included.

## Suggested hardware tiers

These are initial selection targets for the current Stage 1 chain, not certified minimums. All tiers assume a Cortex-M with a hardware **single-precision FPU**, DSP instructions, at least 32-bit DMA transfers where required, and a clock/timer capable of maintaining the audio sample rate.

| Tier | CPU target | Flash | DMA-accessible SRAM | Audio interface | Intended use |
|---|---|---:|---:|---|---|
| **Vừa đủ** | Cortex-M4F, roughly 120–168 MHz | ≥512 KiB | ≥128 KiB, with at least 40 KiB remaining after firmware, stack, and buffers | I2S/SAI or ADC + DMA; UART for diagnostics | Bring-up and frame-by-frame inference. Tight memory/performance margin; real-time deadline must be measured. |
| **Chạy ổn** | Cortex-M4F 168 MHz or Cortex-M7 ≥200 MHz | ≥1 MiB | ≥256 KiB | I2S/SAI + DMA; timer; UART/USB for diagnostics | Recommended baseline for continuous mono capture, buffering, and useful headroom for logging and application code. |
| **Chạy tốt** | Cortex-M7 roughly 400 MHz or faster | ≥2 MiB | ≥512 KiB | I2S/SAI + DMA; optional SD/QSPI if recording WAV locally | More CPU/RAM margin for additional buffering, richer preprocessing, and future model components. CRNN still needs separate sizing and porting. |

For a minimum tier, target measured frame processing below **70%** of the 23.22 ms frame interval; for a comfortable tier, below **50%**. These are engineering headroom targets, not current benchmark results.

## Important STM32F407 constraint

The existing target is STM32F407VG: Cortex-M4F at up to 168 MHz, 1 MiB Flash, 128 KiB main SRAM, and 64 KiB CCM. The current linker places the working buffers in main SRAM; CCM is not DMA-accessible on this MCU. The current ELF build fits the linker budget, but the remaining main-SRAM space must also hold the eventual audio DMA buffers, stack growth, and any communication buffers. Do not count CCM as microphone/I2S DMA memory.

Device limits are from the [ST STM32F405/407 datasheet](https://www.st.com/resource/en/datasheet/stm32f407vg.pdf); the 128 KiB main-SRAM / 64 KiB CCM split is also explicit in this project's linker script.

For 44.1 kHz mono PCM16, the incoming stream is about **88.2 kB/s**; float32 is about **176.4 kB/s**. A live sensor normally needs an audio interface plus DMA and a ring/double buffer, not a WAV filesystem. If the device must read WAV files from storage, add SD/FatFS (or equivalent) and budget storage capacity separately.

## Re-run

```powershell
python embedded/tools/report_mcu_requirements.py
```

Optional paths:

```powershell
python embedded/tools/report_mcu_requirements.py --elf embedded/firmware/out/avs_stm32f407.elf --output embedded/reports/mcu_hardware_requirements.md
```
