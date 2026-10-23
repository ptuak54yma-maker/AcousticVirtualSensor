#!/usr/bin/env python3
"""Report MCU hardware requirements for the currently ported AVS pipeline.

The report separates measurements taken from the configured pipeline/current
firmware build from engineering recommendations. CPU frequency recommendations
are planning targets only until frame execution time is measured on a physical MCU.

Run from the repository root:
    python embedded/tools/report_mcu_requirements.py
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_HEADER = ROOT / "embedded" / "include" / "avs_config.h"
DEFAULT_ELF = ROOT / "embedded" / "firmware" / "out" / "avs_stm32f407.elf"
DEFAULT_OUTPUT = ROOT / "embedded" / "reports" / "mcu_hardware_requirements.md"


def read_config() -> dict[str, int]:
    text = CONFIG_HEADER.read_text(encoding="utf-8")
    values = {
        key: int(value)
        for key, value in re.findall(
            r"^\s*#define\s+AVS_(\w+)\s+(\d+)u?\s*$", text, flags=re.MULTILINE
        )
    }
    required = ("SAMPLE_RATE", "N_FFT", "HOP_LENGTH", "NUM_BINS", "NUM_COMPONENTS")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing AVS constants in {CONFIG_HEADER}: {', '.join(missing)}")
    return values


def find_size_tool() -> str | None:
    tool = shutil.which("arm-none-eabi-size") or shutil.which("arm-none-eabi-size.exe")
    if tool:
        return tool
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates = sorted(
            (Path(local_app_data) / "ArmToolchains").glob("*/bin/arm-none-eabi-size.exe"),
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    return None


def firmware_size(elf: Path) -> tuple[int, int, int] | None:
    tool = find_size_tool()
    if not tool or not elf.is_file():
        return None
    cwd = ROOT
    elf_arg = str(elf)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        ascii_root = Path(local_app_data) / "AVSRepoLink"
        try:
            if ascii_root.exists() and ascii_root.resolve() == ROOT.resolve():
                elf_arg = elf.resolve().relative_to(ROOT).as_posix()
                cwd = ascii_root
        except (OSError, ValueError):
            pass
    try:
        result = subprocess.run(
            [tool, elf_arg], cwd=cwd, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) < 2 or len(rows[-1]) < 4:
        return None
    return tuple(int(value, 10) for value in rows[-1][:3])  # text, data, bss


def kib(value: int) -> str:
    return f"{value / 1024:.1f} KiB"


def build_report(values: dict[str, int], elf: Path) -> str:
    sample_rate = values["SAMPLE_RATE"]
    n_fft = values["N_FFT"]
    hop = values["HOP_LENGTH"]
    bins = values["NUM_BINS"]
    components = values["NUM_COMPONENTS"]
    frames_per_second = sample_rate / hop
    frame_deadline_ms = 1000.0 / frames_per_second

    # Approximate current firmware work buffers for a 32-bit Cortex-M ABI.
    # The ELF/map measurement below is preferred when the cross-toolchain exists.
    nnls_workspace = (
        2 * components * components * 8  # double Gram and Cholesky matrices
        + 5 * components * 8  # double vectors
        + components * 4  # 32-bit size_t active indices
        + components  # passive-set flags
        + 2 * 8  # double scalar values
        + 4  # initialized flag plus alignment allowance
    )
    estimated_work_buffers = (
        2 * n_fft * 4  # FFT real/imag scratch
        + n_fft * 4  # frame
        + n_fft * 4  # window
        + bins * 4  # log spectrum
        + nnls_workspace
        + components * 4  # H output
    )
    dictionary_flash = bins * components * 4
    measured = firmware_size(elf)

    if measured:
        text_bytes, data_bytes, bss_bytes = measured
        flash_bytes = text_bytes + data_bytes
        ram_static = data_bytes + bss_bytes
        build_note = (
            f"ELF size tool: text={kib(text_bytes)}, data={kib(data_bytes)}, "
            f"bss={kib(bss_bytes)}. Flash load estimate (text+data): {kib(flash_bytes)}; "
            f"RAM linker allocation (data+bss): {kib(ram_static)}. The current bss figure "
            "includes the linker's 8 KiB stack reservation."
        )
    else:
        build_note = (
            "No arm-none-eabi-size result was available; use the work-buffer estimate "
            "below and rerun after building with the ARM toolchain for exact ELF section sizes."
        )

    return f"""# MCU hardware requirements — Acoustic Virtual Sensor

Generated from `{CONFIG_HEADER.relative_to(ROOT).as_posix()}`.

## Scope and measured pipeline facts

This sizing covers the currently ported embedded chain: **frame input → FFT/log spectrum → fixed-dictionary NNLS → H_event_sum → peak detection**. The embedded firmware does not yet include a WAV reader, microphone/ADC/I2S capture path, or the project's CRNN inference. Add those resources separately before treating this as a complete product specification.

| Item | Value |
|---|---:|
| Audio sample rate | {sample_rate:,} samples/s, mono |
| FFT frame / hop | {n_fft:,} / {hop:,} samples |
| Frames per second | {frames_per_second:.2f} |
| Time available per frame for real-time operation | {frame_deadline_ms:.2f} ms |
| Spectrum bins / dictionary components | {bins:,} / {components} |
| Frozen float32 dictionary storage | {kib(dictionary_flash)} in Flash (approximately) |
| Current FFT + frame + window + spectrum + NNLS + H buffers | {kib(estimated_work_buffers)} estimated |

The frame deadline is calculated from the configured hop. A real-time implementation must finish each frame, including input and scheduling overhead, before the next frame is due. Execution time has **not** been measured on a physical MCU.

## Current firmware build footprint

{build_note}

ELF examined: `{elf}`. A generated ELF size is a build measurement, not proof that streaming audio fits in the same RAM budget; DMA/audio buffers and the final application stack must be included.

## Suggested hardware tiers

These are initial selection targets for the current Stage 1 chain, not certified minimums. All tiers assume a Cortex-M with a hardware **single-precision FPU**, DSP instructions, at least 32-bit DMA transfers where required, and a clock/timer capable of maintaining the audio sample rate.

| Tier | CPU target | Flash | DMA-accessible SRAM | Audio interface | Intended use |
|---|---|---:|---:|---|---|
| **Vừa đủ** | Cortex-M4F, roughly 120–168 MHz | ≥512 KiB | ≥128 KiB, with at least 40 KiB remaining after firmware, stack, and buffers | I2S/SAI or ADC + DMA; UART for diagnostics | Bring-up and frame-by-frame inference. Tight memory/performance margin; real-time deadline must be measured. |
| **Chạy ổn** | Cortex-M4F 168 MHz or Cortex-M7 ≥200 MHz | ≥1 MiB | ≥256 KiB | I2S/SAI + DMA; timer; UART/USB for diagnostics | Recommended baseline for continuous mono capture, buffering, and useful headroom for logging and application code. |
| **Chạy tốt** | Cortex-M7 roughly 400 MHz or faster | ≥2 MiB | ≥512 KiB | I2S/SAI + DMA; optional SD/QSPI if recording WAV locally | More CPU/RAM margin for additional buffering, richer preprocessing, and future model components. CRNN still needs separate sizing and porting. |

For a minimum tier, target measured frame processing below **70%** of the {frame_deadline_ms:.2f} ms frame interval; for a comfortable tier, below **50%**. These are engineering headroom targets, not current benchmark results.

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
"""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, default=DEFAULT_ELF,
                        help="Optional built ARM ELF used to measure text/data/bss sections")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Markdown report path")
    args = parser.parse_args()

    output = args.output.expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(read_config(), args.elf.expanduser().resolve())
    output.write_text(report, encoding="utf-8")
    print(report)
    printable_output = output.as_posix().encode("ascii", "backslashreplace").decode("ascii")
    print(f"\nReport saved: {printable_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
