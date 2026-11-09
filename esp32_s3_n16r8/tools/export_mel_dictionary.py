"""Export a trained K=48 Mel model to the ESP-IDF firmware header."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=root / "experiments" / "pipeline_k48_mel" / "models" / "W_standard_mel.npz",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "esp32_s3_n16r8" / "espidf_experiments" / "w_standard_mel_data.h",
    )
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as bundle:
        w = np.asarray(bundle["W"], dtype=np.float32)
        bank = np.asarray(bundle["mel_bank"], dtype=np.float32)
        reference = float(bundle["compressor_reference"])
        n_mels = int(bundle["n_mels"])
        gamma = float(bundle["power_gamma"])
        compressor = str(bundle["compressor"].item())
        mel_input = str(bundle["mel_input"].item())
    if w.shape != (n_mels, 48) or bank.shape != (n_mels, 1025):
        raise SystemExit(f"Expected W ({n_mels},48) and Mel bank ({n_mels},1025); got {w.shape} and {bank.shape}.")
    if compressor not in {"log1p", "power", "linear"} or mel_input not in {"power", "magnitude"}:
        raise SystemExit("Unsupported Mel input or compressor.")
    if compressor == "power" and not (0.0 < gamma <= 1.0):
        raise SystemExit("Power compression gamma must be in (0, 1].")
    for name, array in (("W", w), ("Mel bank", bank)):
        if not np.isfinite(array).all() or np.any(array < 0):
            raise SystemExit(f"{name} must contain finite nonnegative values.")
    values_w = w.ravel(order="C")
    offsets = [0]
    bins: list[int] = []
    weights: list[float] = []
    for mel_row in bank:
        active = np.flatnonzero(mel_row)
        bins.extend(int(index) for index in active)
        weights.extend(float(mel_row[index]) for index in active)
        offsets.append(len(bins))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#ifndef AVS_W_STANDARD_MEL_DATA_H\n#define AVS_W_STANDARD_MEL_DATA_H\n\n#include <stdint.h>\n\n")
        f.write(f"#define AVS_W_ROWS {n_mels}u\n#define AVS_W_COLS 48u\n")
        f.write("#define AVS_MEL_REFERENCE %.9ef\n" % reference)
        f.write(f"#define AVS_MEL_INPUT_POWER {1 if mel_input == 'power' else 0}u\n")
        f.write(f"#define AVS_MEL_COMPRESSOR_LOG1P {1 if compressor == 'log1p' else 0}u\n")
        f.write(f"#define AVS_MEL_COMPRESSOR_POWER {1 if compressor == 'power' else 0}u\n")
        f.write("#define AVS_MEL_POWER_GAMMA %.9ef\n" % gamma)
        f.write("/* Generated from experiments/pipeline_k48_mel/models/W_standard_mel.npz. */\n")
        f.write("static const float AVS_W_STANDARD[AVS_W_ROWS * AVS_W_COLS] = {\n")
        for i in range(0, values_w.size, 6):
            f.write("    " + ", ".join(f"{float(x):.9e}f" for x in values_w[i:i + 6]) + ",\n")
        f.write(f"}};\n\n#define AVS_MEL_NNZ {len(bins)}u\n")
        f.write(f"static const uint16_t AVS_MEL_OFFSETS[{n_mels + 1}] = {{\n    ")
        f.write(", ".join(str(x) for x in offsets))
        f.write("\n};\nstatic const uint16_t AVS_MEL_BIN_INDEX[AVS_MEL_NNZ] = {\n")
        for i in range(0, len(bins), 16):
            f.write("    " + ", ".join(str(x) for x in bins[i:i + 16]) + ",\n")
        f.write("};\nstatic const float AVS_MEL_WEIGHTS[AVS_MEL_NNZ] = {\n")
        for i in range(0, len(weights), 6):
            f.write("    " + ", ".join(f"{x:.9e}f" for x in weights[i:i + 6]) + ",\n")
        f.write("};\n\n#endif\n")
    print(f"Wrote espidf_experiments/w_standard_mel_data.h: W={w.shape}, "
          f"Mel nonzeros={len(weights)}, reference={reference:.9g}")


if __name__ == "__main__":
    main()
