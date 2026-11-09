"""Export old W and 64/128-band Mel candidates for the Arduino experiment viewer."""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np


def write_floats(file, name: str, array: np.ndarray, per_line: int = 6) -> None:
    values = np.asarray(array, dtype=np.float32).ravel(order="C")
    file.write(f"static const float {name}[{values.size}] = {{\n")
    for start in range(0, values.size, per_line):
        row = ", ".join(f"{float(v):.9e}f" for v in values[start:start + per_line])
        file.write(f"    {row},\n")
    file.write("};\n\n")


def export_mel(file, prefix: str, bundle_path: Path) -> None:
    with np.load(bundle_path, allow_pickle=False) as data:
        w = np.asarray(data["W"], dtype=np.float32)
        bank = np.asarray(data["mel_bank"], dtype=np.float32)
        reference = float(data["compressor_reference"])
        mel_input = str(data["mel_input"].item())
        compressor = str(data["compressor"].item())
    if w.shape[1] != 48 or bank.shape != (w.shape[0], 1025):
        raise ValueError(f"Unexpected Mel model shapes: W={w.shape}, bank={bank.shape}")
    if mel_input != "magnitude" or compressor != "log1p":
        raise ValueError(f"{prefix} needs magnitude/log1p, got {mel_input}/{compressor}")
    n_mels = w.shape[0]
    offsets = [0]
    indices: list[int] = []
    weights: list[float] = []
    for row in bank:
        active = np.flatnonzero(row)
        indices.extend(int(i) for i in active)
        weights.extend(float(row[i]) for i in active)
        offsets.append(len(indices))
    file.write(f"#define {prefix}_ROWS {n_mels}u\n")
    file.write(f"#define {prefix}_REFERENCE {reference:.9e}f\n")
    file.write(f"#define {prefix}_NNZ {len(indices)}u\n")
    write_floats(file, f"{prefix}_W", w)
    file.write(f"static const uint16_t {prefix}_OFFSETS[{n_mels + 1}] = {{\n    ")
    file.write(", ".join(str(v) for v in offsets))
    file.write("\n};\n")
    file.write(f"static const uint16_t {prefix}_BIN_INDEX[{prefix}_NNZ] = {{\n")
    for start in range(0, len(indices), 16):
        file.write("    " + ", ".join(str(v) for v in indices[start:start + 16]) + ",\n")
    file.write("};\n")
    file.write(f"static const float {prefix}_WEIGHTS[{prefix}_NNZ] = {{\n")
    for start in range(0, len(weights), 6):
        file.write("    " + ", ".join(f"{v:.9e}f" for v in weights[start:start + 6]) + ",\n")
    file.write("};\n\n")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "esp32_s3_n16r8" / "arduino_experiments" / "AVS_ESP32S3_Experiment_Viewer" / "experiment_models.h")
    args = parser.parse_args()
    old_w = np.asarray(np.load(root / "models" / "nmf" / "W_standard.npy"), dtype=np.float32)
    if old_w.shape != (1025, 48):
        raise ValueError(f"Expected legacy W shape (1025,48), got {old_w.shape}")
    mel64 = root / "experiments" / "pipeline_k48_mel" / "models" / "W_standard_mel.npz"
    mel128 = root / "experiments" / "pipeline_k48_mel" / "runs" / "k128_magnitude_log1p" / "W_standard_mel.npz"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        file.write("#ifndef AVS_EXPERIMENT_MODELS_H\n#define AVS_EXPERIMENT_MODELS_H\n\n#include <stdint.h>\n\n")
        file.write("#define AVS_OLD_ROWS 1025u\n")
        write_floats(file, "AVS_OLD_W", old_w)
        export_mel(file, "AVS_MEL64", mel64)
        export_mel(file, "AVS_MEL128", mel128)
        file.write("#endif\n")
    print(f"Wrote Experiment Viewer model header: {args.output.name}")


if __name__ == "__main__":
    main()
