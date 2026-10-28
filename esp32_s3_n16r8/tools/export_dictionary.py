#!/usr/bin/env python3
"""Export the existing frozen NMF dictionary to the ESP-IDF firmware header."""
from pathlib import Path
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../models/nmf/W_standard.npy")
    parser.add_argument("--output", default="main/w_standard_data.h")
    args = parser.parse_args()

    weights = np.load(Path(args.input))
    if weights.shape != (1025, 48):
        raise SystemExit(f"Expected dictionary shape (1025, 48), got {weights.shape}")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise SystemExit("Dictionary must contain finite non-negative values")

    values = weights.astype(np.float32, copy=False).ravel(order="C")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#ifndef AVS_W_STANDARD_DATA_H\n#define AVS_W_STANDARD_DATA_H\n\n")
        f.write("#define AVS_W_ROWS 1025u\n#define AVS_W_COLS 48u\n")
        f.write("/* Generated from the project's frozen models/nmf/W_standard.npy. */\n")
        f.write("static const float AVS_W_STANDARD[AVS_W_ROWS * AVS_W_COLS] = {\n")
        for i in range(0, values.size, 6):
            row = ", ".join(f"{float(x):.9e}f" for x in values[i:i + 6])
            f.write(f"    {row},\n")
        f.write("};\n\n#endif\n")
    print(f"Wrote {output} ({values.size} float32 values)")


if __name__ == "__main__":
    main()
