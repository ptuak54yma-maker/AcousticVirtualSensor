#!/usr/bin/env python3
"""Export the existing W_standard.npy as a C float array; does not train NMF."""
import argparse
from pathlib import Path
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="models/nmf/W_standard.npy")
    parser.add_argument("--output", default="embedded/src/w_standard_data.h")
    args = parser.parse_args()

    w = np.load(Path(args.input))
    if w.shape != (1025, 48):
        raise SystemExit(f"Expected W shape (1025, 48); got {w.shape}")
    if not np.isfinite(w).all() or (w < 0).any():
        raise SystemExit("Dictionary must be finite and non-negative")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#ifndef AVS_W_STANDARD_DATA_H\n#define AVS_W_STANDARD_DATA_H\n\n")
        f.write("#define AVS_W_ROWS 1025u\n#define AVS_W_COLS 48u\n")
        f.write("/* Generated from the frozen W_standard.npy; row-major [freq][component]. */\n")
        f.write("static const float AVS_W_STANDARD[AVS_W_ROWS * AVS_W_COLS] = {\n")
        flat = w.astype(np.float32, copy=False).ravel(order="C")
        for i in range(0, flat.size, 6):
            f.write("    " + ", ".join(f"{float(x):.9e}f" for x in flat[i:i+6]) + ",\n")
        f.write("};\n\n#endif\n")
    # Windows may use a legacy console encoding that cannot print the
    # repository's Vietnamese path. Keep this utility successful in that case.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    print(f"Wrote {out} ({flat.size} float32 values; {flat.nbytes} bytes raw)")

if __name__ == "__main__":
    main()
