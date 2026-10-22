#!/usr/bin/env python3
"""Create a short reproducible 44.1 kHz WAV for C++/Python smoke comparison."""
from pathlib import Path
import argparse
import numpy as np
import soundfile as sf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("embedded/build/avs_smoke.wav"))
    args = parser.parse_args()
    sr = 44100
    n = 8192
    t = np.arange(n, dtype=np.float64) / sr
    signal = (
        0.30 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.12 * np.sin(2.0 * np.pi * 1700.0 * t)
        + 0.02 * np.sin(2.0 * np.pi * 7200.0 * t)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), signal.astype(np.float32), sr, subtype="PCM_16")
    print(f"Wrote smoke WAV: {args.output} ({n} samples, {sr} Hz)")

if __name__ == "__main__":
    main()
