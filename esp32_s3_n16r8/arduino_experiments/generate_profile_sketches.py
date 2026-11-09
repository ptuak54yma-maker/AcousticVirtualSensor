#!/usr/bin/env python3
"""Generate one Arduino IDE sketch per experiment profile."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIEWER = ROOT / "esp32_s3_n16r8" / "arduino_experiments" / "AVS_ESP32S3_Experiment_Viewer"
OUTPUT = ROOT / "esp32_s3_n16r8" / "arduino_experiments" / "AVS_ESP32S3_Profile_Sketches"
SOURCE_INO = VIEWER / "AVS_ESP32S3_Experiment_Viewer.ino"
MODEL_HEADER = VIEWER / "experiment_models.h"
PROFILES = [
    "legacy_nnls", "m64_nnls", "m128_nnls",
    "m64_fista8", "m64_fista16", "m64_fista32", "m64_fista64",
    "m64_fista128", "m64_fista256", "m64_fista512", "m64_fista1024",
    "m64_fista2048", "m64_fista4096", "m128_fista512",
]


def main() -> None:
    if not MODEL_HEADER.exists():
        raise SystemExit("Model header missing. First run tools/export_arduino_experiment_models.py")
    source = SOURCE_INO.read_text(encoding="utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for profile_id, profile in enumerate(PROFILES):
        sketch_dir = OUTPUT / profile
        sketch_dir.mkdir(parents=True, exist_ok=True)
        (sketch_dir / f"{profile}.ino").write_text(
            f"#define AVS_PROFILE_ID {profile_id}\n" + source,
            encoding="utf-8", newline="\n",
        )
        shutil.copy2(MODEL_HEADER, sketch_dir / MODEL_HEADER.name)
    print(f"Generated {len(PROFILES)} sketches under {OUTPUT.relative_to(ROOT)}")
    for profile in PROFILES:
        print(f"  {OUTPUT.relative_to(ROOT) / profile / (profile + '.ino')}")


if __name__ == "__main__":
    main()
