"""Load peak settings fitted by the matching Python model configuration."""
from __future__ import annotations

import json
from pathlib import Path


def load_peak_params(root: Path, profile: str) -> dict[str, float | int]:
    """Return saved prominence/distance without fitting on the evaluation split."""
    experiment = root / "experiments" / "pipeline_k48_mel"
    if profile == "legacy_nnls":
        path = root / "models" / "peak_detection" / "peak_params.json"
        key = None
    elif profile in ("m64_nnls",) or profile.startswith("m64_fista"):
        path = experiment / "results" / "fista_peak_params.json"
        key = "python_scipy_nnls" if profile == "m64_nnls" else f"fista_{profile.removeprefix('m64_fista')}"
    elif profile == "m128_nnls" or profile == "m128_fista2048":
        path = experiment / "runs" / "k128_magnitude_log1p" / "fista_peak_params.json"
        key = "python_scipy_nnls" if profile == "m128_nnls" else "fista_2048"
    else:
        raise ValueError(f"No Python model/peak-parameter mapping for profile {profile!r}.")

    if not path.is_file():
        raise FileNotFoundError(f"Saved Python peak parameters are missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data if key is None else data.get(key)
    if not isinstance(params, dict) or "prominence" not in params or "distance" not in params:
        raise ValueError(f"No matching saved prominence/distance for {profile!r} in {path}")
    return {"prominence": float(params["prominence"]), "distance": int(params["distance"])}
