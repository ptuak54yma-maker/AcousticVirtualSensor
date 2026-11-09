"""Inference helpers for the experimental 64-bin Mel / K=24 NMF pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import nnls

try:
    from . import experiment_config as config
    from .mel_frontend import compress_mel, mel_spectrogram
except ImportError:  # also support direct execution from this folder
    import experiment_config as config
    from mel_frontend import compress_mel, mel_spectrogram


def load_model(path: str | Path = config.MODEL_PATH) -> dict[str, object]:
    """Load W and preprocessing metadata from the experiment's NPZ bundle."""
    with np.load(Path(path), allow_pickle=False) as data:
        w = np.asarray(data["W"], dtype=np.float32)
        mel_bank = np.asarray(data["mel_bank"], dtype=np.float32)
        reference = float(data["compressor_reference"])
        compressor = str(data["compressor"].item())
        gamma = float(data["power_gamma"])
        mel_input = str(data["mel_input"].item())
        n_mels = int(data["n_mels"]) if "n_mels" in data else int(w.shape[0])
    expected_w_shape = (n_mels, config.TOTAL_COMPONENTS)
    if w.shape != expected_w_shape:
        raise ValueError(f"Expected W shape {expected_w_shape}, got {w.shape}.")
    if mel_bank.shape != (n_mels, config.N_FFT // 2 + 1):
        raise ValueError(f"Unexpected Mel bank shape {mel_bank.shape}.")
    return {
        "W": np.ascontiguousarray(w),
        "mel_bank": mel_bank,
        "reference": reference,
        "compressor": compressor,
        "gamma": gamma,
        "mel_input": mel_input,
        "n_mels": n_mels,
    }


def solve_nnls_batch(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Solve min ||v - W H|| with H >= 0, one frame at a time."""
    if v.ndim != 2 or w.ndim != 2 or v.shape[0] != w.shape[0]:
        raise ValueError(f"Incompatible V and W shapes: {v.shape}, {w.shape}.")
    h = np.empty((w.shape[1], v.shape[1]), dtype=np.float32)
    for frame in range(v.shape[1]):
        h[:, frame], _ = nnls(w, v[:, frame], maxiter=config.NNLS_MAX_ITER)
    return h


def extract_features(audio: np.ndarray, model: dict[str, object]) -> dict[str, np.ndarray]:
    """Return compressed Mel features, H, and three source-group envelopes."""
    mel = mel_spectrogram(
        audio,
        mel_bank=model["mel_bank"],
        mel_input=model["mel_input"],
    )
    v = compress_mel(
        mel,
        reference=model["reference"],
        compressor=model["compressor"],
        gamma=model["gamma"],
    )
    h = solve_nnls_batch(v, model["W"])
    e = config.EVENT_COMPONENTS
    b = e + config.BOWL_COMPONENTS
    return {
        "mel_compressed": v,
        "H": h,
        "H_event_sum": np.sum(h[:e], axis=0, dtype=np.float32),
        "H_bowl_sum": np.sum(h[e:b], axis=0, dtype=np.float32),
        "H_env_sum": np.sum(h[b:], axis=0, dtype=np.float32),
    }


def main() -> None:
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Run the experimental K=24 Mel-NMF pipeline.")
    parser.add_argument("wav", type=Path)
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    audio, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if sr != config.SAMPLE_RATE:
        raise SystemExit(f"Expected {config.SAMPLE_RATE} Hz, got {sr} Hz.")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    model = load_model(args.model)
    result = extract_features(audio, model)
    output = args.output or (
        config.EXPERIMENT_DIR / "results" / "features"
        / f"{args.wav.stem}.k24_mel_features.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    print(f"Saved K=24 Mel-NMF features: {output}")
    print(f"W={model['W'].shape}; frames={result['H'].shape[1]}; "
          f"compressor={model['compressor']}; ref={model['reference']:.6g}")


if __name__ == "__main__":
    main()
