"""Train a separate K=24 dictionary in the new Mel/compressor domain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.decomposition import NMF

try:
    from . import experiment_config as config
    from .mel_frontend import compress_mel, fit_reference, make_mel_filterbank, mel_spectrogram
except ImportError:  # also support direct execution from this folder
    import experiment_config as config
    from mel_frontend import compress_mel, fit_reference, make_mel_filterbank, mel_spectrogram


def load_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != config.SAMPLE_RATE:
        raise ValueError(f"{path.name}: expected {config.SAMPLE_RATE} Hz, got {sample_rate} Hz.")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"{path.name}: unsupported audio shape {audio.shape}.")
    audio = audio - np.float32(np.mean(audio, dtype=np.float64))
    return np.ascontiguousarray(audio, dtype=np.float32)


def fit_source_dictionary(
    source_mels: list[np.ndarray],
    reference: float,
    components: int,
    alpha_h: float,
) -> tuple[np.ndarray, int]:
    """Fit one source-specific W with frequency/Mel bins on rows."""
    compressed = [
        compress_mel(mel, reference, config.COMPRESSOR, config.POWER_GAMMA)
        for mel in source_mels
    ]
    v = np.ascontiguousarray(np.concatenate(compressed, axis=1), dtype=np.float32)
    model = NMF(
        n_components=components,
        init=config.NMF_INIT,
        solver="cd",
        beta_loss="frobenius",
        alpha_H=alpha_h,
        l1_ratio=0.0,
        max_iter=config.NMF_MAX_ITER,
        tol=config.NMF_TOL,
        random_state=config.NMF_RANDOM_SEED,
    )
    # For X shaped (Mel bins, frames), fit_transform returns W shaped (Mel, K).
    w = model.fit_transform(v)
    return np.ascontiguousarray(w, dtype=np.float32), int(model.n_iter_)


def main() -> None:
    groups = {
        "event": ("phoi_*.wav", config.EVENT_COMPONENTS, config.NMF_EVENT_ALPHA_H),
        "bowl": ("noisebowl_*.wav", config.BOWL_COMPONENTS, config.NMF_BOWL_ALPHA_H),
        "env": ("noiseevir_*.wav", config.ENV_COMPONENTS, config.NMF_ENV_ALPHA_H),
    }
    mel_bank = make_mel_filterbank()
    group_mels: dict[str, list[np.ndarray]] = {}
    all_mels: list[np.ndarray] = []
    group_files: dict[str, list[str]] = {}

    for name, (pattern, _components, _alpha_h) in groups.items():
        files = sorted(config.TRAIN_DIR.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No {name} source WAVs found for pattern {pattern!r} in {config.TRAIN_DIR}."
            )
        mels = [
            mel_spectrogram(load_mono(path), mel_bank, config.MEL_INPUT)
            for path in files
        ]
        group_mels[name] = mels
        group_files[name] = [str(path.relative_to(config.PROJECT_ROOT)) for path in files]
        all_mels.extend(mels)
        print(f"Prepared {name}: {len(files)} WAV(s), {sum(m.shape[1] for m in mels)} frames")

    # Use a single source-derived scale for every group and for later inference.
    reference = fit_reference(all_mels)
    print(f"Compressor={config.COMPRESSOR}; shared reference={reference:.8g}")

    dictionaries: dict[str, np.ndarray] = {}
    iterations: dict[str, int] = {}
    for name, (_pattern, components, alpha_h) in groups.items():
        print(f"Training W_{name}: K={components}, alpha_H={alpha_h}", flush=True)
        dictionaries[name], iterations[name] = fit_source_dictionary(
            group_mels[name], reference, components, alpha_h
        )
        if iterations[name] >= config.NMF_MAX_ITER:
            print(f"WARNING: W_{name} reached max_iter={config.NMF_MAX_ITER}.", flush=True)
        else:
            print(f"  W_{name} converged in {iterations[name]} iterations.", flush=True)

    w = np.ascontiguousarray(
        np.concatenate(
            [dictionaries["event"], dictionaries["bowl"], dictionaries["env"]], axis=1
        ),
        dtype=np.float32,
    )
    expected_shape = (config.N_MELS, config.TOTAL_COMPONENTS)
    if w.shape != expected_shape:
        raise RuntimeError(f"Expected W shape {expected_shape}, got {w.shape}.")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.MODEL_PATH,
        W=w,
        W_event=dictionaries["event"],
        W_bowl=dictionaries["bowl"],
        W_env=dictionaries["env"],
        mel_bank=mel_bank,
        compressor_reference=np.float32(reference),
        compressor=np.asarray(config.COMPRESSOR),
        power_gamma=np.float32(config.POWER_GAMMA),
        mel_input=np.asarray(config.MEL_INPUT),
        n_fft=np.int32(config.N_FFT),
        hop_length=np.int32(config.HOP_LENGTH),
        n_mels=np.int32(config.N_MELS),
        component_counts=np.asarray(
            [config.EVENT_COMPONENTS, config.BOWL_COMPONENTS, config.ENV_COMPONENTS],
            dtype=np.int32,
        ),
        nmf_iterations=np.asarray(
            [iterations["event"], iterations["bowl"], iterations["env"]], dtype=np.int32
        ),
        nmf_init=np.asarray(config.NMF_INIT),
        source_files=np.asarray(
            [";".join(group_files[name]) for name in ("event", "bowl", "env")]
        ),
    )
    print(f"Saved experimental K=24 dictionary bundle with shape {w.shape}.")


if __name__ == "__main__":
    main()
