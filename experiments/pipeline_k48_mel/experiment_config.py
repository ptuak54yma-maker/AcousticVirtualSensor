"""Parameters for the 48-component log-Mel NMF model for ESP32-S3."""

from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
MODEL_DIR = EXPERIMENT_DIR / "models"
MODEL_PATH = MODEL_DIR / "W_standard_mel.npz"
STANDARD_W_PATH = PROJECT_ROOT / "models" / "nmf" / "W_standard_mel.npy"

SAMPLE_RATE = 44_100
N_FFT = 2_048
HOP_LENGTH = 1_024
WINDOW = "hann"
CENTER = False

N_MELS = 64
F_MIN = 20.0
F_MAX = SAMPLE_RATE / 2
MEL_NORM = "slaney"
MEL_HTK = False
MEL_INPUT = "magnitude"

# The scale is estimated once from the training-source WAVs and saved in the
# model bundle. Inference must reuse that exact value.
COMPRESSOR = "log1p"
POWER_GAMMA = 0.5

EVENT_COMPONENTS = 24
BOWL_COMPONENTS = 12
ENV_COMPONENTS = 12
TOTAL_COMPONENTS = EVENT_COMPONENTS + BOWL_COMPONENTS + ENV_COMPONENTS

NMF_MAX_ITER = 8_000
MAX_TRAIN_FRAMES_PER_WAV = 2_048
NMF_TOL = 1e-4
NMF_RANDOM_SEED = 42
NMF_INIT = "nndsvda"
NMF_EVENT_ALPHA_H = 0.0
NMF_BOWL_ALPHA_H = 0.0
NMF_ENV_ALPHA_H = 0.0

NNLS_MAX_ITER = 2_000