"""
===============================================================================
Acoustic Virtual Sensor - Project Configuration
===============================================================================

All configurable parameters of the project are centralized in this file.
Uses pathlib.Path for cross-platform portability and relative path anchoring.
"""

from pathlib import Path

# =============================================================================
# 1. PROJECT METADATA
# =============================================================================
PROJECT_NAME = "Acoustic Virtual Sensor"
PROJECT_VERSION = "1.0.0"
AUTHOR = "Pham Minh Tu"

# =============================================================================
# 2. PATH DEFINITIONS (PORTABLE WITH PATHLIB)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Model weight directories
MODEL_DIR = PROJECT_ROOT / "models"
NMF_MODEL_DIR = MODEL_DIR / "nmf"
CRNN_MODEL_DIR = MODEL_DIR / "crnn"
PEAK_DETECTION_DIR = MODEL_DIR / "peak_detection"

# Dataset root directories
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_AUDIO_DIR = DATA_DIR / "train"
TEST_AUDIO_DIR = DATA_DIR / "test"
TRAIN_ANNOTATION_DIR = DATA_DIR / "annotations" / "train"
TEST_ANNOTATION_DIR = DATA_DIR / "annotations" / "test"
CRNN_DATASET_DIR = DATA_DIR / "datasets"

# Features Cache directories
FEATURES_DIR = DATA_DIR / "features"
TRAIN_FEATURES_DIR = FEATURES_DIR / "train"
TEST_FEATURES_DIR = FEATURES_DIR / "test"

# Output and Log directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# Ensure runtime directories exist
for folder in [
    LOG_DIR, FIGURES_DIR, RESULTS_DIR, TRAIN_ANNOTATION_DIR,
    TEST_ANNOTATION_DIR, CRNN_DATASET_DIR, TRAIN_FEATURES_DIR,
    TEST_FEATURES_DIR, NMF_MODEL_DIR, CRNN_MODEL_DIR, PEAK_DETECTION_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# File Patterns
PATTERN_CRNN_TRAIN = "Train_*.wav"
PATTERN_NMF_EVENT = "phoi_*.wav"
PATTERN_NMF_BOWL = "noisebowl_*.wav"
PATTERN_NMF_ENV = "noiseevir_*.wav"

# Artifact file paths
W_STANDARD_FILENAME = "W_standard.npy"
W_STANDARD_PATH = NMF_MODEL_DIR / W_STANDARD_FILENAME
CRNN_MODEL_PATH = CRNN_MODEL_DIR / "crnn_model.pt"
PEAK_PARAMS_PATH = PEAK_DETECTION_DIR / "peak_params.json"
EVAL_RESULTS_CSV = RESULTS_DIR / "comparison_results.csv"

# =============================================================================
# 3. INPUT AUDIO & STREAMING PARAMETERS
# =============================================================================
AUDIO_SOURCE = "file"
DEFAULT_TEST_WAV = "Bowl_3.wav"
SAMPLE_RATE = 44100
STREAM_CHUNK_SIZE = 1024

# =============================================================================
# 4. DSP & STFT PARAMETERS
# =============================================================================
N_FFT = 2048
WINDOW_LENGTH = 2048
HOP_LENGTH = 1024
WINDOW_FUNCTION = "hann"
CENTER_STFT = False

SPECTROGRAM_TYPE = "log_magnitude"
LOG_EPSILON = 1.0

# Mel Spectrogram
N_MELS = 128
F_MIN = 20.0
F_MAX = SAMPLE_RATE // 2

# =============================================================================
# 5. FIXED-DICTIONARY NMF PARAMETERS
# =============================================================================
EVENT_COMPONENTS = 24
BOWL_COMPONENTS = 12
ENV_COMPONENTS = 12
TOTAL_COMPONENTS = EVENT_COMPONENTS + BOWL_COMPONENTS + ENV_COMPONENTS  # 48

# NMF Training Hyperparameters (dời từ scripts/train_nmf.py về)
NMF_MAX_ITER = 20000
NMF_SOLVER = "cd"
NMF_BETA_LOSS = "frobenius"
NMF_INIT = "random"
NMF_L1_RATIO = 0.0
NMF_RANDOM_SEED = 42

NMF_EVENT_ALPHA_H = 0.2
NMF_BOWL_ALPHA_H = 0.0
NMF_ENV_ALPHA_H = 0.0

# NNLS Online Limits
NNLS_MAX_ITER = 20000

# =============================================================================
# 6. BASELINE PEAK DETECTION PARAMETERS
# =============================================================================
ENABLE_PEAK_DETECTION = True
DEFAULT_PEAK_PROMINENCE = 0.5
DEFAULT_PEAK_DISTANCE_FRAMES = 3

# =============================================================================
# 7. CRNN ARCHITECTURE & SEQUENCE PARAMETERS
# =============================================================================
CRNN_INPUT_TYPE = "H_event_sum"
CRNN_INPUT_FEATURES = 1

SEQUENCE_LENGTH = 128
SEQUENCE_HOP = 64

# Sub-branch channels & kernels (dời từ src/crnn/model.py về)
CRNN_MEL_CHANNELS = 32
CRNN_CONV2D_KERNEL_SIZE = 3

CRNN_HE_CHANNELS = 16
CRNN_HE_CONV1_KERNEL_SIZE = 5
CRNN_HE_CONV2_KERNEL_SIZE = 3

# Recurrent and Classifier
RNN_TYPE = "GRU"
RNN_HIDDEN_SIZE = 32
RNN_LAYERS = 1
RNN_BIDIRECTIONAL = True
CRNN_DROPOUT = 0.2
CRNN_CLASSIFIER_HIDDEN_SIZE = 32

# =============================================================================
# 8. CRNN TRAINING PARAMETERS
# =============================================================================
CRNN_EPOCHS = 50
CRNN_BATCH_SIZE = 32
CRNN_LEARNING_RATE = 1e-3
CRNN_WEIGHT_DECAY = 1e-4
CRNN_RANDOM_SEED = 42
CRNN_VALIDATION_SPLIT = 0.15  # (dời từ scripts/train_crnn.py về)
USE_POSITIVE_CLASS_WEIGHT = True

# =============================================================================
# 9. POST-PROCESSING & EVALUATION
# =============================================================================
CRNN_THRESHOLD = 0.5
MIN_EVENT_DISTANCE_FRAMES = 3
EVENT_REPRESENTATIVE_METHOD = "maximum_probability"
EVENT_MATCH_TOLERANCE_FRAMES = 3  # (dời từ src/evaluation/event_metrics.py về)

# =============================================================================
# 10. MANUAL ANNOTATION PARAMETERS
# =============================================================================
ANNOTATION_ENABLED = True
ANNOTATION_CLICK_TOLERANCE_FRAMES = 4      # (dời từ src/annotation/manual_label.py về)
ANNOTATION_LOCAL_SEARCH_RADIUS_FRAMES = 3  # (dời từ src/annotation/manual_label.py về)
ANNOTATION_SMOOTHING_ENABLED = True
ANNOTATION_GAUSSIAN_SIGMA = 1.0

# =============================================================================
# 11. RUNTIME VIRTUAL SENSOR
# =============================================================================
WINDOW_SECONDS = 3.0
UPDATE_INTERVAL_SECONDS = 1.0