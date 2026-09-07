"""
===============================================================================
Acoustic Virtual Sensor - Project Configuration
===============================================================================
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

# Dataset root directories
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_AUDIO_DIR = DATA_DIR / "train"
TEST_AUDIO_DIR = DATA_DIR / "test"
TRAIN_ANNOTATION_DIR = DATA_DIR / "annotations" / "train"
TEST_ANNOTATION_DIR = DATA_DIR / "annotations" / "test"
CRNN_DATASET_DIR = DATA_DIR / "datasets"

# Output and Log directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# Ensure runtime directories exist
for folder in [LOG_DIR, FIGURES_DIR, RESULTS_DIR, TRAIN_ANNOTATION_DIR, TEST_ANNOTATION_DIR, CRNN_DATASET_DIR, NMF_MODEL_DIR, CRNN_MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. INPUT AUDIO & STREAMING PARAMETERS
# =============================================================================
AUDIO_SOURCE = "file"  # "file" or "microphone"
DEFAULT_TEST_WAV = "Bowl_0.wav"
SAMPLE_RATE = 44100  # Target sampling rate (Hz)

# Number of audio samples per streaming chunk (~23.22 ms)
STREAM_CHUNK_SIZE = 1024
WINDOW_SECONDS = 3.0  # Sliding window duration for feed rate estimation (~129 frames)
UPDATE_INTERVAL_SECONDS = 1.0

# File Naming Patterns
PATTERN_NMF_EVENT = "phoi_*.wav"
PATTERN_NMF_BOWL  = "noisebowl_*.wav"
PATTERN_NMF_ENV   = "noiseevir_*.wav"
PATTERN_CRNN_TRAIN = "Train_*.wav"
PATTERN_TEST_BOWL  = "Bowl_*.wav"

# =============================================================================
# 4. DSP & STFT PARAMETERS
# =============================================================================
N_FFT = 2048
WINDOW_LENGTH = 2048
HOP_LENGTH = 1024
WINDOW_FUNCTION = "hann"
CENTER_STFT = False  # Exact sample-to-frame alignment

SPECTROGRAM_TYPE = "log_magnitude"  # V = log(1 + |X|)
LOG_EPSILON = 1.0

# Mel Spectrogram (Hybrid CRNN)
N_MELS = 128
F_MIN = 20.0
F_MAX = SAMPLE_RATE // 2  # 22050 Hz

# =============================================================================
# 5. FIXED-DICTIONARY NMF PARAMETERS
# =============================================================================
W_STANDARD_FILENAME = "W_standard.npy"
W_STANDARD_PATH = NMF_MODEL_DIR / W_STANDARD_FILENAME

EVENT_COMPONENTS = 24
BOWL_COMPONENTS = 12
ENV_COMPONENTS = 12
TOTAL_COMPONENTS = EVENT_COMPONENTS + BOWL_COMPONENTS + ENV_COMPONENTS  # 48

NNLS_MAX_ITER = 20000

# =============================================================================
# 6. BASELINE PEAK DETECTION PARAMETERS
# =============================================================================

PEAK_DETECTION_DIR = MODEL_DIR / "peak_detection"
PEAK_PARAMS_PATH = PEAK_DETECTION_DIR / "peak_params.json"
PEAK_DETECTION_DIR.mkdir(parents=True, exist_ok=True)

# Giá trị mặc định dự phòng (Fallback defaults) nếu chưa chạy Grid Search
DEFAULT_PEAK_PROMINENCE = 0.5
DEFAULT_PEAK_DISTANCE_FRAMES = 3

# Grid Search Hyperparameter Space
GRID_SEARCH_PROMINENCES = (0.01, 1.0, 50)  # np.linspace(start, stop, num)
GRID_SEARCH_DISTANCES = (1, 51, 1)          # np.arange(start, stop, step)

# =============================================================================
# 7. CRNN SEQUENCE & ARCHITECTURE PARAMETERS
# =============================================================================
CRNN_INPUT_TYPE = "H_event_sum"
CRNN_INPUT_FEATURES = 1

SEQUENCE_LENGTH = 128  # ~2.97 seconds context
SEQUENCE_HOP = 64

# Topology
CNN_CHANNELS = 32
CNN_KERNEL_SIZE = 5
CNN_LAYERS = 2
RNN_TYPE = "GRU"
RNN_HIDDEN_SIZE = 32
RNN_LAYERS = 1
RNN_BIDIRECTIONAL = True
CRNN_DROPOUT = 0.2

# Training setup
CRNN_EPOCHS = 50
CRNN_BATCH_SIZE = 32
CRNN_LEARNING_RATE = 1e-3
CRNN_WEIGHT_DECAY = 1e-4
CRNN_RANDOM_SEED = 42
USE_POSITIVE_CLASS_WEIGHT = True

# Label smoothing
ANNOTATION_SMOOTHING_ENABLED = True
ANNOTATION_GAUSSIAN_SIGMA = 1.0

# Post-processing
CRNN_THRESHOLD = 0.5
MIN_EVENT_DISTANCE_FRAMES = 3
EVENT_REPRESENTATIVE_METHOD = "maximum_probability"

# =============================================================================
# 8. RUNTIME, GUI & LOGGING PARAMETERS
# =============================================================================
ENABLE_GUI = False
LOG_LEVEL = "INFO"
LOG_FILE = "sensor.log"

WINDOW_TITLE = "Acoustic Virtual Sensor"
WINDOW_WIDTH = 550
WINDOW_HEIGHT = 450
FONT_NAME = "Arial"
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 11
FONT_SIZE_VALUE = 20
REFRESH_RATE_MS = 50

# Output Paths
CRNN_MODEL_PATH = CRNN_MODEL_DIR / "crnn_model.pt"
EVAL_RESULTS_CSV = RESULTS_DIR / "comparison_results.csv"