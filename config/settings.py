"""
Configuration settings for the deepfake detection system
"""
import torch
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FRAMES_DIR = DATA_DIR / "frames"
FACES_DIR = DATA_DIR / "faces"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, FRAMES_DIR, FACES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model paths
DEEPFAKE_MODEL_PATH = MODELS_DIR / "deepfake_model.pth"
FACE_DETECTOR_PATH = MODELS_DIR / "face_detector.tflite"

# Detection settings
DEFAULT_FAKE_THRESHOLD = 0.55
HIGH_CONFIDENCE_THRESHOLD = 0.7
GAP_TOLERANCE = 1.5  # seconds
MIN_SEGMENT_DURATION = 0.5  # seconds

# Video processing
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_FRAMES = 500  # Maximum frames to process (for performance)
FRAME_INTERVAL = 1  # Extract every Nth frame

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "deepfake_detection.log"

class Config:
    """Configuration class"""
    def __init__(self):
        self.BASE_DIR = BASE_DIR
        self.DATA_DIR = DATA_DIR
        self.MODELS_DIR = MODELS_DIR
        self.FRAMES_DIR = FRAMES_DIR
        self.FACES_DIR = FACES_DIR
        self.DEEPFAKE_MODEL_PATH = DEEPFAKE_MODEL_PATH
        self.FACE_DETECTOR_PATH = FACE_DETECTOR_PATH
        self.DEFAULT_FAKE_THRESHOLD = DEFAULT_FAKE_THRESHOLD
        self.HIGH_CONFIDENCE_THRESHOLD = HIGH_CONFIDENCE_THRESHOLD
        self.GAP_TOLERANCE = GAP_TOLERANCE
        self.MIN_SEGMENT_DURATION = MIN_SEGMENT_DURATION
        self.TARGET_SIZE = TARGET_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_FRAMES = MAX_FRAMES
        self.FRAME_INTERVAL = FRAME_INTERVAL
        self.DEVICE = DEVICE
        self.LOG_LEVEL = LOG_LEVEL
        self.LOG_FORMAT = LOG_FORMAT
        self.LOG_FILE = LOG_FILE

config = Config()