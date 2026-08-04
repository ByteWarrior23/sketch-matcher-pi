"""
config.py — Pi Deployment Configuration
Copy this to the Pi along with the model files.

Only contains settings relevant to inference (not training).
"""

from pathlib import Path

# Model paths
MODEL_DIR = Path(__file__).parent / "model_data"
TFLITE_FILENAME = "sketch_matcher.tflite"
PHOTO_EMBEDDINGS_FILENAME = "photo_embeddings.npy"
PHOTO_LABELS_FILENAME = "photo_labels.npy"
LABELS_FILENAME = "labels.json"

# Image preprocessing (MUST match training pipeline)
IMG_SIZE = 224
PADDING = 10
BINARY_THRESHOLD = 128

# Camera
PI_CAMERA_RESOLUTION = (1920, 1080)
PI_CAMERA_FRAMERATE = 30

# Matching
PI_CONFIDENCE_THRESHOLD = 0.75
PI_REJECT_THRESHOLD = 0.45     # below top-1 similarity -> "NOT FOUND"
PI_SHOW_TOP_K = 3

# GPIO pins
PI_GPIO_BUTTON_PIN = 17
PI_GPIO_LED_GREEN = 27
PI_GPIO_LED_RED = 22

# Logging
LOG_LEVEL = "INFO"
