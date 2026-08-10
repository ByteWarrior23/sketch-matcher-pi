"""Rock-Paper-Scissors project config (shared by train/eval/deploy)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Classes (0,1,2 = rock, paper, scissors)
CLASSES = ["rock", "paper", "scissors"]
NUM_CLASSES = len(CLASSES)

# Image pipeline
IMG_SIZE = 224
IMG_CHANNELS = 3
MEAN = 127.5
STD = 127.5  # MobileNetV3 built-in rescale equivalent for TFLite int8 path

# Training
BACKBONE = "mobilenetv3large"
BATCH_SIZE = 32
EPOCHS_STAGE1 = 20   # frozen backbone, head only
EPOCHS_STAGE2 = 30   # unfreeze last 12 layers
EPOCHS_STAGE3 = 40   # full fine-tune
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4
LR_STAGE3 = 1e-5
PATIENCE = 12
RANDOM_SEED = 42

# Paths
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
EXPORT_DIR = ROOT / "pi_deploy" / "model_data"

# Game logic
PI_STRATEGY = "counter"  # counter: Pi picks the move that beats the last player move; random: uniform
HOLD_FRAMES = 12         # frames a move must be consistent before accepted
MIN_CONFIDENCE = 0.75    # below this, show "no gesture"
