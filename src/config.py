
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Reproducibility
RANDOM_SEED = 42

# Image settings
IMAGE_SIZE = 224
BATCH_SIZE = 8

# Training settings
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# Model settings
MODEL_NAME = "resnet18"
NUM_CLASSES = 3
PRETRAINED = True

# Training components
OPTIMIZER_NAME = "Adam"
LOSS_FUNCTION = "CrossEntropyLoss"

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping
CLASS_NAMES = [
    "Meningioma",
    "Glioma",
    "Pituitary Tumor"
]

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix"
]


if __name__ == "__main__":
    print("BRAIN TUMOR DETECTION - CONFIGURATION")
    print("====================================")
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Model:", MODEL_NAME)
    print("Classes:", CLASS_NAMES)
    print("Pretrained:", PRETRAINED)
    print("Device:", DEVICE)
