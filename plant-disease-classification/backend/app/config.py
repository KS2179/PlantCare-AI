"""
Central configuration for the backend.
Reads from environment variables where useful so Docker / local runs
can override paths without touching code.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where the trained model checkpoint lives. Produced by train/train.py.
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "train" / "checkpoints" / "best_model.pt"))

# Class names file produced during training (one class per line, in the
# same order as the model's output logits).
CLASS_NAMES_PATH = Path(os.getenv("CLASS_NAMES_PATH", BASE_DIR / "train" / "checkpoints" / "class_names.txt"))

# Static disease knowledge base (symptoms / causes / prevention / treatment).
DISEASE_INFO_PATH = Path(os.getenv("DISEASE_INFO_PATH", BASE_DIR / "data" / "disease_info.json"))

# SQLite database file for prediction history.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'history.db'}")

# Model architecture used for inference. Must match what train.py produced.
MODEL_ARCH = os.getenv("MODEL_ARCH", "resnet50")  # "resnet50" or "efficientnet_b0"

IMAGE_SIZE = 224
CONFIDENCE_DECIMALS = 4

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "data" / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
