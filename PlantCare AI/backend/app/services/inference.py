import logging
from functools import lru_cache

import torch
import torch.nn.functional as F
from torchvision import transforms

from app.config import MODEL_PATH, CLASS_NAMES_PATH, MODEL_ARCH, IMAGE_SIZE
from app.services.model_factory import build_model
from app.services.preprocessing import preprocess_upload

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained checkpoint has been placed at MODEL_PATH yet."""


def _load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        raise ModelNotAvailableError(
            f"Class names file not found at {CLASS_NAMES_PATH}. Run train/train.py first, "
            "or copy an exported checkpoint + class_names.txt into backend/train/checkpoints/."
        )
    with open(CLASS_NAMES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_model():
    """Loads the model + class names once per process and caches them.
    Raises ModelNotAvailableError with a clear message if training hasn't
    been run yet, rather than crashing the whole API on startup."""
    if not MODEL_PATH.exists():
        raise ModelNotAvailableError(
            f"No trained model found at {MODEL_PATH}. Run train/train.py to produce one, "
            "then restart the API (or set MODEL_PATH to point at your checkpoint)."
        )

    class_names = _load_class_names()
    device = get_device()

    model = build_model(MODEL_ARCH, num_classes=len(class_names), pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Loaded %s with %d classes on %s", MODEL_ARCH, len(class_names), device)
    return model, class_names


def predict(image_bytes: bytes, top_k: int = 3) -> dict:
    """Runs the full pipeline: preprocess -> forward pass -> softmax ->
    top-k labels + confidences. Returns plain python types (no tensors)
    so callers can serialize directly."""
    model, class_names = load_model()
    device = get_device()

    pil_image = preprocess_upload(image_bytes)
    tensor = _transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    k = min(top_k, len(class_names))
    top_probs, top_idxs = torch.topk(probs, k)

    results = [
        {"label": class_names[idx], "confidence": round(float(p), 4)}
        for p, idx in zip(top_probs.tolist(), top_idxs.tolist())
    ]
    return {
        "predicted_class": results[0]["label"],
        "confidence": results[0]["confidence"],
        "top_k": results,
    }
