"""
OpenCV preprocessing. Kept as a distinct step (rather than doing
everything in torchvision transforms) so it's easy to point to this
file specifically when explaining "where does OpenCV come in".
"""
import cv2
import numpy as np
from PIL import Image

from app.config import IMAGE_SIZE


def read_image_bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Decode raw upload bytes into a BGR OpenCV array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image -- unsupported or corrupt file.")
    return img


def denoise_and_resize(img_bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Light denoising + resize. Leaf photos from phone cameras are often
    noisy/compressed, so a mild bilateral filter helps before the model
    ever sees the image."""
    denoised = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=50, sigmaSpace=50)
    resized = cv2.resize(denoised, (size, size), interpolation=cv2.INTER_AREA)
    return resized


def cv2_bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def preprocess_upload(image_bytes: bytes) -> Image.Image:
    """Full pipeline: bytes -> decode -> denoise/resize -> PIL image
    ready for the torchvision normalization transform."""
    img_bgr = read_image_bytes_to_cv2(image_bytes)
    img_bgr = denoise_and_resize(img_bgr)
    return cv2_bgr_to_pil(img_bgr)
