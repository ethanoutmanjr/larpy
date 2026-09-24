"""Image processing utilities."""

import io
from PIL import Image, ImageOps
import numpy as np
from typing import Tuple


def validate_image(image_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate that the bytes represent a valid image.

    Returns:
        (is_valid, error_message)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verify it's actually an image
        return True, ""
    except Exception as e:
        return False, str(e)


def preprocess_image(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> Image.Image:
    """
    Preprocess an image for model inference.

    Args:
        image_bytes: Raw image bytes
        target_size: Target dimensions
        normalize: Whether to normalize pixel values

    Returns:
        Preprocessed PIL Image
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)

    if normalize:
        # Convert to numpy and normalize
        img_array = np.array(img) / 255.0
        img = Image.fromarray((img_array * 255).astype(np.uint8))

    return img


def image_to_tensor(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to normalized numpy array."""
    img = preprocess_image(image_bytes)
    img_array = np.array(img) / 255.0
    return img_array.transpose(2, 0, 1)  # HWC to CHW


def get_image_format(image_bytes: bytes) -> str:
    """Detect image format from bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.format or "JPEG"
    except:
        return "JPEG"


def compress_image(image_bytes: bytes, quality: int = 85) -> bytes:
    """Compress image to reduce upload size."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()
