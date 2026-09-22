from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from controlled_editor.exceptions import ImageLoadError
from controlled_editor.image.image import Image


SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image into the canonical integral representation

    Returns:
        RGB NumPy array with shape (H, W, 3) and dtype uint8.

    Raises:
        ImageLoadError: If the image cannot be loaded.
    """

    image_path = Path(path)

    if not image_path.is_file():
        raise ImageLoadError(f"Image file does not exist: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"Unsupported image format: {image_path.suffix}")

    try:
        with PILImage.open(image_path) as image:
            rgb_image = image.convert("RGB")
            return np.array(rgb_image, dtype=np.uint8).copy()

    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError(f"Unable to load image: {image_path}") from exc

    return Image(data)


def save_image(image: np.ndarray, path: str | Path) -> None:

    image_path = Path(path)

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"Unsupported image format: {image_path.suffix}")

    try:
        PILImage.fromarray(image, mode="RGB").save(image_path)

    except (OSError, ValueError) as exc:
        raise ImageLoadError(f"Unable to save image: {image_path}") from exc
