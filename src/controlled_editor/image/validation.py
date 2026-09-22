from __future__ import annotations

import numpy as np

from controlled_editor.exceptions import ImageValidationError

MIN_IMAGE_DIMENSION = 64
MAX_IMAGE_DIMENSION = 12_000


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ImageValidationError(f"Expected numpy.ndarray, got {type(image).__name__}")

    if image.ndim != 3:
        raise ImageValidationError(
            f"Expected a 3-dimensional image array, got {image.ndim} dimensions"
        )
    height, width, channels = image.shape

    if channels != 3:
        raise ImageValidationError(f"Expected 3 channels (RGB), got {channels}")

    if height < MIN_IMAGE_DIMENSION or width < MIN_IMAGE_DIMENSION:
        raise ImageValidationError(
            "Image dimensions are too small: "
            f"{width}x{height}. Minimum dimension is "
            f"{MAX_IMAGE_DIMENSION}px"
        )

    if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
        raise ImageValidationError(
            "Image dimensions are too large: "
            f"{width}x{height}. Maximum dimension is"
            f"{MAX_IMAGE_DIMENSION}px"
        )

    if image.dtype != np.uint8:
        raise ImageValidationError(f"Expected dtype uint8, got {image.dtype}")

    if not np.isfinite(image).all():
        raise ImageValidationError("Image contains non-finite pixel values")
