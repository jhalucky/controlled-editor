import numpy as np
import pytest

from controlled_editor.exceptions import ImageValidationError
from controlled_editor.image.validation import (
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
    validate_image,
)


def test_valid_image_passes():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    validate_image(image)


def test_non_numpy_input_fails():
    image = [[0, 0, 0]]

    with pytest.raises(ImageValidationError, match="numpy.ndarray"):
        validate_image(image)


def test_two_dimensional_image_fails():
    image = np.zeros((720, 1280), dtype=np.uint8)

    with pytest.raises(ImageValidationError, match="3-dimensional"):
        validate_image(image)


def test_wrong_channel_count_fails():
    image = np.zeros((720, 1280, 4), dtype=np.uint8)

    with pytest.raises(ImageValidationError, match="3 channels"):
        validate_image(image)


def test_wrong_dtype_fails():
    image = np.zeros((720, 1280, 3), dtype=np.float32)

    with pytest.raises(ImageValidationError, match="uint8"):
        validate_image(image)


def test_image_too_small_fails():
    image = np.zeros(
        (MIN_IMAGE_DIMENSION - 1, 1280, 3),
        dtype=np.uint8,
    )

    with pytest.raises(ImageValidationError, match="too small"):
        validate_image(image)


def test_image_too_large_fails():
    image = np.zeros(
        (MAX_IMAGE_DIMENSION + 1, 1000, 3),
        dtype=np.uint8,
    )

    with pytest.raises(ImageValidationError, match="too large"):
        validate_image(image)


def test_single_channel_image_fails():
    image = np.zeros((720, 1280, 1), dtype=np.uint8)

    with pytest.raises(ImageValidationError, match="3 channels"):
        validate_image(image)


def test_empty_dimension_fails():
    image = np.zeros((0, 1280, 3), dtype=np.uint8)

    with pytest.raises(ImageValidationError, match="too small"):
        validate_image(image)
