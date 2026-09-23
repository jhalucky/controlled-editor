import numpy as np
import pytest

from controlled_editor.exceptions import ImageValidationError
from controlled_editor.image.mask import Mask


def test_binary_mask_creation() -> None:
    data = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    mask = Mask(data)

    assert mask.height == 3
    assert mask.width == 3
    assert mask.shape == (3, 3)
    assert mask.is_binary
    assert not mask.is_soft


def test_soft_mask_creation() -> None:
    data = np.array(
        [
            [0.0, 0.2, 0.5],
            [0.1, 0.8, 1.0],
        ],
        dtype=np.float32,
    )

    mask = Mask(data)

    assert mask.shape == (2, 3)
    assert mask.is_soft
    assert not mask.is_binary


def test_mask_must_be_2d() -> None:
    data = np.zeros((10, 10, 1), dtype=np.uint8)

    with pytest.raises(ImageValidationError):
        Mask(data)


def test_binary_mask_only_allows_zero_and_one() -> None:
    data = np.array(
        [
            [0, 1],
            [0, 255],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(ImageValidationError):
        Mask(data)


def test_soft_mask_must_be_between_zero_and_one() -> None:
    data = np.array(
        [
            [0.0, 0.5],
            [1.0, 1.5],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ImageValidationError):
        Mask(data)


def test_soft_mask_rejects_non_finite_values() -> None:
    data = np.array(
        [
            [0.0, np.nan],
            [0.5, 1.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ImageValidationError):
        Mask(data)


def test_mask_rejects_unsupported_dtype() -> None:
    data = np.zeros((10, 10), dtype=np.int32)

    with pytest.raises(ImageValidationError):
        Mask(data)