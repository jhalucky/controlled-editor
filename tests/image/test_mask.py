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


def test_empty_mask() -> None:
    mask = Mask.empty(10, 20)

    assert mask.shape == (10, 20)
    assert mask.is_binary
    assert not np.any(mask.data)


def test_full_mask() -> None:
    mask = Mask.full(10, 20)

    assert mask.shape == (10, 20)
    assert mask.is_binary
    assert np.all(mask.data == 1)


def test_empty_mask_rejects_invalid_dimensions() -> None:
    with pytest.raises(ImageValidationError):
        Mask.empty(0, 10)

    with pytest.raises(ImageValidationError):
        Mask.empty(10, 0)


def test_full_mask_rejects_invalid_dimensions() -> None:
    with pytest.raises(ImageValidationError):
        Mask.full(-1, 10)

    with pytest.raises(ImageValidationError):
        Mask.full(10, -1)


def test_binary_mask_invert() -> None:
    data = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.uint8,
    )

    mask = Mask(data)
    inverted = mask.invert()

    expected = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(inverted.data, expected)


def test_invert_rejects_soft_mask() -> None:
    data = np.array(
        [
            [0.0, 1.0],
            [0.5, 0.2],
        ],
        dtype=np.float32,
    )

    mask = Mask(data)

    with pytest.raises(ImageValidationError):
        mask.invert()


def test_bounding_box() -> None:
    data = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    mask = Mask(data)

    assert mask.bounding_box() == (2, 1, 4, 3)


def test_empty_mask_bounding_box() -> None:
    mask = Mask.empty(10, 10)

    assert mask.bounding_box() is None
