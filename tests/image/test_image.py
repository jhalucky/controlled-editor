import numpy as np
import pytest

from controlled_editor.exceptions import ImageValidationError
from controlled_editor.image.image import Image


def test_image_creation():
    data = np.zeros((720, 1280, 3), dtype=np.uint8)

    image = Image(data)

    assert image.data is data
    assert image.height == 720
    assert image.width == 1280
    assert image.channels == 3
    assert image.shape == (720, 1280, 3)
    assert image.dtype == np.uint8


def test_invalid_image_cannot_be_created():
    data = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(ImageValidationError):
        Image(data)


def test_image_is_frozen():
    data = np.zeros((720, 1280, 3), dtype=np.uint8)
    image = Image(data)

    with pytest.raises(AttributeError):
        image.data = np.ones_like(data)
