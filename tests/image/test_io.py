import numpy as np
import pytest
from PIL import Image

from controlled_editor.exceptions import ImageLoadError
from controlled_editor.image.io import load_image, save_image


def test_load_rgb_image(tmp_path):
    source = tmp_path / "source.png"

    expected = np.zeros((20, 30, 3), dtype=np.uint8)
    expected[:, :, 0] = 255

    Image.fromarray(expected, mode="RGB").save(source)

    result = load_image(source)

    assert result.shape == (20, 30, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(result, expected)


def test_load_rgba_image_converts_to_rgb(tmp_path):
    source = tmp_path / "source.png"

    rgba = np.zeros((10, 15, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255
    rgba[:, :, 3] = 128

    Image.fromarray(rgba, mode="RGBA").save(source)

    result = load_image(source)

    assert result.shape == (10, 15, 3)
    assert result.dtype == np.uint8
    assert np.all(result[:, :, 0] == 255)


def test_load_grayscale_image_converts_to_rgb(tmp_path):
    source = tmp_path / "source.png"

    grayscale = np.full((10, 15), 128, dtype=np.uint8)

    Image.fromarray(grayscale, mode="L").save(source)

    result = load_image(source)

    assert result.shape == (10, 15, 3)
    assert result.dtype == np.uint8
    assert np.all(result[:, :, 0] == 128)
    assert np.all(result[:, :, 1] == 128)
    assert np.all(result[:, :, 2] == 128)


def test_load_missing_file_raises(tmp_path):
    source = tmp_path / "missing.png"

    with pytest.raises(ImageLoadError):
        load_image(source)


def test_load_unsupported_extension_raises(tmp_path):
    source = tmp_path / "image.txt"
    source.write_text("not an image")

    with pytest.raises(ImageLoadError):
        load_image(source)


def test_save_and_reload_image(tmp_path):
    destination = tmp_path / "output.png"

    original = np.zeros((25, 35, 3), dtype=np.uint8)
    original[:, :, 1] = 200

    save_image(original, destination)

    loaded = load_image(destination)

    assert np.array_equal(loaded, original)
