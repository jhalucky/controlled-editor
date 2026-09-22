from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controlled_editor.image.validation import validate_image


@dataclass(frozen=True, slots=True)
class Image:
    data: np.ndarray

    def __post_init__(self) -> None:
        validate_image(self.data)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        return self.data.shape[2]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
