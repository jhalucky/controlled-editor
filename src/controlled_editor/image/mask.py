from dataclasses import dataclass

import numpy as np

from controlled_editor.exceptions import ImageValidationError

@dataclass(frozen=True, slots=True)
class Mask:
    data: np.ndarray

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise ImageValidationError("mask data must not be a Numpy array.")

        if self.data.ndim != 2:
            raise ImageValidationError("mask must be a 2D array.")

        if self.data.dtype == np.uint8:
            if not np.all((self.data == 0) | (self.data == 1)):
                raise ImageValidationError(
                    "Binary mask values must be 0 or 1."
                )

        elif self.data.dtype == np.float32:
            if not np.all(np.isfinite(self.data)):
                raise ImageValidationError("Soft mask contains non-finite values.")

            if np.any(self.data < 0.0) or np.any(self.data > 1.0):
                raise ImageValidationError("Soft mask values must be between 0.0 and 1.0.")

        else:
            raise ImageValidationError(
                "mask dtype must be uint8 for binary masks or float32 for soft masks."
            )


    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def is_binary(self) -> bool:
        return self.data.dtype == np.uint8

    @property
    def is_soft(self) -> bool:
        return self.data.dtype == np.float32