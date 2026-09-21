class ControlledEditorError(Exception):
    """Base exception for the controlled editor."""

class ImageError(ControlledEditorError):
    """Base exception for image-related errors."""

class ImageLoadError(ImageError):
    """Raised when an image cannot be loaded."""

class ImageValidationError(ImageError):
    """Raised when an image fails validation."""

class MaskError(ControlledEditorError):
    """Raised when a mask is invalid."""

class RegionError(ControlledEditorError):
    """Raised when a region or bounding box is invalid."""

    