"""Filospot: For automating the loading, navigation, alignment, filter and export of image and spot data."""

__version__ = "0.1.0"

from .app import main
from .data_navigator import DataNavigator
from .alignment import (
    apply_pixel_translation,
    align_reference_image,
    reset_reference_alignment,
)
from .filtering import filter_spots_by_reference
from .config import DEFAULT_ROOT, IMAGE_SCALE

__all__ = [
    "main",
    "DataNavigator",
    "apply_pixel_translation",
    "align_reference_image",
    "reset_reference_alignment",
    "filter_spots_by_reference",
    "DEFAULT_ROOT",
    "IMAGE_SCALE",
]
