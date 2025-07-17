"""Image alignment utilities."""

import numpy as np
import tifffile
from .data_navigator import remove_layer_by_pattern
from .config import IMAGE_SCALE


def apply_pixel_translation(image_array, offset_z, offset_y, offset_x):
    """
    Apply pixel-level translation to an image array with zero-padding.

    Parameters:
    - image_array: numpy array of shape (Z, Y, X) or (Y, X)
    - offset_z, offset_y, offset_x: translation offsets in pixels

    Returns:
    - Translated image array with same shape, zero-padded where needed
    """
    # Handle both 2D and 3D images
    if image_array.ndim == 2:
        # 2D image: only apply Y, X translation
        translated = np.zeros_like(image_array)

        # Calculate slice indices for source and destination
        src_y_start = max(0, -offset_y)
        src_y_end = min(image_array.shape[0], image_array.shape[0] - offset_y)
        src_x_start = max(0, -offset_x)
        src_x_end = min(image_array.shape[1], image_array.shape[1] - offset_x)

        dst_y_start = max(0, offset_y)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, offset_x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        # Copy the translated region
        translated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image_array[
            src_y_start:src_y_end, src_x_start:src_x_end
        ]

    else:
        # 3D image: apply Z, Y, X translation
        translated = np.zeros_like(image_array)

        # Calculate slice indices for source and destination
        src_z_start = max(0, -offset_z)
        src_z_end = min(image_array.shape[0], image_array.shape[0] - offset_z)
        src_y_start = max(0, -offset_y)
        src_y_end = min(image_array.shape[1], image_array.shape[1] - offset_y)
        src_x_start = max(0, -offset_x)
        src_x_end = min(image_array.shape[2], image_array.shape[2] - offset_x)

        dst_z_start = max(0, offset_z)
        dst_z_end = dst_z_start + (src_z_end - src_z_start)
        dst_y_start = max(0, offset_y)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, offset_x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        # Copy the translated region
        translated[
            dst_z_start:dst_z_end, dst_y_start:dst_y_end, dst_x_start:dst_x_end
        ] = image_array[
            src_z_start:src_z_end, src_y_start:src_y_end, src_x_start:src_x_end
        ]

    return translated


def align_reference_image(
    navigator, viewer, offset_z: int, offset_y: int, offset_x: int
):
    """Apply Z, Y, X pixel offsets to align the reference image with the main image."""
    if not navigator.ref_paths:
        print("No reference images loaded.")
        return

    # Update the stored offsets
    navigator.ref_offset = [offset_z, offset_y, offset_x]

    # Remove any existing aligned reference layer
    remove_layer_by_pattern(viewer, "_ref_aligned")

    # Create new aligned reference layer (keeping original reference layer)
    path_tif, path_ref, path_csv = navigator.get_current()
    if path_ref and path_ref.exists():
        ref_image = tifffile.imread(path_ref)

        # Apply pixel-level translation with zero-padding
        translated_image = apply_pixel_translation(
            ref_image, offset_z, offset_y, offset_x
        )

        # Create aligned layer with clear naming
        aligned_name = (
            f"{path_tif.stem}_ref_aligned_Z{offset_z}_Y{offset_y}_X{offset_x}"
        )
        viewer.add_image(
            translated_image,
            name=aligned_name,
            scale=IMAGE_SCALE,
            colormap="magenta",  # Use different color to distinguish from original
            blending="additive",
        )
        print(f"Created aligned reference layer: {aligned_name}")
        print(f"Applied pixel offsets: Z={offset_z}, Y={offset_y}, X={offset_x}")
        print("Original reference layer preserved for comparison")
        print("Translation applied at pixel level with zero-padding")
    else:
        print("Reference image not found")


def reset_reference_alignment(navigator, viewer):
    """Reset reference image alignment to zero offsets and remove aligned layer."""
    # Remove any existing aligned layer
    remove_layer_by_pattern(viewer, "_ref_aligned")
    print("Removed aligned reference layer")

    # Reset alignment offsets
    navigator.ref_offset = [0, 0, 0]
    print("Reset alignment offsets to zero")
