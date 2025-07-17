"""Spot filtering functionality."""

import numpy as np
import napari
from napari.layers import Image, Points
from scipy import ndimage
from .data_navigator import find_reference_layer, remove_layer_by_pattern


def filter_spots_by_reference(
    viewer: napari.Viewer,
    intensity_threshold: int,
    gaussian_sigma: float,
    use_blur: bool,
    use_aligned: bool,
):
    """Filter spots based on intensity in reference image, optionally with Gaussian blur."""

    # Find the points layer (original spots)
    points_layer = None
    for layer in viewer.layers:
        if (
            isinstance(layer, Points)
            and "_exported" not in layer.name
            and "removed" not in layer.name
        ):
            points_layer = layer
            break

    if not points_layer:
        print("No points layer found to filter")
        return

    # Find the reference image to use
    ref_layer = find_reference_layer(viewer, prefer_aligned=use_aligned)

    if not ref_layer:
        print("No reference image layer found")
        return

    print(f"Using reference layer: {ref_layer.name}")

    # Get reference image data
    ref_image = ref_layer.data

    # Apply Gaussian blur if requested
    if use_blur and gaussian_sigma > 0:
        print(f"Applying Gaussian blur with sigma={gaussian_sigma}")
        ref_image = ndimage.gaussian_filter(ref_image, sigma=gaussian_sigma)

    # Get points data
    points_data = points_layer.data.copy()
    points_to_keep = []
    points_to_remove = []

    print(f"Filtering {len(points_data)} points with threshold {intensity_threshold}")

    # Check each point's intensity in the reference image
    for point in points_data:
        # Convert point coordinates to integer indices
        if ref_image.ndim == 3:
            z, y, x = int(round(point[0])), int(round(point[1])), int(round(point[2]))
            # Check bounds
            if (
                0 <= z < ref_image.shape[0]
                and 0 <= y < ref_image.shape[1]
                and 0 <= x < ref_image.shape[2]
            ):
                intensity = ref_image[z, y, x]
            else:
                intensity = 0  # Out of bounds, consider as low intensity
        else:
            y, x = int(round(point[0])), int(round(point[1]))
            # Check bounds
            if 0 <= y < ref_image.shape[0] and 0 <= x < ref_image.shape[1]:
                intensity = ref_image[y, x]
            else:
                intensity = 0  # Out of bounds, consider as low intensity

        # Filter based on intensity threshold
        if intensity >= intensity_threshold:
            points_to_remove.append(point)
        else:
            points_to_keep.append(point)

    print(
        f"Keeping {len(points_to_keep)} points, removing {len(points_to_remove)} points"
    )

    # Update the original points layer with filtered points
    if points_to_keep:
        points_layer.data = np.array(points_to_keep)
    else:
        points_layer.data = np.empty((0, points_data.shape[1]))

    # Create a new layer for removed points if any were removed
    if points_to_remove:
        removed_layer_name = f"{points_layer.name}_removed_thresh{intensity_threshold}"
        # Remove existing removed layer with same name
        remove_layer_by_pattern(viewer, removed_layer_name)

        viewer.add_points(
            np.array(points_to_remove),
            name=removed_layer_name,
            face_color="red",
            border_color="red",
            size=10,  # Use fixed size for removed points
            symbol="cross",
            ndim=points_layer.ndim,
            scale=points_layer.scale,
        )
        print(f"Created removed points layer: {removed_layer_name}")

    print("Spot filtering completed")


def update_threshold_range(viewer: napari.Viewer, filter_widget, use_aligned: bool):
    """Update the threshold spinbox range based on current reference image."""
    # Find the reference image to use
    ref_layer = find_reference_layer(viewer, prefer_aligned=use_aligned)

    if ref_layer:
        ref_image = ref_layer.data

        # Apply blur if requested to get the actual image that will be used
        if (
            hasattr(filter_widget, "blur_check")
            and hasattr(filter_widget, "sigma_spin")
            and filter_widget.blur_check.value
            and filter_widget.sigma_spin.value > 0
        ):
            ref_image = ndimage.gaussian_filter(
                ref_image, sigma=filter_widget.sigma_spin.value
            )

        # Calculate min and max values
        min_val = int(np.min(ref_image))
        max_val = int(np.max(ref_image))

        # Update the SpinBox range - this works for SpinBox widgets
        filter_widget.threshold_spin.min = min_val
        filter_widget.threshold_spin.max = max_val

        # Set a reasonable default (e.g., 75th percentile)
        default_val = int(np.percentile(ref_image, 75))

        # Update the threshold value if it's still at default or out of new range
        current_threshold = filter_widget.threshold_spin.value
        if (
            current_threshold == 1000
            or current_threshold < min_val
            or current_threshold > max_val
        ):
            filter_widget.threshold_spin.value = default_val

        print(f"Image intensity range: {min_val} - {max_val}")
        print(f"Updated threshold range to: {min_val} - {max_val}")
        print(f"Set threshold to: {filter_widget.threshold_spin.value}")
    else:
        print("No reference layer found for threshold range update")


def update_reference_contrast(viewer: napari.Viewer, filter_widget, use_aligned: bool):
    """Update reference layer contrast limits based on current threshold."""
    if not (
        hasattr(filter_widget, "contrast_check") and filter_widget.contrast_check.value
    ):
        return

    threshold = filter_widget.threshold_spin.value

    # Find the reference layer being used
    ref_layer = find_reference_layer(viewer, prefer_aligned=use_aligned)

    if ref_layer and isinstance(ref_layer, Image):
        # Set contrast limits to highlight the threshold
        try:
            # Get the actual numpy array from the layer
            ref_image = np.asarray(ref_layer.data)
            min_val = float(np.min(ref_image))
            threshold_val = float(threshold)

            # Set contrast limits to emphasize the threshold
            # Lower limit: minimum value of the image
            # Upper limit: threshold value to highlight what will be filtered
            ref_layer.contrast_limits = [min_val, threshold_val]
            print(f"Updated contrast limits: [{min_val}, {threshold_val}]")
            print(
                f"Threshold visualization: values >= {threshold_val} will be filtered"
            )
        except Exception as e:
            print(f"Could not update contrast limits: {e}")
            print(f"Threshold visualization: values >= {threshold} will be filtered")
    elif ref_layer:
        print(f"Reference layer found but not an Image layer: {type(ref_layer)}")
        print(f"Threshold visualization: values >= {threshold} will be filtered")
