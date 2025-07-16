"""For automating the loading, navigation, alignment, filter and export of image and spot data."""

from pathlib import Path

import napari
import numpy as np
import pandas as pd
import tifffile
from magicgui import magicgui
from magicgui.widgets import (
    Container,
    PushButton,
    ComboBox,
    SpinBox,
    CheckBox,
    FloatSpinBox,
)
from napari.layers import Image, Points
from scipy import ndimage

DEFAULT_ROOT = Path(
    "/g/diz/_Lab_General/Leanne_and_Martin/Filopodia_analysis_mDia1_diff_sizes_V2/"
)
DEFAULT_DIR_TIF = DEFAULT_ROOT / "Filopodia_analysis_qin_mDia_channels/CAezrin/600/"
DEFAULT_DIR_REF = DEFAULT_ROOT / "Filopodia_analysis_qin_actin_channels/CAezrin/600/"
DEFAULT_DIR_CSV = DEFAULT_ROOT / "Filopodia_analysis_qin_results/CAezrin/600/"


class DataNavigator:
    def __init__(self):
        self.tif_paths = []
        self.ref_paths = []
        self.csv_paths = []
        self.index = 0
        self.ref_offset = [0, 0, 0]  # Z, Y, X offsets for reference image alignment

    def load_paths(self, dir_tif: Path, dir_ref: Path, dir_csv: Path):
        print("Loading paths from:")
        print(f"  TIF directory: {dir_tif}")
        print(f"  REF directory: {dir_ref}")
        print(f"  CSV directory: {dir_csv}")

        tif_list = sorted(dir_tif.glob("*.tif"))
        print(f"Found {len(tif_list)} TIF files")

        # Only look for reference images if reference directory is different from image directory
        if dir_ref != dir_tif:
            ref_list = [dir_ref / tif.name for tif in tif_list]
            print(f"Looking for {len(ref_list)} reference files in different directory")
        else:
            ref_list = []
            print(
                "Reference directory same as TIF directory - skipping reference images"
            )

        csv_list = [dir_csv / tif.with_suffix(".csv").name for tif in tif_list]
        print(f"Looking for {len(csv_list)} CSV files")

        if not tif_list:
            raise FileNotFoundError("No .tif files found.")
        if not all(p.exists() for p in csv_list):
            missing_csv = [p for p in csv_list if not p.exists()]
            print(f"Missing CSV files: {missing_csv}")
            raise FileNotFoundError("Some corresponding .csv files are missing.")

        # Check reference images only if reference directory is different from image directory
        if ref_list:
            existing_ref = [p for p in ref_list if p.exists()]
            missing_ref = [p for p in ref_list if not p.exists()]
            print(f"Found {len(existing_ref)} reference images")
            if missing_ref:
                print(f"Missing reference images: {missing_ref}")
                print(
                    "Warning: Some reference images are missing. Reference images will not be loaded."
                )
                ref_list = []

        self.tif_paths = tif_list
        self.ref_paths = ref_list
        self.csv_paths = csv_list
        self.index = 0
        print(
            f"Final: {len(self.tif_paths)} TIF, {len(self.ref_paths)} REF, {len(self.csv_paths)} CSV"
        )

    def has_next(self):
        return self.index < len(self.tif_paths) - 1

    def has_prev(self):
        return self.index > 0

    def get_current(self):
        ref_path = self.ref_paths[self.index] if self.ref_paths else None
        return self.tif_paths[self.index], ref_path, self.csv_paths[self.index]

    def next(self):
        if self.has_next():
            self.index += 1
        return self.get_current()

    def prev(self):
        if self.has_prev():
            self.index -= 1
        return self.get_current()


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


navigator = DataNavigator()
viewer = napari.Viewer()


def clear_layers():
    for layer in list(viewer.layers):
        viewer.layers.remove(layer)


def load_indexed_pair():
    path_tif, path_ref, path_csv = navigator.get_current()
    print(f"Loading image: {path_tif}")
    print(f"Reference path: {path_ref}")
    print(f"CSV path: {path_csv}")

    image = tifffile.imread(path_tif)
    spots = pd.read_csv(path_csv)[["z", "y", "x"]]
    viewer.add_image(
        image,
        name=path_tif.stem + "_raw",
        scale=(0.3000000, 0.0405217, 0.0405217),
    )
    # Load reference image if available
    if path_ref:
        print(f"Reference path exists: {path_ref.exists()}")
        if path_ref.exists():
            print(f"Loading reference image: {path_ref}")
            ref_image = tifffile.imread(path_ref)
            viewer.add_image(
                ref_image,
                name=path_tif.stem + "_ref",
                scale=(0.3000000, 0.0405217, 0.0405217),
                colormap="green",
                blending="additive",
                translate=navigator.ref_offset,  # Apply any existing offset
            )
            print("Reference image loaded successfully")
            if any(offset != 0 for offset in navigator.ref_offset):
                print(
                    f"Applied stored offsets: Z={navigator.ref_offset[0]}, Y={navigator.ref_offset[1]}, X={navigator.ref_offset[2]}"
                )
        else:
            print(f"Reference image file does not exist: {path_ref}")
    else:
        print("No reference path provided")

    viewer.add_points(
        spots,
        name=path_csv.stem,
        face_color="orange",
        border_color="orange",
        size=10,
        symbol="ring",
        ndim=3,
        scale=(0.3000000, 0.0405217, 0.0405217),
    )

    # Automatically set up filter parameters when new images are loaded
    if path_ref and path_ref.exists():
        update_threshold_range()
        update_reference_contrast()


@magicgui(
    call_button="Load Directories",
    dir_tif={"label": "Raw spots", "mode": "d"},
    dir_ref={"label": "References", "mode": "d"},
    dir_csv={"label": "Detections", "mode": "d"},
)
def load_directories(
    dir_tif: Path = DEFAULT_DIR_TIF,
    dir_ref: Path = DEFAULT_DIR_REF,
    dir_csv: Path = DEFAULT_DIR_CSV,
):
    try:
        navigator.load_paths(dir_tif, dir_ref, dir_csv)
        clear_layers()
        load_indexed_pair()
        update_image_choices()  # Update the dropdown choices after loading
        print(f"Loaded {len(navigator.tif_paths)} images.")
        if navigator.ref_paths:
            print(f"Loaded {len(navigator.ref_paths)} reference images.")
        print(f"Current image: {navigator.index + 1}/{len(navigator.tif_paths)}")
    except Exception as e:
        print(f"Error: {e}")


# Navigation functions (without decorators - will be used in combined widget)
def load_next():
    if navigator.has_next():
        navigator.next()
        clear_layers()
        load_indexed_pair()
        update_image_choices()  # Update dropdown selection
        print(f"Image {navigator.index + 1}/{len(navigator.tif_paths)}")
    else:
        print("Already at last image.")


def load_previous():
    if navigator.has_prev():
        navigator.prev()
        clear_layers()
        load_indexed_pair()
        update_image_choices()  # Update dropdown selection
        print(f"Image {navigator.index + 1}/{len(navigator.tif_paths)}")
    else:
        print("Already at first image.")


def go_to_image_func(image_selection: str):
    if not navigator.tif_paths:
        print("No images loaded. Please load directories first.")
        return

    if not image_selection:
        print("No image selected in dropdown.")
        return

    try:
        # Find the index of the selected image
        selected_index = next(
            i
            for i, path in enumerate(navigator.tif_paths)
            if path.stem == image_selection
        )
        navigator.index = selected_index
        clear_layers()
        load_indexed_pair()
        update_image_choices()  # Update dropdown selection to sync with current image
        print(
            f"Jumped to image {selected_index + 1}/{len(navigator.tif_paths)}: {image_selection}"
        )
    except StopIteration:
        print(f"Image '{image_selection}' not found in loaded images.")


# Combined Navigation Widget
def create_navigation_widget():
    """Create a combined navigation widget with Previous, Next, and Go to Image controls"""

    # Create individual widgets
    previous_btn = PushButton(text="Previous")
    next_btn = PushButton(text="Next")

    image_combo = ComboBox(choices=[], label="Images:")

    # Create padded go button container to center and control width
    # Create a wider button text to take up more space naturally
    go_btn = PushButton(text="   Go to Image   ")  # Add padding to button text

    # Connect button callbacks
    previous_btn.clicked.connect(load_previous)
    next_btn.clicked.connect(load_next)
    go_btn.clicked.connect(lambda: go_to_image_func(image_combo.value))

    # Create navigation buttons container (horizontal layout)
    nav_buttons_container = Container(
        widgets=[previous_btn, next_btn], layout="horizontal"
    )

    # The button will naturally take more space than empty containers
    go_btn_container = Container(widgets=[image_combo, go_btn], layout="vertical")

    # Create main container (vertical layout)
    main_container = Container(
        widgets=[go_btn_container, nav_buttons_container], layout="vertical"
    )

    # Store references for updating dropdown choices
    main_container.image_combo = image_combo

    return main_container


# Create the navigation widget
navigation_widget = create_navigation_widget()


def update_image_choices():
    """Update the choices in the image selection dropdown"""
    if navigator.tif_paths:
        choices = [path.stem for path in navigator.tif_paths]
        navigation_widget.image_combo.choices = choices

        # Always set current selection to the currently loaded image
        if 0 <= navigator.index < len(choices):
            current_image = choices[navigator.index]
            navigation_widget.image_combo.value = current_image
            print(
                f"Updated dropdown with {len(choices)} image choices, current: {current_image}"
            )
        else:
            # Fallback: if index is out of bounds, show the first image
            if choices:
                navigation_widget.image_combo.value = choices[0]
                print(
                    f"Updated dropdown with {len(choices)} image choices, showing first image as fallback"
                )
    else:
        navigation_widget.image_combo.choices = []
        navigation_widget.image_combo.value = ""
        print("Cleared dropdown choices - no images loaded")


# Alignment functions (without decorators - will be used in combined widget)
def align_reference_func(offset_z: int, offset_y: int, offset_x: int):
    """Apply Z, Y, X pixel offsets to align the reference image with the main image"""
    if not navigator.ref_paths:
        print("No reference images loaded.")
        return

    # Update the stored offsets
    navigator.ref_offset = [offset_z, offset_y, offset_x]

    # Check if there's already an aligned reference layer and remove it
    aligned_layer = None
    for layer in viewer.layers:
        if "_ref_aligned" in layer.name:
            aligned_layer = layer
            break

    if aligned_layer:
        viewer.layers.remove(aligned_layer)

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
            scale=(0.3000000, 0.0405217, 0.0405217),
            colormap="magenta",  # Use different color to distinguish from original
            blending="additive",
        )
        print(f"Created aligned reference layer: {aligned_name}")
        print(f"Applied pixel offsets: Z={offset_z}, Y={offset_y}, X={offset_x}")
        print("Original reference layer preserved for comparison")
        print("Translation applied at pixel level with zero-padding")
    else:
        print("Reference image not found")


def reset_alignment_func():
    """Reset reference image alignment to zero offsets and remove aligned layer"""
    # Remove any existing aligned layer
    aligned_layer = None
    for layer in viewer.layers:
        if "_ref_aligned" in layer.name:
            aligned_layer = layer
            break

    if aligned_layer:
        viewer.layers.remove(aligned_layer)
        print("Removed aligned reference layer")

    # Reset the spinbox values in the alignment widget
    alignment_widget.offset_z_spin.value = 0
    alignment_widget.offset_y_spin.value = 0
    alignment_widget.offset_x_spin.value = 0

    # Reset stored offsets
    navigator.ref_offset = [0, 0, 0]
    print("Reset alignment offsets to zero")


# Combined Alignment Widget
def create_alignment_widget():
    """Create a combined alignment widget with offset controls and buttons"""

    # Create offset controls
    offset_z_spin = SpinBox(value=0, min=-50, max=50, step=1, label="Z")
    offset_y_spin = SpinBox(value=0, min=-50, max=50, step=1, label="Y")
    offset_x_spin = SpinBox(value=0, min=-50, max=50, step=1, label="X")

    # Create buttons
    apply_btn = PushButton(text="Apply Offset")
    reset_btn = PushButton(text="Reset")

    # Connect button callbacks
    apply_btn.clicked.connect(
        lambda: align_reference_func(
            offset_z_spin.value, offset_y_spin.value, offset_x_spin.value
        )
    )
    reset_btn.clicked.connect(reset_alignment_func)

    # Create offset containers (each row with label and spinbox)
    zyx_container = Container(
        widgets=[offset_z_spin, offset_y_spin, offset_x_spin], layout="horizontal"
    )

    # Create button container
    button_container = Container(widgets=[apply_btn, reset_btn], layout="horizontal")

    # Create main container (vertical layout)
    main_container = Container(
        widgets=[zyx_container, button_container], layout="vertical"
    )

    # Store references for accessing spinbox values
    main_container.offset_z_spin = offset_z_spin
    main_container.offset_y_spin = offset_y_spin
    main_container.offset_x_spin = offset_x_spin

    return main_container


# Create the alignment widget
alignment_widget = create_alignment_widget()


def update_threshold_range():
    """Update the threshold spinbox range based on current reference image"""
    # Find the reference image to use
    ref_layer = None
    if hasattr(filter_widget, "aligned_check") and filter_widget.aligned_check.value:
        # Look for aligned reference layer
        for layer in viewer.layers:
            if "_ref_aligned" in layer.name:
                ref_layer = layer
                break

    if not ref_layer:
        # Look for original reference layer
        for layer in viewer.layers:
            if "_ref" in layer.name and "_aligned" not in layer.name:
                ref_layer = layer
                break

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


def update_reference_contrast():
    """Update reference layer contrast limits based on current threshold"""
    if not (
        hasattr(filter_widget, "contrast_check") and filter_widget.contrast_check.value
    ):
        return

    threshold = filter_widget.threshold_spin.value

    # Find the reference layer being used
    ref_layer = None
    if hasattr(filter_widget, "aligned_check") and filter_widget.aligned_check.value:
        for layer in viewer.layers:
            if "_ref_aligned" in layer.name:
                ref_layer = layer
                break

    if not ref_layer:
        for layer in viewer.layers:
            if "_ref" in layer.name and "_aligned" not in layer.name:
                ref_layer = layer
                break

    if ref_layer and isinstance(ref_layer, Image):
        # Set contrast limits to highlight the threshold
        try:
            # Get the actual numpy array from the layer
            ref_image = np.asarray(ref_layer.data)
            min_val = float(np.min(ref_image))
            max_val = float(np.max(ref_image))
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


# Filter functions (without decorators - will be used in combined widget)
def filter_spots_by_reference_func(
    intensity_threshold: int,
    gaussian_sigma: float,
    use_blur: bool,
    use_aligned: bool,
):
    """Filter spots based on intensity in reference image, optionally with Gaussian blur"""

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
    ref_layer = None
    if use_aligned:
        # Look for aligned reference layer
        for layer in viewer.layers:
            if "_ref_aligned" in layer.name:
                ref_layer = layer
                break
        if not ref_layer:
            print("No aligned reference layer found. Using original reference.")

    if not ref_layer:
        # Look for original reference layer
        for layer in viewer.layers:
            if "_ref" in layer.name and "_aligned" not in layer.name:
                ref_layer = layer
                break

    if not ref_layer:
        print("No reference image layer found")
        return

    print(f"Using reference layer: {ref_layer.name}")

    # Get reference image data
    ref_image = ref_layer.data

    # Apply Gaussian blur if requested
    if use_blur and gaussian_sigma > 0:
        print(f"Applying Gaussian blur with sigma={gaussian_sigma}")
        if ref_image.ndim == 3:
            ref_image = ndimage.gaussian_filter(ref_image, sigma=gaussian_sigma)
        else:
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
        for layer in list(viewer.layers):
            if layer.name == removed_layer_name:
                viewer.layers.remove(layer)
                break

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


# Combined Filter Widget
def create_filter_widget():
    """Create a combined filter widget with all filtering controls"""

    # Create parameter controls
    threshold_spin = SpinBox(
        value=65535, min=0, max=65535, step=25, label="Intensity Threshold"
    )
    sigma_spin = FloatSpinBox(value=1.0, min=0.0, max=5.0, step=0.1, label="Sigma =")

    # Create checkboxes
    blur_check = CheckBox(value=True, text="Apply Gaussian Blur")
    aligned_check = CheckBox(value=False, text="Use Aligned Reference")
    contrast_check = CheckBox(value=True, text="Update Contrast Live")

    # Create action buttons
    filter_btn = PushButton(text="Filter Spots")

    # Connect button callbacks
    filter_btn.clicked.connect(
        lambda: filter_spots_by_reference_func(
            threshold_spin.value,
            sigma_spin.value,
            blur_check.value,
            aligned_check.value,
        )
    )

    # Create organized containers
    # Parameters row
    params_container = Container(widgets=[threshold_spin], layout="vertical")

    blur_check_container = Container(
        widgets=[blur_check, sigma_spin], layout="horizontal"
    )

    filter_btn_container = Container(
        widgets=[filter_btn],
        layout="horizontal",
    )

    # Main container (vertical layout)
    main_container = Container(
        widgets=[
            params_container,
            blur_check_container,
            Container(widgets=[aligned_check]),
            Container(widgets=[contrast_check]),
            filter_btn_container,
        ],
        layout="vertical",
    )

    # Store references for accessing values
    main_container.threshold_spin = threshold_spin
    main_container.sigma_spin = sigma_spin
    main_container.blur_check = blur_check
    main_container.aligned_check = aligned_check
    main_container.contrast_check = contrast_check

    return main_container


# Create the filter widget
filter_widget = create_filter_widget()


# Automatically set up live contrast updates when checkbox is enabled
def setup_contrast_live_updates():
    """Set up live contrast updates based on checkbox state"""
    if filter_widget.contrast_check.value:
        try:
            # Connect threshold changes to contrast updates
            filter_widget.threshold_spin.changed.connect(
                lambda: update_reference_contrast()
            )

            # Connect parameter changes to range and contrast updates
            filter_widget.aligned_check.changed.connect(
                lambda: (update_threshold_range(), update_reference_contrast())
            )

            filter_widget.blur_check.changed.connect(
                lambda: (update_threshold_range(), update_reference_contrast())
            )

            filter_widget.sigma_spin.changed.connect(
                lambda: (
                    update_threshold_range()
                    if filter_widget.blur_check.value
                    else None,
                    update_reference_contrast(),
                )
            )

            print("Live contrast updates enabled")
        except Exception as e:
            print(f"Could not connect live update events: {e}")
    else:
        try:
            # Disconnect all the event connections
            filter_widget.threshold_spin.changed.disconnect()
            filter_widget.aligned_check.changed.disconnect()
            filter_widget.blur_check.changed.disconnect()
            filter_widget.sigma_spin.changed.disconnect()

            print("Live contrast updates disabled")
        except Exception as e:
            print(f"Could not disconnect live update events: {e}")


# Connect the checkbox to toggle live updates
filter_widget.contrast_check.changed.connect(setup_contrast_live_updates)

# Enable live updates by default (since checkbox starts as True)
setup_contrast_live_updates()


@magicgui(
    call_button="Export to CSV",
    output_dir={"label": "Output Directory", "mode": "d"},
)
def export_csv(layer: Points, output_dir: Path = DEFAULT_ROOT):
    output_dir.mkdir(parents=True, exist_ok=True)
    path_out = output_dir / f"{layer.name}_exported.csv"
    df = pd.DataFrame(layer.data, columns=["z", "y", "x"])
    df.to_csv(path_out, index=False)
    print(f"Exported to: {path_out}")


viewer.window.add_dock_widget(load_directories, area="right", name="Select Folders")
viewer.window.add_dock_widget(navigation_widget, area="right", name="Navigation")
viewer.window.add_dock_widget(
    alignment_widget, area="right", name="Reference Alignment"
)
viewer.window.add_dock_widget(filter_widget, area="right", name="Filter Spots")
viewer.window.add_dock_widget(export_csv, area="right", name="Export Spots")

napari.run()
