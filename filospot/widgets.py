"""GUI widgets for the filospot application."""

from pathlib import Path
import pandas as pd
import napari
from napari.layers import Points
from magicgui import magicgui
from magicgui.widgets import (
    Container,
    PushButton,
    ComboBox,
    SpinBox,
    CheckBox,
    FloatSpinBox,
)

from .data_navigator import load_image_data, clear_layers
from .alignment import align_reference_image, reset_reference_alignment
from .filtering import (
    filter_spots_by_reference,
    update_threshold_range,
    update_reference_contrast,
)


def create_navigation_widget(navigator, viewer: napari.Viewer):
    """Create a combined navigation widget with Previous, Next, and Go to Image controls."""

    # Create individual widgets
    previous_btn = PushButton(text="Previous")
    next_btn = PushButton(text="Next")
    image_combo = ComboBox(choices=[], label="Images:")
    go_btn = PushButton(text="   Go to Image   ")

    # Navigation functions
    def load_next():
        if navigator.has_next():
            navigator.next()
            clear_layers(viewer)
            load_image_data(navigator, viewer)
            update_image_selection()
            print(f"Image {navigator.index + 1}/{len(navigator.tif_paths)}")
        else:
            print("Already at last image.")

    def load_previous():
        if navigator.has_prev():
            navigator.prev()
            clear_layers(viewer)
            load_image_data(navigator, viewer)
            update_image_selection()
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
            clear_layers(viewer)
            load_image_data(navigator, viewer)
            update_image_selection()
            print(
                f"Jumped to image {selected_index + 1}/{len(navigator.tif_paths)}: {image_selection}"
            )
        except StopIteration:
            print(f"Image '{image_selection}' not found in loaded images.")

    def update_image_selection():
        """Update only the selection in the dropdown - called after successful navigation."""
        if navigator.tif_paths:
            # Ensure choices are populated if they're empty (defensive programming)
            current_choices = getattr(image_combo, "choices", [])
            if not current_choices:
                print("Warning: Combobox choices are empty, repopulating...")
                choices = [path.stem for path in navigator.tif_paths]
                image_combo.choices = choices
                print(f"Repopulated dropdown with {len(choices)} choices")

            if 0 <= navigator.index < len(navigator.tif_paths):
                current_image = navigator.tif_paths[navigator.index].stem
                image_combo.value = current_image
                print(f"Updated dropdown selection to: {current_image}")
            else:
                print(f"Warning: Navigation index {navigator.index} is out of bounds")

    # Connect button callbacks
    previous_btn.clicked.connect(load_previous)
    next_btn.clicked.connect(load_next)
    go_btn.clicked.connect(lambda: go_to_image_func(image_combo.value))

    # Create containers
    nav_buttons_container = Container(
        widgets=[previous_btn, next_btn], layout="horizontal"
    )
    go_btn_container = Container(widgets=[image_combo, go_btn], layout="vertical")
    main_container = Container(
        widgets=[go_btn_container, nav_buttons_container], layout="vertical"
    )

    # Store references
    main_container.image_combo = image_combo
    main_container.update_image_selection = update_image_selection

    return main_container


def create_alignment_widget(navigator, viewer: napari.Viewer):
    """Create a combined alignment widget with offset controls and buttons."""

    # Create offset controls
    offset_z_spin = SpinBox(value=0, min=-50, max=50, step=1, label="Z")
    offset_y_spin = SpinBox(value=0, min=-50, max=50, step=1, label="Y")
    offset_x_spin = SpinBox(value=0, min=-50, max=50, step=1, label="X")

    # Create buttons
    apply_btn = PushButton(text="Apply Offset")
    reset_btn = PushButton(text="Reset")

    # Connect button callbacks
    apply_btn.clicked.connect(
        lambda: align_reference_image(
            navigator,
            viewer,
            offset_z_spin.value,
            offset_y_spin.value,
            offset_x_spin.value,
        )
    )
    reset_btn.clicked.connect(lambda: reset_reference_alignment(navigator, viewer))

    # Create containers
    zyx_container = Container(
        widgets=[offset_z_spin, offset_y_spin, offset_x_spin], layout="horizontal"
    )
    button_container = Container(widgets=[apply_btn, reset_btn], layout="horizontal")
    main_container = Container(
        widgets=[zyx_container, button_container], layout="vertical"
    )

    # Store references
    main_container.offset_z_spin = offset_z_spin
    main_container.offset_y_spin = offset_y_spin
    main_container.offset_x_spin = offset_x_spin

    return main_container


def create_filter_widget(viewer: napari.Viewer):
    """Create a combined filter widget with all filtering controls."""

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
        lambda: filter_spots_by_reference(
            viewer,
            threshold_spin.value,
            sigma_spin.value,
            blur_check.value,
            aligned_check.value,
        )
    )

    # Create containers
    params_container = Container(widgets=[threshold_spin], layout="vertical")
    blur_check_container = Container(
        widgets=[blur_check, sigma_spin], layout="horizontal"
    )
    filter_btn_container = Container(widgets=[filter_btn], layout="horizontal")

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

    # Store references
    main_container.threshold_spin = threshold_spin
    main_container.sigma_spin = sigma_spin
    main_container.blur_check = blur_check
    main_container.aligned_check = aligned_check
    main_container.contrast_check = contrast_check

    # Setup live contrast updates
    def setup_contrast_live_updates():
        """Set up live contrast updates based on checkbox state."""
        if contrast_check.value:
            try:
                # Connect threshold changes to contrast updates
                threshold_spin.changed.connect(
                    lambda: update_reference_contrast(
                        viewer, main_container, aligned_check.value
                    )
                )

                # Connect parameter changes to range and contrast updates
                aligned_check.changed.connect(
                    lambda: (
                        update_threshold_range(
                            viewer, main_container, aligned_check.value
                        ),
                        update_reference_contrast(
                            viewer, main_container, aligned_check.value
                        ),
                    )
                )

                blur_check.changed.connect(
                    lambda: (
                        update_threshold_range(
                            viewer, main_container, aligned_check.value
                        ),
                        update_reference_contrast(
                            viewer, main_container, aligned_check.value
                        ),
                    )
                )

                sigma_spin.changed.connect(
                    lambda: (
                        update_threshold_range(
                            viewer, main_container, aligned_check.value
                        )
                        if blur_check.value
                        else None,
                        update_reference_contrast(
                            viewer, main_container, aligned_check.value
                        ),
                    )
                )

                print("Live contrast updates enabled")
            except Exception as e:
                print(f"Could not connect live update events: {e}")
        else:
            try:
                # Disconnect all the event connections
                threshold_spin.changed.disconnect()
                aligned_check.changed.disconnect()
                blur_check.changed.disconnect()
                sigma_spin.changed.disconnect()

                print("Live contrast updates disabled")
            except Exception as e:
                print(f"Could not disconnect live update events: {e}")

    # Connect the checkbox to toggle live updates
    contrast_check.changed.connect(setup_contrast_live_updates)

    # Enable live updates by default (since checkbox starts as True)
    setup_contrast_live_updates()

    return main_container


def create_directory_loader(
    navigator,
    viewer: napari.Viewer,
    navigation_widget,
    filter_widget,
    default_root: Path,
):
    """Create the directory loader widget."""

    DEFAULT_DIR_TIF = default_root / "Filopodia_analysis_qin_mDia_channels/CAezrin/600/"
    DEFAULT_DIR_REF = (
        default_root / "Filopodia_analysis_qin_actin_channels/CAezrin/600/"
    )
    DEFAULT_DIR_CSV = default_root / "Filopodia_analysis_qin_results/CAezrin/600/"

    def update_image_choices():
        """Update the choices in the image selection dropdown - only called when directories are loaded."""
        if navigator.tif_paths:
            choices = [path.stem for path in navigator.tif_paths]
            navigation_widget.image_combo.choices = choices

            # Set current selection to the currently loaded image
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

    def setup_combobox_protection():
        """Set up global protection for the combobox against napari layer events."""

        def on_layer_change(*args, **kwargs):
            """Event handler to protect combobox when layers change."""
            if navigator.tif_paths and hasattr(navigation_widget, "image_combo"):
                # Check if combobox was cleared and restore it
                current_choices = getattr(navigation_widget.image_combo, "choices", [])
                if not current_choices:
                    choices = [path.stem for path in navigator.tif_paths]
                    navigation_widget.image_combo.choices = choices

                    # Restore current selection
                    if 0 <= navigator.index < len(navigator.tif_paths):
                        current_image = navigator.tif_paths[navigator.index].stem
                        navigation_widget.image_combo.value = current_image
                        print(
                            f"Auto-restored combobox after layer change: {len(choices)} choices, current: {current_image}"
                        )

        # Connect to napari layer events
        try:
            # Connect to various layer list events that might clear the combobox
            viewer.layers.events.inserted.connect(on_layer_change)
            viewer.layers.events.removed.connect(on_layer_change)
            viewer.layers.events.reordered.connect(on_layer_change)
            print("Combobox protection events connected successfully")
        except Exception as e:
            print(f"Warning: Could not connect all combobox protection events: {e}")

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
            clear_layers(viewer)
            load_image_data(navigator, viewer)
            update_image_choices()
            print(f"Loaded {len(navigator.tif_paths)} images.")
            if navigator.ref_paths:
                print(f"Loaded {len(navigator.ref_paths)} reference images.")
            print(f"Current image: {navigator.index + 1}/{len(navigator.tif_paths)}")
            setup_combobox_protection()

            # Automatically set up filter parameters when new images are loaded
            if (
                navigator.ref_paths
                and navigator.get_current()[1]
                and navigator.get_current()[1].exists()
            ):
                update_threshold_range(viewer, filter_widget, False)
                update_reference_contrast(viewer, filter_widget, False)
        except Exception as e:
            print(f"Error: {e}")

    return load_directories


def create_export_widget(default_root: Path):
    """Create the export widget."""

    @magicgui(
        call_button="Export to CSV",
        output_dir={"label": "Output Directory", "mode": "d"},
    )
    def export_csv(layer: Points, output_dir: Path = default_root):
        output_dir.mkdir(parents=True, exist_ok=True)
        path_out = output_dir / f"{layer.name}_exported.csv"
        df = pd.DataFrame(layer.data, columns=["z", "y", "x"])
        df.to_csv(path_out, index=False)
        print(f"Exported to: {path_out}")

    return export_csv
