"""Data navigation and loading functionality."""

from pathlib import Path
import tifffile
import pandas as pd
import napari
from .config import IMAGE_SCALE


class DataNavigator:
    """Handles navigation through image datasets and corresponding CSV files."""

    def __init__(self):
        self.tif_paths = []
        self.ref_paths = []
        self.csv_paths = []
        self.index = 0
        self.ref_offset = [0, 0, 0]  # Z, Y, X offsets for reference image alignment

    def load_paths(self, dir_tif: Path, dir_ref: Path, dir_csv: Path):
        """Load and validate file paths from the specified directories."""
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
        """Check if there's a next image to navigate to."""
        return self.index < len(self.tif_paths) - 1

    def has_prev(self):
        """Check if there's a previous image to navigate to."""
        return self.index > 0

    def get_current(self):
        """Get the current image, reference, and CSV file paths."""
        ref_path = self.ref_paths[self.index] if self.ref_paths else None
        return self.tif_paths[self.index], ref_path, self.csv_paths[self.index]

    def next(self):
        """Navigate to the next image."""
        if self.has_next():
            self.index += 1
        return self.get_current()

    def prev(self):
        """Navigate to the previous image."""
        if self.has_prev():
            self.index -= 1
        return self.get_current()


def load_image_data(navigator: "DataNavigator", viewer):
    """Load the current image data into the napari viewer."""
    path_tif, path_ref, path_csv = navigator.get_current()
    print(f"Loading image: {path_tif}")
    print(f"Reference path: {path_ref}")
    print(f"CSV path: {path_csv}")

    image = tifffile.imread(path_tif)
    spots = pd.read_csv(path_csv)[["z", "y", "x"]]
    viewer.add_image(
        image,
        name=path_tif.stem + "_raw",
        scale=IMAGE_SCALE,
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
                scale=IMAGE_SCALE,
                colormap="green",
                blending="additive",
            )
            print("Reference image loaded successfully")
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
        scale=IMAGE_SCALE,
    )


def clear_layers(viewer):
    """Clear all layers from the napari viewer."""
    for layer in list(viewer.layers):
        viewer.layers.remove(layer)


def remove_layer_by_pattern(viewer, pattern: str):
    """Remove layer(s) matching the given pattern."""
    layers_to_remove = [layer for layer in viewer.layers if pattern in layer.name]
    for layer in layers_to_remove:
        viewer.layers.remove(layer)
    return len(layers_to_remove) > 0


def find_reference_layer(viewer, prefer_aligned=False):
    """Find the reference layer to use, optionally preferring aligned version."""
    ref_layer = None

    if prefer_aligned:
        # Look for aligned reference layer first
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

    return ref_layer
