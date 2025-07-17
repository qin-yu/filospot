"""Main application entry point for filospot."""

import napari
from .config import DEFAULT_ROOT
from .data_navigator import DataNavigator
from .widgets import (
    create_navigation_widget,
    create_alignment_widget,
    create_filter_widget,
    create_directory_loader,
    create_export_widget,
)


def main():
    """Main application entry point."""
    # Initialize core components
    navigator = DataNavigator()
    viewer = napari.Viewer()

    # Create widgets
    navigation_widget = create_navigation_widget(navigator, viewer)
    alignment_widget = create_alignment_widget(navigator, viewer)
    filter_widget = create_filter_widget(viewer)

    # Create directory loader and export widgets with proper dependencies
    load_directories = create_directory_loader(
        navigator, viewer, navigation_widget, filter_widget, DEFAULT_ROOT
    )
    export_csv = create_export_widget(DEFAULT_ROOT)

    # Add widgets to the viewer
    viewer.window.add_dock_widget(load_directories, area="right", name="Select Folders")
    viewer.window.add_dock_widget(navigation_widget, area="right", name="Navigation")
    viewer.window.add_dock_widget(
        alignment_widget, area="right", name="Reference Alignment"
    )
    viewer.window.add_dock_widget(filter_widget, area="right", name="Filter Spots")
    viewer.window.add_dock_widget(export_csv, area="right", name="Export Spots")

    # Start the napari event loop
    napari.run()


if __name__ == "__main__":
    main()
