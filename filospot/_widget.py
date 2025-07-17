"""Napari plugin widget interface for filospot."""

from typing import TYPE_CHECKING

import napari
from magicgui.widgets import Container, Label

from .config import DEFAULT_ROOT
from .data_navigator import DataNavigator
from .widgets import (
    create_navigation_widget,
    create_alignment_widget,
    create_filter_widget,
    create_directory_loader,
    create_export_widget,
)

if TYPE_CHECKING:
    import napari


# Global navigator instance for the plugin
_navigator = None


def get_navigator() -> DataNavigator:
    """Get or create the global navigator instance."""
    global _navigator
    if _navigator is None:
        _navigator = DataNavigator()
    return _navigator


def make_filospot_widget():
    """Create a comprehensive filospot widget with all functionality organized in sections."""
    navigator = get_navigator()
    viewer = napari.current_viewer()

    # Create all sub-widgets
    navigation_widget = create_navigation_widget(navigator, viewer)
    alignment_widget = create_alignment_widget(navigator, viewer)
    filter_widget = create_filter_widget(viewer)
    load_widget = create_directory_loader(
        navigator, viewer, navigation_widget, filter_widget, DEFAULT_ROOT
    )
    export_widget = create_export_widget(DEFAULT_ROOT)

    # Create section headers as simple labels with styling
    load_header = Label(value="Import", name="section_header")
    nav_header = Label(value="Navigation", name="section_header")
    align_header = Label(value="Align Reference", name="section_header")
    filter_header = Label(value="Filtering", name="section_header")
    export_header = Label(value="Export", name="section_header")

    # Style the headers to make them stand out
    for header in [load_header, nav_header, align_header, filter_header, export_header]:
        header.native.setStyleSheet(
            "font-weight: bold; color: #F49E17; margin-top: 10px;"
        )

    # Combine all widgets in a logical order
    container = Container(
        widgets=[
            load_header,
            load_widget,
            nav_header,
            navigation_widget,
            align_header,
            alignment_widget,
            filter_header,
            filter_widget,
            export_header,
            export_widget,
        ],
        labels=False,
        layout="vertical",
        scrollable=False,
    )

    # Remove container padding and margins
    container.margins = (0, 0, 0, 0)  # left, top, right, bottom
    container.native.setContentsMargins(0, 0, 0, 0)

    # Optionally reduce spacing between widgets
    if hasattr(container.native.layout(), "setSpacing"):
        container.native.layout().setSpacing(2)  # Reduce space between widgets

    return container
