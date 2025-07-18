"""Tests for the filospot napari plugin."""

import os
import pytest

# Set up headless environment
os.environ["QT_QPA_PLATFORM"] = "offscreen"

pytest_plugins = ["pytestqt"]


def test_plugin_widgets_available():
    """Test that the main plugin widget is available."""
    from filospot._widget import make_filospot_widget

    # This should not raise any import errors
    assert callable(make_filospot_widget)


def test_plugin_manifest_exists():
    """Test that the napari plugin manifest exists."""
    import filospot
    from pathlib import Path

    manifest_path = Path(filospot.__file__).parent / "napari.yaml"
    assert manifest_path.exists()


def test_data_navigator_import():
    """Test that core components can be imported."""
    from filospot import DataNavigator

    navigator = DataNavigator()
    assert navigator is not None


def test_widget_creation_with_viewer():
    """Test widget creation with a napari viewer."""
    import napari

    viewer = napari.Viewer(show=False)

    from filospot._widget import make_filospot_widget

    # This should create a widget without errors
    widget = make_filospot_widget()
    assert widget is not None

    viewer.close()
