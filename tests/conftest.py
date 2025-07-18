"""Configuration for pytest."""

import pytest


@pytest.fixture
def make_napari_viewer():
    """Create a napari viewer for testing."""
    import napari

    def _make_napari_viewer():
        return napari.Viewer(show=False)

    return _make_napari_viewer
