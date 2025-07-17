"""Configuration and constants for the filospot application."""

from pathlib import Path

# Default directories - modify these to match your data structure
DEFAULT_ROOT = Path(
    "/g/diz/_Lab_General/Leanne_and_Martin/Filopodia_analysis_mDia1_diff_sizes_V2/"
)

# Image scale parameters (Z, Y, X) for napari visualization
IMAGE_SCALE = (0.3000000, 0.0405217, 0.0405217)
