# Filospot

A [napari] plugin for automating the loading, navigation, alignment, filter and export of image and spot data.

## Installation

You can install `filospot` via [pip]:

    pip install filospot

To install latest development version:

    pip install git+https://github.com/qin-yu/filospot.git

Or install from source:

    git clone https://github.com/qin-yu/filospot.git
    cd filospot
    pip install -e .

## Usage

### As a napari plugin

After installation, the plugin will be available in napari. You can access the filospot widget through:
- `Plugins > filospot > Filospot - Image and Spot Analysis`

The plugin provides a unified interface with sections for:
- **Import**: Load directories containing raw images, reference images, and CSV spot detection files
- **Navigation**: Browse through image datasets with Previous/Next buttons and dropdown selection
- **Align Reference**: Apply pixel-level translation to align reference images with main images
- **Filtering**: Filter detected spots based on intensity thresholds in reference images
- **Export**: Export filtered spots to CSV format

### As a standalone application

```bash
filospot
```

This launches the full napari application with filospot widgets pre-loaded.

### Programmatically

```python
from filospot import main
main()
```

Or use individual components:

```python
from filospot import DataNavigator, filter_spots_by_reference
from filospot.alignment import apply_pixel_translation

# Create a data navigator
navigator = DataNavigator()
navigator.load_paths(tif_dir, ref_dir, csv_dir)

# Apply image alignment
aligned_image = apply_pixel_translation(image, 0, 5, -2)
```

## Features

- **Multi-format support**: Load TIFF images and CSV spot detection files
- **Reference alignment**: Pixel-level translation alignment with visual feedback
- **Advanced filtering**: Filter spots by intensity with optional Gaussian blur
- **Live visualization**: Real-time contrast updates and threshold visualization
- **Batch processing**: Navigate through entire datasets efficiently
- **Export functionality**: Save filtered results to CSV

## Project Structure

- `filospot/app.py` - Main application entry point
- `filospot/_widget.py` - Napari plugin widget interface
- `filospot/data_navigator.py` - Data loading and navigation functionality
- `filospot/alignment.py` - Image alignment utilities
- `filospot/filtering.py` - Spot filtering functionality
- `filospot/widgets.py` - GUI widgets and interface components
- `filospot/config.py` - Configuration and constants
- `filospot/napari.yaml` - Napari plugin manifest
- `tests/` - Test suite for plugin functionality
- `conda-recipe/` - Conda package recipe for conda-forge distribution

## Requirements

- Python >= 3.8
- napari[all]
- numpy
- pandas
- tifffile
- magicgui
- scipy

[napari]: https://napari.org
[pip]: https://pypi.org/project/pip/
