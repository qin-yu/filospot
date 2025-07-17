"""Setup script for filospot package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="filospot",
    version="0.1.0",
    author="Qin Yu",
    description="For automating the loading, navigation, alignment, filter and export of image and spot data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "napari[all]",
        "numpy",
        "pandas",
        "tifffile",
        "magicgui",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "filospot=filospot:main",
        ],
    },
)
