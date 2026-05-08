"""
Microscope Imaging - General-purpose microscopy imaging utilities.

Provides modality-independent tools for:
- Bayer pattern debayering (CPU and GPU)
- Background/flat-field correction
- OME-TIFF writing with metadata
- Z-stack projection operators
- Autofocus metrics and tissue detection

This package has no dependencies on microscope hardware or specific
imaging modalities (PPM, fluorescence, etc.), making it suitable for
any microscopy workflow.
"""

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("microscope-imageprocessing")
except Exception:
    __version__ = "0.1.0.dev"

from microscope_imageprocessing.debayering import CPUDebayer
from microscope_imageprocessing.correction.background import BackgroundCorrectionUtils
from microscope_imageprocessing.io.writer import ome_tiff_writer

__all__ = [
    "CPUDebayer",
    "BackgroundCorrectionUtils",
    "ome_tiff_writer",
]
