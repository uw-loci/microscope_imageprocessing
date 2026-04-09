"""OME-TIFF writing utilities for microscopy image files.

Provides standards-compliant OME-TIFF output with resolution metadata,
suitable for any microscopy modality.
"""

import platform
from typing import Optional
import numpy as np
import tifffile as tf
import logging

import microscope_imageprocessing

logger = logging.getLogger(__name__)


def ome_tiff_writer(
    filename: str,
    pixel_size_um: float,
    data: np.ndarray,
    compression: Optional[str] = None,
):
    """Write OME-TIFF file with metadata.

    Args:
        filename: Output filename
        pixel_size_um: Pixel size in micrometers
        data: Image data array
        compression: Compression type (None, "lzw", "zlib", "deflate", etc.)
                    None = uncompressed (default for scientific data compatibility)
                    "lzw" requires the 'imagecodecs' package
                    "zlib"/"deflate" work without additional dependencies
    """
    with tf.TiffWriter(filename) as tif:
        options = {
            "photometric": "rgb" if len(data.shape) == 3 else "minisblack",
            "resolutionunit": "CENTIMETER",
            "maxworkers": 2,
        }

        if compression is not None:
            options["compression"] = compression.lower()

        description = (
            f"microscope_imageprocessing={microscope_imageprocessing.__version__}"
            f" python={platform.python_version()}"
        )

        tif.write(
            data,
            resolution=(1e4 / pixel_size_um, 1e4 / pixel_size_um),
            description=description,
            **options,
        )


def format_imagetags(tags: dict) -> dict:
    """Format image tags by grouping by prefix."""
    dx = {}
    for k in set([key.split("-")[0] for key in tags]):
        dx.update({k: {key: tags[key] for key in tags if key.startswith(k)}})
    return dx


