"""OME-TIFF writing utilities for microscopy image files.

Provides standards-compliant OME-TIFF output with resolution metadata,
suitable for any microscopy modality.

The :func:`ome_tiff_writer` function is a thin 2D adapter over
:class:`~microscope_imageprocessing.io.ome_writer.StackWriter` configured
with ``size_t=size_z=size_c=1``. It preserves the pre-refactor pixel data,
photometric interpretation, compression, and physical-size metadata so
existing callers see no behavior change. See :func:`ome_tiff_writer` for
the exact byte differences vs the legacy writer.
"""

import platform
from typing import Optional

import numpy as np

import microscope_imageprocessing
from microscope_imageprocessing.io.ome_writer import StackWriter


def ome_tiff_writer(
    filename: str,
    pixel_size_um: float,
    data: np.ndarray,
    compression: Optional[str] = None,
):
    """Write a single-plane OME-TIFF with physical pixel size metadata.

    Thin 2D adapter over :class:`StackWriter`: constructs a writer with
    ``size_t=size_z=size_c=1`` and writes one plane. The ``data`` array is
    stored verbatim as grayscale ``(Y, X)`` with ``photometric='minisblack'``
    or RGB ``(Y, X, 3)`` with ``photometric='rgb'`` (SamplesPerPixel=3,
    CONTIG PlanarConfig) per the existing stitched-OME-TIFF format.

    Byte differences vs the pre-refactor direct-tifffile writer:
      * The secondary ``ImageDescription`` tag that tifffile emits by
        default (the shape-JSON ``{"shape": [...]}`` -- tifffile's
        round-tripping aid) is suppressed by :class:`StackWriter`'s
        ``metadata=None`` call. This is cosmetic and does not affect
        readback by :func:`tifffile.imread` or downstream OME tooling.
      * ``StripOffsets`` shifts to reflect the shorter header. Pixel bytes
        themselves are byte-identical.
      * All other tags (Compression, PhotometricInterpretation, XResolution,
        YResolution, ResolutionUnit, BitsPerSample, SamplesPerPixel,
        PlanarConfiguration, the primary ``ImageDescription`` version
        string, Software) are unchanged.

    Args:
        filename: Output filename.
        pixel_size_um: Pixel size in micrometers.
        data: Image data array. ``(Y, X)`` grayscale or ``(Y, X, 3)`` RGB.
        compression: Compression type (None, "lzw", "zlib", "deflate", ...).
            None = uncompressed (default for scientific data compatibility).
            "lzw" requires the 'imagecodecs' package.
            "zlib"/"deflate" work without additional dependencies.
    """
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        photometric = "rgb"
    else:
        photometric = "minisblack"

    size_y, size_x = int(arr.shape[0]), int(arr.shape[1])

    description = (
        f"microscope_imageprocessing={microscope_imageprocessing.__version__}"
        f" python={platform.python_version()}"
    )

    writer = StackWriter(
        filename,
        size_t=1,
        size_z=1,
        size_c=1,
        size_y=size_y,
        size_x=size_x,
        dtype=arr.dtype,
        pixel_size_um=pixel_size_um,
        channel_names=["image"],
        granularity="single",
        bigtiff=False,
        compression=compression.lower() if compression is not None else None,
        photometric=photometric,
        description_override=description,
    )
    try:
        writer.write_frame(arr, t=0, z=0, c=0)
    finally:
        writer.close()


def format_imagetags(tags: dict) -> dict:
    """Format image tags by grouping by prefix."""
    dx = {}
    for k in set([key.split("-")[0] for key in tags]):
        dx.update({k: {key: tags[key] for key in tags if key.startswith(k)}})
    return dx
