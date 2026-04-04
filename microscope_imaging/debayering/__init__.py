"""Bayer pattern demosaicing (debayering) for raw camera images.

Provides CPU-based bilinear interpolation for converting single-channel
Bayer pattern images into full RGB images. Supports RGGB, GRBG, GBRG,
and BGGR patterns.
"""

from microscope_imaging.debayering.cpu import CPUDebayer

__all__ = ["CPUDebayer"]
