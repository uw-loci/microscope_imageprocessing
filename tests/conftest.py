"""Shared fixtures for microscope_imageprocessing tests.

Provides synthetic test images that exercise the key code paths without
requiring real microscopy data. All generated images are small (<=64x64)
to keep tests fast.
"""

import numpy as np
import pytest


@pytest.fixture
def bayer_rggb_uint16():
    """64x64 synthetic RGGB Bayer pattern image (uint16).

    Creates a gradient pattern where each 2x2 cell has:
      R  G1
      G2  B
    with channel-specific intensity ramps so debayered output has
    visibly different R, G, B planes.
    """
    h, w = 64, 64
    img = np.zeros((h, w), dtype=np.uint16)
    # Red pixels (even row, even col) -- ramp from 1000 to 40000
    img[0::2, 0::2] = np.linspace(1000, 40000, (h // 2) * (w // 2)).reshape(h // 2, w // 2).astype(np.uint16)
    # Green pixels (even row odd col + odd row even col)
    img[0::2, 1::2] = 20000
    img[1::2, 0::2] = 20000
    # Blue pixels (odd row, odd col) -- ramp inverted
    img[1::2, 1::2] = np.linspace(40000, 1000, (h // 2) * (w // 2)).reshape(h // 2, w // 2).astype(np.uint16)
    return img


@pytest.fixture
def bayer_rggb_uint8():
    """64x64 synthetic RGGB Bayer pattern image (uint8)."""
    h, w = 64, 64
    img = np.zeros((h, w), dtype=np.uint8)
    img[0::2, 0::2] = 200  # R
    img[0::2, 1::2] = 128  # G1
    img[1::2, 0::2] = 128  # G2
    img[1::2, 1::2] = 80   # B
    return img


@pytest.fixture
def rgb_image_uint8():
    """64x64x3 synthetic RGB image (uint8) with uneven illumination.

    Center is brighter than edges, simulating vignetting for
    background correction tests.
    """
    h, w = 64, 64
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    # Radial falloff (vignetting)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    falloff = 1.0 - 0.4 * (dist / dist.max())
    base = (falloff * 180).clip(0, 255).astype(np.uint8)
    img = np.stack([base, base, base], axis=-1)
    return img


@pytest.fixture
def flat_background_uint8():
    """64x64x3 uniform background image for flat-field correction."""
    return np.full((64, 64, 3), 200, dtype=np.uint8)


@pytest.fixture
def z_stack_uint16():
    """List of 5 synthetic 32x32 uint16 images simulating a Z-stack.

    Middle plane (index 2) has highest contrast (sharpest focus).
    """
    stack = []
    for i in range(5):
        # Gaussian blur increases away from center plane
        sigma = abs(i - 2) * 2.0 + 0.5
        from scipy.ndimage import gaussian_filter
        sharp = np.random.RandomState(42).randint(0, 60000, (32, 32), dtype=np.uint16)
        blurred = gaussian_filter(sharp.astype(np.float64), sigma=sigma)
        stack.append(np.clip(blurred, 0, 65535).astype(np.uint16))
    return stack
