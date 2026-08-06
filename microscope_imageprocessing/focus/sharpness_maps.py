"""Per-PIXEL sharpness maps.

The focus metrics in :mod:`microscope_imageprocessing.focus.metrics` answer
"how in-focus is this *frame*" and return a single float. These answer "how
in-focus is each *pixel*" and return a 2D map the same size as the image.

They are separate because they serve a different question, and deliberately
live outside :mod:`microscope_imageprocessing.zstack` because nothing about
them is Z-stack specific. A sharpness map is useful on its own:

  - fusing a Z-stack into an extended-depth-of-field image
    (:func:`microscope_imageprocessing.zstack.edf.extended_depth_of_field`),
  - diagnosing sample or stage tilt by showing which part of a single field
    is in focus (the OWS3 2026-08-05 case: every tile sharp on the right and
    blurred on the left, a systematic ~0.13 degree tilt),
  - masking a tile to the region that is actually in focus.

Every map function takes an image (2D grayscale, or multi-channel which is
reduced with the same equal-weighted mean the autofocus metrics use) and
returns a ``float64`` array of the same height and width, where larger means
sharper. The values are NOT comparable between different map functions or
between images with different exposure -- only within one map, and across a
Z-stack of the same field, which is what the fusion needs.

Local response is averaged over a square ``window``. Raw per-pixel gradient
is far too noisy to pick a focal plane from: on a 16-bit brightfield tile
the shot noise in flat background swamps the real focus signal, and the
argmax over Z becomes a coin flip. The window is the single most important
parameter here.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np
from scipy import ndimage

from microscope_imageprocessing.focus.metrics import to_gray

logger = logging.getLogger(__name__)

#: Default local averaging window, in pixels. Odd so it is symmetric about the
#: pixel.
#:
#: 9 is a REASONED STARTING POINT, not a measured optimum: large enough that
#: single-pixel noise cannot produce a spurious argmax, small enough not to
#: smear the in-focus region's boundary across a whole cell at ~0.65 um/px. It
#: has not been swept against real stacks. The right value scales with pixel
#: size and with how noisy the camera is, so it is exposed as a parameter
#: rather than baked in -- if fused output looks blocky, raise it; if the
#: boundary between in-focus regions looks smeared, lower it.
DEFAULT_WINDOW = 9


def _prepare(image: np.ndarray) -> np.ndarray:
    """Reduce to 2D float64 grayscale, sharing the autofocus reduction."""
    gray = to_gray(image)
    return np.asarray(gray, dtype=np.float64)


def _smooth(response: np.ndarray, window: int) -> np.ndarray:
    """Average a raw per-pixel response over a square window."""
    if window <= 1:
        return response
    return ndimage.uniform_filter(response, size=window, mode="nearest")


def tenengrad_map(image: np.ndarray, window: int = DEFAULT_WINDOW) -> np.ndarray:
    """Per-pixel squared gradient magnitude (Tenengrad), locally averaged.

    The per-pixel counterpart of the ``tenengrad`` focus metric, so a stack
    fused with this agrees with what the autofocus was optimising. A good
    default for stained brightfield tissue, where focus shows up as edge
    contrast rather than as intensity.
    """
    gray = _prepare(image)
    gy, gx = np.gradient(gray)
    return _smooth(gx * gx + gy * gy, window)


def modified_laplacian_map(image: np.ndarray, window: int = DEFAULT_WINDOW) -> np.ndarray:
    """Per-pixel sum-modified-Laplacian (Nayar and Nakagawa), locally averaged.

    Takes absolute second differences in X and Y *separately* before summing,
    so opposite-sign curvature in the two axes cannot cancel -- which is the
    failure mode of a plain Laplacian on elongated structures such as fibres
    or a wing vein. Sharper-peaked in Z than Tenengrad, and correspondingly
    more sensitive to noise in flat regions.
    """
    gray = _prepare(image)
    lx = np.abs(2.0 * gray - np.roll(gray, 1, axis=1) - np.roll(gray, -1, axis=1))
    ly = np.abs(2.0 * gray - np.roll(gray, 1, axis=0) - np.roll(gray, -1, axis=0))
    # np.roll wraps, so the border columns/rows are meaningless -- copy their
    # neighbour inwards rather than leaving a bright false edge that would win
    # the argmax along the entire tile boundary.
    lx[:, 0] = lx[:, 1]
    lx[:, -1] = lx[:, -2]
    ly[0, :] = ly[1, :]
    ly[-1, :] = ly[-2, :]
    return _smooth(lx + ly, window)


def variance_map(image: np.ndarray, window: int = DEFAULT_WINDOW) -> np.ndarray:
    """Per-pixel local variance, over the same square window.

    Cheapest of the three and the most forgiving of noise, but it responds to
    local contrast rather than to edges specifically, so a dark blob on a
    bright field scores well whether or not its boundary is sharp. Reasonable
    when the sample is low-contrast and the others are picking up noise.
    """
    gray = _prepare(image)
    size = max(2, window)
    mean = ndimage.uniform_filter(gray, size=size, mode="nearest")
    mean_of_squares = ndimage.uniform_filter(gray * gray, size=size, mode="nearest")
    return np.maximum(mean_of_squares - mean * mean, 0.0)


#: Registry of per-pixel sharpness maps, by canonical name.
SHARPNESS_MAPS: Dict[str, Callable[..., np.ndarray]] = {
    "tenengrad": tenengrad_map,
    "modified_laplacian": modified_laplacian_map,
    "variance": variance_map,
}


def resolve_sharpness_map(name: str) -> Callable[..., np.ndarray]:
    """Look up a sharpness-map function by canonical name.

    Raises :class:`KeyError` on an unknown name rather than substituting a
    default, matching :func:`microscope_imageprocessing.focus.resolve_metric`:
    silently swapping the operator would change results with no trace.
    """
    if name not in SHARPNESS_MAPS:
        raise KeyError(
            f"Unknown sharpness map '{name}'. Available: {', '.join(sorted(SHARPNESS_MAPS))}"
        )
    return SHARPNESS_MAPS[name]


def list_sharpness_map_names() -> list:
    """Canonical sharpness-map names, sorted."""
    return sorted(SHARPNESS_MAPS)
