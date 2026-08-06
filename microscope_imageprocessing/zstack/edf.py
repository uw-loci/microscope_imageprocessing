"""Extended depth of field: fuse a Z-stack into one all-in-focus image.

For each pixel, pick the Z-plane where that pixel is sharpest and take its
value from there. Unlike max/min/mean, this is a *focus* operator: it makes
no assumption that the wanted signal is the brightest or the darkest, which
is why the existing projections are wrong for brightfield. A max projection
of a brightfield stack preferentially selects the brightest pixels, and in
transmitted light the brightest pixels are the empty background and the most
defocused tissue -- so it actively degrades the thing you wanted.

Where this is worth using: when a single field spans more than a depth of
field in Z, so no single plane is in focus everywhere. That happens with a
tilted sample or stage, with thick or non-flat tissue, and with anything
mounted unevenly. Autofocus cannot help there -- it chooses one Z for the
whole field, so part of the field is defocused whatever it chooses. Measured
case (OWS3, 2026-08-05): a 0.5 NA air 10x, 1337 um field with roughly 3 um
depth of field, blurred at the left edge and sharp at the right on every
tile, which is about 0.13 degrees of tilt.

Cost: one exposure per plane. Three planes is usually enough to cover a few
micrometres of tilt, which is cheaper than the fifteen-step autofocus sweep
that cannot fix it.

The sharpness measurement lives in
:mod:`microscope_imageprocessing.focus.sharpness_maps` so it can be used
without a Z-stack; this module only does the selection and fusion.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Tuple, Union

import numpy as np
from scipy import ndimage

from microscope_imageprocessing.focus.sharpness_maps import (
    DEFAULT_WINDOW,
    resolve_sharpness_map,
)

logger = logging.getLogger(__name__)

#: Default median-filter size applied to the chosen-plane index map. The
#: per-pixel argmax is independently noisy: in a flat region neighbouring
#: pixels can select different planes for no physical reason, and the fused
#: image then shows salt-and-pepper texture stitched from several planes.
#: Real focal surfaces are smooth, so filtering the index map costs little
#: and removes that. 0 disables it.
#:
#: 5 is a REASONED STARTING POINT, not a measured optimum, and has not been
#: swept against real stacks. Larger values enforce a smoother focal surface
#: -- good for a tilted flat sample, bad where focus genuinely steps (a fold,
#: or a torn section), since the median will then bridge across the step.
DEFAULT_INDEX_SMOOTH = 5


def extended_depth_of_field(
    stack: List[np.ndarray],
    metric: str = "tenengrad",
    window: int = DEFAULT_WINDOW,
    index_smooth: int = DEFAULT_INDEX_SMOOTH,
    return_height_map: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Fuse a Z-stack by selecting the sharpest plane per pixel.

    Args:
        stack: Z-planes, in Z order. All must share a shape and dtype.
            2D grayscale or 3D multi-channel (H, W, C) are both accepted.
        metric: Sharpness map name -- ``tenengrad`` (default),
            ``modified_laplacian`` or ``variance``. See
            :mod:`microscope_imageprocessing.focus.sharpness_maps`.
        window: Local averaging window for the sharpness map, in pixels.
        index_smooth: Median-filter size for the chosen-plane index map.
            0 disables smoothing.
        return_height_map: Also return the per-pixel chosen plane index.

    Returns:
        The fused image, with the input dtype and shape. When
        ``return_height_map`` is set, a ``(fused, height_map)`` tuple, where
        the height map is ``uint8`` plane indices -- useful for checking
        whether the selected surface looks like a plane (tilt) or noise.

    Raises:
        ValueError: Empty stack, or planes whose shapes disagree.
        KeyError: Unknown ``metric``.

    For multi-channel input the sharpness is computed on the grayscale
    reduction and the SAME plane is taken for every channel of a pixel, so
    colour cannot be split across planes -- picking channels independently
    would produce colour fringing at every point where the focal surface
    crosses a plane boundary.
    """
    if not stack:
        raise ValueError("Cannot compute EDF of an empty stack")

    first = stack[0]
    for i, plane in enumerate(stack):
        if plane.shape != first.shape:
            raise ValueError(
                f"Z-plane {i} has shape {plane.shape}, expected {first.shape} "
                "-- every plane must be the same size"
            )

    if len(stack) == 1:
        logger.debug("EDF on a single-plane stack is a no-op")
        if return_height_map:
            return first.copy(), np.zeros(first.shape[:2], dtype=np.uint8)
        return first.copy()

    sharpness_fn = resolve_sharpness_map(metric)

    # (Z, H, W) sharpness. Computed plane by plane rather than on a stacked
    # array so peak memory stays at one float64 map per plane rather than a
    # float64 copy of the whole stack.
    maps = np.stack([sharpness_fn(plane, window=window) for plane in stack], axis=0)

    index = np.argmax(maps, axis=0).astype(np.int32)

    if index_smooth and index_smooth > 1:
        # Median rather than mean: averaging plane indices would invent
        # intermediate planes that were never acquired, and at a step in the
        # focal surface it would select a plane that is sharp in neither
        # region. The median always returns an index some neighbour chose.
        index = ndimage.median_filter(index, size=index_smooth, mode="nearest")

    # Where no plane is meaningfully sharper than the others -- empty
    # background, saturated regions -- the argmax is arbitrary and would
    # scatter the fused output across planes. Pin those to the middle plane
    # instead, which is the conventional "nothing to choose" answer and keeps
    # background visually continuous.
    sharp_max = maps.max(axis=0)
    sharp_min = maps.min(axis=0)
    spread = sharp_max - sharp_min
    middle = len(stack) // 2
    # Scale-free threshold: a pixel counts as undecided when the best plane
    # beats the worst by less than 1% of the field's median best score.
    reference = float(np.median(sharp_max))
    # A zero reference means nothing in the field has measurable sharpness -- a
    # blank or saturated field, or a stack of identical planes. Every argmax is
    # then a tie broken by array order, which silently returns plane 0 and
    # looks like a real choice. An infinite threshold marks the whole field
    # undecided, which is the honest answer.
    threshold = 0.01 * reference if reference > 0 else np.inf
    undecided = spread < threshold
    if undecided.any():
        index = np.where(undecided, middle, index)
        logger.debug(
            "EDF: %.1f%% of pixels had no clear focal plane; pinned to plane %d",
            100.0 * float(undecided.mean()),
            middle,
        )

    arr = np.stack(stack, axis=0)
    if arr.ndim == 4:
        # (Z, H, W, C): broadcast the per-pixel plane choice across channels.
        gather = index[np.newaxis, :, :, np.newaxis]
        fused = np.take_along_axis(arr, gather, axis=0)[0]
    else:
        gather = index[np.newaxis, :, :]
        fused = np.take_along_axis(arr, gather, axis=0)[0]

    fused = fused.astype(first.dtype, copy=False)

    if return_height_map:
        return fused, index.astype(np.uint8)
    return fused


def focus_height_map(
    stack: List[np.ndarray],
    metric: str = "tenengrad",
    window: int = DEFAULT_WINDOW,
    index_smooth: int = DEFAULT_INDEX_SMOOTH,
) -> np.ndarray:
    """Per-pixel index of the sharpest plane, without fusing.

    The focal surface on its own. A plane-like ramp across the field means
    the sample or stage is tilted and tells you the direction; a flat map
    means the field is genuinely within one depth of field; a noisy map means
    there is not enough contrast for the measurement to mean anything.
    """
    _, height = extended_depth_of_field(
        stack,
        metric=metric,
        window=window,
        index_smooth=index_smooth,
        return_height_map=True,
    )
    return height


def make_edf_projection(
    metric: str = "tenengrad",
    window: int = DEFAULT_WINDOW,
    index_smooth: int = DEFAULT_INDEX_SMOOTH,
) -> Callable[[List[np.ndarray]], np.ndarray]:
    """Build a registry-shaped EDF projection with specific settings.

    The projection registry's contract is ``List[ndarray] -> ndarray``, with no
    room for parameters, so a caller that wants non-default settings needs this
    rather than :func:`get_projection`. Validates eagerly: an unknown metric
    raises here, at configuration time, instead of part-way through an
    acquisition that has already cost hours of stage time.

    Args:
        metric: Sharpness map name -- see
            :mod:`microscope_imageprocessing.focus.sharpness_maps`.
        window: Local averaging window for the sharpness map, in pixels.
        index_smooth: Median-filter size for the chosen-plane index map;
            0 disables it.

    Returns:
        A callable taking a Z-stack and returning the fused image.

    Raises:
        KeyError: Unknown ``metric``.
        ValueError: Negative ``window`` or ``index_smooth``.
    """
    resolve_sharpness_map(metric)  # fail now, not mid-acquisition
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if index_smooth < 0:
        raise ValueError(f"index_smooth must be >= 0, got {index_smooth}")

    def _projection(stack: List[np.ndarray]) -> np.ndarray:
        result = extended_depth_of_field(
            stack, metric=metric, window=window, index_smooth=index_smooth
        )
        assert isinstance(result, np.ndarray)  # return_height_map defaults False
        return result

    _projection.__name__ = f"edf_{metric}_w{window}_s{index_smooth}"
    return _projection


def edf_projection(stack: List[np.ndarray]) -> np.ndarray:
    """Registry-compatible EDF projection using default settings.

    Adapts :func:`extended_depth_of_field` to the
    ``List[ndarray] -> ndarray`` signature the projection registry uses.
    """
    result = extended_depth_of_field(stack)
    assert isinstance(result, np.ndarray)  # return_height_map defaults False
    return result


__all__ = [
    "DEFAULT_INDEX_SMOOTH",
    "edf_projection",
    "extended_depth_of_field",
    "focus_height_map",
    "make_edf_projection",
]
