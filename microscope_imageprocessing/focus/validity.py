"""Validity checks for autofocus strategies.

A validity check answers the modality-specific question: "is there
enough signal in this image for an autofocus search to be meaningful?"
The answer drives the strategy's failure-mode handling (defer to
later tile, proceed anyway, or pop the manual dialog).

Each check is a pure function:
    f(image, **params) -> (ok: bool, stats: dict)

The stats dict is logging-friendly and is included in the run_log so
post-hoc analysis can see why a tile was accepted or rejected.

The four canonical checks come from the focus_metrics_manifest.yml,
which is the single source of truth for parameter names, types, and
defaults. The ``resolve_validity_check(name)`` dispatcher wires a name
from a YAML strategy entry to the matching function.

Implementation choices for checks that previously had multiple incarnations:

  - ``texture_and_area`` -- the dense-tissue gate from
    ``microscope_control.autofocus.core.has_sufficient_signal`` is
    consolidated here. Modality-specific tissue_mask_range tweaks were
    moved to the per-strategy YAML (modality_defaults section); this
    function takes whatever range the caller passes.
  - ``bright_spot_count`` -- ``SparseSignalStrategy._compute_spots`` is
    consolidated. The scipy-vs-fallback connected-component logic moves
    here (scipy is a hard dependency of microscope_imageprocessing).
  - ``total_gradient_energy`` -- ``DarkFieldStrategy.is_valid``'s
    gradient-energy math is consolidated.
  - ``always_false`` -- the trivial check used by manual_only.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from microscope_imageprocessing.focus.manifest import (
    UnknownMetricError,
    get_manifest,
)

logger = logging.getLogger(__name__)


# Validity check signature: (image, **params) -> (ok, stats)
ValidityCheckFn = Callable[..., Tuple[bool, Dict[str, Any]]]


# --------------------------------------------------------------- helpers


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Reduce an image to a 2D float32 array. RGB averages all channels;
    2D inputs are returned as-is. Float32 (not float64) because validity
    math is statistical, not score-comparative."""
    if image.ndim == 3:
        return np.mean(image, axis=2).astype(np.float32)
    return image.astype(np.float32)


# ---------------------------------------------------------- texture_and_area


def texture_and_area(
    image: np.ndarray,
    texture_threshold: float = 0.010,
    tissue_area_threshold: float = 0.200,
    rgb_brightness_threshold: float = 240.0,
    tissue_mask_range: Tuple[float, float] = (0.10, 0.90),
    median_floor: float = 15.0,
    **_unused: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """Dense-tissue validity gate. Used by the dense_texture strategy.

    Three sub-checks combined:
      1. RGB brightness rejection: blank glass is very bright (~240+);
         skip those tiles before the more expensive checks.
      2. Texture: stddev of gradient magnitude inside the tissue mask
         must exceed ``texture_threshold``.
      3. Area: the fraction of pixels inside ``tissue_mask_range`` must
         exceed ``tissue_area_threshold``.

    The ``median_floor`` parameter is a brightness floor consulted by
    the brightness loop (separate concern); included in the stats so a
    caller can decide on its own.
    """
    rgb_mean = None
    avg_brightness = None
    if image.ndim == 3 and rgb_brightness_threshold is not None:
        rgb_mean = np.mean(image, axis=(0, 1))
        avg_brightness = float(np.mean(rgb_mean))
        if avg_brightness > rgb_brightness_threshold:
            return False, {
                "validity_check": "texture_and_area",
                "rejected_reason": "rgb_brightness",
                "avg_brightness": avg_brightness,
                "rgb_brightness_threshold": rgb_brightness_threshold,
                "rgb_mean": rgb_mean.tolist(),
            }

    img_gray = _to_gray(image)
    img_range = float(img_gray.max() - img_gray.min())
    if img_range < 1e-9:
        return False, {
            "validity_check": "texture_and_area",
            "rejected_reason": "no_dynamic_range",
            "texture": 0.0,
            "area": 0.0,
        }
    img_norm = (img_gray - img_gray.min()) / (img_range + 1e-10)

    gy, gx = np.gradient(img_norm)
    gradient_magnitude = np.sqrt(gx * gx + gy * gy)

    lo, hi = tissue_mask_range
    tissue_mask = (img_norm > lo) & (img_norm < hi)

    if np.any(tissue_mask):
        tissue_texture = float(np.std(gradient_magnitude[tissue_mask]))
        tissue_area_fraction = float(np.sum(tissue_mask) / tissue_mask.size)
    else:
        tissue_texture = 0.0
        tissue_area_fraction = 0.0

    sufficient_texture = tissue_texture > texture_threshold
    sufficient_area = tissue_area_fraction > tissue_area_threshold
    ok = sufficient_texture and sufficient_area

    return ok, {
        "validity_check": "texture_and_area",
        "texture": tissue_texture,
        "texture_threshold": texture_threshold,
        "area": tissue_area_fraction,
        "area_threshold": tissue_area_threshold,
        "sufficient_texture": sufficient_texture,
        "sufficient_area": sufficient_area,
        "tissue_mask_range": list(tissue_mask_range),
        "rgb_mean": rgb_mean.tolist() if rgb_mean is not None else None,
        "avg_brightness": avg_brightness,
        "median_floor": median_floor,  # informational
    }


# ---------------------------------------------------------- bright_spot_count


def bright_spot_count(
    image: np.ndarray,
    spot_sigma_above_bg: float = 5.0,
    spot_min_separation_px: int = 8,
    min_spots: int = 3,
    min_peak_intensity: float = 20.0,
    bright_pixel_floor: float = 50.0,
    **_unused: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """Sparse-signal validity gate. Used by the sparse_signal strategy.

    Counts bright local maxima above an adaptive background. The
    background is estimated robustly via median + MAD. A spot must be
    above ``min_peak_intensity`` AND above ``bg_median +
    spot_sigma_above_bg * bg_sigma`` to count.

    ``spot_min_separation_px`` parameter is reserved for a future
    nearest-neighbour merge step; the current implementation uses
    scipy's connected-component labelling which already merges
    touching pixels.

    ``bright_pixel_floor`` is a separate brightness-check parameter,
    included in stats for the caller's brightness loop.
    """
    from scipy import ndimage as _ndimage

    gray = _to_gray(image)
    if gray.max() > gray.min():
        gray_8bit = ((gray - gray.min()) / (gray.max() - gray.min()) * 255.0).astype(np.float32)
    else:
        gray_8bit = gray.astype(np.float32)

    bg_median = float(np.median(gray_8bit))
    bg_mad = float(np.median(np.abs(gray_8bit - bg_median))) + 1e-6
    bg_sigma = bg_mad * 1.4826
    spot_threshold = max(bg_median + spot_sigma_above_bg * bg_sigma, min_peak_intensity)

    fg_mask = gray_8bit > spot_threshold
    if not np.any(fg_mask):
        spot_count = 0
    else:
        _, n_spots = _ndimage.label(fg_mask)
        spot_count = int(n_spots)

    p99 = float(np.percentile(gray_8bit, 99))
    ok = spot_count >= min_spots

    return ok, {
        "validity_check": "bright_spot_count",
        "spot_count": spot_count,
        "min_spots": min_spots,
        "spot_threshold": float(spot_threshold),
        "bg_median": bg_median,
        "bg_sigma": bg_sigma,
        "p99": p99,
        "spot_min_separation_px": spot_min_separation_px,  # informational
        "bright_pixel_floor": bright_pixel_floor,  # informational
    }


# ---------------------------------------------------- total_gradient_energy


def total_gradient_energy(
    image: np.ndarray,
    min_gradient_energy: float = 0.002,
    **_unused: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """Whole-FOV gradient validity gate. Used by the dark_field strategy.

    Computes the mean squared gradient magnitude of the image normalized
    to [0, 1]. Passes when the mean exceeds ``min_gradient_energy``.
    Background-dominated samples (SHG, dark-field) where the whole frame
    is signal use this -- there is no spatial mask to apply.
    """
    gray = _to_gray(image)
    img_range = float(gray.max() - gray.min())
    if img_range < 1e-9:
        return False, {
            "validity_check": "total_gradient_energy",
            "gradient_energy": 0.0,
            "min_gradient_energy": min_gradient_energy,
            "rejected_reason": "no_dynamic_range",
        }
    normalized = (gray - gray.min()) / (img_range + 1e-10)
    gy, gx = np.gradient(normalized)
    gradient_energy = float(np.mean(gx * gx + gy * gy))
    ok = gradient_energy >= min_gradient_energy
    return ok, {
        "validity_check": "total_gradient_energy",
        "gradient_energy": gradient_energy,
        "min_gradient_energy": min_gradient_energy,
    }


# ------------------------------------------------------------- always_false


def always_false(image: np.ndarray, **_unused: Any) -> Tuple[bool, Dict[str, Any]]:
    """Trivial check that always rejects. Used by manual_only so the
    workflow's on_failure=MANUAL handler always pops the manual focus
    dialog. Centralising it here keeps the validity dispatcher uniform."""
    return False, {"validity_check": "always_false"}


# ------------------------------------------------- chroma_deviation


def chroma_deviation(
    image: np.ndarray,
    min_chroma: float = 12.0,
    chroma_area_threshold: float = 0.150,
    white_reference: Optional[np.ndarray] = None,
    saturation_ceiling: float = 250.0,
    **_unused: Any,
) -> Tuple[bool, Dict[str, Any]]:
    """Is there stained material in view -- judged by COLOUR, not sharpness.

    Every other check here asks whether the image is structured, which makes them all
    defocus-dependent: blur destroys spatial structure, so a badly out-of-focus field of
    tissue looks exactly like blank glass to them. That is fatal for the job they get used
    for -- deciding, from a scan that is out of focus almost everywhere, whether there is
    anything worth focusing ON.

    Colour survives defocus. Blur spreads a stained pixel over its neighbours but does not
    change what wavelengths were absorbed, so an H&E field stays pink/purple however soft it
    is, while blank glass under brightfield stays neutral. That is how a person recognises
    tissue in a badly focused field, and it is what this measures: the fraction of pixels
    whose colour is far enough from neutral grey.

    Chroma is max(R,G,B) - min(R,G,B), which is the unnormalised saturation. Deliberately NOT
    a hue angle: hue is meaningless and numerically unstable on near-neutral pixels, which is
    most of a mostly-blank field. Deliberately NOT normalised per frame either -- that is the
    flaw in ``texture_and_area``'s area term, where per-frame min/max scaling makes the mask
    cover nearly every pixel of any unimodal field, so the area fraction reads ~1.0 whether or
    not there is tissue (observed at 0.9989 and 0.99992 on fields that had none).

    :param image: HxWx3 RGB. A 2-D (monochrome) image has no colour information and returns
        False -- honestly, rather than by pretending a grey level means something.
    :param min_chroma: how far from neutral, in 8-bit counts, a pixel must be to count as
        stained. Sensor noise and slight illumination cast put blank glass in the low single
        digits; H&E sits well above it even when badly blurred.
    :param chroma_area_threshold: fraction of pixels that must clear ``min_chroma``.
    :param white_reference: optional per-pixel background (a collected flat field). When
        given, the frame is divided by it first, which removes the illumination's own colour
        cast and vignetting -- both of which otherwise add chroma that is not the sample's.
    :param saturation_ceiling: pixels at or above this in every channel are clipped and their
        colour is not trustworthy; they are excluded from the fraction.
    """
    stats: Dict[str, Any] = {"validity_check": "chroma_deviation"}
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        stats["rejected_reason"] = "not_colour"
        return False, stats

    rgb = image[:, :, :3].astype(np.float32)

    if white_reference is not None:
        ref = np.asarray(white_reference, dtype=np.float32)
        if ref.shape[:2] == rgb.shape[:2] and ref.ndim == 3 and ref.shape[2] >= 3:
            # Flat-field: divide, then rescale to the reference's own mean so the numbers stay
            # in 8-bit-ish units and min_chroma keeps its meaning.
            safe = np.where(ref[:, :, :3] <= 1.0, 1.0, ref[:, :, :3])
            rgb = rgb / safe * float(np.mean(safe))
        else:
            stats["white_reference"] = "ignored (shape mismatch)"

    hi = rgb.max(axis=2)
    lo = rgb.min(axis=2)
    chroma = hi - lo

    usable = lo < saturation_ceiling
    usable_count = int(np.count_nonzero(usable))
    if usable_count == 0:
        stats["rejected_reason"] = "all_pixels_clipped"
        return False, stats

    stained = np.count_nonzero((chroma >= min_chroma) & usable)
    fraction = stained / usable_count

    stats.update(
        {
            "chroma_fraction": float(fraction),
            "chroma_area_threshold": chroma_area_threshold,
            "median_chroma": float(np.median(chroma[usable])),
            "min_chroma": min_chroma,
            "usable_fraction": usable_count / float(chroma.size),
        }
    )
    return bool(fraction > chroma_area_threshold), stats


# ------------------------------------------------------------- registry

# Names MUST match the manifest's validity_checks list. The test suite
# enforces parity so a manifest edit cannot drift from the implementation.
_IMPLEMENTATIONS: Dict[str, ValidityCheckFn] = {
    "texture_and_area": texture_and_area,
    "bright_spot_count": bright_spot_count,
    "total_gradient_energy": total_gradient_energy,
    "chroma_deviation": chroma_deviation,
    "always_false": always_false,
}


def resolve_validity_check(name: str) -> ValidityCheckFn:
    """Look up the validity-check function for a manifest name.

    Raises :class:`UnknownMetricError` for unknown names. Mirrors
    ``resolve_metric``'s contract: never silently substitutes, the
    error message names the available options.
    """
    manifest = get_manifest()
    if name not in manifest.validity_checks:
        raise UnknownMetricError(
            f"Unknown validity check '{name}'. " f"Available: {sorted(manifest.validity_checks)}."
        )
    impl = _IMPLEMENTATIONS.get(name)
    if impl is None:
        raise UnknownMetricError(
            f"Validity check '{name}' is in the manifest but has no "
            f"implementation in microscope_imageprocessing.focus.validity. "
            f"This is a packaging bug; please report."
        )
    return impl


def list_validity_check_names() -> list[str]:
    """All canonical validity-check names, in manifest order."""
    return list(get_manifest().validity_checks)
