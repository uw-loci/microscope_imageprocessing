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
from typing import Any, Callable, Dict, Tuple

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
        gray_8bit = (
            (gray - gray.min()) / (gray.max() - gray.min()) * 255.0
        ).astype(np.float32)
    else:
        gray_8bit = gray.astype(np.float32)

    bg_median = float(np.median(gray_8bit))
    bg_mad = float(np.median(np.abs(gray_8bit - bg_median))) + 1e-6
    bg_sigma = bg_mad * 1.4826
    spot_threshold = max(
        bg_median + spot_sigma_above_bg * bg_sigma, min_peak_intensity
    )

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


# ------------------------------------------------------------- registry

# Names MUST match the manifest's validity_checks list. The test suite
# enforces parity so a manifest edit cannot drift from the implementation.
_IMPLEMENTATIONS: Dict[str, ValidityCheckFn] = {
    "texture_and_area": texture_and_area,
    "bright_spot_count": bright_spot_count,
    "total_gradient_energy": total_gradient_energy,
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
            f"Unknown validity check '{name}'. "
            f"Available: {sorted(manifest.validity_checks)}."
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
