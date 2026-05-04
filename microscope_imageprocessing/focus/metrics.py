"""Focus metric implementations and dispatcher.

The single home for every focus metric in QPSC. Each metric is a
function ``f(image: np.ndarray) -> float`` where ``image`` is a 2D
grayscale array (preprocessing-extracted from multi-channel input by
``_to_gray``).

The ``resolve_metric(name)`` dispatcher is the only supported way to
look up a metric function. It validates against the manifest and
raises :class:`UnknownMetricError` (never silently substitutes) on a
miss -- the previous fall-back behavior masked drift bugs.

Implementation choices for metrics that previously had multiple
incarnations:

  - ``laplacian_variance`` -- scipy.ndimage.laplace + .var(). Matches
    the historical pycromanager standard-AF implementation and is the
    academic standard. The earlier numpy 4-neighbour stencil in
    streaming_focus.py is replaced.
  - ``brenner_gradient`` -- sum of squared lag-2 horizontal differences
    (the Brenner 1976 definition). Matches the streaming-AF
    implementation; the np.gradient-based version in pycromanager is
    replaced.
  - ``tenengrad`` -- sum of squared first differences in X and Y. The
    numpy-only form from streaming_focus.py (no skimage dep needed).
  - ``sobel`` -- skimage.filters.sobel + .var(). Matches the
    pycromanager standard-AF implementation; brings skimage into the
    streaming AF code path's dependency set, but skimage is already a
    dependency of microscope_imageprocessing.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from microscope_imageprocessing.focus.manifest import (
    UnknownMetricError,
    get_manifest,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- helpers

def _to_gray(image: np.ndarray) -> np.ndarray:
    """Reduce a multi-component frame to a 2D grayscale array.

    For 3D input with >= 3 components: BT.601 luminance
    Y = 0.299*R + 0.587*G + 0.114*B (standard photographic
    grayscale). For 2-component 3D input: average of the two
    channels. For 1-component 3D input (some camera SDKs return
    ``(H, W, 1)``): squeeze. For 2D input: passthrough. Always
    promotes to ``float64`` so downstream metric math is exact.

    PRIOR DESIGN (failed PPM 40x 2026-05-04): green-channel-only
    reduction missed the focus signal on JAI 3-CCD raw frames of
    eosin-stained tissue. Eosin's strongest absorption is in the
    green band, so the GREEN channel sees relatively flat
    illumination at every Z while the RED channel carries most of
    the focus-relevant contrast. The user could see histogram
    change clearly in the live viewer (which displays luminance)
    but every metric reported only 3-4% modulation across a 5 um
    scan and the gaussian fit committed Z up to 5 um from truth.
    """
    if image is None:
        return np.empty((0, 0), dtype=np.float64)
    arr = np.asarray(image)
    if arr.size == 0:
        return arr.astype(np.float64, copy=False)
    if arr.ndim == 3:
        nch = arr.shape[2]
        if nch >= 3:
            arr_f = arr.astype(np.float64, copy=False)
            return (0.299 * arr_f[:, :, 0]
                    + 0.587 * arr_f[:, :, 1]
                    + 0.114 * arr_f[:, :, 2])
        if nch == 2:
            arr_f = arr.astype(np.float64, copy=False)
            return 0.5 * (arr_f[:, :, 0] + arr_f[:, :, 1])
        arr = arr[:, :, 0]
    return arr.astype(np.float64, copy=False)


# ---------------------------------------------------------- numpy-only

def _normalized_variance(gray: np.ndarray) -> float:
    """var / mean. Sensitive to focus when the lamp vignette is small
    relative to sample variance. Known to fail at low magnifications
    where vignette dominates -- the OWS3 BF 10x bug from 2026-05-01."""
    mean = gray.mean()
    if mean <= 1e-9:
        return 0.0
    return float(gray.var() / mean)


def _vollath_f5(gray: np.ndarray) -> float:
    """Vollath's F5: sum I[x,y]*I[x+1,y] - N*mean(I)^2.

    Autocorrelation form -- effectively suppresses uncorrelated noise.
    Useful for sparse-signal modalities (fluorescence, LSM) where
    background shot noise dominates raw variance.
    """
    if gray.ndim != 2 or gray.shape[1] < 2:
        return 0.0
    shifted_product = float((gray[:, :-1] * gray[:, 1:]).sum())
    n = float(gray.size)
    mean = float(gray.mean())
    return shifted_product - n * mean * mean


def _tenengrad(gray: np.ndarray) -> float:
    """Sum of squared first differences in X and Y. Inherently
    high-pass; ignores low-spatial-frequency illumination. Streaming-AF
    default for tissue (BF, PPM) since 2026-05-02."""
    if gray.ndim != 2 or gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float((gx * gx).sum() + (gy * gy).sum())


def _brenner_gradient(gray: np.ndarray) -> float:
    """Brenner 1976: sum of squared lag-2 horizontal differences.

    High-pass and immune to slow illumination gradients. Cheap.
    The 2D ``np.gradient`` variant previously used in pycromanager has
    been replaced with the canonical lag-2 form.
    """
    if gray.ndim != 2 or gray.shape[1] < 3:
        return 0.0
    d = gray[:, 2:] - gray[:, :-2]
    return float((d * d).sum())


def _p98_p2(gray: np.ndarray) -> float:
    """Histogram spread: p98 - p2. Robust to outliers; good fallback
    when peak validation on a primary metric fails.

    Always cheap to compute -- the standard AF code path computes this
    alongside the primary metric for free (one pass over the same
    pixels) and uses it as a fallback when the primary's peak fails
    validation. ``role: fallback`` in the manifest documents this.
    """
    if gray.size == 0:
        return 0.0
    return float(np.percentile(gray, 98) - np.percentile(gray, 2))


def _none(gray: np.ndarray) -> float:
    """No-op metric. Strategy ``manual_only`` uses this so the AF
    pipeline always falls through to the manual-focus dialog without
    any score computation. Keeping it in the registry rather than
    special-casing in callers means the lookup table is uniform."""
    return 0.0


# -------------------------------------------------------- scipy-backed

def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian-filtered image (scipy 3x3 stencil).

    High-pass; kills low-spatial illumination. Matches the historical
    pycromanager implementation. The numerical value differs from the
    earlier numpy stencil in streaming_focus.py but the focus ranking
    is monotonic with it -- callers that compare scores within a
    single sweep are unaffected.
    """
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    from scipy.ndimage import laplace
    lap = laplace(gray)
    return float(lap.var())


# ------------------------------------------------------- skimage-backed

def _sobel(gray: np.ndarray) -> float:
    """Variance of the Sobel-filtered image. Edge-energy metric;
    works well on high-contrast features."""
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    from skimage.filters import sobel as skimage_sobel
    return float(skimage_sobel(gray).var())


def _robust_sharpness_metric(gray: np.ndarray) -> float:
    """Particle-resistant sharpness: median filter + Otsu-masked
    Laplacian variance. ~20 ms on a 2500x1900 frame.

    Used by modality-aware strategies (dense_texture) where outlier
    bright particles (dust, debris) would otherwise dominate a plain
    Laplacian variance.
    """
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    from skimage.filters import laplace, median, threshold_otsu
    from skimage.morphology import disk
    filtered = median(gray, disk(3))
    lap = laplace(filtered)
    threshold = threshold_otsu(gray)
    mask = gray > (threshold * 0.5)
    if mask.any():
        return float(lap[mask].var())
    return float(lap.var())


def _hybrid_sharpness_metric(gray: np.ndarray) -> float:
    """Soft-masked Brenner gradient on Gaussian-smoothed image. ~8 ms
    on a 2500x1900 frame.

    Compromise between Brenner's speed and robust_sharpness's particle
    resistance. Soft mask weights mid-gray pixels highest, suppressing
    saturated and very dark regions.
    """
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    from skimage.filters import gaussian
    smoothed = gaussian(gray, sigma=1.5)
    gy, gx = np.gradient(smoothed.astype(np.float32))
    gradient_magnitude = gx**2 + gy**2
    rng = float(gray.max() - gray.min())
    if rng < 1e-10:
        return 0.0
    normalized = (gray - gray.min()) / (rng + 1e-10)
    weight_mask = 1.0 - np.abs(normalized - 0.5) * 2.0
    return float(np.mean(gradient_magnitude * weight_mask))


# ------------------------------------------------------------- registry

# The runtime registry. Names MUST match the manifest's metrics list;
# the test suite enforces this and the manifest is the source of truth.
_IMPLEMENTATIONS: dict[str, Callable[[np.ndarray], float]] = {
    "tenengrad": _tenengrad,
    "laplacian_variance": _laplacian_variance,
    "brenner_gradient": _brenner_gradient,
    "normalized_variance": _normalized_variance,
    "vollath_f5": _vollath_f5,
    "sobel": _sobel,
    "p98_p2": _p98_p2,
    "robust_sharpness_metric": _robust_sharpness_metric,
    "hybrid_sharpness_metric": _hybrid_sharpness_metric,
    "none": _none,
}


def resolve_metric(name: str) -> Callable[[np.ndarray], float]:
    """Look up the metric function for a canonical name.

    Raises :class:`UnknownMetricError` for unknown names. If the name
    is in the manifest's ``removed_aliases``, the error message names
    the canonical replacement so the user can update their YAML.

    The returned callable accepts either a 2D grayscale array or a
    multi-channel frame (3D). Multi-channel frames are reduced to the
    green/index-1 channel before computation.
    """
    manifest = get_manifest()
    canonical = manifest.removed_aliases.get(name)
    if canonical is not None:
        raise UnknownMetricError(
            f"Metric '{name}' was renamed to '{canonical}'. "
            f"Update your autofocus YAML or run "
            f"scripts/migrate_autofocus_yaml.py."
        )
    if name not in manifest.metrics:
        raise UnknownMetricError(
            f"Unknown metric '{name}'. Available: {sorted(manifest.metrics)}."
        )
    impl = _IMPLEMENTATIONS.get(name)
    if impl is None:
        raise UnknownMetricError(
            f"Metric '{name}' is in the manifest but has no implementation "
            f"in microscope_imageprocessing.focus.metrics. This is a packaging "
            f"bug; please report."
        )
    return _wrap_with_preprocessing(impl)


def _wrap_with_preprocessing(
    impl: Callable[[np.ndarray], float],
) -> Callable[[np.ndarray], float]:
    """Adapt a 2D-grayscale metric to also accept multi-channel input."""
    def metric(image: np.ndarray) -> float:
        gray = _to_gray(image)
        if gray.size == 0:
            return 0.0
        try:
            return float(impl(gray))
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Focus metric %s raised: %s", impl.__name__, e)
            return 0.0
    metric.__name__ = impl.__name__
    metric.__doc__ = impl.__doc__
    return metric


def modality_default_metric(
    modality: Optional[str],
    fallback: str = "tenengrad",
) -> str:
    """Return the canonical metric name for a modality.

    Wraps :meth:`FocusMetricsManifest.modality_default_metric` and
    applies the codebase-wide fallback. Centralized so streaming AF,
    standard AF, and any future caller share the same answer.
    """
    name = get_manifest().modality_default_metric(modality)
    return name if name is not None else fallback


def list_metric_names() -> list[str]:
    """All canonical metric names, in manifest order. Convenience
    wrapper for callers (tests, GUIs) that need to iterate."""
    return list(get_manifest().metrics)
