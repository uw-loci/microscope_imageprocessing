"""Tests for focus metric implementations and the resolve_metric dispatcher.

The metric implementations are the consolidation target -- previously
they lived in 3+ places with different names. These tests lock in two
properties:

  1. **Manifest-vs-implementation parity.** Every name in the manifest
     has an implementation; every implementation is named in the
     manifest. The drift bug class that motivated this refactor must
     not silently re-emerge.
  2. **Focus monotonicity.** Each metric, given a sharp image and a
     blurred-version of the same image, produces a strictly larger
     score on the sharp one. This is the minimum behavioural
     guarantee a focus metric must satisfy.

Numeric stability and absolute scores are not tested -- callers
compare scores within a single sweep, so absolute values are not part
of the contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from microscope_imageprocessing.focus import (
    UnknownMetricError,
    clear_cache,
    list_metric_names,
    modality_default_metric,
    resolve_metric,
)
from microscope_imageprocessing.focus import metrics as metrics_module


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


# --------------------------------------------------------------- fixtures


@pytest.fixture
def sharp_image() -> np.ndarray:
    """A vertical sharp edge: left half 0, right half 1000.

    Has both a strong horizontal gradient (good for tenengrad / sobel /
    brenner) and a sharp Laplacian step (good for laplacian_variance).
    Wide dynamic range so p98_p2 separates from a flat blur.
    """
    img = np.zeros((128, 128), dtype=np.float64)
    img[:, 64:] = 1000.0
    return img


@pytest.fixture
def blurred_image() -> np.ndarray:
    """The same sharp edge smeared with an erf profile sigma=8."""
    img = np.zeros((128, 128), dtype=np.float64)
    sigma = 8.0
    for x in range(128):
        img[:, x] = 1000.0 * 0.5 * (1.0 + math.erf((x - 64) / (sigma * math.sqrt(2))))
    return img


# ---------------------------------------------------- manifest <-> code


class TestManifestImplementationParity:
    def test_every_manifest_metric_has_implementation(self):
        for name in list_metric_names():
            assert (
                name in metrics_module._IMPLEMENTATIONS
            ), f"Manifest metric {name!r} has no implementation in metrics._IMPLEMENTATIONS."

    def test_every_implementation_is_in_manifest(self):
        manifest_names = set(list_metric_names())
        for name in metrics_module._IMPLEMENTATIONS:
            assert name in manifest_names, (
                f"Implementation {name!r} is not declared in the manifest. "
                f"Add a metric entry or remove the implementation."
            )

    def test_no_duplicate_implementations(self):
        # Every implementation function is registered under exactly one
        # name. Aliases must go through the alias-table, not by binding
        # the same callable to two registry keys.
        callables = list(metrics_module._IMPLEMENTATIONS.values())
        assert len(callables) == len(set(map(id, callables)))


# ----------------------------------------------------- dispatcher behavior


class TestResolveMetric:
    def test_returns_callable_for_each_canonical_name(self):
        for name in list_metric_names():
            fn = resolve_metric(name)
            assert callable(fn)

    def test_unknown_name_raises(self):
        with pytest.raises(UnknownMetricError, match="Unknown metric"):
            resolve_metric("not_a_real_metric")

    def test_deprecated_alias_raises_with_canonical_name_in_message(self):
        # The whole point of the rename: the loader names the new
        # spelling so users can fix their YAML in seconds.
        with pytest.raises(UnknownMetricError, match="renamed to 'vollath_f5'"):
            resolve_metric("volath5")
        with pytest.raises(UnknownMetricError, match="renamed to 'tenengrad'"):
            resolve_metric("tenenbaum_gradient")

    def test_resolved_metric_handles_3d_input(self):
        # Multi-channel frames should reduce to the green channel and
        # produce a non-zero score on a textured input.
        fn = resolve_metric("tenengrad")
        rgb = np.zeros((64, 64, 3), dtype=np.float64)
        rgb[:, 32:, 1] = 1000.0  # green channel has a sharp edge
        assert fn(rgb) > 0

    def test_resolved_metric_handles_2d_input(self):
        fn = resolve_metric("tenengrad")
        gray = np.zeros((64, 64), dtype=np.float64)
        gray[:, 32:] = 1000.0
        assert fn(gray) > 0

    def test_resolved_metric_returns_zero_on_empty(self):
        fn = resolve_metric("tenengrad")
        assert fn(np.empty((0, 0))) == 0.0

    def test_resolved_metric_returns_zero_on_none(self):
        fn = resolve_metric("tenengrad")
        assert fn(None) == 0.0


# ----------------------------------------------------- focus monotonicity

# The focus contract: sharp > blurred for every metric except 'none'
# (which is intentionally constant). Parametrize so a future metric
# addition gets the test for free.
#
# 'hybrid_sharpness_metric' weights pixels by proximity to mid-gray
# (the soft mask), so a binary edge with only extreme values produces
# zero weight everywhere. It needs its own textured fixture below.

_BASIC_FOCUSING_METRICS = [
    name
    for name in metrics_module._IMPLEMENTATIONS
    if name not in ("none", "hybrid_sharpness_metric")
]


@pytest.mark.parametrize("metric_name", _BASIC_FOCUSING_METRICS)
def test_metric_favours_sharp_over_blurred(
    metric_name: str, sharp_image: np.ndarray, blurred_image: np.ndarray
):
    fn = resolve_metric(metric_name)
    sharp_score = fn(sharp_image)
    blurred_score = fn(blurred_image)
    assert sharp_score > blurred_score, (
        f"Metric '{metric_name}' should rank sharp > blurred but got "
        f"sharp={sharp_score:.6f} blurred={blurred_score:.6f}."
    )


def test_hybrid_sharpness_favours_sharp_over_blurred():
    # hybrid_sharpness_metric weights mid-gray pixels highest, so it
    # needs a fixture with intermediate intensities. A random texture
    # bounded to the mid-gray band has both: lots of mid-gray pixels
    # for the soft mask, and high spatial frequency that disappears
    # when blurred.
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    sharp = rng.uniform(300.0, 700.0, size=(128, 128))
    blurred = gaussian_filter(sharp, sigma=4.0)
    fn = resolve_metric("hybrid_sharpness_metric")
    assert fn(sharp) > fn(blurred), (
        f"hybrid_sharpness_metric should rank textured-sharp > textured-blurred "
        f"but got sharp={fn(sharp):.6f} blurred={fn(blurred):.6f}."
    )


def test_none_is_constant():
    fn = resolve_metric("none")
    img1 = np.random.rand(64, 64)
    img2 = np.zeros((64, 64))
    assert fn(img1) == fn(img2) == 0.0


# ------------------------------------------------- modality dispatch


class TestModalityDefault:
    def test_brightfield_resolves_to_tenengrad(self):
        assert modality_default_metric("Brightfield") == "tenengrad"
        assert modality_default_metric("BF") == "tenengrad"
        assert modality_default_metric("ppm") == "tenengrad"

    def test_fluorescence_resolves_to_vollath_f5(self):
        assert modality_default_metric("Fluorescence") == "vollath_f5"
        assert modality_default_metric("LSM") == "vollath_f5"

    def test_unknown_modality_uses_fallback(self):
        assert modality_default_metric("unknown_modality") == "tenengrad"
        assert modality_default_metric(None) == "tenengrad"
        assert modality_default_metric("", fallback="laplacian_variance") == "laplacian_variance"

    def test_explicit_fallback_argument(self):
        # Fallback only applies when the modality is missing; a known
        # modality still returns its mapped metric.
        assert modality_default_metric("brightfield", fallback="sobel") == "tenengrad"


# ----------------------------------------------- behavioural sanity


def test_list_metric_names_includes_canonicals():
    names = list_metric_names()
    assert "tenengrad" in names
    assert "laplacian_variance" in names
    assert "p98_p2" in names
    assert "none" in names
    # Aliases must NOT leak into the canonical name list.
    assert "volath5" not in names
    assert "tenenbaum_gradient" not in names


def test_p98_p2_separates_dynamic_range():
    # p98_p2 is the histogram-spread fallback; it should react strongly
    # to a wide dynamic range (sharp edge has 0..1000) vs a uniform
    # field (zero spread).
    fn = resolve_metric("p98_p2")
    sharp = np.zeros((64, 64), dtype=np.float64)
    sharp[:, 32:] = 1000.0
    flat = np.full((64, 64), 500.0, dtype=np.float64)
    assert fn(sharp) > 100.0
    assert fn(flat) == 0.0
