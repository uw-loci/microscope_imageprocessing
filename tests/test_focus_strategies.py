"""Tests for autofocus strategies and the build_strategy factory.

Strategies are the assembly layer over metrics + validity checks.
These tests lock in:

  1. **Manifest-vs-implementation parity.** Every strategy in the
     manifest has a class.
  2. **Each strategy honours its declared default failure mode** so
     the workflow's failure-handling branches reach the right path.
  3. **build_strategy applies YAML overrides correctly** -- including
     dropping unknown keys (so a YAML annotation like 'description'
     doesn't crash the constructor).
  4. **Each strategy delegates to the correct validity + score
     functions** -- the consolidation invariant.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from microscope_imageprocessing.focus import (
    DarkFieldStrategy,
    DenseTextureStrategy,
    ManualOnlyStrategy,
    SparseSignalStrategy,
    StrategyFailureMode,
    build_strategy,
    clear_cache,
    list_strategy_names,
)
from microscope_imageprocessing.focus import strategies as strategies_module


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------- manifest <-> code


def test_every_manifest_strategy_has_class():
    for name in list_strategy_names():
        assert (
            name in strategies_module._STRATEGY_CLASSES
        ), f"Manifest strategy {name!r} has no class registered."


def test_every_class_is_in_manifest():
    manifest_names = set(list_strategy_names())
    for name in strategies_module._STRATEGY_CLASSES:
        assert name in manifest_names, f"Strategy class {name!r} is not declared in the manifest."


# ------------------------------------------------- failure mode defaults


def test_dense_texture_defers():
    s = DenseTextureStrategy()
    assert s.on_failure is StrategyFailureMode.DEFER


def test_sparse_signal_proceeds():
    s = SparseSignalStrategy()
    assert s.on_failure is StrategyFailureMode.PROCEED


def test_dark_field_proceeds():
    s = DarkFieldStrategy()
    assert s.on_failure is StrategyFailureMode.PROCEED


def test_manual_only_pops_dialog():
    s = ManualOnlyStrategy()
    assert s.on_failure is StrategyFailureMode.MANUAL


# ------------------------------------------------- delegation invariants


class TestDenseTexture:
    def test_is_valid_passes_dense_image(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(50, 200, size=(128, 128)).astype(np.float32)
        s = DenseTextureStrategy()
        ok, stats = s.is_valid(img)
        assert ok is True
        assert stats["strategy"] == "dense_texture"
        assert stats["validity_check"] == "texture_and_area"

    def test_score_is_positive_on_textured_image(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(50, 200, size=(128, 128)).astype(np.float32)
        s = DenseTextureStrategy()
        assert s.score(img) > 0

    def test_brightness_check_below_floor_fails(self):
        # Mostly-dark image with a single bright pixel -- median is near 0.
        img = np.zeros((64, 64), dtype=np.float32)
        img[0, 0] = 1000.0
        s = DenseTextureStrategy(median_floor=15.0)
        ok, stats = s.brightness_acceptable(img)
        assert ok is False
        assert stats["brightness_check"] == "median_floor"


class TestSparseSignal:
    def _spotty_image(self, n=5):
        rng = np.random.default_rng(0)
        img = rng.normal(10.0, 2.0, size=(128, 128)).astype(np.float32)
        img = np.clip(img, 0, None)
        for cy, cx in [(20, 20), (20, 90), (60, 50), (100, 30), (100, 100)][:n]:
            yy, xx = np.mgrid[0:128, 0:128]
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            img += 200.0 * np.exp(-r2 / 4.0)
        return img

    def test_is_valid_when_enough_spots(self):
        img = self._spotty_image(n=5)
        s = SparseSignalStrategy(min_spots=3)
        ok, stats = s.is_valid(img)
        assert ok is True
        assert stats["validity_check"] == "bright_spot_count"

    def test_score_falls_back_when_too_few_foreground_pixels(self):
        # Flat image: foreground mask empty -> brenner fallback runs and
        # returns the whole-FOV score (typically tiny but defined).
        img = np.full((64, 64), 50.0, dtype=np.float32)
        s = SparseSignalStrategy()
        # Must not raise; falls back to brenner. Score may be 0 because
        # the image is constant, but the fallback path is what we test.
        score = s.score(img)
        assert score >= 0.0

    def test_score_uses_masked_image_when_spots_present(self):
        img = self._spotty_image(n=5)
        s = SparseSignalStrategy()
        score = s.score(img)
        assert score > 0.0


class TestDarkField:
    def test_is_valid_passes_textured_image(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1000, size=(64, 64)).astype(np.float32)
        s = DarkFieldStrategy()
        ok, stats = s.is_valid(img)
        assert ok is True
        assert stats["validity_check"] == "total_gradient_energy"

    def test_brightness_p99_check(self):
        img = np.full((64, 64), 1.0, dtype=np.float32)
        s = DarkFieldStrategy(p99_floor=30.0)
        ok, _ = s.brightness_acceptable(img)
        # All pixels are 1; after normalisation by max=1 they become 255
        # (since gray/max*255), so p99 = 255 which exceeds 30 -- passes.
        # The dim case is when max=0; covered by a separate test.
        assert ok is True

    def test_brightness_zero_image(self):
        img = np.zeros((64, 64), dtype=np.float32)
        s = DarkFieldStrategy()
        ok, stats = s.brightness_acceptable(img)
        assert ok is False
        assert stats["p99"] < stats["floor"]


class TestManualOnly:
    def test_always_invalid(self):
        img = np.random.default_rng(0).uniform(0, 255, (64, 64)).astype(np.float32)
        s = ManualOnlyStrategy()
        ok, stats = s.is_valid(img)
        assert ok is False
        assert stats["validity_check"] == "always_false"

    def test_score_is_zero(self):
        img = np.random.default_rng(0).uniform(0, 255, (64, 64)).astype(np.float32)
        s = ManualOnlyStrategy()
        assert s.score(img) == 0.0

    def test_brightness_always_acceptable(self):
        img = np.zeros((64, 64), dtype=np.float32)
        s = ManualOnlyStrategy()
        ok, _ = s.brightness_acceptable(img)
        assert ok is True


# ------------------------------------------------- build_strategy factory


class TestBuildStrategy:
    def test_unknown_strategy_falls_back_to_dense_texture(self, caplog):
        with caplog.at_level(logging.WARNING):
            s = build_strategy("not_a_real_strategy")
        assert isinstance(s, DenseTextureStrategy)
        assert any("not_a_real_strategy" in r.message for r in caplog.records)

    def test_no_params_uses_defaults(self):
        s = build_strategy("dense_texture")
        assert isinstance(s, DenseTextureStrategy)
        assert s.texture_threshold == 0.010

    def test_validity_params_get_flattened(self):
        # YAML structure: {validity_params: {texture_threshold: 0.05}}
        s = build_strategy(
            "dense_texture",
            {"validity_params": {"texture_threshold": 0.05}},
        )
        assert s.texture_threshold == 0.05

    def test_score_metric_renamed_for_constructor(self):
        # YAML key 'score_metric' becomes constructor arg 'score_metric_name'.
        s = build_strategy("dense_texture", {"score_metric": "tenengrad"})
        assert s.score_metric_name == "tenengrad"

    def test_unknown_params_dropped_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            s = build_strategy(
                "dense_texture",
                {"texture_threshold": 0.02, "future_param": 999},
            )
        assert s.texture_threshold == 0.02
        assert any("future_param" in r.message for r in caplog.records)

    def test_description_and_validity_check_keys_ignored(self):
        # YAML annotations the strategy class doesn't consume.
        s = build_strategy(
            "dense_texture",
            {
                "description": "a dense thing",
                "validity_check": "texture_and_area",
                "brightness_check": "median_floor",
            },
        )
        assert isinstance(s, DenseTextureStrategy)

    def test_on_failure_override(self):
        s = build_strategy("dense_texture", {"on_failure": "manual"})
        assert s.on_failure is StrategyFailureMode.MANUAL

    def test_invalid_on_failure_keeps_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            s = build_strategy("dense_texture", {"on_failure": "garbage"})
        assert s.on_failure is StrategyFailureMode.DEFER  # default
        assert any("garbage" in r.message for r in caplog.records)


def test_list_strategy_names_matches_manifest():
    # The four "core" strategies must always be exposed; the manifest
    # may declare additional ones (e.g. dense_fluorescence) and the
    # registry must keep up with it.
    names = set(list_strategy_names())
    assert {
        "dense_texture",
        "sparse_signal",
        "dark_field",
        "manual_only",
    }.issubset(names)


# ------------------------------------------------- saturation handling
# Per-strategy saturation tolerance drives the AF auto-exposure reducer
# (microscope_command_server _guard_af_saturation). Regression cover for
# the 2026-05-31 PPM 40x focus runaway, where a saturated red channel
# inverted the focus metric and walked the stage. See
# claude-reports/2026-06-02_autofocus-focus-runaway.md.


def _rgb_with_channel_saturated(channel: int, fraction: float) -> np.ndarray:
    """100x100 RGB uint8 image with `fraction` of pixels clipped in `channel`."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    n_rows = int(round(fraction * 100))
    img[:n_rows, :, channel] = 255
    return img


class TestSaturationFraction:
    def test_worst_channel_fraction_rgb(self):
        img = _rgb_with_channel_saturated(channel=0, fraction=0.20)
        frac = strategies_module.worst_channel_saturation_fraction(img)
        assert frac == pytest.approx(0.20)

    def test_monochrome_fraction(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:10, :] = 255
        assert strategies_module.worst_channel_saturation_fraction(img) == pytest.approx(0.10)

    def test_float_0_1_uses_high_level(self):
        img = np.zeros((10, 10), dtype=np.float32)
        img[:1, :] = 1.0  # 10% at full scale
        assert strategies_module.worst_channel_saturation_fraction(img) == pytest.approx(0.10)

    def test_empty_image_is_zero(self):
        assert strategies_module.worst_channel_saturation_fraction(None) == 0.0


class TestSaturationTolerance:
    def test_dense_texture_tolerates_5pct_not_20pct(self):
        s = build_strategy("dense_texture")
        assert s.saturation_threshold == pytest.approx(0.10)
        assert s.saturation_acceptable(_rgb_with_channel_saturated(2, 0.05))[0] is True
        assert s.saturation_acceptable(_rgb_with_channel_saturated(0, 0.20))[0] is False

    def test_sparse_rejects_even_5pct(self):
        # A sparse field clips all its signal in a few percent of pixels, so
        # the tolerance must be far tighter than dense tissue.
        s = build_strategy("sparse_signal")
        assert s.saturation_threshold == pytest.approx(0.03)
        assert s.saturation_acceptable(_rgb_with_channel_saturated(1, 0.05))[0] is False
        assert s.saturation_acceptable(_rgb_with_channel_saturated(1, 0.02))[0] is True

    def test_dark_field_tight_like_sparse(self):
        s = build_strategy("dark_field")
        assert s.saturation_threshold == pytest.approx(0.03)
        assert s.saturation_acceptable(_rgb_with_channel_saturated(1, 0.05))[0] is False

    def test_manual_only_never_reduces(self):
        s = build_strategy("manual_only")
        # manual_only never runs auto AF; saturation is always "acceptable".
        assert s.saturation_acceptable(_rgb_with_channel_saturated(0, 0.99))[0] is True

    def test_yaml_override_of_threshold(self):
        s = build_strategy("dense_texture", {"validity_params": {"saturation_threshold": 0.25}})
        assert s.saturation_threshold == pytest.approx(0.25)
        assert s.saturation_acceptable(_rgb_with_channel_saturated(0, 0.20))[0] is True

    def test_stats_payload_shape(self):
        s = build_strategy("dense_texture")
        ok, stats = s.saturation_acceptable(_rgb_with_channel_saturated(0, 0.20))
        assert ok is False
        assert stats["strategy"] == "dense_texture"
        assert stats["saturation_fraction"] == pytest.approx(0.20)
        assert stats["saturation_threshold"] == pytest.approx(0.10)
