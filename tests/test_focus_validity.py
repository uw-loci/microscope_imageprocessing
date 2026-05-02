"""Tests for focus validity checks and the resolve_validity_check dispatcher.

The validity checks are the second consolidation target. Like
metrics, they previously lived in 3+ places (core.has_sufficient_signal,
SparseSignalStrategy._compute_spots, DarkFieldStrategy.is_valid). These
tests lock in:

  1. **Manifest-vs-implementation parity.** Every validity check named
     in the manifest has an implementation, and every implementation is
     in the manifest.
  2. **Behavioural contract.** Each check's ``ok`` boolean reacts the
     correct way to images that should pass and images that should fail.
  3. **Stats dict shape.** The stats dict carries the keys callers rely
     on for logging and post-hoc analysis.
"""
from __future__ import annotations

import numpy as np
import pytest

from microscope_imageprocessing.focus import (
    UnknownMetricError,
    clear_cache,
    list_validity_check_names,
    resolve_validity_check,
)
from microscope_imageprocessing.focus import validity as validity_module


@pytest.fixture(autouse=True)
def _reset_cache():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------- manifest <-> code

class TestManifestImplementationParity:
    def test_every_manifest_check_has_implementation(self):
        for name in list_validity_check_names():
            assert name in validity_module._IMPLEMENTATIONS, (
                f"Manifest validity check {name!r} has no implementation."
            )

    def test_every_implementation_is_in_manifest(self):
        manifest_names = set(list_validity_check_names())
        for name in validity_module._IMPLEMENTATIONS:
            assert name in manifest_names, (
                f"Implementation {name!r} is not declared in the manifest."
            )


# ----------------------------------------------------- dispatcher behavior

class TestResolveValidityCheck:
    def test_returns_callable_for_each_canonical_name(self):
        for name in list_validity_check_names():
            fn = resolve_validity_check(name)
            assert callable(fn)

    def test_unknown_name_raises(self):
        with pytest.raises(UnknownMetricError, match="Unknown validity check"):
            resolve_validity_check("not_a_real_check")


# ------------------------------------------------- texture_and_area

class TestTextureAndArea:
    def test_dense_texture_image_passes(self):
        # Random grayscale fills the tissue mask range (0.1..0.9 normalized)
        # and has gradient texture. Should pass with default thresholds.
        rng = np.random.default_rng(0)
        img = rng.uniform(50, 200, size=(128, 128)).astype(np.float32)
        ok, stats = validity_module.texture_and_area(img)
        assert ok is True
        assert stats["validity_check"] == "texture_and_area"
        assert stats["sufficient_texture"] is True
        assert stats["sufficient_area"] is True

    def test_blank_field_fails(self):
        # Constant image: no gradient, no texture, no area.
        img = np.full((128, 128), 128.0, dtype=np.float32)
        ok, stats = validity_module.texture_and_area(img)
        assert ok is False
        # The "no_dynamic_range" early return tag.
        assert stats.get("rejected_reason") == "no_dynamic_range"

    def test_rgb_blank_glass_rejected_by_brightness(self):
        # Bright RGB tile (avg 250) hits the rgb_brightness early-rejection.
        img = np.full((128, 128, 3), 250.0, dtype=np.float32)
        ok, stats = validity_module.texture_and_area(
            img, rgb_brightness_threshold=240.0
        )
        assert ok is False
        assert stats.get("rejected_reason") == "rgb_brightness"
        assert stats["avg_brightness"] >= 240.0

    def test_thresholds_tunable(self):
        # Same image, different thresholds: a permissive threshold should
        # pass an image a strict threshold rejects. Random texture sits
        # in the middle: high enough for the loose threshold, too low
        # for an unrealistically strict one.
        rng = np.random.default_rng(1)
        img = rng.uniform(50, 200, size=(128, 128)).astype(np.float32)
        ok_strict, stats_strict = validity_module.texture_and_area(
            img, texture_threshold=10.0
        )
        ok_loose, stats_loose = validity_module.texture_and_area(
            img, texture_threshold=0.001
        )
        assert ok_strict is False
        assert stats_strict["sufficient_texture"] is False
        assert ok_loose is True
        assert stats_loose["sufficient_texture"] is True

    def test_unused_kwargs_tolerated(self):
        # Strategies pass the full param dict; the function should
        # accept and ignore keys it does not consume.
        img = np.random.default_rng(2).uniform(50, 200, size=(64, 64)).astype(np.float32)
        ok, _ = validity_module.texture_and_area(
            img,
            texture_threshold=0.001,
            tissue_area_threshold=0.001,
            future_param=12345,
        )
        assert ok is True


# ------------------------------------------------- bright_spot_count

class TestBrightSpotCount:
    def _sparse_spots_image(self, n_spots: int = 5) -> np.ndarray:
        # Dark background with bright Gaussian spots. The image is
        # 128x128 8-bit-equivalent.
        rng = np.random.default_rng(42)
        img = rng.normal(10.0, 2.0, size=(128, 128)).astype(np.float32)
        img = np.clip(img, 0, None)
        # Add bright spots far apart so the labelling resolves them.
        for cy, cx in [(20, 20), (20, 90), (60, 50), (100, 30), (100, 100)][:n_spots]:
            yy, xx = np.mgrid[0:128, 0:128]
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            img += 200.0 * np.exp(-r2 / 4.0)
        return img

    def test_passes_when_enough_spots(self):
        img = self._sparse_spots_image(n_spots=5)
        ok, stats = validity_module.bright_spot_count(img, min_spots=3)
        assert ok is True
        assert stats["spot_count"] >= 3
        assert "bg_median" in stats and "bg_sigma" in stats

    def test_fails_when_too_few_spots(self):
        # Two spots, min_spots=3 -> fail.
        img = self._sparse_spots_image(n_spots=2)
        ok, stats = validity_module.bright_spot_count(img, min_spots=3)
        assert ok is False
        assert stats["spot_count"] < 3

    def test_fails_on_blank(self):
        img = np.full((64, 64), 50.0, dtype=np.float32)
        ok, stats = validity_module.bright_spot_count(img, min_spots=1)
        # No bright spots above background; even a permissive min_spots
        # cannot pass when there are zero connected components.
        assert ok is False
        assert stats["spot_count"] == 0


# ------------------------------------------------- total_gradient_energy

class TestTotalGradientEnergy:
    def test_textured_image_passes(self):
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1000, size=(64, 64)).astype(np.float32)
        ok, stats = validity_module.total_gradient_energy(img)
        assert ok is True
        assert stats["gradient_energy"] >= stats["min_gradient_energy"]

    def test_smooth_image_fails(self):
        # Linear gradient has very low gradient energy after normalization.
        y, x = np.mgrid[0:64, 0:64]
        img = (x + y).astype(np.float32)
        ok, stats = validity_module.total_gradient_energy(
            img, min_gradient_energy=0.1
        )
        assert ok is False
        assert stats["gradient_energy"] < 0.1

    def test_constant_image_fails(self):
        img = np.full((64, 64), 100.0, dtype=np.float32)
        ok, stats = validity_module.total_gradient_energy(img)
        assert ok is False
        assert stats.get("rejected_reason") == "no_dynamic_range"


# ------------------------------------------------------------- always_false

class TestAlwaysFalse:
    def test_returns_false_regardless_of_input(self):
        img = np.random.default_rng(0).uniform(0, 255, (64, 64)).astype(np.float32)
        ok, stats = validity_module.always_false(img)
        assert ok is False
        assert stats["validity_check"] == "always_false"

    def test_handles_blank(self):
        img = np.zeros((4, 4), dtype=np.float32)
        ok, stats = validity_module.always_false(img)
        assert ok is False


# ------------------------------------------------- behavioural sanity

def test_list_validity_check_names_includes_canonicals():
    names = list_validity_check_names()
    assert set(names) == {
        "texture_and_area",
        "bright_spot_count",
        "total_gradient_energy",
        "always_false",
    }
