"""Tests for Z-stack projection operators."""

import numpy as np
from microscope_imageprocessing.zstack.projections import (
    max_intensity_projection,
    min_intensity_projection,
    mean_projection,
    sum_projection,
    std_projection,
    get_projection,
    generate_z_offsets,
)


class TestProjections:
    """Test Z-stack projection operators with synthetic data."""

    def test_max_projection(self, z_stack_uint16):
        result = max_intensity_projection(z_stack_uint16)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint16
        # Max should be >= any individual plane
        for plane in z_stack_uint16:
            assert np.all(result >= plane)

    def test_min_projection(self, z_stack_uint16):
        result = min_intensity_projection(z_stack_uint16)
        assert result.shape == (32, 32)
        # Min should be <= any individual plane
        for plane in z_stack_uint16:
            assert np.all(result <= plane)

    def test_mean_projection(self, z_stack_uint16):
        result = mean_projection(z_stack_uint16)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint16
        # Mean should be between min and max
        mn = min_intensity_projection(z_stack_uint16).astype(float)
        mx = max_intensity_projection(z_stack_uint16).astype(float)
        assert np.all(result.astype(float) >= mn - 1)
        assert np.all(result.astype(float) <= mx + 1)

    def test_sum_projection_no_overflow(self):
        """Sum of uint16 planes should not overflow."""
        planes = [np.full((4, 4), 60000, dtype=np.uint16) for _ in range(10)]
        result = sum_projection(planes)
        assert result.dtype == np.uint16
        assert result.max() == 65535  # Clipped to uint16 max

    def test_std_projection(self, z_stack_uint16):
        result = std_projection(z_stack_uint16)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint16

    def test_registry_lookup(self):
        for name in ["max", "min", "mean", "sum", "std"]:
            fn = get_projection(name)
            assert callable(fn)

    def test_registry_unknown_raises(self):
        try:
            get_projection("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


class TestZOffsets:
    """Test Z-offset generation."""

    def test_symmetric_offsets(self):
        offsets = generate_z_offsets(z_range_um=10.0, z_step_um=2.0)
        assert offsets[0] == -5.0
        assert offsets[-1] == 5.0
        assert len(offsets) == 6  # -5, -3, -1, 1, 3, 5

    def test_zero_range_returns_single(self):
        offsets = generate_z_offsets(z_range_um=0.0, z_step_um=1.0)
        assert offsets == [0.0]

    def test_step_larger_than_range(self):
        offsets = generate_z_offsets(z_range_um=2.0, z_step_um=5.0)
        assert len(offsets) == 1
        assert offsets[0] == -1.0
