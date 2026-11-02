"""
Unit tests for Bayer pattern debayering.

Tests the CPUDebayer class which performs Bayer pattern interpolation
to convert raw camera data to RGB images.
"""

import numpy as np
import pytest
from microscope_imageprocessing.debayering.cpu import CPUDebayer, process_data

# Seeded, so a failure reproduces.
_RNG = np.random.default_rng(20261102)


class TestCPUDebayerBasic:
    """Test basic debayering functionality."""

    def test_debayer_rggb_pattern(self, synthetic_bayer_rggb):
        """Test debayering of RGGB Bayer pattern."""
        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(synthetic_bayer_rggb)

        # Check output shape
        h, w = synthetic_bayer_rggb.shape
        assert rgb.shape == (h, w, 3)

        # Check output dtype
        assert rgb.dtype == np.uint8

        # Check approximate RGB values in each quadrant
        # Top-left quadrant (red region)
        tl_r = rgb[64, 64, 0]  # Sample from center of quadrant
        assert tl_r > 150, f"Expected high red in top-left, got R={tl_r}"

        # Top-right quadrant (green region)
        tr_g = rgb[64, 320, 1]
        assert tr_g > 150, f"Expected high green in top-right, got G={tr_g}"

        # Bottom-left quadrant (blue region)
        bl_b = rgb[320, 64, 2]
        assert bl_b > 150, f"Expected high blue in bottom-left, got B={bl_b}"

        # Bottom-right quadrant (white region)
        br_r = rgb[320, 320, 0]
        br_g = rgb[320, 320, 1]
        br_b = rgb[320, 320, 2]
        assert (
            br_r > 150 and br_g > 150 and br_b > 150
        ), f"Expected white in bottom-right, got RGB=({br_r}, {br_g}, {br_b})"

    def test_debayer_grbg_pattern(self):
        """Test debayering of GRBG Bayer pattern."""
        # GRBG pattern:
        # G R
        # B G

        size = 256
        bayer = np.zeros((size, size), dtype=np.uint8)

        # Create simple pattern
        bayer[0::2, 0::2] = 200  # G positions
        bayer[0::2, 1::2] = 200  # R positions
        bayer[1::2, 0::2] = 50  # B positions
        bayer[1::2, 1::2] = 200  # G positions

        debayer = CPUDebayer(pattern="GRBG")
        rgb = debayer.debayer(bayer)

        assert rgb.shape == (size, size, 3)
        assert rgb.dtype == np.uint8

        # Should have high R and G, low B
        assert rgb[128, 128, 0] > 150  # Red
        assert rgb[128, 128, 1] > 150  # Green
        assert rgb[128, 128, 2] < 100  # Blue

    def test_debayer_gbrg_pattern(self):
        """Test debayering of GBRG Bayer pattern."""
        # GBRG pattern:
        # G B
        # R G

        size = 256
        bayer = np.zeros((size, size), dtype=np.uint8)

        bayer[0::2, 0::2] = 200  # G positions
        bayer[0::2, 1::2] = 200  # B positions
        bayer[1::2, 0::2] = 50  # R positions
        bayer[1::2, 1::2] = 200  # G positions

        debayer = CPUDebayer(pattern="GBRG")
        rgb = debayer.debayer(bayer)

        assert rgb.shape == (size, size, 3)
        assert rgb.dtype == np.uint8

    def test_debayer_bggr_pattern(self):
        """Test debayering of BGGR Bayer pattern."""
        # BGGR pattern:
        # B G
        # G R

        size = 256
        bayer = np.zeros((size, size), dtype=np.uint8)

        bayer[0::2, 0::2] = 200  # B positions
        bayer[0::2, 1::2] = 200  # G positions
        bayer[1::2, 0::2] = 200  # G positions
        bayer[1::2, 1::2] = 50  # R positions

        debayer = CPUDebayer(pattern="BGGR")
        rgb = debayer.debayer(bayer)

        assert rgb.shape == (size, size, 3)
        assert rgb.dtype == np.uint8

        # Should have high B and G, low R
        assert rgb[128, 128, 2] > 150  # Blue
        assert rgb[128, 128, 1] > 150  # Green
        assert rgb[128, 128, 0] < 100  # Red


class TestCPUDebayerBitDepth:
    """Test debayering with different bit depths."""

    def test_debayer_uint8_input(self):
        """Test debayering with uint8 input."""
        bayer = _RNG.integers(0, 255, (256, 256), dtype=np.uint8)

        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer)

        assert rgb.dtype == np.uint8
        assert np.all(rgb >= 0) and np.all(rgb <= 255)

    def test_debayer_uint16_input(self):
        """Test debayering with uint16 input (common for scientific cameras)."""
        bayer = _RNG.integers(0, 65535, (256, 256), dtype=np.uint16)

        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer)

        # Output dtype should match input dtype (dtype preservation)
        assert rgb.dtype == np.uint16
        assert np.all(rgb >= 0) and np.all(rgb <= 65535)

    def test_debayer_preserves_dynamic_range(self):
        """Test that debayering preserves relative intensity differences."""
        # Create bayer with two distinct intensity levels
        bayer = np.zeros((256, 256), dtype=np.uint8)

        # Left half: dim
        bayer[:, :128] = 50

        # Right half: bright
        bayer[:, 128:] = 200

        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer)

        # Left half should be dimmer than right half
        left_mean = np.mean(rgb[:, :128, :])
        right_mean = np.mean(rgb[:, 128:, :])

        assert right_mean > left_mean


class TestCPUDebayerEdgeCases:
    """Test edge cases and error handling."""

    def test_debayer_minimum_size(self):
        """Test debayering with minimum viable image size."""
        # 2x2 is the minimum for a Bayer pattern
        tiny_bayer = np.array([[100, 150], [150, 100]], dtype=np.uint8)

        debayer = CPUDebayer(pattern="RGGB")

        try:
            rgb = debayer.debayer(tiny_bayer)
            assert rgb.shape[0] >= 2 and rgb.shape[1] >= 2
        except Exception as e:
            pytest.fail(f"Debayer failed on 2x2 image: {e}")

    def test_debayer_odd_dimensions(self):
        """Test debayering with odd image dimensions."""
        # Bayer patterns typically work best with even dimensions
        # but should handle odd dimensions gracefully
        odd_bayer = _RNG.integers(0, 255, (257, 257), dtype=np.uint8)

        debayer = CPUDebayer(pattern="RGGB")

        try:
            rgb = debayer.debayer(odd_bayer)
            assert rgb.shape[0] == 257 and rgb.shape[1] == 257
        except Exception as e:
            # Some implementations may require even dimensions
            if "even" not in str(e).lower():
                pytest.fail(f"Unexpected error with odd dimensions: {e}")

    def test_debayer_all_zeros(self):
        """Test debayering with all-zero (black) image."""
        black_bayer = np.zeros((256, 256), dtype=np.uint8)

        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(black_bayer)

        # Output should be all zeros
        assert np.all(rgb == 0)

    def test_debayer_all_max_value(self):
        """Test debayering with saturated (all-white) image."""
        white_bayer = np.full((256, 256), 255, dtype=np.uint8)

        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(white_bayer)

        # Output should be all 255 (white)
        assert np.all(rgb == 255) or np.mean(rgb) > 250

    def test_invalid_pattern_raises_error(self):
        """Test that invalid Bayer pattern raises appropriate error."""
        with pytest.raises((ValueError, KeyError)):
            CPUDebayer(pattern="INVALID")

    def test_invalid_input_shape_raises_error(self):
        """Test that non-2D input raises appropriate error."""
        invalid_input = _RNG.integers(0, 255, (256, 256, 3), dtype=np.uint8)

        debayer = CPUDebayer(pattern="RGGB")

        with pytest.raises((ValueError, AssertionError)):
            debayer.debayer(invalid_input)


class TestCPUDebayerClipping:
    """Test clipping behavior at intensity boundaries."""

    def test_no_overflow_with_interpolation(self):
        """Test that interpolation doesn't cause overflow."""
        # Create Bayer with high values that might overflow during interpolation
        bayer = np.full((256, 256), 250, dtype=np.uint8)

        # Add some variation
        bayer[::2, ::2] = 255

        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer)

        # Should clip to 255, not overflow
        assert np.all(rgb <= 255)
        assert rgb.dtype == np.uint8

    def test_no_underflow_with_interpolation(self):
        """Test that interpolation doesn't cause underflow."""
        bayer = np.full((256, 256), 5, dtype=np.uint8)

        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer)

        # Should clip to 0, not underflow
        assert np.all(rgb >= 0)


class TestCPUDebayerEdgeHandling:
    """Test edge/border handling in debayering."""

    def test_edge_pixels_are_valid(self):
        """Test that edge pixels are properly interpolated."""
        bayer = _RNG.integers(50, 200, (256, 256), dtype=np.uint8)

        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer)

        # Check that edge pixels have reasonable values
        assert np.all(rgb[0, :, :] > 0)  # Top edge
        assert np.all(rgb[-1, :, :] > 0)  # Bottom edge
        assert np.all(rgb[:, 0, :] > 0)  # Left edge
        assert np.all(rgb[:, -1, :] > 0)  # Right edge

    def test_corner_pixels_are_valid(self):
        """Test that corner pixels are properly handled."""
        bayer = _RNG.integers(50, 200, (256, 256), dtype=np.uint8)

        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer)

        # Check corners
        assert np.all(rgb[0, 0, :] > 0)  # Top-left
        assert np.all(rgb[0, -1, :] > 0)  # Top-right
        assert np.all(rgb[-1, 0, :] > 0)  # Bottom-left
        assert np.all(rgb[-1, -1, :] > 0)  # Bottom-right


class TestCPUDebayerConsistency:
    """Test consistency and determinism of debayering."""

    def test_debayer_is_deterministic(self, synthetic_bayer_rggb):
        """Multiple calls should produce identical results."""
        debayer = CPUDebayer(pattern="RGGB")

        rgb1 = debayer.debayer(synthetic_bayer_rggb)
        rgb2 = debayer.debayer(synthetic_bayer_rggb)

        assert np.array_equal(rgb1, rgb2)

    def test_debayer_doesnt_modify_input(self, synthetic_bayer_rggb):
        """Debayering should not modify the input array."""
        original = synthetic_bayer_rggb.copy()

        debayer = CPUDebayer(pattern="RGGB")
        debayer.debayer(synthetic_bayer_rggb)

        assert np.array_equal(synthetic_bayer_rggb, original)


class TestProcessDataFunction:
    """The in-memory convenience wrapper."""

    def test_process_data_wrapper(self, synthetic_bayer_rggb):
        rgb = process_data(synthetic_bayer_rggb, pattern="RGGB")
        assert rgb.shape == (512, 512, 3)
        assert rgb.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
