"""Tests for Bayer pattern demosaicing."""

import numpy as np
from microscope_imageprocessing.debayering import CPUDebayer


class TestCPUDebayer:
    """Test CPU debayering with various patterns and dtypes."""

    def test_rggb_uint16_shape(self, bayer_rggb_uint16):
        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer_rggb_uint16)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint16

    def test_rggb_uint8_shape(self, bayer_rggb_uint8):
        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer_rggb_uint8)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_preserves_dtype(self, bayer_rggb_uint16, bayer_rggb_uint8):
        debayer = CPUDebayer(pattern="RGGB")
        assert debayer.debayer(bayer_rggb_uint16).dtype == np.uint16
        assert debayer.debayer(bayer_rggb_uint8).dtype == np.uint8

    def test_no_negative_values(self, bayer_rggb_uint16):
        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer_rggb_uint16)
        assert rgb.min() >= 0

    def test_all_patterns_produce_valid_output(self, bayer_rggb_uint8):
        for pattern in ["RGGB", "GRBG", "GBRG", "BGGR"]:
            debayer = CPUDebayer(pattern=pattern)
            rgb = debayer.debayer(bayer_rggb_uint8)
            assert rgb.shape == (64, 64, 3), f"Failed for pattern {pattern}"

    def test_channels_differ(self, bayer_rggb_uint16):
        """Debayered R and G channels should differ (R varies, G is constant)."""
        debayer = CPUDebayer(pattern="RGGB")
        rgb = debayer.debayer(bayer_rggb_uint16)
        r_std = rgb[:, :, 0].astype(float).std()
        g_std = rgb[:, :, 1].astype(float).std()
        # R has a gradient (high std), G is constant 20000 (low std)
        assert r_std > g_std * 2
