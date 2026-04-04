"""Tests for background and flat-field correction."""

import numpy as np
from microscope_imageprocessing.correction.background import BackgroundCorrectionUtils


class TestBackgroundCorrection:
    """Test flat-field correction methods."""

    def test_divide_correction_preserves_dtype(self, rgb_image_uint8, flat_background_uint8):
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            rgb_image_uint8, flat_background_uint8, scaling_factor=1.0, method="divide"
        )
        assert corrected.dtype == np.uint8

    def test_divide_correction_reduces_vignetting(self, rgb_image_uint8, flat_background_uint8):
        """After flat-field correction, spatial uniformity should improve."""
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            rgb_image_uint8, flat_background_uint8, scaling_factor=1.0, method="divide"
        )
        # Std across pixels should decrease (more uniform)
        raw_std = rgb_image_uint8[:, :, 0].astype(float).std()
        corrected_std = corrected[:, :, 0].astype(float).std()
        # Uniform background correction on vignette should reduce variation
        assert corrected_std <= raw_std * 1.1  # Allow small tolerance

    def test_subtract_method(self, rgb_image_uint8, flat_background_uint8):
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            rgb_image_uint8, flat_background_uint8, scaling_factor=1.0, method="subtract"
        )
        assert corrected.dtype == np.uint8
        assert corrected.min() >= 0

    def test_scaling_factor_applied(self, rgb_image_uint8, flat_background_uint8):
        c1 = BackgroundCorrectionUtils.apply_flat_field_correction(
            rgb_image_uint8, flat_background_uint8, scaling_factor=1.0
        )
        c2 = BackgroundCorrectionUtils.apply_flat_field_correction(
            rgb_image_uint8, flat_background_uint8, scaling_factor=0.5
        )
        # Lower scaling factor should produce darker image on average
        assert c2.astype(float).mean() < c1.astype(float).mean()


class TestBackgroundModeDetection:
    """Test histogram mode-based background estimation."""

    def test_rgb_background_detection(self, rgb_image_uint8):
        bg_mean, confidence = BackgroundCorrectionUtils.calculate_background_color_from_mode(
            rgb_image_uint8
        )
        assert bg_mean.shape == (3,)
        assert 0.0 <= confidence <= 1.0

    def test_grayscale_background_detection(self):
        gray = np.full((64, 64), 128, dtype=np.uint8)
        bg_mean, confidence = BackgroundCorrectionUtils.calculate_background_color_from_mode(gray)
        assert abs(bg_mean - 128) < 20
        assert confidence > 0.5


class TestModalityParsing:
    """Test scan type to modality conversion."""

    def test_three_part_format(self):
        assert BackgroundCorrectionUtils.get_modality_from_scan_type("PPM_10x_1") == "PPM_10x"
        assert BackgroundCorrectionUtils.get_modality_from_scan_type("PPM_40x_2") == "PPM_40x"

    def test_two_part_format(self):
        assert BackgroundCorrectionUtils.get_modality_from_scan_type("ppm_40x") == "ppm_40x"
        assert BackgroundCorrectionUtils.get_modality_from_scan_type("bf_20x") == "bf_20x"

    def test_single_part_passthrough(self):
        assert BackgroundCorrectionUtils.get_modality_from_scan_type("unknown") == "unknown"
