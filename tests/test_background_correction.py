"""
Unit tests for background correction and flat-field correction.

Tests the apply_flat_field_correction function and background detection methods.
"""

import numpy as np
import pytest
from microscope_imageprocessing.correction.background import BackgroundCorrectionUtils

# Seeded, so a failure reproduces.
_RNG = np.random.default_rng(20261102)


class TestCalculateBackgroundColorFromMode:
    """Test background color detection from histogram mode."""

    def test_background_detection_uniform_image(self, uniform_background_image):
        """Test background detection on uniform white background."""
        bg_color, confidence = BackgroundCorrectionUtils.calculate_background_color_from_mode(
            uniform_background_image
        )

        # Should detect the uniform background value
        assert bg_color is not None
        assert confidence > 0.8  # High confidence for uniform background
        assert 14500 < bg_color < 15500  # Near the fixture value of 15000

    def test_background_detection_with_tissue(self, synthetic_raw_image):
        """Test background detection on image with tissue and background."""
        bg_color, confidence = BackgroundCorrectionUtils.calculate_background_color_from_mode(
            synthetic_raw_image
        )

        assert bg_color is not None
        assert np.isfinite(bg_color)
        assert 0 <= confidence <= 1.0

    def test_background_detection_returns_tuple(self, uniform_background_image):
        """Test that function returns (color, confidence) tuple."""
        result = BackgroundCorrectionUtils.calculate_background_color_from_mode(
            uniform_background_image
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        bg_color, confidence = result
        assert isinstance(bg_color, (int, float, np.number))
        assert isinstance(confidence, (float, np.floating))

    def test_background_detection_low_confidence_for_varied_image(self):
        """Test that confidence is low for images without clear background."""
        # Image with high variation, no clear background mode
        varied_image = _RNG.integers(0, 65535, (256, 256), dtype=np.uint16)

        bg_color, confidence = BackgroundCorrectionUtils.calculate_background_color_from_mode(
            varied_image
        )

        # Confidence should be lower for varied images
        assert confidence < 0.9


class TestApplyFlatFieldCorrection:
    """Test flat-field correction (background correction)."""

    def test_divide_method_basic(self, synthetic_raw_image, synthetic_background_image):
        """Test basic divide method for flat-field correction."""
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            synthetic_background_image,
            method="divide",
            scaling_factor=1.0,
        )

        # Check output properties
        assert corrected.shape == synthetic_raw_image.shape
        assert corrected.dtype == synthetic_raw_image.dtype

        # Corrected image should have values in valid range
        assert np.all(corrected >= 0)
        assert np.all(corrected <= 65535)  # uint16 max

    def test_subtract_method_basic(self, synthetic_raw_image, synthetic_background_image):
        """Test basic subtract method for background correction."""
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            synthetic_background_image,
            method="subtract",
            scaling_factor=1.0,
        )

        assert corrected.shape == synthetic_raw_image.shape
        assert corrected.dtype == synthetic_raw_image.dtype

        # Should not have negative values (clipped to 0)
        assert np.all(corrected >= 0)

    def test_divide_removes_vignetting(self, synthetic_raw_image, synthetic_background_image):
        """Test that divide method reduces vignetting effect."""
        # Apply correction
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            synthetic_background_image,
            method="divide",
            scaling_factor=1.0,
        )

        # Vignetting causes lower values at edges
        # After correction, center-to-edge variance should be reduced

        # Compare center vs edge variance before and after
        center_roi = synthetic_raw_image[128:384, 128:384]
        edge_roi = synthetic_raw_image[0:100, 0:100]

        center_corrected = corrected[128:384, 128:384]
        edge_corrected = corrected[0:100, 0:100]

        # Center and edge should be more similar after correction
        before_diff = abs(np.mean(center_roi) - np.mean(edge_roi))
        after_diff = abs(np.mean(center_corrected) - np.mean(edge_corrected))

        assert after_diff < before_diff

    def test_scaling_factor_effect(self, synthetic_raw_image, synthetic_background_image):
        """Test that scaling factor affects output intensity."""
        corrected_1x = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            synthetic_background_image,
            method="divide",
            scaling_factor=1.0,
        )

        corrected_2x = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            synthetic_background_image,
            method="divide",
            scaling_factor=2.0,
        )

        # 2x scaling should produce brighter result
        mean_1x = np.mean(corrected_1x)
        mean_2x = np.mean(corrected_2x)

        assert mean_2x > mean_1x

    def test_epsilon_prevents_division_by_zero(self, synthetic_raw_image):
        """Test that epsilon parameter prevents division by zero."""
        # Background with some zero values
        zero_background = np.ones((512, 512), dtype=np.uint16)
        zero_background[100:200, 100:200] = 0  # Some zero region

        # Should not crash or produce inf/nan
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            synthetic_raw_image,
            zero_background,
            method="divide",
            scaling_factor=1.0,
            epsilon=1.0,
        )

        assert np.all(np.isfinite(corrected))
        assert not np.any(np.isinf(corrected))

    def test_shape_mismatch_handling(self, synthetic_raw_image):
        """Test handling of transposed or mismatched background dimensions."""
        # Transposed background (common issue)
        background_transposed = (
            np.ones(
                (synthetic_raw_image.shape[1], synthetic_raw_image.shape[0]),
                dtype=np.uint16,
            )
            * 15000
        )

        # Should handle transpose automatically or raise clear error
        try:
            corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
                synthetic_raw_image,
                background_transposed,
                method="divide",
                scaling_factor=1.0,
            )
            # If it succeeds, check output is valid
            assert corrected.shape == synthetic_raw_image.shape
        except (ValueError, AssertionError) as e:
            # Should raise clear error about shape mismatch
            assert "shape" in str(e).lower() or "dimension" in str(e).lower()


class TestApplyFlatFieldCorrectionEdgeCases:
    """Test edge cases for flat-field correction."""

    def test_uint8_images(self):
        """Test correction with uint8 images."""
        raw_uint8 = _RNG.integers(50, 200, (256, 256), dtype=np.uint8)
        bg_uint8 = _RNG.integers(100, 250, (256, 256), dtype=np.uint8)

        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            raw_uint8, bg_uint8, method="divide", scaling_factor=1.0
        )

        assert corrected.dtype == np.uint8
        assert np.all(corrected >= 0) and np.all(corrected <= 255)

    def test_identical_raw_and_background(self):
        """Test when raw and background are identical."""
        image = _RNG.integers(5000, 15000, (256, 256), dtype=np.uint16)

        # Divide method should produce near-uniform result
        corrected_divide = BackgroundCorrectionUtils.apply_flat_field_correction(
            image, image.copy(), method="divide", scaling_factor=1.0
        )

        # Should be close to scaling_factor * mean
        assert np.std(corrected_divide) < np.std(image)

        # Subtract method with identical images and scaling_factor=1.0 returns original
        # Formula: img - (bg - bg*scale) = img - (bg - bg*1.0) = img - 0 = img
        corrected_subtract = BackgroundCorrectionUtils.apply_flat_field_correction(
            image, image.copy(), method="subtract", scaling_factor=1.0
        )

        # Should return approximately the same image
        assert np.allclose(corrected_subtract, image, rtol=0.01)

    def test_background_brighter_than_raw(self):
        """Test when background is brighter than raw (normal case)."""
        raw = np.full((256, 256), 10000, dtype=np.uint16)
        background = np.full((256, 256), 15000, dtype=np.uint16)

        # This is the expected scenario
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            raw, background, method="divide", scaling_factor=1.0
        )

        assert np.all(np.isfinite(corrected))

    def test_background_dimmer_than_raw(self):
        """Test when background is unexpectedly dimmer than raw."""
        raw = np.full((256, 256), 15000, dtype=np.uint16)
        background = np.full((256, 256), 10000, dtype=np.uint16)

        # Unusual but should handle gracefully
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            raw, background, method="divide", scaling_factor=1.0
        )

        # Should not crash, may clip to max value
        assert np.all(np.isfinite(corrected))
        assert np.all(corrected <= 65535)

    def test_clipping_to_valid_range(self):
        """Test that output is clipped to valid range for dtype."""
        raw = np.full((256, 256), 30000, dtype=np.uint16)
        background = np.full((256, 256), 10000, dtype=np.uint16)

        # With large scaling factor, might exceed uint16 max
        corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
            raw, background, method="divide", scaling_factor=5.0
        )

        # Should clip to uint16 max
        assert np.all(corrected <= 65535)


class TestValidateBackgroundImages:
    """Test background image validation function."""

    def test_validate_with_all_present(self, tmp_path):
        """Test validation when all required background images are present."""
        # Create the on-disk layout the validator expects: <modality>/<angle>/background.tif
        bg_dir = tmp_path / "backgrounds"
        modality_dir = bg_dir / "ppm_20x"

        for angle in [0, 45, 90, 135]:
            angle_dir = modality_dir / str(angle)
            angle_dir.mkdir(parents=True)
            (angle_dir / "background.tif").touch()

        is_valid, missing = BackgroundCorrectionUtils.validate_background_images(
            bg_dir, modality="ppm_20x", required_angles=[0, 45, 90, 135]
        )
        assert is_valid is True
        assert missing == []

    def test_validate_with_missing_angle(self, tmp_path):
        """Test validation when a required background angle is missing."""
        bg_dir = tmp_path / "backgrounds"
        modality_dir = bg_dir / "ppm_20x"

        # Create only some background images (missing 135 degree)
        for angle in [0, 45, 90]:
            angle_dir = modality_dir / str(angle)
            angle_dir.mkdir(parents=True)
            (angle_dir / "background.tif").touch()

        is_valid, missing = BackgroundCorrectionUtils.validate_background_images(
            bg_dir, modality="ppm_20x", required_angles=[0, 45, 90, 135]
        )
        # Should report the missing angle
        assert is_valid is False
        assert 135 in missing

    def test_validate_with_missing_modality_directory(self, tmp_path):
        """Test validation when entire modality directory is missing."""
        bg_dir = tmp_path / "backgrounds"
        bg_dir.mkdir(parents=True)

        # Don't create the modality subdirectory
        is_valid, missing = BackgroundCorrectionUtils.validate_background_images(
            bg_dir, modality="ppm_20x", required_angles=[0, 45, 90, 135]
        )
        # A missing modality directory means all angles are missing
        assert is_valid is False
        assert missing == [0, 45, 90, 135]


class TestGetModalityFromScanType:
    """Modality key from a scan type: the run index is dropped, nothing else."""

    def test_standard_scan_type_format(self):
        get = BackgroundCorrectionUtils.get_modality_from_scan_type
        assert get("PPM_10x_1") == "PPM_10x"
        assert get("PPM_20x_2") == "PPM_20x"
        assert get("Brightfield_40x_1") == "Brightfield_40x"

    def test_modality_without_run_index_is_unchanged(self):
        assert BackgroundCorrectionUtils.get_modality_from_scan_type("ppm_40x") == "ppm_40x"

    def test_malformed_scan_type_is_returned_as_is(self):
        get = BackgroundCorrectionUtils.get_modality_from_scan_type
        assert get("invalid") == "invalid"
        # No magnification in the second part: not a modality key, returned unchanged.
        assert get("PPM_ten_1") == "PPM_ten_1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
