"""
Background and flat-field correction utilities.

This module contains utilities for background correction and flat-field
correction with modality support for microscopy imaging.
"""

import pathlib
from typing import List, Tuple, Dict, Optional
import numpy as np
import tifffile as tf
import logging

logger = logging.getLogger(__name__)


class BackgroundCorrectionUtils:
    """Utilities for background and flat-field correction with modality support."""

    @staticmethod
    def calculate_background_color_from_mode(
        image: np.ndarray, bin_size: int = 5, mode_tolerance: int = 10
    ) -> Tuple[np.ndarray, float]:
        """
        Find background color using histogram mode approach.

        Args:
            image: RGB image array (HxWx3) or grayscale image (HxW)
            bin_size: Size of histogram bins (smaller = more precise but noisier)
            mode_tolerance: Pixels within this range of mode are considered background

        Returns:
            Tuple of (background_mean, confidence_score)
            For RGB images, background_mean is array of shape (3,)
            For grayscale images, background_mean is a scalar
        """
        # Check if image is RGB or grayscale
        is_rgb = image.ndim == 3

        # Convert to grayscale for mode detection
        if is_rgb:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            # For grayscale uint16 images, normalize to uint8 range for histogram
            if image.dtype == np.uint16:
                gray = (image / 256).astype(np.uint8)
            else:
                gray = image.astype(np.uint8)

        # Calculate histogram with reasonable bins
        hist, bins = np.histogram(gray, bins=256 // bin_size, range=(0, 255))

        # Find the mode (highest peak)
        mode_idx = np.argmax(hist)
        mode_value = (bins[mode_idx] + bins[mode_idx + 1]) / 2

        # Find pixels near the mode (likely background)
        background_mask = np.abs(gray - mode_value) < mode_tolerance

        # Calculate confidence (what fraction of image is background)
        confidence = np.sum(background_mask) / gray.size

        # Get background pixel values
        if is_rgb:
            background_pixels = image[background_mask]
            if len(background_pixels) > 0:
                background_mean = background_pixels.mean(axis=0)
            else:
                # Fallback to image mean if no background found
                background_mean = image.mean(axis=(0, 1))
                confidence = 0.0
        else:
            # For grayscale images
            if np.sum(background_mask) > 0:
                background_mean = image[background_mask].mean()
            else:
                # Fallback to image mean if no background found
                background_mean = image.mean()
                confidence = 0.0

        return background_mean, confidence

    @staticmethod
    def get_modality_from_scan_type(scan_type: str) -> str:
        """
        Extract modality identifier from scan type.

        The scan type format is: ScanType_Objectivex_acquisitionnumber
        We need to extract: ScanType_Objectivex

        Examples:
        - "PPM_10x_1" -> "PPM_10x"
        - "PPM_40x_2" -> "PPM_40x"
        - "CAMM_20x_1" -> "CAMM_20x"
        - "PPM_4x_3" -> "PPM_4x"
        - "ppm_40x" -> "ppm_40x" (already modality, no acquisition number)

        Args:
            scan_type: Full scan type string

        Returns:
            Modality string combining scan type and objective
        """
        parts = scan_type.split("_")

        if len(parts) >= 3:
            # Full format: ScanType_Objective_AcquisitionNumber (e.g., "PPM_40x_1")
            scan_type_part = parts[0]
            objective_part = parts[1]

            if "x" in objective_part.lower():
                modality = f"{scan_type_part}_{objective_part}"
                return modality
            else:
                logger.warning(f"Unexpected scan type format: {scan_type}")
                return scan_type
        elif len(parts) == 2:
            # Two-part format: ScanType_Objective (e.g., "ppm_40x")
            # Already a modality string without acquisition number
            scan_type_part = parts[0]
            objective_part = parts[1]

            if "x" in objective_part.lower():
                return scan_type  # Already in modality format
            else:
                logger.warning(f"Unexpected scan type format: {scan_type}")
                return scan_type
        else:
            logger.warning(f"Scan type has unexpected number of parts: {scan_type}")
            return scan_type

    @staticmethod
    def load_background_images(
        background_dir: pathlib.Path,
        angles: List[float],
        logger=None,
        modality: Optional[str] = None,
    ) -> Tuple[Dict[float, np.ndarray], Dict[float, float], Dict[float, List[float]]]:
        """
        Load background images and calculate consistent scaling factors for each angle.

        Supports multiple directory structures:
        - Direct angle files: background_dir/angle.tif
        - Angle subdirectories: background_dir/angle/background.tif
        - Modality subdirectories: background_dir/modality/angle/background.tif

        Args:
            background_dir: Directory containing background images
            angles: List of angles to load backgrounds for
            logger: Optional logger instance
            modality: Optional modality subdirectory name

        Returns:
            Tuple of (background_images_dict, scaling_factors_dict, white_balance_dict)
        """
        backgrounds = {}
        scaling_factors = {}
        white_balance_coeffs = {}

        if logger:
            logger.info(f"Loading background images from: {background_dir}")
            if modality:
                logger.info(f"Using modality subdirectory: {modality}")

        # Determine search directory
        search_dir = background_dir / modality if modality else background_dir

        if modality and not search_dir.exists():
            if logger:
                logger.error(f"Modality directory not found: {search_dir}")
            return backgrounds, scaling_factors, white_balance_coeffs

        # Non-rotation modalities (brightfield, widefield fluorescence) pass
        # an empty angles list. Treat this as a single background at angle 0
        # so the search loop below finds `0.0.tif` / `background.tif`. Also
        # auto-descend into a single WB-mode subdirectory if the caller
        # pointed at the base path instead of the wb subfolder -- the Java
        # resolver and the Python save path can disagree when
        # background_settings.yml is missing from the wb subdir.
        if not angles:
            angles = [0.0]
            if logger:
                logger.info("No angles specified -- loading single non-rotation background at angle 0")

            # If no .tif files live directly in search_dir, look for a single
            # child directory (typically "off" for brightfield/fluorescence
            # or a WB mode like "simple") that contains .tif files.
            try:
                direct_tifs = list(search_dir.glob("*.tif"))
                if not direct_tifs and search_dir.is_dir():
                    subdirs_with_tifs = [
                        d for d in search_dir.iterdir()
                        if d.is_dir() and any(d.glob("*.tif"))
                    ]
                    if len(subdirs_with_tifs) == 1:
                        if logger:
                            logger.info(
                                "No .tif in %s; descending into single WB subdirectory: %s",
                                search_dir, subdirs_with_tifs[0].name,
                            )
                        search_dir = subdirs_with_tifs[0]
                    elif len(subdirs_with_tifs) > 1:
                        if logger:
                            logger.warning(
                                "Multiple WB subdirectories found in %s (%s); "
                                "cannot auto-select. Caller should point directly at the WB subdir.",
                                search_dir, [d.name for d in subdirs_with_tifs],
                            )
            except Exception as e:
                if logger:
                    logger.debug("Error scanning for WB subdirectories: %s", e)

        for angle in angles:
            background_found = False
            attempted_paths = []
            background_file = None

            # Search order: most specific to most general.
            # Rotation modalities (PPM) use per-angle files (e.g. 7.0.tif).
            # Non-rotation modalities (brightfield, fluorescence, monochrome)
            # pass angle=0.0 as a sentinel and use background.tif -- this is
            # the preferred name because 0.0.tif is confusing when the
            # modality has no polarization/rotation angles.
            search_paths = [
                # Direct angle file (per-angle, PPM)
                search_dir / f"{angle}.tif",
                # Angle subdirectory (legacy)
                search_dir / str(angle) / "background.tif",
                # Non-rotation single background file (BF/fluorescence/mono)
                search_dir / "background.tif",
            ]

            for path in search_paths:
                attempted_paths.append(str(path))
                if path.exists():
                    background_found = True
                    background_file = path
                    break

            if background_found:
                try:
                    background_img = tf.imread(str(background_file))
                    backgrounds[angle] = background_img

                    if logger:
                        logger.info(f"  [OK] Loaded background for {angle} deg: {background_file}")

                    # Calculate the scaling factor for this background
                    bg_float = background_img.astype(np.float32)
                    bg_mean_all = bg_float.mean()

                    # No intensity scaling - flat-field formula (img * bg_mean/bg_pixel)
                    # already preserves overall brightness while correcting spatial variation.
                    # Previous 240-target scaling for 90 deg caused tiling artifacts by pushing
                    # corrected edge pixels near saturation.
                    scaling_factor = 1.0
                    if logger:
                        logger.info(
                            f"    Background mean intensity at {angle} deg: {bg_mean_all:.1f}"
                        )

                    scaling_factors[angle] = scaling_factor

                    # Calculate white balance from background
                    bg_mean = background_img.mean(axis=(0, 1))  # Mean R,G,B
                    if len(bg_mean) >= 3 and bg_mean[1] > 0:  # Check G channel
                        # Calculate relative gains to normalize to green channel
                        white_balance_coeffs[angle] = [
                            bg_mean[1] / bg_mean[0] if bg_mean[0] > 0 else 1.0,  # R correction
                            1.0,  # G reference
                            bg_mean[1] / bg_mean[2] if bg_mean[2] > 0 else 1.0,  # B correction
                        ]
                        if logger:
                            logger.info(
                                f"    White balance coeffs for {angle} deg: {white_balance_coeffs[angle]}"
                            )
                    else:
                        white_balance_coeffs[angle] = [1.0, 1.0, 1.0]

                except Exception as e:
                    if logger:
                        logger.error(f"  [ERROR] Failed to load background for {angle} deg: {e}")
            else:
                if logger:
                    logger.warning(f"  [FAIL] Background not found for {angle} deg")
                    logger.warning(f"    Searched paths:")
                    for path in attempted_paths:
                        logger.warning(f"      - {path}")

        if logger and backgrounds:
            logger.info(f"Successfully loaded {len(backgrounds)}/{len(angles)} background images")
        elif logger:
            logger.warning("No background images were loaded")

        return backgrounds, scaling_factors, white_balance_coeffs

    @staticmethod
    def validate_background_images(
        background_dir: pathlib.Path, modality: str, required_angles: List[float], logger=None
    ) -> Tuple[bool, List[float]]:
        """
        Validate that all required background images exist for a specific modality.

        Returns:
            (is_valid, missing_angles) tuple
        """
        missing_angles = []
        modality_dir = background_dir / modality

        if not modality_dir.exists():
            if logger:
                logger.error(f"Modality directory not found: {modality_dir}")
            return False, required_angles

        for angle in required_angles:
            angle_dir = modality_dir / str(angle)
            background_file = angle_dir / "background.tif"

            if not background_file.exists():
                missing_angles.append(angle)

        is_valid = len(missing_angles) == 0

        if not is_valid and logger:
            logger.error(f"Missing background images for {modality} angles: {missing_angles}")

        return is_valid, missing_angles

    @staticmethod
    def apply_flat_field_correction(
        image: np.ndarray,
        background: np.ndarray,
        scaling_factor: float,
        method: str = "divide",
        epsilon: float = 0.1,
    ) -> np.ndarray:
        """
        Apply flat-field correction with pre-calculated scaling for consistency.

        The scaling_factor is calculated once per background image and applied
        to all tiles to ensure seamless stitching.

        Args:
            image: Image to correct
            background: Background image
            scaling_factor: Pre-calculated scaling factor
            method: Correction method ('divide' or 'subtract')
            epsilon: Small value to prevent division by zero (default: 0.1)

        Returns:
            Corrected image
        """
        # Ensure float precision for calculations
        img_float = image.astype(np.float32)
        bg_float = background.astype(np.float32)

        # Track if we need to transpose the result back to original image shape
        original_shape = image.shape
        transposed_image = False

        # Handle shape mismatch - background may be (C,H,W) while image is (H,W,C)
        if img_float.shape != bg_float.shape:
            if len(bg_float.shape) == 3 and len(img_float.shape) == 3:
                # Check if we need to transpose from (C,H,W) to (H,W,C)
                if (
                    bg_float.shape[0] == img_float.shape[2]
                    and bg_float.shape[1:] == img_float.shape[:2]
                ):
                    bg_float = np.transpose(bg_float, (1, 2, 0))  # (C,H,W) -> (H,W,C)
                # Or check if we need to transpose from (H,W,C) to (C,H,W)
                elif (
                    bg_float.shape[:2] == img_float.shape[1:]
                    and bg_float.shape[2] == img_float.shape[0]
                ):
                    img_float = np.transpose(img_float, (2, 0, 1))  # (H,W,C) -> (C,H,W)
                    transposed_image = True

        # Ensure shapes match after potential transpose
        if img_float.shape != bg_float.shape:
            raise ValueError(
                f"Shape mismatch after transpose attempt: image {img_float.shape} vs background {bg_float.shape}"
            )

        # Prevent division by zero with small epsilon
        bg_float = np.where(bg_float < epsilon, epsilon, bg_float)

        if method == "divide":
            # Proper flatfield correction: normalize by background mean to preserve brightness
            bg_mean = bg_float.mean()

            # The correction formula: Image * (background_mean / background_pixel)
            # This normalizes illumination while preserving overall brightness
            corrected = img_float * (bg_mean / bg_float)

            # Apply the scaling factor for consistency across tiles
            corrected = corrected * scaling_factor

            # For debugging: check if background has illumination variation
            bg_std = bg_float.std()
            bg_variation = bg_std / bg_mean
            if bg_variation < 0.02:  # Less than 2% variation
                # Background is very uniform - flat-field correction will have
                # minimal effect. Note: bright/uncrossed angles (e.g. 90deg)
                # naturally have low variation (typically 3-5%) and this is
                # expected -- the previous 5% threshold produced false positives.
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Background image has low variation (std/mean = {bg_variation:.3f}) - may not correct illumination properly"
                )

        elif method == "subtract":
            # For subtraction, we need a different approach
            # Scale the background to match the image intensity range
            bg_scaled = bg_float * scaling_factor
            corrected = img_float - (bg_float - bg_scaled)
            corrected = np.maximum(corrected, 0)  # No negative values

        # Final clipping to valid range
        if image.dtype == np.uint8:
            max_val = 255
        elif image.dtype == np.uint16:
            max_val = 65535
        else:
            max_val = 255  # Default for unknown types

        # Clip to valid range
        corrected = np.clip(corrected, 0, max_val).astype(image.dtype)

        # Transpose back to original shape if we transposed the image
        if transposed_image:
            corrected = np.transpose(corrected, (1, 2, 0))  # (C,H,W) -> (H,W,C)

        return corrected
