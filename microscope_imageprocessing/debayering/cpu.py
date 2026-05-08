import numpy as np
from scipy import ndimage
from multiprocessing import Pool
import os
import tifffile


class CPUDebayer:
    """CPU-optimized Bayer pattern demosaicing using bilinear interpolation."""

    def __init__(
        self,
        pattern: str = "RGGB",
        image_bit_clipmax: int = 65535,
        image_dtype: type = np.uint16,
        convolution_mode: str = "reflect",
    ) -> None:
        """Initialize debayering with a Bayer pattern configuration.

        Args:
            pattern: Bayer filter pattern. One of 'RGGB', 'GRBG', 'GBRG', 'BGGR'.
            image_bit_clipmax: Maximum pixel value for clipping (default 65535 for 16-bit).
            image_dtype: Output data type for 16-bit images (default np.uint16).
            convolution_mode: Edge handling mode for scipy.ndimage.convolve (default 'reflect').
        """
        self.pattern = pattern
        self.image_bit_clipmax = image_bit_clipmax
        self.image_dtype = image_dtype
        self.convolution_mode = convolution_mode
        self._setup_masks()

    def _setup_masks(self):
        """Create color channel masks for the Bayer pattern"""
        patterns = {
            "RGGB": {"R": (0, 0), "G": [(0, 1), (1, 0)], "B": (1, 1)},
            "GRBG": {"R": (0, 1), "G": [(0, 0), (1, 1)], "B": (1, 0)},
            "GBRG": {"R": (1, 0), "G": [(0, 0), (1, 1)], "B": (0, 1)},
            "BGGR": {"R": (1, 1), "G": [(0, 1), (1, 0)], "B": (0, 0)},
        }
        self.masks = patterns[self.pattern]

    def debayer(self, bayer_img: np.ndarray) -> np.ndarray:
        """Perform CPU debayering using bilinear interpolation.

        Preserves input dtype - uint8 input produces uint8 output, uint16 input
        produces uint16 output. Works internally in float32 for precision, then
        converts back to input dtype.

        Args:
            bayer_img: 2D single-channel Bayer pattern image (H, W).

        Returns:
            RGB image as (H, W, 3) array in the same dtype as input.
        """
        # Store input dtype to preserve it in output
        input_dtype = bayer_img.dtype

        # Determine appropriate clipping max based on input dtype
        if input_dtype == np.uint8:
            clip_max = 255
        else:
            clip_max = self.image_bit_clipmax

        img = bayer_img.astype(np.float32)
        h, w = img.shape

        # Initialize RGB output
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        # Extract channels directly using slicing (faster than masks)
        r_y, r_x = self.masks["R"]
        rgb[r_y::2, r_x::2, 0] = img[r_y::2, r_x::2]

        for g_y, g_x in self.masks["G"]:
            rgb[g_y::2, g_x::2, 1] = img[g_y::2, g_x::2]

        b_y, b_x = self.masks["B"]
        rgb[b_y::2, b_x::2, 2] = img[b_y::2, b_x::2]

        # Bilinear interpolation kernels
        kernels = {
            "r": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4,
            "g": np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 4,
            "b": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4,
        }

        # Interpolate each channel
        for i, kernel in enumerate([kernels["r"], kernels["g"], kernels["b"]]):
            rgb[:, :, i] = ndimage.convolve(rgb[:, :, i], kernel, mode=self.convolution_mode)

        # Restore original values
        rgb[r_y::2, r_x::2, 0] = img[r_y::2, r_x::2]
        for g_y, g_x in self.masks["G"]:
            rgb[g_y::2, g_x::2, 1] = img[g_y::2, g_x::2]
        rgb[b_y::2, b_x::2, 2] = img[b_y::2, b_x::2]

        # Clip and convert back to input dtype
        return np.clip(rgb, 0, clip_max).astype(input_dtype)


def process_image(filepath: str, pattern: str = "GRBG") -> str:
    """Debayer a single TIFF image file and save as numpy array.

    Args:
        filepath: Path to a raw Bayer pattern TIFF file.
        pattern: Bayer filter pattern (default 'GRBG').

    Returns:
        Path to the saved RGB numpy file (.npy).
    """
    # filepath, pattern = args
    # Load image (assuming raw bayer data)
    image = tifffile.imread(filepath)
    assert image.ndim == 2
    # bayer = np.fromfile(filepath, dtype=np.uint8).reshape(2000, 2000)

    # Process
    debayer = CPUDebayer(pattern=pattern)
    rgb = debayer.debayer(image)

    # Save result
    output_path = filepath.replace(".raw", "_rgb.npy")
    np.save(output_path, rgb)
    return output_path


def process_data(image: np.ndarray, pattern: str = "GRBG") -> np.ndarray:
    """Debayer an in-memory Bayer pattern image.

    Args:
        image: 2D single-channel Bayer pattern image.
        pattern: Bayer filter pattern (default 'GRBG').

    Returns:
        RGB image as (H, W, 3) array.
    """
    assert image.ndim == 2
    debayer = CPUDebayer(pattern=pattern)
    rgb = debayer.debayer(image)
    return rgb


def batch_process(file_list: list, pattern: str = "GRBG", n_workers: int = None) -> list:
    """Debayer multiple images in parallel using multiprocessing.

    Args:
        file_list: List of paths to raw Bayer pattern TIFF files.
        pattern: Bayer filter pattern (default 'GRBG').
        n_workers: Number of parallel workers (default: CPU count).

    Returns:
        List of paths to saved RGB numpy files.
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    args = [(f, pattern) for f in file_list]

    with Pool(n_workers) as pool:
        results = pool.map(process_image, args)

    return results


# Usage example
if __name__ == "__main__":
    # Single image
    # bayer = np.random.randint(0, 256, (2000, 2000), dtype=np.uint8)
    # debayer = CPUDebayer(pattern="RGGB")
    # rgb = debayer.debayer(bayer)
    pass
    # Batch processing
    # files = ['image1.raw', 'image2.raw', ...]
    # results = batch_process(files, pattern='RGGB', n_workers=8)
