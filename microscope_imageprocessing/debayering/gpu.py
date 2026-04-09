import cupy as cp
import numpy as np

class GPUDebayer:
    """GPU-accelerated Bayer pattern demosaicing using CuPy"""

    def __init__(self, pattern='RGGB'):
        """Initialize with Bayer pattern type"""
        self.pattern = pattern
        self._setup_masks()

    def _setup_masks(self):
        """Create color channel masks for the Bayer pattern"""
        # Define pattern positions (0,0 is top-left)
        patterns = {
            'RGGB': {'R': (0,0), 'G': [(0,1), (1,0)], 'B': (1,1)},
            'GRBG': {'R': (0,1), 'G': [(0,0), (1,1)], 'B': (1,0)},
            'GBRG': {'R': (1,0), 'G': [(0,0), (1,1)], 'B': (0,1)},
            'BGGR': {'R': (1,1), 'G': [(0,1), (1,0)], 'B': (0,0)}
        }
        self.masks = patterns[self.pattern]

    def debayer(self, bayer_img):
        """Perform GPU debayering using bilinear interpolation"""
        # Transfer to GPU
        img = cp.asarray(bayer_img, dtype=cp.float32)
        h, w = img.shape

        # Initialize RGB output
        rgb = cp.zeros((h, w, 3), dtype=cp.float32)

        # Extract color channels using masks
        r_mask = cp.zeros((h, w), dtype=bool)
        g_mask = cp.zeros((h, w), dtype=bool)
        b_mask = cp.zeros((h, w), dtype=bool)

        # Red channel
        r_y, r_x = self.masks['R']
        r_mask[r_y::2, r_x::2] = True

        # Green channel (two positions)
        for g_y, g_x in self.masks['G']:
            g_mask[g_y::2, g_x::2] = True

        # Blue channel
        b_y, b_x = self.masks['B']
        b_mask[b_y::2, b_x::2] = True

        # Apply masks
        rgb[:,:,0][r_mask] = img[r_mask]
        rgb[:,:,1][g_mask] = img[g_mask]
        rgb[:,:,2][b_mask] = img[b_mask]

        # Bilinear interpolation using convolution
        kernels = {
            'r': cp.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=cp.float32) / 4,
            'g': cp.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=cp.float32) / 4,
            'b': cp.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=cp.float32) / 4
        }

        # Interpolate missing values
        from cupyx.scipy import ndimage

        rgb[:,:,0] = ndimage.convolve(rgb[:,:,0], kernels['r'], mode='reflect')
        rgb[:,:,1] = ndimage.convolve(rgb[:,:,1], kernels['g'], mode='reflect')
        rgb[:,:,2] = ndimage.convolve(rgb[:,:,2], kernels['b'], mode='reflect')

        # Restore original values where they exist
        rgb[:,:,0][r_mask] = img[r_mask]
        rgb[:,:,1][g_mask] = img[g_mask]
        rgb[:,:,2][b_mask] = img[b_mask]

        # Clip values and convert back to CPU, preserving input dtype
        max_val = cp.iinfo(cp.asarray(bayer_img).dtype).max if bayer_img.dtype.kind == 'u' else 255
        rgb = cp.clip(rgb, 0, max_val)
        return cp.asnumpy(rgb).astype(bayer_img.dtype)

# Usage example
if __name__ == "__main__":
    # Create synthetic Bayer image for testing
    bayer = np.random.randint(0, 256, (2000, 2000), dtype=np.uint8)

    # Initialize debayer
    debayer = GPUDebayer(pattern='RGGB')

    # Process image
    rgb_image = debayer.debayer(bayer)
    print(f"Output shape: {rgb_image.shape}")  # (2000, 2000, 3)
