"""Tests for OME-TIFF writer."""

import tempfile
import os
import numpy as np
import tifffile
from microscope_imageprocessing.io.writer import ome_tiff_writer


class TestOmeTiffWriter:
    """Test OME-TIFF writing with metadata."""

    def test_write_rgb(self, rgb_image_uint8):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            ome_tiff_writer(path, pixel_size_um=0.5, data=rgb_image_uint8)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            # Read back and verify
            loaded = tifffile.imread(path)
            assert loaded.shape == rgb_image_uint8.shape
            np.testing.assert_array_equal(loaded, rgb_image_uint8)
        finally:
            os.unlink(path)

    def test_write_grayscale_uint16(self):
        data = np.random.randint(0, 65535, (32, 32), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            ome_tiff_writer(path, pixel_size_um=1.0, data=data)
            loaded = tifffile.imread(path)
            np.testing.assert_array_equal(loaded, data)
        finally:
            os.unlink(path)

    def test_write_with_compression(self, rgb_image_uint8):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            ome_tiff_writer(path, pixel_size_um=0.5, data=rgb_image_uint8, compression="zlib")
            assert os.path.exists(path)
            loaded = tifffile.imread(path)
            np.testing.assert_array_equal(loaded, rgb_image_uint8)
        finally:
            os.unlink(path)
