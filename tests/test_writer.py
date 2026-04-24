"""Tests for OME-TIFF writer."""

import hashlib
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

import microscope_imageprocessing
from microscope_imageprocessing.io.ome_writer import StackWriter
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


class TestOmeTiffWriterIsStackWriterAlias:
    """The 2D adapter is a thin shim over StackWriter.

    These tests verify the adapter produces output that is byte-identical
    to a direct StackWriter call with matching parameters (t=z=c=1,
    granularity='single', bigtiff=False, legacy description override).
    """

    @staticmethod
    def _direct_stack_writer_equivalent(
        path, pixel_size_um, data, compression=None
    ):
        """Direct StackWriter call matching the 2D adapter's construction."""
        arr = np.asarray(data)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            photometric = "rgb"
        else:
            photometric = "minisblack"
        description = (
            f"microscope_imageprocessing={microscope_imageprocessing.__version__}"
            f" python={platform.python_version()}"
        )
        w = StackWriter(
            path,
            size_t=1,
            size_z=1,
            size_c=1,
            size_y=int(arr.shape[0]),
            size_x=int(arr.shape[1]),
            dtype=arr.dtype,
            pixel_size_um=pixel_size_um,
            channel_names=["image"],
            granularity="single",
            bigtiff=False,
            compression=compression.lower() if compression is not None else None,
            photometric=photometric,
            description_override=description,
        )
        try:
            w.write_frame(arr, t=0, z=0, c=0)
        finally:
            w.close()

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: np.random.RandomState(1).randint(0, 65535, (32, 48), dtype=np.uint16),
            lambda: np.random.RandomState(2).randint(0, 255, (32, 48, 3), dtype=np.uint8),
            lambda: np.random.RandomState(3).randint(0, 255, (16, 24), dtype=np.uint8),
        ],
        ids=["uint16-2d", "uint8-rgb", "uint8-2d"],
    )
    def test_byte_identical_to_direct_stackwriter(self, factory):
        data = factory()
        with tempfile.TemporaryDirectory() as tmp:
            p_adapter = Path(tmp) / "adapter.tif"
            p_direct = Path(tmp) / "direct.tif"
            ome_tiff_writer(str(p_adapter), pixel_size_um=0.5, data=data)
            self._direct_stack_writer_equivalent(
                str(p_direct), pixel_size_um=0.5, data=data
            )
            sha_adapter = hashlib.sha256(p_adapter.read_bytes()).hexdigest()
            sha_direct = hashlib.sha256(p_direct.read_bytes()).hexdigest()
            assert sha_adapter == sha_direct, (
                f"adapter SHA {sha_adapter} != direct SHA {sha_direct}"
            )

    def test_byte_identical_with_compression(self):
        data = np.random.RandomState(4).randint(0, 255, (32, 48, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            p_adapter = Path(tmp) / "adapter.tif"
            p_direct = Path(tmp) / "direct.tif"
            ome_tiff_writer(str(p_adapter), pixel_size_um=0.5, data=data, compression="zlib")
            self._direct_stack_writer_equivalent(
                str(p_direct), pixel_size_um=0.5, data=data, compression="zlib"
            )
            assert p_adapter.read_bytes() == p_direct.read_bytes()

    def test_legacy_description_preserved(self):
        """Primary ImageDescription is the legacy version string, NOT
        OME-XML -- downstream tooling that parsed the legacy string
        continues to work unchanged."""
        data = np.zeros((16, 24), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            p = f.name
        try:
            ome_tiff_writer(p, pixel_size_um=1.0, data=data)
            with tifffile.TiffFile(p) as tf:
                desc = tf.pages[0].tags["ImageDescription"].value
            assert desc.startswith("microscope_imageprocessing=")
            assert "python=" in desc
            assert "<OME" not in desc
        finally:
            os.unlink(p)

    def test_physical_size_metadata_preserved(self):
        """XResolution / YResolution / ResolutionUnit must match the
        pre-refactor writer (1e4/pixel_size_um in CENTIMETER)."""
        data = np.zeros((8, 8), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            p = f.name
        try:
            ome_tiff_writer(p, pixel_size_um=0.5, data=data)
            with tifffile.TiffFile(p) as tf:
                pg = tf.pages[0]
                xres = pg.tags["XResolution"].value
                yres = pg.tags["YResolution"].value
                unit = pg.tags["ResolutionUnit"].value
            assert xres == (20000, 1)
            assert yres == (20000, 1)
            # RESUNIT.CENTIMETER = 3
            assert int(unit) == 3
        finally:
            os.unlink(p)

    def test_rgb_uses_rgb_photometric(self):
        data = np.random.RandomState(5).randint(0, 255, (8, 12, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            p = f.name
        try:
            ome_tiff_writer(p, pixel_size_um=1.0, data=data)
            with tifffile.TiffFile(p) as tf:
                pg = tf.pages[0]
                # PHOTOMETRIC.RGB = 2
                assert int(pg.photometric) == 2
                assert pg.samplesperpixel == 3
                # PLANARCONFIG.CONTIG = 1
                assert int(pg.planarconfig) == 1
        finally:
            os.unlink(p)

    def test_grayscale_uses_minisblack(self):
        data = np.zeros((8, 12), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            p = f.name
        try:
            ome_tiff_writer(p, pixel_size_um=1.0, data=data)
            with tifffile.TiffFile(p) as tf:
                pg = tf.pages[0]
                # PHOTOMETRIC.MINISBLACK = 1
                assert int(pg.photometric) == 1
                assert pg.samplesperpixel == 1
        finally:
            os.unlink(p)
