"""Tests for the streaming multi-dimensional OME-TIFF writer.

These tests target the Task #1 scope: construction, happy-path writes for
all three granularities, abort semantics, streaming (out-of-order frame
arrival), and OME-XML correctness.

ASCII-only per project policy.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from microscope_imageprocessing.io.ome_writer import StackWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _frame(y: int, x: int, value: int, dtype=np.uint16) -> np.ndarray:
    """Construct a frame whose pixels encode a scalar value for later checks."""
    arr = np.full((y, x), fill_value=value, dtype=dtype)
    return arr


def _ome_xml_from_file(path: Path) -> str:
    """Return the OME-XML ImageDescription from the first IFD of path."""
    with tifffile.TiffFile(str(path)) as tif:
        return tif.pages[0].description


# ---------------------------------------------------------------------------
# 2D round-trip
# ---------------------------------------------------------------------------
class Test2DRoundTrip:
    """StackWriter with size_t=size_z=size_c=1 acts like a single 2D writer."""

    def test_roundtrip_single_granularity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sample.ome.tiff"
            frame = _frame(32, 48, value=1234, dtype=np.uint16)

            with StackWriter(
                out,
                size_t=1,
                size_z=1,
                size_c=1,
                size_y=32,
                size_x=48,
                dtype=np.uint16,
                pixel_size_um=0.5,
                channel_names=["BF"],
                granularity="single",
            ) as w:
                w.write_frame(frame, t=0, z=0, c=0)

            assert out.exists()
            with tifffile.TiffFile(str(out)) as tif:
                assert len(tif.pages) == 1
                page = tif.pages[0]
                assert page.shape == (32, 48)
                data = page.asarray()
                np.testing.assert_array_equal(data, frame)

            xml = _ome_xml_from_file(out)
            assert 'SizeX="48"' in xml
            assert 'SizeY="32"' in xml
            assert 'DimensionOrder="XYZCT"' in xml
            assert 'PhysicalSizeX="0.5"' in xml
            assert 'PhysicalSizeY="0.5"' in xml

    def test_per_timepoint_with_single_t_still_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stem"
            with StackWriter(
                out,
                size_t=1,
                size_z=1,
                size_c=1,
                size_y=16,
                size_x=16,
                dtype=np.uint8,
                pixel_size_um=1.0,
                channel_names=["BF"],
                granularity="per_timepoint",
            ) as w:
                w.write_frame(_frame(16, 16, 5, np.uint8), t=0, z=0, c=0)

            produced = list(Path(tmp).glob("*.ome.tiff"))
            assert len(produced) == 1
            assert produced[0].name == "stem_t0000.ome.tiff"


# ---------------------------------------------------------------------------
# Multi-dim synthetic stack
# ---------------------------------------------------------------------------
class TestMultiDimStack:
    """Write a (T=2, Z=3, C=2, Y=10, X=10) stack and verify layout."""

    def test_twelve_frames_round_trip_single_file(self):
        T, Z, C, Y, X = 2, 3, 2, 10, 10
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cube.ome.tiff"
            written = {}
            with StackWriter(
                out,
                size_t=T,
                size_z=Z,
                size_c=C,
                size_y=Y,
                size_x=X,
                dtype=np.uint16,
                pixel_size_um=0.325,
                z_step_um=1.0,
                time_increment_s=15.0,
                channel_names=["DAPI", "GFP"],
                granularity="single",
            ) as w:
                for t in range(T):
                    for z in range(Z):
                        for c in range(C):
                            # Encode a unique scalar per (t,z,c) so we can
                            # verify the XYCZT ordering on disk.
                            val = 1 + t * 100 + z * 10 + c
                            f = _frame(Y, X, val, np.uint16)
                            written[(t, z, c)] = val
                            w.write_frame(f, t=t, z=z, c=c)

            with tifffile.TiffFile(str(out)) as tif:
                assert len(tif.pages) == T * Z * C

            xml = _ome_xml_from_file(out)
            assert f'SizeT="{T}"' in xml
            assert f'SizeZ="{Z}"' in xml
            assert f'SizeC="{C}"' in xml
            assert 'DimensionOrder="XYZCT"' in xml
            assert 'PhysicalSizeZ="1.0"' in xml
            assert 'TimeIncrement="15.0"' in xml

            # The writer emits one <TiffData> per IFD with explicit
            # FirstZ/FirstC/FirstT plus a matching <Plane> element with
            # TheZ/TheC/TheT. Both kinds of mapping must be present so any
            # reader (BioFormats, QuPath, Fiji) gets unambiguous IFD->plane
            # assignment regardless of whether it prefers DimensionOrder
            # inference or per-plane mapping.
            assert (
                xml.count("<TiffData") == T * Z * C
            ), f"expected {T * Z * C} TiffData entries, got {xml.count('<TiffData')}"
            assert (
                xml.count("<Plane ") == T * Z * C
            ), f"expected {T * Z * C} Plane entries, got {xml.count('<Plane ')}"

            # Verify each plane's value matches its (TheT, TheC, TheZ)
            # assignment. DimensionOrder="XYZCT" means Z fastest among IFDs,
            # C next, T slowest -- write loop is for t: for c: for z.
            with tifffile.TiffFile(str(out)) as tif:
                plane_idx = 0
                for t in range(T):
                    for c in range(C):
                        for z in range(Z):
                            data = tif.pages[plane_idx].asarray()
                            assert data[0, 0] == written[(t, z, c)], (
                                f"plane {plane_idx} (t={t},c={c},z={z}) "
                                f"expected {written[(t, z, c)]} got {data[0, 0]}"
                            )
                            plane_idx += 1

    def test_per_timepoint_layout_counts_match(self):
        T, Z, C, Y, X = 3, 2, 2, 8, 8
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stack.ome.tiff"
            with StackWriter(
                out,
                size_t=T,
                size_z=Z,
                size_c=C,
                size_y=Y,
                size_x=X,
                dtype=np.uint16,
                pixel_size_um=1.0,
                z_step_um=0.5,
                channel_names=["DAPI", "GFP"],
                granularity="per_timepoint",
            ) as w:
                for t in range(T):
                    for z in range(Z):
                        for c in range(C):
                            w.write_frame(
                                _frame(Y, X, t * 100 + z * 10 + c, np.uint16),
                                t=t,
                                z=z,
                                c=c,
                            )

            per_t_files = sorted(Path(tmp).glob("*_t????.ome.tiff"))
            assert len(per_t_files) == T
            with tifffile.TiffFile(str(per_t_files[0])) as tif:
                assert len(tif.pages) == Z * C
            xml = _ome_xml_from_file(per_t_files[0])
            assert 'SizeT="1"' in xml
            assert f'SizeZ="{Z}"' in xml
            assert f'SizeC="{C}"' in xml


# ---------------------------------------------------------------------------
# Abort semantics
# ---------------------------------------------------------------------------
class TestAbort:
    """Writes 5 of 10 planes then aborts; file is valid with reduced SizeT."""

    def test_partial_abort_single_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "aborted.ome.tiff"
            w = StackWriter(
                out,
                size_t=10,
                size_z=1,
                size_c=1,
                size_y=8,
                size_x=8,
                dtype=np.uint16,
                pixel_size_um=1.0,
                time_increment_s=5.0,
                channel_names=["BF"],
                granularity="single",
            )
            try:
                for t in range(5):
                    w.write_frame(_frame(8, 8, t + 1, np.uint16), t=t, z=0, c=0)
                w.abort()
            finally:
                # abort() should be idempotent and safe; close() must no-op.
                w.close()

            assert out.exists()
            with tifffile.TiffFile(str(out)) as tif:
                # Aborted writer collapses to the dense bounding box over
                # written indices: t in [0, 5) -> SizeT=5, one plane each.
                assert len(tif.pages) == 5

            xml = _ome_xml_from_file(out)
            assert 'SizeT="5"' in xml


# ---------------------------------------------------------------------------
# Streaming / out-of-order arrival
# ---------------------------------------------------------------------------
class TestOutOfOrderArrival:
    """write_frame called out of order must still produce a correct stack."""

    def test_reverse_t_order_then_close(self):
        T, Z, C, Y, X = 2, 1, 1, 4, 4
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "order.ome.tiff"
            with StackWriter(
                out,
                size_t=T,
                size_z=Z,
                size_c=C,
                size_y=Y,
                size_x=X,
                dtype=np.uint8,
                pixel_size_um=1.0,
                channel_names=["BF"],
                granularity="single",
            ) as w:
                # t=1 arrives BEFORE t=0
                w.write_frame(_frame(Y, X, 200, np.uint8), t=1, z=0, c=0)
                w.write_frame(_frame(Y, X, 100, np.uint8), t=0, z=0, c=0)

            with tifffile.TiffFile(str(out)) as tif:
                assert len(tif.pages) == T
                assert tif.pages[0].asarray()[0, 0] == 100
                assert tif.pages[1].asarray()[0, 0] == 200


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------
class TestConstructorValidation:
    def test_rejects_mismatched_channel_names_length(self, tmp_path):
        with pytest.raises(ValueError, match="channel_names length"):
            StackWriter(
                tmp_path / "x.ome.tiff",
                size_t=1,
                size_z=1,
                size_c=2,
                size_y=4,
                size_x=4,
                dtype=np.uint8,
                pixel_size_um=1.0,
                channel_names=["only_one"],
            )

    def test_rejects_unknown_granularity(self, tmp_path):
        with pytest.raises(ValueError, match="granularity"):
            StackWriter(
                tmp_path / "x.ome.tiff",
                size_t=1,
                size_z=1,
                size_c=1,
                size_y=4,
                size_x=4,
                dtype=np.uint8,
                pixel_size_um=1.0,
                channel_names=["BF"],
                granularity="bogus",
            )

    def test_rejects_zero_sizes(self, tmp_path):
        with pytest.raises(ValueError):
            StackWriter(
                tmp_path / "x.ome.tiff",
                size_t=0,
                size_z=1,
                size_c=1,
                size_y=4,
                size_x=4,
                dtype=np.uint8,
                pixel_size_um=1.0,
                channel_names=["BF"],
            )

    def test_write_frame_rejects_bad_indices(self, tmp_path):
        w = StackWriter(
            tmp_path / "x.ome.tiff",
            size_t=1,
            size_z=1,
            size_c=1,
            size_y=4,
            size_x=4,
            dtype=np.uint8,
            pixel_size_um=1.0,
            channel_names=["BF"],
        )
        try:
            with pytest.raises(ValueError, match="t index"):
                w.write_frame(_frame(4, 4, 1, np.uint8), t=5, z=0, c=0)
            with pytest.raises(ValueError, match="frame shape"):
                w.write_frame(_frame(3, 3, 1, np.uint8), t=0, z=0, c=0)
        finally:
            w.close()
