"""OME MapAnnotation support: arbitrary key/value metadata on a written stack.

Some quantities carry handling rules a reader cannot infer from the pixels --
an axial orientation map is the motivating case, since averaging or mirroring
it incorrectly yields a plausible-looking wrong answer rather than an error.
OME has no dedicated attribute for that, so it goes in a MapAnnotation.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import tifffile

from microscope_imageprocessing.io.ome_writer import StackWriter


def _write(tmp_path, **kwargs):
    p = tmp_path / "img.ome.tif"
    w = StackWriter(
        str(p),
        size_t=1,
        size_z=1,
        size_c=1,
        size_y=8,
        size_x=8,
        dtype=np.dtype("float32"),
        pixel_size_um=0.1715,
        channel_names=["Retardance (nm)"],
        granularity="single",
        bigtiff=False,
        **kwargs,
    )
    try:
        w.write_frame(np.zeros((8, 8), np.float32), t=0, z=0, c=0)
    finally:
        w.close()
    return p


def _xml(path):
    desc = tifffile.TiffFile(str(path)).pages[0].tags["ImageDescription"].value
    root = ET.fromstring(desc)
    return root, {"o": root.tag.split("}")[0].strip("{")}


def test_annotations_round_trip(tmp_path):
    p = _write(tmp_path, map_annotations={"a.units": "radians", "a.axial": "true"})
    root, ns = _xml(p)
    got = {m.get("K"): m.text for m in root.findall(".//o:M", ns)}
    assert got == {"a.units": "radians", "a.axial": "true"}


def test_values_are_stringified(tmp_path):
    """Callers pass floats and ints; OME-XML only holds text."""
    p = _write(tmp_path, map_annotations={"swing": 0.03, "n": 5})
    root, ns = _xml(p)
    got = {m.get("K"): m.text for m in root.findall(".//o:M", ns)}
    assert got == {"swing": "0.03", "n": "5"}


def test_none_values_are_dropped_not_written_as_the_string_none(tmp_path):
    p = _write(tmp_path, map_annotations={"present": "yes", "absent": None})
    root, ns = _xml(p)
    got = {m.get("K"): m.text for m in root.findall(".//o:M", ns)}
    assert got == {"present": "yes"}


def test_image_references_the_annotation(tmp_path):
    """An unreferenced MapAnnotation is not attached to anything."""
    p = _write(tmp_path, map_annotations={"k": "v"})
    root, ns = _xml(p)
    ref = root.find(".//o:AnnotationRef", ns).get("ID")
    assert ref == root.find(".//o:MapAnnotation", ns).get("ID")


@pytest.mark.parametrize("kwargs", [{}, {"map_annotations": None}, {"map_annotations": {}}])
def test_omitting_annotations_changes_nothing(tmp_path, kwargs):
    """Additive feature: existing callers must see identical output."""
    p = _write(tmp_path, **kwargs)
    root, ns = _xml(p)
    assert root.find(".//o:StructuredAnnotations", ns) is None
    assert root.find(".//o:AnnotationRef", ns) is None


def test_pixels_still_read_back(tmp_path):
    p = _write(tmp_path, map_annotations={"k": "v"})
    assert tifffile.imread(str(p)).shape == (8, 8)
