"""Streaming multi-dimensional OME-TIFF writer.

Owned by the Z-stack + time-lapse refactor. Replaces the per-frame OME-TIFF
writes that lose multi-dimensional (T, Z, C) structure with a single
``StackWriter`` class that accepts 2D frames as they are acquired and emits a
proper OME-XML description on close.

ASCII-only per project policy: this module runs on Windows cp1252 as well as
Linux/WSL. Do not use Unicode characters (arrows, Greek letters, deg sign) in
code, logging, or comments.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.etree import ElementTree as ET

import numpy as np
import tifffile as tf

import microscope_imageprocessing

logger = logging.getLogger(__name__)


# Allowed granularity values accepted by the constructor.
_GRANULARITY_SINGLE = "single"
_GRANULARITY_PER_T = "per_timepoint"
_GRANULARITY_PER_C_FOLDER = "per_channel_folder"
_VALID_GRANULARITIES = (
    _GRANULARITY_SINGLE,
    _GRANULARITY_PER_T,
    _GRANULARITY_PER_C_FOLDER,
)


# OME-XML namespace constants (kept local so this module has no extra deps).
_OME_XMLNS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_OME_XSI = "http://www.w3.org/2001/XMLSchema-instance"
_OME_SCHEMA_LOC = (
    "http://www.openmicroscopy.org/Schemas/OME/2016-06 "
    "http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd"
)


class StackWriter:
    """Thread-safe streaming OME-TIFF writer for multi-dimensional stacks.

    Accepts 2D frames one at a time via :meth:`write_frame` and assembles a
    canonical ``(T, Z, C, Y, X)`` OME-TIFF on :meth:`close`. Frames may arrive
    out of order along any of the T, Z, C axes -- they are stored in an
    internal plane-indexed buffer and serialized in XYCZT DimensionOrder on
    close.

    Granularity modes:
      * ``single`` - write a single OME-TIFF at ``output_path``.
      * ``per_timepoint`` - ``output_path`` is treated as a stem; one file per
        timepoint is written as ``<stem>_t{NNNN}.ome.tiff``.
      * ``per_channel_folder`` - ``output_path`` is treated as a folder stem;
        one subfolder per channel is created and files within each folder
        follow the per-timepoint naming.

    The writer is safe to call from multiple threads; an internal
    ``threading.Lock`` serializes buffer mutations and file operations.

    Cancellation: call :meth:`abort` at any time to finalize whatever planes
    have been written so far. The resulting file is a valid OME-TIFF whose
    ``SizeT`` / ``SizeZ`` / ``SizeC`` reflect only the planes that were
    actually written.

    Note: Task #1 scope lands this class as a well-typed shell with a working
    happy-path implementation exercised by unit tests. Downstream teams will
    hook this into the acquisition loop in later tasks.
    """

    def __init__(
        self,
        output_path: os.PathLike,
        *,
        size_t: int,
        size_z: int,
        size_c: int,
        size_y: int,
        size_x: int,
        dtype: np.dtype,
        pixel_size_um: float,
        z_step_um: Optional[float] = None,
        time_increment_s: Optional[float] = None,
        channel_names: Sequence[str],
        channel_metadata: Optional[Sequence[Dict[str, Any]]] = None,
        granularity: str = _GRANULARITY_PER_T,
        bigtiff: bool = True,
        compression: Optional[str] = None,
        software_tag: str = "QPSC",
        photometric: Optional[str] = "minisblack",
        description_override: Optional[str] = None,
    ) -> None:
        if size_t < 1 or size_z < 1 or size_c < 1:
            raise ValueError(
                "size_t, size_z, size_c must all be >= 1 "
                f"(got T={size_t}, Z={size_z}, C={size_c})"
            )
        if size_y < 1 or size_x < 1:
            raise ValueError(
                f"size_y and size_x must be >= 1 (got Y={size_y}, X={size_x})"
            )
        if len(channel_names) != size_c:
            raise ValueError(
                f"channel_names length {len(channel_names)} does not match "
                f"size_c {size_c}"
            )
        if granularity not in _VALID_GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {_VALID_GRANULARITIES}; got {granularity!r}"
            )

        self.output_path = Path(output_path)
        self.size_t = int(size_t)
        self.size_z = int(size_z)
        self.size_c = int(size_c)
        self.size_y = int(size_y)
        self.size_x = int(size_x)
        self.dtype = np.dtype(dtype)
        self.pixel_size_um = float(pixel_size_um)
        self.z_step_um = float(z_step_um) if z_step_um is not None else None
        self.time_increment_s = (
            float(time_increment_s) if time_increment_s is not None else None
        )
        self.channel_names: List[str] = list(channel_names)
        self.channel_metadata: List[Dict[str, Any]] = (
            [dict(m) for m in channel_metadata]
            if channel_metadata is not None
            else [{} for _ in range(self.size_c)]
        )
        if len(self.channel_metadata) != self.size_c:
            raise ValueError(
                f"channel_metadata length {len(self.channel_metadata)} does not "
                f"match size_c {self.size_c}"
            )
        self.granularity = granularity
        self.bigtiff = bool(bigtiff)
        self.compression = compression
        self.software_tag = software_tag
        # `photometric=None` lets tifffile auto-detect (e.g. RGB for
        # (Y, X, 3) uint8 frames), matching the legacy 2D adapter's
        # `photometric="rgb" if ndim==3 else "minisblack"` dispatch.
        # `description_override` replaces the generated OME-XML in the
        # first IFD; the 2D adapter uses this to preserve its byte layout
        # for existing callers.
        self.photometric = photometric
        self.description_override = description_override

        self._lock = threading.Lock()
        self._closed = False
        self._aborted = False

        # Plane buffers keyed by (t, z, c). We buffer all frames in memory and
        # serialize on close. The per-tile memory cost is acceptable for the
        # acquisition sizes QPSC produces (single-point, Z+T stacks). Larger
        # grid-tiled acquisitions use per-tile files whose writers are
        # short-lived.
        self._frames: Dict[tuple, np.ndarray] = {}
        self._plane_metadata: Dict[tuple, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "StackWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # An exception in the caller's block -- finalize whatever we have
            # and propagate.
            try:
                self.abort()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("StackWriter.abort() failed during __exit__: %s", e)
            return None
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write_frame(
        self,
        image: np.ndarray,
        *,
        t: int,
        z: int,
        c: int,
        plane_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Buffer a single 2D frame at indices (t, z, c).

        Calls from multiple threads are serialized internally. The frame is
        validated for shape and dtype before being stored. Later calls with
        the same (t, z, c) overwrite earlier ones.
        """
        if image is None:
            raise ValueError("image must not be None")
        if not (0 <= t < self.size_t):
            raise ValueError(f"t index {t} out of range [0, {self.size_t})")
        if not (0 <= z < self.size_z):
            raise ValueError(f"z index {z} out of range [0, {self.size_z})")
        if not (0 <= c < self.size_c):
            raise ValueError(f"c index {c} out of range [0, {self.size_c})")

        arr = np.asarray(image)
        if arr.shape[:2] != (self.size_y, self.size_x):
            raise ValueError(
                f"frame shape {arr.shape} does not match declared "
                f"(size_y, size_x) = ({self.size_y}, {self.size_x})"
            )

        with self._lock:
            if self._closed:
                raise RuntimeError("StackWriter is already closed")
            if self._aborted:
                raise RuntimeError("StackWriter has been aborted; cannot write")
            self._frames[(t, z, c)] = arr
            if plane_metadata is not None:
                self._plane_metadata[(t, z, c)] = dict(plane_metadata)

    def close(self) -> None:
        """Finalize the OME-TIFF file(s) and release resources."""
        with self._lock:
            if self._closed:
                return
            try:
                self._finalize(aborted=False)
            finally:
                self._closed = True

    def abort(self) -> None:
        """Finalize whatever planes have been written and mark aborted.

        The resulting OME-TIFF is a valid file whose OME-XML SizeT/Z/C
        reflect only the set of written planes (collapsed to a dense bounding
        box over the written indices).
        """
        with self._lock:
            if self._closed:
                return
            try:
                self._aborted = True
                self._finalize(aborted=True)
            finally:
                self._closed = True

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def _effective_dims(self) -> tuple:
        """Return (T, Z, C) reflecting either the declared sizes or the
        dense bounding box over written planes when aborted.
        """
        if not self._aborted:
            return self.size_t, self.size_z, self.size_c

        if not self._frames:
            return 1, 1, 1

        ts = {k[0] for k in self._frames}
        zs = {k[1] for k in self._frames}
        cs = {k[2] for k in self._frames}
        # Collapse to a [0, max+1) range -- OME-XML requires contiguous
        # dimensions. Any missing planes within that range are filled with
        # zeros at write time (see _planes_for_t).
        return max(ts) + 1, max(zs) + 1, max(cs) + 1

    def _finalize(self, *, aborted: bool) -> None:
        eff_t, eff_z, eff_c = self._effective_dims()
        if aborted and not self._frames:
            logger.info(
                "StackWriter.abort() called with no frames written; skipping file write"
            )
            return

        if self.granularity == _GRANULARITY_SINGLE:
            self._write_single_file(eff_t, eff_z, eff_c)
        elif self.granularity == _GRANULARITY_PER_T:
            self._write_per_timepoint(eff_t, eff_z, eff_c)
        elif self.granularity == _GRANULARITY_PER_C_FOLDER:
            self._write_per_channel_folder(eff_t, eff_z, eff_c)
        else:  # pragma: no cover - constructor validates
            raise ValueError(f"Unknown granularity: {self.granularity}")

    def _resolve_single_path(self) -> Path:
        """Return a concrete file path for the `single` granularity."""
        p = self.output_path
        if p.suffix == "":
            # Treat as stem.
            return p.with_name(p.name + ".ome.tiff")
        return p

    def _resolve_per_timepoint_path(self, t_idx: int) -> Path:
        """Return the file path for timepoint t_idx under `per_timepoint`."""
        p = self.output_path
        if p.suffix:
            # Drop the suffix chain (.ome.tiff handled conservatively).
            stem_name = p.name
            for ext in (".tiff", ".tif"):
                if stem_name.endswith(ext):
                    stem_name = stem_name[: -len(ext)]
                    break
            if stem_name.endswith(".ome"):
                stem_name = stem_name[: -len(".ome")]
            stem = p.with_name(stem_name)
        else:
            stem = p
        return stem.with_name(f"{stem.name}_t{t_idx:04d}.ome.tiff")

    def _resolve_per_channel_folder_path(self, c_idx: int, t_idx: int) -> Path:
        """Return the file path for (channel=c_idx, timepoint=t_idx)
        under `per_channel_folder`.
        """
        root = self.output_path
        chan = self.channel_names[c_idx] if c_idx < len(self.channel_names) else f"c{c_idx}"
        chan_folder = root / _sanitize_folder(chan)
        return chan_folder / f"{root.name}_{_sanitize_folder(chan)}_t{t_idx:04d}.ome.tiff"

    # Writers for each granularity -----------------------------------------
    def _write_single_file(self, eff_t: int, eff_z: int, eff_c: int) -> None:
        path = self._resolve_single_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        ome_xml = self._build_ome_xml(
            file_count=1,
            files_size_t=eff_t,
            eff_t=eff_t,
            eff_z=eff_z,
            eff_c=eff_c,
            file_index=0,
            t_offset=0,
        )
        self._write_file(path, eff_t, eff_z, eff_c, t_offset=0, ome_xml=ome_xml)

    def _write_per_timepoint(self, eff_t: int, eff_z: int, eff_c: int) -> None:
        for t in range(eff_t):
            path = self._resolve_per_timepoint_path(t)
            path.parent.mkdir(parents=True, exist_ok=True)
            ome_xml = self._build_ome_xml(
                file_count=eff_t,
                files_size_t=1,
                eff_t=eff_t,
                eff_z=eff_z,
                eff_c=eff_c,
                file_index=t,
                t_offset=t,
            )
            # Write frames for this single timepoint only.
            self._write_file(path, 1, eff_z, eff_c, t_offset=t, ome_xml=ome_xml)

    def _write_per_channel_folder(self, eff_t: int, eff_z: int, eff_c: int) -> None:
        # One subfolder per channel. Within each folder use per-timepoint
        # layout (safe default per the plan).
        for c in range(eff_c):
            for t in range(eff_t):
                path = self._resolve_per_channel_folder_path(c, t)
                path.parent.mkdir(parents=True, exist_ok=True)
                ome_xml = self._build_ome_xml(
                    file_count=eff_t,
                    files_size_t=1,
                    eff_t=eff_t,
                    eff_z=eff_z,
                    eff_c=1,
                    file_index=t,
                    t_offset=t,
                    c_filter=c,
                )
                self._write_file(
                    path,
                    1,
                    eff_z,
                    1,
                    t_offset=t,
                    c_filter=c,
                    ome_xml=ome_xml,
                )

    def _write_file(
        self,
        path: Path,
        size_t_in_file: int,
        size_z_in_file: int,
        size_c_in_file: int,
        *,
        t_offset: int,
        c_filter: Optional[int] = None,
        ome_xml: str,
    ) -> None:
        """Stream planes for a single OME-TIFF file in XYCZT plane order."""
        options: Dict[str, Any] = {
            "resolutionunit": "CENTIMETER",
        }
        if self.photometric is not None:
            options["photometric"] = self.photometric
        if self.compression is not None:
            options["compression"] = self.compression

        # The 2D adapter sets description_override AND short-circuits
        # bigtiff to stay inside the classic TIFF limit; honor both.
        with tf.TiffWriter(str(path), bigtiff=self.bigtiff) as tif:
            plane_idx = 0
            total_planes = size_t_in_file * size_z_in_file * size_c_in_file
            for local_t in range(size_t_in_file):
                global_t = t_offset + local_t
                for local_c in range(size_c_in_file):
                    global_c = c_filter if c_filter is not None else local_c
                    for local_z in range(size_z_in_file):
                        frame = self._frames.get((global_t, local_z, global_c))
                        if frame is None:
                            # Missing plane (aborted write). Fill zeros so the
                            # OME-XML plane count stays consistent.
                            frame = np.zeros(
                                (self.size_y, self.size_x), dtype=self.dtype
                            )
                        per_plane_opts = dict(options)
                        if plane_idx == 0:
                            per_plane_opts["description"] = (
                                self.description_override
                                if self.description_override is not None
                                else ome_xml
                            )
                            per_plane_opts["metadata"] = None
                        tif.write(
                            frame.astype(self.dtype, copy=False),
                            contiguous=False,
                            resolution=(
                                1e4 / self.pixel_size_um,
                                1e4 / self.pixel_size_um,
                            ),
                            **per_plane_opts,
                        )
                        plane_idx += 1
            logger.debug(
                "StackWriter wrote %d/%d planes to %s",
                plane_idx,
                total_planes,
                path,
            )

    # ------------------------------------------------------------------
    # OME-XML assembly
    # ------------------------------------------------------------------
    def _build_ome_xml(
        self,
        *,
        file_count: int,
        files_size_t: int,
        eff_t: int,
        eff_z: int,
        eff_c: int,
        file_index: int,
        t_offset: int,
        c_filter: Optional[int] = None,
    ) -> str:
        """Produce an OME-XML description for one file.

        file_count / files_size_t describe the chosen layout so we can emit
        per-file SizeT correctly (1 for per_timepoint, eff_t for single).
        """
        ome = ET.Element(
            "OME",
            {
                "xmlns": _OME_XMLNS,
                "xmlns:xsi": _OME_XSI,
                "xsi:schemaLocation": _OME_SCHEMA_LOC,
                "UUID": f"urn:uuid:{uuid.uuid4()}",
                "Creator": (
                    f"{self.software_tag} "
                    f"microscope_imageprocessing={microscope_imageprocessing.__version__} "
                    f"python={platform.python_version()}"
                ),
            },
        )

        image_id = f"Image:{file_index}"
        image = ET.SubElement(ome, "Image", {"ID": image_id, "Name": image_id})

        pix_size_t = files_size_t
        pix_size_c = 1 if c_filter is not None else eff_c

        # DimensionOrder reflects the physical IFD layout produced by
        # _write_file (T outer, C middle, Z inner) so readers that ignore
        # TiffData and fall back to DimensionOrder still map IFDs correctly.
        pixels_attrs = {
            "ID": f"Pixels:{file_index}",
            "DimensionOrder": "XYZCT",
            "Type": _ome_pixel_type(self.dtype),
            "SizeX": str(self.size_x),
            "SizeY": str(self.size_y),
            "SizeZ": str(eff_z),
            "SizeC": str(pix_size_c),
            "SizeT": str(pix_size_t),
            "PhysicalSizeX": str(self.pixel_size_um),
            "PhysicalSizeY": str(self.pixel_size_um),
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeYUnit": "um",
        }
        if self.z_step_um is not None and eff_z > 1:
            pixels_attrs["PhysicalSizeZ"] = str(self.z_step_um)
            pixels_attrs["PhysicalSizeZUnit"] = "um"
        if self.time_increment_s is not None and pix_size_t > 1:
            pixels_attrs["TimeIncrement"] = str(self.time_increment_s)
            pixels_attrs["TimeIncrementUnit"] = "s"
        pixels = ET.SubElement(image, "Pixels", pixels_attrs)

        # Channel entries.
        channel_range = [c_filter] if c_filter is not None else range(eff_c)
        for local_idx, c_idx in enumerate(channel_range):
            chan_attrs = {
                "ID": f"Channel:{file_index}:{local_idx}",
                "Name": self.channel_names[c_idx]
                if c_idx < len(self.channel_names)
                else f"Channel {c_idx}",
                "SamplesPerPixel": "1",
            }
            meta = (
                self.channel_metadata[c_idx]
                if c_idx < len(self.channel_metadata)
                else {}
            )
            for xml_attr, meta_key in (
                ("ExcitationWavelength", "excitation_nm"),
                ("EmissionWavelength", "emission_nm"),
                ("IlluminationType", "illumination_type"),
                ("Color", "color"),
            ):
                if meta_key in meta and meta[meta_key] is not None:
                    chan_attrs[xml_attr] = str(meta[meta_key])
            ET.SubElement(pixels, "Channel", chan_attrs)

        # TiffData elements map each IFD in the TIFF to its (T, C, Z) slot.
        # OME-TIFF REQUIRES these for multi-dim stacks -- BioFormats (what
        # QuPath uses) falls back to DimensionOrder only when TiffData is
        # absent, which silently collapses Z-stacks in some readers.
        # Iteration order here MUST match _write_file's loop order so IFD N
        # corresponds to the Nth TiffData entry.
        ifd_index = 0
        channel_count = 1 if c_filter is not None else eff_c
        for local_t in range(pix_size_t):
            for local_c in range(channel_count):
                for z_idx in range(eff_z):
                    ET.SubElement(
                        pixels,
                        "TiffData",
                        {
                            "FirstT": str(local_t),
                            "FirstC": str(local_c),
                            "FirstZ": str(z_idx),
                            "IFD": str(ifd_index),
                            "PlaneCount": "1",
                        },
                    )
                    ifd_index += 1

        # Per-plane metadata (optional supplementary info per OME spec).
        for local_t in range(pix_size_t):
            global_t = t_offset + local_t
            for local_c, c_idx in enumerate(channel_range):
                global_c = c_idx
                for z_idx in range(eff_z):
                    plane_meta = self._plane_metadata.get((global_t, z_idx, global_c), {})
                    plane_attrs = {
                        "TheT": str(local_t),
                        "TheC": str(local_c),
                        "TheZ": str(z_idx),
                    }
                    for xml_attr, meta_key, unit_attr, unit_val in (
                        ("DeltaT", "delta_t_s", "DeltaTUnit", "s"),
                        ("ExposureTime", "exposure_s", "ExposureTimeUnit", "s"),
                        ("PositionX", "position_x_um", "PositionXUnit", "um"),
                        ("PositionY", "position_y_um", "PositionYUnit", "um"),
                        ("PositionZ", "position_z_um", "PositionZUnit", "um"),
                    ):
                        if meta_key in plane_meta and plane_meta[meta_key] is not None:
                            plane_attrs[xml_attr] = str(plane_meta[meta_key])
                            plane_attrs[unit_attr] = unit_val
                    ET.SubElement(pixels, "Plane", plane_attrs)

        # TIFF UUIDs are optional and not required for tifffile readback.
        ET.register_namespace("", _OME_XMLNS)
        xml_bytes = ET.tostring(ome, encoding="unicode")
        return xml_bytes


def _ome_pixel_type(dtype: np.dtype) -> str:
    """Map a numpy dtype to an OME-XML PixelType string."""
    dt = np.dtype(dtype)
    mapping = {
        np.dtype(np.uint8): "uint8",
        np.dtype(np.int8): "int8",
        np.dtype(np.uint16): "uint16",
        np.dtype(np.int16): "int16",
        np.dtype(np.uint32): "uint32",
        np.dtype(np.int32): "int32",
        np.dtype(np.float32): "float",
        np.dtype(np.float64): "double",
    }
    if dt in mapping:
        return mapping[dt]
    raise ValueError(f"Unsupported OME PixelType for numpy dtype {dt}")


def _sanitize_folder(name: str) -> str:
    """ASCII-only filesystem-safe folder name."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    result = "".join(out).strip("._-")
    return result or "channel"
