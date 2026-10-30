"""
Flat-field profiles for detectors with no usable background image.

A laser scanner or a PMT shades its field just as a camera does, but an empty field reads only the
detector's offset, so the measured-background route in ``background.py`` has nothing to divide by.
BaSiC (Peng et al. 2017, Nat Commun 8:14836) estimates the shading from the acquired tiles instead:
across many fields the sample moves and the shading does not.

The model here is ``raw = offset + flatfield * signal``. The offset is one number, subtracted before
the fit and added back after correction, so the background and existing thresholds keep their
values. Dividing raw values directly would scale the offset as well; on the OWS3 SHG data the offset
(about 6,500 counts) is larger than most of the signal, so that would paint the shading onto the background.

A fit produces a :class:`FlatfieldProfile`, saved as ``flatfield.tif`` and ``profile.yml``. The
profile belongs to a detector, objective, zoom and channel, not to a run, so it also corrects data it
cannot be fitted from, such as a single-position Z-stack or time-lapse.

``_fit_shading`` is a NumPy port of the "approximate" fitting mode of BaSiCPy (MIT license,
Copyright 2022 BaSiCPy developing collaboration group). The package itself needs PyTorch and
``scipy<1.13``, which the microscope server's environment cannot take. Its darkfield and temporal
baseline terms are left out: the offset takes the place of the first, and the second would remove
real changes in signal over a time-lapse.
"""

import argparse
import dataclasses
import datetime
import json
import logging
import pathlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tifffile as tf
import yaml
from scipy.fft import dctn, idctn

logger = logging.getLogger(__name__)

FLATFIELD_FILE = "flatfield.tif"
PROFILE_FILE = "profile.yml"
PREVIEW_FILE = "flatfield_preview.png"
DEFAULT_WORKING_SIZE = 128
DEFAULT_MAX_IMAGES = 500
MM_CHANNEL_TAG = 51123
_MM_CHANNEL_NAME = re.compile(r"img_channel(\d+)_")

# BaSiCPy's defaults for the approximate fit
_EPSILON = 0.1
_RHO = 1.5
_MU_COEF = 12.5
_MAX_MU_COEF = 1e7
_MAX_ITERATIONS = 500
_MAX_REWEIGHT_ITERATIONS = 10
_OPTIMIZATION_TOL = 1e-3
_REWEIGHTING_TOL = 1e-2

# Empty fields hold detector noise, which the optics do not shade, so they pull a fit toward flat:
# on the OWS3 SHG run, fitting every tile recovered 0.83 of a known shading and fitting the brighter
# tiles 0.93. Too few images matter more than that, so all are kept below this count.
_MIN_IMAGES_TO_SELECT = 40


def _shrink(values: np.ndarray, threshold) -> np.ndarray:
    """Soft threshold: pull every value toward zero by ``threshold``."""
    return np.sign(values) * np.clip(np.abs(values) - threshold, 0, None)


def _fit_step(
    images: np.ndarray, weight: np.ndarray, smoothness: float, init_mu: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One weighted decomposition of ``images`` into shading, per-image brightness and residual."""
    image_norm = np.linalg.norm(images)
    max_mu = init_mu * _MAX_MU_COEF
    mu = init_mu

    shading = np.ones(images.shape[1:], dtype=np.float32)
    shading_dct = dctn(shading, norm="ortho")
    brightness = images.mean(axis=(1, 2)) / images.mean()
    residual = np.zeros_like(images)
    dual = np.zeros_like(images)

    for _ in range(_MAX_ITERATIONS):
        shading = idctn(shading_dct, norm="ortho")
        background = shading[None] * brightness[:, None, None]
        update = (images - residual - background + dual / mu).mean(axis=0)
        shading_dct = _shrink(shading_dct + dctn(update, norm="ortho"), smoothness / mu)
        shading = idctn(shading_dct, norm="ortho")

        background = shading[None] * brightness[:, None, None]
        residual = _shrink(residual + (images - background - residual + dual / mu), weight / mu)
        remainder = images - residual
        brightness = np.clip(remainder.mean(axis=(1, 2)) / remainder.mean(), 0, None)
        background = shading[None] * brightness[:, None, None]

        fit_error = images - background - residual
        dual = dual + mu * fit_error
        mu = min(mu * _RHO, max_mu)
        if np.linalg.norm(fit_error) / image_norm <= _OPTIMIZATION_TOL:
            break
    return shading, brightness, residual


def _fit_shading(images: np.ndarray) -> np.ndarray:
    """
    Estimate the smooth shading common to a stack of offset-subtracted images.

    Args:
        images: float32 array (N, H, W), at the working size

    Returns:
        Shading (H, W) normalized to a mean of 1
    """
    mean_image = images.mean(axis=0)
    mean_image = mean_image / mean_image.mean()
    smoothness = np.abs(dctn(mean_image, norm="ortho")).sum() / 400 * 0.5
    spectral_norm = np.linalg.svd(images.reshape(len(images), -1), compute_uv=False)[0]
    init_mu = _MU_COEF / spectral_norm

    weight = np.ones_like(images)
    previous = None
    shading = None
    for _ in range(_MAX_REWEIGHT_ITERATIONS):
        raw_shading, brightness, residual = _fit_step(images, weight, smoothness, init_mu)
        background = raw_shading[None] * brightness[:, None, None]
        shading = background.mean(axis=0)
        if not shading.mean() > 0:  # also catches NaN, from a brightness that collapsed
            raise ValueError(
                "The fit found nothing common to the images: they are too sparse, or the offset "
                "is too high"
            )
        shading = shading / shading.mean()

        relative = residual / (background.mean(axis=(1, 2))[:, None, None] + 1e-6)
        weight = 1.0 / (np.abs(relative) + _EPSILON)
        weight = weight * weight.size / weight.sum()

        if previous is not None:
            change = np.abs(shading - previous).sum() / np.abs(previous).sum()
            if change <= _REWEIGHTING_TOL:
                break
        previous = shading
    return shading


def estimate_offset(images: Sequence[np.ndarray], dark_fraction: float = 0.1) -> float:
    """
    Estimate the detector offset from the darkest images of a set.

    The offset is the level of an empty field: the median of the per-image medians of the darkest
    images. A low percentile of the pixels would be wrong, since it reads the bottom of the noise.
    It needs some empty fields; for confluent tissue, pass a measured offset to :func:`fit_profile`.

    Args:
        images: images from one detector
        dark_fraction: share of the images, darkest first, taken as empty fields

    Returns:
        The offset in raw counts
    """
    medians = np.sort([float(np.median(image)) for image in images])
    return float(np.median(medians[: max(1, int(len(medians) * dark_fraction))]))


def _shrink_to_working_size(image: np.ndarray, working_size: int) -> np.ndarray:
    return cv2.resize(
        image.astype(np.float32), (working_size, working_size), interpolation=cv2.INTER_AREA
    )


@dataclasses.dataclass
class FlatfieldProfile:
    """
    A fitted flat-field: shading normalized to a mean of 1, plus the detector offset.

    Attributes:
        flatfield: float32 array (H, W) with a mean of 1
        offset: detector offset in raw counts
        metadata: what it was fitted from and for; saved in ``profile.yml``
    """

    flatfield: np.ndarray
    offset: float
    metadata: Dict = dataclasses.field(default_factory=dict)

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Correct one raw plane in the detector's own pixel frame, before any averaging, projection,
        flip or rotation.

        Args:
            image: single-channel image, the same size as the profile

        Returns:
            The corrected image in the same dtype, and how many pixels clipped at its range
        """
        if image.shape != self.flatfield.shape:
            raise ValueError(
                f"Image shape {image.shape} does not match the flat-field {self.flatfield.shape}"
            )
        corrected = (image.astype(np.float64) - self.offset) / self.flatfield + self.offset
        if not np.issubdtype(image.dtype, np.integer):
            return corrected.astype(image.dtype), 0
        limits = np.iinfo(image.dtype)
        rounded = np.floor(corrected + 0.5)
        clipped = int(np.count_nonzero((rounded < limits.min) | (rounded > limits.max)))
        return np.clip(rounded, limits.min, limits.max).astype(image.dtype), clipped

    def save(self, folder: pathlib.Path) -> pathlib.Path:
        """Write the profile and a picture of the shading to a folder, creating it if needed."""
        folder = pathlib.Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        tf.imwrite(folder / FLATFIELD_FILE, self.flatfield.astype(np.float32))
        record = dict(self.metadata)
        record.update(
            offset=self.offset,
            width=int(self.flatfield.shape[1]),
            height=int(self.flatfield.shape[0]),
        )
        with open(folder / PROFILE_FILE, "w", encoding="ascii") as stream:
            yaml.safe_dump(record, stream, sort_keys=False)
        low, high = float(self.flatfield.min()), float(self.flatfield.max())
        scaled = (self.flatfield - low) / max(high - low, 1e-6) * 255
        cv2.imwrite(
            str(folder / PREVIEW_FILE),
            cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_VIRIDIS),
        )
        return folder

    @classmethod
    def load(cls, folder: pathlib.Path) -> "FlatfieldProfile":
        """Read a profile written by :meth:`save`."""
        folder = pathlib.Path(folder)
        with open(folder / PROFILE_FILE, encoding="ascii") as stream:
            record = yaml.safe_load(stream)
        flatfield = tf.imread(folder / FLATFIELD_FILE).astype(np.float32)
        return cls(flatfield=flatfield, offset=float(record.pop("offset")), metadata=record)


def fit_profile(
    images: Sequence[np.ndarray],
    offset: Optional[float] = None,
    working_size: int = DEFAULT_WORKING_SIZE,
    metadata: Optional[Dict] = None,
) -> FlatfieldProfile:
    """
    Fit a flat-field profile to images from one detector, objective, zoom and channel.

    Fit on raw planes or averaged frames. A maximum projection reads the offset too high and a sum
    multiplies it, so neither is valid input.

    Args:
        images: at least two single-channel images of the same size, from different fields.
            Dozens with varied content give a usable fit. From 40 images up, the brighter half
            is fitted.
        offset: detector offset in raw counts; estimated from the darkest images when None
        working_size: images are reduced to this size for the fit
        metadata: extra fields for ``profile.yml``

    Returns:
        The fitted profile, at the images' full size
    """
    if len(images) < 2:
        raise ValueError("A flat-field fit needs at least two images")
    height, width = images[0].shape
    if offset is None:
        offset = estimate_offset(images)
    small = np.stack(
        [
            np.clip(_shrink_to_working_size(image, working_size) - offset, 0, None)
            for image in images
        ]
    )
    if small.mean() <= 0:
        raise ValueError(f"No signal above the offset of {offset:.1f}")
    if len(small) >= _MIN_IMAGES_TO_SELECT:
        order = np.argsort(small.mean(axis=(1, 2)))
        small = small[order[len(small) // 2 :]]

    shading = _fit_shading(small)
    flatfield = cv2.resize(shading, (width, height), interpolation=cv2.INTER_LINEAR)
    flatfield = flatfield / flatfield.mean()

    record = dict(metadata or {})
    record.update(
        algorithm="basic-approximate",
        images_used=len(small),
        images_read=len(images),
        working_size=working_size,
        fitted=datetime.datetime.now().isoformat(timespec="seconds"),
        flatfield_min=float(flatfield.min()),
        flatfield_max=float(flatfield.max()),
    )
    return FlatfieldProfile(flatfield=flatfield, offset=float(offset), metadata=record)


def _page_channel(page) -> Optional[int]:
    """The channel of a MicroManager TIFF page, from its per-page metadata tag."""
    tag = page.tags.get(MM_CHANNEL_TAG)
    if tag is None:
        return None
    value = tag.value
    if isinstance(value, (bytes, str)):
        value = json.loads(value)
    channel = value.get("ChannelIndex", value.get("Channel-Index"))
    return None if channel is None else int(channel)


def read_tile_planes(
    folder: pathlib.Path, channel: Optional[int] = None, max_images: int = DEFAULT_MAX_IMAGES
) -> List[np.ndarray]:
    """
    Read raw planes for a fit from a tile folder, whatever its layout.

    Handles QPSC tile folders, MicroManager single-plane series (``img_channel000_...tif``) and
    MMStack files. RGB tiles are skipped: they have measured backgrounds. Files are sampled evenly,
    since a fit needs variety rather than every tile.

    Args:
        folder: searched recursively
        channel: MicroManager channel index; None takes every plane
        max_images: most planes to read

    Returns:
        Single-channel planes, all the size of the first one found
    """
    folder = pathlib.Path(folder)
    files = sorted(
        path
        for pattern in ("*.tif", "*.tiff")
        for path in folder.rglob(pattern)
        if not path.name.endswith(".stitch-info.txt")
    )
    if channel is not None:
        named = [(path, _MM_CHANNEL_NAME.match(path.name)) for path in files]
        files = [path for path, match in named if match is None or int(match.group(1)) == channel]
    if not files:
        raise FileNotFoundError(f"No TIFF files in {folder}")

    picked = np.unique(np.linspace(0, len(files) - 1, min(max_images, len(files))).astype(int))
    planes: List[np.ndarray] = []
    for index in picked:
        with tf.TiffFile(files[index]) as tiff:
            for page in tiff.pages:
                if channel is not None and _page_channel(page) not in (None, channel):
                    continue
                plane = page.asarray()
                if plane.ndim != 2 or (planes and plane.shape != planes[0].shape):
                    continue
                planes.append(plane)
                break
    if not planes:
        raise ValueError(f"No single-channel planes in {folder}")
    return planes


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Fit a profile from a tile folder: ``python -m ...correction.flatfield <tiles> <profile>``."""
    parser = argparse.ArgumentParser(description="Fit a BaSiC flat-field profile from tiles.")
    parser.add_argument("tiles", type=pathlib.Path, help="folder of tiles, searched recursively")
    parser.add_argument("profile", type=pathlib.Path, help="folder to write the profile to")
    parser.add_argument("--channel", type=int, help="MicroManager channel index")
    parser.add_argument("--offset", type=float, help="measured detector offset in raw counts")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    args = parser.parse_args(argv)

    planes = read_tile_planes(args.tiles, args.channel, args.max_images)
    profile = fit_profile(
        planes, offset=args.offset, metadata={"source": str(args.tiles), "channel": args.channel}
    )
    profile.save(args.profile)
    print(
        f"Fitted {profile.metadata['images_used']} of {len(planes)} planes: offset {profile.offset:.1f}, flat-field "
        f"{profile.metadata['flatfield_min']:.3f} to {profile.metadata['flatfield_max']:.3f}"
    )


if __name__ == "__main__":
    main()
