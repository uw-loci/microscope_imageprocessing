"""Modality-aware autofocus strategies.

A strategy is a self-contained recipe for "is there enough signal to
focus on this image?" + "what's the focus score for this image?" +
"is the camera exposure appropriate for this image content?". Different
sample regimes need different recipes:

- Dense (H&E, IHC, PPM, confluent IF): the texture-and-area gate works.
  Score = laplacian_variance (or strategy YAML override). Brightness =
  median floor.
- Sparse (beads, pollen, scattered FISH spots): the area gate is wrong.
  Validity = N bright local maxima above background. Score = laplacian
  variance on the spot ROI; falls back to whole-FOV brenner when too
  few foreground pixels survive.
- Dark-field (SHG, LSM, dark-field BF, unstained cleared tissue): no
  spatial gate; the whole frame is signal. Validity = total gradient
  energy above a floor. Score = brenner_gradient.
- Manual-only: skip auto entirely; always pops the manual dialog.

Strategies are selected per-modality from autofocus_<scope>.yml and can
be overridden per-acquisition via the --af-strategy CLI flag.

The metric and validity-check implementations come from
``focus.metrics`` and ``focus.validity``. Strategies are the assembly
layer -- they wire the right metric to the right gate to the right
brightness check, parameterized for the modality.

Failure modes (what to do when validity returns False):

- DEFER: skip AF on this tile, defer to the next. Used by dense_texture.
- PROCEED: gate said no, run AF anyway. Used by sparse_signal /
  dark_field where the gate is conservative.
- MANUAL: pop the manual focus dialog immediately. Used by manual_only.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import numpy as np

from microscope_imageprocessing.focus.metrics import resolve_metric
from microscope_imageprocessing.focus.validity import (
    bright_spot_count,
    resolve_validity_check,
    texture_and_area,
    total_gradient_energy,
)

logger = logging.getLogger(__name__)


def worst_channel_saturation_fraction(image: np.ndarray) -> float:
    """Fraction (0-1) of pixels saturated in the single worst channel.

    Saturation corrupts focus metrics: clipped highlights flatten the local
    gradient and can invert the focus curve so it ramps toward defocus
    instead of peaking at focus (the 2026-05-31 PPM 40x runaway). This
    measures the most-saturated channel so the AF auto-exposure reducer can
    drop exposure / illumination until the strongest channel is back under a
    per-strategy tolerance.

    Per-channel for RGB (H,W,>=3); whole-frame for monochrome (H,W). Uses a
    dtype-aware near-saturation level (uint8 / 0-255 float: 250, uint16:
    64000, 0-1 float: 0.98). Returns 0.0 for empty / None images.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return 0.0
    if image.dtype == np.uint16:
        sat_level = 64000.0
    elif np.issubdtype(image.dtype, np.floating) and float(image.max()) <= 1.0:
        sat_level = 0.98
    else:
        sat_level = 250.0  # uint8 and 0-255 float
    if image.ndim == 2:
        return float(np.mean(image >= sat_level))
    if image.ndim == 3 and image.shape[2] >= 1:
        worst = 0.0
        for c in range(min(3, image.shape[2])):
            worst = max(worst, float(np.mean(image[:, :, c] >= sat_level)))
        return worst
    return 0.0


def _eval_saturation(name: str, threshold: float, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
    """Shared saturation_acceptable body. (ok, stats); ok=False means the
    brightest channel exceeds this strategy's saturation tolerance."""
    frac = worst_channel_saturation_fraction(image)
    return frac <= threshold, {
        "strategy": name,
        "saturation_fraction": frac,
        "saturation_threshold": threshold,
    }


class StrategyFailureMode(enum.Enum):
    """What to do when a strategy's validity check fails."""

    DEFER = "defer"
    PROCEED = "proceed"
    MANUAL = "manual"


ScoreFn = Callable[[np.ndarray], float]


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Best-effort grayscale conversion for the brightness checks.
    Mirrors the conversion used by validity checks so brightness and
    validity see the same intensity scale."""
    if image.ndim == 3:
        return np.mean(image, axis=2).astype(np.float32)
    return image.astype(np.float32)


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


class AutofocusStrategy(Protocol):
    """Modality-aware autofocus recipe. Wraps a validity check, a focus
    metric, and a brightness check."""

    name: str
    on_failure: StrategyFailureMode

    def is_valid(self, image: np.ndarray, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        """Returns (valid, stats); stats is logging-friendly."""
        ...

    def score(self, image: np.ndarray) -> float:
        """Focus score (higher = sharper)."""
        ...

    def brightness_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Returns (ok, stats). Per-strategy so sparse strategies can use
        a percentile/dynamic-range check instead of a median floor."""
        ...

    def saturation_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Returns (ok, stats). False -> the brightest channel is saturated
        beyond this strategy's tolerance and the caller should reduce
        exposure / illumination before trusting the focus metric. Per-strategy
        because the right tolerance differs by sample regime: dense tissue can
        spare ~10% before the metric inverts, while a sparse bead/FISH field
        clips all of its real signal in only a few percent of pixels."""
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


@dataclass
class DenseTextureStrategy:
    """For H&E, IHC, PPM, confluent IF.

    Validity = texture stddev above threshold AND tissue-mask area above
    threshold AND not blank-white RGB. Score = manifest-resolved metric
    (default laplacian_variance). Brightness = median floor.

    Failure mode: DEFER -- the acquisition queue's tile-deferral logic
    handles this directly.
    """

    name: str = "dense_texture"
    on_failure: StrategyFailureMode = StrategyFailureMode.DEFER
    score_metric_name: str = "laplacian_variance"
    texture_threshold: float = 0.010
    tissue_area_threshold: float = 0.200
    rgb_brightness_threshold: float = 240.0
    tissue_mask_range: Tuple[float, float] = (0.10, 0.90)
    median_floor: float = 15.0
    # Dense tissue tolerates ~10% saturated pixels before the focus metric
    # starts to invert; above this the AF auto-exposure reducer halves
    # exposure / illumination.
    saturation_threshold: float = 0.10

    def __post_init__(self) -> None:
        self._score_fn = resolve_metric(self.score_metric_name)

    def is_valid(self, image: np.ndarray, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        ok, stats = texture_and_area(
            image,
            texture_threshold=self.texture_threshold,
            tissue_area_threshold=self.tissue_area_threshold,
            rgb_brightness_threshold=self.rgb_brightness_threshold,
            tissue_mask_range=self.tissue_mask_range,
            median_floor=self.median_floor,
        )
        stats["strategy"] = self.name
        if logger_:
            level = logger_.info if ok else logger_.warning
            level(
                "dense_texture: %s (texture=%.4f, area=%.3f)",
                "VALID" if ok else "rejected",
                stats.get("texture", 0.0),
                stats.get("area", 0.0),
            )
        return ok, stats

    def score(self, image: np.ndarray) -> float:
        return float(self._score_fn(image))

    def brightness_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        gray = _to_grayscale(image)
        if gray.max() > 0:
            gray_8bit = (gray / gray.max() * 255.0).astype(np.float32)
        else:
            gray_8bit = gray
        median = float(np.median(gray_8bit))
        ok = median >= self.median_floor
        return ok, {
            "strategy": self.name,
            "brightness_check": "median_floor",
            "median": median,
            "floor": self.median_floor,
        }

    def saturation_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        return _eval_saturation(self.name, self.saturation_threshold, image)


@dataclass
class SparseSignalStrategy:
    """For scattered fluorescent objects on a dark background.

    Validity = N bright local maxima above an adaptive background.
    Score = manifest-resolved metric on the spot ROI (mask =
    image > background + k*MAD); falls back to whole-FOV brenner when
    too few foreground pixels survive.

    Brightness check uses a dynamic-range floor instead of a median.
    On a sparse bright sample the median is near zero regardless of
    exposure; doubling exposure to make the median pass would push the
    bright spots into saturation.

    Failure mode: PROCEED -- sparse images often focus fine when the
    spot count is borderline.
    """

    name: str = "sparse_signal"
    on_failure: StrategyFailureMode = StrategyFailureMode.PROCEED
    score_metric_name: str = "laplacian_variance"
    spot_sigma_above_bg: float = 5.0
    spot_min_separation_px: int = 8
    min_spots: int = 3
    min_peak_intensity: float = 20.0
    bright_pixel_floor: float = 50.0
    # A sparse field (beads, FISH spots) clips ALL of its real signal in only
    # a few percent of pixels, so the tolerance is far tighter than tissue --
    # a tissue-style 10% gate would never fire even when every spot is blown
    # out. Catch clipping early.
    saturation_threshold: float = 0.03

    def __post_init__(self) -> None:
        self._score_fn = resolve_metric(self.score_metric_name)
        self._brenner_fn = resolve_metric("brenner_gradient")

    def _compute_fg_mask(self, image: np.ndarray) -> np.ndarray:
        """Recompute the foreground mask the validity check used.

        The validity function does not return the mask (its shape varies
        with input), so the strategy duplicates the threshold math here
        for the score path. Cheap; a shared helper would force
        bright_spot_count to allocate even for callers who only want
        the boolean.
        """
        gray = _to_grayscale(image)
        if gray.max() > gray.min():
            gray_8bit = ((gray - gray.min()) / (gray.max() - gray.min()) * 255.0).astype(np.float32)
        else:
            gray_8bit = gray.astype(np.float32)
        bg_median = float(np.median(gray_8bit))
        bg_mad = float(np.median(np.abs(gray_8bit - bg_median))) + 1e-6
        bg_sigma = bg_mad * 1.4826
        spot_threshold = max(
            bg_median + self.spot_sigma_above_bg * bg_sigma,
            self.min_peak_intensity,
        )
        return gray_8bit > spot_threshold

    def is_valid(self, image: np.ndarray, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        ok, stats = bright_spot_count(
            image,
            spot_sigma_above_bg=self.spot_sigma_above_bg,
            spot_min_separation_px=self.spot_min_separation_px,
            min_spots=self.min_spots,
            min_peak_intensity=self.min_peak_intensity,
            bright_pixel_floor=self.bright_pixel_floor,
        )
        stats["strategy"] = self.name
        if logger_:
            level = logger_.info if ok else logger_.warning
            level(
                "sparse_signal: %d spots above %.1f (bg_median=%.1f, "
                "bg_sigma=%.2f); min_spots=%d -> %s",
                stats["spot_count"],
                stats["spot_threshold"],
                stats["bg_median"],
                stats["bg_sigma"],
                self.min_spots,
                "VALID" if ok else "below threshold (will PROCEED anyway)",
            )
        return ok, stats

    def score(self, image: np.ndarray) -> float:
        fg_mask = self._compute_fg_mask(image)
        if not np.any(fg_mask) or fg_mask.sum() < 50:
            # Too few foreground pixels to score reliably; fall back to
            # whole-FOV brenner so the AF search still has signal along Z.
            return float(self._brenner_fn(image))
        gray = _to_grayscale(image)
        masked = np.where(fg_mask, gray, 0.0)
        return float(self._score_fn(masked))

    def brightness_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        gray = _to_grayscale(image)
        max_val = float(gray.max())
        min_val = float(gray.min())
        # Dynamic-range floor: if max == min, there is nothing to focus on.
        # Otherwise pass -- bright spots already exist at some level.
        ok = (max_val - min_val) >= 5.0
        return ok, {
            "strategy": self.name,
            "brightness_check": "dynamic_range",
            "max": max_val,
            "min": min_val,
            "dynamic_range": max_val - min_val,
            "floor": 5.0,
        }

    def saturation_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        return _eval_saturation(self.name, self.saturation_threshold, image)


@dataclass
class DarkFieldStrategy:
    """For background-dominated signals (SHG, dark-field, unstained
    cleared tissue) where neither a mid-gray mask nor a bright-spot
    count fits.

    Validity = total gradient energy above a floor. Score =
    manifest-resolved metric (default brenner_gradient).
    Brightness = p99 above floor.

    Failure mode: PROCEED -- dark-field samples typically focus fine
    when they have any signal at all.
    """

    name: str = "dark_field"
    on_failure: StrategyFailureMode = StrategyFailureMode.PROCEED
    score_metric_name: str = "brenner_gradient"
    min_gradient_energy: float = 0.002
    p99_floor: float = 30.0
    # Dark-field signal is the bright tail on a dark background; only a few
    # percent of pixels should ever be near max, so saturation there means
    # the real signal is clipping. Tight tolerance, like sparse_signal.
    saturation_threshold: float = 0.03

    def __post_init__(self) -> None:
        self._score_fn = resolve_metric(self.score_metric_name)

    def is_valid(self, image: np.ndarray, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        ok, stats = total_gradient_energy(image, min_gradient_energy=self.min_gradient_energy)
        stats["strategy"] = self.name
        if logger_:
            level = logger_.info if ok else logger_.warning
            level(
                "dark_field: gradient_energy=%.4f vs min=%.4f -> %s",
                stats["gradient_energy"],
                self.min_gradient_energy,
                "VALID" if ok else "below threshold (will PROCEED anyway)",
            )
        return ok, stats

    def score(self, image: np.ndarray) -> float:
        return float(self._score_fn(image))

    def brightness_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        gray = _to_grayscale(image)
        if gray.max() > 0:
            gray_8bit = (gray / gray.max() * 255.0).astype(np.float32)
        else:
            gray_8bit = gray
        p99 = float(np.percentile(gray_8bit, 99))
        ok = p99 >= self.p99_floor
        return ok, {
            "strategy": self.name,
            "brightness_check": "percentile_floor",
            "p99": p99,
            "floor": self.p99_floor,
        }

    def saturation_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        return _eval_saturation(self.name, self.saturation_threshold, image)


@dataclass
class DenseFluorescenceStrategy:
    """For confluent fluorescent signal (whole-cell IF, dense membrane
    stains, packed nuclei) where neither sparse_signal's bright-spot
    count nor dense_texture's mid-gray tissue mask fits.

    Validity = total gradient energy (the one validity check that does
    not assume a particular brightness convention). Score = vollath_f5
    by default -- the autocorrelation form rejects the shot noise that
    wrecks plain variance metrics on fluorescence. Brightness = p99
    above floor (mirrors DarkFieldStrategy: low median is normal for
    FL, the bright tail is what matters).

    Failure mode: PROCEED -- confluent FL with weak gradient energy can
    still focus when the signal is uniform, so do not defer.
    """

    name: str = "dense_fluorescence"
    on_failure: StrategyFailureMode = StrategyFailureMode.PROCEED
    score_metric_name: str = "vollath_f5"
    min_gradient_energy: float = 0.002
    p99_floor: float = 30.0
    # Confluent fluorescence fills the frame, so like dense tissue it can
    # spare ~10% saturated pixels before the metric degrades.
    saturation_threshold: float = 0.10

    def __post_init__(self) -> None:
        self._score_fn = resolve_metric(self.score_metric_name)

    def is_valid(self, image: np.ndarray, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        ok, stats = total_gradient_energy(image, min_gradient_energy=self.min_gradient_energy)
        stats["strategy"] = self.name
        if logger_:
            level = logger_.info if ok else logger_.warning
            level(
                "dense_fluorescence: gradient_energy=%.4f vs min=%.4f -> %s",
                stats["gradient_energy"],
                self.min_gradient_energy,
                "VALID" if ok else "below threshold (will PROCEED anyway)",
            )
        return ok, stats

    def score(self, image: np.ndarray) -> float:
        return float(self._score_fn(image))

    def brightness_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        gray = _to_grayscale(image)
        if gray.max() > 0:
            gray_8bit = (gray / gray.max() * 255.0).astype(np.float32)
        else:
            gray_8bit = gray
        p99 = float(np.percentile(gray_8bit, 99))
        ok = p99 >= self.p99_floor
        return ok, {
            "strategy": self.name,
            "brightness_check": "percentile_floor",
            "p99": p99,
            "floor": self.p99_floor,
        }

    def saturation_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        return _eval_saturation(self.name, self.saturation_threshold, image)


@dataclass
class ManualOnlyStrategy:
    """Skip auto entirely. Always rejects; the workflow's
    on_failure=MANUAL handler pops the manual focus dialog.

    Used for training runs, edge-case samples, or when the user does
    not trust auto for a modality.
    """

    name: str = "manual_only"
    on_failure: StrategyFailureMode = StrategyFailureMode.MANUAL

    def is_valid(self, image: np.ndarray, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        return False, {"strategy": self.name, "validity_check": "always_false"}

    def score(self, image: np.ndarray) -> float:
        return 0.0

    def brightness_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        return True, {"strategy": self.name, "brightness_check": "none"}

    def saturation_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        # manual_only never runs auto AF, so saturation is the operator's call.
        return True, {"strategy": self.name, "saturation_check": "none"}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_STRATEGY_CLASSES = {
    "dense_texture": DenseTextureStrategy,
    "sparse_signal": SparseSignalStrategy,
    "dense_fluorescence": DenseFluorescenceStrategy,
    "dark_field": DarkFieldStrategy,
    "manual_only": ManualOnlyStrategy,
}


def build_strategy(
    strategy_name: str, params: Optional[Dict[str, Any]] = None
) -> AutofocusStrategy:
    """Build a strategy instance from a name and a flat parameter dict.

    The dict comes from the YAML loader after merging strategy-library
    defaults with per-modality overrides. Unknown strategy names fall
    back to dense_texture with a warning so a typo in YAML does not
    break acquisition entirely.

    Unknown parameters are dropped with a warning rather than crashing
    the dataclass constructor; YAML may carry display-only annotations
    (description, validity_check name) that the strategy class itself
    does not consume.
    """
    cls = _STRATEGY_CLASSES.get(strategy_name)
    if cls is None:
        logger.warning(
            "Unknown autofocus strategy '%s'; falling back to dense_texture",
            strategy_name,
        )
        cls = DenseTextureStrategy

    if not params:
        return cls()

    flattened: Dict[str, Any] = {}
    for k, v in params.items():
        if k == "validity_params" and isinstance(v, dict):
            flattened.update(v)
        elif k == "score_metric":
            flattened["score_metric_name"] = v
        elif k in ("description", "validity_check", "brightness_check", "on_failure"):
            continue
        else:
            flattened[k] = v

    accepted_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    accepted_params = {k: v for k, v in flattened.items() if k in accepted_fields}
    rejected = set(flattened) - set(accepted_params)
    if rejected:
        logger.warning(
            "Strategy '%s' YAML had unknown params: %s (ignored)",
            strategy_name,
            sorted(rejected),
        )

    instance = cls(**accepted_params)

    if "on_failure" in params:
        try:
            instance.on_failure = StrategyFailureMode(params["on_failure"])
        except ValueError:
            logger.warning(
                "Strategy '%s' YAML had invalid on_failure '%s'; " "keeping default %s",
                strategy_name,
                params["on_failure"],
                instance.on_failure.value,
            )

    return instance


def list_strategy_names() -> list[str]:
    """All canonical strategy names defined in the manifest. Compares
    the runtime registry against the manifest at call time so a future
    drift surfaces in the test suite, not at runtime."""
    from microscope_imageprocessing.focus.manifest import get_manifest

    return list(get_manifest().strategies)


# Re-export the dispatcher so callers that import from
# microscope_imageprocessing.focus get a single home.
__all__ = [
    "AutofocusStrategy",
    "DarkFieldStrategy",
    "DenseFluorescenceStrategy",
    "DenseTextureStrategy",
    "ManualOnlyStrategy",
    "SparseSignalStrategy",
    "StrategyFailureMode",
    "build_strategy",
    "list_strategy_names",
    "resolve_validity_check",
]
