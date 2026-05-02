"""Loader for focus_metrics_manifest.yml.

The manifest is the single source of truth for which focus metrics,
validity checks, and strategies exist in QPSC. Both Python (this
package) and Java (qupath-extension-qpsc) read it. See the manifest
file's header comment for schema details.

Discovery order (matches the Java implementation):

  1. ``$QPSC_CONFIG_DIR/focus_metrics_manifest.yml`` (env override)
  2. Alongside the active ``config_<scope>.yml`` -- caller passes the
     directory via ``load_manifest(config_dir=...)``.
  3. The packaged default at
     ``microscope_imageprocessing/focus/_packaged_manifest.yml``.

The packaged copy is bundled at install time so tests and dev work
without a configurations checkout. It is regenerated from the canonical
file in CI; do not edit the packaged copy by hand.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class UnknownMetricError(KeyError):
    """Raised when a metric name is not in the manifest.

    The error message names the canonical replacement when the
    requested name appears in ``removed_aliases``. Callers should let
    this propagate -- the previous silent-fallback behavior masked
    drift bugs.
    """


@dataclass(frozen=True)
class MetricSpec:
    name: str
    group: str  # recommended | advanced | special
    badge: str  # high | medium | low | na
    best_for: str
    avoid_when: str
    requires: str  # numpy | scipy | skimage | scipy+skimage
    supported_paths: Tuple[str, ...]  # subset of {streaming, standard, strategy}
    role: Optional[str] = None  # "fallback" for p98_p2; None otherwise


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str  # float | int | list_of_float
    default: Any
    range: Optional[Tuple[float, float]] = None
    length: Optional[int] = None  # for list types


@dataclass(frozen=True)
class ValidityCheckSpec:
    name: str
    description: str
    params: Tuple[ParamSpec, ...]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    score_metric_default: str
    validity_check: str
    on_failure: str  # defer | proceed | manual


@dataclass(frozen=True)
class FocusMetricsManifest:
    schema_version: int
    metrics: Dict[str, MetricSpec]
    removed_aliases: Dict[str, str]  # old_name -> canonical_name
    validity_checks: Dict[str, ValidityCheckSpec]
    strategies: Dict[str, StrategySpec]
    modality_defaults: Dict[str, str]  # modality_key (lowercase) -> metric_name
    source_path: Path = field(compare=False, repr=False)

    def metrics_by_group(self, group: str) -> List[MetricSpec]:
        """All metrics in the given group, in manifest order."""
        return [m for m in self.metrics.values() if m.group == group]

    def modality_default_metric(self, modality: Optional[str]) -> Optional[str]:
        """Look up the default metric for a modality string.

        Case-insensitive. Returns the canonical metric name if matched,
        ``None`` if no mapping exists. Caller decides what fallback to
        apply (typically ``tenengrad``).
        """
        if not modality:
            return None
        return self.modality_defaults.get(modality.strip().lower())


def _parse_metric(entry: Mapping[str, Any]) -> MetricSpec:
    paths = entry.get("supported_paths") or []
    return MetricSpec(
        name=entry["name"],
        group=entry.get("group", "advanced"),
        badge=entry.get("badge", "medium"),
        best_for=str(entry.get("best_for", "")).strip(),
        avoid_when=str(entry.get("avoid_when", "")).strip(),
        requires=entry.get("requires", "numpy"),
        supported_paths=tuple(paths),
        role=entry.get("role"),
    )


def _parse_param(name: str, entry: Mapping[str, Any]) -> ParamSpec:
    rng = entry.get("range")
    return ParamSpec(
        name=name,
        type=entry["type"],
        default=entry.get("default"),
        range=tuple(rng) if rng is not None else None,
        length=entry.get("length"),
    )


def _parse_validity_check(entry: Mapping[str, Any]) -> ValidityCheckSpec:
    params_dict = entry.get("params") or {}
    params = tuple(_parse_param(k, v) for k, v in params_dict.items())
    return ValidityCheckSpec(
        name=entry["name"],
        description=str(entry.get("description", "")).strip(),
        params=params,
    )


def _parse_strategy(entry: Mapping[str, Any]) -> StrategySpec:
    return StrategySpec(
        name=entry["name"],
        description=str(entry.get("description", "")).strip(),
        score_metric_default=entry["score_metric_default"],
        validity_check=entry["validity_check"],
        on_failure=entry.get("on_failure", "defer"),
    )


def _parse_manifest(doc: Mapping[str, Any], source_path: Path) -> FocusMetricsManifest:
    metrics = {e["name"]: _parse_metric(e) for e in doc.get("metrics", [])}
    aliases = dict(doc.get("removed_aliases", {}) or {})
    vchecks = {e["name"]: _parse_validity_check(e) for e in doc.get("validity_checks", [])}
    strategies = {e["name"]: _parse_strategy(e) for e in doc.get("strategies", [])}
    modality = {str(k).lower(): v for k, v in (doc.get("modality_defaults") or {}).items()}

    # Cross-reference checks -- catch drift in the manifest itself before
    # any caller sees a confusing runtime error.
    metric_names = set(metrics)
    for s in strategies.values():
        if s.score_metric_default not in metric_names:
            raise ValueError(
                f"Strategy '{s.name}' default score_metric '{s.score_metric_default}' "
                f"is not in the manifest's metrics list."
            )
        if s.validity_check not in vchecks:
            raise ValueError(
                f"Strategy '{s.name}' validity_check '{s.validity_check}' "
                f"is not in the manifest's validity_checks list."
            )
    for mod, name in modality.items():
        if name not in metric_names:
            raise ValueError(
                f"modality_defaults['{mod}'] -> '{name}' is not in the manifest's metrics list."
            )
    overlap = aliases.keys() & metric_names
    if overlap:
        raise ValueError(
            f"Names appear in BOTH metrics and removed_aliases: {sorted(overlap)}. "
            f"Pick one canonical spelling."
        )

    return FocusMetricsManifest(
        schema_version=int(doc.get("schema_version", 1)),
        metrics=metrics,
        removed_aliases=aliases,
        validity_checks=vchecks,
        strategies=strategies,
        modality_defaults=modality,
        source_path=source_path,
    )


def _candidate_paths(config_dir: Optional[Path]) -> List[Path]:
    candidates: List[Path] = []
    env = os.environ.get("QPSC_CONFIG_DIR")
    if env:
        candidates.append(Path(env) / "focus_metrics_manifest.yml")
    if config_dir is not None:
        candidates.append(Path(config_dir) / "focus_metrics_manifest.yml")
    candidates.append(Path(__file__).parent / "_packaged_manifest.yml")
    return candidates


def load_manifest(config_dir: Optional[Path] = None) -> FocusMetricsManifest:
    """Load the focus metrics manifest from the first discoverable path.

    Args:
        config_dir: Directory that holds the active ``config_<scope>.yml``.
            If passed, looked up second (after ``$QPSC_CONFIG_DIR``).

    Raises:
        FileNotFoundError: if no candidate path resolves -- including
            the packaged default. This indicates a broken installation,
            not a user error.
        ValueError: if the manifest is structurally invalid (missing
            keys, dangling references, alias collisions).
    """
    last_err: Optional[Exception] = None
    for path in _candidate_paths(config_dir):
        if not path.is_file():
            continue
        try:
            with path.open() as f:
                doc = yaml.safe_load(f) or {}
        except Exception as e:
            last_err = e
            logger.warning("Could not read focus manifest at %s: %s", path, e)
            continue
        manifest = _parse_manifest(doc, source_path=path)
        logger.debug(
            "Loaded focus manifest from %s (%d metrics, %d strategies)",
            path, len(manifest.metrics), len(manifest.strategies),
        )
        return manifest

    raise FileNotFoundError(
        "No focus_metrics_manifest.yml found. Searched: "
        + ", ".join(str(p) for p in _candidate_paths(config_dir))
        + (f". Last error: {last_err}" if last_err else "")
    )


# Module-level cache. Most callers want the manifest once per process;
# tests reset it via clear_cache() between fixtures.
_cached: Optional[FocusMetricsManifest] = None
_cached_config_dir: Optional[Path] = None


def get_manifest(config_dir: Optional[Path] = None) -> FocusMetricsManifest:
    """Cached load. Re-loads only when the config_dir changes.

    Use ``load_manifest()`` directly if you need to bypass the cache.
    """
    global _cached, _cached_config_dir
    if _cached is None or _cached_config_dir != config_dir:
        _cached = load_manifest(config_dir)
        _cached_config_dir = config_dir
    return _cached


def clear_cache() -> None:
    """Drop the cached manifest. Mainly for tests."""
    global _cached, _cached_config_dir
    _cached = None
    _cached_config_dir = None
