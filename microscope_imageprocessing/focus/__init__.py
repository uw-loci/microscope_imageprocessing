"""Focus metrics, validity checks, and strategies for autofocus.

The single source of truth for which metrics, validity checks, and
strategies exist in QPSC. Both Python (this package) and Java
(qupath-extension-qpsc) read the same ``focus_metrics_manifest.yml``
to populate dispatchers and GUI dropdowns.

Public API:

  - ``load_manifest(config_dir=None)`` / ``get_manifest(...)`` -- load
    the YAML manifest and return a frozen :class:`FocusMetricsManifest`.
  - ``resolve_metric(name)`` -- look up a metric callable by canonical
    name. Raises :class:`UnknownMetricError` on unknown or deprecated
    names; never silently substitutes.
  - ``modality_default_metric(modality)`` -- streaming-AF modality
    default lookup, with a codebase-wide fallback.
  - ``list_metric_names()`` -- iterate canonical names.
  - ``MetricSpec``, ``StrategySpec``, ``ValidityCheckSpec``,
    ``ParamSpec``, ``FocusMetricsManifest`` -- typed manifest entries.

Validity checks and strategy classes are added in later workstreams.
"""
from microscope_imageprocessing.focus.manifest import (
    FocusMetricsManifest,
    MetricSpec,
    ParamSpec,
    StrategySpec,
    UnknownMetricError,
    ValidityCheckSpec,
    clear_cache,
    get_manifest,
    load_manifest,
)
from microscope_imageprocessing.focus.metrics import (
    list_metric_names,
    modality_default_metric,
    resolve_metric,
)

__all__ = [
    "FocusMetricsManifest",
    "MetricSpec",
    "ParamSpec",
    "StrategySpec",
    "UnknownMetricError",
    "ValidityCheckSpec",
    "clear_cache",
    "get_manifest",
    "load_manifest",
    "list_metric_names",
    "modality_default_metric",
    "resolve_metric",
]
