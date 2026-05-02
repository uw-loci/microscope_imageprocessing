"""Focus metrics, validity checks, and strategies for autofocus.

The single source of truth for which metrics, validity checks, and
strategies exist in QPSC. Both Python (this package) and Java
(qupath-extension-qpsc) read the same ``focus_metrics_manifest.yml``
to populate dispatchers and GUI dropdowns.

Public API (current):

  - ``load_manifest(config_dir=None)`` -- one-shot load of the manifest.
  - ``get_manifest(config_dir=None)`` -- cached load.
  - ``UnknownMetricError`` -- raised by future ``resolve_metric`` on
    unknown or deprecated names. Never silently translated.
  - ``MetricSpec``, ``StrategySpec``, ``ValidityCheckSpec``,
    ``ParamSpec``, ``FocusMetricsManifest`` -- typed manifest entries.

Once metric implementations, validity checks, and strategies are added
in subsequent workstreams, this module also exports ``resolve_metric``,
``get_validity_check``, and the strategy classes.
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
]
