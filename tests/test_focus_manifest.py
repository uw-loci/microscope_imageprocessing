"""Tests for the focus_metrics_manifest.yml loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from microscope_imageprocessing.focus import (
    FocusMetricsManifest,
    clear_cache,
    get_manifest,
    load_manifest,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with a clean manifest cache."""
    clear_cache()
    yield
    clear_cache()


class TestPackagedManifestLoads:
    """The packaged copy must always be discoverable -- if this fails
    the install is broken and every other test would cascade."""

    def test_loads_packaged_default(self):
        m = load_manifest()
        assert isinstance(m, FocusMetricsManifest)
        assert m.schema_version >= 1
        assert m.source_path.name == "_packaged_manifest.yml"

    def test_get_manifest_caches(self):
        a = get_manifest()
        b = get_manifest()
        assert a is b


class TestManifestStructure:
    """Lock in the canonical metric set from M1.

    These tests are the contract: if a future edit removes a metric or
    a strategy, the test fails loudly so the GUI / loader changes that
    must accompany it cannot be forgotten.
    """

    def test_ten_canonical_metrics(self):
        m = load_manifest()
        assert set(m.metrics) == {
            "tenengrad",
            "laplacian_variance",
            "brenner_gradient",
            "normalized_variance",
            "vollath_f5",
            "sobel",
            "p98_p2",
            "robust_sharpness_metric",
            "hybrid_sharpness_metric",
            "none",
        }

    def test_metric_groups_assigned(self):
        m = load_manifest()
        groups = {name: spec.group for name, spec in m.metrics.items()}
        # Recommended bucket -- the three the GUI surfaces above the fold.
        assert groups["tenengrad"] == "recommended"
        assert groups["laplacian_variance"] == "recommended"
        assert groups["brenner_gradient"] == "recommended"
        # Special bucket -- never picked by users casually.
        assert groups["p98_p2"] == "special"
        assert groups["none"] == "special"

    def test_p98_p2_marked_as_fallback(self):
        m = load_manifest()
        assert m.metrics["p98_p2"].role == "fallback"

    def test_supported_paths_for_path_only_metrics(self):
        m = load_manifest()
        # vollath_f5 lives only in the streaming AF code path today.
        assert m.metrics["vollath_f5"].supported_paths == ("streaming",)
        # sobel lives only in standard / strategy paths (skimage dep).
        assert "streaming" not in m.metrics["sobel"].supported_paths
        # tenengrad must be available everywhere -- it's the recommended
        # default for tissue.
        assert set(m.metrics["tenengrad"].supported_paths) == {
            "streaming",
            "standard",
            "strategy",
        }

    def test_removed_aliases_are_documented(self):
        m = load_manifest()
        assert m.removed_aliases == {
            "volath5": "vollath_f5",
            "tenenbaum_gradient": "tenengrad",
        }

    def test_alias_targets_exist_as_metrics(self):
        m = load_manifest()
        for old, new in m.removed_aliases.items():
            assert new in m.metrics, f"Alias {old!r} targets {new!r} which is not a real metric."

    def test_shipped_validity_checks(self):
        m = load_manifest()
        assert set(m.validity_checks) == {
            "texture_and_area",
            "bright_spot_count",
            "total_gradient_energy",
            "chroma_deviation",
            "always_false",
        }

    def test_shipped_strategies(self):
        m = load_manifest()
        # The four "core" strategies must always be present so existing
        # autofocus YAMLs keep validating; additive strategies are
        # allowed (manifest is the source of truth for what ships).
        assert {
            "dense_texture",
            "sparse_signal",
            "dark_field",
            "manual_only",
        }.issubset(set(m.strategies))

    def test_strategies_reference_real_metrics_and_checks(self):
        m = load_manifest()
        for s in m.strategies.values():
            assert s.score_metric_default in m.metrics
            assert s.validity_check in m.validity_checks

    def test_modality_default_lookups(self):
        m = load_manifest()
        assert m.modality_default_metric("Brightfield") == "tenengrad"
        assert m.modality_default_metric("BF") == "tenengrad"
        assert m.modality_default_metric("ppm") == "tenengrad"
        assert m.modality_default_metric("fluorescence") == "vollath_f5"
        assert m.modality_default_metric("LSM") == "vollath_f5"
        # Unknown modality returns None; caller decides what to do.
        assert m.modality_default_metric("unknown_modality") is None
        # Empty / None inputs are tolerated.
        assert m.modality_default_metric(None) is None
        assert m.modality_default_metric("") is None


class TestValiditySpec:
    def test_texture_and_area_params_present(self):
        m = load_manifest()
        params = {p.name: p for p in m.validity_checks["texture_and_area"].params}
        assert set(params) == {
            "texture_threshold",
            "tissue_area_threshold",
            "rgb_brightness_threshold",
            "tissue_mask_range",
            "median_floor",
        }

    def test_param_types_and_defaults(self):
        m = load_manifest()
        params = {p.name: p for p in m.validity_checks["texture_and_area"].params}
        assert params["texture_threshold"].type == "float"
        assert params["texture_threshold"].default == 0.010
        assert params["texture_threshold"].range == (0.0, 1.0)
        assert params["tissue_mask_range"].type == "list_of_float"
        assert params["tissue_mask_range"].length == 2
        assert params["tissue_mask_range"].default == [0.10, 0.90]

    def test_int_params_are_typed_int(self):
        m = load_manifest()
        params = {p.name: p for p in m.validity_checks["bright_spot_count"].params}
        # min_spots / spot_min_separation_px are integers in Python; the
        # GUI must render them as Spinner<Integer> not double-validated
        # text fields. The 'type: int' annotation in the manifest is
        # what drives that.
        assert params["min_spots"].type == "int"
        assert params["spot_min_separation_px"].type == "int"

    def test_always_false_has_no_params(self):
        m = load_manifest()
        assert m.validity_checks["always_false"].params == ()


class TestRefuseInvalidManifests:
    """Cross-reference checks must reject malformed manifests at load
    time -- this catches authoring errors before any caller sees a
    confusing runtime error."""

    def _write(self, tmp_path: Path, body: str) -> Path:
        p = tmp_path / "focus_metrics_manifest.yml"
        p.write_text(body)
        return tmp_path

    def test_strategy_referencing_unknown_metric_rejected(self, tmp_path):
        cfg = self._write(
            tmp_path,
            """
schema_version: 1
metrics:
  - name: tenengrad
    group: recommended
    badge: high
    requires: numpy
    supported_paths: [streaming, standard, strategy]
validity_checks:
  - name: always_false
    description: ""
    params: {}
strategies:
  - name: bad
    description: refers to a metric that does not exist
    score_metric_default: not_a_real_metric
    validity_check: always_false
    on_failure: defer
""",
        )
        with pytest.raises(ValueError, match="not_a_real_metric"):
            load_manifest(config_dir=cfg)

    def test_strategy_referencing_unknown_validity_check_rejected(self, tmp_path):
        cfg = self._write(
            tmp_path,
            """
schema_version: 1
metrics:
  - name: tenengrad
    group: recommended
    badge: high
    requires: numpy
    supported_paths: [streaming]
validity_checks:
  - name: always_false
    description: ""
    params: {}
strategies:
  - name: bad
    description: refers to a check that does not exist
    score_metric_default: tenengrad
    validity_check: not_a_real_check
    on_failure: defer
""",
        )
        with pytest.raises(ValueError, match="not_a_real_check"):
            load_manifest(config_dir=cfg)

    def test_modality_default_pointing_at_unknown_metric_rejected(self, tmp_path):
        cfg = self._write(
            tmp_path,
            """
schema_version: 1
metrics:
  - name: tenengrad
    group: recommended
    badge: high
    requires: numpy
    supported_paths: [streaming]
validity_checks: []
strategies: []
modality_defaults:
  brightfield: not_a_real_metric
""",
        )
        with pytest.raises(ValueError, match="not_a_real_metric"):
            load_manifest(config_dir=cfg)

    def test_alias_and_metric_overlap_rejected(self, tmp_path):
        # If 'tenengrad' is BOTH a current metric AND a removed alias,
        # the loader's behaviour would be ambiguous. Refuse.
        cfg = self._write(
            tmp_path,
            """
schema_version: 1
metrics:
  - name: tenengrad
    group: recommended
    badge: high
    requires: numpy
    supported_paths: [streaming]
validity_checks: []
strategies: []
removed_aliases:
  tenengrad: laplacian_variance
""",
        )
        with pytest.raises(ValueError, match="metrics and removed_aliases"):
            load_manifest(config_dir=cfg)


class TestDiscoveryOrder:
    def test_env_var_wins_over_config_dir(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "env"
        config_dir = tmp_path / "cfg"
        env_dir.mkdir()
        config_dir.mkdir()
        # Both manifests are valid but tagged differently in
        # description so we can tell which one was loaded.
        valid_body = """
schema_version: 99
metrics:
  - name: tenengrad
    group: recommended
    badge: high
    requires: numpy
    supported_paths: [streaming]
validity_checks: []
strategies: []
"""
        (env_dir / "focus_metrics_manifest.yml").write_text(valid_body)
        (config_dir / "focus_metrics_manifest.yml").write_text(
            valid_body.replace("schema_version: 99", "schema_version: 42")
        )
        monkeypatch.setenv("QPSC_CONFIG_DIR", str(env_dir))
        m = load_manifest(config_dir=config_dir)
        assert m.schema_version == 99

    def test_config_dir_wins_over_packaged_default(self, tmp_path):
        config_dir = tmp_path / "cfg"
        config_dir.mkdir()
        (config_dir / "focus_metrics_manifest.yml").write_text("""
schema_version: 7
metrics:
  - name: tenengrad
    group: recommended
    badge: high
    requires: numpy
    supported_paths: [streaming]
validity_checks: []
strategies: []
""")
        m = load_manifest(config_dir=config_dir)
        assert m.schema_version == 7
        assert "cfg" in str(m.source_path)

    def test_falls_through_to_packaged_default(self, tmp_path):
        # Neither env nor config_dir resolves. Packaged default wins.
        empty = tmp_path / "empty"
        empty.mkdir()
        m = load_manifest(config_dir=empty)
        assert m.source_path.name == "_packaged_manifest.yml"


def test_metricspec_is_frozen():
    # Spec classes are dataclasses with frozen=True so callers cannot
    # accidentally mutate the cached registry. Lock that in.
    m = load_manifest()
    spec = m.metrics["tenengrad"]
    with pytest.raises(Exception):
        spec.group = "advanced"  # type: ignore[misc]
