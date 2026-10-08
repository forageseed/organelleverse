"""Managed annotation-QC service, lazy facade, and Agent operation tests.

The service must publish one canonical, content-addressed annotation-QC Result
under the managed run store, expose it through the lazy ``ov.qc.annotation``
facade and the ``qc.annotation`` operation, and keep the direct Python, Registry,
and Agent JSON call paths byte-identical. Reuse must re-verify every cached
artifact and identity; a mutated cache fails closed as ``OrganelleInputError``.

These tests are written before the service, facade, and spec exist so their RED
failure proves the wiring is missing.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import organelleverse as ov
from organelleverse.assembly.contracts import canonical_json_bytes
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.frozen import FrozenMap
from organelleverse.operations import OperationRegistry
from organelleverse.operations.adapters import invoke_json
from organelleverse.quality_control.annotation_contracts import (
    AnnotationQcReport,
    AnnotationQcRunManifest,
)
from organelleverse.quality_control.annotation_policy import EXPECTED_PCG_NAMES
from organelleverse.quality_control.annotation_service import (
    aggregate_decision,  # pyright: ignore[reportPrivateUsage]
    run_annotation_qc,
)
from organelleverse.quality_control.operations import ANNOTATION_QC_SPEC
from organelleverse.runtime import managed_runs_root
from tests.quality_control.annotation_helpers import annotation_result_fixture

_REPORT_KIND = "annotation_qc_report"
_MANIFEST_KIND = "annotation_qc_run_manifest"


def _set_managed_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(cache_root))
    return cache_root


# ---------------------------------------------------------------------------
# Direct / Registry / Agent identity
# ---------------------------------------------------------------------------


def test_direct_registry_and_agent_annotation_qc_are_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")
    registry = OperationRegistry()
    registry.register(ANNOTATION_QC_SPEC, ov.qc.annotation)

    direct = ov.qc.annotation(source)
    registered = registry.invoke("qc.annotation", input=source, parameters={})
    response = invoke_json(
        {
            "operation_id": "qc.annotation",
            "input": source.model_dump(mode="json"),
            "parameters": {},
        },
        registry=registry,
        granted_side_effects={"read_files", "write_files"},
    )

    assert response["ok"] is True
    assert direct == registered
    assert response["result"] == direct.model_dump(mode="json")
    assert direct.operation_id == "qc.annotation"
    assert direct.operation_version == "1.0"
    assert direct.metrics["qc_decision"] == "needs_review"


def test_annotation_qc_result_carries_exactly_two_managed_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = _set_managed_cache(monkeypatch, tmp_path)
    isolated_cwd = tmp_path / "cwd"
    isolated_cwd.mkdir()
    monkeypatch.chdir(isolated_cwd)
    source = annotation_result_fixture(tmp_path / "source")

    result = ov.qc.annotation(source)

    runs_root = managed_runs_root()
    assert {artifact.kind for artifact in result.artifacts} == {
        _REPORT_KIND,
        _MANIFEST_KIND,
    }
    assert all(Path(artifact.uri).is_relative_to(runs_root) for artifact in result.artifacts)
    assert all(artifact.resolve().is_file() for artifact in result.artifacts)
    # Managed output only: nothing is written under the caller's working directory.
    assert not (isolated_cwd / ".organelleverse").exists()
    # The cache root is isolated from the default home store.
    assert str(runs_root).startswith(str(cache_root))


def test_independent_annotation_qc_calls_reuse_one_managed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")

    first = ov.qc.annotation(source)
    second = ov.qc.annotation(source)

    assert second == first
    assert len(list((managed_runs_root() / "qc.annotation").iterdir())) == 1


def test_report_result_and_manifest_share_one_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")

    result = run_annotation_qc(source)

    report_artifact = next(
        artifact for artifact in result.artifacts if artifact.kind == _REPORT_KIND
    )
    manifest_artifact = next(
        artifact for artifact in result.artifacts if artifact.kind == _MANIFEST_KIND
    )
    report = AnnotationQcReport.model_validate_json(report_artifact.resolve().read_bytes())
    manifest = AnnotationQcRunManifest.model_validate_json(manifest_artifact.resolve().read_bytes())

    provenance = result.provenance
    assert provenance is not None
    assert provenance.run_manifest_id == manifest.run_manifest_id
    assert manifest.source_annotation_result_id == source.object_id
    assert report.source_annotation_result_id == source.object_id
    assert report.source_annotation_object_id == manifest.source_annotation_object_id
    assert report.source_annotation_object_id
    assert report.decision == result.metrics["qc_decision"]
    assert report.policy_version == manifest.policy_version
    assert provenance.upstream_run_manifest_ids == (
        source.provenance.run_manifest_id if source.provenance else "",
    )
    # Source artifact hashes are linked through provenance.
    assert provenance.input_artifact_hashes == tuple(
        artifact.sha256 for artifact in source.artifacts
    )
    # The manifest records the report as its output; it cannot list its own
    # content-addressed artifact without a circular identity.
    assert manifest.outputs == (report_artifact,)
    assert manifest.parameters_hash == provenance.parameters_hash


def test_cached_artifact_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")

    first = ov.qc.annotation(source)
    report = next(artifact for artifact in first.artifacts if artifact.kind == _REPORT_KIND)
    Path(report.uri).write_text('{"tampered": true}\n')

    with pytest.raises(OrganelleInputError) as caught:
        ov.qc.annotation(source)
    assert caught.value.code == "qc.annotation_cache_conflict"
    details = caught.value.details
    assert isinstance(details, FrozenMap)
    assert "path" in details
    assert "field" in details


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("status", "ok"),
        ("flags", ["forged"]),
        ("findings", []),
        ("summary_text", "forged"),
    ],
)
def test_cached_result_semantic_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")
    result = ov.qc.annotation(source)
    _rewrite_cached_result(result, lambda payload: payload.__setitem__(field, replacement))

    with pytest.raises(OrganelleInputError) as caught:
        ov.qc.annotation(source)

    assert caught.value.code == "qc.annotation_cache_conflict"
    assert isinstance(caught.value.details, FrozenMap)
    assert "path" in caught.value.details
    assert "field" in caught.value.details


def test_cached_non_count_metric_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")
    result = ov.qc.annotation(source)

    def mutate(payload: dict[str, object]) -> None:
        metrics = dict(cast(dict[str, object], payload["metrics"]))
        metrics["profile_id"] = "forged"
        payload["metrics"] = metrics

    _rewrite_cached_result(result, mutate)

    with pytest.raises(OrganelleInputError) as caught:
        ov.qc.annotation(source)
    assert caught.value.code == "qc.annotation_cache_conflict"


def test_cached_artifact_uri_escape_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")
    result = ov.qc.annotation(source)
    report = next(item for item in result.artifacts if item.kind == _REPORT_KIND)
    external = tmp_path / "outside-report.json"
    shutil.copy2(report.uri, external)

    def mutate(payload: dict[str, object]) -> None:
        artifacts = [dict(item) for item in cast(list[dict[str, object]], payload["artifacts"])]
        artifacts[0]["uri"] = str(external)
        payload["artifacts"] = artifacts

    _rewrite_cached_result(result, mutate)

    with pytest.raises(OrganelleInputError) as caught:
        ov.qc.annotation(source)
    assert caught.value.code == "qc.annotation_cache_conflict"


def test_cached_extra_artifact_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")
    result = ov.qc.annotation(source)

    def mutate(payload: dict[str, object]) -> None:
        artifacts = list(cast(list[dict[str, object]], payload["artifacts"]))
        artifacts.append(dict(artifacts[0]))
        payload["artifacts"] = artifacts

    _rewrite_cached_result(result, mutate)

    with pytest.raises(OrganelleInputError) as caught:
        ov.qc.annotation(source)
    assert caught.value.code == "qc.annotation_cache_conflict"


def test_cached_report_identity_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")
    result = ov.qc.annotation(source)
    report = next(item for item in result.artifacts if item.kind == _REPORT_KIND)
    report_path = Path(report.uri)
    report_payload = json.loads(report_path.read_text())
    report_payload["policy_version"] = "forged"
    report_payload.pop("object_id", None)
    forged_report = AnnotationQcReport.model_validate(report_payload)
    report_path.write_bytes(canonical_json_bytes(forged_report.model_dump(mode="json")))
    rebuilt = ArtifactRef.from_path(
        report_path,
        kind=report.kind,
        format=report.format,
        media_type=report.media_type,
    )

    def mutate(payload: dict[str, object]) -> None:
        artifacts = [dict(item) for item in cast(list[dict[str, object]], payload["artifacts"])]
        artifacts[0]["sha256"] = rebuilt.sha256
        artifacts[0]["size_bytes"] = rebuilt.size_bytes
        payload["artifacts"] = artifacts

    _rewrite_cached_result(result, mutate)

    with pytest.raises(OrganelleInputError) as caught:
        ov.qc.annotation(source)
    assert caught.value.code == "qc.annotation_cache_conflict"


def test_export_compatibility_passes_for_a_valid_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")

    result = run_annotation_qc(source)
    report = _report(result)
    export_check = next(
        check for check in report.checks if check.check_id == "annotation.export_compatibility"
    )
    assert export_check.status == "pass"


def test_resolver_failure_leaves_no_managed_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")
    annotation = next(artifact for artifact in source.artifacts if artifact.kind == "annotation")
    Path(annotation.uri).write_text("{}")

    with pytest.raises(OrganelleInputError):
        ov.qc.annotation(source)
    assert not (managed_runs_root() / "qc.annotation").exists()


# ---------------------------------------------------------------------------
# Decision mapping
# ---------------------------------------------------------------------------


def test_full_profile_recovers_ready_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source", genes=EXPECTED_PCG_NAMES)

    result = ov.qc.annotation(source)

    assert result.metrics["qc_decision"] == "ready"
    assert result.status == "ok"
    assert result.flags == ("qc_ready",)


def test_invalid_structure_yields_not_ready_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(
        tmp_path / "source", genes=EXPECTED_PCG_NAMES, invalid_extra="atp1"
    )

    result = ov.qc.annotation(source)

    assert result.metrics["qc_decision"] == "not_ready"
    assert result.status == "warning"


def test_aggregate_decision_precedence_is_fixed() -> None:
    from organelleverse.quality_control.contracts import QcCheck

    def check(status: str) -> QcCheck:
        return QcCheck(check_id="x", category="c", status=status, message="m")  # type: ignore[arg-type]

    assert aggregate_decision((check("pass"),), required_evidence_missing=False) == "ready"
    assert aggregate_decision((check("pass"),), required_evidence_missing=True) == (
        "insufficient_evidence"
    )
    assert (
        aggregate_decision((check("pass"), check("warn")), required_evidence_missing=True)
        == "needs_review"
    )
    assert (
        aggregate_decision((check("warn"), check("fail")), required_evidence_missing=True)
        == "not_ready"
    )


# ---------------------------------------------------------------------------
# Facade and operation registration
# ---------------------------------------------------------------------------


def test_annotation_facade_is_lazy_and_sorted() -> None:
    import importlib

    api = importlib.import_module("organelleverse.quality_control.api")
    assert ov.qc.annotation is api.annotation
    assert ov.qc.__all__ == [
        "annotation",
        "assembly",
        "compare_assembly_to_reference",
        "filter_long_reads",
        "filter_short_reads",
        "read_statistics",
        "write",
    ]


def test_annotation_qc_spec_is_result_to_result_consume() -> None:
    assert ANNOTATION_QC_SPEC.operation_id == "qc.annotation"
    assert ANNOTATION_QC_SPEC.contract_version == "1.0"
    assert ANNOTATION_QC_SPEC.stage.value == "consume"
    assert ANNOTATION_QC_SPEC.input_kind.value == "result"
    assert ANNOTATION_QC_SPEC.output_kind.value == "result"
    assert ANNOTATION_QC_SPEC.organelle_types == ("mitochondrion",)
    assert ANNOTATION_QC_SPEC.fallback.allowed is False
    assert ANNOTATION_QC_SPEC.cacheable is False
    assert set(ANNOTATION_QC_SPEC.side_effects) == {"read_files", "write_files"}


def test_annotation_qc_parameter_schema_is_closed_with_no_properties() -> None:
    registry = OperationRegistry()
    registry.register(ANNOTATION_QC_SPEC, ov.qc.annotation)
    schema = registry.parameter_schema("qc.annotation")
    assert schema["additionalProperties"] is False
    assert schema["properties"] == {}


def test_qc_annotation_is_in_release_catalog_once() -> None:
    specs = ov.operations.list()
    annotation_specs = [spec for spec in specs if spec.operation_id == "qc.annotation"]
    assert len(annotation_specs) == 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _report(result: ov.OrganelleResult) -> AnnotationQcReport:
    report_artifact = next(
        artifact for artifact in result.artifacts if artifact.kind == _REPORT_KIND
    )
    return AnnotationQcReport.model_validate_json(Path(report_artifact.uri).read_bytes())


def _rewrite_cached_result(
    result: ov.OrganelleResult,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    result_path = Path(result.artifacts[0].uri).parent / "result.json"
    payload: dict[str, object] = json.loads(result_path.read_text())
    mutate(payload)
    payload.pop("object_id", None)
    rewritten = ov.OrganelleResult.model_validate(payload)
    result_path.write_bytes(canonical_json_bytes(rewritten.model_dump(mode="json")))
