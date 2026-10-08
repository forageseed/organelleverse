"""Unified ``qc.write`` materialization for annotation-QC Results.

``ov.qc.write`` must dispatch a ``qc.annotation`` Result to the annotation-QC
writer, render the full human/machine bundle, verify the canonical artifacts and
Result/report/manifest identities independently of the service, and reuse or
refuse destinations with the same atomic semantics as the assembly writer. The
assembly writer path must remain unchanged.

These tests are written before the annotation writer exists so their RED failure
proves the dispatch and rendering are missing.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import cast

import pytest

import organelleverse as ov
from organelleverse.assembly.contracts import canonical_json_bytes
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import OperationRegistry
from organelleverse.operations.adapters import invoke_json
from organelleverse.operations.spec import SideEffect
from organelleverse.quality_control.annotation_contracts import (
    AnnotationQcReport,
    AnnotationQcRunManifest,
)
from organelleverse.quality_control.operations import QC_WRITE_SPEC

from .annotation_helpers import annotation_result_fixture

_REPORT_KIND = "annotation_qc_report"
_MANIFEST_KIND = "annotation_qc_run_manifest"
_DIRECTORY_FILES = {
    "annotation_qc_summary.md",
    "annotation_qc_report.json",
    "annotation_qc_run_manifest.json",
    "summary.csv",
    "checks.csv",
    "features.csv",
    "gene_profile.csv",
    "rejected_cds_candidates.csv",
}
_XLSX_SHEETS = (
    "summary",
    "checks",
    "features",
    "gene_profile",
    "rejected_cds_candidates",
)


def _set_managed_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))


def _qc_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> OrganelleResult:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source")
    return ov.qc.annotation(source)


def _report(result: OrganelleResult) -> AnnotationQcReport:
    artifact = next(item for item in result.artifacts if item.kind == _REPORT_KIND)
    return AnnotationQcReport.model_validate_json(Path(artifact.uri).read_bytes())


def _manifest(result: OrganelleResult) -> AnnotationQcRunManifest:
    artifact = next(item for item in result.artifacts if item.kind == _MANIFEST_KIND)
    return AnnotationQcRunManifest.model_validate_json(Path(artifact.uri).read_bytes())


# ---------------------------------------------------------------------------
# Directory bundle
# ---------------------------------------------------------------------------


def test_qc_write_materializes_complete_annotation_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    bundle = tmp_path / "bundle"

    written = ov.qc.write(qc, output=bundle)

    assert written.operation_id == "qc.write"
    assert written.operation_version == "1.2"
    assert written.status == qc.status
    assert written.metrics == qc.metrics
    assert "qc_materialized" in written.flags
    assert {path.name for path in bundle.iterdir()} == _DIRECTORY_FILES


def test_directory_bundle_round_trips_every_canonical_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    report = _report(qc)
    bundle = tmp_path / "bundle"
    ov.qc.write(qc, output=bundle)

    reparsed_report = AnnotationQcReport.model_validate_json(
        (bundle / "annotation_qc_report.json").read_bytes()
    )
    reparsed_manifest = AnnotationQcRunManifest.model_validate_json(
        (bundle / "annotation_qc_run_manifest.json").read_bytes()
    )
    assert reparsed_report == report
    assert reparsed_manifest.run_manifest_id == qc.provenance.run_manifest_id

    with (bundle / "summary.csv").open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["decision"] == report.decision
    assert summary_rows[0]["profile_recovery_fraction"]

    with (bundle / "checks.csv").open(newline="") as handle:
        check_rows = list(csv.DictReader(handle))
    assert len(check_rows) == len(report.checks)
    assert [row["check_id"] for row in check_rows] == [item.check_id for item in report.checks]

    with (bundle / "features.csv").open(newline="") as handle:
        feature_rows = list(csv.DictReader(handle))
    assert len(feature_rows) == len(report.feature_summaries)
    assert feature_rows[0]["feature_id"] == report.feature_summaries[0].feature_id

    with (bundle / "gene_profile.csv").open(newline="") as handle:
        profile_rows = list(csv.DictReader(handle))
    expected_genes = len(report.gene_profile_assessment.expected_pcg_names) + len(
        report.gene_profile_assessment.variable_pcg_names
    )
    assert len(profile_rows) == expected_genes
    assert {row["profile_id"] for row in profile_rows} == {
        report.gene_profile_assessment.profile_id
    }
    assert {row["class"] for row in profile_rows} == {"expected", "variable"}

    with (bundle / "rejected_cds_candidates.csv").open(newline="") as handle:
        rejected_rows = list(csv.DictReader(handle))
    assert len(rejected_rows) == len(report.rejected_candidates)


def test_gene_profile_csv_records_status_and_copy_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(
        tmp_path / "source",
        genes=("atp1", "cob", "cox1"),
        duplicate_gene="atp1",
    )
    qc = ov.qc.annotation(source)
    bundle = tmp_path / "bundle"
    ov.qc.write(qc, output=bundle)

    with (bundle / "gene_profile.csv").open(newline="") as handle:
        rows = {row["gene_name"]: row for row in csv.DictReader(handle)}

    assert rows["atp1"]["class"] == "expected"
    assert rows["atp1"]["status"] == "duplicated"
    assert rows["atp1"]["copy_count"] == "2"
    assert rows["cob"]["status"] == "recovered"
    assert rows["cob"]["copy_count"] == "1"
    # A variable gene absent from the fixture is reported, never warned.
    assert rows["rps10"]["class"] == "variable"
    assert rows["rps10"]["status"] == "absent"
    assert rows["rps10"]["copy_count"] == "0"


def test_rejected_candidates_table_preserves_ordered_parts_and_issue_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_managed_cache(monkeypatch, tmp_path)
    source = annotation_result_fixture(tmp_path / "source", rejected_gene="cox2")
    qc = ov.qc.annotation(source)
    bundle = tmp_path / "bundle"
    ov.qc.write(qc, output=bundle)

    with (bundle / "rejected_cds_candidates.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["gene_name"] == "cox2"
    assert row["strand"] == "1"
    assert row["parts"] == "5|41|1"
    assert row["issue_codes"] == "premature_stop"
    assert "premature stop codon" in row["issue_messages"]


# ---------------------------------------------------------------------------
# Single-file views and XLSX
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "expected_name"),
    [
        (".md", "annotation_qc_summary.md"),
        (".json", "annotation_qc_report.json"),
        (".csv", "summary.csv"),
    ],
)
def test_single_file_views_equal_the_directory_view(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suffix: str, expected_name: str
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    directory = tmp_path / "directory"
    ov.qc.write(qc, output=directory)
    single = tmp_path / f"view{suffix}"

    ov.qc.write(qc, output=single)

    assert single.read_bytes() == (directory / expected_name).read_bytes()


def test_xlsx_contains_every_annotation_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    output = tmp_path / "report.xlsx"
    second = tmp_path / "second.xlsx"

    ov.qc.write(qc, output=output)
    ov.qc.write(qc, output=second)

    with zipfile.ZipFile(output) as workbook:
        xml = workbook.read("xl/workbook.xml").decode()
    for sheet in _XLSX_SHEETS:
        assert f'name="{sheet}"' in xml
    assert output.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("suffix", [".txt", ".tsv", ".html", ".bed"])
def test_unsupported_single_file_suffix_fails_without_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suffix: str
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    output = tmp_path / f"report{suffix}"

    with pytest.raises(OrganelleInputError) as captured:
        ov.qc.write(qc, output=output)
    assert captured.value.code == "qc.unsupported_output_suffix"
    assert not output.exists()


# ---------------------------------------------------------------------------
# Dispatch equivalence, reuse, refusal, atomicity
# ---------------------------------------------------------------------------


def test_direct_registry_agent_and_top_level_write_use_qc_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    registry = OperationRegistry()
    registry.register(QC_WRITE_SPEC, ov.qc.write)

    direct = ov.qc.write(qc, output=tmp_path / "direct")
    registered = registry.invoke("qc.write", input=qc, parameters={"output": tmp_path / "registry"})
    response = invoke_json(
        {
            "operation_id": "qc.write",
            "input": qc.model_dump(mode="json"),
            "parameters": {"output": str(tmp_path / "agent")},
        },
        registry=registry,
        granted_side_effects={SideEffect.READ_FILES, SideEffect.WRITE_FILES},
    )
    top_level = ov.write(qc, tmp_path / "top-level")

    assert response["ok"] is True
    assert direct.operation_id == registered.operation_id == top_level.operation_id == "qc.write"
    assert response["result"]["operation_id"] == "qc.write"
    for candidate in ("registry", "agent", "top-level"):
        assert (tmp_path / candidate / "checks.csv").read_bytes() == (
            tmp_path / "direct" / "checks.csv"
        ).read_bytes()


def test_identical_destination_is_byte_identical_and_reusable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    bundle = tmp_path / "bundle"

    first = ov.qc.write(qc, output=bundle)
    before = {
        path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()
    }
    second = ov.qc.write(qc, output=bundle)

    assert second == first
    assert before == {
        path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()
    }


def test_changed_destination_is_refused_and_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    bundle = tmp_path / "bundle"
    ov.qc.write(qc, output=bundle)
    (bundle / "checks.csv").write_text("changed\n")

    with pytest.raises(OrganelleInputError) as captured:
        ov.qc.write(qc, output=bundle)
    assert captured.value.code == "qc.destination_conflict"
    assert (bundle / "checks.csv").read_text() == "changed\n"


def test_writer_rejects_failed_and_non_annotation_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    written = ov.qc.write(qc, output=tmp_path / "ok")
    assert written.operation_version == "1.2"

    failed = OrganelleResult(
        operation_id="qc.annotation",
        scope="mitochondrion",
        status="failed",
        errors=({"code": "qc.annotation_failed", "message": "failed"},),
    )
    non_qc = OrganelleResult(
        operation_id="annotation.annotate",
        scope="mitochondrion",
        status="ok",
    )
    for candidate in (failed, non_qc):
        with pytest.raises(OrganelleInputError) as captured:
            ov.qc.write(candidate, output=tmp_path / candidate.operation_id)
        assert captured.value.code == "qc.write_input_contract"


def test_tampered_canonical_artifact_fails_before_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    report = next(item for item in qc.artifacts if item.kind == _REPORT_KIND)
    Path(report.uri).write_text('{"tampered": true}\n')
    bundle = tmp_path / "bundle"

    with pytest.raises(OrganelleInputError) as captured:
        ov.qc.write(qc, output=bundle)
    assert captured.value.code == "qc.artifact_digest_mismatch"
    assert not bundle.exists()


def test_report_manifest_identity_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    forged = _forge_manifest(qc, source_annotation_result_id="result:sha256:" + "f" * 64)
    bundle = tmp_path / "bundle"

    with pytest.raises(OrganelleInputError) as captured:
        ov.qc.write(forged, output=bundle)
    assert captured.value.code == "qc.artifact_contract_invalid"
    assert not bundle.exists()


def test_writer_rejects_extra_qc_result_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    extra_path = tmp_path / "extra.json"
    extra_path.write_text("{}\n")
    extra = ArtifactRef.from_path(
        extra_path,
        kind="unexpected",
        format="json",
        media_type="application/json",
    )
    forged = qc.model_copy(update={"artifacts": (*qc.artifacts, extra)})
    bundle = tmp_path / "bundle"

    with pytest.raises(OrganelleInputError) as captured:
        ov.qc.write(forged, output=bundle)

    assert captured.value.code == "qc.write_input_contract"
    assert not bundle.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary_text", "forged summary"),
        ("metrics", {"qc_decision": "needs_review", "forged": True}),
        ("flags", ("qc_needs_review", "forged")),
    ],
)
def test_writer_rejects_result_projection_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    forged = qc.model_copy(update={field: value})
    bundle = tmp_path / field

    with pytest.raises(OrganelleInputError) as captured:
        ov.qc.write(forged, output=bundle)

    assert captured.value.code == "qc.artifact_contract_invalid"
    assert not bundle.exists()


def test_writer_rejects_provenance_parameter_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    assert qc.provenance is not None
    provenance = qc.provenance.model_copy(update={"parameters_hash": "f" * 64})
    forged = qc.model_copy(update={"provenance": provenance})
    bundle = tmp_path / "bundle"

    with pytest.raises(OrganelleInputError) as captured:
        ov.qc.write(forged, output=bundle)

    assert captured.value.code == "qc.artifact_contract_invalid"
    assert not bundle.exists()


def test_writer_rejects_resigned_manifest_output_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    forged = _forge_manifest(qc, outputs=[])
    assert forged.provenance is not None
    manifest = _manifest(forged)
    provenance = forged.provenance.model_copy(update={"run_manifest_id": manifest.run_manifest_id})
    forged = forged.model_copy(update={"provenance": provenance})
    bundle = tmp_path / "bundle"

    with pytest.raises(OrganelleInputError) as captured:
        ov.qc.write(forged, output=bundle)

    assert captured.value.code == "qc.artifact_contract_invalid"
    assert not bundle.exists()


def test_failed_publish_rolls_back_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qc = _qc_result(monkeypatch, tmp_path)
    bundle = tmp_path / "bundle"
    parent = bundle.parent
    real_replace = ov.quality_control.writer.os.replace

    def fail_publish(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == bundle:
            raise OSError("injected publish failure")
        real_replace(source, destination)

    monkeypatch.setattr("organelleverse.quality_control.writer.os.replace", fail_publish)

    with pytest.raises(OrganelleInputError) as captured:
        ov.qc.write(qc, output=bundle)
    assert captured.value.code == "qc.destination_conflict"
    assert not bundle.exists()
    assert not any(path.name.startswith(".bundle.tmp-") for path in parent.iterdir())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _forge_manifest(result: OrganelleResult, **overrides: object) -> OrganelleResult:
    """Rewrite the on-disk manifest model and rebuild the Result artifact ref."""

    manifest_artifact = next(item for item in result.artifacts if item.kind == _MANIFEST_KIND)
    manifest_path = Path(manifest_artifact.uri)
    manifest = AnnotationQcRunManifest.model_validate_json(manifest_path.read_bytes())
    payload = cast(dict[str, object], manifest.model_dump(mode="json"))
    payload.update(overrides)
    payload.pop("object_id", None)
    forged = AnnotationQcRunManifest.model_validate(payload)
    manifest_path.write_bytes(canonical_json_bytes(forged.model_dump(mode="json")))
    rebuilt = ArtifactRef.from_path(
        manifest_path,
        kind=manifest_artifact.kind,
        format=manifest_artifact.format,
        media_type=manifest_artifact.media_type,
    )
    artifacts = tuple(rebuilt if item.kind == _MANIFEST_KIND else item for item in result.artifacts)
    return result.model_copy(update={"artifacts": artifacts})
