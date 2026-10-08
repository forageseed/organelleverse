from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.result import OrganelleResult
from organelleverse.quality_control.contracts import ErrorCandidate
from organelleverse.quality_control.service import run_assembly_qc
from organelleverse.quality_control.writer import _write_error_bed, materialize_result

from .test_service import _install_deterministic_collectors
from .test_static import _publish_assembly_evidence


def _qc_result(monkeypatch, tmp_path: Path) -> OrganelleResult:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    return run_assembly_qc(published.result)


def test_directory_bundle_contains_human_and_machine_outputs(monkeypatch, tmp_path: Path) -> None:
    result = _qc_result(monkeypatch, tmp_path)
    output = tmp_path / "published"

    written = materialize_result(result, output=output)

    expected = {
        "assembly_qc_summary.md",
        "assembly_qc_report.json",
        "assembly_qc_run_manifest.json",
        "summary.csv",
        "checks.csv",
        "sequences.csv",
        "junction_support.csv",
        "repeat_support.csv",
        "marker_hits.csv",
        "error_candidates.csv",
        "coverage_windows.csv",
        "assembly_statistics.csv",
        "library_mapping_summaries.csv",
        "gfa_summary.csv",
        "alternative_configurations.csv",
        "marker_profile_assessment.csv",
        "base_error_candidates.bed",
        "structural_error_candidates.bed",
        "kmer_metrics.tsv",
        "coverage_any_alignment.bed.gz",
        "coverage_confident_primary.bed.gz",
        "logs",
    }
    assert {item.name for item in output.iterdir()} == expected
    assert written.operation_id == "qc.write"
    assert written.status == result.status
    assert written.provenance is not None
    assert written.provenance.input_object_ids == (result.object_id,)
    assert "score" not in (output / "assembly_qc_summary.md").read_text().lower()
    with (output / "checks.csv").open(newline="") as handle:
        checks = list(csv.DictReader(handle))
    assert checks
    assert "assessment_status" in checks[0]
    with (output / "repeat_support.csv").open(newline="") as handle:
        repeats = list(csv.DictReader(handle))
    assert repeats == [
        {
            "repeat_id": "",
            "sequence_id": "",
            "length": "",
            "orientation": "",
            "path_occurrences": "",
            "spanning_reads": "",
            "status": "",
            "assessment_status": "not_assessed",
        }
    ]
    assert (output / "base_error_candidates.bed").read_bytes() == b""
    assert (output / "structural_error_candidates.bed").read_bytes() == b""
    assert (output / "kmer_metrics.tsv").read_text() == (
        "metric\tvalue\nassembly_kmer_support_fraction\t0.75\n"
    )


@pytest.mark.parametrize(
    ("suffix", "expected_name"),
    [
        (".md", "assembly_qc_summary.md"),
        (".json", "assembly_qc_report.json"),
        (".csv", "checks.csv"),
    ],
)
def test_single_file_views_equal_the_directory_view(
    monkeypatch, tmp_path: Path, suffix: str, expected_name: str
) -> None:
    result = _qc_result(monkeypatch, tmp_path)
    directory = tmp_path / "directory"
    materialize_result(result, output=directory)
    single = tmp_path / f"view{suffix}"

    materialize_result(result, output=single)

    assert single.read_bytes() == (directory / expected_name).read_bytes()


def test_xlsx_contains_every_evidence_table(monkeypatch, tmp_path: Path) -> None:
    result = _qc_result(monkeypatch, tmp_path)
    output = tmp_path / "report.xlsx"
    second = tmp_path / "second.xlsx"

    materialize_result(result, output=output)
    materialize_result(result, output=second)

    with zipfile.ZipFile(output) as workbook:
        xml = workbook.read("xl/workbook.xml").decode()
    for sheet in (
        "summary",
        "checks",
        "sequences",
        "junction_support",
        "repeat_support",
        "marker_hits",
        "error_candidates",
        "coverage_windows",
        "assembly_statistics",
        "library_mapping_summaries",
        "gfa_summary",
        "alternative_configurations",
        "marker_profile_assessment",
    ):
        assert f'name="{sheet}"' in xml
    assert output.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("suffix", [".txt", ".tsv", ".html", ".bed"])
def test_unsupported_single_file_suffix_fails_without_output(
    monkeypatch, tmp_path: Path, suffix: str
) -> None:
    result = _qc_result(monkeypatch, tmp_path)
    output = tmp_path / f"report{suffix}"

    with pytest.raises(OrganelleInputError) as captured:
        materialize_result(result, output=output)
    assert captured.value.code == "qc.unsupported_output_suffix"
    assert not output.exists()


def test_writer_rejects_non_qc_failed_and_second_generation_results(
    monkeypatch, tmp_path: Path
) -> None:
    qc_result = _qc_result(monkeypatch, tmp_path)
    non_qc = OrganelleResult(
        operation_id="assembly.assemble",
        scope="mitochondrion",
        status="ok",
    )
    failed = OrganelleResult(
        operation_id="qc.assembly",
        scope="mitochondrion",
        status="failed",
        errors=(
            {
                "code": "qc.mapping_failed",
                "message": "failed",
            },
        ),
    )
    written = materialize_result(qc_result, output=tmp_path / "first")
    assert written.operation_version == "1.2"

    for candidate in (non_qc, failed, written):
        with pytest.raises(OrganelleInputError) as captured:
            materialize_result(candidate, output=tmp_path / candidate.operation_id)
        assert captured.value.code == "qc.write_input_contract"


def test_assembly_writer_bundle_is_unchanged_by_annotation_dispatch(
    monkeypatch, tmp_path: Path
) -> None:
    result = _qc_result(monkeypatch, tmp_path)
    output = tmp_path / "assembly"

    written = materialize_result(result, output=output)

    assert written.operation_id == "qc.write"
    assert written.summary_text == "Materialized assembly QC output."
    expected = {
        "assembly_qc_summary.md",
        "assembly_qc_report.json",
        "assembly_qc_run_manifest.json",
        "summary.csv",
        "checks.csv",
        "sequences.csv",
        "junction_support.csv",
        "repeat_support.csv",
        "marker_hits.csv",
        "error_candidates.csv",
        "coverage_windows.csv",
        "assembly_statistics.csv",
        "library_mapping_summaries.csv",
        "gfa_summary.csv",
        "alternative_configurations.csv",
        "marker_profile_assessment.csv",
        "base_error_candidates.bed",
        "structural_error_candidates.bed",
        "kmer_metrics.tsv",
        "coverage_any_alignment.bed.gz",
        "coverage_confident_primary.bed.gz",
        "logs",
    }
    assert {item.name for item in output.iterdir()} == expected


def test_structural_error_projection_is_valid_bed6(tmp_path: Path) -> None:
    output = tmp_path / "errors.bed"
    candidate = ErrorCandidate(
        candidate_id="ctg1:small_collapse:1",
        sequence_id="ctg1",
        start=49,
        end=50,
        error_type="small_collapse",
        supporting_observations=3,
        status="candidate",
    )

    _write_error_bed((candidate,), output)

    assert output.read_text() == (
        "ctg1\t49\t50\tctg1:small_collapse:1|small_collapse|candidate\t3\t.\n"
    )


def test_tampered_canonical_artifact_fails_before_destination(monkeypatch, tmp_path: Path) -> None:
    result = _qc_result(monkeypatch, tmp_path)
    report = next(item for item in result.artifacts if item.kind == "assembly_qc_report")
    Path(report.uri).write_text('{"tampered":true}\n')
    output = tmp_path / "published"

    with pytest.raises(OrganelleInputError) as captured:
        materialize_result(result, output=output)
    assert captured.value.code == "qc.artifact_digest_mismatch"
    assert not output.exists()


def test_identical_destination_is_idempotent_but_conflict_is_preserved(
    monkeypatch, tmp_path: Path
) -> None:
    result = _qc_result(monkeypatch, tmp_path)
    output = tmp_path / "published"
    first = materialize_result(result, output=output)
    before = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }

    second = materialize_result(result, output=output)

    assert second == first
    assert before == {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    (output / "checks.csv").write_text("changed\n")
    with pytest.raises(OrganelleInputError) as captured:
        materialize_result(result, output=output)
    assert captured.value.code == "qc.destination_conflict"
    assert (output / "checks.csv").read_text() == "changed\n"
