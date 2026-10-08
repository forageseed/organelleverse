"""Shared, environment-controlled assembly QC release gate.

The four backend gates (Oatk, HiMT, GetOrganelle, PMAT2) do **not** re-run an
assembly backend. Each consumes a real, already-produced assembly ``Result``
(serialized ``result.json`` from that backend's assembly release gate) and pipes
it through ``qc.assembly -> qc.write(directory)`` then reparses every output.

Because a real Result and real mapping tools are not available in every
environment, the gate is **environment-controlled**: it skips explicitly with a
message that states what it needs, rather than fabricating a pass or running a
backend. Run it only in the release environment where the assembly Result and
``minimap2``/``samtools`` are present::

    ORGANELLEVERSE_QC_OATK_RESULT=/path/to/oatk_mito_result.json \\
        pytest -m release_assembly_qc -q
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.result import OrganelleResult
from organelleverse.core.serialization import load_result
from organelleverse.quality_control.contracts import (
    AssemblyQcReport,
    AssemblyQcRunManifest,
)

_DIRECTORY_BUNDLE_FILES = (
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
    "coverage_any_alignment.bed.gz",
    "coverage_confident_primary.bed.gz",
    "logs",
)


def _require_real_result(env_var: str, backend: str) -> Path:
    raw = os.environ.get(env_var)
    if not raw:
        pytest.skip(
            f"assembly QC release gate for {backend} is environment-controlled "
            f"and was not run: set {env_var} to a real result.json produced by "
            f"the {backend} assembly release gate. The QC gate reuses that "
            f"Result and does not re-run the backend."
        )
    path = Path(raw)
    if not path.is_file():
        pytest.fail(f"assembly QC release gate for {backend}: {env_var}={path} is not a file")
    return path


def _require_mapping_tools() -> None:
    missing = [tool for tool in ("minimap2", "samtools") if shutil.which(tool) is None]
    if missing:
        pytest.fail(f"assembly QC release gate requires on PATH: {', '.join(missing)}")


def _assert_source_backend(source: OrganelleResult, expected_backend: str) -> None:
    assert source.operation_id == "assembly.assemble"
    assert source.status != "failed"
    assert source.provenance is not None
    assert source.provenance.actual_backend == expected_backend
    assert source.provenance.attempted_backends == (expected_backend,)


def _assert_bundle(
    bundle: Path,
    *,
    source: OrganelleResult,
    qc: OrganelleResult,
    written: OrganelleResult,
) -> AssemblyQcReport:
    missing = [name for name in _DIRECTORY_BUNDLE_FILES if not (bundle / name).exists()]
    assert not missing, f"QC bundle missing files: {missing}"
    assert (bundle / "logs").is_dir()

    report = AssemblyQcReport.model_validate_json((bundle / "assembly_qc_report.json").read_bytes())
    manifest = AssemblyQcRunManifest.model_validate_json(
        (bundle / "assembly_qc_run_manifest.json").read_bytes()
    )
    assert report.source_assembly_result_id == source.object_id
    assert report.source_run_manifest_id == manifest.source_run_manifest_id
    assert report.decision == qc.metrics["qc_decision"]
    assert manifest.source_assembly_result_id == source.object_id
    assert qc.provenance is not None
    assert qc.provenance.run_manifest_id == manifest.run_manifest_id
    assert written.provenance is not None
    assert written.provenance.input_object_ids == (qc.object_id,)

    markdown = (bundle / "assembly_qc_summary.md").read_text()
    assert markdown.startswith("# Assembly QC summary\n")
    assert "score" not in markdown.lower()
    for filename in _DIRECTORY_BUNDLE_FILES:
        if not filename.endswith(".csv"):
            continue
        with (bundle / filename).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, f"{filename} must contain data or an explicit not_assessed row"
        assert "assessment_status" in rows[0], filename

    files = {path.resolve() for path in bundle.rglob("*") if path.is_file()}
    recorded = {Path(artifact.uri).resolve() for artifact in written.artifacts}
    assert recorded == files
    for artifact in written.artifacts:
        current = ArtifactRef.from_path(
            artifact.uri,
            kind=artifact.kind,
            format=artifact.format,
            media_type=artifact.media_type,
        )
        assert current.sha256 == artifact.sha256
        assert current.size_bytes == artifact.size_bytes
    return report


def _assert_zero_coverage_assessed(qc: OrganelleResult, backend: str) -> None:
    assert qc.metrics["any_alignment_coverage_breadth"] is not None
    assert qc.metrics["any_alignment_zero_coverage_bases"] == 0, (
        f"{backend} fixture has artificial zero-coverage bases: "
        f"{qc.metrics['any_alignment_zero_coverage_bases']}"
    )


def run_assembly_qc_release_gate(
    tmp_path: Path,
    *,
    backend: str,
    expected_backend: str,
    env_var: str,
    require_zero_coverage_assessed: bool = False,
) -> None:
    """Pipe one real assembly Result through the QC release chain and reparse it."""

    result_path = _require_real_result(env_var, backend)
    _require_mapping_tools()

    source = load_result(result_path)
    _assert_source_backend(source, expected_backend)

    qc = ov.qc.assembly(source)
    assert qc.operation_id == "qc.assembly"
    assert qc.status != "failed", qc.summary_text

    bundle = tmp_path / "bundle"
    written = ov.qc.write(qc, output=bundle)
    assert written.operation_id == "qc.write"
    assert written.status == qc.status

    _assert_bundle(bundle, source=source, qc=qc, written=written)

    decision = qc.metrics["qc_decision"]
    assert decision in {"ready", "needs_review"}, (
        f"{backend} fixture produced a publication-blocking decision: {decision}"
    )

    if require_zero_coverage_assessed:
        # Low-depth PMAT2 HiFi: policy mapping filters must not manufacture a
        # zero-coverage interval that the underlying reads do cover.
        _assert_zero_coverage_assessed(qc, backend)
