from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.core.result import OrganelleResult
from organelleverse.quality_control.service import run_assembly_qc
from organelleverse.quality_control.writer import materialize_result
from tests.release._assembly_qc_gate import (
    _assert_bundle,
    _assert_source_backend,
    _assert_zero_coverage_assessed,
)

from .test_service import _install_deterministic_collectors
from .test_static import _publish_assembly_evidence


def test_release_gate_requires_the_claimed_backend(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)

    _assert_source_backend(published.result, "oatk")
    with pytest.raises(AssertionError):
        _assert_source_backend(published.result, "himt")


def test_release_gate_strictly_reparses_the_complete_bundle(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    qc = run_assembly_qc(published.result)
    written = materialize_result(qc, output=tmp_path / "bundle")

    report = _assert_bundle(
        tmp_path / "bundle",
        source=published.result,
        qc=qc,
        written=written,
    )

    assert report.decision == "insufficient_evidence"


def test_low_depth_gate_rejects_unassessed_zero_coverage() -> None:
    assessed = OrganelleResult(
        operation_id="qc.assembly",
        scope="mitochondrion",
        status="ok",
        metrics={
            "any_alignment_coverage_breadth": 1.0,
            "any_alignment_zero_coverage_bases": 0,
        },
    )
    unassessed = OrganelleResult(
        operation_id="qc.assembly",
        scope="mitochondrion",
        status="warning",
        metrics={
            "any_alignment_coverage_breadth": None,
            "any_alignment_zero_coverage_bases": None,
        },
    )

    _assert_zero_coverage_assessed(assessed, "PMAT2")
    with pytest.raises(AssertionError):
        _assert_zero_coverage_assessed(unassessed, "PMAT2")
