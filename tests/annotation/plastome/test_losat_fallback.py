"""A failing LOSAT run falls back to NCBI BLAST+ instead of failing the plastome annotation."""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.annotation.plastome import blast
from organelleverse.annotation.plastome.models import BlastTools
from organelleverse.core.errors import OrganelleExecutionError


def _tools(**kw):
    base = {"blastn": "/usr/bin/blastn", "makeblastdb": "/usr/bin/makeblastdb", "tblastn": "/usr/bin/tblastn",
            "losat": "/opt/LOSAT"}
    base.update(kw)
    return BlastTools(**base)


def _fail_with_losat(calls):
    def fake(target_seq, queries, *, work_dir, tools, min_identity, qcoverage_range, threads=1):
        calls.append(tools.losat)
        if tools.losat is not None:
            raise OrganelleExecutionError(code="external_command_failed", message="LOSAT failed with exit code -6")
        return {"q1": "hit"}
    return fake


def test_losat_failure_repeats_with_ncbi(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(blast, "_run_plastome_blasts", _fail_with_losat(calls))
    hits = blast.run_plastome_blasts("ACGT", (), work_dir=tmp_path, tools=_tools(), min_identity=0.6,
                                     qcoverage_range=(0.5, 1.0))
    assert hits == {"q1": "hit"}
    assert calls == ["/opt/LOSAT", None]


def test_losat_failure_without_ncbi_still_raises(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(blast, "_run_plastome_blasts", _fail_with_losat(calls))
    with pytest.raises(OrganelleExecutionError):
        blast.run_plastome_blasts("ACGT", (), work_dir=tmp_path, tools=_tools(blastn=None), min_identity=0.6,
                                  qcoverage_range=(0.5, 1.0))
    assert calls == ["/opt/LOSAT"]


def test_ir_search_falls_back_too(monkeypatch, tmp_path: Path):
    calls = []

    def fake(target_seq, *, work_dir, tools, min_ir_length, threads=1):
        calls.append(tools.losat)
        if tools.losat is not None:
            raise OrganelleExecutionError(code="external_command_failed", message="LOSAT failed")
        return ((1, 10, 1), (20, 29, -1))

    monkeypatch.setattr(blast, "_detect_ir_regions", fake)
    got = blast.detect_ir_regions("ACGT", work_dir=tmp_path, tools=_tools(tblastn=None), min_ir_length=10)
    assert got == ((1, 10, 1), (20, 29, -1))
    assert calls == ["/opt/LOSAT", None]
