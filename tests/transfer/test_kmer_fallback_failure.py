"""With no aligner the k-mer scan runs; when it fails the failure is returned, not a crash."""

from __future__ import annotations

from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.transfer import transfer_core


@pytest.fixture
def no_aligner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transfer_core, "alignment_transfer", lambda *args, **kwargs: None)


def _fasta(path: Path, sequence: str) -> Path:
    path.write_text(f">s\n{sequence}\n", encoding="utf-8")
    return path


def test_detect_numt_reports_sequences_shorter_than_k_when_no_aligner(
    no_aligner: None, tmp_path: Path
) -> None:
    nuclear = _fasta(tmp_path / "nuclear.fasta", "ACGT" * 7)
    mito = _fasta(tmp_path / "mito.fasta", "ACGT" * 5)

    result = ov.population.detect_numt(nuclear, mito)

    assert result.status == "failed"
    assert result.errors[0].code.endswith("sequences_too_short")


def test_detect_mtpt_reports_sequences_shorter_than_k_when_no_aligner(
    no_aligner: None, tmp_path: Path
) -> None:
    mito = _fasta(tmp_path / "mito.fasta", "ACGT" * 7)
    plastid = _fasta(tmp_path / "plastid.fasta", "ACGT" * 5)

    result = ov.transfer.detect_mtpt(mito, plastid)

    assert result.status == "failed"
