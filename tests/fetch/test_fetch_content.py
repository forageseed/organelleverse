"""Content-integrity checks: catch a 200-status error page wearing a genome's extension."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.fetch._content import SHARED_INCLUDE_KINDS, validate_downloaded_content


def test_shared_include_kinds_covers_the_full_cross_source_vocabulary() -> None:
    assert {
        "protein",
        "cds",
        "gff3",
        "genome",
        "tpm",
        "rna",
        "seq-report",
    } == SHARED_INCLUDE_KINDS


def test_a_real_gzipped_fasta_passes(tmp_path: Path) -> None:
    path = tmp_path / "genome.fa.gz"
    path.write_bytes(gzip.compress(b">seq1\nACGTACGT\n"))
    validate_downloaded_content(path, kind="genome", gzipped=True, source="pgd")


def test_an_html_error_page_saved_as_a_fasta_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "genome.fa.gz"
    path.write_bytes(gzip.compress(b"<html><body>blocked</body></html>"))
    with pytest.raises(OrganelleExecutionError) as raised:
        validate_downloaded_content(path, kind="genome", gzipped=True, source="pgd")
    assert raised.value.code == "network.pgd.content_invalid"
    assert raised.value.retryable is True


def test_a_real_gff_passes(tmp_path: Path) -> None:
    path = tmp_path / "genes.gff3"
    path.write_bytes(b"##gff-version 3\nChr1\tsrc\tgene\t1\t100\t.\t+\t.\tID=g1\n")
    validate_downloaded_content(path, kind="gff3", gzipped=False, source="gir")


def test_a_non_gff_body_is_rejected_for_gff_kind(tmp_path: Path) -> None:
    path = tmp_path / "genes.gff3"
    path.write_bytes(b"not gff content at all")
    with pytest.raises(OrganelleExecutionError) as raised:
        validate_downloaded_content(path, kind="gff3", gzipped=False, source="gir")
    assert raised.value.code == "network.gir.content_invalid"


def test_an_ungzipped_fasta_is_checked_directly(tmp_path: Path) -> None:
    path = tmp_path / "prot.fasta"
    path.write_bytes(b">AT1G01010.1\nMEEQVGFGFRPNDEEL\n")
    validate_downloaded_content(path, kind="protein", gzipped=False, source="imp")
