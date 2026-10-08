"""fasta_sequence_lengths reads bytes now; the record ids and lengths must not change."""

from __future__ import annotations

from pathlib import Path

from organelleverse._losat import fasta_sequence_lengths


def test_lengths_ignore_blank_lines_crlf_descriptions_and_case(tmp_path: Path) -> None:
    path = tmp_path / "x.fa"
    path.write_bytes(
        b">chr1 first record\r\nACGT\r\nacgtN\r\n\r\n>chr2\nAC\nGT\n>\n  \n>chr3 \tdesc\nTTTT\n"
    )
    assert fasta_sequence_lengths(path) == {"chr1": 9, "chr2": 4, "": 0, "chr3": 4}


def test_non_ascii_header_does_not_raise(tmp_path: Path) -> None:
    path = tmp_path / "u.fa"
    path.write_bytes(">séq1 描述\nACGT\n".encode("utf-8"))
    assert fasta_sequence_lengths(path) == {"séq1": 4}
