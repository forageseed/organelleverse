from __future__ import annotations

from pathlib import Path

import organelleverse as ov


def _fasta(path: Path, *records: tuple[str, str]) -> Path:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records), encoding="utf-8")
    return path


def test_align_takes_what_read_returned_like_the_path(tmp_path: Path) -> None:
    genes = _fasta(tmp_path / "genes.fasta", ("g1", "ACGTACGT"), ("g2", "ACGTTCGT"))

    from_path = ov.phylogeny.align(genes)
    from_read = ov.phylogeny.align(ov.phylogeny.read(genes))

    assert from_read.status == from_path.status
    assert from_read.summary_text == from_path.summary_text


def test_build_tree_refuses_sequences_that_are_not_an_alignment(tmp_path: Path) -> None:
    unaligned = _fasta(tmp_path / "u.fasta", ("a", "ACGT"), ("b", "ACG"), ("c", "ACGTT"))

    for source in (unaligned, ov.phylogeny.read(unaligned)):
        result = ov.phylogeny.build_tree(source)

        assert result.status == "failed"
        assert result.errors[0].code == "phylogeny.build_tree.not_aligned"


def test_build_tree_plans_an_alignment_from_read_or_path(tmp_path: Path) -> None:
    aligned = _fasta(tmp_path / "a.fasta", ("a", "ACGT"), ("b", "AC-T"), ("c", "ACGA"))

    from_path = ov.phylogeny.build_tree(aligned, method="fasttree")
    from_read = ov.phylogeny.build_tree(ov.phylogeny.read(aligned), method="fasttree")

    assert from_path.status == from_read.status == "ok"
    assert from_read.metrics["argv"] == from_path.metrics["argv"]
    assert str(aligned) in from_read.metrics["argv"]


def test_trim_alignment_refuses_unaligned_sequences(tmp_path: Path) -> None:
    unaligned = _fasta(tmp_path / "u.fasta", ("a", "ACGT"), ("b", "ACG"))

    result = ov.phylogeny.trim_alignment(ov.phylogeny.read(unaligned))

    assert result.status == "failed"
    assert result.errors[0].code == "phylogeny.trim_alignment.not_aligned"


def test_haplotype_network_takes_what_read_returned(tmp_path: Path) -> None:
    aligned = _fasta(
        tmp_path / "h.fasta",
        ("s1", "ACGTACGTAC"),
        ("s2", "ACGTACGTAC"),
        ("s3", "ACGTTCGTAC"),
        ("s4", "ACGTTCGAAC"),
    )

    from_path = ov.phylogeny.haplotype_network(aligned)
    from_read = ov.phylogeny.haplotype_network(ov.phylogeny.read(aligned))

    assert from_read.status == from_path.status == "ok"
    assert from_read.metrics["n_haplotypes"] == from_path.metrics["n_haplotypes"]
