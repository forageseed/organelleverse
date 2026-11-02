"""classify_sequences: a sequence on the other strand, or split over contigs, is shared, not private."""

from __future__ import annotations

import random
from pathlib import Path

from organelleverse.io_genome import read_fasta_genome
from organelleverse.pangenome.pangenome import classify_sequences

_COMP = str.maketrans("ACGT", "TGCA")


def _dna(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _genome(tmp_path: Path, name: str, records: dict[str, str]):
    path = tmp_path / f"{name}.fasta"
    path.write_text("".join(f">{sid}\n{seq}\n" for sid, seq in records.items()))
    return read_fasta_genome(path, organelle="mitochondrion", species=name)


def test_reverse_strand_and_split_copies_are_core(tmp_path: Path) -> None:
    shared = _dna(4000, 1)
    genomes = [
        _genome(tmp_path, "forward", {"chr": shared}),
        _genome(tmp_path, "reverse", {"chr": shared.translate(_COMP)[::-1]}),
        # neither contig holds half of the sequence on its own
        _genome(tmp_path, "split", {"c1": shared[:1800], "c2": shared[1800:]}),
    ]
    metrics = classify_sequences(genomes).metrics
    assert (metrics["core"], metrics["variable"], metrics["private"]) == (4, 0, 0)


def test_unique_sequence_is_still_private(tmp_path: Path) -> None:
    shared = _dna(4000, 1)
    genomes = [
        _genome(tmp_path, "a", {"chr": shared, "extra": _dna(1500, 9)}),
        _genome(tmp_path, "b", {"chr": shared}),
    ]
    metrics = classify_sequences(genomes).metrics
    assert (metrics["core"], metrics["private"]) == (2, 1)
