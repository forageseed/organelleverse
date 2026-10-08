"""Maximal dispersed repeats against synthetic and real mitochondrial DNA."""

from pathlib import Path

from organelleverse._bio import read_fasta
from organelleverse._sequtil import reverse_complement
from organelleverse.structure.structure import _find_repeats
from organelleverse.structure.structure_core import compute_multiconf


def test_merges_adjacent_kmer_hits_into_one_direct_repeat() -> None:
    block = "ACGTTGCACTAGGATCCGTACGATCGTTACGGCATC"
    repeats = _find_repeats(block + "N" * 20 + block, 10)
    assert {tuple(r["positions"]): r["length"] for r in repeats if r["type"] == "direct"} == {
        (1, len(block) + 21): len(block)
    }


def test_merges_inverted_hits_in_opposite_direction() -> None:
    block = "ACGTTGCACTAGGATCCGTACGATCGTTACGGCATC"
    repeats = _find_repeats(block + "N" * 20 + reverse_complement(block), 10)
    assert any(
        r["type"] == "inverted"
        and r["positions"] == [1, len(block) + 21]
        and r["length"] == len(block)
        for r in repeats
    )


def test_can_bridge_one_mismatch_on_same_diagonal(tmp_path: Path) -> None:
    block = "ACGTTGCACTAGGATCCGTACGATCGTTACGGCATC"
    changed = block[:18] + ("A" if block[18] != "A" else "C") + block[19:]
    seq = block + "N" * 20 + changed
    exact = _find_repeats(seq, 10)
    approximate = _find_repeats(seq, 10, max_mismatches=1)
    assert not any(r["length"] == len(block) for r in exact)
    assert any(
        r["type"] == "direct"
        and r["positions"] == [1, len(block) + 21]
        and r["length"] == len(block)
        and r["mismatches"] == 1
        for r in approximate
    )
    fasta = tmp_path / "one-mismatch.fasta"
    fasta.write_text(f">demo\n{seq}\n", encoding="ascii")
    assert any(r["length"] == len(block) for r in compute_multiconf(fasta, 10, 1)["repeats"])


def test_can_bridge_one_mismatch_in_inverted_repeat() -> None:
    block = "ACGTTGCACTAGGATCCGTACGATCGTTACGGCATC"
    inverted = reverse_complement(block)
    changed = inverted[:18] + ("A" if inverted[18] != "A" else "C") + inverted[19:]
    repeats = _find_repeats(block + "N" * 20 + changed, 10, max_mismatches=1)
    assert any(
        r["type"] == "inverted"
        and r["positions"] == [1, len(block) + 21]
        and r["length"] == len(block)
        and r["mismatches"] == 1
        for r in repeats
    )


def test_nc_037304_matches_reported_repeat_pairs() -> None:
    reference = (
        Path(__file__).resolve().parents[2]
        / "examples/data/arabidopsis_mitochondrion.fasta"
    )
    identifier, sequence = read_fasta(reference)[0]
    assert identifier == "NC_037304.1"
    result = compute_multiconf(reference)
    assert result["direct_repeat_pairs"] == 39
    assert result["inverted_repeat_pairs"] == 41
    assert len(result["repeats"]) == 80
    for repeat in result["repeats"]:
        first, second = (position - 1 for position in repeat["positions"])
        length = repeat["length"]
        assert first + length <= second
        first_copy = sequence[first : first + length]
        second_copy = sequence[second : second + length]
        assert first_copy == (
            second_copy if repeat["type"] == "direct" else reverse_complement(second_copy)
        )
    assert [
        (r["type"], r["length"]) for r in sorted(result["repeats"], key=lambda r: -r["length"])[:2]
    ] == [
        ("inverted", 6590),
        ("direct", 4193),
    ]
