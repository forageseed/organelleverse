"""barcode.identify must find a match regardless of strand and of reference length."""

from __future__ import annotations

import random
from pathlib import Path

from organelleverse.barcode.barcode import identify

_COMP = str.maketrans("ACGT", "TGCA")


def _dna(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mutate(seq: str, rate: float, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT".replace(b, "")) if rng.random() < rate else b for b in seq)


def _write(path: Path, records: dict[str, str]) -> Path:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records.items()))
    return path


def _refs(tmp_path: Path) -> tuple[Path, str]:
    target = _dna(6000, 1)
    refs = _write(tmp_path / "refs.fasta", {"Species_a": target, "Species_b": _dna(6000, 2), "Species_c": _dna(6000, 3)})
    return refs, target


def test_reverse_complement_query_is_identified(tmp_path: Path) -> None:
    refs, target = _refs(tmp_path)
    query = _write(tmp_path / "q.fasta", {"q": target[1000:1700].translate(_COMP)[::-1]})
    result = identify(query, refs)
    assert result.metrics["best_match"] == "Species_a"
    assert result.metrics["containment"] == 1.0
    assert "identified" in result.flags


def test_short_diverged_query_against_long_reference_is_identified(tmp_path: Path) -> None:
    refs, target = _refs(tmp_path)
    query = _write(tmp_path / "q.fasta", {"q": _mutate(target[2000:2650], 0.03, 7)})
    result = identify(query, refs)
    assert result.metrics["best_match"] == "Species_a"
    assert result.metrics["jaccard"] < 0.1  # the old Jaccard call missed this match
    assert "identified" in result.flags


def test_unrelated_query_is_not_identified(tmp_path: Path) -> None:
    refs, _ = _refs(tmp_path)
    query = _write(tmp_path / "q.fasta", {"q": _dna(700, 99)})
    result = identify(query, refs)
    assert "identified" not in result.flags
