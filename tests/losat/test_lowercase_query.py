"""LOSAT loses every minus-strand hit for a lower-case nucleotide query; run_losat must not.

Seen on real data: the rice mitochondrion FASTA is lower-case, and searching it
against the nuclear genome returned half the NCBI blastn HSPs (776 vs 1,538),
missing a 46.7 kb, 99.9% identical reverse-strand NUMT.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from organelleverse import _losat
from organelleverse._losat import resolve_losat, run_losat

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _random_dna(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _write(path: Path, name: str, sequence: str, *, width: int = 60) -> Path:
    body = "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))
    path.write_text(f">{name}\n{body}\n", encoding="utf-8")
    return path


def _minus_strand_case(tmp_path: Path, *, lower_query: bool) -> tuple[Path, Path]:
    core = _random_dna(1500, seed=21)
    query = _random_dna(300, seed=22) + core + _random_dna(300, seed=23)
    subject = _random_dna(500, seed=24) + core.translate(_COMPLEMENT)[::-1] + _random_dna(500, seed=25)
    return (
        _write(tmp_path / "query.fa", "q", query.lower() if lower_query else query),
        _write(tmp_path / "subject.fa", "s", subject),
    )


needs_losat = pytest.mark.skipif(resolve_losat() is None, reason="LOSAT binary not available")


@needs_losat
@pytest.mark.parametrize("lower_query", [False, True])
def test_minus_strand_hit_is_found_whatever_the_query_case(tmp_path: Path, lower_query: bool) -> None:
    query, subject = _minus_strand_case(tmp_path, lower_query=lower_query)

    rows = run_losat("blastn", query, subject, evalue=1e-5)

    long_rows = [row for row in rows if int(row[3]) >= 1400]
    assert len(long_rows) == 1
    sstart, send = int(long_rows[0][8]), int(long_rows[0][9])
    assert sstart > send, "the subject copy is the reverse complement, so sstart > send"


@needs_losat
def test_soft_masked_query_keeps_its_minus_strand_hit(tmp_path: Path) -> None:
    """Soft-masking (a lower-case stretch inside an upper-case query) is the common real-world form."""
    query, subject = _minus_strand_case(tmp_path, lower_query=False)
    sequence = "".join(line for line in query.read_text().splitlines()[1:])
    masked = sequence[:700] + sequence[700:900].lower() + sequence[900:]
    _write(query, "q", masked)

    rows = run_losat("blastn", query, subject, evalue=1e-5)

    assert any(int(row[3]) >= 1400 for row in rows)


def test_uppercase_copy_is_used_only_for_lowercase_nucleotide_queries(tmp_path: Path) -> None:
    upper = _write(tmp_path / "upper.fa", "u", "ACGT" * 30)
    lower = _write(tmp_path / "lower.fa", "l", "acgt" * 30)

    with _losat._uppercase_query("blastn", upper) as same:
        assert same == upper
    with _losat._uppercase_query("blastp", lower) as untouched:  # protein query: never rewritten
        assert untouched == lower
    with _losat._uppercase_query("blastn", lower) as copy:
        assert copy != lower
        text = Path(copy).read_text()
        assert text.startswith(">l\n") and text.split("\n", 1)[1].replace("\n", "") == "ACGT" * 30
        kept = Path(copy)
    assert not kept.exists(), "the temporary copy is removed afterwards"
    assert lower.read_text().split("\n", 1)[1].replace("\n", "") == "acgt" * 30, "the input is never modified"


def test_unreadable_query_path_is_left_to_the_runner(tmp_path: Path) -> None:
    """Managed runners may receive paths this process cannot open; they must not raise here."""
    missing = tmp_path / "not_here.fa"
    with _losat._uppercase_query("blastn", missing) as same:
        assert same == missing
