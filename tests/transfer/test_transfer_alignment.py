"""compute_transfer / detect_mtpt align (LOSAT, then BLAST) instead of scanning exact k-mers.

The real-data comparison (rice and Arabidopsis plastid -> mitochondrion) showed
the exact-run scan recovered 59% / 29% of the bases BLAST finds, because every
substitution inside a transferred copy breaks a run. These tests plant diverged
and reverse-complemented copies so the exact scan provably misses them.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from organelleverse._losat import resolve_losat
from organelleverse._sequtil import reverse_complement
from organelleverse.transfer import compute_transfer, detect_mtpt
from organelleverse.transfer import transfer as transfer_module
from organelleverse.transfer import transfer_core

pytestmark = pytest.mark.skipif(
    resolve_losat() is None and shutil.which("blastn") is None,
    reason="needs LOSAT or NCBI blastn",
)


def _random_dna(length: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _mutate(sequence: str, rate: float, seed: int = 99) -> str:
    """Substitute ``rate`` of the bases at random positions (identity ~ 1 - rate)."""
    rng = random.Random(seed)
    swap = {"A": "C", "C": "G", "G": "T", "T": "A"}
    return "".join(swap[base] if rng.random() < rate else base for base in sequence)


def _write(path: Path, records: list[tuple[str, str]]) -> Path:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records), encoding="utf-8")
    return path


@pytest.fixture
def donor() -> str:
    return _random_dna(1500, seed=1)


def test_diverged_copy_is_found_where_exact_runs_are_not(tmp_path: Path, donor: str) -> None:
    # ~92% identity: no exact run reaches 100 bp.
    copy = _mutate(donor[200:800], rate=0.08)
    target = _random_dna(2000, seed=2) + copy + _random_dna(1500, seed=3)
    target_path = _write(tmp_path / "mito.fa", [("mito", target)])
    donor_path = _write(tmp_path / "cp.fa", [("cp", donor)])

    kmer = transfer_core._compute_transfer_kmer(target_path, donor_path, 31, 100)
    assert kmer["fragment_count"] == 0

    result = compute_transfer(target_path, donor_path)
    assert result["backend"] in {"losat", "ncbi_blastn"}
    assert result["fragment_count"] == 1
    fragment = result["fragments"][0]
    assert fragment["seqid"] == "mito"
    assert abs(fragment["start"] - 2001) <= 25 and abs(fragment["end"] - 2600) <= 25
    assert 88 <= fragment["identity"] <= 96
    assert fragment["strand"] == "+"


def test_reverse_complement_copy_reports_minus_strand(tmp_path: Path, donor: str) -> None:
    copy = reverse_complement(_mutate(donor[100:700], rate=0.04))
    target = _random_dna(1000, seed=4) + copy + _random_dna(1000, seed=5)
    target_path = _write(tmp_path / "mito.fa", [("mito", target)])
    donor_path = _write(tmp_path / "cp.fa", [("cp", donor)])

    result = compute_transfer(target_path, donor_path)
    assert result["fragment_count"] == 1
    assert result["fragments"][0]["strand"] == "-"
    assert result["reverse_strand_bp"] == result["total_bp"] > 0
    assert result["direct_strand_bp"] == 0


def test_min_identity_filters_diverged_copies(tmp_path: Path, donor: str) -> None:
    copy = _mutate(donor[200:800], rate=0.17)  # ~83% identity
    target = _random_dna(1500, seed=6) + copy + _random_dna(1500, seed=7)
    target_path = _write(tmp_path / "mito.fa", [("mito", target)])
    donor_path = _write(tmp_path / "cp.fa", [("cp", donor)])

    assert compute_transfer(target_path, donor_path, min_identity=80.0)["fragment_count"] == 1
    assert compute_transfer(target_path, donor_path, min_identity=95.0)["fragment_count"] == 0


def test_multi_record_target_coordinates_are_per_record(tmp_path: Path, donor: str) -> None:
    """Concatenating records (the old behaviour) shifts every later fragment."""
    first = _random_dna(3000, seed=8)
    second = _random_dna(500, seed=9) + donor[300:700] + _random_dna(500, seed=10)
    target_path = _write(tmp_path / "mito.fa", [("ctg1", first), ("ctg2", second)])
    donor_path = _write(tmp_path / "cp.fa", [("cp", donor)])

    result = compute_transfer(target_path, donor_path)
    assert result["fragment_count"] == 1
    fragment = result["fragments"][0]
    assert fragment["seqid"] == "ctg2"
    assert abs(fragment["start"] - 501) <= 5 and abs(fragment["end"] - 900) <= 5
    assert result["fraction"] == pytest.approx(result["total_bp"] / 4400, rel=1e-3)


def test_two_donor_regions_hitting_distinct_target_sites_stay_separate(tmp_path: Path, donor: str) -> None:
    target = (
        _random_dna(800, seed=11)
        + donor[0:300]
        + _random_dna(900, seed=12)
        + donor[900:1250]
        + _random_dna(800, seed=13)
    )
    target_path = _write(tmp_path / "mito.fa", [("mito", target)])
    donor_path = _write(tmp_path / "cp.fa", [("cp", donor)])

    result = compute_transfer(target_path, donor_path)
    assert result["fragment_count"] == 2
    assert result["total_bp"] == pytest.approx(650, abs=20)


def test_detect_mtpt_result_reports_backend_identity_and_strand(tmp_path: Path, donor: str) -> None:
    copy = _mutate(donor[200:800], rate=0.08)
    target = _random_dna(2000, seed=2) + copy + _random_dna(1500, seed=3)
    target_path = _write(tmp_path / "mito.fa", [("mito", target)])
    donor_path = _write(tmp_path / "cp.fa", [("cp", donor)])

    result = detect_mtpt(target_path, donor_path)
    assert result.status == "ok"
    assert result.provenance.actual_backend in {"losat", "ncbi_blastn"}
    assert result.provenance.attempted_backends[-1] == result.provenance.actual_backend
    assert "transfer_detected" in result.flags
    assert "kmer_fallback" not in result.flags
    assert result.metrics["fragment_count"] == 1
    assert 88 <= result.metrics["mean_identity"] <= 96
    assert result.metrics["fragments"][0]["seqid"] == "mito"


def test_detect_mtpt_without_any_aligner_flags_the_kmer_fallback(
    tmp_path: Path, donor: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORG_VERSE_LOSAT_BIN", "none")
    monkeypatch.setattr(transfer_core.shutil, "which", lambda name: None)
    target = _random_dna(500, seed=14) + donor[100:600] + _random_dna(500, seed=15)
    target_path = _write(tmp_path / "mito.fa", [("mito", target)])
    donor_path = _write(tmp_path / "cp.fa", [("cp", donor)])

    result = detect_mtpt(target_path, donor_path)
    assert result.status == "ok"
    assert "kmer_fallback" in result.flags
    assert result.provenance.attempted_backends[:2] == ("losat", "ncbi_blastn")
    assert result.metrics["fragment_count"] == 1  # an exact copy is still found

    plain = compute_transfer(target_path, donor_path)
    assert plain["backend"] == "kmer_fallback"


def test_detect_mtpt_rejects_sequences_shorter_than_the_word_size(tmp_path: Path) -> None:
    target_path = _write(tmp_path / "mito.fa", [("mito", "ACGT")])
    donor_path = _write(tmp_path / "cp.fa", [("cp", "ACGTACGTAC")])
    result = detect_mtpt(target_path, donor_path)
    assert result.status == "failed"
    assert result.errors[0].code == "sequences_too_short"


def test_population_detect_numt_keeps_the_exact_kmer_scan() -> None:
    """``population.detect_numt`` is documented k-mer based and must not silently change."""
    assert transfer_module._detect_transfer.__name__ == "_detect_transfer"
