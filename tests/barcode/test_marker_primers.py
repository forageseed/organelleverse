"""Marker primer design on hypervariable regions — primer3-backed behavior tests."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from organelleverse._bio import write_fasta
from organelleverse.barcode import primers
from organelleverse.barcode.primers import (
    _conserved,
    _sample_mismatches,
    design_marker_primers,
)
from organelleverse.core.errors import OrganelleDependencyError


def _alignment(path: Path, *, snps: dict[int, dict[int, str]] | None = None) -> list[str]:
    """4 samples x 400 bp; conserved flanks, variable core at columns 161-240.

    ``snps`` maps sample index → {0-based column: replacement base}.
    """
    rng = random.Random(11)
    ref = "".join(rng.choice("ACGT") for _ in range(400))
    seqs = [list(ref) for _ in range(4)]
    for i in range(160, 240):
        for s in seqs[1:]:
            s[i] = rng.choice("ACGT")
    for sample, changes in (snps or {}).items():
        for column, base in changes.items():
            seqs[sample][column] = base
    names = [f"s{i}" for i in range(4)]
    write_fasta(path, list(zip(names, ("".join(s) for s in seqs), strict=True)))
    return names


def test_candidates_cover_region_and_report_discrimination(tmp_path: Path) -> None:
    fasta = tmp_path / "aln.fa"
    _alignment(fasta)
    result = design_marker_primers(
        fasta,
        regions=[{"start": 161, "end": 240}],
        product_size_min=120,
        product_size_max=220,
    )
    assert result.status == "ok"
    metrics = result.metrics
    assert metrics["n_candidates"] == 1
    assert list(metrics["skipped_regions"]) == []
    candidate = metrics["candidates"][0]
    assert (candidate["region_start"], candidate["region_end"]) == (161, 240)
    assert candidate["amplicon_start"] <= 161
    assert candidate["amplicon_end"] >= 240
    assert 120 <= candidate["product_size"] <= 220
    # every sample anneals perfectly at both primers
    assert candidate["forward"]["max_mismatches"] == 0
    assert candidate["reverse"]["max_mismatches"] == 0
    assert all(v == 0 for v in candidate["forward"]["mismatches_by_sample"].values())
    # amplicon resolves all four samples
    assert candidate["distinct_haplotypes"] == 4
    assert candidate["samples_uniquely_distinguished"] == 4
    assert 0.1 < candidate["amplicon_pi"] < 0.6
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_hotspot_discovery_mode_finds_and_targets_peak_window(tmp_path: Path) -> None:
    fasta = tmp_path / "aln.fa"
    _alignment(fasta)
    discovered = design_marker_primers(
        fasta, window_size=100, step=50, product_size_min=120, product_size_max=220
    )
    metrics = discovered.metrics
    assert metrics["n_candidates"] == 1
    candidate = metrics["candidates"][0]
    # the merged hotspot is reported as the region; its peak π window (151-250)
    # is what the amplicon must cover.
    assert (candidate["region_start"], candidate["region_end"]) == (101, 300)
    assert candidate["amplicon_start"] <= 151
    assert candidate["amplicon_end"] >= 250
    assert 120 <= candidate["product_size"] <= 220


def test_polymorphic_priming_site_is_filtered(tmp_path: Path) -> None:
    """If every priming site within product range is polymorphic, no pair passes."""
    fasta = tmp_path / "aln.fa"
    # columns 40-360 polymorphic in every sample → no conserved priming site
    rng = random.Random(3)
    ref = "".join(rng.choice("ACGT") for _ in range(400))
    seqs = [list(ref) for _ in range(4)]
    for i in range(40, 360):
        for s in seqs[1:]:
            s[i] = rng.choice("ACGT")
    write_fasta(fasta, [(f"s{i}", "".join(s)) for i, s in enumerate(seqs)])
    result = design_marker_primers(
        fasta,
        regions=[{"start": 161, "end": 240}],
        product_size_min=120,
        product_size_max=220,
        max_primer_mismatches=1,
    )
    assert result.metrics["n_candidates"] == 0
    assert list(result.metrics["skipped_regions"]) == [
        {"start": 161, "end": 240, "reason": "no primer pair passed conservation filters"}
    ]


def test_invalid_and_oversized_regions_are_skipped(tmp_path: Path) -> None:
    fasta = tmp_path / "aln.fa"
    _alignment(fasta)
    result = design_marker_primers(
        fasta,
        regions=[{"start": 0, "end": 50}, {"start": 100, "end": 5000}, {"start": 300, "end": 100}],
        product_size_min=120,
        product_size_max=220,
    )
    reasons = [s["reason"] for s in result.metrics["skipped_regions"]]
    assert reasons == [
        "region outside alignment coordinates",
        "region outside alignment coordinates",
        "region outside alignment coordinates",
    ]
    result = design_marker_primers(
        fasta,
        regions=[{"start": 161, "end": 240}],
        product_size_min=120,
        product_size_max=150,  # region (80 bp) fits, but asks the impossible… no: fits
    )
    assert result.metrics["n_candidates"] == 1
    oversized = design_marker_primers(
        fasta,
        regions=[{"start": 1, "end": 400}],
        product_size_min=120,
        product_size_max=220,
    )
    assert list(oversized.metrics["skipped_regions"]) == [
        {"start": 1, "end": 400, "reason": "region larger than max product size"}
    ]


def test_reverse_primer_three_prime_orientation() -> None:
    """A 3' mismatch is the LEFTmost base of a reverse primer's span."""
    names = ["ref", "mut_left", "mut_right"]
    cols = list(range(10, 20))  # alignment columns of the reverse-primer span
    primer_seq = "GTACGTACGT"  # 5'→3' = reverse complement of the plus-strand span
    ref_seq = "T" * 10 + "ACGTACGTAC" + "T" * 6  # span (cols 10-19) = ACGTACGTAC
    seqs = [ref_seq, ref_seq, ref_seq]
    seqs[1] = ref_seq[:10] + "G" + ref_seq[11:]  # leftmost span base mutates → 3' end
    seqs[2] = ref_seq[:19] + "G" + ref_seq[20:]  # rightmost span base mutates → 5' end
    total, prime = _sample_mismatches(seqs, names, cols, primer_seq, revcomp=True)
    assert total == {"ref": 0, "mut_left": 1, "mut_right": 1}
    assert prime == {"ref": 0, "mut_left": 1, "mut_right": 0}
    assert not _conserved(total, prime, 5)  # both counted as total mismatches
    assert _conserved({"a": 1, "b": 1}, {"a": 0, "b": 0}, 1)
    assert not _conserved({"a": 2}, {"a": 0}, 1)


def test_sample_mismatches_count_gaps(tmp_path: Path) -> None:
    names = ["a", "b"]
    seqs = ["ACGTACGT", "ACG-ACGT"]
    total, _prime = _sample_mismatches(seqs, names, [0, 1, 2, 3], "ACGT")
    assert total == {"a": 0, "b": 1}  # the gap counts as a mismatch


def test_missing_primer3_raises_dependency_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "aln.fa"
    _alignment(fasta)

    def _boom():
        raise OrganelleDependencyError(
            code="barcode.design_marker_primers.missing_primer3",
            message="install primer3-py",
        )

    monkeypatch.setattr(primers, "_load_primer3", _boom)
    with pytest.raises(OrganelleDependencyError) as excinfo:
        design_marker_primers(fasta, regions=[{"start": 161, "end": 240}])
    assert excinfo.value.code == "barcode.design_marker_primers.missing_primer3"


def test_too_few_sequences_fails_closed(tmp_path: Path) -> None:
    fasta = tmp_path / "one.fa"
    fasta.write_text(">only\nACGTACGT\n")
    result = design_marker_primers(fasta)
    assert result.status == "failed"
    assert result.errors[0].code == "barcode.design_marker_primers.too_few_sequences"
