"""Multi-sample depth windows and SNV/InDel fractions (heteroplasmy_panel)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from organelleverse.heteroplasmy_panel import summarize_depth_windows, summarize_variant_fractions

from tests.bam_fixtures import bam_header

rubam = pytest.importorskip("rubam")

READ = 40


def _reference(tmp_path: Path) -> dict[str, str]:
    rng = random.Random(3)
    seqs = {
        "MT": "".join(rng.choice("ACGT") for _ in range(400)),
        "PT": "".join(rng.choice("ACGT") for _ in range(240)),
    }
    (tmp_path / "mt.fa").write_text(f">MT\n{seqs['MT']}\n", encoding="utf-8")
    (tmp_path / "pt.fa").write_text(f">PT\n{seqs['PT']}\n", encoding="utf-8")
    return seqs


def _bam(
    path: Path, seqs: dict[str, str], *, snv: int = 0, deletion: int = 0, low: str = ""
) -> Path:
    """Ten reads start at every 4th position of each contig (depth 100 inside).

    ``snv`` of every 10 reads covering MT:101 carry A/C/G/T != ref there; ``deletion``
    of every 10 reads covering MT:201 lack MT 202-203 (a 2-bp deletion anchored at
    201). ``low`` contig gets only one read per start (depth ~10).
    """
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": len(seq)} for name, seq in seqs.items()],
    }
    unsorted = path.with_suffix(".unsorted.bam")
    n = 0
    with rubam.AlignmentFile(
        str(unsorted), "wb", header=bam_header(header, Path(unsorted).parent)
    ) as out:
        for rid, (name, seq) in enumerate(seqs.items()):
            for start in range(0, len(seq) - READ - 2, 4):
                for copy in range(1 if name == low else 10):
                    query = list(seq[start : start + READ])
                    cigar = [(0, READ)]
                    if name == "MT" and start <= 100 < start + READ and copy < snv:
                        ref = seq[100]
                        query[100 - start] = next(b for b in "ACGT" if b != ref)
                    if name == "MT" and start <= 199 and start + READ > 205 and copy < deletion:
                        left = 201 - start
                        query = list(
                            seq[start : start + left] + seq[start + left + 2 : start + READ + 2]
                        )
                        cigar = [(0, left), (2, 2), (0, READ - left)]
                    read = rubam.AlignedSegment()
                    read.query_name = f"r{n}"
                    read.query_sequence = "".join(query)
                    read.flag = 0
                    read.reference_id = rid
                    read.reference_start = start
                    read.mapping_quality = 60
                    read.cigartuples = cigar
                    read.query_qualities = rubam.qualitystring_to_array("I" * len(query))
                    out.write(read)
                    n += 1
    rubam.sort(str(unsorted), str(path))
    rubam.index(str(path))
    return path


def test_depth_windows_are_relative_to_each_sample_median(tmp_path: Path) -> None:
    seqs = _reference(tmp_path)
    bams = {
        "wt": _bam(tmp_path / "wt.bam", seqs),
        "low_pt": _bam(tmp_path / "b.bam", seqs, low="PT"),
    }
    result = summarize_depth_windows(
        bams, window_size=50, organelles={"MT": "mitochondrion", "PT": "plastid"}
    )
    assert result.status == "ok"
    profiles = {(p["sample"], p["seqid"]): p for p in result.metrics["profiles"]}
    assert len(profiles[("wt", "MT")]["mean_depth"]) == 8
    # Uniform 10x lower plastid depth is still 0 relative to that sample's own plastid median.
    assert profiles[("low_pt", "PT")]["median_depth"] == pytest.approx(10, abs=0.5)
    assert profiles[("low_pt", "PT")]["log2_ratio"][2] == pytest.approx(0, abs=0.05)
    assert profiles[("wt", "MT")]["log2_ratio"][0] < -0.5  # first window ramps up from 0
    assert {c["organelle"] for c in result.metrics["contigs"]} == {"mitochondrion", "plastid"}
    only_mt = summarize_depth_windows(bams, window_size=50, sequences=["MT"])
    assert {p["seqid"] for p in only_mt.metrics["profiles"]} == {"MT"}
    assert summarize_depth_windows(bams, sequences=["chr9"]).status == "failed"


def test_depth_windows_reject_mismatched_references(tmp_path: Path) -> None:
    seqs = _reference(tmp_path)
    other = {"MT": seqs["MT"][:300], "PT": seqs["PT"]}
    bams = {"a": _bam(tmp_path / "a.bam", seqs), "b": _bam(tmp_path / "b.bam", other)}
    result = summarize_depth_windows(bams)
    assert result.status == "failed"
    assert result.errors[0].code == "heteroplasmy.depth_failed"


def test_snv_and_indel_fractions_in_every_sample(tmp_path: Path) -> None:
    seqs = _reference(tmp_path)
    bams = {
        "mutant": _bam(tmp_path / "m.bam", seqs, snv=2, deletion=1),
        "wt": _bam(tmp_path / "w.bam", seqs),
    }
    repeats = [{"sequence_id": "MT", "positions": [195, 330], "length": 20, "type": "direct"}]
    result = summarize_variant_fractions(
        bams,
        {"mitochondrion": tmp_path / "mt.fa", "plastid": tmp_path / "pt.fa"},
        homology_mask=False,
        min_alt_fraction=0.05,
        repeats=repeats,
    )
    assert result.status == "ok", result.summary_text
    sites = {s["site_id"]: s for s in result.metrics["sites"]}
    snv = next(s for s in sites.values() if s["kind"] == "SNV")
    indel = next(s for s in sites.values() if s["kind"] == "InDel")
    assert (snv["seqid"], snv["position"]) == ("MT", 101)
    assert (indel["position"], indel["ref"], indel["alt"]) == (
        201,
        seqs["MT"][200:203],
        seqs["MT"][200],
    )
    assert snv["scope"] == "mitochondrion" and not snv["in_repeat"] and indel["in_repeat"]
    rows = {(r["sample"], r["site_id"]): r for r in result.metrics["fractions"]}
    assert rows[("mutant", snv["site_id"])]["alt_fraction"] == pytest.approx(0.2)
    # Reads starting at 168..196 (8 of the 10 spanning starts) carry the deletion.
    assert rows[("mutant", indel["site_id"])]["alt_fraction"] == pytest.approx(0.08)
    assert rows[("mutant", snv["site_id"])]["category"] == "other"
    assert rows[("mutant", indel["site_id"])]["category"] == "repeat_related"
    # Measured, not only discovered, in the wild type: zero alt reads -> low confidence, undetected.
    assert rows[("wt", snv["site_id"])]["alt_fraction"] == 0
    assert rows[("wt", snv["site_id"])]["detected"] is False
    assert rows[("wt", snv["site_id"])]["category"] == "low_confidence"
    assert len(result.metrics["fractions"]) == 2 * len(sites)


def test_variants_carried_by_the_control_are_low_confidence(tmp_path: Path) -> None:
    seqs = _reference(tmp_path)
    bams = {
        "wt": _bam(tmp_path / "w.bam", seqs, snv=2),
        "mutant": _bam(tmp_path / "m.bam", seqs, snv=3, deletion=1),
    }
    refs = {"mitochondrion": tmp_path / "mt.fa", "plastid": tmp_path / "pt.fa"}
    result = summarize_variant_fractions(
        bams, refs, homology_mask=False, min_alt_fraction=0.05, control_samples=["wt"]
    )
    sites = {s["kind"]: s for s in result.metrics["sites"]}
    assert sites["SNV"]["in_control"] and not sites["InDel"]["in_control"]
    rows = {(r["sample"], r["site_id"]): r["category"] for r in result.metrics["fractions"]}
    assert rows[("mutant", sites["SNV"]["site_id"])] == "low_confidence"
    assert rows[("mutant", sites["InDel"]["site_id"])] == "other"
    bad = summarize_variant_fractions(bams, refs, homology_mask=False, control_samples=["nope"])
    assert bad.errors[0].code == "heteroplasmy.unknown_control"


def test_homology_mask_needs_both_organelles(tmp_path: Path) -> None:
    seqs = _reference(tmp_path)
    bams = {"wt": _bam(tmp_path / "w.bam", seqs)}
    result = summarize_variant_fractions(bams, {"mitochondrion": tmp_path / "mt.fa"})
    assert result.errors[0].code == "heteroplasmy.mask_needs_reference"
