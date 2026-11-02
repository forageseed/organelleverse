"""MTPT grading against plastid / mitochondrial reference panels."""

from __future__ import annotations

import random
import shutil

import pytest

from organelleverse.transfer.mtpt_grading import (
    _best_hits,
    classify_fragment,
    grade_mtpt,
    organelle_reference_fasta,
    refine_ambiguous,
)


def test_four_tiers_follow_the_published_thresholds() -> None:
    strong_p = {"coverage": 80.0, "identity": 97.0}
    strong_m = {"coverage": 40.0, "identity": 90.0}
    weak = {"coverage": 20.0, "identity": 99.0}
    assert classify_fragment(strong_p, None) == "clean_plastid"
    assert classify_fragment(strong_p, weak) == "clean_plastid"
    assert classify_fragment(strong_p, strong_m) == "ambiguous"
    assert classify_fragment(weak, strong_m) == "conserved_mtdna"
    assert classify_fragment(None, None) == "undetermined"
    # identity must exceed (not equal) the threshold
    assert classify_fragment({"coverage": 80.0, "identity": 90.0}, None) == "undetermined"


def test_ambiguous_subtype_by_identity_margin() -> None:
    assert refine_ambiguous({"identity": 99.0}, {"identity": 96.0}) == "closer_to_plastid"
    assert refine_ambiguous({"identity": 93.0}, {"identity": 98.5}) == "closer_to_mito"
    assert refine_ambiguous({"identity": 97.0}, {"identity": 96.5}) == "equidistant"
    assert refine_ambiguous({"identity": 97.0}, {"identity": 96.5}, identity_margin=0.2) == "closer_to_plastid"


def test_best_hits_union_coverage_and_exclusion() -> None:
    rows = [
        # qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen
        ["c0", "NC_1", "98.0", "400", "1", "400", "1", "400", "0", "700", "1000"],
        ["c0", "NC_1", "96.0", "400", "301", "700", "900", "1300", "0", "650", "1000"],
        ["c0", "NC_2", "99.9", "1000", "1", "1000", "1", "1000", "0", "1800", "1000"],
    ]
    organisms = {"NC_1": "Oryza rufipogon", "NC_2": "Oryza sativa Japonica Group"}
    best = _best_hits(rows, organisms, exclude="Oryza sativa")
    assert best["c0"]["accession"] == "NC_1"
    assert best["c0"]["coverage"] == 70.0  # union 1-700 of 1000
    assert best["c0"]["identity"] == 97.0
    assert _best_hits(rows, organisms, exclude=None)["c0"]["accession"] == "NC_2"


GBFF = """LOCUS       NC_000001               20 bp    DNA     circular PLN 01-JAN-2026
ACCESSION   NC_000001
VERSION     NC_000001.1
SOURCE      plastid Testus plantus
  ORGANISM  Testus plantus
            Eukaryota; Viridiplantae; Streptophyta; Embryophyta.
FEATURES             Location/Qualifiers
ORIGIN
        1 acgtacgtac gtacgtacgt
//
LOCUS       NC_000002               10 bp    DNA     circular VRT 01-JAN-2026
VERSION     NC_000002.1
  ORGANISM  Homo sapiens
            Eukaryota; Metazoa; Chordata.
ORIGIN
        1 ttttttttgg
//
"""


def test_reference_fasta_keeps_green_plants(tmp_path) -> None:
    gbff = tmp_path / "ref.gbff"
    gbff.write_text(GBFF)
    out = tmp_path / "plastid.fa"
    stats = organelle_reference_fasta(gbff, out)
    assert stats == {"records": 1, "bases": 20}
    text = out.read_text()
    assert text.startswith(">NC_000001.1 organism=Testus plantus\nACGTACGTACGTACGTACGT")
    assert organelle_reference_fasta(gbff, out, lineage=None)["records"] == 2


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT".replace(b, "")) if rng.random() < rate else b for b in seq)


@pytest.mark.skipif(
    not (shutil.which("blastn") and shutil.which("makeblastdb")), reason="needs NCBI BLAST+"
)
def test_end_to_end_grading(tmp_path) -> None:
    rng = random.Random(3)

    def dna(n: int) -> str:
        return "".join(rng.choice("ACGT") for _ in range(n))

    a, b, r = dna(1200), dna(900), dna(800)  # transfer, shared old transfer, native mt region
    cp = dna(5000) + a + dna(4000) + b + dna(4000) + r + dna(5000)
    mito = dna(6000) + a + dna(7000) + b + dna(6000) + r + dna(8000)
    (tmp_path / "cp.fa").write_text(">cp\n" + cp + "\n")
    (tmp_path / "mt.fa").write_text(">mt\n" + mito + "\n")
    plastid_ref = tmp_path / "plastid_ref.fa"
    plastid_ref.write_text(
        ">NC_P1 organism=Other plant\n"
        + dna(3000) + _mutate(a, 0.03, rng) + dna(3000) + _mutate(b, 0.03, rng)
        + dna(3000) + _mutate(r, 0.15, rng) + dna(3000) + "\n"
    )
    mito_ref = tmp_path / "mito_ref.fa"
    mito_ref.write_text(
        ">NC_M1 organism=Other plant\n"
        + dna(4000) + _mutate(b, 0.04, rng) + dna(4000) + _mutate(r, 0.03, rng) + dna(4000) + "\n"
        + ">NC_SELF organism=Query species\n" + mito + "\n"
    )
    result = grade_mtpt(
        tmp_path / "mt.fa", tmp_path / "cp.fa",
        plastid_reference=plastid_ref, mitochondrion_reference=mito_ref,
        exclude_organism="Query species",
    )
    assert result.status == "ok", result.summary_text
    tiers = {f["start"] // 1000: f["tier"] for f in result.metrics["fragments"]}
    assert sorted(tiers.values()) == ["ambiguous", "clean_plastid", "conserved_mtdna"]
    assert result.metrics["bp_by_tier"]["conserved_mtdna"] >= 780
    assert result.metrics["retained_fraction"] < result.metrics["raw_fraction"]
    assert "possible_self_in_reference" not in result.flags
    ambiguous = [f for f in result.metrics["fragments"] if f["tier"] == "ambiguous"]
    assert ambiguous[0]["ambiguous_subtype"] in ("closer_to_plastid", "closer_to_mito", "equidistant")
    assert ambiguous[0]["mito_species_with_copy"] == 1  # "Other plant"; the query species is excluded
    assert sum(result.metrics["bp_by_ambiguous_subtype"].values()) == result.metrics["bp_by_tier"]["ambiguous"]
    clean = [f for f in result.metrics["fragments"] if f["tier"] == "clean_plastid"]
    assert clean[0]["ambiguous_subtype"] is None and clean[0]["mito_species_with_copy"] == 0

    unexcluded = grade_mtpt(
        tmp_path / "mt.fa", tmp_path / "cp.fa",
        plastid_reference=plastid_ref, mitochondrion_reference=mito_ref,
    )
    assert "possible_self_in_reference" in unexcluded.flags
