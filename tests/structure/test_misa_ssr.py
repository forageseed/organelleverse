"""MISA copy thresholds and the NC_000932 plastome benchmark."""

from collections import Counter
from pathlib import Path

from Bio import SeqIO

from organelleverse.structure.structure import _find_ssrs, repeats
from organelleverse.structure.structure_core import compute_repeats


def test_misa_thresholds_for_each_motif_size() -> None:
    motifs = ("A", "AT", "ATG", "ATGC", "ATGCA", "ATGCAG")
    thresholds = (10, 5, 4, 3, 3, 3)
    for unit_len, (motif, copies) in enumerate(zip(motifs, thresholds, strict=True), start=1):
        assert _find_ssrs(motif * (copies - 1), unit_len, unit_len, 3) == []
        matches = _find_ssrs(motif * copies, unit_len, unit_len, 3)
        assert len(matches) == 1
        assert matches[0]["copies"] == copies


def test_records_do_not_join_across_fasta_boundaries(tmp_path: Path) -> None:
    fasta = tmp_path / "two.fasta"
    fasta.write_text(">first\nAAAAAAAAA\n>second\nAAAAAAAAA\n", encoding="ascii")
    assert compute_repeats(fasta)["ssr_count"] == 0
    assert repeats(fasta).metrics["ssr_count"] == 0

    fasta.write_text(">first\nAAAAAAAAAA\n>second\nATATATATAT\n", encoding="ascii")
    matches = compute_repeats(fasta)["ssrs"]
    assert {(m["sequence_id"], m["unit"], m["start"], m["end"]) for m in matches} == {
        ("first", "A", 1, 10),
        ("first", "AA", 1, 10),
        ("second", "AT", 1, 10),
    }


def test_nc_000932_matches_misa_reference_count(tmp_path: Path) -> None:
    reference = (
        Path(__file__).resolve().parents[2]
        / "src/organelleverse/annotation/data/plastome/references/Arabidopsis_thaliana_chloroplast.gb"
    )
    record = SeqIO.read(reference, "genbank")
    assert record.id == "NC_000932.1"
    fasta = tmp_path / "NC_000932.fasta"
    SeqIO.write(record, fasta, "fasta")

    result = compute_repeats(fasta)
    assert result["ssr_count"] == 232
    assert Counter(len(ssr["unit"]) for ssr in result["ssrs"]) == {
        1: 69,
        2: 87,
        3: 30,
        4: 39,
        5: 6,
        6: 1,
    }
    assert repeats(fasta).metrics["ssr_count"] == 232
