"""Coordinate and candidate evidence regressions for ORF/CMS screening."""

from pathlib import Path

import pytest

from organelleverse._bio import read_fasta
from organelleverse._sequtil import reverse_complement
from organelleverse.annotation.research import _predict_orfs, _write_cds_fasta
from organelleverse.phenotype.cms import cms, compute_cms_candidates


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("offset", [0, 1, 2])
def test_six_frame_coordinates_roundtrip_to_original_genome(tmp_path, strand, offset):
    coding = "ATG" + "AAA" * 30 + "TAA"
    oriented = coding if strand == "+" else reverse_complement(coding)
    genome = "C" * (15 + offset) + oriented + "C" * 6
    expected = (16 + offset, 15 + offset + len(coding))
    features = [
        feature
        for feature in _predict_orfs("contig", genome, 30)
        if feature.strand == strand and (feature.start, feature.end) == expected
    ]
    assert len(features) == 1
    fasta = tmp_path / "cds.fa"
    _write_cds_fasta(fasta, [("contig", genome)], features)
    assert read_fasta(fasta)[0][1] == coding


def test_real_nc_000932_negative_psba_cds_roundtrip(tmp_path):
    from Bio import SeqIO

    reference = Path(__file__).parents[2] / (
        "src/organelleverse/annotation/data/plastome/references/Arabidopsis_thaliana_chloroplast.gb"
    )
    record = SeqIO.read(reference, "genbank")
    assert record.id == "NC_000932.1"
    feature = next(
        item
        for item in record.features
        if item.type == "CDS" and item.qualifiers.get("gene") == ["psbA"]
    )
    assert feature.location.strand == -1
    coding = str(feature.extract(record.seq))
    genome = "C" * 15 + reverse_complement(coding) + "C" * 6
    predicted = [
        item
        for item in _predict_orfs("psbA", genome, 300)
        if item.strand == "-" and (item.start, item.end) == (16, 15 + len(coding))
    ]
    assert len(predicted) == 1
    fasta = tmp_path / "psba.cds.fa"
    _write_cds_fasta(fasta, [("psbA", genome)], predicted)
    extracted = read_fasta(fasta)[0][1]
    assert extracted == coding
    from Bio.Seq import Seq

    assert str(Seq(extracted).translate(to_stop=True)) == feature.qualifiers["translation"][0]


@pytest.mark.parametrize("strand", ["+", "-"])
def test_cms_public_and_core_agree_and_expose_screening_evidence(tmp_path, strand):
    coding = "ATG" + "CTG" * 50 + "TAA"
    oriented = coding if strand == "+" else reverse_complement(coding)
    path = tmp_path / "cms.fa"
    path.write_text(">candidate\n" + "C" * 15 + oriented + "C" * 6 + "\n")
    public = cms(path)
    core = compute_cms_candidates(path)
    assert public.status == "ok"
    assert public.metrics["candidates"] == core["count"] == 1
    row = public.metrics["candidate_table"][0]
    assert row["sequence_id"] == "candidate"
    assert (row["start"], row["end"], row["strand"]) == (16, 15 + len(coding), strand)
    assert row["protein_sequence"] == core["candidates"][0]["protein_sequence"] == "M" + "L" * 50
    assert row["evidence"] == "hydrophobicity_screen_only"
    assert row["tm_method"] == "kyte_doolittle_window"
    assert "hydrophobicity_screen_only" in public.flags
