"""Trans-splicing classification separates IR copies from real split genes.

Pooling exons by gene name alone read every IR-duplicated chloroplast gene
(rpl2, ycf2, ndhB, ...) as trans-spliced because the two copies sit tens of
kb apart. Classification must be per locus.
"""

from __future__ import annotations

from pathlib import Path

from organelleverse.io_genome import read_genbank_genome
from organelleverse.trans_splicing.trans_splicing import detect_trans_splicing

_HEADER = """LOCUS       fake          {length} bp    DNA     circular     01-JAN-2026
DEFINITION  synthetic.
ACCESSION   FAKE
VERSION     FAKE.1
ORIGIN
"""


def _wrap(sequence: str, width: int = 60) -> str:
    return "\n".join(
        f"{i + 1:>9} {sequence[i : i + width]}" for i in range(0, len(sequence), width)
    )


def _write_genbank(path: Path, sequence: str, features: list[str]) -> None:
    text = _HEADER.format(length=len(sequence)).replace("ORIGIN\n", "")
    text += "FEATURES             Location/Qualifiers\n"
    for feature in features:
        for line in feature.splitlines():
            text += f"     {line}\n"
    text += "ORIGIN\n" + _wrap(sequence) + "\n//\n"
    path.write_text(text)


def test_ir_duplicated_gene_is_cis(tmp_path: Path) -> None:
    sequence = "A" * 30000 + "C" * 600 + "G" * 20000 + "T" * 600
    features = [
        "CDS             30001..30600",
        '                /gene="rpl2"',
        "CDS             50601..51200",
        '                /gene="rpl2"',
    ]
    path = tmp_path / "ir_dup.gb"
    _write_genbank(path, sequence, features)
    genome = read_genbank_genome(path, organelle="plastid", species="Synthetic")
    result = detect_trans_splicing(genome, min_exon_gap=5000)
    assert dict(result.metrics)["trans_splicing"] == 0
    assert dict(result.metrics)["cis_splicing"] == 1


def test_single_feature_with_far_apart_exons_is_trans(tmp_path: Path) -> None:
    sequence = "A" * 30000 + "C" * 600 + "G" * 20000 + "T" * 600
    features = [
        "CDS             join(30001..30600,50601..51200)",
        '                /gene="rps12"',
    ]
    path = tmp_path / "split_gene.gb"
    _write_genbank(path, sequence, features)
    genome = read_genbank_genome(path, organelle="plastid", species="Synthetic")
    result = detect_trans_splicing(genome, min_exon_gap=5000)
    genes = [finding.metric for finding in (result.findings or ())]
    assert genes == ["rps12"]


def test_close_exon_features_of_one_gene_still_merge(tmp_path: Path) -> None:
    """Exon-style annotations (one feature per exon) must not be split apart."""
    sequence = "A" * 30000 + "C" * 300 + "G" * 300 + "T" * 300
    features = [
        "CDS             30001..30300",
        '                /gene="nad1"',
        "CDS             30601..30900",
        '                /gene="nad1"',
    ]
    path = tmp_path / "exon_pairs.gb"
    _write_genbank(path, sequence, features)
    genome = read_genbank_genome(path, organelle="mitochondrion", species="Synthetic")
    result = detect_trans_splicing(genome, min_exon_gap=5000)
    assert dict(result.metrics)["trans_splicing"] == 0
