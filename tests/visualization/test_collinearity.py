"""OrganelleVerse-drawn collinearity: gene links, blocks, orientation, alignments, rendering."""

from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.errors import OrganelleParameterError
from organelleverse.core.result import OrganelleResult
from organelleverse.visualization import compose_figure, plot_collinearity
from organelleverse.visualization.collinearity import _gene_key, plot_collinearity_ov

GENE = 900  # bp per gene, 100 bp spacers


def _genbank(path: Path, genes: list[tuple[str, int]], *, extra: list = (), organelle="mitochondrion"):
    """``genes``: (name, strand) laid out left to right; ``extra``: raw SeqFeatures."""
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord

    length = len(genes) * (GENE + 100) + 1000
    record = SeqRecord(Seq("ACGT" * (length // 4)), id=path.stem, name=path.stem,
                       description=f"{path.stem} {organelle} genome",
                       annotations={"molecule_type": "DNA", "topology": "circular",
                                    "organism": f"Species {path.stem}"})
    record.features.append(SeqFeature(FeatureLocation(0, length), type="source",
                                      qualifiers={"organelle": [organelle],
                                                  "organism": [f"Species {path.stem}"]}))
    for i, (name, strand) in enumerate(genes):
        start = 500 + i * (GENE + 100)
        kind = "tRNA" if name.lower().startswith("trn") else "CDS"
        size = 72 if kind == "tRNA" else GENE
        record.features.append(SeqFeature(FeatureLocation(start, start + size, strand=strand),
                                          type=kind, qualifiers={"gene": [name]}))
    record.features.extend(extra)
    from Bio import SeqIO

    SeqIO.write(record, str(path), "genbank")
    return path


ORDER = ["psbA", "matK", "rbcL", "atpB", "accD", "psaA", "psbB", "petB", "rpoC1", "ndhF"]


def test_gene_keys_are_normalised():
    assert _gene_key("trnD(gtc)") == "trnd-guc"
    assert _gene_key("trnH-GUG") == "trnh-gug"
    assert _gene_key("trnD-GUC; trnD(gtc)") == "trnd-guc"
    assert _gene_key("trnfM-CAU II") == "trnfm-cau"
    assert _gene_key("pbf1") == "psbn" and _gene_key("cytb") == "cob" and _gene_key("rrnS") == "rrn18"
    assert _gene_key("orf102") is None and _gene_key("BRDIC_r001") is None


def test_an_inverted_segment_is_one_inverted_block(tmp_path):
    a = _genbank(tmp_path / "a.gb", [(n, 1) for n in ORDER])
    # rbcL..psaA reversed in order and strand: one inversion
    inverted = ORDER[:2] + [(n) for n in reversed(ORDER[2:6])] + ORDER[6:]
    b = _genbank(tmp_path / "b.gb", [(n, -1 if n in ORDER[2:6] else 1) for n in inverted])
    plot = plot_collinearity([a, b])
    pair = plot.metrics["pairs"][0]
    assert pair["n_links"] == len(ORDER)
    assert pair["n_inverted_links"] == 4
    assert [blk["inverted"] for blk in pair["blocks"]] == [False, True, False]
    assert pair["blocks"][1]["n_genes"] == 4
    assert "inversion" in plot.flags
    assert plot.metrics["genomes"][0]["name"] == "Species a"


def test_trnas_match_by_amino_acid_and_rnas_are_named_from_products(tmp_path):
    from Bio.SeqFeature import FeatureLocation, SeqFeature

    unnamed = SeqFeature(FeatureLocation(12000, 12072, strand=1), type="tRNA",
                         qualifiers={"locus_tag": ["X_cpR001"], "product": ["tRNA-Gly"]})
    a = _genbank(tmp_path / "a.gb", [("psbA", 1), ("trnH", 1), ("rbcL", 1), ("trnL", 1)], extra=[unnamed])
    b = _genbank(tmp_path / "b.gb", [("psbA", 1), ("trnH-GUG", 1), ("rbcL", 1), ("trnL-UAA", 1),
                                     ("trnG-GCC", 1), ("ycf99", 1)])
    pair = plot_collinearity([a, b]).metrics["pairs"][0]
    linked = {link["gene"] for link in pair["links"]}
    assert {"trnH", "trnL", "trnG"} <= linked  # trnG named from the product "tRNA-Gly"
    assert list(pair["only_lower"]) == ["ycf99"] and not pair["only_upper"]


def test_anchor_orientation_reverse_complements_the_other_strand(tmp_path):
    a = _genbank(tmp_path / "a.gb", [(n, 1) for n in ORDER])
    # the same genome written from the other strand
    b = _genbank(tmp_path / "b.gb", [(n, -1) for n in reversed(ORDER)])
    as_given = plot_collinearity([a, b]).metrics["pairs"][0]
    assert as_given["n_inverted_links"] == len(ORDER)
    plot = plot_collinearity([a, b], rotate_to="psbA", orient="anchor")
    pair = plot.metrics["pairs"][0]
    assert pair["n_inverted_links"] == 0 and pair["n_blocks"] == 1
    assert plot.metrics["genomes"][1]["reverse_complemented"] is True
    assert "reverse_complemented" in plot.flags
    with pytest.raises(OrganelleParameterError):
        plot_collinearity([a, b], orient="anchor")


def test_alignment_tables_replace_gene_links(tmp_path):
    a = _genbank(tmp_path / "a.gb", [(n, 1) for n in ORDER])
    b = _genbank(tmp_path / "b.gb", [(n, 1) for n in ORDER])
    table = tmp_path / "a_b.tsv"
    table.write_text(
        "a\tb\t98.5\t3000\t0\t0\t1\t3000\t1\t3000\t0\t5000\n"     # forward
        "a\tb\t91.0\t2000\t0\t0\t4001\t6000\t8000\t6001\t0\t3000\n"  # inverted
        "a\tb\t99.0\t150\t0\t0\t7001\t7150\t7001\t7150\t0\t200\n"   # too short
        "x\tb\t99.0\t900\t0\t0\t1\t900\t1\t900\t0\t1000\n"          # unknown query
    )
    plot = plot_collinearity([a, b], blast=table)
    pair = plot.metrics["pairs"][0]
    assert pair["source"] == "alignment" and pair["n_links"] == 2
    assert [link["inverted"] for link in pair["links"]] == [False, True]
    assert pair["skipped_rows"] == 1 and "alignment_rows_skipped" in plot.flags


def test_renders_from_serialized_metrics_and_composes(tmp_path):
    a = _genbank(tmp_path / "a.gb", [(n, 1) for n in ORDER])
    b = _genbank(tmp_path / "b.gb", [(n, 1) for n in ORDER])
    plot = plot_collinearity_ov([a, b], gene_labels=["rbcL", "ndhF"])
    assert plot.metrics["width_mm"] == 183.0
    restored = OrganelleResult.model_validate_json(plot.model_dump_json())
    for suffix in ("png", "svg", "pdf"):
        out = tmp_path / f"collinearity.{suffix}"
        assert ov.write(restored, out).artifacts and out.stat().st_size > 0
    figure = compose_figure([plot, plot], width="double", columns=1)
    assert ov.write(figure, tmp_path / "composed.png").artifacts


def test_bad_arguments_are_parameter_errors(tmp_path):
    a = _genbank(tmp_path / "a.gb", [(n, 1) for n in ORDER])
    with pytest.raises(OrganelleParameterError):
        plot_collinearity([a])
    with pytest.raises(OrganelleParameterError):
        plot_collinearity([a, a], scale="log")
    with pytest.raises(OrganelleParameterError):
        plot_collinearity([a, a], alignments=[])
    with pytest.raises(ValueError):
        plot_collinearity([a, a], method="mauve")
