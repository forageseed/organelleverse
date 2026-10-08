"""Plastid gene classes and legend for OGDraw-style maps."""

from __future__ import annotations

import pytest

from organelleverse.visualization.ogdraw import GENE_COLORS, classify_gene, legend_rows


@pytest.mark.parametrize(
    ("name", "category"),
    [
        ("psaA", "photosystem_I"),
        ("psbA", "photosystem_II"),
        ("petB", "cytochrome_b6f"),
        ("ndhB", "complex_I"),
        ("rbcL", "rubisco"),
        ("clpP", "clp_protease"),
        ("ycf3", "ycf"),
        ("atpF", "atp_synthase"),
        ("rpoC1", "rna_pol"),
        ("rps12", "ribo_SSU"),
        ("matK", "maturase"),
        ("trnK-UUU", "tRNA"),
        ("rrn16", "rRNA"),
        ("accD", "other"),
    ],
)
def test_plastid_genes_get_plastid_classes(name: str, category: str) -> None:
    assert classify_gene(name, "plastid") == category


def test_default_keeps_ogdrawr_mitochondrial_rules() -> None:
    # OGDrawR has no plastid classes: ndh/psb stay "other" unless asked.
    assert classify_gene("ndhB") == "other"
    assert classify_gene("psbA") == "other"
    assert classify_gene("nad5") == "complex_I"
    assert classify_gene("cox1", "mito") == "complex_IV"


def test_legend_follows_organelle_and_every_row_has_a_colour() -> None:
    plastid = dict(legend_rows("plastid"))
    mito = dict(legend_rows("mito"))
    assert "photosystem_II" in plastid and "complex_IV" not in plastid
    assert "complex_IV" in mito and "photosystem_II" not in mito
    for key in {*plastid, *mito}:
        assert key in GENE_COLORS
