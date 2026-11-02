"""A genome map's result scope is the organelle its GenBank records declare."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from organelleverse.visualization.ogdraw import plot_ogdraw_map
from organelleverse.visualization.plots import plot_genome_map


def _genbank(path: Path, organelle: str | None, name: str = "rec1") -> Path:
    rng = random.Random(3)
    record = SeqRecord(
        Seq("".join(rng.choice("ACGT") for _ in range(3_000))),
        id=name,
        name=name,
        description="test record",
        annotations={"molecule_type": "DNA", "topology": "circular"},
    )
    qualifiers = {"organelle": [organelle]} if organelle else {}
    record.features.append(
        SeqFeature(SimpleLocation(0, 3_000, strand=1), type="source", qualifiers=qualifiers)
    )
    record.features.append(
        SeqFeature(SimpleLocation(100, 400, strand=1), type="gene", qualifiers={"gene": ["psbA"]})
    )
    SeqIO.write([record], path, "genbank")
    return path


@pytest.mark.parametrize("plot", [plot_genome_map, plot_ogdraw_map])
@pytest.mark.parametrize(
    ("declared", "scope"),
    [("plastid:chloroplast", "plastid"), ("mitochondrion", "mitochondrion"), (None, "none")],
)
def test_scope_follows_the_declared_organelle(tmp_path, plot, declared, scope) -> None:
    gb = _genbank(tmp_path / "one.gb", declared)
    assert plot([gb]).scope == scope


def test_records_that_declare_different_organelles_are_mixed(tmp_path) -> None:
    cp = _genbank(tmp_path / "cp.gb", "plastid:chloroplast", name="cp")
    mt = _genbank(tmp_path / "mt.gb", "mitochondrion", name="mt")
    assert plot_genome_map([cp, mt]).scope == "mixed"
