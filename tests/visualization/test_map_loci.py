"""Loci, IR band, label scaling and names shared by the OGDraw and structure maps."""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

import organelleverse as ov
from organelleverse.capabilities.adapters.visualization import _STRUCTURE_GENBANK
from organelleverse.core.errors import OrganelleInputError
from organelleverse.visualization import plot_structure_map
from organelleverse.visualization.features import annotation_rows, record_display_name
from organelleverse.visualization.ogdraw import (
    _genbank_display_name,
    draw_mito_map,
    parse_genbank,
    plot_ogdraw_map,
)
from organelleverse.visualization.plots import plot_genome_map


def _mito_record(length: int = 120_000) -> SeqRecord:
    """A mitochondrion written the way older OrganelleVerse releases wrote nad2/nad5.

    Each split gene has one plain ``gene`` span over its CDS join and no
    ``/trans_splicing``: nad2 mixes strands, nad5 stays on one strand but its
    exons sit 60 kb apart, cox2 is an ordinary 1 kb cis intron.
    """
    rng = random.Random(7)
    record = SeqRecord(
        Seq("".join(rng.choice("ACGT") for _ in range(length))),
        id="mt1",
        name="mt1",
        description="Testus mitochondrialis mitochondrion, complete genome.",
        annotations={"molecule_type": "DNA", "organism": "."},
    )

    def add(kind, parts, name):
        location = parts[0] if len(parts) == 1 else CompoundLocation(parts)
        record.features.append(SeqFeature(location, type=kind, qualifiers={"gene": [name]}))

    record.features.append(
        SeqFeature(SimpleLocation(0, length, strand=1), type="source",
                   qualifiers={"organelle": ["mitochondrion"]})
    )
    add("gene", [SimpleLocation(10_000, 50_400, strand=1)], "nad2")
    add("CDS", [SimpleLocation(50_000, 50_400, strand=1), SimpleLocation(10_000, 10_300, strand=-1)], "nad2")
    add("gene", [SimpleLocation(60_000, 120_000, strand=1)], "nad5")
    add("CDS", [SimpleLocation(60_000, 60_300, strand=1), SimpleLocation(119_000, 119_400, strand=1)], "nad5")
    add("gene", [SimpleLocation(2_000, 3_600, strand=1)], "cox2")
    add("CDS", [SimpleLocation(2_000, 2_300, strand=1), SimpleLocation(3_300, 3_600, strand=1)], "cox2")
    add("gene", [SimpleLocation(80_000, 80_072, strand=1)], "trnM")
    add("tRNA", [SimpleLocation(80_000, 80_072, strand=1)], "trnM")
    add("gene", [SimpleLocation(90_000, 90_072, strand=-1)], "trnM")
    add("tRNA", [SimpleLocation(90_000, 90_072, strand=-1)], "trnM")
    return record


@pytest.fixture
def mito_gb(tmp_path):
    path = tmp_path / "annotation.gb"
    SeqIO.write([_mito_record()], path, "genbank")
    return path


def test_gene_span_over_a_split_locus_is_not_drawn_as_one_block():
    rows = annotation_rows(_mito_record())
    assert [r["feature_type"] for r in rows] == ["CDS", "CDS", "CDS", "tRNA", "tRNA"]
    by_gene = {r["gene"]: r for r in rows if r["gene"] != "trnM"}
    assert (by_gene["nad2"]["splicing"], by_gene["nad2"]["trans_basis"]) == ("trans", "mixed_strand")
    assert (by_gene["nad5"]["splicing"], by_gene["nad5"]["trans_basis"]) == ("trans", "distance")
    assert (by_gene["cox2"]["splicing"], by_gene["cox2"]["gaps"]) == ("cis", [1000])


def test_ogdraw_parsing_keeps_every_copy_and_each_exon_strand(mito_gb):
    genes = parse_genbank(mito_gb).genes
    assert [g.name for g in genes].count("trnM") == 2  # OGDrawR kept only the first
    nad2 = next(g for g in genes if g.name == "nad2")
    assert nad2.trans_spliced and nad2.exon_strands == [1, -1]
    # No block spans the 40 kb between nad2's exons.
    assert max(b - a for g in genes for a, b in g.exons) < 1_000


def test_ogdraw_map_numbers_inferred_trans_exons(mito_gb, tmp_path):
    out = tmp_path / "map.svg"
    ov.write(plot_ogdraw_map(mito_gb, dpi=50), out)
    svg = out.read_text(encoding="utf-8")
    assert "nad5·1" in svg and "nad5·2" in svg and "nad2·2" in svg


def test_display_name_prefers_the_species_over_the_file_stem(mito_gb):
    assert _genbank_display_name(mito_gb) == "Testus mitochondrialis"
    record = _mito_record()
    record.features[0].qualifiers["organism"] = ["Arabidopsis_thaliana"]
    assert record_display_name(record, "x") == "Arabidopsis thaliana"


def test_fonts_are_printed_sizes_whatever_the_canvas(mito_gb):
    # Journal style: type is set in points at printed size; a bigger canvas
    # gives labels room, it does not make them bigger.
    parsed = parse_genbank(mito_gb)
    sizes = {}
    for inches in (12, 7.2):
        fig, ax = plt.subplots(figsize=(inches, inches))
        try:
            draw_mito_map(parsed, genome_name="G", ax=ax)
            sizes[inches] = {t.get_text(): t.get_fontsize() for t in ax.texts}
        finally:
            plt.close(fig)
    assert sizes[7.2]["G"] == sizes[12]["G"] == 8.5
    assert sizes[7.2][" cox2 "] == sizes[12][" cox2 "] == pytest.approx(6.5)


def _svg_font_sizes(svg_text: str) -> list[float]:
    import re

    return [float(v) for v in re.findall(r"font(?:-size)?:\s*(?:\w+\s+)*?([\d.]+)px", svg_text)]


def test_default_map_is_double_column_and_no_text_below_5_pt(mito_gb, tmp_path):
    out = tmp_path / "map.svg"
    ov.write(plot_ogdraw_map(mito_gb), out)
    import xml.etree.ElementTree as ET

    width = ET.parse(out).getroot().attrib["width"]
    assert width.endswith("pt") and float(width[:-2]) == pytest.approx(183 / 25.4 * 72, rel=0.08)
    sizes = _svg_font_sizes(out.read_text(encoding="utf-8"))
    assert sizes and min(sizes) >= 5.0 * 1.0  # SVG px at 72 dpi == pt


def test_structure_map_text_floor_and_shared_panel_baseline(mito_gb, tmp_path):
    out = tmp_path / "structure.svg"
    ov.write(plot_structure_map(mito_gb, show_ir=False), out)
    sizes = _svg_font_sizes(out.read_text(encoding="utf-8"))
    assert sizes and min(sizes) >= 5.0


def test_two_maps_compose_into_one_lettered_figure(mito_gb, tmp_path):
    from organelleverse.visualization import compose_figure

    figure = compose_figure([plot_ogdraw_map(mito_gb), plot_genome_map(mito_gb)],
                            titles=["first", "second"])
    layout = figure.metrics["layout"]
    assert layout["columns"] == 2 and list(layout["letters"]) == ["a", "b"]
    assert layout["width_mm"] == pytest.approx(183)
    out = tmp_path / "fig.svg"
    ov.write(figure, out)
    text = out.read_text(encoding="utf-8")
    assert ">a<" in text and ">b<" in text and "first" in text
    with pytest.raises(ValueError, match="cannot be drawn as a panel"):
        compose_figure([plot_structure_map(mito_gb, show_ir=False)])


def test_crowded_ring_shrinks_labels_instead_of_overprinting():
    record = _mito_record(20_000)
    record.features = record.features[:1]
    for i in range(200):
        record.features.append(
            SeqFeature(SimpleLocation(i * 100, i * 100 + 50, strand=1), type="gene",
                       qualifiers={"gene": [f"orf{i}"]})
        )
    from organelleverse.visualization.ogdraw import _parse_genbank_record

    parsed = _parse_genbank_record(record)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    try:
        draw_mito_map(parsed, ax=ax, draw_center_text=False, draw_legend=False)
        label_sizes = {t.get_fontsize() for t in ax.texts}
    finally:
        plt.close(fig)
    assert max(label_sizes) < 6.5  # shrunk from the default 6.5 pt label size
    assert min(label_sizes) >= 5.0  # never below the journal floor


def test_region_band_is_drawn_and_named():
    parsed = parse_genbank_from_text()
    regions = {
        "LSC": {"start": 0, "length": 500},
        "IRb": {"start": 500, "length": 150},
        "SSC": {"start": 650, "length": 200},
        "IRa": {"start": 850, "length": 150},
    }
    fig, ax = plt.subplots()
    try:
        draw_mito_map(parsed, ax=ax, regions=regions)
        names = {t.get_text() for t in ax.texts}
    finally:
        plt.close(fig)
    assert {"LSC", "IRb", "SSC", "IRa"} <= names


def parse_genbank_from_text():
    from io import StringIO

    from organelleverse.visualization.ogdraw import _parse_genbank_record

    return _parse_genbank_record(SeqIO.read(StringIO(_STRUCTURE_GENBANK.content), "genbank"))


def _synthetic_plastome(tmp_path):
    rng = random.Random(11)

    def seq(n):
        return "".join(rng.choice("ACGT") for _ in range(n))

    lsc, ir, ssc = seq(40_000), seq(9_000), seq(7_000)
    full = lsc + ir + ssc + str(Seq(ir).reverse_complement())
    record = SeqRecord(Seq(full), id="cp1", name="cp1", description="Testus plastidus chloroplast.",
                       annotations={"molecule_type": "DNA"})
    record.features.append(
        SeqFeature(SimpleLocation(0, len(full), strand=1), type="source",
                   qualifiers={"organelle": ["plastid:chloroplast"]})
    )
    record.features.append(
        SeqFeature(SimpleLocation(1_000, 2_000, strand=1), type="gene", qualifiers={"gene": ["psbA"]})
    )
    path = tmp_path / "cp.gb"
    SeqIO.write([record], path, "genbank")
    return path


def test_plastid_map_resolves_the_quadripartite_band(tmp_path):
    path = _synthetic_plastome(tmp_path)
    regions = plot_ogdraw_map(path, dpi=50).metrics["regions"]
    assert set(regions) == {"LSC", "IRb", "SSC", "IRa"}
    assert regions["IRb"]["length"] == pytest.approx(9_000, abs=50)
    assert plot_ogdraw_map(path, show_ir=False).metrics["regions"] == {}
    assert plot_genome_map(path).metrics["regions"] == regions
    out = tmp_path / "cp.svg"
    ov.write(plot_ogdraw_map(path, dpi=50), out)
    assert "IRa" in out.read_text(encoding="utf-8")


def test_explicit_ir_on_a_record_without_one_fails(tmp_path):
    path = tmp_path / "s.gb"
    path.write_text(_STRUCTURE_GENBANK.content, encoding="utf-8")
    with pytest.raises(OrganelleInputError, match="quadripartite"):
        plot_ogdraw_map(path, show_ir=True)
    assert plot_ogdraw_map(path).metrics["regions"] == {}


def test_structure_map_still_matches_the_ogdraw_loci(mito_gb):
    rows = plot_structure_map(mito_gb).metrics["features"]
    assert {r["gene"]: r["splicing"] for r in rows if r["gene"] != "trnM"} == {
        "nad2": "trans",
        "nad5": "trans",
        "cox2": "cis",
    }


_TINY_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
    '<rect width="20" height="20" fill="red"/></svg>'
)


def _svg_only_runner(argv, env):
    """gbdraw without CairoSVG: writes the SVG, prints a hint, exits 0."""
    import subprocess
    from pathlib import Path

    prefix = Path(argv[argv.index("-o") + 1])
    prefix.with_suffix(".svg").write_text(_TINY_SVG, encoding="utf-8")
    return subprocess.CompletedProcess(argv, 0, "CairoSVG is not installed. Cannot generate: PNG", "")


def test_gbdraw_svg_only_output_is_converted_to_the_requested_png(tmp_path):
    pytest.importorskip("cairosvg")
    from organelleverse.visualization.gbdraw import plot_gbdraw

    gb = tmp_path / "s.gb"
    gb.write_text(_STRUCTURE_GENBANK.content, encoding="utf-8")
    plot = plot_gbdraw(gb, mode="circular", organelle="plastid", executable="gbdraw",
                       runner=_svg_only_runner)
    out = tmp_path / "map.png"
    ov.write(plot, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_gbdraw_svg_only_without_cairosvg_names_the_dependency(tmp_path, monkeypatch):
    import sys

    from organelleverse.visualization.gbdraw import plot_gbdraw

    monkeypatch.setitem(sys.modules, "cairosvg", None)
    gb = tmp_path / "s.gb"
    gb.write_text(_STRUCTURE_GENBANK.content, encoding="utf-8")
    plot = plot_gbdraw(gb, mode="circular", organelle="plastid", executable="gbdraw",
                       runner=_svg_only_runner)
    with pytest.raises(Exception, match="CairoSVG"):
        ov.write(plot, tmp_path / "map.png")
