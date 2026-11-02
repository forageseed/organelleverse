"""Circos-style transfer chord diagram: link normalization, layout, labels and rendering."""

from __future__ import annotations

import csv
import random
import re
import shutil
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

import organelleverse as ov
from organelleverse._losat import resolve_losat
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.core.result import OrganelleResult
from organelleverse.transfer import transfer_core
from organelleverse.visualization import plot_transfer_circos, transfer_links
from organelleverse.visualization.circos import _crossings, _layout

#: detect_mtpt (LOSAT) fragments for the fixture below: a reverse psbA piece, an
#: IR trnN copy that aligns to both IR arms, and an rbcL piece.
FRAGMENTS = [
    {"seqid": "MTFIX.1", "start": 495, "end": 700, "length": 206, "identity": 99.52, "strand": "-",
     "hsps": [{"source_seqid": "CPFIX.1", "source_start": 151, "source_end": 355, "start": 495,
               "end": 700, "identity": 99.52, "strand": "-", "alignment_length": 206}]},
    {"seqid": "MTFIX.1", "start": 1401, "end": 1500, "length": 100, "identity": 100.0, "strand": "+",
     "hsps": [{"source_seqid": "CPFIX.1", "source_start": 1291, "source_end": 1390, "start": 1401,
               "end": 1500, "identity": 100.0, "strand": "+", "alignment_length": 100},
              {"source_seqid": "CPFIX.1", "source_start": 2811, "source_end": 2910, "start": 1401,
               "end": 1500, "identity": 100.0, "strand": "-", "alignment_length": 100}]},
    {"seqid": "MTFIX.1", "start": 2401, "end": 2700, "length": 300, "identity": 100.0, "strand": "+",
     "hsps": [{"source_seqid": "CPFIX.1", "source_start": 601, "source_end": 900, "start": 2401,
               "end": 2700, "identity": 100.0, "strand": "+", "alignment_length": 300}]},
]


def _dna(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _rc(s: str) -> str:
    return str(Seq(s).reverse_complement())


def _record(seq, rid, organelle, description, features):
    rec = SeqRecord(Seq(seq), id=rid, name=rid.split(".")[0], description=description)
    rec.annotations.update({"molecule_type": "DNA", "topology": "circular", "organism": "Fixture plant"})
    source = SeqFeature(SimpleLocation(0, len(seq), 1), type="source",
                        qualifiers={"organelle": [organelle], "organism": ["Fixture plant"]})
    rec.features = [source, *features]
    return rec


def _feature(kind, gene, *parts):
    location = (
        SimpleLocation(*parts[0])
        if len(parts) == 1
        else CompoundLocation([SimpleLocation(*p) for p in parts])
    )
    return SeqFeature(location, type=kind, qualifiers={"gene": [gene]})


def _write_fixture(directory: Path) -> tuple[Path, Path]:
    """Mito 3.8 kb + quadripartite plastid 3 kb (IR 600 bp), same layout as the bundle fixture."""
    rng = random.Random(11)
    lsc, ir, ssc = _dna(rng, 1200), _dna(rng, 600), _dna(rng, 600)
    plastid = lsc + ir + ssc + _rc(ir)
    cp = _record(plastid, "CPFIX.1", "plastid:chloroplast", "Fixture plant chloroplast, complete genome", [
        _feature("gene", "psbA", (100, 400, -1)), _feature("CDS", "psbA", (100, 400, -1)),
        _feature("gene", "trnN-GUU", (1300, 1372, 1)), _feature("tRNA", "trnN-GUU", (1300, 1372, 1)),
        _feature("gene", "trnN-GUU", (2828, 2900, -1)), _feature("tRNA", "trnN-GUU", (2828, 2900, -1)),
        _feature("gene", "rbcL", (600, 900, 1)), _feature("CDS", "rbcL", (600, 900, 1)),
    ])
    mito = (_dna(rng, 500) + _rc(plastid[150:350]) + _dna(rng, 700) + plastid[1290:1390]
            + _dna(rng, 900) + plastid[600:900] + _dna(rng, 1100))
    mt = _record(mito, "MTFIX.1", "mitochondrion", "Fixture plant mitochondrion, complete genome", [
        _feature("gene", "nad1", (100, 300, 1), (3300, 3500, 1)),
        _feature("CDS", "nad1", (100, 300, 1), (3300, 3500, 1)),
        _feature("gene", "trnN-GUU", (1400, 1472, 1)), _feature("tRNA", "trnN-GUU", (1400, 1472, 1)),
    ])
    mito_path, cp_path = directory / "mito.gb", directory / "plastid.gb"
    SeqIO.write(mt, mito_path, "genbank")
    SeqIO.write(cp, cp_path, "genbank")
    return mito_path, cp_path


@pytest.fixture
def genomes(tmp_path):
    return _write_fixture(tmp_path)


# ── transfer core keeps donor coordinates ───────────────────────────────────


def test_merged_fragments_keep_every_hsp_with_donor_coordinates():
    rows = [
        # qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
        ["cp", "mt", "100.0", "100", "0", "0", "1291", "1390", "1401", "1500", "1e-40", "180"],
        ["cp", "mt", "98.0", "100", "2", "0", "2811", "2910", "1500", "1401", "1e-40", "170"],
        ["cp", "mt", "70.0", "500", "0", "0", "1", "500", "9001", "9500", "1e-5", "40"],
    ]
    fragments = transfer_core._merge_hits(transfer_core._hits(rows, min_identity=80, min_len=100))
    assert len(fragments) == 1
    fragment = fragments[0]
    assert (fragment["start"], fragment["end"], fragment["identity"]) == (1401, 1500, 99.0)
    assert [(h["source_start"], h["source_end"], h["strand"]) for h in fragment["hsps"]] == [
        (1291, 1390, "+"),
        (2811, 2910, "-"),
    ]
    assert all(h["source_seqid"] == "cp" for h in fragment["hsps"])


# ── normalization ────────────────────────────────────────────────────────────


def test_mtpt_fragments_become_one_link_per_hsp():
    links, unlinked, kind = transfer_links({"fragments": FRAGMENTS})
    assert kind == "MTPT" and not unlinked
    assert [(lk["recipient_start"], lk["donor_start"], lk["strand"]) for lk in links] == [
        (495, 151, "-"), (1401, 1291, "+"), (1401, 2811, "-"), (2401, 601, "+"),
    ]
    assert links[1]["fragment"] == links[2]["fragment"] == 1


def test_kmer_fallback_fragments_are_reported_unlinked(genomes):
    mito, cp = genomes
    rows = [*FRAGMENTS, {"seqid": "MTFIX.1", "start": 3001, "end": 3200, "length": 200, "strand": "+"}]
    plot = plot_transfer_circos(rows, [mito], [cp])
    assert plot.metrics["unlinked"][0]["recipient_start"] == 3001
    with pytest.raises(OrganelleInputError, match="donor coordinates"):
        plot_transfer_circos([rows[-1]], [mito], [cp])


def test_numt_candidates_infer_kind_from_donor_and_split_scales(tmp_path, genomes):
    _mito, cp = genomes
    nuclear = tmp_path / "nuclear.fa"
    rng = random.Random(3)
    nuclear.write_text(
        "".join(f">chr{i}\n{_dna(rng, 60_000)}\n" for i in (1, 2))
        + ">scaffold_99\nACGTACGTAC\n", encoding="utf-8"
    )
    candidates = [
        {"nuclear_seqid": "chr2", "nuclear_start": 30_001, "nuclear_end": 30_300,
         "organelle_seqid": "CPFIX.1", "organelle_start": 601, "organelle_end": 900,
         "identity": 97.5, "length": 300, "strand": "+"},
    ]
    plot = plot_transfer_circos({"candidates": candidates}, [nuclear], [cp], show_genes=False)
    metrics = plot.metrics
    assert metrics["transfer_kind"] == "NUPT"
    assert metrics["scale"] == "per_group"
    # unlinked tiny scaffolds are dropped, chromosomes kept
    assert [s["seqid"] for s in metrics["segments"] if s["role"] == "recipient"] == ["chr1", "chr2"]
    recipient = sum(s["end_deg"] - s["start_deg"] for s in metrics["segments"] if s["role"] == "recipient")
    donor = sum(s["end_deg"] - s["start_deg"] for s in metrics["segments"] if s["role"] == "donor")
    assert recipient == pytest.approx(donor)


def test_flat_links_and_min_length(genomes):
    mito, cp = genomes
    links = [
        {"recipient_seqid": "MTFIX", "recipient_start": 2401, "recipient_end": 2700,
         "donor_seqid": "CPFIX", "donor_start": 601, "donor_end": 900, "identity": 91},
        {"recipient_seqid": "MTFIX.1", "recipient_start": 10, "recipient_end": 60,
         "donor_seqid": "CPFIX.1", "donor_start": 10, "donor_end": 60},
    ]
    plot = plot_transfer_circos(links, [mito], [cp], min_length=100)
    metrics = plot.metrics
    assert metrics["link_count"] == 1 and metrics["dropped_below_min_length"] == 1
    # accession without version resolves to the record id
    assert metrics["links"][0]["recipient_seqid"] == "MTFIX.1"
    assert list(metrics["color_range"]) == [90, 100.0]
    with pytest.raises(OrganelleInputError, match=">= 1000 bp"):
        plot_transfer_circos(links, [mito], [cp], min_length=1000)


def test_bad_inputs_fail_explicitly(genomes):
    mito, cp = genomes
    with pytest.raises(OrganelleParameterError):
        plot_transfer_circos(FRAGMENTS, [mito], [cp], color_by="gc")
    with pytest.raises(OrganelleParameterError):
        plot_transfer_circos(FRAGMENTS, [mito], [cp], order="random")
    wrong = [{**FRAGMENTS[0], "seqid": "other"}]
    with pytest.raises(OrganelleInputError, match="not in the recipient"):
        plot_transfer_circos(wrong, [mito], [cp])
    outside = [{"recipient_seqid": "MTFIX.1", "recipient_start": 3700, "recipient_end": 3900,
                "donor_seqid": "CPFIX.1", "donor_start": 1, "donor_end": 200}]
    with pytest.raises(OrganelleInputError, match="outside"):
        plot_transfer_circos(outside, [mito], [cp])


# ── genes, regions, labels ──────────────────────────────────────────────────


def test_plastid_regions_genes_and_transferred_gene_labels(genomes):
    mito, cp = genomes
    plot = plot_transfer_circos(FRAGMENTS, [mito], [cp], gene_labels="transferred")
    metrics = plot.metrics
    segments = {s["seqid"]: s for s in metrics["segments"]}
    assert set(segments["CPFIX.1"]["regions"]) == {"LSC", "IRb", "SSC", "IRa"}
    assert segments["MTFIX.1"]["regions"] == {}
    assert segments["MTFIX.1"]["label"] == "mtDNA" and segments["CPFIX.1"]["label"] == "cpDNA"
    labels = {(row["seqid"], row["gene"]) for row in metrics["gene_labels"]}
    # rbcL fully carried; psbA 205/301 bp >= half; both IR copies of trnN; the mito trnN copy
    assert labels == {("CPFIX.1", "rbcL"), ("CPFIX.1", "psbA"), ("CPFIX.1", "trnN-GUU"),
                      ("MTFIX.1", "trnN-GUU")}
    assert all(row["transferred"] for row in metrics["gene_labels"])
    assert sum(1 for row in metrics["gene_labels"] if row["gene"] == "trnN-GUU" and row["seqid"] == "CPFIX.1") == 2
    psba = next(lk for lk in metrics["links"] if lk["donor_start"] == 151)
    assert [dict(g) for g in psba["donor_genes"]] == [{"gene": "psbA", "fraction": round(205 / 300, 3)}]
    strict = plot_transfer_circos(FRAGMENTS, [mito], [cp], gene_labels="transferred",
                                  gene_label_min_fraction=0.9)
    assert "psbA" not in {row["gene"] for row in strict.metrics["gene_labels"]}
    assert plot_transfer_circos(FRAGMENTS, [mito], [cp], gene_labels="none").metrics["gene_labels"] == ()


def test_auto_labels_name_every_gene_like_the_ogdraw_map_when_they_fit(genomes, monkeypatch):
    from organelleverse.visualization import circos as transfer_circos

    mito, cp = genomes
    metrics = plot_transfer_circos(FRAGMENTS, [mito], [cp]).metrics
    assert dict(metrics["gene_label_modes"]) == {"MTFIX.1": "all", "CPFIX.1": "all"}
    rows = {(row["seqid"], row["gene"]): row for row in metrics["gene_labels"]}
    assert rows[("MTFIX.1", "nad1")]["transferred"] is False  # named, not carried
    assert rows[("CPFIX.1", "rbcL")]["transferred"] is True
    assert rows[("CPFIX.1", "psbA")]["strand"] == -1
    # a segment whose names would not fit falls back to the transferred genes
    # an inner (minus-strand) ring with room for one name: the mito has no minus-strand
    # names and keeps all; the plastid's two inward names (psbA, IRa trnN) do not fit at
    # 6.5 pt until each transferred name is given once (psbA alone inside), and names
    # never shrink below 6.5 pt; the caption, not the figure, says what was left out
    def crowded_inner(span, unit_in, count, radius=None):
        return 4.0 if radius is not None and radius < 0.85 and count >= 2 else 6.5

    monkeypatch.setattr(transfer_circos, "_label_font", crowded_inner)
    crowded = plot_transfer_circos(FRAGMENTS, [mito], [cp]).metrics
    assert dict(crowded["gene_label_modes"]) == {"MTFIX.1": "all", "CPFIX.1": "transferred_once"}
    assert list(crowded["gene_label_notes"]) == [
        "cpDNA: only genes carried by a transfer are named, each name once"
    ]
    assert "each name once" in crowded["caption"]
    assert {r["font_pt"] for r in crowded["gene_labels"]} == {6.5}
    assert sorted(r["gene"] for r in crowded["gene_labels"] if r["seqid"] == "CPFIX.1") == [
        "psbA", "rbcL", "trnN-GUU"
    ]
    with pytest.raises(OrganelleParameterError):
        plot_transfer_circos(FRAGMENTS, [mito], [cp], gene_labels="some")


def test_minus_strand_names_sit_inside_the_ring_like_ogdraw(tmp_path, genomes):
    from organelleverse.visualization.circos import _BAND_OUT_MIN, _R_BAND_OUT, _inner_shift

    mito, cp = genomes
    inside = plot_transfer_circos(FRAGMENTS, [mito], [cp])
    assert inside.metrics["minus_labels"] == "inside"
    assert any(r["strand"] == -1 for r in inside.metrics["gene_labels"])  # psbA, IRa trnN
    rows = [{"gene": "trnS(GGA)", "strand": -1, "font_pt": 6.5, "transferred": True}]
    shift = _inner_shift(rows, unit_in=1.7, minus_inside=True)
    assert shift < 0 and _R_BAND_OUT + shift >= _BAND_OUT_MIN  # band recedes, within its floor
    assert _inner_shift(rows, unit_in=1.7, minus_inside=False) == 0.0
    assert inside.metrics["minus_labels_drawn"] == "inside"
    # a ring too small for inward names (band would pass its floor) puts them outside
    from organelleverse.visualization.circos import _fit_labels

    metrics = inside.metrics
    _rows, _modes, _notes, drawn_inside = _fit_labels(
        metrics["segments"], metrics["gene_label_candidates"], "auto", 0.15, True
    )
    assert drawn_inside is False
    outside = plot_transfer_circos(FRAGMENTS, [mito], [cp], minus_labels="outside")
    for name, plot in (("inside", inside), ("outside", outside)):
        ov.write(plot, tmp_path / f"{name}.svg")
    assert "\u2212 strand inside" in (tmp_path / "inside.svg").read_text(encoding="utf-8")
    with pytest.raises(OrganelleParameterError):
        plot_transfer_circos(FRAGMENTS, [mito], [cp], minus_labels="left")


def test_zoom_inset_shows_one_ribbon_at_true_scale(tmp_path, genomes):
    mito, cp = genomes
    metrics = plot_transfer_circos(FRAGMENTS, [mito], [cp]).metrics
    zoom = metrics["zoom"]
    shortest = min(metrics["links"], key=lambda lk: (lk["length"], int(lk["id"][1:])))
    assert zoom["link"] == shortest["id"] and zoom["length"] == shortest["length"] == 100
    assert zoom["span_bp"] >= 5 * zoom["length"]
    for role in ("recipient", "donor"):
        window = zoom[role]
        assert list(window["frag"]) == [shortest[f"{role}_start"], shortest[f"{role}_end"]]
        assert window["start"] <= window["frag"][0] <= window["frag"][1] <= window["end"]
        assert window["end"] - window["start"] + 1 == zoom["span_bp"]
    assert {g["gene"] for g in zoom["recipient"]["genes"]} == {"trnN-GUU"}
    assert "Inset: ribbon" in metrics["caption"]
    assert plot_transfer_circos(FRAGMENTS, [mito], [cp], zoom="L4").metrics["zoom"]["link"] == "L4"
    assert plot_transfer_circos(FRAGMENTS, [mito], [cp], zoom=None).metrics["zoom"] is None
    with pytest.raises(OrganelleParameterError):
        plot_transfer_circos(FRAGMENTS, [mito], [cp], zoom="L99")
    ov.write(plot_transfer_circos(FRAGMENTS, [mito], [cp]), tmp_path / "zoom.svg")
    assert "at true scale" in (tmp_path / "zoom.svg").read_text(encoding="utf-8")


def test_fasta_inputs_draw_without_genes(tmp_path, genomes):
    mito, cp = genomes
    mito_fa, cp_fa = tmp_path / "mito.fa", tmp_path / "cp.fa"
    SeqIO.write(SeqIO.read(mito, "genbank"), mito_fa, "fasta")
    SeqIO.write(SeqIO.read(cp, "genbank"), cp_fa, "fasta")
    metrics = plot_transfer_circos(FRAGMENTS, [mito_fa], [cp_fa]).metrics
    assert metrics["transfer_kind"] == "MTPT"
    assert all(not s["genes"] and not s["regions"] for s in metrics["segments"])
    assert not metrics["gene_labels"]


# ── layout ───────────────────────────────────────────────────────────────────


def _three_contigs(tmp_path) -> tuple[list[Path], Path, list[dict]]:
    rng = random.Random(5)
    paths = []
    for name in ("ctgA", "ctgB", "ctgC"):
        path = tmp_path / f"{name}.fa"
        path.write_text(f">{name}\n{_dna(rng, 20_000)}\n", encoding="utf-8")
        paths.append(path)
    cp = tmp_path / "cp.fa"
    cp.write_text(f">cp\n{_dna(rng, 30_000)}\n", encoding="utf-8")
    # ctgA links to the start of cp, ctgC to its end: in file order every pair crosses
    links = []
    for contig, base in (("ctgA", 5_000), ("ctgB", 15_000), ("ctgC", 25_000)):
        for k in range(3):
            links.append({"recipient_seqid": contig, "recipient_start": 2_000 + 5_000 * k,
                          "recipient_end": 2_400 + 5_000 * k, "donor_seqid": "cp",
                          "donor_start": base + 1_000 * k, "donor_end": base + 1_000 * k + 400,
                          "identity": 90 + k})
    return paths, cp, links


def test_auto_order_reduces_ribbon_crossings(tmp_path):
    contigs, cp, links = _three_contigs(tmp_path)
    kept = plot_transfer_circos(links, contigs, [cp], order="input").metrics
    auto = plot_transfer_circos(links, contigs, [cp]).metrics
    assert kept["crossings"] == kept["crossings_input_order"] > 0
    assert auto["crossings"] < auto["crossings_input_order"] == kept["crossings"]
    assert [s["seqid"] for s in kept["segments"]] == ["ctgA", "ctgB", "ctgC", "cp"]
    assert not any(s["reversed"] for s in kept["segments"])


def test_crossing_count_matches_chord_geometry():
    segments = [{"seqid": "a", "length": 100, "role": "recipient"},
                {"seqid": "b", "length": 100, "role": "donor"}]
    placed = _layout(segments, set(), "proportional")
    parallel = [
        {"recipient_seqid": "a", "recipient_start": 10, "recipient_end": 10, "donor_seqid": "b",
         "donor_start": 90, "donor_end": 90, "length": 1},
        {"recipient_seqid": "a", "recipient_start": 90, "recipient_end": 90, "donor_seqid": "b",
         "donor_start": 10, "donor_end": 10, "length": 1},
    ]
    assert _crossings(placed, parallel) == 0
    crossed = [dict(parallel[0], donor_start=10, donor_end=10), dict(parallel[1], donor_start=90, donor_end=90)]
    assert _crossings(placed, crossed) == 1
    flipped = _layout(segments, {"b"}, "proportional")
    assert _crossings(flipped, crossed) == 0


# ── rendering ────────────────────────────────────────────────────────────────


def _font_sizes(svg: Path) -> list[float]:
    text = svg.read_text(encoding="utf-8")
    return [float(v) for v in re.findall(r"font(?:-size)?:[^;\"]*?([\d.]+)px", text)]


def test_write_renders_png_svg_pdf_and_links_table(tmp_path, genomes):
    mito, cp = genomes
    plot = plot_transfer_circos(FRAGMENTS, [mito], [cp])
    assert not plot.artifacts
    for suffix in ("png", "svg", "pdf"):
        result = ov.write(plot, tmp_path / f"circos.{suffix}")
        assert {a.kind for a in result.artifacts} == {"figure", "figure_data"}
        assert (tmp_path / f"circos.{suffix}").stat().st_size > 1000
    svg = (tmp_path / "circos.svg").read_text(encoding="utf-8")
    assert "rbcL" in svg and 'id="L1"' in svg  # editable text, ribbons addressable by link id
    rows = list(csv.DictReader((tmp_path / "circos.links.tsv").open(), delimiter="\t"))
    assert [r["id"] for r in rows] == ["L1", "L2", "L3", "L4"]
    assert rows[3]["donor_genes"] == "rbcL:1"


def test_printed_size_and_journal_type(tmp_path, genomes):
    mito, cp = genomes
    double = plot_transfer_circos(FRAGMENTS, [mito], [cp])
    assert double.metrics["width_mm"] == 183.0 and double.metrics["height_mm"] <= 170.0
    assert "MTPT" in double.metrics["caption"]
    for name, plot in (("double", double), ("single", plot_transfer_circos(FRAGMENTS, [mito], [cp], width="single"))):
        ov.write(plot, tmp_path / f"{name}.svg")
        svg = (tmp_path / f"{name}.svg").read_text(encoding="utf-8")
        width_pt = float(re.search(r'<svg[^>]*width="([\d.]+)pt"', svg).group(1))
        assert width_pt == pytest.approx(plot.metrics["width_mm"] / 25.4 * 72, abs=0.5)
        sizes = _font_sizes(tmp_path / f"{name}.svg")
        assert min(sizes) >= 5.0  # nothing below the 5 pt floor
        assert max(sizes) <= 8.0  # no in-figure title; body 7 pt, ticks 6 pt
        assert 6.0 in sizes and 7.0 in sizes
    with pytest.raises(OrganelleParameterError):
        plot_transfer_circos(FRAGMENTS, [mito], [cp], width="poster")


def test_draws_as_a_panel_of_a_composed_figure(tmp_path, genomes):
    from organelleverse.visualization import compose_figure, plot_ogdraw_map

    mito, cp = genomes
    chord = plot_transfer_circos(FRAGMENTS, [mito], [cp])
    assert chord.panel is not None and chord.panel.header.endswith("4 ribbons")
    figure = compose_figure([chord, plot_ogdraw_map([cp], show_ir=False)])
    assert list(figure.metrics["layout"]["letters"]) == ["a", "b"]
    out = ov.write(figure, tmp_path / "composite.pdf")
    assert Path(out.artifacts[0].uri).stat().st_size > 1000
    # one gene-name size across both panels (italic text = gene names)
    ov.write(figure, tmp_path / "composite.svg")
    svg = (tmp_path / "composite.svg").read_text(encoding="utf-8")
    italic = {
        float(re.search(r"([\d.]+)px", style).group(1))
        for style in re.findall(r'<text\b[^>]*style="([^"]*italic[^"]*)"', svg)
    }
    assert len(italic) == 1, italic
    restored = OrganelleResult.model_validate(chord.model_dump(mode="json"))
    from organelleverse.visualization.circos import _restore_transfer_circos

    assert _restore_transfer_circos(restored).panel is not None


def test_serialized_result_renders_through_write(tmp_path, genomes):
    mito, cp = genomes
    plot = plot_transfer_circos(FRAGMENTS, [mito], [cp], color_by="length")
    restored = OrganelleResult.model_validate(plot.model_dump(mode="json"))
    assert type(restored) is OrganelleResult  # renderer gone, as after a capability call
    published = ov.write(restored, tmp_path / "restored.png")
    assert Path(published.artifacts[0].uri).name == "restored.png"


@pytest.mark.skipif(
    resolve_losat() is None and shutil.which("blastn") is None, reason="needs LOSAT or NCBI blastn"
)
def test_detect_mtpt_result_draws_directly(tmp_path, genomes):
    from organelleverse.transfer import detect_mtpt

    mito, cp = genomes
    mito_fa, cp_fa = tmp_path / "mito.fa", tmp_path / "cp.fa"
    SeqIO.write(SeqIO.read(mito, "genbank"), mito_fa, "fasta")
    SeqIO.write(SeqIO.read(cp, "genbank"), cp_fa, "fasta")
    result = detect_mtpt(mito_fa, cp_fa)
    plot = plot_transfer_circos(result, [mito], [cp])
    donors = sorted((lk["donor_start"], lk["donor_end"]) for lk in plot.metrics["links"])
    # the trnN copy links to both IR arms
    assert any(abs(a - 1291) <= 5 for a, _ in donors) and any(abs(a - 2811) <= 5 for a, _ in donors)
    assert plot.metrics["transfer_kind"] == "MTPT"
