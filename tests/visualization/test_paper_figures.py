"""IR boundary, π profile, repeat summary and gene presence figures."""

from __future__ import annotations

import csv

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

import organelleverse as ov
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.core.result import OrganelleResult
from organelleverse.visualization import paper_figures as pf

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _seq(n: int, seed: int) -> str:
    """Deterministic pseudo-random DNA (periodic toy sequences make IR detection quadratic)."""
    x, out = seed, []
    for _ in range(n):
        x = (1103515245 * x + 12345) % 2**31
        out.append("ACGT"[(x >> 16) & 3])
    return "".join(out)


def _rc(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _genbank(path, seq, features, *, organism="Testus plastidus", organelle="plastid:chloroplast",
             accession="TEST1"):
    record = SeqRecord(Seq(seq), id=f"{accession}.1", name=accession,
                       description=f"{organism} chloroplast, complete genome")
    record.annotations.update(molecule_type="DNA", topology="circular", organism=organism)
    record.features.append(SeqFeature(FeatureLocation(0, len(seq), 1), type="source",
                                      qualifiers={"organism": [organism], "organelle": [organelle]}))
    for ftype, name, start, end, strand, *extra in features:
        qualifiers = {**({"gene": [name]} if name else {}), **(extra[0] if extra else {})}
        record.features.append(SeqFeature(FeatureLocation(start - 1, end, strand), type=ftype,
                                          qualifiers=qualifiers))
    SeqIO.write(record, path, "genbank")
    return path


def _plastome(path, *, organism="Testus plastidus", accession="TEST1"):
    """LSC 3000 / IRb 800 / SSC 1200 / IRa 800 with genes placed at the junctions.

    Each single-copy region starts and ends with A8: A never pairs with A across
    the inverted copies, so the detector's extension stops exactly at the junction.
    """
    ir = _seq(800, 11)
    guard = "A" * 8
    seq = guard + _seq(2984, 3) + guard + ir + guard + _seq(1184, 5) + guard + _rc(ir)
    return _genbank(path, seq, [
        ("gene", "psbA", 400, 1300, -1),
        ("gene", "trnH-GUG", 20, 95, -1),
        ("gene", "rps19", 2951, 3100, 1),        # crosses JLB (3000|3001)
        ("gene", "rpl2", 3300, 3700, -1),
        ("gene", "ndhF", 3850, 4300, -1),        # SSC side of JSB, 49 bp away
        ("gene", "ycf1", 4800, 5200, -1),        # crosses JSA (5000|5001)
    ], organism=organism, accession=accession)


# ---------------------------------------------------------------------------
# IR boundaries
# ---------------------------------------------------------------------------


def test_ir_boundaries_reads_junction_genes_and_distances(tmp_path):
    first = _plastome(tmp_path / "a.gb", organism="Alpha one")
    second = _genbank(tmp_path / "b.gb", _seq(5000, 17), [("gene", "psbA", 100, 900, -1)],
                      organism="Beta two", accession="TEST2")
    plot = pf.plot_ir_boundaries([first, second], window=1000)
    genomes = plot.metrics["genomes"]
    assert [g["name"] for g in genomes] == ["Alpha one", "Beta two"]
    a = genomes[0]
    assert a["ir_detected"]
    lengths = {k: v["length"] for k, v in a["regions"].items()}
    assert lengths == {"LSC": 3000, "IRb": 800, "SSC": 1200, "IRa": 800}
    junctions = {j["id"]: j for j in a["junctions"]}
    assert [j["position"] for j in a["junctions"]] == [3000, 3800, 5000, 5800]
    rps19 = next(g for g in junctions["JLB"]["genes"] if g["gene"] == "rps19")
    assert rps19["relation"] == "crosses"
    assert (rps19["left_bp"], rps19["right_bp"]) == (50, 100)
    ndhf = next(g for g in junctions["JSB"]["genes"] if g["gene"] == "ndhF")
    assert ndhf["relation"] == "right" and ndhf["distance"] == 49
    # No gene crosses JSB, so the nearest gene on the IRb side is drawn too.
    rpl2 = next(g for g in junctions["JSB"]["genes"] if g["gene"] == "rpl2")
    assert rpl2["relation"] == "left" and rpl2["distance"] == 100
    # A crossing gene is the nearest gene on both sides: rps19 stands alone at JLB.
    assert [g["gene"] for g in junctions["JLB"]["genes"]] == ["rps19"]
    assert any(g["gene"] == "ycf1" and g["relation"] == "crosses" for g in junctions["JSA"]["genes"])
    # The IRa/LSC junction wraps the origin; trnH sits 19 bp into the LSC.
    trnh = next(g for g in junctions["JLA"]["genes"] if g["gene"] == "trnH-GUG")
    assert trnh["relation"] == "right" and trnh["distance"] == 19
    assert not genomes[1]["ir_detected"] and not genomes[1]["junctions"]
    assert "ir_not_detected" in plot.flags


def test_ir_boundaries_writes_figure_and_table_and_survives_serialization(tmp_path):
    gb = _plastome(tmp_path / "a.gb")
    plot = pf.plot_ir_boundaries([gb, gb], names=["One", "Two"])
    published = ov.write(plot, tmp_path / "ir.svg")
    assert {a.kind for a in published.artifacts} == {"figure", "figure_data"}
    svg = (tmp_path / "ir.svg").read_text()
    assert "rps19" in svg and "JLB" in svg
    rows = list(csv.DictReader((tmp_path / "ir.junctions.tsv").open(), delimiter="\t"))
    assert {r["genome"] for r in rows} == {"One", "Two"}
    restored = OrganelleResult.model_validate_json(plot.model_dump_json())
    gb.unlink()
    again = ov.write(restored, tmp_path / "again.png")
    assert (tmp_path / "again.png").stat().st_size > 1000
    assert again.metrics["genomes"][0]["name"] == "One"


def test_ir_boundaries_rejects_multi_record_files(tmp_path):
    gb = _plastome(tmp_path / "a.gb")
    both = tmp_path / "both.gb"
    both.write_text(gb.read_text() * 2)
    with pytest.raises(OrganelleInputError):
        pf.plot_ir_boundaries([both])


# ---------------------------------------------------------------------------
# nucleotide diversity
# ---------------------------------------------------------------------------


def _alignment(tmp_path):
    """Four 3,000 bp sequences, diverse in 1,000-1,600; the reference has a 10-column gap."""
    base = list(_seq(3000, 7))
    rows = []
    for i in range(4):
        seq = base[:]
        for p in range(len(seq)):
            if (1000 <= p < 1600 and (p + i) % 5 == 0) or (p % 397 == 0 and i == 3):
                seq[p] = "ACGT"[("ACGT".index(seq[p]) + 1 + i) % 4]
        rows.append(seq)
    reference = rows[0][:100] + ["-"] * 10 + rows[0][100:]
    others = [r[:100] + list(_seq(10, 23)) + r[100:] for r in rows[1:]]
    path = tmp_path / "aln.fasta"
    path.write_text("".join(f">s{i}\n{''.join(r)}\n" for i, r in enumerate([reference, *others])))
    gb = _genbank(tmp_path / "ref.gb", "".join(rows[0]), [
        ("gene", "psbA", 1, 400, -1),
        ("gene", "matK", 1000, 1700, 1),
        ("gene", "rbcL", 2200, 2900, 1),
    ])
    return path, gb


def test_diversity_maps_alignment_columns_to_reference_and_names_peaks(tmp_path):
    aln, gb = _alignment(tmp_path)
    plot = pf.plot_nucleotide_diversity(aln, genbank_path=gb, show_ir=False, top_n=2)
    m = plot.metrics
    assert m["coordinate"] == "reference" and m["reference_length"] == 3000
    first = m["windows"][0]
    assert (first["start"], first["end"]) == (1, 600)
    assert (first["x_start"], first["x_end"]) == (1, 590)  # 10 reference gap columns dropped
    best = max(m["peaks"], key=lambda p: p["pi"])
    assert best["label"] == "matK" and best["pi"] == m["max_pi"]
    assert all(p["label"] for p in m["peaks"])  # the lower peak is named by its spacer
    assert m["max_pi"] == max(w["pi"] for w in m["windows"])
    ov.write(plot, tmp_path / "pi.png")
    rows = list(csv.DictReader((tmp_path / "pi.windows.tsv").open(), delimiter="\t"))
    assert any(r["peak_label"] == "matK" for r in rows)


def test_diversity_threshold_labels_each_run_once(tmp_path):
    aln, gb = _alignment(tmp_path)
    plot = pf.plot_nucleotide_diversity(aln, genbank_path=gb, show_ir=False, threshold=0.05)
    peaks = plot.metrics["peaks"]
    assert len(peaks) == 1 and peaks[0]["label"] == "matK"
    assert peaks[0]["pi"] >= 0.05
    ov.write(plot, tmp_path / "pi.svg")
    assert "matK" in (tmp_path / "pi.svg").read_text()


def test_diversity_accepts_precomputed_windows_and_guards_coordinates(tmp_path):
    from organelleverse.diversity.diversity_core import compute_nucleotide_diversity

    aln, gb = _alignment(tmp_path)
    windows = compute_nucleotide_diversity(aln, window_size=600, step=200)["windows"]
    plot = pf.plot_nucleotide_diversity(windows=windows)
    assert plot.metrics["coordinate"] == "alignment"
    with pytest.raises(OrganelleParameterError):
        pf.plot_nucleotide_diversity(windows=windows, genbank_path=gb)
    mapped = pf.plot_nucleotide_diversity(aln, windows=windows, genbank_path=gb, show_ir=False)
    assert mapped.metrics["coordinate"] == "reference"
    short = _genbank(tmp_path / "short.gb", _seq(2000, 3), [("gene", "matK", 1, 500, 1)])
    with pytest.raises(OrganelleInputError):
        pf.plot_nucleotide_diversity(aln, genbank_path=short, show_ir=False)
    with pytest.raises(OrganelleParameterError):
        pf.plot_nucleotide_diversity()


# ---------------------------------------------------------------------------
# repeats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("A", "A/T"), ("T", "A/T"), ("C", "C/G"), ("AT", "AT/AT"), ("TA", "AT/AT"),
     ("AG", "AG/CT"), ("GA", "AG/CT"), ("TC", "AG/CT"), ("AAT", "AAT/ATT"), ("TAA", "AAT/ATT"),
     ("ATT", "AAT/ATT"), ("AAAT", "AAAT/ATTT")],
)
def test_motif_class_follows_misa_convention(unit, expected):
    assert pf.motif_class(unit) == expected


def test_repeat_summary_counts_types_motifs_and_length_classes(tmp_path):
    ssrs = [
        [{"unit": "A", "start": 1, "end": 12}, {"unit": "T", "start": 30, "end": 40},
         {"unit": "AT", "start": 50, "end": 60}, {"unit": "AAT", "start": 70, "end": 84},
         {"unit": "AA", "start": 1, "end": 12}, {"unit": "ATAT", "start": 50, "end": 60}],
        [{"unit": "C", "start": 1, "end": 10}],
    ]
    repeats = [
        [{"type": "direct", "length": 31, "positions": [1, 500]},
         {"type": "inverted", "length": 45, "positions": [10, 900]},
         {"type": "inverted", "length": 26264, "positions": [84171, 128215]}],
        [{"type": "direct", "length": 39, "positions": [3, 300]}],
    ]
    plot = pf.plot_repeat_summary([], ssrs=ssrs, repeats=repeats, names=["Alpha", "Beta"])
    a, b = plot.metrics["samples"]
    assert a["ssr_types"] == {"mono": 2, "di": 1, "tri": 1, "tetra": 0, "penta": 0, "hexa": 0}
    assert a["motifs"] == {"A/T": 2, "AT/AT": 1, "AAT/ATT": 1}
    assert a["non_primitive_skipped"] == 2 and a["ssr_total"] == 4
    assert b["motifs"] == {"C/G": 1}
    bins = [x["label"] for x in plot.metrics["length_bins"]]
    first = f"30{pf.EN_DASH}39"
    assert bins[0] == first and bins[-1] == "≥80"
    assert a["repeat_bins"][first] == {"direct": 1, "inverted": 0}
    assert a["repeat_bins"]["≥80"] == {"direct": 0, "inverted": 1}
    assert b["repeat_types"] == {"direct": 1, "inverted": 0}
    ov.write(plot, tmp_path / "rep.pdf")
    rows = list(csv.DictReader((tmp_path / "rep.repeats.tsv").open(), delimiter="\t"))
    assert {"genome": "Alpha", "table": "ssr_motif", "class": "A/T", "count": "2"} in rows


def test_repeat_summary_runs_the_existing_detectors_from_fasta(tmp_path):
    from organelleverse.structure.structure_core import compute_multiconf, compute_repeats

    unit = _seq(40, 31)
    seq = _seq(300, 1) + "A" * 12 + _seq(300, 2) + unit + _seq(300, 4) + unit + _seq(200, 6)
    fasta = tmp_path / "g.fasta"
    fasta.write_text(f">g\n{seq}\n")
    plot = pf.plot_repeat_summary([fasta], min_repeat_len=30)
    sample = plot.metrics["samples"][0]
    assert sample["name"] == "g"
    primitive = [r for r in compute_repeats(fasta)["ssrs"] if pf._primitive(r["unit"])]
    assert sample["ssr_total"] == len(primitive) >= 1
    assert sample["repeat_total"] == len(compute_multiconf(fasta, min_repeat_len=30)["repeats"])
    assert sample["repeat_types"]["direct"] >= 1
    ov.write(plot, tmp_path / "one.png")
    with pytest.raises(OrganelleParameterError):
        pf.plot_repeat_summary([fasta], ssrs=[[]])


# ---------------------------------------------------------------------------
# gene presence
# ---------------------------------------------------------------------------


def _presence_files(tmp_path):
    seq = _seq(4000, 9)
    a = _genbank(tmp_path / "A.gb", seq, [
        ("gene", "psbA", 1, 300, 1), ("gene", "ndhB", 400, 900, 1),
        ("tRNA", "trnH-GUG", 1000, 1075, -1), ("rRNA", "rrn16S", 1200, 2600, 1),
        ("gene", "ycf3", 2700, 3000, 1),
    ], organism="Alpha one", accession="ACC1")
    b = _genbank(tmp_path / "B.gb", seq, [
        ("gene", "psbA", 1, 300, 1), ("gene", "ndhB", 400, 900, 1, {"pseudo": [""]}),
        ("tRNA", "trnH", 1000, 1075, -1), ("rRNA", "rrn16", 1200, 2600, 1),
        ("gene", "pafI", 2700, 3000, 1),
    ], organism="Beta two", accession="ACC2")
    c = _genbank(tmp_path / "C.gb", seq, [
        ("gene", "psbA", 1, 300, 1),
        ("tRNA", "", 1000, 1075, -1, {"product": ["tRNA-His"], "note": ["tRNA-His(GUG)"]}),
        ("rRNA", "", 1200, 2600, 1, {"product": ["16S ribosomal RNA"]}),
    ], organism="Gamma three", accession="ACC3")
    return [a, b, c]


def test_gene_presence_statuses_names_and_tree_order(tmp_path):
    files = _presence_files(tmp_path)
    tree = tmp_path / "t.nwk"
    tree.write_text("(Gamma_three:0.2,(Beta_two:0.1,ACC1:0.1):0.3,Extra_tip:0.5);\n")
    plot = pf.plot_gene_presence(files, tree_path=tree, transferred={"Gamma three": ["ndhB"]})
    m = plot.metrics
    assert m["order"] == "tree"
    assert list(m["species"]) == ["Gamma three", "Beta two", "Alpha one"]
    genes = [g["gene"] for g in m["genes"]]
    assert m["trna_names"] == "amino_acid"  # one genome writes trnH without an anticodon
    assert set(genes) == {"psbA", "ndhB", "trnH", "rrn16", "ycf3"}
    assert genes.index("psbA") < genes.index("ndhB") < genes.index("trnH")  # OGDraw class order
    status = {(sp, g): s for sp, row in zip(m["species"], m["matrix"], strict=True)
              for g, s in zip(genes, row, strict=True)}
    assert status[("Alpha one", "ndhB")] == "present"
    assert status[("Beta two", "ndhB")] == "pseudogene"
    assert status[("Gamma three", "ndhB")] == "transferred"
    assert status[("Beta two", "ycf3")] == "present"  # pafI is ycf3
    assert status[("Gamma three", "ycf3")] == "absent"
    # Gamma names its tRNA and rRNA only through /product.
    assert status[("Gamma three", "trnH")] == "present"
    assert status[("Gamma three", "rrn16")] == "present"
    assert m["tree_segments"]
    ov.write(plot, tmp_path / "gp.svg")
    rows = list(csv.DictReader((tmp_path / "gp.matrix.tsv").open(), delimiter="\t"))
    assert rows[1]["genome"] == "Beta two" and rows[1]["ndhB"] == "pseudogene"


def test_gene_presence_variable_only_cluster_and_errors(tmp_path):
    files = _presence_files(tmp_path)
    plot = pf.plot_gene_presence(files, variable_only=True)
    assert plot.metrics["order"] == "cluster"
    assert {g["gene"] for g in plot.metrics["genes"]} == {"ndhB", "ycf3"}
    assert plot.metrics["tree_segments"]
    restored = OrganelleResult.model_validate_json(plot.model_dump_json())
    ov.write(restored, tmp_path / "c.png")
    anticodon = pf.plot_gene_presence(files[::2], trna_names="anticodon", order="input")
    assert "trnH-GUG" in [g["gene"] for g in anticodon.metrics["genes"]]
    bad = tmp_path / "bad.nwk"
    bad.write_text("(Nobody,Else);\n")
    with pytest.raises(OrganelleInputError):
        pf.plot_gene_presence(files, tree_path=bad)
    with pytest.raises(OrganelleParameterError):
        pf.plot_gene_presence(files, order="sideways")


def test_gene_presence_wraps_long_gene_lists_into_balanced_blocks():
    genes = [{"gene": f"g{i}", "category": f"c{i // 7}"} for i in range(110)]
    blocks = pf._blocks(genes, 60)
    assert [i for block in blocks for i in block] == list(range(110))
    assert len(blocks) == 2 and all(len(block) <= 60 for block in blocks)
    # The split falls between classes, not inside one.
    first, second = blocks
    assert genes[first[-1]]["category"] != genes[second[0]]["category"]
    assert pf._blocks(genes, 0) == [list(range(110))]
    huge = [{"gene": f"g{i}", "category": "one"} for i in range(130)]
    assert [len(b) for b in pf._blocks(huge, 60)] == [43, 43, 44]  # fewest blocks, balanced
    tail = [{"gene": f"g{i}", "category": "a" if i < 57 else "b" if i < 106 else "r"}
            for i in range(110)]
    assert [len(b) for b in pf._blocks(tail, 60)] == [57, 53]
    # No two-block split avoids tearing a class here, so the balanced one wins.
    mito = [{"gene": f"g{i}", "category": "a" if i < 58 else "t" if i < 103 else "r"}
            for i in range(111)]
    assert [len(b) for b in pf._blocks(mito, 56)] == [55, 56]


def test_figures_are_journal_sized_panels_and_compose(tmp_path):
    from organelleverse.visualization.compose import compose_figure

    ir = pf.plot_ir_boundaries([_plastome(tmp_path / "a.gb")] * 2, names=["One", "Two"])
    gp = pf.plot_gene_presence(_presence_files(tmp_path))
    for plot in (ir, gp):
        assert plot.metrics["width_mm"] == 183.0
        assert 0 < plot.metrics["height_mm"] <= 170
        assert plot.panel is not None and plot.panel.aspect > 0
    single = pf.plot_ir_boundaries([tmp_path / "a.gb"], width="single")
    assert single.metrics["width_mm"] == 89.0
    restored = pf.restore_paper_figure(OrganelleResult.model_validate_json(gp.model_dump_json()))
    assert restored.panel is not None
    figure = compose_figure([ir, gp])
    ov.write(figure, tmp_path / "composite.pdf")
    assert (tmp_path / "composite.pdf").stat().st_size > 1000
    ov.write(ir, tmp_path / "ir.svg")
    svg = (tmp_path / "ir.svg").read_text()
    assert "rps19" in svg  # text stays editable text (svg.fonttype none)
