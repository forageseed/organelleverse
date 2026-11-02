"""Pangenome circle: VCF tracks, the graph skeleton, the real Citrus fixture and restore."""

from __future__ import annotations

import random
import re

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

import organelleverse as ov
from organelleverse.capabilities.adapters.visualization import (
    _PAN_CIRCULAR_GENBANK,
    _PAN_CIRCULAR_GFA,
    _PAN_CIRCULAR_VCF,
)
from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.core.result import OrganelleResult
from organelleverse.visualization.pan_circular import _vcf_tracks, plot_pan_circular
from organelleverse.visualization.pan_graph import pangenome_skeleton

_HEADER = (
    "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
)


def _write_fixture(tmp_path):
    paths = {}
    for item in (_PAN_CIRCULAR_GENBANK, _PAN_CIRCULAR_VCF, _PAN_CIRCULAR_GFA):
        path = tmp_path / item.relative_path
        path.write_text(item.content, encoding="utf-8")
        paths[item.relative_path] = path
    return paths


def test_vcf_counts_genomes_and_computes_pi_with_the_implicit_reference(tmp_path):
    vcf = tmp_path / "v.vcf"
    vcf.write_text(
        _HEADER
        # SNP: alleles ref,1,0,1 -> p = 0.5, pi = 4/3 * 0.5
        + "ref#1#chr\t150\t>1>3\tA\tG\t60\t.\tLV=0\tGT\t1\t0\t1\n"
        # nested in >1>3, which is reported itself: skipped
        + "ref#1#chr\t151\t>5>7\tA\tC\t60\t.\tLV=1;PS=>1>3\tGT\t1\t1\t1\n"
        # 2-bp deletion in s2, s3 has no call
        + "ref#1#chr\t250\t>7>9\tACG\tA\t60\t.\tLV=0\tGT\t0\t1\t.\n"
        # 80-bp insertion is an SV: counted per genome, not in pi
        + f"ref#1#chr\t260\t>9>11\tA\tA{'T' * 80}\t60\t.\tLV=0\tGT\t0\t0\t1\n", encoding="utf-8"
    )
    tracks = _vcf_tracks(vcf, "chr", [0, 200], [200, 400])
    assert tracks["samples"] == ["s1", "s2", "s3"]
    assert tracks["reference_haplotype"] == "ref"
    assert (tracks["snp_count"], tracks["indel_count"], tracks["sv_sites"]) == (1, 1, 1)
    assert tracks["counts"].tolist() == [[1, 0], [0, 1], [1, 1]]
    assert tracks["no_call"].tolist() == [[0, 0], [0, 0], [0, 0]]  # s3 is called at the SV
    assert tracks["snp_pi"][0] == pytest.approx((4 / 3) * 0.5 / 200)
    # deletion: alleles ref,0,1 -> p = 1/3, pi = 3/2 * (1 - 5/9)
    assert tracks["indel_pi"][1] == pytest.approx(1.5 * (1 - 5 / 9) / 200)


def test_vcf_no_call_window_is_marked_only_when_every_record_misses_the_genome(tmp_path):
    vcf = tmp_path / "v.vcf"
    vcf.write_text(_HEADER + "ref#1#chr\t10\t.\tA\tG\t60\t.\t.\tGT\t1\t.\t.\n", encoding="utf-8")
    tracks = _vcf_tracks(vcf, "chr", [0], [100])
    assert tracks["no_call"].tolist() == [[0], [1], [1]]


def test_skeleton_collapses_small_bubbles_and_keeps_presence_classes(tmp_path):
    gfa = tmp_path / "g.gfa"
    gfa.write_text(
        "H\tVN:Z:1.0\n"
        "S\tA\t*\tLN:i:1000\nS\tx\t*\tLN:i:1\nS\ty\t*\tLN:i:1\nS\tB\t*\tLN:i:1000\n"
        "S\tP\t*\tLN:i:800\nS\tC\t*\tLN:i:1000\n"
        "P\ts1#1#c\tA+,x+,B+,P+,C+\t*\n"
        "P\ts2#1#c\tA+,y+,B+,C+\t*\n"
        "W\ts3\t1\tc\t0\t3001\t>A>y>B>C\n", encoding="utf-8"
    )
    skeleton = pangenome_skeleton(gfa, max_blocks=3, reference="s2")
    # The SNP bubble (x/y) goes first; then C, A and B are one unbranched core
    # run around the circle, and the s1-only insertion P stays its own block.
    core, private = skeleton["blocks"]
    assert (core["length"], core["class"], core["genomes"]) == (3000, "core", 3)
    assert (private["length"], private["class"], private["genomes"]) == (800, "private", 1)
    assert (core["ref_start"], core["ref_end"]) == (0, 3001)
    assert private["ref_start"] is None
    assert skeleton["genomes"] == 3 and skeleton["raw_nodes"] == 6
    links = {(e["source"], e["target"]): e["genomes"] for e in skeleton["links"]}
    # s1 detours through P; s2/s3 close the core block on itself
    assert links == {("b1+", "b1+"): 2, ("b1+", "b2+"): 1, ("b2+", "b1+"): 1}


def test_real_citrus_fixture_draws_all_five_layers_and_restores(tmp_path):
    files = _write_fixture(tmp_path)
    plot = plot_pan_circular(
        files["reference.gb"],
        variants=files["variants.vcf"],
        pangenome_gfa=files["graph.gfa"],
        bins=30,
        graph_max_blocks=12,
        dpi=60,
    )
    metrics = plot.metrics
    for flag in (
        "vcf_input",
        "ogdraw_outer",
        "variant_heatmap",
        "pi_from_genotypes",
        "reference_haplotype_included",
        "ov_graph_center",
    ):
        assert flag in plot.flags
    assert sorted(metrics["heatmap"]["samples"]) == sorted(
        ["NC_057142.1", "PQ636878.1", "NC_037463.1", "OZ535733.1"]
    )
    assert metrics["variant_unit"] == "sites"
    assert metrics["snp_count"] > 0 and metrics["indel_count"] > 0
    assert {row["gene"] for row in metrics["loci"]} >= {"atp1", "ccmFC"}
    graph = metrics["graph"]
    assert graph["genomes"] == 5 and 1 <= len(graph["blocks"]) <= 12
    assert set(metrics["graph_layout"]) == {b["name"] for b in graph["blocks"]}
    assert metrics["center_image"] == "ov_graph"
    # the centre is an overview no larger than the full skeleton kept for the sidecar
    assert len(metrics["graph_overview"]["blocks"]) <= len(graph["blocks"])
    # graph coverage separates unaligned windows from missing alleles, per genome
    assert "coverage_from_graph" in plot.flags
    summary = {row["sample"]: row for row in metrics["heatmap"]["summary"]}
    assert set(summary) == set(metrics["heatmap"]["samples"])
    for row in summary.values():
        assert row["no_call_records"] == (
            row["no_call_records_aligned"] + row["no_call_records_unaligned"]
        )
        assert sum(row[k] for k in row if k.startswith("windows_")) == metrics["variant_bins"]
    assert "unaligned" in metrics["caption"] and "Lavender" in metrics["caption"]

    out = tmp_path / "pan.png"
    published = ov.write(plot, out)
    assert out.exists() and published.artifacts
    work = out.with_suffix("")
    assert (work / "pan_variant_matrix.tsv").read_text(encoding="utf-8").startswith("accession\t0-1000")
    assert (work / "pan_graph_blocks.tsv").exists()
    assert (work / "pan_graph_full.pdf").exists() and (work / "pan_missing_summary.tsv").exists()

    # A serialized Result redraws through the writer without the input files.
    for path in files.values():
        path.unlink()
    restored = OrganelleResult.model_validate(plot.model_dump(mode="json"))
    svg = tmp_path / "restored.svg"
    ov.write(restored, svg)
    text = svg.read_text(encoding="utf-8")
    assert "Pangenome graph" in text and "Nucleotide diversity" in text


def test_figure_is_drawn_at_journal_size_and_composes_as_a_panel(tmp_path):
    from organelleverse.visualization.compose import compose_figure
    from organelleverse.visualization.style import FONT_PT, MAX_HEIGHT_MM

    files = _write_fixture(tmp_path)
    plot = plot_pan_circular(
        files["reference.gb"],
        variants=files["variants.vcf"],
        pangenome_gfa=files["graph.gfa"],
        bins=30,
        graph_max_blocks=12,
        dpi=60,
    )
    width_in, height_in = plot.metrics["figsize"]
    assert width_in * 25.4 == pytest.approx(183.0, abs=0.1)
    assert height_in * 25.4 <= MAX_HEIGHT_MM
    svg = tmp_path / "pan.svg"
    ov.write(plot, svg)
    sizes = [
        float(s)
        for s in re.findall(r"font(?:-size)?:\s*(?:[a-z]+\s+)*([0-9.]+)px", svg.read_text(encoding="utf-8"))
    ]
    # printed sizes: nothing below the 5 pt floor, no figure title above 8 pt
    assert sizes and min(sizes) >= FONT_PT["min"] and max(sizes) <= FONT_PT["title"]

    assert plot.panel is not None and plot.panel.aspect == 1.0
    figure = compose_figure([plot, plot], titles=["a", "b"])
    out = tmp_path / "composite.png"
    ov.write(figure, out)
    assert out.exists()


def test_bandage_renderer_is_an_explicit_choice_and_fails_without_bandage(tmp_path):
    files = _write_fixture(tmp_path)
    with pytest.raises(OrganelleDependencyError):
        plot_pan_circular(
            files["reference.gb"],
            pangenome_gfa=files["graph.gfa"],
            graph_renderer="bandage",
            bandage_executable=tmp_path / "no-such-Bandage",
        )


def test_variant_table_input_is_still_accepted_and_labelled_as_calls(tmp_path):
    files = _write_fixture(tmp_path)
    table = tmp_path / "calls.tsv"
    table.write_text("accession\tposition\ttype\na\t100\tSNP\na\t5000\tINDEL\nb\t100\tSNP\n", encoding="utf-8")
    plot = plot_pan_circular(files["reference.gb"], variants=table, bins=10)
    assert plot.metrics["variant_unit"] == "calls"
    assert plot.metrics["diversity_kind"] == "calls_per_genome"
    assert (plot.metrics["snp_count"], plot.metrics["indel_count"]) == (2, 1)
    assert plot.metrics["center_image"] == "none"
    assert "center_summary" in plot.flags


def _plastome(tmp_path):
    rng = random.Random(5)

    def seq(n):
        return "".join(rng.choice("ACGT") for _ in range(n))

    lsc, ir, ssc = seq(40_000), seq(9_000), seq(7_000)
    full = lsc + ir + ssc + str(Seq(ir).reverse_complement())
    record = SeqRecord(
        Seq(full),
        id="cp1",
        name="cp1",
        description="Testus plastidus chloroplast.",
        annotations={"molecule_type": "DNA"},
    )
    record.features.append(
        SeqFeature(
            SimpleLocation(0, len(full), strand=1),
            type="source",
            qualifiers={"organelle": ["plastid:chloroplast"]},
        )
    )
    for start, strand, name in ((1_000, 1, "psbA"), (60_000, -1, "rbcL")):
        record.features.append(
            SeqFeature(
                SimpleLocation(start, start + 900, strand=strand),
                type="gene",
                qualifiers={"gene": [name]},
            )
        )
    path = tmp_path / "cp.gb"
    SeqIO.write([record], path, "genbank")
    return path


def test_plastid_reference_gets_the_ir_band_and_gc_overlay(tmp_path):
    path = _plastome(tmp_path)
    plot = plot_pan_circular(path, gc=True, dpi=50)
    assert set(plot.metrics["regions"]) == {"LSC", "IRb", "SSC", "IRa"}
    assert "ir_band" in plot.flags and "gc_overlay" in plot.flags
    assert len(plot.metrics["gc"]) == plot.metrics["variant_bins"]
    assert plot.metrics["title"] == "Testus plastidus plastid pan-genome"
    out = tmp_path / "cp.svg"
    ov.write(plot, out)
    assert "IRa" in out.read_text(encoding="utf-8")
    assert "ir_band" not in plot_pan_circular(path, show_ir=False).flags
