"""NUMT/NUPT chromosome figures: tracks and ideogram."""

from __future__ import annotations

import itertools
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from organelleverse.visualization import suite_plots as sp
from organelleverse.visualization import transfer_plots as tp

LENGTHS = {"10": 23_000_000, "2": 36_000_000, "1": 43_000_000}


def _candidate(seqid: str, start: int, length: int = 300, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "nuclear_seqid": seqid,
        "nuclear_start": start,
        "nuclear_end": start + length - 1,
        "length": length,
        "identity": 92.5,
        "organelle_seqid": "mt",
        "organelle_gene": "",
        "nuclear_gene": "",
        "evidence_level": 0,
        "depth_supported": None,
        "longread_linked": None,
    }
    row.update(extra)
    return row


def _many(n: int, seed: int = 1, **extra: object) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        seqid = rng.choice(list(LENGTHS))
        rows.append(
            _candidate(
                seqid,
                rng.randrange(1, LENGTHS[seqid] - 60_000),
                rng.choice([150, 800, 4_000, 25_000]),
                identity=rng.uniform(80, 100),
                **extra,
            )
        )
    return rows


def _legend_texts(spec: dict[str, object]) -> list[str]:
    fig = tp._Painter(spec).draw()
    texts = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
    texts += [legend.get_title().get_text() for legend in fig.legends]
    plt.close(fig)
    return texts


def _tracks_spec(plot: object) -> dict[str, object]:
    # the compute-only plot keeps its prepared spec in the renderer closure
    renderer = plot.renderer  # type: ignore[attr-defined]
    cells = {
        name: cell.cell_contents
        for name, cell in zip(renderer.__code__.co_freevars, renderer.__closure__, strict=True)
    }
    return cells["spec"]


def test_natural_chromosome_order() -> None:
    names = ["chr10", "chr2", "Chr1", "chrX", "scaffold_10", "scaffold_2", "1B", "1A", "3"]
    assert sorted(names, key=tp.chromosome_sort_key) == [
        "Chr1",
        "1A",
        "1B",
        "chr2",
        "3",
        "chr10",
        "chrX",
        "scaffold_2",
        "scaffold_10",
    ]


def test_evidence_categories_follow_the_validation_layers() -> None:
    assert tp.evidence_category({"evidence_level": 0}) == "candidate"
    assert tp.evidence_category({"evidence_level": 1}) == "one_layer"
    assert tp.evidence_category({"evidence_level": 2}) == "both"
    assert tp.evidence_category({}) == "unspecified"
    flags = {"depth_supported": False, "longread_linked": False, "evidence_level": 0}
    assert tp.evidence_category(flags) == "not_validated"
    assert tp.evidence_category({**flags, "longread_linked": True}) == "longread"
    assert tp.evidence_category({**flags, "depth_supported": True}) == "depth"
    assert tp.evidence_category({"depth_supported": True, "longread_linked": True}) == "both"


def test_tracks_draw_only_given_chromosomes_and_count_unplaced(tmp_path: Path) -> None:
    rows = [_candidate("1", 1_000), _candidate("10", 5_000), _candidate("2", 9_000)]
    rows += [_candidate("Syng_TIGR_013", 100), _candidate("AC160949", 200)] * 2
    plot = sp.plot_transfer_tracks(rows, chromosome_lengths=LENGTHS)
    metrics = dict(plot.metrics)
    assert list(metrics["chromosomes"]) == ["1", "2", "10"]
    assert metrics["drawn_candidates"] == 3
    assert metrics["unplaced_candidates"] == 4
    assert dict(metrics["unplaced_seqids"]) == {"AC160949": 2, "Syng_TIGR_013": 2}
    assert "4 on 2 unplaced sequences not drawn" in _tracks_spec(plot)["note"]
    out = plot.save(tmp_path / "tracks.png")
    assert out.stat().st_size > 10_000


def test_tracks_read_lengths_from_fai_and_drop_small_contigs(tmp_path: Path) -> None:
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">unused\nACGT\n")
    Path(f"{fasta}.fai").write_text(
        "1\t43000000\t0\t60\t61\n2\t36000000\t0\t60\t61\nctg9\t50000\t0\t60\t61\n"
    )
    plot = sp.plot_transfer_tracks([_candidate("1", 10), _candidate("ctg9", 10)], fasta)
    assert list(plot.metrics["chromosomes"]) == ["1", "2"]
    assert plot.metrics["unplaced_candidates"] == 1


def test_tracks_legend_names_evidence_instead_of_levels() -> None:
    rows = _many(60)
    for index, row in enumerate(rows):
        row["depth_supported"] = index % 3 == 0
        row["longread_linked"] = index % 2 == 0
    texts = _legend_texts(_tracks_spec(sp.plot_transfer_tracks(rows, chromosome_lengths=LENGTHS)))
    assert "Evidence" in texts
    assert any(text.startswith("Read depth + long read") for text in texts)
    assert any(text.startswith("Long-read linked") for text in texts)
    assert not any("level" in text for text in texts)
    assert "Fragment length" in texts


def test_first_layer_only_is_labelled_as_alignment_candidates() -> None:
    plot = sp.plot_transfer_tracks(_many(30), chromosome_lengths=LENGTHS, transfer_label="NUMT")
    texts = _legend_texts(_tracks_spec(plot))
    assert any("candidate (alignment only)" in text for text in texts)


def test_tracks_axis_is_in_mb_and_rows_follow_natural_order() -> None:
    spec = _tracks_spec(sp.plot_transfer_tracks(_many(20), chromosome_lengths=LENGTHS))
    painter = tp._Painter(spec)
    fig = painter.draw()
    assert painter.ax.get_xlabel() == "Position (Mb)"
    assert [t.get_text() for t in painter.ax.get_yticklabels()] == ["Chr1", "Chr2", "Chr10"]
    plt.close(fig)


def test_combined_tracks_with_identity_labels_and_centromeres_render_fast(tmp_path: Path) -> None:
    numt = _many(1_500, seed=2)
    numt[0]["organelle_gene"] = "nad5"
    nupt = _many(900, seed=3, organelle_seqid="plastid")
    plot = sp.plot_transfer_tracks(
        numt,
        chromosome_lengths=LENGTHS,
        plastid_candidates=nupt,
        color_by="identity",
        label_top=5,
        centromeres={"1": (16_000_000, 17_000_000)},
    )
    assert list(plot.metrics["transfers"]) == ["NUMT", "NUPT"]
    started = time.perf_counter()
    out = plot.save(tmp_path / "combined.png")
    elapsed = time.perf_counter() - started
    assert out.stat().st_size > 50_000
    # batched drawing: thousands of insertions stay well under the old ~40 s
    assert elapsed < 15


def test_tracks_keep_the_private_path_renderer(tmp_path: Path) -> None:
    out = sp._render_transfer_tracks(_many(10), tmp_path / "legacy.png", chromosome_lengths=LENGTHS)
    assert out.exists()


def test_tracks_reject_empty_input() -> None:
    with pytest.raises(ValueError):
        sp.plot_transfer_tracks([])


def test_ideogram_defaults_without_annotation_draw_no_gene_markers(tmp_path: Path) -> None:
    result = {"candidates": _many(40), "organelle": "mitochondrion"}
    prepared = sp.nuclear_transfer_ideogram(LENGTHS, result)
    metrics = sp._result_metrics(prepared)
    assert metrics["panel_label"] is None
    assert metrics["nuclear_annotated"] is False
    assert metrics["gene_hit_count"] == 0
    assert [row["name"] for row in metrics["chromosomes"]] == ["1", "2", "10"]
    out = sp.write_nuclear_transfer_ideogram(prepared, tmp_path / "ideogram.png")
    assert out.exists()


def test_ideogram_axis_is_mb_and_gene_legend_lists_present_classes() -> None:
    rows = _many(40)
    rows[0]["nuclear_gene"] = "Os01g0100100"
    prepared = sp.nuclear_transfer_ideogram(LENGTHS, rows)
    metrics = sp._result_metrics(prepared)
    assert metrics["nuclear_annotated"] is True
    spec = {
        "orientation": "vertical",
        "chromosomes": tp.normalize_chromosomes(
            {"name": k, "length_bp": v} for k, v in LENGTHS.items()
        ),
        "sets": tp.build_transfer_sets(rows, primary_label="NUMT"),
        "gene_classes": True,
    }
    painter = tp._Painter(spec)
    fig = painter.draw()
    assert painter.ax.get_ylabel() == "Position (Mb)"
    texts = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
    plt.close(fig)
    assert "Within nuclear gene" in texts and "Intergenic" in texts
    assert not any("new-gene" in text or "un-gene" in text for text in texts)


def test_ideogram_combined_numt_left_nupt_right(tmp_path: Path) -> None:
    prepared = sp.nuclear_transfer_ideogram(
        LENGTHS,
        _many(50),
        plastid_fragments=_many(30, seed=9),
        label_top=3,
        centromeres={"2": (15_000_000, 16_000_000)},
        color_by="evidence_level",
    )
    metrics = sp._result_metrics(prepared)
    assert metrics["transfer_label"] == "NUMT + NUPT"
    assert len(metrics["plastid_fragments"]) == 30
    assert metrics["chromosomes"][1]["centromere_start_mb"] == pytest.approx(15.0)
    assert tp.TRANSFER_SIDES["NUMT"] < 0 < tp.TRANSFER_SIDES["NUPT"]
    assert sp.save_plot(prepared, tmp_path / "combined.svg").exists()


def test_ideogram_accepts_legacy_fragment_and_gene_hit_rows(tmp_path: Path) -> None:
    prepared = sp.nuclear_transfer_ideogram(
        [{"chromosome": "chr2", "length": 30}, {"chromosome": "chr1", "length": 40}],
        [{"chromosome": "chr1", "start": 5.0, "end": 5.2}, {"chromosome": "chr2", "start": 7.1}],
        gene_hits=[{"chromosome": "chr1", "position": 5.1, "class": "NUMT->new-gene"}],
        panel_label="C",
    )
    metrics = sp._result_metrics(prepared)
    assert [row["name"] for row in metrics["chromosomes"]] == ["chr1", "chr2"]
    assert metrics["fragments"][0]["start"] == 5_000_000
    assert metrics["nuclear_annotated"] is True
    out = sp.write_nuclear_transfer_ideogram(prepared, tmp_path / "legacy.png", width=5, height=3)
    assert out.exists()


def test_spread_separates_labels_without_overlap() -> None:
    placed = tp._spread([10.0, 10.0, 10.1, 30.0], [1.0, 1.0, 1.0, 1.0])
    gaps = [b - a for a, b in itertools.pairwise(placed)]
    assert all(gap >= 1.0 - 1e-9 for gap in gaps)
    assert placed[-1] == pytest.approx(30.0)


# -- per-sample summaries ------------------------------------------------------

_GFF = """##gff-version 3
chr1\tt\tgene\t10000\t20000\t.\t+\t.\tID=g1
chr1\tt\tmRNA\t10000\t20000\t.\t+\t.\tID=g1.1;Parent=g1
chr1\tt\texon\t10000\t12000\t.\t+\t.\tParent=g1.1
chr1\tt\texon\t18000\t20000\t.\t+\t.\tParent=g1.1
chr1\tt\tgene\t50000\t60000\t.\t-\t.\tID=g2
chr1\tt\texon\t50000\t60000\t.\t-\t.\tParent=g2
"""

_RM_OUT = """   SW   perc perc perc  query      position in query           matching       repeat
score   div. del. ins.  sequence    begin     end    (left)    repeat         class/family

  239   29.4  1.9  0.6  chr1        10500    10800 (100)  +  Gypsy1         LTR/Gypsy   1  300   (0)  1
  120   10.0  0.0  0.0  chr1        14000    14100 (100)  +  hAT1           DNA/hAT-Ac  1  100   (0)  2
"""

_EXPECTED_REGIONS = {
    (10500, 10800): "Exon",
    (14000, 14300): "Intron",
    (9000, 9300): "Promoter",  # upstream of a + gene
    (20500, 20800): "Terminator",
    (60500, 60800): "Promoter",  # upstream of a - gene is to its right
    (49000, 49300): "Terminator",
    (30000, 30300): "Intergenic",
    (11900, 12200): "Intron",  # 101 bp exon vs 199 bp intron: majority wins
}


def _region_candidates() -> list[dict[str, object]]:
    return [_candidate("chr1", start, end - start + 1) for start, end in _EXPECTED_REGIONS]


def test_length_bins_follow_the_published_ranges() -> None:
    assert tp.length_bin_labels() == [
        "200–399",  # noqa: RUF001
        "400–599",  # noqa: RUF001
        "600–799",  # noqa: RUF001
        "800–999",  # noqa: RUF001
        "1,000–2,999",  # noqa: RUF001
        "≥3,000",
    ]
    rows = [_candidate("1", 1, n) for n in (150, 200, 399, 400, 999, 1_000, 2_999, 3_000, 56_000)]
    result = tp.transfer_length_bins(rows)
    assert result["below_min"] == 1
    assert list(result["counts"].values()) == [2, 1, 0, 1, 2, 2]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("LTR/Gypsy", "LTR"),
        ("Gypsy_LTR_retrotransposon", "LTR"),
        ("DNA/hAT-Ac", "DNA"),
        ("Mutator_TIR_transposon", "DNA"),
        ("MITE/DTM", "MITE"),
        ("RC/Helitron", "Helitron"),
        ("helitron", "Helitron"),
        ("LINE/L1", "LINE"),
        ("SINE/tRNA", "SINE"),
        ("rRNA", "rRNA"),
        ("Simple_repeat", "Simple_repeat"),
        ("Low_complexity", "Simple_repeat"),
        ("Unknown", "other"),
        ("", "nonTE"),
    ],
)
def test_te_classes_map_repeatmasker_and_edta_names(text: str, expected: str) -> None:
    assert tp.te_category(text) == expected


def test_regions_are_strand_aware_and_take_the_majority(tmp_path: Path) -> None:
    gff = tmp_path / "genes.gff3"
    gff.write_text(_GFF)
    summary = tp.summarize_transfer_regions(_region_candidates(), gff)
    got = {(row["start"], row["end"]): row["region"] for row in summary["candidates"]}
    assert got == _EXPECTED_REGIONS
    assert summary["te_annotated"] is False
    assert {row["te_class"] for row in summary["candidates"]} == {"not_annotated"}
    assert sum(summary["region_fraction"].values()) == pytest.approx(1.0)


def test_regions_fall_back_to_transcripts_without_gene_lines(tmp_path: Path) -> None:
    gff = tmp_path / "mrna_only.gff3"
    gff.write_text("\n".join(line for line in _GFF.splitlines() if "\tgene\t" not in line))
    gff.write_text(gff.read_text().replace("\texon\t50000\t60000", "\tmRNA\t50000\t60000"))
    summary = tp.summarize_transfer_regions(_region_candidates(), gff)
    got = {(row["start"], row["end"]): row["region"] for row in summary["candidates"]}
    assert got[(14000, 14300)] == "Intron"
    assert got[(60500, 60800)] == "Promoter"


def test_te_class_from_repeatmasker_out_and_te_gff(tmp_path: Path) -> None:
    gff = tmp_path / "genes.gff3"
    gff.write_text(_GFF)
    rm_out = tmp_path / "genome.fa.out"
    rm_out.write_text(_RM_OUT)
    by_out = tp.summarize_transfer_regions(_region_candidates(), gff, rm_out)
    classes = {(r["start"], r["end"]): r["te_class"] for r in by_out["candidates"]}
    assert classes[(10500, 10800)] == "LTR"
    assert classes[(14000, 14300)] == "nonTE"  # TEs cover 101/301 bp < te_min_fraction
    loose = tp.summarize_transfer_regions(_region_candidates(), gff, rm_out, te_min_fraction=0.2)
    assert {(r["start"], r["end"]): r["te_class"] for r in loose["candidates"]}[
        (14000, 14300)
    ] == "DNA"
    te_gff = tmp_path / "te.gff3"
    te_gff.write_text(
        "chr1\tEDTA\thelitron\t30000\t30300\t.\t+\t.\tID=te1;Classification=DNA/Helitron\n"
    )
    by_gff = tp.summarize_transfer_regions(_region_candidates(), gff, te_gff)
    assert {(r["start"], r["end"]): r["te_class"] for r in by_gff["candidates"]}[
        (30000, 30300)
    ] == "Helitron"


def test_length_profile_has_one_legend_entry_per_group(tmp_path: Path) -> None:
    samples = {f"hap{i}": _many(40, seed=i) for i in range(1, 5)}
    samples["R108"] = _many(30, seed=9)
    groups = {**{f"hap{i}": "M. sativa ZM4" for i in range(1, 5)}, "R108": "M. truncatula"}
    plot = tp.plot_transfer_length_profile(samples, groups=groups)
    assert set(plot.metrics["samples"]) == set(samples)
    out = plot.save(tmp_path / "profile.png")
    assert out.stat().st_size > 10_000
    fig_spec = {
        name: cell.cell_contents
        for name, cell in zip(
            plot.renderer.__code__.co_freevars, plot.renderer.__closure__, strict=True
        )
    }["spec"]
    assert fig_spec["group_names"] == ["M. sativa ZM4", "M. truncatula"]


def test_region_te_bars_render_for_several_haplotypes(tmp_path: Path) -> None:
    gff = tmp_path / "genes.gff3"
    gff.write_text(_GFF)
    rm_out = tmp_path / "genome.fa.out"
    rm_out.write_text(_RM_OUT)
    summaries = {
        name: tp.summarize_transfer_regions(_region_candidates(), gff, rm_out)
        for name in ("hap1", "hap2", "hap3")
    }
    plot = tp.plot_transfer_region_te(summaries, transfer_label="NUPT", panel_label="F")
    percent = plot.metrics["percent"]["hap1"]
    assert sum(sum(v.values()) for v in percent.values()) == pytest.approx(100.0)
    assert plot.save(tmp_path / "region_te.png").exists()


# -- printed size, panels and composites ---------------------------------------


def test_figures_are_drawn_at_journal_width_with_printed_point_sizes() -> None:
    from organelleverse.visualization.style import FONT_PT, publication_style, width_inches

    spec = _tracks_spec(sp.plot_transfer_tracks(_many(30), chromosome_lengths=LENGTHS))
    with publication_style():
        fig = tp._Painter(spec).draw()
        try:
            width, height = fig.get_size_inches()
            assert width == pytest.approx(width_inches("double"))
            assert height * 25.4 <= 170 + 1e-6
            sizes = {
                text.get_fontsize()
                for legend in fig.legends
                for text in [*legend.get_texts(), legend.get_title()]
                if text.get_text()
            }
            assert sizes == {FONT_PT["body"]}
            tick_sizes = {t.get_fontsize() for t in fig.axes[0].get_xticklabels()}
            assert tick_sizes == {FONT_PT["tick"]}
        finally:
            plt.close(fig)


def test_ideogram_numbers_the_longest_insertions_and_lists_them() -> None:
    rows = _many(40)
    rows[0]["length"] = 56_000
    rows[0]["organelle_gene"] = "nad5"
    prepared = sp.nuclear_transfer_ideogram(LENGTHS, rows, label_top=2)
    plot = sp.plot_nuclear_transfer_ideogram(prepared)
    spec = {
        name: cell.cell_contents
        for name, cell in zip(
            plot.renderer.__code__.co_freevars, plot.renderer.__closure__, strict=True
        )
    }["spec"]
    painter = tp._Painter(spec)
    fig = painter.draw()
    texts = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
    plt.close(fig)
    assert painter.callouts[0][0] == 1
    assert any(text.startswith("56.0 kb · nad5 · NUMT") for text in texts)


def test_transfer_plots_compose_into_a_lettered_figure(tmp_path: Path) -> None:
    from organelleverse.visualization import compose_figure

    tracks = sp.plot_transfer_tracks(_many(60), chromosome_lengths=LENGTHS, title="Tracks")
    ideogram = sp.plot_nuclear_transfer_ideogram(
        sp.nuclear_transfer_ideogram(LENGTHS, _many(40), plastid_fragments=_many(30, seed=5))
    )
    profile = tp.plot_transfer_length_profile({"a": _many(40), "b": _many(30, seed=4)})
    gff = tmp_path / "genes.gff3"
    gff.write_text(_GFF)
    bars = tp.plot_transfer_region_te(
        {"a": tp.summarize_transfer_regions(_region_candidates(), gff)}
    )
    for plot in (tracks, ideogram, profile, bars):
        assert plot.panel is not None
    figure = compose_figure([ideogram, profile, tracks, bars], columns=2)
    out = figure.save(tmp_path / "composite.png")
    assert out.stat().st_size > 20_000


# -- honest TE panel, compact composite legend, per-sample markers -------------


def _region_summary(tmp_path: Path, te_text: str | None) -> dict[str, object]:
    gff = tmp_path / "genes.gff3"
    gff.write_text(_GFF)
    te = None
    if te_text is not None:
        te = tmp_path / "te.out"
        te.write_text(te_text)
    return tp.summarize_transfer_regions(_region_candidates(), gff, te)


def test_region_panel_never_draws_missing_te_annotation_as_no_te(tmp_path: Path) -> None:
    summary = _region_summary(tmp_path, None)
    plot = tp.plot_transfer_region_te({"a": summary, "b": summary}, groups={"a": "G", "b": "G"})
    assert plot.metrics["te_mode"] == "not_evaluated"
    assert "TE classes not evaluated" in plot.metrics["caption"]
    handles = plot.panel.draw(plt.figure().add_subplot(), False)
    assert [h.get_label() for h in handles] == ["G (n = 2)"]
    plt.close("all")


def test_region_panel_without_any_te_overlap_is_regions_only(tmp_path: Path) -> None:
    far_away = _RM_OUT.replace("10500    10800", "90500    90800").replace(
        "14000    14100", "95000    95100"
    )
    summary = _region_summary(tmp_path, far_away)
    plot = tp.plot_transfer_region_te({"a": summary})
    assert plot.metrics["te_mode"] == "no_te_overlap"
    assert "No insertion is at least 50% covered" in plot.metrics["caption"]


def test_region_panel_splits_by_te_class_only_with_real_te_hits(tmp_path: Path) -> None:
    plot = tp.plot_transfer_region_te({"a": _region_summary(tmp_path, _RM_OUT)})
    assert plot.metrics["te_mode"] == "te_classes"
    labels = [h.get_label() for h in plot.panel.draw(plt.figure().add_subplot(), False)]
    plt.close("all")
    assert labels == ["LTR", "nonTE"]


def test_composite_legend_is_short_and_density_goes_to_the_caption() -> None:
    plot = sp.plot_transfer_tracks(
        _many(40), chromosome_lengths=LENGTHS, plastid_candidates=_many(30, seed=3)
    )
    fig = plt.figure(figsize=(3.5, 3.0))
    labels = [h.get_label() for h in plot.panel.draw(fig.add_axes((0, 0, 1, 1)), False)]
    plt.close(fig)
    assert labels[:2] == ["NUMT", "NUPT"]
    assert not any("per 500 kb" in label or "Fragment" in label for label in labels)
    caption = plot.metrics["caption"]
    assert "insertions per 500 kb" in caption and "alignment only" in caption


def test_profile_gives_each_sample_its_marker_and_an_end_label() -> None:
    samples = {f"hap{i}": _many(30, seed=i) for i in range(1, 4)}
    plot = tp.plot_transfer_length_profile(samples, groups=dict.fromkeys(samples, "G"))
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_axes((0.15, 0.3, 0.8, 0.6))
    handles = plot.panel.draw(ax, True)
    markers = {line.get_marker() for line in ax.get_lines() if len(line.get_xdata()) > 1}
    names = {text.get_text() for text in ax.texts}
    plt.close(fig)
    assert handles == []
    assert len(markers) == 3
    assert {"hap1", "hap2", "hap3"} <= names
