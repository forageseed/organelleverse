"""Population / pangenome comparison figures: numbers, rendering and restore."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.core.frozen import thaw_json
from organelleverse.core.result import OrganelleResult
from organelleverse.population.population import FST_ESTIMATORS
from organelleverse.visualization import popgen_data as data
from organelleverse.visualization import popgen_plots

sys.path.insert(0, str(Path(__file__).parent))
import itertools

import popgen_fixture as toy

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture()
def files(tmp_path):
    return toy.write_all(tmp_path / "toy")


def _metrics(plot):
    return thaw_json(plot.metrics)


def _write_all_formats(plot, directory: Path, stem: str):
    for suffix in (".png", ".svg", ".pdf"):
        released = ov.write(plot, directory / f"{stem}{suffix}")
        kinds = [a.kind for a in released.artifacts]
        assert kinds[0] == "figure" and "figure_data" in kinds
        assert Path(released.artifacts[0].uri).stat().st_size > 1000
    assert (directory / f"{stem}.tsv").is_file()


def _restored_renders(plot, path: Path):
    """A JSON round-trip of the Result (no renderer) still renders through ov.write."""
    restored = OrganelleResult.model_validate_json(plot.model_dump_json())
    released = ov.write(restored, path)
    assert Path(released.artifacts[0].uri).is_file()


# ---------------------------------------------------------------- F_ST


def test_fst_windows_use_the_population_estimator_and_tile_the_genome(files):
    scan = data.fst_windows(files["toy.vcf"], toy.GROUPS, estimator="hudson", window_size=500, step=250)
    windows = scan["windows"]
    assert windows[0]["start"] == 1 and windows[-1]["end"] == toy.LENGTH
    assert all(b["start"] - a["start"] == 250 for a, b in itertools.pairwise(windows))
    snps = data.biallelic_snps(files["toy.vcf"], "REF")
    assert len(snps) == len(toy.FIXED) + len(toy.WILD_ONLY) + len(toy.SHARED)  # deletion record excluded
    first = windows[0]
    inside = [p for p in snps if first["start"] <= p <= first["end"]]
    site = FST_ESTIMATORS["hudson"]
    expected = sum(site(snps[p], toy.WILD, toy.CROP) for p in inside) / len(inside)
    assert first["fst"] == pytest.approx(round(expected, 4))
    assert first["fst"] > 0.9  # fixed differences
    # windows below min_snps are gaps, not zeros
    assert any(w["fst"] is None and w["n_snps"] < 3 for w in windows)


def test_reference_sample_counts_as_ref_allele(files):
    groups = {**toy.GROUPS, "REF": "wild"}
    without = data.fst_windows(files["toy.vcf"], groups, window_size=500, step=500)
    assert without["samples_missing_from_vcf"] == ["REF"]
    assert without["group_sizes"]["wild"] == 4
    with_ref = data.fst_windows(files["toy.vcf"], groups, window_size=500, step=500, reference_sample="REF")
    assert with_ref["group_sizes"]["wild"] == 5 and with_ref["samples_missing_from_vcf"] == []


def test_plot_fst_scan_renders_highlights_and_restores(files, tmp_path):
    plot = ov.visualization.plot_fst_scan(
        files["toy.vcf"],
        groups=toy.GROUPS,
        pair=["wild", "crop"],
        estimator="weir_cockerham",
        window_size=500,
        step=250,
        highlight=[{"start": 601, "end": 900, "label": "SV"}],
    )
    m = _metrics(plot)
    assert m["plot_kind"] == "plot_fst_scan" and m["estimator"] == "weir_cockerham"
    assert m["highlights"] == [{"start": 601, "end": 900, "label": "SV"}]
    assert plot.artifacts == ()  # compute-only
    _write_all_formats(plot, tmp_path, "fst")
    _restored_renders(plot, tmp_path / "restored" / "fst.png")
    auto = ov.visualization.fst_scan(files["toy.vcf"], groups=toy.GROUPS, window_size=500, step=250, highlight="auto")
    assert _metrics(auto)["highlights"][0]["start"] == 1


def test_fst_needs_two_named_groups_and_one_contig(files, tmp_path):
    three = {**toy.GROUPS, "C4": "other"}
    with pytest.raises(OrganelleParameterError):
        data.fst_windows(files["toy.vcf"], three)
    assert data.fst_windows(files["toy.vcf"], three, pair=["wild", "crop"])["group_sizes"] == {"wild": 4, "crop": 3}
    two = tmp_path / "two.vcf"
    two.write_text(files["toy.vcf"].read_text().replace("##contig=<ID=REF,length=2000>", "##contig=<ID=REF,length=2000>\n##contig=<ID=X,length=5>"))
    with pytest.raises(OrganelleInputError):
        data.fst_windows(two, toy.GROUPS)
    with pytest.raises(OrganelleParameterError):
        data.fst_windows(files["toy.vcf"], toy.GROUPS, estimator="nei")


# ---------------------------------------------------------------- π ratio


def test_pi_ratio_from_alignment_matches_vcf_pseudo_alignment(files):
    from organelleverse.diversity.diversity import _window_pi_s

    kwargs = dict(pair=["wild", "crop"], window_size=500, step=250, min_sites=100)
    aln = data.pi_ratio_windows(toy.GROUPS, alignment=files["toy.aln.fasta"], reference="REF", **kwargs)
    vcf = data.pi_ratio_windows(
        toy.GROUPS, vcf=files["toy.vcf"], reference_fasta=files["ref.fasta"], **kwargs
    )
    window = next(w for w in aln["windows"] if w["start"] == 1001)
    seqs = [toy.sample_sequence(s) for s in toy.WILD]
    expected, _, _ = _window_pi_s([[s[i] for s in seqs] for i in range(1000, 1500)])
    assert window["pi_a"] == pytest.approx(round(expected, 7))
    assert window["pi_b"] == 0 and window["ratio"] > 1
    # identical except where C2's deletion is masked in the VCF route
    for a, v in zip(aln["windows"], vcf["windows"], strict=False):
        if v["end"] < 1700 or v["start"] > 1703:
            assert (a["pi_a"], a["pi_b"]) == (v["pi_a"], v["pi_b"])
    assert vcf["masked_records"] == 1 and vcf["masked_bases"] == 4
    assert aln["pseudocount"] == pytest.approx(min(x for w in aln["windows"] for x in (w["pi_a"], w["pi_b"]) if x) / 2)


def test_pseudo_alignment_masks_non_snp_alt_and_missing(files, tmp_path):
    ref, seqs, _counts = data.pseudo_alignment_from_vcf(files["toy.vcf"], files["ref.fasta"], ["C2", "W1", "REF"], reference_sample="REF")
    assert seqs["C2"][1699:1703] == "NNNN"
    assert seqs["W1"][1699:1703] != "NNNN"
    assert seqs["REF"] == ref
    missing = tmp_path / "missing.vcf"
    missing.write_text(files["toy.vcf"].read_text().replace("\tGT\t0\t", "\tGT\t.\t", 1))
    _, seqs, _ = data.pseudo_alignment_from_vcf(missing, files["ref.fasta"], ["W1"])
    assert seqs["W1"][50] == "N"


def test_plot_pi_ratio_labels_peaks_with_genes(files, tmp_path):
    plot = ov.visualization.plot_pi_ratio(
        toy.GROUPS,
        alignment=files["toy.aln.fasta"],
        reference="REF",
        pair=["wild", "crop"],
        window_size=500,
        step=250,
        min_sites=100,
        genbank=files["ref.gb"],
        top_fraction=0.1,
        log_scale=True,
    )
    m = _metrics(plot)
    assert m["peaks"] and m["peaks"][0]["label"] == "geneB"
    assert "genbank" not in m  # no absolute paths in metrics
    _write_all_formats(plot, tmp_path, "pi")
    _restored_renders(plot, tmp_path / "restored" / "pi.svg")
    strict = data.pi_ratio_windows(toy.GROUPS, alignment=files["toy.aln.fasta"], pair=["wild", "crop"], window_size=500, step=250, min_sites=1000)
    assert all(w["ratio"] is None for w in strict["windows"])
    with pytest.raises(OrganelleParameterError):
        data.pi_ratio_windows(toy.GROUPS, alignment=files["toy.aln.fasta"], vcf=files["toy.vcf"])


def test_fst_pi_stack_shares_the_axis(files, tmp_path):
    fst = ov.visualization.plot_fst_scan(files["toy.vcf"], groups=toy.GROUPS, pair=["wild", "crop"], window_size=500, step=250)
    pi = ov.visualization.plot_pi_ratio(toy.GROUPS, alignment=files["toy.aln.fasta"], reference="REF", pair=["wild", "crop"], window_size=500, step=250, min_sites=100)
    fst_json = OrganelleResult.model_validate_json(fst.model_dump_json())
    stack = ov.visualization.plot_fst_pi_stack(fst_json, pi)
    _write_all_formats(stack, tmp_path, "stack")
    header = (tmp_path / "stack.tsv").read_text().splitlines()[0].split("\t")
    assert header == ["start", "end", "fst", "n_snps", "pi_a", "pi_b", "ratio"]
    _restored_renders(stack, tmp_path / "restored" / "stack.pdf")
    with pytest.raises(OrganelleParameterError):
        ov.visualization.plot_fst_pi_stack(pi, fst)
    shorter = ov.visualization.plot_fst_scan(windows=[{"start": 1, "end": 10, "fst": 0.1}], genome_length=10)
    with pytest.raises(OrganelleParameterError):
        ov.visualization.plot_fst_pi_stack(shorter, pi)


# ---------------------------------------------------------------- coverage and SVs


def test_reference_coverage_from_gfa_and_paf_agree(files):
    gfa = data.reference_coverage_from_gfa(files["toy.gfa"], "REF", samples=toy.SAMPLES)
    assert gfa["genome_length"] == toy.LENGTH and gfa["reference"] == "REF#1#1"
    assert gfa["coverage"]["W4"] == [(1, 600), (901, 2000)]
    assert gfa["coverage"]["W1"] == [(1, 2000)]
    paf = data.coverage_from_paf(files["toy.paf"])
    assert {s: paf["coverage"][s] for s in toy.SAMPLES} == gfa["coverage"]
    with pytest.raises(OrganelleInputError):
        data.reference_coverage_from_gfa(files["toy.gfa"], "REF", samples=["nobody"])


def test_presence_svs_find_the_polymorphic_deletion(files):
    cov = data.reference_coverage_from_gfa(files["toy.gfa"], "REF", samples=toy.SAMPLES)
    svs = data.presence_svs(cov["coverage"], cov["genome_length"], min_length=100)
    assert [(sv["start"], sv["end"]) for sv in svs] == [(601, 900)]
    sv = svs[0]
    assert {s for s, present in sv["presence"].items() if not present} == toy.LACK_N2
    assert data.presence_svs(cov["coverage"], cov["genome_length"], min_length=301) == []


def test_sv_schematic_rows_split_homologous_and_sv_blocks(files):
    rows = data.sv_schematic_rows(files["toy.gfa"], "REF", 1, 2000, ["W1", "W4", "C1"], min_sv_length=100)
    by = {r["sample"]: r for r in rows["rows"]}
    assert list(by) == ["REF", "W1", "W4", "C1"]  # reference row first
    kinds = lambda s: [(b["kind"], b["end"] - b["start"], b["sv"]) for b in by[s]["blocks"]]  # noqa: E731
    assert kinds("REF") == [("homologous", 600, None), ("sv", 300, "SV1"), ("homologous", 1100, None)]
    assert kinds("W4") == [("homologous", 1700, None)]
    assert kinds("C1") == [("homologous", 1100, None), ("sv", 200, "SV2"), ("homologous", 600, None)]
    svs = {sv["id"]: sv for sv in rows["svs"]}
    assert (svs["SV1"]["ref_start"], svs["SV1"]["ref_end"], svs["SV1"]["in_reference"]) == (601, 900, True)
    assert svs["SV2"]["in_reference"] is False and svs["SV2"]["carriers"] == ["C1"]
    # C1's first homologous node sits under the same node of the reference row
    assert by["C1"]["align"] == [0, 0]
    assert by["C1"]["edge_positions"]["start:n5"] == 1300 and by["REF"]["edge_positions"]["start:n5"] == 1400
    # 1-bp variable runs are bubbles, not SVs
    assert data.sv_schematic_rows(files["toy.gfa"], "REF", 1, 2000, ["C1"], min_sv_length=500)["svs"] == []


def test_plot_sv_schematic_renders_and_restores(files, tmp_path):
    plot = ov.visualization.plot_sv_schematic(
        files["toy.gfa"], reference="REF", start=1, end=2000, samples=["W1", "W4", "C1"],
        labels={"W1": "wild W1", "C1": "crop C1"},
    )
    m = _metrics(plot)
    assert [r["label"] for r in m["rows"]] == ["REF", "wild W1", "W4", "crop C1"]
    _write_all_formats(plot, tmp_path, "schematic")
    _restored_renders(plot, tmp_path / "restored" / "schematic.png")
    with pytest.raises(OrganelleParameterError):
        ov.visualization.plot_sv_schematic(files["toy.gfa"], reference="REF")


# ---------------------------------------------------------------- SV frequency


def _svs():
    return [
        {"id": "SVa", "start": 1, "end": 500, "length": 500,
         "presence": {**{s: True for s in toy.WILD}, **{s: False for s in toy.CROP}}},
        {"id": "SVb", "start": 600, "end": 900, "length": 301,
         "presence": {s: s not in {"W4", "C1"} for s in toy.SAMPLES}},
        {"id": "SVc", "start": 1000, "end": 1100, "length": 101, "presence": {s: True for s in toy.SAMPLES}},
    ]


def test_sv_frequency_fisher_bh_and_untestable_rows(tmp_path):
    from scipy import stats

    plot = ov.visualization.plot_sv_frequency(_svs(), groups=toy.GROUPS, group_order=["wild", "crop"])
    m = _metrics(plot)
    assert m["test"] == "exact" and m["test_description"] == "two-sided Fisher's exact test"
    assert m["unit"].startswith("every sample") and m["group_units"] == {"wild": 4, "crop": 4}
    rows = {r["id"]: r for r in m["svs"]}
    assert rows["SVa"]["p"] == pytest.approx(stats.fisher_exact([[4, 0], [0, 4]])[1])
    assert rows["SVb"]["p"] == pytest.approx(stats.fisher_exact([[3, 1], [3, 1]])[1])
    assert rows["SVa"]["counts"] == {
        "wild": {"present": 4, "absent": 0, "missing": 0},
        "crop": {"present": 0, "absent": 4, "missing": 0},
    }
    assert rows["SVa"]["denominator"] == {"wild": 4, "crop": 4}
    assert rows["SVc"]["p"] is None and rows["SVc"]["stars"] == "n.t."
    # stars on the figure are the recomputed BH q
    q = data.benjamini_hochberg([rows["SVa"]["p"], rows["SVb"]["p"], None])
    assert rows["SVa"]["q"] == pytest.approx(q[0]) and rows["SVb"]["q"] == pytest.approx(q[1])
    assert rows["SVa"]["stars"] == popgen_plots._stars(q[0])
    assert "Fisher" in m["caption"] and "Denominator" in m["caption"] and "BH" in m["caption"]
    _write_all_formats(plot, tmp_path, "freq")
    header = (tmp_path / "freq.tsv").read_text().splitlines()[0].split("\t")
    assert {"wild denominator", "wild missing", "p", "q", "stars"} <= set(header)
    _restored_renders(plot, tmp_path / "restored" / "freq.png")


def test_exact_presence_test_matches_fisher_and_brute_force():
    import itertools
    from math import comb

    from scipy import stats

    for table in ([[3, 1], [0, 4]], [[5, 2], [1, 6]], [[0, 3], [2, 2]], [[7, 0], [6, 1]]):
        (a, b), (c, d) = table
        assert data.exact_presence_test([a, c], [b, d]) == pytest.approx(stats.fisher_exact(table)[1])

    def brute(present, absent):
        sizes = [p + a for p, a in zip(present, absent, strict=True)]
        total, n = sum(present), sum(sizes)
        prob = lambda xs: math.prod(comb(s, x) for s, x in zip(sizes, xs, strict=True)) / comb(n, total)  # noqa: E731
        observed = prob(present)
        tables = [xs for xs in itertools.product(*(range(s + 1) for s in sizes)) if sum(xs) == total]
        return sum(prob(xs) for xs in tables if prob(xs) <= observed * (1 + 1e-7))

    for present, absent in (([4, 0, 0], [0, 2, 2]), ([3, 1, 2], [1, 3, 0]), ([5, 0, 1], [0, 3, 2])):
        assert data.exact_presence_test(present, absent) == pytest.approx(brute(present, absent))
    assert data.exact_presence_test([3, 2], [0, 0]) is None  # no variation
    assert data.exact_presence_test([3, 0], [1, 0]) is None  # one non-empty group


def test_sv_frequency_exact_for_three_groups_chi2_on_request_and_from_gfa(files):
    from scipy import stats

    three = {**toy.GROUPS, "C3": "third", "C4": "third"}
    m = _metrics(ov.visualization.plot_sv_frequency(_svs(), groups=three))
    assert m["test"] == "exact" and m["test_description"].startswith("Freeman")
    sva = next(r for r in m["svs"] if r["id"] == "SVa")
    assert sva["p"] == pytest.approx(data.exact_presence_test([4, 0, 0], [0, 2, 2]))
    chi = _metrics(ov.visualization.plot_sv_frequency(_svs(), groups=three, test="chi2"))
    sva = next(r for r in chi["svs"] if r["id"] == "SVa")
    chi2, p, _, expected = stats.chi2_contingency([[4, 0], [0, 2], [0, 2]], correction=False)
    assert sva["p"] == pytest.approx(p) and sva["statistic"] == pytest.approx(chi2)
    assert sva["min_expected"] == pytest.approx(expected.min()) and "SVa" in chi["low_expected_svs"]
    with pytest.raises(OrganelleParameterError):
        ov.visualization.plot_sv_frequency(_svs(), groups=three, test="fisher")
    # graph input: identical mitogenomes (W1-W3, and C1-C3) count once per group
    from_graph = _metrics(
        ov.visualization.plot_sv_frequency(groups=toy.GROUPS, gfa=files["toy.gfa"], reference="REF", min_length=100)
    )
    assert [(r["start"], r["end"]) for r in from_graph["svs"]] == [(601, 900)]
    assert from_graph["group_sizes"] == {"wild": 4, "crop": 4}
    assert from_graph["group_units"] == {"wild": 2, "crop": 2}
    assert from_graph["svs"][0]["counts"]["wild"] == {"present": 1, "absent": 1, "missing": 0}
    assert from_graph["svs"][0]["counts"]["crop"] == {"present": 1, "absent": 1, "missing": 0}
    every = _metrics(
        ov.visualization.plot_sv_frequency(
            groups=toy.GROUPS, gfa=files["toy.gfa"], reference="REF", min_length=100, collapse_identical=False
        )
    )
    assert every["svs"][0]["counts"]["wild"] == {"present": 3, "absent": 1, "missing": 0}


def test_identical_path_groups_ignore_origin_and_strand(tmp_path):
    gfa = tmp_path / "rot.gfa"
    gfa.write_text(
        "H\tVN:Z:1.0\nS\ta\tACGT\nS\tb\tGGCC\nS\tc\tTTAA\nS\td\tCAGT\n"
        "L\ta\t+\tb\t+\t0M\nL\tb\t+\tc\t+\t0M\nL\tc\t+\ta\t+\t0M\nL\tb\t+\td\t+\t0M\n"
        "P\tA#1#1\ta+,b+,c+\t*\n"
        "P\tB#1#1\tb+,c+,a+\t*\n"  # rotated
        "P\tC#1#1\tc-,b-,a-\t*\n"  # reverse strand
        "P\tD#1#1\ta+,b+,d+\t*\n"  # different
    )
    units = data.identical_path_groups(gfa)
    assert units["A"] == units["B"] == units["C"] != units["D"]


def test_partial_coverage_is_missing_and_units_take_the_majority():
    # s2 carries 300 of the 800 bp SV (merged across its 300-bp island): 0.375 -> not called
    coverage = {"s1": [(1, 1000)], "s2": [(1, 100), (401, 700), (901, 1000)], "s3": [(1, 1000)]}
    (sv,) = data.presence_svs(coverage, 1000, min_length=50, merge_gap=400)
    assert (sv["start"], sv["end"]) == (101, 900)
    assert sv["fraction"]["s2"] == pytest.approx(0.375)
    assert sv["presence"] == {"s1": True, "s2": None, "s3": True} and sv["missing"] == 1
    table = data.sv_presence_table([sv], {"s1": "g", "s2": "g", "s3": "h"}, units={"s1": "u", "s2": "u"})
    assert table[0]["counts"]["g"] == {"present": 1, "absent": 0, "missing": 0}  # majority of called
    tie = {"id": "x", "presence": {"a": True, "b": False}}
    assert data.sv_presence_table([tie], {"a": "g", "b": "g"}, units={"a": "u", "b": "u"})[0]["counts"]["g"] == {
        "present": 0, "absent": 0, "missing": 1,
    }
    with pytest.raises(OrganelleParameterError):
        data.presence_svs(coverage, 1000, present_min=0.2, absent_max=0.5)


def test_benjamini_hochberg_known_values():
    q = data.benjamini_hochberg([0.01, 0.04, 0.03, None, 0.5])
    assert q[3] is None
    assert [q[0], q[1], q[2], q[4]] == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])
    assert math.isclose(q[4], 0.5)


# ---------------------------------------------------------------- presence tracks


def test_presence_tracks_order_annotations_genes_and_svs(files, tmp_path):
    source = {s: ("Pt" if s == "W2" else "Mt") for s in toy.SAMPLES}
    plot = ov.visualization.plot_presence_tracks(
        gfa=files["toy.gfa"],
        reference="REF",
        start=1,
        end=2000,
        groups=toy.GROUPS,
        group_order=["crop", "wild"],
        annotations={"DNA source": source},
        genbank=files["ref.gb"],
        min_sv_length=100,
    )
    m = _metrics(plot)
    order = [r["sample"] for r in m["rows"]]
    assert order[:4] and {*order[:4]} == set(toy.CROP) and order[0] == "C4"  # carriers of the SV first
    assert m["palette"]["DNA source"] == {"Mt": popgen_plots.SOURCE_COLORS["Mt"], "Pt": popgen_plots.SOURCE_COLORS["Pt"]}
    assert [(sv["start"], sv["end"]) for sv in m["svs"]] == [(601, 900)]
    assert [g["name"] for g in m["genes"]] == ["geneA", "geneB", "trnX"]
    assert m["genes"][1]["strand"] == -1
    _write_all_formats(plot, tmp_path, "tracks")
    _restored_renders(plot, tmp_path / "restored" / "tracks.svg")
    paf = ov.visualization.plot_presence_tracks(paf=files["toy.paf"], start=500, end=1000, groups=toy.GROUPS, svs=[{"id": "x", "start": 601, "end": 900}])
    assert _metrics(paf)["rows"][0]["blocks"]
    with pytest.raises(OrganelleParameterError):
        ov.visualization.plot_presence_tracks(gfa=files["toy.gfa"], reference="REF", start=10, end=5000, groups=toy.GROUPS)


def test_operations_are_exported_and_writer_dispatches(tmp_path):
    for name in ("fst_scan", "pi_ratio", "fst_pi_stack", "sv_schematic", "sv_frequency", "presence_tracks"):
        assert getattr(ov.visualization, name) is getattr(ov.visualization, f"plot_{name}")
        assert f"visualization.plot_{name}" in popgen_plots.POPGEN_OPERATIONS
    payload = json.loads(ov.visualization.plot_sv_frequency(_svs(), groups=toy.GROUPS).model_dump_json())
    payload.pop("object_id", None)
    bogus = OrganelleResult.model_validate({**payload, "metrics": {"plot_kind": "nope"}})
    with pytest.raises(OrganelleInputError):
        ov.write(bogus, tmp_path / "x.png")


# ---------------------------------------------------------------- journal layout


def _all_plots(files):
    fst = ov.visualization.plot_fst_scan(files["toy.vcf"], groups=toy.GROUPS, pair=["wild", "crop"], window_size=500, step=250)
    pi = ov.visualization.plot_pi_ratio(toy.GROUPS, alignment=files["toy.aln.fasta"], reference="REF", pair=["wild", "crop"], window_size=500, step=250, min_sites=100, genbank=files["ref.gb"])
    return [
        fst,
        pi,
        ov.visualization.plot_fst_pi_stack(fst, pi),
        ov.visualization.plot_sv_schematic(files["toy.gfa"], reference="REF", start=1, end=2000, samples=["W1", "W4", "C1"]),
        ov.visualization.plot_sv_frequency(_svs(), groups=toy.GROUPS),
        ov.visualization.plot_presence_tracks(gfa=files["toy.gfa"], reference="REF", start=1, end=2000, groups=toy.GROUPS, genbank=files["ref.gb"]),
    ]


def _pdf_font_sizes(path: Path) -> list[float]:
    fitz = pytest.importorskip("fitz")
    sizes = []
    with fitz.open(path) as doc:
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    sizes.extend(span["size"] for span in line["spans"] if span["text"].strip())
    return sizes


def test_figures_are_journal_sized_with_no_text_below_5_pt(files, tmp_path):
    for plot in _all_plots(files):
        m = _metrics(plot)
        assert m["width"] == "double" and m["dpi"] == 600
        path = tmp_path / f"{m['plot_kind']}.pdf"
        ov.write(plot, path)
        sizes = _pdf_font_sizes(path)
        assert sizes and min(sizes) >= 4.95, (m["plot_kind"], min(sizes))
        fitz = pytest.importorskip("fitz")
        with fitz.open(path) as doc:
            width_mm = doc[0].rect.width / 72 * 25.4
            height_mm = doc[0].rect.height / 72 * 25.4
        assert width_mm <= 186 and height_mm <= 172, (m["plot_kind"], width_mm, height_mm)  # 183 mm + export pad
    with pytest.raises(OrganelleParameterError):
        ov.visualization.plot_sv_frequency(_svs(), groups=toy.GROUPS, width="poster")


def test_popgen_plots_compose_into_one_lettered_figure(files, tmp_path):
    from organelleverse.visualization.compose import compose_figure

    plots = _all_plots(files)
    for plot in plots:
        assert plot.panel is not None and plot.panel.aspect > 0
    restored = popgen_plots._restore_popgen_plot(OrganelleResult.model_validate_json(plots[3].model_dump_json()))
    assert restored.panel is not None
    figure = compose_figure([plots[2], plots[5], restored, plots[4]], titles=["Fst / pi", "presence", "SVs", "frequency"])
    assert _metrics(figure)["layout"]["letters"] == ["a", "b", "c", "d"]
    released = ov.write(figure, tmp_path / "composite.pdf")
    sizes = _pdf_font_sizes(Path(released.artifacts[0].uri))
    assert min(sizes) >= 4.95


def test_tracks_name_rows_or_index_them_and_write_tables(files, tmp_path):
    plot = ov.visualization.plot_presence_tracks(
        gfa=files["toy.gfa"], reference="REF", start=1, end=2000, groups=toy.GROUPS, genbank=files["ref.gb"],
        min_sv_length=100,
    )
    ov.write(plot, tmp_path / "tracks.pdf")
    rows = (tmp_path / "tracks.rows.tsv").read_text().splitlines()
    assert rows[0].split("\t")[:4] == ["row", "label", "sample", "group"] and len(rows) == 1 + len(toy.SAMPLES)
    assert (tmp_path / "tracks.svs.tsv").read_text().splitlines()[1].split("\t")[:3] == ["SV1", "601", "900"]
    fitz = pytest.importorskip("fitz")
    with fitz.open(tmp_path / "tracks.pdf") as doc:
        spans = [s for b in doc[0].get_text("dict")["blocks"] for line in b.get("lines", []) for s in line["spans"]]
    names = [s["size"] for s in spans if s["text"].strip() in toy.SAMPLES]
    assert len(names) == len(toy.SAMPLES) and min(names) >= 6.4  # sample names at >= 6.5 pt
    groups = [s["text"] for s in spans if s["text"].startswith(("wild", "crop"))]
    assert groups  # groups labelled directly next to their rows
    from organelleverse.visualization import popgen_render

    rows36 = [{"label": f"s{i}"} for i in range(36)]
    picks, size, mode = popgen_render._row_label_plan(rows36, 1.8)
    assert mode == "index" and size >= 6.5 and [t for _, t in picks][:3] == ["1", "2", "4"]
    picks, size, mode = popgen_render._row_label_plan(rows36, 3.0)
    assert mode == "names" and len(picks) == 36
