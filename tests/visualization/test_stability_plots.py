"""Multi-sample organelle stability figures (depth profile, frequencies, site heatmap)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.core.errors import OrganelleParameterError
from organelleverse.core.result import OrganelleResult
from organelleverse.visualization import (
    plot_depth_profile,
    plot_variant_frequencies,
    plot_variant_site_heatmap,
)

pytest.importorskip("matplotlib")

SAMPLES = ["Col-0", "msh1-a", "msh1-b", "reca3"]
GROUPS = {"Col-0": "WT", "msh1-a": "msh1", "msh1-b": "msh1", "reca3": "reca3"}

GENBANK = """LOCUS       MTX                     1200 bp    DNA     circular PLN 01-JAN-2000
DEFINITION  test mitochondrion.
ACCESSION   MTX
VERSION     MTX.1
FEATURES             Location/Qualifiers
     source          1..1200
                     /organism="Test plant"
                     /organelle="mitochondrion"
     gene            101..400
                     /gene="nad1"
     CDS             101..400
                     /gene="nad1"
     gene            complement(601..680)
                     /gene="trnM"
     tRNA            complement(601..680)
                     /gene="trnM"
ORIGIN
""" + "".join(f"{i * 60 + 1:>9} " + " ".join(["acgtacgtac"] * 6) + "\n" for i in range(20)) + "//\n"


def _depth() -> dict:
    profiles = []
    for s in SAMPLES:
        for seqid, n in (("MTX.1", 12), ("PTX.1", 6)):
            ratio = [0.0] * n
            if s.startswith("msh1") and seqid == "MTX.1":
                ratio[3] = 1.6
                ratio[9] = -2.5
            profiles.append({"sample": s, "seqid": seqid, "median_depth": 100.0,
                             "mean_depth": [100.0 * 2**r for r in ratio], "log2_ratio": ratio})
    return {"window_size": 100, "samples": SAMPLES, "profiles": profiles,
            "contigs": [{"seqid": "PTX.1", "length": 600, "organelle": "plastid"},
                        {"seqid": "MTX.1", "length": 1200, "organelle": "mitochondrion"}]}


def _variants() -> dict:
    sites = [
        {"site_id": "a", "scope": "mitochondrion", "seqid": "MTX.1", "position": 150, "ref": "A",
         "alt": "G", "kind": "SNV", "masked": False, "in_repeat": False},
        {"site_id": "b", "scope": "mitochondrion", "seqid": "MTX.1", "position": 700, "ref": "AT",
         "alt": "A", "kind": "InDel", "masked": False, "in_repeat": True},
        {"site_id": "c", "scope": "plastid", "seqid": "PTX.1", "position": 90, "ref": "C",
         "alt": "T", "kind": "SNV", "masked": True, "in_repeat": False},
        {"site_id": "d", "scope": "plastid", "seqid": "PTX.1", "position": 300, "ref": "G",
         "alt": "A", "kind": "SNV", "masked": False, "in_repeat": False},
    ]
    values = {
        "a": {"Col-0": 0.0, "msh1-a": 0.31, "msh1-b": 0.12, "reca3": 0.02},
        "b": {"Col-0": 0.0, "msh1-a": 0.07, "msh1-b": 0.0, "reca3": 0.0},
        "c": {"Col-0": 0.06, "msh1-a": 0.06, "msh1-b": 0.05, "reca3": None},
        "d": {"Col-0": 0.011, "msh1-a": 0.02, "msh1-b": 0.03, "reca3": 0.015},
    }
    rows = []
    for site in sites:
        for s in SAMPLES:
            f = values[site["site_id"]][s]
            category = ("low_confidence" if site["masked"] else
                        "repeat_related" if site["in_repeat"] else "other")
            rows.append({"sample": s, "site_id": site["site_id"], "alt_count": 0 if f is None else int(f * 200),
                         "depth": 0 if f is None else 200, "alt_fraction": f, "ci95_lower": None,
                         "ci95_upper": None, "detected": bool(f and f >= 0.01), "category": category})
    return {"samples": SAMPLES, "references": {"mitochondrion": ["MTX.1"], "plastid": ["PTX.1"]},
            "thresholds": {"min_alt_fraction": 0.01}, "sites": sites, "fractions": rows}


def test_depth_profile_orders_by_organelle_and_group_and_draws_tracks(tmp_path: Path) -> None:
    gb = tmp_path / "mt.gb"
    gb.write_text(GENBANK, encoding="utf-8")
    repeats = [{"sequence_id": "MTX.1", "positions": [200, 900], "length": 120, "type": "inverted"},
               {"sequence_id": "MTX.1", "positions": [10, 30], "length": 15, "type": "direct"}]
    plot = plot_depth_profile(_depth(), groups=GROUPS, sample_order=["reca3", "msh1-a", "Col-0", "msh1-b"],
                              genbank_paths=gb, repeats=repeats, min_arc_length=100)
    spec = plot.metrics
    assert [c["seqid"] for c in spec["contigs"]] == ["MTX.1", "PTX.1"]
    assert [i["sample"] for i in spec["layout"]] == ["reca3", "msh1-a", "msh1-b", "Col-0"]
    assert {b["gene"]: b["category"] for b in spec["genes"]["MTX.1"]} == {"nad1": "complex_I", "trnM": "tRNA"}
    assert [a["length"] for a in spec["arcs"]] == [120]
    assert not plot.artifacts
    for suffix in ("png", "svg", "pdf"):
        written = ov.write(plot, tmp_path / f"depth.{suffix}")
        assert {a.kind for a in written.artifacts} == {"figure", "figure_data"}
    with (tmp_path / "depth.depth.tsv").open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 4 * (12 + 6)
    assert rows[0]["sample"] == "reca3" and rows[0]["seqid"] == "MTX.1"


def test_depth_ridgeline_mode_renders(tmp_path: Path) -> None:
    plot = plot_depth_profile(_depth(), groups=GROUPS, mode="ridgeline")
    assert ov.write(plot, tmp_path / "ridge.png").artifacts


def test_frequency_points_only_at_or_above_threshold(tmp_path: Path) -> None:
    plot = plot_variant_frequencies(_variants(), groups=GROUPS, threshold=0.05)
    spec = plot.metrics
    assert list(spec["scopes"]) == ["mitochondrion", "plastid"]
    assert list(spec["kinds"]) == ["SNV", "InDel"]
    drawn = {(p["sample"], p["site_id"]): p["category"] for p in spec["points"]}
    assert drawn[("msh1-a", "a")] == "other"
    assert drawn[("msh1-a", "b")] == "repeat_related"
    assert drawn[("Col-0", "c")] == "low_confidence"
    assert all(p["fraction"] >= 0.05 for p in spec["points"])
    assert sorted(b["fraction"] for b in spec["bands"]) == [0.011, 0.015, 0.02, 0.02, 0.03]
    assert "\u2265 0.05" in spec["caption"] and "between 0.01 and 0.05" in spec["caption"]
    assert "caption" in plot_depth_profile(_depth()).metrics
    written = ov.write(plot, tmp_path / "freq.svg")
    assert written.artifacts[0].uri.endswith("freq.svg")
    assert (tmp_path / "freq.frequencies.tsv").exists()


def test_site_heatmap_keeps_recurrent_sites_and_flags_truncation(tmp_path: Path) -> None:
    plot = plot_variant_site_heatmap(_variants(), groups=GROUPS, threshold=0.05)
    rows = plot.metrics["rows"]
    assert [r["site_id"] for r in rows] == ["a", "b", "c"]
    assert [r["recurrence"] for r in rows] == [2, 1, 3]
    assert rows[2]["values"][-1] is None and rows[2]["category"] == "low_confidence"
    assert rows[1]["label"] == "mt 700 \u2212T"
    assert ov.write(plot, tmp_path / "sites.pdf").artifacts
    top = plot_variant_site_heatmap(_variants(), threshold=0.05, max_sites=1)
    assert [r["site_id"] for r in top.metrics["rows"]] == ["c"] and "sites_truncated" in top.flags


def test_reference_sample_cancels_shared_bias_and_low_confidence_sites_can_be_dropped(tmp_path: Path) -> None:
    plot = plot_depth_profile(_depth(), groups=GROUPS, reference_sample="msh1-a")
    ratios = {(p["sample"], p["seqid"]): p["log2_ratio"] for p in plot.metrics["profiles"]}
    assert set(ratios[("msh1-b", "MTX.1")]) == {0.0}
    assert ratios[("Col-0", "MTX.1")][3] == -1.6
    ov.write(plot, tmp_path / "vs_ref.png")
    with (tmp_path / "vs_ref.depth.tsv").open() as fh:
        assert "log2_vs_msh1-a" in fh.readline()
    with pytest.raises(OrganelleParameterError):
        plot_depth_profile(_depth(), reference_sample="nope")
    kept = plot_variant_site_heatmap(_variants(), threshold=0.05, include_low_confidence=False)
    assert [r["site_id"] for r in kept.metrics["rows"]] == ["a", "b"]


def test_tracks_mode_zooms_into_a_region_with_one_axis_per_sample(tmp_path: Path) -> None:
    gb = tmp_path / "mt.gb"
    gb.write_text(GENBANK, encoding="utf-8")
    plot = plot_depth_profile(_depth(), groups=GROUPS, mode="tracks", region=["MTX.1", 201, 900],
                              genbank_paths=gb, width="onehalf")
    assert dict(plot.metrics["region"]) == {"seqid": "MTX.1", "start": 201, "end": 900}
    assert [c["seqid"] for c in plot.metrics["contigs"]] == ["MTX.1"]
    ov.write(plot, tmp_path / "tracks.pdf")
    with (tmp_path / "tracks.depth.tsv").open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 4 * 7 and rows[0]["start"] == "201" and rows[6]["end"] == "900"
    for region in (None, ["MTX.1", 0, 900], ["MTX.1", 900, 200], ["chrZ", 1, 10]):
        with pytest.raises(OrganelleParameterError):
            plot_depth_profile(_depth(), mode="tracks", region=region)


def test_figures_compose_into_one_lettered_page(tmp_path: Path) -> None:
    from organelleverse.visualization import compose_figure

    panels = [plot_depth_profile(_depth(), groups=GROUPS), plot_variant_frequencies(_variants(), groups=GROUPS),
              plot_variant_site_heatmap(_variants(), groups=GROUPS)]
    assert all(p.panel is not None and p.panel.aspect > 0 for p in panels)
    figure = compose_figure(panels, columns=1)
    assert ov.write(figure, tmp_path / "composite.pdf").artifacts
    restored = OrganelleResult.model_validate_json(panels[0].model_dump_json())
    assert ov.write(restored, tmp_path / "restored.png").artifacts


def test_serialized_results_render_through_ov_write(tmp_path: Path) -> None:
    for plot in (plot_depth_profile(_depth(), groups=GROUPS),
                 plot_variant_frequencies(_variants()),
                 plot_variant_site_heatmap(_variants())):
        restored = OrganelleResult.model_validate_json(plot.model_dump_json())
        assert ov.write(restored, tmp_path / f"{plot.operation_id}.png").artifacts


def test_sqrt_scale_and_bad_parameters() -> None:
    assert plot_variant_frequencies(_variants(), sqrt_scale=False).metrics["sqrt_scale"] is False
    with pytest.raises(OrganelleParameterError):
        plot_variant_frequencies(_variants(), threshold=1.5)
    with pytest.raises(OrganelleParameterError):
        plot_depth_profile(_depth(), sample_order=["Col-0", "nope"])
    assert math.isclose(plot_depth_profile(_depth(), limit=3).metrics["limit"], 3)
