"""OV-drawn assembly graph: compute-only result, ov.write rendering and restore."""

import json

import pytest

import organelleverse as ov
from organelleverse.core.errors import OrganelleParameterError
from organelleverse.core.result import OrganelleResult
from organelleverse.visualization.assembly_graph import graph_layout, plot_assembly_graph

GFA = (
    "S\tctg.A\t*\tLN:i:40000\tdp:f:40.0\n"
    "S\tctg.B\t*\tLN:i:25000\tdp:f:42.0\n"
    "S\tctg.R\t*\tLN:i:3000\tdp:f:81.0\n"
    "S\tctg.C\t*\tLN:i:30000\tdp:f:44.0\n"
    "L\tctg.A\t+\tctg.R\t+\t0M\nL\tctg.R\t+\tctg.B\t+\t0M\n"
    "L\tctg.B\t+\tctg.R\t+\t0M\nL\tctg.R\t+\tctg.A\t+\t0M\n"
    "L\tctg.C\t+\tctg.C\t+\t0M\n"
    "P\tmt\tctg.A+,ctg.R+,ctg.B+,ctg.R+\t*\n"
)


def _plot(tmp_path, **kwargs):
    path = tmp_path / "graph.gfa"
    path.write_text(GFA)
    return plot_assembly_graph(path, title="fixture", **kwargs)


def test_compute_only_metrics(tmp_path):
    plot = _plot(tmp_path)
    assert not plot.artifacts
    metrics = plot.metrics
    assert (metrics["nodes"], metrics["edges"], metrics["repeat_count"]) == (4, 5, 1)
    assert metrics["single_copy_depth"] == 42.0
    labels = {row["name"]: row["label"] for row in metrics["segments"]}
    assert labels == {"ctg.A": "A", "ctg.B": "B", "ctg.R": "R", "ctg.C": "C"}
    (repeat,) = metrics["repeats"]
    assert repeat["status"] == "candidate_only"
    assert list(metrics["insets"]) == ["ctg.R"]
    assert "conformations_candidate_only" in plot.flags
    assert set(metrics["layout"]) == {"ctg.A", "ctg.B", "ctg.R", "ctg.C"}


def test_layout_is_deterministic_and_separates_components(tmp_path):
    first = _plot(tmp_path).metrics["layout"]
    second = _plot(tmp_path).metrics["layout"]
    assert first == second
    main = [x for name in ("ctg.A", "ctg.B", "ctg.R") for x, _ in first[name]]
    ring = [x for x, _ in first["ctg.C"]]
    assert max(main) < min(ring) or max(ring) < min(main)


def test_layout_without_links_keeps_every_segment():
    layout = graph_layout(
        [{"name": "a", "length": 100}, {"name": "b", "length": 5000}], [], seed=0
    )
    assert set(layout) == {"a", "b"} and len(layout["b"]) > len(layout["a"])


def test_write_svg_png_pdf_with_table(tmp_path):
    plot = _plot(tmp_path)
    for suffix in ("svg", "png", "pdf"):
        published = ov.write(plot, tmp_path / f"figure.{suffix}")
        kinds = {artifact.kind for artifact in published.artifacts}
        assert kinds == {"figure", "figure_data"}
    table = (tmp_path / "figure.conformations.tsv").read_text().splitlines()
    assert table[0].startswith("repeat_id\trepeat\tkind")
    assert len(table) == 3
    assert "Conf." in (tmp_path / "figure.svg").read_text()


def test_serialized_result_renders_through_ov_write(tmp_path):
    plot = _plot(tmp_path)
    plain = OrganelleResult.model_validate(plot.model_dump(mode="python"))
    published = ov.write(plain, tmp_path / "restored.svg")
    assert (tmp_path / "restored.svg").is_file()
    assert published.operation_id == "visualization.plot_assembly_graph"


def test_repeats_are_numbered_along_the_master_circle(tmp_path):
    plot = _plot(tmp_path)
    (repeat,) = plot.metrics["repeats"]
    assert repeat["id"] == "R1"
    segment = next(row for row in plot.metrics["segments"] if row["name"] == "ctg.R")
    assert segment["repeat_id"] == "R1" and segment["color"] == repeat["color"]
    master = {m["length"] for m in plot.metrics["master_molecules"]}
    assert master == {71000, 30000}


def test_triptych_renders_master_and_subgenomic_panels(tmp_path):
    plot = _plot(tmp_path, style="triptych")
    assert not plot.metrics["insets"]
    ov.write(plot, tmp_path / "triptych.svg")
    svg = (tmp_path / "triptych.svg").read_text()
    assert "Representative circular path" in svg and "Candidate recombination paths" in svg


def test_mtpt_connectors_are_summed_per_segment_and_region():
    from organelleverse.visualization.assembly_graph import _homology

    regions = {
        "LSC": {"start": 0, "length": 600},
        "IRb": {"start": 600, "length": 100},
        "SSC": {"start": 700, "length": 200},
        "IRa": {"start": 900, "length": 100},
    }
    plastid = {
        "reference": {"length": 1000, "regions": regions},
        "fragments": [
            {"id": "MTPT1", "segment": "m", "start": 1, "end": 100, "length": 100,
             "donor_blocks": [{"start": 11, "end": 110, "identity": 99.0}],
             "spanning_reads": 5, "integration": "supported"},
            {"id": "MTPT2", "segment": "m", "start": 301, "end": 600, "length": 300,
             "donor_blocks": [{"start": 201, "end": 500, "identity": 95.0}],
             "spanning_reads": 1, "integration": "unsupported"},
            {"id": "MTPT3", "segment": "m", "start": 701, "end": 760, "length": 60,
             "donor_blocks": [{"start": 621, "end": 680, "identity": 90.0}, {"start": 921, "end": 980, "identity": 90.0}],
             "spanning_reads": None, "integration": "not_assessed"},
        ],
    }
    rows = {row["plastid_segment"]: row for row in _homology(plastid, None, [{"name": "m", "length": 1000}], 0)}
    assert set(rows) == {"pt.LSC", "pt.IR"}  # IRa folds onto the collapsed IR: one connector
    lsc = rows["pt.LSC"]
    assert (lsc["mtpt_count"], lsc["total_length"], lsc["assessed"], lsc["integrated"]) == (2, 400, 2, 1)
    assert lsc["identity"] == round((99.0 * 100 + 95.0 * 300) / 400, 2)
    assert rows["pt.IR"]["mtpt_count"] == 1


def test_origin_colouring_needs_a_plastid(tmp_path):
    with pytest.raises(OrganelleParameterError):
        _plot(tmp_path, color_by="origin")


def test_ovasm_run_directory_uses_evidence_pairings(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    gfa = GFA.replace("ctg.", "s.")
    (run / "organelle.gfa").write_text(gfa)
    (run / "assembly.gfa").write_text(GFA.replace("ctg.", ""))
    (run / "organelle.json").write_text(json.dumps({"sample": "s"}))
    evidence = {"repeats": [{"repeat": "R+", "entries": ["A+", "B+"], "exits": ["A+", "B+"], "pairing_reads": [3, 9]}]}
    (run / "evidence.json").write_text(json.dumps(evidence))
    plot = plot_assembly_graph(run)
    (repeat,) = plot.metrics["repeats"]
    assert repeat["support_source"] == "pairings"
    assert [c["proportion"] for c in repeat["configurations"]] == [0.75, 0.25]
    assert "ovasm evidence.json" in plot.metrics["frequency_source"]
    ov.write(plot, tmp_path / "run.png")


def test_printed_size_and_caption(tmp_path):
    from matplotlib import pyplot as plt

    plot = _plot(tmp_path, style="triptych", width="single")
    ov.write(plot, tmp_path / "single.png")
    caption = (tmp_path / "single.caption.txt").read_text()
    assert caption.startswith("fixture.") and "candidates" in caption
    image = plt.imread(tmp_path / "single.png")
    assert abs(image.shape[1] / 600 * 25.4 - 89) < 3  # 89 mm at 600 dpi, tight bbox not used


def test_network_is_a_compose_figure_panel(tmp_path):
    from organelleverse.visualization import compose_figure

    plot = _plot(tmp_path)
    assert plot.panel is not None and 0.3 <= plot.panel.aspect <= 0.8
    figure = compose_figure([plot, plot], width="double")
    ov.write(figure, tmp_path / "composed.pdf")
    assert (tmp_path / "composed.pdf").stat().st_size > 0


def test_without_conformations_draws_only_the_graph(tmp_path):
    plot = _plot(tmp_path, show_conformations=False)
    assert not plot.metrics["insets"]
    ov.write(plot, tmp_path / "plain.png")
    assert (tmp_path / "plain.png").stat().st_size > 0


def test_segment_labels_never_overlap_on_a_dense_graph(tmp_path):
    """Segment labels stay clear of each other when short segments crowd the network panel."""
    import itertools

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from organelleverse.visualization.assembly_graph import _figure
    from organelleverse.visualization.style import publication_style

    names = [f"ctg.s{i}" for i in range(36)]
    lines = [f"S\t{n}\t*\tLN:i:{2000 + 700 * (i % 5)}\tdp:f:{40 + i % 7}.0\n" for i, n in enumerate(names)]
    lines += [f"L\t{a}\t+\t{b}\t+\t0M\n" for a, b in zip(names, names[1:] + names[:1])]
    lines += [f"L\t{names[i]}\t+\t{names[(i + 9) % 36]}\t+\t0M\n" for i in range(0, 36, 6)]
    path = tmp_path / "dense.gfa"
    path.write_text("".join(lines))
    plot = plot_assembly_graph(path, show_conformations=False, width="single")  # 89 mm: crowded
    with publication_style():
        fig = _figure(plot.metrics)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer) for t in fig.axes[0].texts if t.get_text().endswith("\u00d7")]
        plt.close(fig)
    assert len(boxes) == 36
    assert not [pair for pair in itertools.combinations(boxes, 2) if pair[0].overlaps(pair[1])]


def test_mtpt_transfer_bands_take_the_donor_region_tint():
    """Transfers are light bands tinted by donor region, wider for more transferred sequence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Polygon

    from organelleverse.visualization import assembly_graph as ag

    line = np.column_stack([np.linspace(0.0, 10.0, 20), np.zeros(20)])
    curves = {"m": line, "pt.LSC": line + [0, 5], "pt.IR": line + [0, 8], "pt.SSC": line + [0, 11]}
    homology = [
        {"segment": "m", "plastid_segment": name, "segment_at": 0.5, "plastid_at": 0.5,
         "total_length": length, "assessed": 0, "integrated": 0}
        for name, length in (("pt.IR", 1000), ("pt.SSC", 250), ("pt.LSC", 4000))
    ]
    metrics = {
        "homology": homology,
        "plastid_graph": {"segments": [{"name": f"pt.{r}", "region": r} for r in ("LSC", "IR", "SSC")]},
    }
    fig, ax = plt.subplots()
    ag._draw_homology(ax, metrics, curves, 0.05)
    bands = [patch for patch in ax.patches if isinstance(patch, Polygon)]
    plt.close(fig)
    assert [to_rgb(band.get_facecolor()) for band in bands] == [
        to_rgb(ag.MTPT_BAND_COLORS[region]) for region in ("LSC", "IR", "SSC")
    ]
    widths = []
    for band in bands:
        outline = band.get_xy()[:96]  # 48 points out, 48 back; the closing point dropped
        widths.append(float(np.linalg.norm(outline[24] - outline[95 - 24])))
    assert widths[0] > widths[1] > widths[2] > 0


def test_conformations_drawn_from_reported_paths_are_labelled(tmp_path):
    """When a one-ended fork blocks spelling, rings come from the GFA paths and say so."""
    path = tmp_path / "blocked.gfa"
    path.write_text(
        GFA.replace("P\tmt\tctg.A+,ctg.R+,ctg.B+,ctg.R+\t*\n", "")
        + "S\tctg.T\t*\tLN:i:900\tdp:f:5.0\nL\tctg.A\t+\tctg.T\t+\t0M\n"
        + "P\ts#1#mt.1\tctg.A+,ctg.R+,ctg.B+,ctg.R+\t0M,0M,0M\tTP:Z:circular\n"
    )
    plot = plot_assembly_graph(path, title="paths")
    (row,) = plot.metrics["repeats"]
    assert {conf["molecules_source"] for conf in row["configurations"]} == {"paths", "paths_recombined"}
    ov.write(plot, tmp_path / "paths.svg")
    svg = (tmp_path / "paths.svg").read_text()
    assert "GFA path" in svg and "GFA paths, recombined" in svg
    table = (tmp_path / "paths.conformations.tsv").read_text().splitlines()
    assert table[0].endswith("\tmolecules_source")
    assert {line.split("\t")[-1] for line in table[1:]} == {"paths", "paths_recombined"}
    assert "come from the assembler's paths in the GFA (GFA;" in (tmp_path / "paths.caption.txt").read_text()
