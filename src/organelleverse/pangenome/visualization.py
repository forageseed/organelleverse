"""Publication SVG/PDF/PNG panels paired with exact tabular sources.

Every panel uses supplied analysis values. No layout is presented as a GFA
topology, and annotation coverage is distinguished from read coverage.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from Bio import Phylo
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrow

COLORS = {"core": "#0072B2", "shell": "#E69F00", "cloud": "#009E73", "unobserved": "#999999"}


def _figure(width: float = 8, height: float = 4) -> tuple[Figure, Any]:
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def _table(path: Path, fields: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)
    return str(path)


def _validate_formats(formats: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(formats)
    if not selected or len(set(selected)) != len(selected) or set(selected) - {"svg", "pdf", "png"}:
        raise ValueError("Choose unique publication formats from svg, pdf and png")
    return selected


def _save_figures(figure: Figure, prefix: Path, formats: Sequence[str]) -> dict[str, str]:
    outputs = {}
    with matplotlib.rc_context(
        {"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans"}
    ):
        for kind in _validate_formats(formats):
            destination = prefix.with_suffix(f".{kind}")
            metadata = {"Software" if kind == "png" else "Creator": "OrganelleVerse"}
            figure.savefig(destination, format=kind, dpi=300, metadata=metadata)
            outputs[kind] = str(destination)
    figure.clear()
    return outputs


def render_matrix(
    rows: Sequence[str],
    columns: Sequence[str],
    matrix: Sequence[Sequence[int | float]],
    output_prefix: str | Path,
    *,
    title: str = "Presence / absence",
    value_label: str = "Presence",
    formats: Sequence[str] = ("svg",),
) -> dict[str, str]:
    """Export the full source matrix and an explicitly labelled heatmap."""
    values = np.asarray(matrix)
    if values.shape != (len(rows), len(columns)) or not rows or not columns:
        raise ValueError("Matrix labels must match a nonempty rectangular matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Plot matrix must contain finite nonnegative values")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = _figure(max(7, len(columns) * 0.35), min(20, max(4, len(rows) * 0.18)))
    plotted = axes.imshow(values, aspect="auto", interpolation="nearest", cmap="cividis")
    axes.set_xticks(range(len(columns)), columns, rotation=90)
    # Large source matrices remain complete; tick thinning only affects labels.
    stride = max(1, int(np.ceil(len(rows) / 60)))
    indices = list(range(0, len(rows), stride))
    axes.set_yticks(indices, [rows[index] for index in indices])
    axes.set_title(title)
    figure.colorbar(plotted, ax=axes, label=value_label)
    return {
        **_save_figures(figure, prefix, formats),
        "table": _table(
            prefix.with_suffix(".tsv"),
            ["feature", *columns],
            [[label, *row] for label, row in zip(rows, matrix, strict=True)],
        ),
    }


def render_node_summary(
    pav: Mapping[str, Any], output_dir: str | Path, *, formats: Sequence[str] = ("svg",)
) -> dict[str, str]:
    """Frequency histogram and mutually exclusive core/shell/cloud counts."""
    nodes, frequencies, classes = pav["nodes"], pav["frequencies"], pav["classes"]
    unit = pav.get("frequency_denominator", "paths")
    if unit not in {"paths", "samples"}:
        raise ValueError("Frequency denominator must be paths or samples")
    if not (len(nodes) == len(frequencies) == len(classes)) or not nodes:
        raise ValueError("Node summary arrays must have equal nonzero lengths")
    if any(value < 0 or value > 1 or not np.isfinite(value) for value in frequencies):
        raise ValueError("Node frequencies must be finite and between zero and one")
    if any(label not in COLORS for label in classes):
        raise ValueError("Each node must have exactly one core/shell/cloud/unobserved class")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    table = _table(
        directory / "node_frequency.tsv",
        ["node", "frequency", "class"],
        list(zip(nodes, frequencies, classes, strict=True)),
    )
    figure, axes = _figure()
    axes.hist(frequencies, bins=np.linspace(0, 1, 21), color=COLORS["core"], edgecolor="white")
    axes.set(
        xlabel=f"Fraction of {unit} containing node",
        ylabel="Graph nodes",
        title="Node presence frequency",
        xlim=(0, 1),
    )
    frequency_figures = _save_figures(figure, directory / "node_frequency", formats)
    counts = Counter(classes)
    figure, axes = _figure(5, 4)
    axes.bar(list(COLORS), [counts[label] for label in COLORS], color=list(COLORS.values()))
    axes.set(ylabel="Graph nodes", title="Node frequency classes")
    return {
        **{f"frequency_{kind}": path for kind, path in frequency_figures.items()},
        **{
            f"class_{kind}": path
            for kind, path in _save_figures(figure, directory / "node_classes", formats).items()
        },
        "frequency_table": table,
        "class_table": _table(
            directory / "node_classes.tsv",
            ["class", "node_count"],
            [[label, counts[label]] for label in COLORS],
        ),
    }


def render_gene_arrows(
    arrows: Sequence[Mapping[str, Any]],
    path_bounds: Mapping[str, tuple[int, int]],
    output_dir: str | Path,
    *,
    formats: Sequence[str] = ("svg",),
) -> dict[str, str]:
    """Draw supplied feature parts in native path coordinates at a shared scale."""
    if not path_bounds:
        raise ValueError("Gene arrows require path bounds")
    paths = list(path_bounds)
    for row in arrows:
        if row["path"] not in path_bounds or row["strand"] not in {"+", "-", "."}:
            raise ValueError("Gene arrows require known paths and valid strands")
        start, end = path_bounds[row["path"]]
        if not start <= row["start"] < row["end"] <= end:
            raise ValueError("Gene arrow is outside its path bounds")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Greedy interval packing separates overlapping loci and their labels.
    # It changes only the display lanes; all feature coordinates stay intact.
    span = max(end for _, end in path_bounds.values()) - min(
        start for start, _ in path_bounds.values()
    )
    placed = []
    groups = []
    base = 0
    for path in paths:
        lane_ends = []
        path_rows = sorted(
            (row for row in arrows if row["path"] == path),
            key=lambda row: (row["start"], row["end"], row["gene"]),
        )
        for row in path_rows:
            center = (row["start"] + row["end"]) / 2
            # Reserve label width at 7 pt in the 10-inch export, with spacing.
            half_label = span * 0.005 * max(3, len(str(row["gene"])))
            left = min(row["start"], center - half_label)
            right = max(row["end"], center + half_label)
            lane = next((i for i, end in enumerate(lane_ends) if end < left), len(lane_ends))
            if lane == len(lane_ends):
                lane_ends.append(right)
            else:
                lane_ends[lane] = right
            placed.append((row, base + lane))
        count = max(1, len(lane_ends))
        groups.append((path, base, count))
        base += count + 1
    figure, axes = _figure(10, max(3, base * 0.38 + 1.2))
    for path, low, count in groups:
        for lane in range(count):
            axes.plot(path_bounds[path], [low + lane] * 2, color="#dddddd", linewidth=0.5)
    for row, y in placed:
        start, end = row["start"], row["end"]
        if row["strand"] == ".":
            axes.plot([start, end], [y, y], color=COLORS["core"], linewidth=3)
        else:
            origin, delta = (start, end - start) if row["strand"] == "+" else (end, start - end)
            axes.add_patch(
                FancyArrow(
                    origin,
                    y,
                    delta,
                    0,
                    width=0.16,
                    head_width=0.3,
                    head_length=min((end - start) * 0.25, span * 0.007),
                    length_includes_head=True,
                    color=COLORS["core"],
                )
            )
        axes.text((start + end) / 2, y + 0.23, row["gene"], ha="center", fontsize=7)
    axes.set_yticks([low + (count - 1) / 2 for _, low, count in groups], paths)
    axes.set(
        xlabel="Path coordinate (bp; zero-based)",
        title="Annotated loci (overlapping features on separate lanes)",
        ylim=(-0.6, base - 0.1),
    )
    fields = [
        "path",
        "sample",
        "gene",
        "locus_id",
        "source_locus_tag",
        "part",
        "start",
        "end",
        "strand",
    ]
    return {
        **_save_figures(figure, directory / "gene_arrows", formats),
        "table": _table(
            directory / "gene_arrows.tsv",
            fields,
            [[row.get(field, "") for field in fields] for row in arrows],
        ),
        "bounds_table": _table(
            directory / "path_bounds.tsv",
            ["path", "start", "end"],
            [[path, *bounds] for path, bounds in path_bounds.items()],
        ),
    }


def render_tree(
    tree: Mapping[str, Any], output_dir: str | Path, *, formats: Sequence[str] = ("svg",)
) -> dict[str, str]:
    """Draw actual Newick branch lengths with explicit scientific tree semantics."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    msa_mode = tree.get("mode") == "msa"
    name = "msa_tree" if msa_mode else "node_pav_tree"
    parsed = Phylo.read(io.StringIO(tree["newick"]), "newick")
    figure, axes = _figure(9, max(3, len(tree["paths"]) * 0.35))
    Phylo.draw(parsed, axes=axes, do_show=False)
    axes.set(
        title=(
            f"RAxML-NG maximum-likelihood tree ({tree.get('model', 'declared model')})"
            if msa_mode
            else "Node-PAV Jaccard / UPGMA similarity tree"
        ),
        xlabel="Substitutions per site" if msa_mode else "Jaccard distance / 2 (UPGMA height)",
        ylabel="",
    )
    newick_path = directory / f"{name}.nwk"
    newick_path.write_text(tree["newick"] + "\n", encoding="utf-8")
    files = {**_save_figures(figure, directory / name, formats), "newick": str(newick_path)}
    if msa_mode:
        files["branch_table"] = _table(
            directory / "msa_tree_branches.tsv",
            ["descendant_taxa", "branch_length", "bootstrap_support"],
            [
                [
                    ";".join(t.name for t in clade.get_terminals()),
                    clade.branch_length,
                    clade.confidence,
                ]
                for clade in parsed.find_clades()
            ],
        )
    else:
        files["distance_table"] = _table(
            directory / "jaccard_distances.tsv",
            ["path", *tree["paths"]],
            [
                [path, *values]
                for path, values in zip(tree["paths"], tree["distances"], strict=True)
            ],
        )
        files["bootstrap_table"] = _table(
            directory / "bootstrap_clades.tsv",
            ["clade_paths", "support", "replicate_count"],
            [
                [";".join(clade["paths"]), clade["support"], clade.get("replicate_count")]
                for clade in tree["clades"]
            ],
        )
    return files


def write_report(
    output_dir: str | Path,
    pav: Mapping[str, Any] | None,
    tree: Mapping[str, Any] | None = None,
    annotation_result: Mapping[str, Any] | None = None,
    *,
    formats: Sequence[str] = ("svg",),
) -> list[Path]:
    """Write supplied results as selected SVG/PDF/300-dpi PNG and source tables.

    Optional analyses are omitted if absent, never simulated. The caller owns
    artifact/provenance publication. Node frequency labels retain the declared paths or samples denominator.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    formats = _validate_formats(formats)
    files = []
    if pav is not None:
        files = list(render_node_summary(pav, directory, formats=formats).values())
        files.extend(
            render_matrix(
                pav["nodes"],
                pav["paths"],
                pav["matrix"],
                directory / "node_pav",
                formats=formats,
                title=f"Graph-node presence across {pav.get('frequency_denominator', 'paths')}",
            ).values()
        )
    if tree is not None:
        files.extend(render_tree(tree, directory, formats=formats).values())
    if annotation_result is not None and annotation_result["genes"]:
        files.extend(
            render_matrix(
                annotation_result["genes"],
                annotation_result["samples"],
                annotation_result["copy_matrix"],
                directory / "gene_copy_number",
                formats=formats,
                title="Annotated gene copies",
                value_label="Supplied loci",
            ).values()
        )
        files.extend(
            render_matrix(
                annotation_result["genes"],
                annotation_result["samples"],
                annotation_result["pav_matrix"],
                directory / "gene_pav",
                formats=formats,
                title="Annotated gene presence",
            ).values()
        )
        files.extend(
            render_gene_arrows(
                annotation_result["gene_arrows"],
                annotation_result["path_bounds"],
                directory,
                formats=formats,
            ).values()
        )
        for name in ["bin_coverage", "projection"]:
            rows = annotation_result[name]
            if rows:
                fields = list(rows[0])
                files.append(
                    _table(
                        directory / f"{name}.tsv",
                        fields,
                        [[row[field] for field in fields] for row in rows],
                    )
                )
        bins = annotation_result["bin_coverage"]
        figure, axes = _figure(9, 4)
        paths = list(dict.fromkeys(row["path"] for row in bins))
        for path in paths:
            selected = [row for row in bins if row["path"] == path]
            axes.plot(
                [(row["start"] + row["end"]) / 2 for row in selected],
                [row["annotated_fraction"] for row in selected],
                label=path,
            )
        axes.set(
            xlabel="Path coordinate (bp)",
            ylabel="Annotated fraction",
            ylim=(0, 1),
            title="Union annotation coverage per bin",
        )
        axes.legend(fontsize=7)
        files.extend(_save_figures(figure, directory / "bin_coverage", formats).values())
    figure_sources = {
        "node_frequency": ["node_frequency.tsv"],
        "node_classes": ["node_classes.tsv"],
        "node_pav": ["node_pav.tsv"],
        "node_pav_tree": ["node_pav_tree.nwk", "jaccard_distances.tsv", "bootstrap_clades.tsv"],
        "msa_tree": ["msa_tree.nwk", "msa_tree_branches.tsv"],
        "gene_copy_number": ["gene_copy_number.tsv"],
        "gene_pav": ["gene_pav.tsv"],
        "gene_arrows": ["gene_arrows.tsv", "path_bounds.tsv"],
        "bin_coverage": ["bin_coverage.tsv"],
    }
    files.append(
        _table(
            directory / "figure_sources.tsv",
            ["figure", "source_files"],
            [
                [Path(path).name, ";".join(figure_sources[Path(path).stem])]
                for path in files
                if Path(path).suffix.lstrip(".") in formats
            ],
        )
    )
    return [Path(path) for path in files]


def render_graph_summary(
    statistics: Mapping[str, Any],
    output_dir: str | Path,
    *,
    formats: Sequence[str] = ("svg",),
) -> list[Path]:
    """Show actual GFA counts and distributions, including graphs without paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12, 3.5), layout="constrained")
    FigureCanvasAgg(figure)
    overview, lengths_axes, degree_axes = figure.subplots(1, 3)
    summary_keys = ("nodes", "edges", "paths", "components", "total_bp", "n50")
    overview.axis("off")
    overview.set_title("GFA graph summary")
    overview.text(
        0.05,
        0.95,
        "\n".join(
            f"{key.replace('_', ' ')}: {statistics[key] if statistics[key] is not None else 'unknown'}"
            for key in summary_keys
        ),
        transform=overview.transAxes,
        va="top",
        linespacing=1.7,
    )
    lengths = [value for value in statistics["node_lengths"].values() if value is not None]
    lengths_axes.hist(lengths, bins=30, color=COLORS["core"])
    lengths_axes.set(xlabel="Known segment length (bp)", ylabel="Segments")
    degrees = Counter(statistics["degree"].values())
    degree_axes.bar(list(degrees), list(degrees.values()), color=COLORS["shell"])
    degree_axes.set(xlabel="Link degree (loops count twice)", ylabel="Segments")
    images = _save_figures(figure, directory / "graph_summary", formats)
    tables = [
        _table(
            directory / "graph_summary.tsv",
            ["metric", "value"],
            [[key, statistics[key]] for key in summary_keys],
        ),
        _table(
            directory / "node_lengths.tsv",
            ["node", "length_bp"],
            list(statistics["node_lengths"].items()),
        ),
        _table(
            directory / "node_degree.tsv", ["node", "degree"], list(statistics["degree"].items())
        ),
    ]
    sources = directory / "figure_sources.tsv"
    exists = sources.exists()
    with sources.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if not exists:
            writer.writerow(["figure", "source_files"])
        for name in images.values():
            writer.writerow([Path(name).name, ";".join(Path(table).name for table in tables)])
    return [Path(name) for name in [*images.values(), *tables, str(sources)]]
