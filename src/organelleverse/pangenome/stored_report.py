"""Exact whole-cohort summaries and explicitly bounded matrix previews."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .pav_store import iter_pav_rows, load_pav_metadata, read_pav_page
from .visualization import COLORS, _figure, _save_figures, _table, render_matrix, write_report


def write_stored_report(
    store_directory, output_directory, tree=None, annotation=None, *, formats=("svg",)
):
    """Keep complete matrices in their captured TSV/Parquet artifacts.

    The frequency histogram uses weighted counts for every node, not the
    preview. Matrix figures show the first 500 rows and 100 biological samples
    in source order, with those bounds recorded in the title and source table.
    """
    source, directory = Path(store_directory), Path(output_directory)
    metadata = load_pav_metadata(source)
    directory.mkdir(parents=True, exist_ok=True)
    files = write_report(directory, None, tree, annotation, formats=formats)
    source_rows = []
    frequencies = directory / "node_frequency.tsv"
    with frequencies.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["node", "count", "frequency", "class"])
        for row in iter_pav_rows(source):
            writer.writerow([row["node_id"], row["count"], row["frequency"], row["category"]])
    files.append(frequencies)
    figure, axes = _figure()
    counts = metadata.sample_frequency_counts
    axes.hist(
        [count / len(metadata.samples) for count in counts],
        weights=list(counts.values()),
        bins=np.linspace(0, 1, 21),
        color=COLORS["core"],
        edgecolor="white",
    )
    axes.set(
        xlabel="Fraction of samples containing node",
        ylabel="Graph nodes",
        title=f"Node presence frequency · all {metadata.node_count} nodes",
        xlim=(0, 1),
    )
    panels = _save_figures(figure, directory / "node_frequency", formats)
    files.extend(Path(path) for path in panels.values())
    source_rows.extend([Path(path).name, "node_frequency.tsv"] for path in panels.values())
    classes = metadata.sample_class_counts
    files.append(
        Path(
            _table(
                directory / "node_classes.tsv",
                ["class", "node_count"],
                [[label, classes.get(label, 0)] for label in COLORS],
            )
        )
    )
    figure, axes = _figure(5, 4)
    axes.bar(list(COLORS), [classes.get(label, 0) for label in COLORS], color=list(COLORS.values()))
    axes.set(ylabel="Graph nodes", title="Node frequency classes · all nodes")
    panels = _save_figures(figure, directory / "node_classes", formats)
    files.extend(Path(path) for path in panels.values())
    source_rows.extend([Path(path).name, "node_classes.tsv"] for path in panels.values())
    preview = read_pav_page(source, limit=500, column_limit=100)
    rows = preview["rows"]
    title = (
        f"Node PAV preview · first {len(rows)}/{metadata.node_count} nodes, "
        f"{len(preview['columns'])}/{len(metadata.samples)} samples"
    )
    panels = render_matrix(
        [row["node_id"] for row in rows],
        preview["columns"],
        [row["presence"] for row in rows],
        directory / "node_pav_preview",
        title=title,
        formats=formats,
    )
    files.extend(Path(path) for path in panels.values())
    source_rows.extend(
        [Path(path).name, "node_pav_preview.tsv;matrix-preview.json"]
        for path in panels.values()
        if Path(path).suffix.lstrip(".") in formats
    )
    policy = directory / "matrix-preview.json"
    policy.write_text(
        json.dumps(
            {
                "selection": "first rows and columns in canonical store order; no clustering or subsampling",
                "shown_nodes": len(rows),
                "total_nodes": metadata.node_count,
                "shown_samples": len(preview["columns"]),
                "total_samples": len(metadata.samples),
                "complete_tables": {
                    unit: ref.model_dump(mode="json") for unit, ref in metadata.tsvs.items()
                },
                "table_reference_base": "parent workflow artifact directory",
                "summary_scope": "all nodes and all biological samples",
            },
            indent=2,
        )
        + "\n"
    )
    files.append(policy)
    with (directory / "figure_sources.tsv").open("a", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(source_rows)
    return files
