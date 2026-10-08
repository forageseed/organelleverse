"""Publication panels of recorded stage resources, with exact tabular sources."""

from __future__ import annotations

import csv
from pathlib import Path

from .visualization import _figure, _save_figures


def render_stage_benchmark(record: dict, directory: Path, *, formats=("svg",)) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    # Report generation cannot include its own completed timing. Do not invent
    # or retroactively overwrite a value in an already published artifact.
    stages = [
        stage
        for stage in record["stages"]
        if stage["status"] == "completed" and stage["name"] != "report"
    ]
    source = directory / "stage_benchmark.tsv"
    fields = (
        "name",
        "status",
        "wall_seconds",
        "cpu_seconds",
        "peak_memory_bytes",
        "cpu_scope",
        "memory_scope",
    )
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows([stage.get(field) for field in fields] for stage in stages)
    outputs = [source]
    panels = []
    for metric, label, scale in (
        ("wall_seconds", "Elapsed time (s)", 1),
        ("cpu_seconds", "Stage CPU time (s)", 1),
        ("peak_memory_bytes", "OS maximum RSS (MiB)", 1048576),
    ):
        measured = [stage for stage in stages if stage.get(metric) is not None]
        if not measured:
            continue
        figure, axes = _figure(8, 4)
        axes.barh(
            [stage["name"] for stage in measured],
            [stage[metric] / scale for stage in measured],
            color="#0072B2",
        )
        axes.invert_yaxis()
        axes.set(title="Measured analysis stages before report generation")
        axes.set_xlabel(
            label + "\nIncludes stage startup; RSS is an OS maximum, not a process-tree total.",
            fontsize=9,
        )
        rendered = _save_figures(figure, directory / ("stage_" + metric), formats)
        panels.extend(Path(path) for path in rendered.values())
    outputs.extend(panels)
    manifest = directory / "figure_sources.tsv"
    existed = manifest.exists()
    with manifest.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if not existed:
            writer.writerow(["figure", "source"])
        writer.writerows((path.name, source.name) for path in panels)
    outputs.append(manifest)
    return outputs
