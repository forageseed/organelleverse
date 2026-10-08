"""Windowed graph-shared sequence coverage, not pairwise nucleotide identity."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..core.artifacts import ArtifactRef
from .graph import _sequence, load_gfa, path_steps


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def graph_shared_windows(
    gfa_path: str | Path, *, bin_size: int = 1000, max_rows: int = 1_000_000
) -> dict[str, Any]:
    """Fraction of each reference window covered by nodes in another sample.

    Sample identity is the parser's explicit W/PanSN grouping. Presence is OR
    across the comparison sample's paths/molecules; copies do not add coverage.
    Reference spans are unioned before binning, including overlapping nodes and
    repeated visits. Bins start at native W SeqStart (P: zero), and remain local
    to each recorded path. Missing sequence is allowed for known-length 0M paths;
    nonzero overlaps require stored sequences that prove an exact match.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (bin_size, max_rows)):
        raise ValueError("bin_size and max_rows must be positive integers")
    graph = load_gfa(gfa_path)
    steps = path_steps(gfa_path)
    if not graph.paths:
        raise ValueError("Graph-shared windows require recorded sample paths")
    samples = sorted({record.sample for record in graph.paths.values()})
    if len(samples) < 2:
        raise ValueError("Graph-shared windows require at least two biological samples")
    sample_nodes: dict[str, set[str]] = defaultdict(set)
    bounds = {}
    for name, record in graph.paths.items():
        if record.kind == "W" and record.start is None:
            raise ValueError(f"Native path coordinates are unknown for {name}")
        sample_nodes[record.sample].update(node for node, _ in record.steps)
        if any(step["overlap_previous"] for step in steps[name]):
            _sequence(graph, record)
        start = record.start or 0
        bounds[name] = (start, max((s["end"] for s in steps[name]), default=start))
    row_count = sum(
        (end - start + bin_size - 1) // bin_size * (len(samples) - 1)
        for start, end in bounds.values()
    )
    if row_count > max_rows:
        raise ValueError(
            f"Graph-shared window output needs {row_count} rows, exceeding max_rows={max_rows}; increase bin_size or select a subgraph"
        )
    rows = []
    for name, record in graph.paths.items():
        start, end = bounds[name]
        for sample in samples:
            if sample == record.sample:
                continue
            shared = _union(
                [
                    (step["start"], step["end"])
                    for step in steps[name]
                    if step["node"] in sample_nodes[sample]
                ]
            )
            cursor = 0
            for left in range(start, end, bin_size):
                right = min(end, left + bin_size)
                while cursor < len(shared) and shared[cursor][1] <= left:
                    cursor += 1
                covered = 0
                index = cursor
                while index < len(shared) and shared[index][0] < right:
                    covered += max(0, min(right, shared[index][1]) - max(left, shared[index][0]))
                    index += 1
                rows.append(
                    {
                        "path": name,
                        "reference_sample": record.sample,
                        "comparison_sample": sample,
                        "start": left,
                        "end": right,
                        "shared_bp": covered,
                        "window_bp": right - left,
                        "shared_fraction": covered / (right - left),
                    }
                )
    return {
        "source_graph": ArtifactRef.from_path(
            gfa_path, kind="pangenome_graph", format="gfa"
        ).model_dump(mode="json"),
        "metric": "graph_shared_fraction",
        "bin_size": bin_size,
        "samples": samples,
        "path_bounds": bounds,
        "rows": rows,
        "coordinate_semantics": "0-based half-open; bins anchored at each path's native start; W intervals remain separate",
        "interpretation": "Fraction of reference path bases covered by graph nodes present in another biological sample; not nucleotide identity, homology, or read coverage",
        "sample_semantics": "Presence union across comparison-sample paths; reference sample excluded",
    }


def render_shared_windows(
    data: dict[str, Any],
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("svg",),
    max_paths: int = 100,
    workers: int = 1,
) -> dict[str, str]:
    """Export exact rows plus one labelled profile per path, without truncation."""
    from .visualization import _table

    if workers < 1:
        raise ValueError("workers must be positive")
    if data.get("metric") != "graph_shared_fraction":
        raise ValueError("Expected graph_shared_fraction data")
    if not formats or any(fmt not in {"svg", "pdf", "png"} for fmt in formats):
        raise ValueError("Supported formats are svg, pdf, png")
    by_path: dict[str, list[dict]] = defaultdict(list)
    for row in data["rows"]:
        by_path[row["path"]].append(row)
    if len(by_path) > max_paths:
        raise ValueError(
            f"Profile rendering exceeds max_paths={max_paths}; no paths were silently omitted"
        )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    columns = [
        "path",
        "reference_sample",
        "comparison_sample",
        "start",
        "end",
        "shared_bp",
        "window_bp",
        "shared_fraction",
    ]
    outputs = {
        "table": _table(
            directory / "graph_shared_windows.tsv",
            columns,
            [[row[key] for key in columns] for row in data["rows"]],
        )
    }
    jobs = [(index, name, rows, directory, formats) for index, (name, rows) in enumerate(by_path.items(), 1)]
    if workers == 1 or len(jobs) < 2:
        for job in jobs:
            outputs.update(_render_shared_path(job))
    else:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        # Separate processes keep Matplotlib state isolated. map preserves
        # source order; every worker owns distinct filenames.
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")) as pool:
            for rendered in pool.map(_render_shared_path, jobs):
                outputs.update(rendered)
    return outputs


def _render_shared_path(job):
    from .visualization import _figure
    index, name, rows, directory, formats = job
    outputs = {}
    figure, axes = _figure(9, 4)
    samples: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        samples[row["comparison_sample"]].append(row)
    for sample, windows in samples.items():
        windows.sort(key=lambda row: row["start"])
        axes.stairs(
            [row["shared_fraction"] for row in windows],
            [row["start"] for row in windows] + [windows[-1]["end"]],
            label=sample,
            baseline=None,
        )
    axes.set(
        xlabel="Reference path coordinate (bp)",
        ylabel="Graph-shared fraction",
        ylim=(0, 1.02),
        title=name,
    )
    axes.legend(title="Comparison sample", loc="best")
    for fmt in dict.fromkeys(formats):
        target = directory / f"graph_shared_path_{index:04d}.{fmt}"
        figure.savefig(target, format=fmt, dpi=180)
        outputs[f"path_{index:04d}_{fmt}"] = str(target)
    figure.clear()
    return outputs


def render_shared_window_batches(data, output_dir, *, formats=("svg",), max_paths=100, workers=1):
    """Render every reference path in bounded batches, retaining every sample.

    Each batch has an exact source TSV. Small cohorts keep their existing
    filenames; larger cohorts use separate numbered directories.
    """
    if max_paths < 1:
        raise ValueError("max_paths must be positive")
    by_path = defaultdict(list)
    for row in data["rows"]:
        by_path[row["path"]].append(row)
    names = list(by_path)
    if len(names) <= max_paths:
        yield render_shared_windows(data, output_dir, formats=formats, max_paths=max_paths, workers=workers)
        return
    for offset in range(0, len(names), max_paths):
        selected = names[offset:offset + max_paths]
        batch = {**data, "rows": [row for name in selected for row in by_path[name]]}
        yield render_shared_windows(
            batch, Path(output_dir) / f"shared_profiles_{offset // max_paths + 1:04d}",
            formats=formats, max_paths=max_paths, workers=workers,
        )
