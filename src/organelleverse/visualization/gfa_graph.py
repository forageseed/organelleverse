"""Bandage-backed GFA assembly-graph visualization."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.data import OrganelleData
from ..core.result import OrganelleResult
from .plot_object import failed_plot_result, plot_result

Runner = Callable[[list[str], dict[str, str]], Any]

_BANDAGE_CANDIDATES = (
    "Bandage",
    "BandageNG",
    "bandage-ng",
    "Bandage-NG",
    "bandage",
)


@dataclass(frozen=True)
class GfaSegment:
    name: str
    length: int
    depth: float


@dataclass(frozen=True)
class GfaEdge:
    source: str
    target: str
    source_orient: str
    target_orient: str


@dataclass(frozen=True)
class GfaGraph:
    segments: tuple[GfaSegment, ...]
    edges: tuple[GfaEdge, ...]
    paths: tuple[str, ...] = ()


def plot_gfa_graph(
    gfa: str | Path | OrganelleData,
    *,
    organelle: str = "mito",
    executable: str | Path | None = None,
    runner: Runner | None = None,
    labels: tuple[str, ...] = ("name", "length", "depth"),
    color_by: str = "depth",
    width: int | None = None,
    height: int = 1000,
    font_size: int | None = None,
    node_width: float | None = None,
    edge_width: float | None = None,
    edge_length: float | None = None,
    outline: float | None = None,
    node_length: float | None = None,
    min_node_length: float | None = None,
    text_outline: float | None = None,
    center_labels: bool = False,
    single_arrowheads: bool = False,
    linear: bool = False,
    extra_args: tuple[str, ...] = (),
    title: str = "",
    layout: str = "bandage",
    dpi: int | None = None,
) -> OrganelleResult:
    """Render a GFA graph with Bandage/Bandage-NG (compute-only).

    The image layout and drawing are delegated to the professional Bandage
    graph viewer at write time. OrganelleVerse resolves inputs, resolves the
    executable, parses graph metrics, and returns a structured plot; the
    Bandage subprocess runs when the plot is materialized via
    ``ov.write(plot, output)``. A missing executable returns a failed result
    at compute time.
    """
    if layout != "bandage":
        raise ValueError("layout must be 'bandage'")
    if dpi is not None and dpi <= 0:
        raise ValueError("dpi must be greater than zero")
    gfa_path, resolved_organelle = _resolve_gfa_input(gfa, organelle)

    exe = _find_bandage(executable, runner=runner)
    if exe is None:
        return failed_plot_result(
            "plot_gfa_graph",
            organelle=resolved_organelle,
            code="visualization.missing_bandage",
            message=(
                "Bandage/Bandage-NG executable not found. Install with "
                "`mamba install -c bioconda bandage` or pass executable=..."
            ),
            method="bandage",
            suggested_action={
                "install": "mamba install -c bioconda bandage",
                "parameter": "executable",
            },
            flags=("missing_bandage",),
        )

    graph = read_gfa_graph(gfa_path)
    metrics = _graph_metrics(graph)
    style = dict(
        labels=labels,
        color_by=color_by,
        width=width,
        height=height,
        font_size=font_size,
        node_width=node_width,
        edge_width=edge_width,
        edge_length=edge_length,
        outline=outline,
        node_length=node_length,
        min_node_length=min_node_length,
        text_outline=text_outline,
        center_labels=center_labels,
        single_arrowheads=single_arrowheads,
        linear=linear,
        extra_args=extra_args,
    )

    def _render(path: str | Path) -> Path:
        return _render_gfa_graph(exe, gfa_path, path, runner=runner, title=title, **style)

    return plot_result(
        "plot_gfa_graph",
        _render,
        organelle=resolved_organelle,
        method="bandage",
        metrics={
            **metrics,
            "method": "bandage",
            "renderer": Path(exe).name,
            "title": title,
            "layout": layout,
            "dpi": dpi,
        },
        key_findings=(
            {"metric": "nodes", "value": metrics["nodes"]},
            {"metric": "edges", "value": metrics["edges"]},
            {"metric": "renderer", "value": Path(exe).name},
        ),
        flags=("graph_rendered", "third_party_renderer"),
        summary=(
            f"Bandage will render GFA graph with {metrics['nodes']} segment(s) "
            f"and {metrics['edges']} link(s)."
        ),
    )


def _render_gfa_graph(
    executable: str,
    gfa_path: Path,
    output: str | Path,
    *,
    runner: Runner | None,
    title: str,
    **style: Any,
) -> Path:
    """Private path-taking renderer backing :func:`plot_gfa_graph`.

    Runs Bandage to materialize ``output``. Writer-time failures (nonzero exit
    or missing output) raise so ``ov.write`` leaves no partial destination.
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = _build_bandage_image_argv(executable, gfa_path, out, **style)
    env = _bandage_env()
    run_result = _run_bandage(argv, env=env, runner=runner)
    return_code = _return_code(run_result)
    if return_code != 0:
        raise RuntimeError(_failure_summary(run_result))
    if not out.exists():
        raise RuntimeError(f"Bandage exited successfully but did not create {out}.")
    return out


def read_gfa_graph(path: str | Path) -> GfaGraph:
    """Parse a GFA file into lightweight metrics records using gfapy."""
    import gfapy

    gfa = gfapy.Gfa.from_file(str(path))
    segments: list[GfaSegment] = []
    for seg in gfa.segments:
        seq = str(seg.sequence) if seg.sequence and seg.sequence != "*" else ""
        length_tag = _line_number(seg, "LN")
        length = int(length_tag) if length_tag is not None else len(seq)
        depth = _line_number(seg, "DP", "KC", "RC") or 0.0
        segments.append(GfaSegment(name=str(seg.name), length=length, depth=float(depth)))

    edges: list[GfaEdge] = []
    for link in gfa.dovetails:
        edges.append(
            GfaEdge(
                source=str(link.from_segment.name),
                target=str(link.to_segment.name),
                source_orient=str(link.from_orient),
                target_orient=str(link.to_orient),
            )
        )

    paths = tuple(str(path_line.name) for path_line in gfa.paths)
    return GfaGraph(tuple(segments), tuple(edges), paths)


def _resolve_gfa_input(
    gfa: str | Path | OrganelleData,
    organelle: str,
) -> tuple[Path, str]:
    if isinstance(gfa, OrganelleData):
        graph = gfa.artifacts.get("gfa") or gfa.artifacts.get("graph")
        if graph is None:
            raise ValueError("OrganelleData input needs a 'gfa' or 'graph' artifact.")
        return graph.resolve(), str(gfa.metadata.get("organelle", organelle))
    return Path(gfa), organelle


def _find_bandage(
    executable: str | Path | None,
    *,
    runner: Runner | None,
) -> str | None:
    if executable is not None:
        exe = str(executable)
        if runner is not None:
            return exe
        if os.sep in exe:
            return exe if Path(exe).exists() else None
        return shutil.which(exe)
    for candidate in _BANDAGE_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _build_bandage_image_argv(
    executable: str,
    gfa_path: Path,
    output: Path,
    *,
    labels: tuple[str, ...],
    color_by: str,
    width: int | None,
    height: int,
    font_size: int | None,
    node_width: float | None,
    edge_width: float | None,
    edge_length: float | None,
    outline: float | None,
    node_length: float | None,
    min_node_length: float | None,
    text_outline: float | None,
    center_labels: bool,
    single_arrowheads: bool,
    linear: bool,
    extra_args: tuple[str, ...],
) -> list[str]:
    argv = [executable, "image", str(gfa_path), str(output)]
    if height:
        argv.extend(["--height", str(height)])
    if width is not None:
        argv.extend(["--width", str(width)])
    if "name" in labels:
        argv.append("--names")
    if "length" in labels:
        argv.append("--lengths")
    if "depth" in labels:
        argv.append("--depth")
    if color_by == "depth":
        argv.extend(["--colour", "depth"])
    elif color_by == "uniform":
        argv.extend(["--colour", "uniform"])
    _add_option(argv, "--fontsize", font_size)
    _add_option(argv, "--nodewidth", node_width)
    _add_option(argv, "--edgewidth", edge_width)
    _add_option(argv, "--edgelen", edge_length)
    _add_option(argv, "--outline", outline)
    _add_option(argv, "--nodelen", node_length)
    _add_option(argv, "--minnodlen", min_node_length)
    _add_option(argv, "--toutline", text_outline)
    if center_labels:
        argv.append("--centre")
    if single_arrowheads:
        argv.append("--singlearr")
    if linear:
        argv.append("--linear")
    argv.extend(extra_args)
    return argv


def _add_option(argv: list[str], flag: str, value: float | int | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _bandage_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


def _run_bandage(
    argv: list[str],
    *,
    env: dict[str, str],
    runner: Runner | None,
) -> Any:
    if runner is not None:
        return runner(argv, env)
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _return_code(run_result: Any) -> int:
    if run_result is None:
        return 0
    if isinstance(run_result, int):
        return run_result
    return int(getattr(run_result, "returncode", 0))


def _failure_summary(run_result: Any) -> str:
    stderr = str(getattr(run_result, "stderr", "") or "").strip()
    stdout = str(getattr(run_result, "stdout", "") or "").strip()
    message = stderr or stdout or "Bandage rendering failed."
    return message[:1000]


def _line_number(line: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(line, name, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _graph_metrics(graph: GfaGraph) -> dict[str, Any]:
    degree = {s.name: 0 for s in graph.segments}
    for edge in graph.edges:
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1
    return {
        "nodes": len(graph.segments),
        "edges": len(graph.edges),
        "paths": len(graph.paths),
        "branch_nodes": sum(1 for d in degree.values() if d > 2),
        "max_length": max((s.length for s in graph.segments), default=0),
        "max_depth": max((s.depth for s in graph.segments), default=0.0),
    }
