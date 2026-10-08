"""Read evidence and a representative sequence for an assembled graph.

After ``assemble()``, long reads can check the graph itself: ``ovasm evidence`` counts the
reads that cross each link, and ``ovasm linearize`` threads one sequence through the
read-supported phased paths. A link no read crosses is a likely assembly error, and a
linearization that is not decisive means the reads support several configurations about
equally (common in plant mitochondria), so the sequence is one representative of them.

Typical use::

    result = assemble(recruited, organelle="mitochondrion", method="himt")
    report = assess_assembly_graph(
        result, [Path("recruit/mitochondrion.fastq")], preset="hifi", out_dir=Path("evidence")
    )
    report.flags  # e.g. ("graph_unsupported_links",)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .._ovasm import run_evidence, run_linearize
from ..core.errors import OrganelleInputError
from ..core.result import Finding, OrganelleResult

__all__ = ["GraphAssessment", "assess_assembly_graph", "findings_for", "flags_for", "graph_of"]

#: Flag names, in the order they are reported.
UNSUPPORTED_LINKS = "graph_unsupported_links"
NO_LINEAR_PATH = "linearization_unresolved"
UNDECIDED = "linearization_undecided"
REPEATS_NOT_PLACED = "linearization_repeats_not_placed"


@dataclass(frozen=True)
class GraphAssessment:
    """Evidence and linearization of one assembly graph."""

    gfa: Path
    evidence: dict[str, Any]
    evidence_json: Path
    linearization: dict[str, Any] | None
    linearization_json: Path | None
    #: Best linearization as FASTA; ``None`` when no path visits every unique segment.
    linear_fasta: Path | None
    findings: tuple[Finding, ...]
    flags: tuple[str, ...]


def graph_of(result: OrganelleResult) -> Path:
    """The assembly graph (GFA) published with an assembly result.

    Backends that also publish their raw graph (e.g. Oatk's ``backend_full_graph.gfa``) are
    resolved to the normalized ``assembly.gfa``.
    """
    graphs = [
        Path(a.uri) for a in result.artifacts if a.kind == "assembly_output" and a.format == "gfa"
    ]
    primary = [g for g in graphs if g.name == "assembly.gfa"]
    if len(primary) == 1:
        return primary[0]
    if len(graphs) != 1:
        raise OrganelleInputError(
            code="assembly.graph_unavailable",
            message="the assembly result must carry exactly one GFA assembly output",
            details={"status": result.status, "gfa_outputs": [str(g) for g in graphs]},
        )
    return graphs[0]


def findings_for(evidence: dict[str, Any], linear: dict[str, Any] | None) -> list[Finding]:
    """Findings from an ``evidence.json`` report and, if run, its ``linearize.json``."""
    total = len(evidence["links"])
    out = [
        Finding(
            code="assembly.graph_links_supported",
            metric="links_supported",
            value=evidence["links_supported"],
            unit=f"of {total} links",
        )
    ]
    out += [
        Finding(
            code="assembly.graph_link_unsupported",
            metric=f"{link['from']} -> {link['to']}",
            value=0,
            unit="reads",
        )
        for link in evidence["links"]
        if link["reads"] == 0
    ]
    out += [
        Finding(
            code="assembly.graph_branch_minority",
            metric=branch["from"],
            value=round(branch["minority_fraction"], 4),
            unit="fraction",
        )
        for branch in evidence["branches"]
        if branch["minority_fraction"] is not None
    ]
    if linear is not None:
        out += [
            Finding(code="assembly.graph_anchor_unbridged", metric=name, value=0, unit="bridges")
            for name in linear.get("unbridged_anchors", [])
        ]
    if linear is None or linear["best"] is None:
        return out
    best = linear["best"]
    lin = [
        ("molecules", linear.get("molecules", 1), "molecules"),
        ("path", best["path"], ""),
        ("length", best["length"], "bp"),
        ("circular", best["circular"], ""),
        ("decisive", bool(linear["decisive"]), ""),
    ]
    if linear["margin_ln"] is not None:
        lin.append(("margin_ln", round(linear["margin_ln"], 4), "ln likelihood"))
    out += [
        Finding(code="assembly.graph_linearization", metric=m, value=v, unit=u) for m, v, u in lin
    ]
    return out


def flags_for(evidence: dict[str, Any], linear: dict[str, Any] | None) -> list[str]:
    """Flags for problems a caller should act on (see the module constants)."""
    flags: list[str] = []
    if evidence["links_supported"] < len(evidence["links"]):
        flags.append(UNSUPPORTED_LINKS)
    if linear is not None:
        # some part of the graph no cover of a few molecules explains (see its path ends)
        if linear["best"] is None or linear.get("unsolved_components", 0):
            flags.append(NO_LINEAR_PATH)
        elif not linear["decisive"]:
            flags.append(UNDECIDED)
        if linear["repeats_not_placed"]:
            flags.append(REPEATS_NOT_PLACED)
    return flags


def assess_assembly_graph(
    source: OrganelleResult | Path,
    reads: Sequence[Path],
    *,
    out_dir: Path,
    preset: Literal["hifi", "ont"] = "hifi",
    linearize: bool = True,
    min_anchor: int = 500,
    min_reads: int = 2,
    bootstrap: int = 1000,
    threads: int | None = None,
) -> GraphAssessment:
    """Score an assembly graph against long reads and thread a representative sequence.

    ``source`` is an ``assemble()`` result or a GFA path. ``reads`` should be the organelle
    reads (e.g. from ``recruit_long_reads``); whole-genome reads work but scan slower.
    Writes ``evidence.json``, ``linearize.json`` and ``linear.fasta`` into ``out_dir``.
    """
    gfa = (graph_of(source) if isinstance(source, OrganelleResult) else source).resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = run_evidence(
        gfa,
        [p.resolve() for p in reads],
        out_json=out_dir / "evidence.json",
        preset=preset,
        min_anchor=min_anchor,
        bootstrap=bootstrap,
        threads=threads,
    )
    linear = None
    fasta = out_dir / "linear.fasta"
    fasta.unlink(missing_ok=True)  # never report a stale sequence from an earlier run
    if linearize:
        linear = run_linearize(
            gfa,
            out_dir / "evidence.json",
            out_fasta=fasta,
            out_json=out_dir / "linearize.json",
            min_reads=min_reads,
        )
    return GraphAssessment(
        gfa=gfa,
        evidence=evidence,
        evidence_json=out_dir / "evidence.json",
        linearization=linear,
        linearization_json=out_dir / "linearize.json" if linear is not None else None,
        linear_fasta=fasta if fasta.is_file() else None,
        findings=tuple(findings_for(evidence, linear)),
        flags=tuple(flags_for(evidence, linear)),
    )
