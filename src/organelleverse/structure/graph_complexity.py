"""Assembly-graph complexity of an organelle GFA.

Classes follow Brown et al. 2026 (303 plant mitogenomes): ``simple`` is one
segment (linear or circular), ``complex`` is one connected graph of more than
one segment, ``multicomponent`` is more than one weakly connected component.
Alongside the class: segment/link/component counts, total and per-component
length, dead ends (segment ends without a link), double forks (segments
branching on both ends - the repeat nodes whose recombination gives
alternative conformations) and self-loops.

Only S and L lines are read, and overlaps are not interpreted, so any GFA1
graph (ovasm, oatk, PMAT, GetOrganelle, Flye, ...) works.

A raw assembler graph often holds more than the mitogenome (oatk's whole
graph also carries the plastome and stray components). Given ``sequences``
(e.g. the assembler's mitochondrial FASTA), only components holding a
segment whose sampled 31-mers fall mostly in those sequences are kept.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from organelleverse._public_validation import validated_public

from .._bio import read_fasta
from ..core.errors import OrganelleInputError
from ..core.result import Finding, OperationSuggestion, OrganelleResult
from ..inputs import Graph
from ..sequences import as_path
from .structure import _OPERATION_VERSION, _provenance, _sha256_file

__all__ = ["graph_complexity", "graph_complexity_metrics", "select_components", "write_subgraph"]


_K = 31
_PROBE_STEP = 50


def _read_gfa(
    path: Path, sequences: dict[str, str] | None = None
) -> tuple[dict[str, int], list[tuple[str, str, str, str]]]:
    segments: dict[str, int] = {}
    links: list[tuple[str, str, str, str]] = []
    for raw in path.read_text().splitlines():
        fields = raw.split("\t")
        if fields[0] == "S" and len(fields) >= 3:
            length = None
            for tag in fields[3:]:
                if tag.startswith("LN:i:"):
                    length = int(tag[5:])
            if length is None:
                length = 0 if fields[2] == "*" else len(fields[2])
            segments[fields[1]] = length
            if sequences is not None:
                sequences[fields[1]] = "" if fields[2] == "*" else fields[2].upper()
        elif fields[0] == "L" and len(fields) >= 5:
            links.append((fields[1], fields[2], fields[3], fields[4]))
    if not segments:
        raise OrganelleInputError(
            code="structure.graph_empty",
            message="The GFA file declares no segments.",
            details={"path": str(path)},
        )
    for source, _, target, _ in links:
        for name in (source, target):
            if name not in segments:
                raise OrganelleInputError(
                    code="structure.graph_dangling_link",
                    message="A GFA link names a segment that is not declared.",
                    details={"segment": name},
                )
    return segments, links


def _reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def select_components(
    segment_sequences: dict[str, str],
    links: list[tuple[str, str, str, str]],
    reference: str,
    *,
    min_kmer_fraction: float = 0.5,
) -> set[str]:
    """Segments of the components that hold a segment found in ``reference``.

    A segment is found when at least ``min_kmer_fraction`` of its 31-mers,
    sampled every 50 bp, occur in ``reference`` on either strand.
    """
    reference = reference.upper()
    kmers: set[str] = set()
    for strand in (reference, _reverse_complement(reference)):
        kmers.update(strand[i : i + _K] for i in range(len(strand) - _K + 1))
    found = set()
    for name, seq in segment_sequences.items():
        if len(seq) < _K:
            continue
        probes = [seq[i : i + _K] for i in range(0, len(seq) - _K + 1, _PROBE_STEP)]
        if sum(probe in kmers for probe in probes) >= min_kmer_fraction * len(probes):
            found.add(name)
    parent = {name: name for name in segment_sequences}

    def root(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for source, _, target, _ in links:
        parent[root(source)] = root(target)
    kept_roots = {root(name) for name in found}
    return {name for name in segment_sequences if root(name) in kept_roots}


def write_subgraph(
    gfa: str | Path,
    sequences: str | Path,
    destination: str | Path,
    *,
    min_kmer_fraction: float = 0.5,
) -> Path:
    """Write the GFA restricted to the components holding ``sequences``.

    Keeps the header, the selected segments, the links between them and the
    paths that stay inside them, so any GFA reader (OV's own graph figure
    included) sees only the mitochondrial components of a raw assembler graph.
    """
    source = Path(as_path(gfa))
    segment_sequences: dict[str, str] = {}
    _, links = _read_gfa(source, segment_sequences)
    reference = "".join(sequence for _, sequence in read_fasta(Path(as_path(sequences))))
    keep = select_components(segment_sequences, links, reference, min_kmer_fraction=min_kmer_fraction)
    if not keep:
        raise OrganelleInputError(
            code="structure.graph_selection_empty",
            message="no graph segment matches the given sequences",
        )
    out = Path(destination)
    with source.open() as handle, out.open("w") as writer:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            kind = fields[0]
            if kind == "H":
                writer.write(line)
            elif kind == "S" and fields[1] in keep:
                writer.write(line)
            elif kind == "L" and len(fields) > 3 and fields[1] in keep and fields[3] in keep:
                writer.write(line)
            elif kind == "P" and len(fields) > 2 and all(
                step.rstrip("+-") in keep for step in fields[2].split(",") if step
            ):
                writer.write(line)
    return out


def graph_complexity_metrics(
    segments: dict[str, int], links: list[tuple[str, str, str, str]]
) -> dict[str, object]:
    """Complexity metrics of a graph given ``{segment: length}`` and GFA L-line tuples."""
    # A link leaves ``source`` at its 3' end when '+', 5' end when '-', and
    # enters ``target`` at its 5' end when '+', 3' end when '-'. A link and its
    # reverse complement are the same adjacency, so ends are deduplicated.
    adjacencies: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    for source, s_orient, target, t_orient in links:
        a = (source, "3" if s_orient == "+" else "5")
        b = (target, "5" if t_orient == "+" else "3")
        adjacencies.add((min(a, b), max(a, b)))
    end_degree: dict[tuple[str, str], int] = defaultdict(int)
    self_loops = 0
    parent = {name: name for name in segments}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for a, b in adjacencies:
        end_degree[a] += 1
        end_degree[b] += 1
        if a[0] == b[0]:
            self_loops += 1
        parent[find(a[0])] = find(b[0])

    components: dict[str, list[str]] = defaultdict(list)
    for name in segments:
        components[find(name)].append(name)
    dead_ends = [
        {"segment": name, "end": end}
        for name in segments
        for end in ("5", "3")
        if end_degree[(name, end)] == 0
    ]
    double_forks = sorted(
        name for name in segments if end_degree[(name, "5")] >= 2 and end_degree[(name, "3")] >= 2
    )
    per_component = []
    for members in components.values():
        member_set = set(members)
        n_links = sum(1 for a, _ in adjacencies if a[0] in member_set)
        open_ends = sum(1 for end in dead_ends if end["segment"] in member_set)
        if len(members) == 1:
            shape = "circular" if open_ends == 0 else "linear" if open_ends == 2 else "partial"
        else:
            shape = "closed_graph" if open_ends == 0 else "open_graph"
        per_component.append(
            {
                "segments": sorted(members),
                "n_segments": len(members),
                "n_links": n_links,
                "length": sum(segments[name] for name in members),
                "dead_ends": open_ends,
                "shape": shape,
            }
        )
    per_component.sort(key=lambda row: (-int(row["length"]), row["segments"][0]))
    n_components = len(per_component)
    if n_components > 1:
        complexity = "multicomponent"
    elif len(segments) == 1:
        complexity = "simple"
    else:
        complexity = "complex"
    mean_segments = len(segments) / n_components
    return {
        "complexity": complexity,
        "n_segments": len(segments),
        "n_links": len(adjacencies),
        "n_components": n_components,
        "total_length": sum(segments.values()),
        "mean_segments_per_component": mean_segments,
        "log_mean_segments_per_component": math.log(mean_segments),
        "n_dead_ends": len(dead_ends),
        "dead_ends": dead_ends,
        "n_double_forks": len(double_forks),
        "double_forks": double_forks,
        "n_self_loops": self_loops,
        "components": per_component,
    }


@validated_public("structure.graph_complexity")
def graph_complexity(
    gfa: str | Path | Graph,
    *,
    sequences: str | Path | None = None,
    min_kmer_fraction: float = 0.5,
) -> OrganelleResult:
    """Classify an organelle assembly graph as simple, complex or multicomponent.

    ``simple``: one segment (circular when its ends join, else linear);
    ``complex``: one connected graph of several segments; ``multicomponent``:
    several disconnected graphs (check small ones for nuclear or contaminant
    sequence before calling them chromosomes). Also reports dead ends (an
    unresolved assembly usually has them; a finished one usually has none),
    double forks (repeat segments branching on both ends) and
    ``log_mean_segments_per_component``, the continuous complexity measure
    used in phylogenetic models.

    ``sequences`` (a FASTA, e.g. the assembler's mitochondrial contigs)
    restricts the count to the components that hold them: oatk's raw graph,
    for one, also carries the plastome and stray components that would
    otherwise make every genome look multicomponent. Dropped components are
    reported under ``selection``.
    """
    gfa = as_path(gfa)
    path = Path(gfa)
    started_at = datetime.now(UTC)
    if not 0 < min_kmer_fraction <= 1:
        raise OrganelleInputError(
            code="structure.graph_selection",
            message="min_kmer_fraction must be in (0, 1]",
        )
    segment_sequences: dict[str, str] | None = {} if sequences is not None else None
    segments, links = _read_gfa(path, segment_sequences)
    selection = None
    if sequences is not None and segment_sequences is not None:
        if not any(segment_sequences.values()):
            raise OrganelleInputError(
                code="structure.graph_no_sequences",
                message="selecting by sequences needs segment sequences in the GFA",
            )
        reference = "".join(sequence for _, sequence in read_fasta(Path(as_path(sequences))))
        keep = select_components(
            segment_sequences, links, reference, min_kmer_fraction=min_kmer_fraction
        )
        if not keep:
            raise OrganelleInputError(
                code="structure.graph_selection_empty",
                message="no graph segment matches the given sequences",
            )
        all_metrics = graph_complexity_metrics(segments, links)
        selection = {
            "segments_total": len(segments),
            "segments_kept": len(keep),
            "components_total": all_metrics["n_components"],
            "length_dropped": sum(length for name, length in segments.items() if name not in keep),
        }
        segments = {name: length for name, length in segments.items() if name in keep}
        links = [link for link in links if link[0] in keep and link[2] in keep]
    metrics = graph_complexity_metrics(segments, links)
    if selection is not None:
        selection["components_dropped"] = selection["components_total"] - metrics["n_components"]
        metrics["selection"] = selection
    flags = [str(metrics["complexity"])]
    if metrics["n_dead_ends"]:
        flags.append("has_dead_ends")
    if metrics["n_double_forks"]:
        flags.append("has_double_forks")
    return OrganelleResult(
        operation_id="structure.graph_complexity",
        operation_version=_OPERATION_VERSION,
        scope="mitochondrion",
        status="ok",
        summary_text=(
            f"{metrics['complexity']} graph: {metrics['n_segments']} segments, "
            f"{metrics['n_links']} links, {metrics['n_components']} component(s), "
            f"{metrics['total_length']:,} bp; {metrics['n_dead_ends']} dead ends, "
            f"{metrics['n_double_forks']} double forks."
        ),
        metrics=metrics,
        findings=(
            Finding(code="structure.graph_complexity", metric="complexity", value=metrics["complexity"]),
            Finding(code="structure.graph_segments", metric="n_segments", value=metrics["n_segments"]),
            Finding(
                code="structure.graph_components", metric="n_components", value=metrics["n_components"]
            ),
        ),
        flags=tuple(flags),
        # The graph itself is drawn by OV (no Bandage needed); with sequences
        # the drawing keeps the same mitochondrial components.
        suggested_operations=(
            OperationSuggestion(
                operation_id="visualization.plot_assembly_graph",
                reason_code="structure.graph_complexity.draw_graph",
                parameter_changes={
                    "gfa": str(path),
                    **({} if sequences is None else {"sequences": str(sequences)}),
                },
            ),
        ),
        provenance=_provenance(
            "structure.graph_complexity",
            parameters={
                "sequences": None if sequences is None else str(sequences),
                "min_kmer_fraction": min_kmer_fraction,
            },
            input_artifact_hashes=(_sha256_file(path),),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
    )
