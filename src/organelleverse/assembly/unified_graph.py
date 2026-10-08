"""One graph format for every assembly backend (OV-GFA v1), and its route into the pangenome.

HiMT, Oatk, PMAT and GetOrganelle write graphs with different overlaps (0M, ~1 kb, k),
segment names and depth tags. ``ovasm unify`` rewrites any of them as OV-GFA v1:

- blunt (0M) links: overlaps are cut out once, splitting segments rather than copying them;
- segments named ``<sample>.<name>[.<piece>]`` so graphs of many samples never collide;
- ``dp:f`` depth whatever tag the backend used (the header records the rule);
- ``ev:i`` read support on links, from :func:`assess_assembly_graph`;
- PanSN P lines ``<sample>#<hap>#<molecule>``: haplotype 1 is the representative
  linearization, 2.. are near-tied alternative configurations (written only when the
  linearization is not decisive).

These are the pangenome module's own requirements (zero overlaps, P paths, stored sequences),
so the graph passes ``load_gfa``, QC and the conversion/overview prechecks as is.
:func:`export_representatives` writes haplotype 1 of each sample as FASTA, the input of the
pangenome construction workflow.

Typical use::

    assessment = assess_assembly_graph(result, reads, out_dir=Path("evidence"))
    unified = unify_assembly_graph(
        result, sample="NIP", out_dir=Path("unified"), assessment=assessment
    )
    export_representatives([unified.gfa, other.gfa], Path("pan_input"))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .._ovasm import run_unify
from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from .graph_assessment import GraphAssessment, graph_of

__all__ = ["UnifiedGraph", "export_representatives", "unify_assembly_graph"]

Organelle = Literal["mitochondrion", "plastid"]


@dataclass(frozen=True)
class UnifiedGraph:
    """An OV-GFA v1 graph and the ``ovasm unify`` report that produced it."""

    gfa: Path
    report: dict[str, Any]


def unify_assembly_graph(
    source: OrganelleResult | Path,
    *,
    sample: str,
    out_dir: Path,
    assessment: GraphAssessment | None = None,
    backend: str | None = None,
    organelle: Organelle | None = None,
    max_alternatives: int = 3,
) -> UnifiedGraph:
    """Rewrite an assembly graph as OV-GFA v1.

    ``source`` is an ``assemble()`` result (backend and organelle are read from it) or a GFA
    path. With ``assessment`` (from :func:`assess_assembly_graph` on the same graph), links
    carry read support and the linearization becomes PanSN paths; without it the graph has no
    paths, so it can be inspected and converted but not used for construction or PAV.
    """
    if isinstance(source, OrganelleResult):
        gfa = graph_of(source).resolve()
        provenance = source.provenance
        backend = backend or (provenance.actual_backend if provenance is not None else "")
        if organelle is None and source.scope == "mitochondrion":
            organelle = "mitochondrion"
        elif organelle is None and source.scope == "plastid":
            organelle = "plastid"
    else:
        gfa = source.resolve()
    if assessment is not None and assessment.gfa.resolve() != gfa:
        raise OrganelleInputError(
            code="assembly.assessment_mismatch",
            message="the assessment was made on a different graph",
            details={"graph": str(gfa), "assessed_graph": str(assessment.gfa)},
        )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_gfa = out_dir / f"{sample}.ov.gfa"
    report = run_unify(
        gfa,
        sample=sample,
        backend=backend or "unknown",
        organelle=organelle or "mitochondrion",
        out_gfa=out_gfa,
        out_json=out_dir / f"{sample}.unify.json",
        evidence_json=assessment.evidence_json if assessment is not None else None,
        linearize_json=assessment.linearization_json if assessment is not None else None,
        max_alternatives=max_alternatives,
    )
    return UnifiedGraph(gfa=out_gfa, report=report)


def export_representatives(graphs: Sequence[Path], out_dir: Path) -> list[Path]:
    """Write haplotype 1 of every OV-GFA as ``<sample>.fa``, for pangenome construction.

    Every graph must carry a representative path (``<sample>#1#<molecule>``) and samples must
    be distinct; the returned files form the ``dataset_path`` directory of the workflow.
    """
    from ..pangenome.graph import path_sequences

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: dict[str, Path] = {}
    for gfa in graphs:
        sequences = path_sequences(gfa)
        firsts = [name for name in sequences if name.split("#")[1:2] == ["1"]]
        if len(firsts) != 1:
            raise OrganelleInputError(
                code="assembly.representative_missing",
                message="an OV-GFA must carry exactly one haplotype-1 path to export",
                details={"graph": str(gfa), "paths": sorted(sequences)},
            )
        sample = firsts[0].split("#")[0]
        if sample in seen:
            raise OrganelleInputError(
                code="assembly.duplicate_sample",
                message="two graphs use the same sample name",
                details={"sample": sample, "graphs": [str(seen[sample]), str(gfa)]},
            )
        seen[sample] = gfa
        target = out_dir / f"{sample}.fa"
        sequence = sequences[firsts[0]]
        lines = [sequence[i : i + 80] for i in range(0, len(sequence), 80)]
        target.write_text(f">{sample}\n" + "\n".join(lines) + "\n", encoding="ascii")
        written.append(target)
    return written
