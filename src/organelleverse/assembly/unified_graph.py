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
import json
import re
import shutil
import tempfile
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
    """Write every configuration-1 molecule as ``<sample>.fa``.

    Every graph must carry a representative path (``<sample>#1#<molecule>``) and samples must
    be distinct; configuration numbers are not evidence of biological haplotypes.
    ``representatives.json`` preserves the original path and molecule identities.
    All inputs are checked before publishing a new output directory.
    """
    from ..pangenome.graph import path_sequences

    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"representative destination already exists: {out_dir}")
    prepared: list[tuple[str, list[tuple[str, str]]]] = []
    records: list[dict[str, object]] = []
    seen: dict[str, Path] = {}
    for gfa in graphs:
        sequences = path_sequences(gfa)
        firsts = [name for name in sequences if name.split("#")[1:2] == ["1"]]
        if not firsts:
            raise OrganelleInputError(
                code="assembly.representative_missing",
                message="an OV-GFA must carry configuration-1 paths to export",
                details={"graph": str(gfa), "paths": sorted(sequences)},
            )
        sample = firsts[0].split("#")[0]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample) or any(
            name.split("#")[0] != sample for name in sequences
        ):
            raise OrganelleInputError(
                code="assembly.invalid_sample",
                message="each graph must have one filename-safe sample identity",
                details={"graph": str(gfa)},
            )
        if sample in seen:
            raise OrganelleInputError(
                code="assembly.duplicate_sample",
                message="two graphs use the same sample name",
                details={"sample": sample, "graphs": [str(seen[sample]), str(gfa)]},
            )
        seen[sample] = gfa
        molecules = []
        for name in firsts:
            record_id = sample if len(firsts) == 1 else name
            molecules.append((record_id, sequences[name]))
            records.append({"sample": sample, "fasta": f"{sample}.fa",
                            "record_id": record_id, "source_graph": str(gfa.resolve()),
                            "source_path": name, "configuration": 1,
                            "molecule": name.split("#", 2)[2], "length": len(sequences[name])})
        prepared.append((sample, molecules))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        for sample, molecules in prepared:
            with (staging / f"{sample}.fa").open("w", encoding="ascii") as handle:
                for record_id, sequence in molecules:
                    handle.write(f">{record_id}\n")
                    handle.writelines(sequence[i:i + 80] + "\n" for i in range(0, len(sequence), 80))
        (staging / "representatives.json").write_text(
            json.dumps({"schema_version": "organelleverse.representatives.v1", "records": records}, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return [out_dir / f"{sample}.fa" for sample, _ in prepared]
