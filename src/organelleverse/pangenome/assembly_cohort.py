"""Explicit biological sample membership at the assembly/pangenome boundary.

One assembly result per biological sample and organelle is required. Alternative
configurations and multiple molecules never create additional sample identities.
Missing sequence is unknown, not an observed biological absence.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ..core.artifacts import ArtifactRef
from ..core.result import OrganelleResult
from ..operations.spec import StrictSpecModel
from ._gfa import load_gfa


class AssemblySample(StrictSpecModel):
    sample: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    group: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    species: str = Field(min_length=1)
    result: OrganelleResult
    unified_graph: ArtifactRef | None = None

    @field_validator("result", mode="before")
    @classmethod
    def load_result_contract(cls, value):
        return OrganelleResult.model_validate(value)

    @field_validator("unified_graph", mode="before")
    @classmethod
    def load_graph_contract(cls, value):
        return ArtifactRef.model_validate(value) if value is not None else None


class AssemblyCohortMember(StrictSpecModel):
    sample: str
    group: str
    species: str
    result_id: str
    graph: ArtifactRef | None = None
    state: Literal["failed", "excluded", "graph_only", "representative"]
    representative_paths: tuple[str, ...] = ()
    alternative_paths: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    source_artifacts: tuple[ArtifactRef, ...] = ()

    @field_validator("graph", mode="before")
    @classmethod
    def load_graph_contract(cls, value):
        return ArtifactRef.model_validate(value) if value is not None else None

    @field_validator("source_artifacts", mode="before")
    @classmethod
    def load_sources(cls, value):
        return tuple(ArtifactRef.model_validate(a) for a in value)


class AssemblyCohort(StrictSpecModel):
    schema_version: Literal["organelleverse.assembly-cohort.v1"] = "organelleverse.assembly-cohort.v1"
    organelle: Literal["mitochondrion", "plastid"]
    members: tuple[AssemblyCohortMember, ...]

    @model_validator(mode="after")
    def unique_samples(self) -> AssemblyCohort:
        if not self.members or len({m.sample for m in self.members}) != len(self.members):
            raise ValueError("cohort requires unique biological sample identities")
        return self


def verify_source(artifact: ArtifactRef) -> Path:
    """Use the existing artifact identity contract, including actual source bytes."""
    path = artifact.resolve()
    observed = ArtifactRef.from_path(path, kind=artifact.kind, format=artifact.format,
                                     media_type=artifact.media_type)
    if observed.object_id != artifact.object_id:
        raise ValueError(f"source artifact changed: {path}")
    return path


def from_assemblies(samples: Sequence[AssemblySample]) -> AssemblyCohort:
    """Freeze explicit groups and observed graph/sequence eligibility.

    OVASM publishes ``organelle.gfa``. Other backends must supply a verified
    OV-GFA via ``unified_graph`` after graph assessment/unification. Excluded and
    failed samples remain in the cohort; no completeness threshold is invented.
    """
    if not samples or len({s.sample for s in samples}) != len(samples):
        raise ValueError("provide one assembly result per unique biological sample")
    scope = samples[0].result.scope
    if scope not in {"mitochondrion", "plastid"} or any(s.result.scope != scope for s in samples):
        raise ValueError("a cohort must contain exactly one organelle")
    members = []
    for item in samples:
        result = item.result
        common = dict(sample=item.sample, group=item.group, species=item.species,
                      result_id=result.object_id, source_artifacts=result.artifacts)
        if result.status == "failed":
            members.append(AssemblyCohortMember(**common, state="failed", limitations=("assembly_failed",)))
            continue
        matches = [a for a in result.artifacts if a.kind == "assembly_output"
                   and a.format == "gfa" and a.resolve().name == "organelle.gfa"]
        graph = item.unified_graph or (matches[0] if len(matches) == 1 else None)
        if graph is None:
            members.append(AssemblyCohortMember(**common, state="excluded", limitations=("unified_graph_required",)))
            continue
        parsed = load_gfa(verify_source(graph))
        headers = [field[5:] for h in parsed.headers for field in h.split("\t") if field.startswith("sn:Z:")]
        if headers != [item.sample] or any(p.sample != item.sample for p in parsed.paths.values()):
            raise ValueError(f"graph identity differs from biological sample {item.sample}")
        if not parsed.segments or any(l.overlap != "0M" for l in parsed.links):
            raise ValueError("cohort requires nonempty blunt OV-GFA graphs")
        if any(s.sequence is None for s in parsed.segments.values()):
            raise ValueError("cohort graphs must store every segment sequence")
        representative = tuple(p.name for p in parsed.paths.values() if p.haplotype == "1")
        alternatives = tuple(p.name for p in parsed.paths.values() if p.haplotype != "1")
        limitations = (*result.flags, "absence_not_assessed", "physical_topology_not_certified")
        if not representative:
            limitations += ("representative_unresolved",)
        if alternatives:
            limitations += ("alternative_configurations_not_independent_samples",)
        members.append(AssemblyCohortMember(
            **common, graph=graph, state="representative" if representative else "graph_only",
            representative_paths=representative, alternative_paths=alternatives,
            limitations=tuple(dict.fromkeys(limitations)),
        ))
    return AssemblyCohort(organelle=scope, members=tuple(members))
