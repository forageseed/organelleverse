"""Pure static analysis and evidence resolution for assembly QC.

These functions perform no scientific computation and no I/O beyond reading the
declared input files. They validate FASTA/GFA syntax and referential integrity,
and resolve a published assembly :class:`~organelleverse.core.result.OrganelleResult`
into a fully integrity-checked :class:`ResolvedAssemblyEvidence`.

Manifest loading (architectural note): a published assembly Result carries the
canonical manifest bytes in its ``assembly_run_manifest`` artifact and the
loadable full dump in its ``assembly_run_record`` artifact
(see ``assembly/service.py`` ``_write_manifest_evidence``). The canonical form
is a semantic dict and is not directly loadable by
``AssemblyRunManifest.model_validate_json``; the loadable model is therefore read
from ``assembly_run_record`` and then cross-checked against the canonical
artifact, mirroring ``_load_reusable_result``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from organelleverse.assembly.contracts import AssemblyInputPayload
from organelleverse.assembly.manifests import AssemblyRunManifest
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult

from .contracts import (
    AssemblyStatistics,
    GfaSummary,
    InputLibraryEvidence,
    ResolvedAssemblyEvidence,
    SequenceSummary,
)
from .fasta import read_fasta

__all__ = [
    "compute_assembly_statistics",
    "resolve_assembly_evidence",
    "summarize_fasta",
    "validate_gfa",
]


# ---------------------------------------------------------------------------
# FASTA summary
# ---------------------------------------------------------------------------


def summarize_fasta(path: str | Path) -> tuple[SequenceSummary, ...]:
    """Return one strict :class:`SequenceSummary` per FASTA record.

    Identifiers are preserved in file order. GC and ambiguous-base counts use an
    uppercased view of each sequence; the stored sequence is not modified. The
    shared :func:`read_fasta` reader enforces file existence, non-emptiness,
    and unique identifiers.
    """
    summaries: list[SequenceSummary] = []
    for sequence_id, sequence in read_fasta(path):
        upper = sequence.upper()
        length = len(upper)
        gc = upper.count("G") + upper.count("C")
        gc_fraction = gc / length if length else 0.0
        ambiguous_bases = sum(1 for base in upper if base not in "ACGT")
        summaries.append(
            SequenceSummary(
                sequence_id=sequence_id,
                role="primary",
                length=length,
                gc_fraction=gc_fraction,
                ambiguous_bases=ambiguous_bases,
            )
        )
    return tuple(summaries)


def compute_assembly_statistics(path: str | Path) -> AssemblyStatistics:
    """Return descriptive whole-assembly contiguity and composition statistics.

    Pure description: these values are not part of any pass/fail decision.
    ``n50``/``l50`` use the standard contig definition over lengths sorted
    descending. GC and ambiguous-base totals are computed from an uppercased
    view of the FASTA so they are exact (not derived from rounded per-sequence
    fractions).
    """
    lengths: list[int] = []
    total_gc = 0
    total_ambiguous = 0
    total_length = 0
    for _sequence_id, sequence in read_fasta(path):
        upper = sequence.upper()
        length = len(upper)
        lengths.append(length)
        total_length += length
        total_gc += upper.count("G") + upper.count("C")
        total_ambiguous += sum(1 for base in upper if base not in "ACGT")
    n50, l50 = _n50_l50(lengths)
    return AssemblyStatistics(
        sequence_count=len(lengths),
        total_length=total_length,
        largest_sequence_length=max(lengths) if lengths else 0,
        n50=n50,
        l50=l50,
        overall_gc_fraction=(total_gc / total_length) if total_length else 0.0,
        ambiguous_bases=total_ambiguous,
    )


def _n50_l50(lengths: list[int]) -> tuple[int, int]:
    """Standard contig N50/L50 over lengths sorted descending."""
    if not lengths:
        return 0, 0
    ordered = sorted(lengths, reverse=True)
    half = sum(ordered) / 2
    cumulative = 0
    for index, length in enumerate(ordered, start=1):
        cumulative += length
        if cumulative >= half:
            return length, index
    return ordered[-1], len(ordered)


# ---------------------------------------------------------------------------
# GFA validation
# ---------------------------------------------------------------------------


def validate_gfa(path: str | Path) -> GfaSummary:
    """Parse ``S``, ``L``, and ``P`` records and derive topology counts.

    Duplicate segments and unknown link/path references raise a typed
    :class:`OrganelleInputError`. Topology counts (component, branch, tip,
    parallel edge) are derived from the undirected segment graph without any
    circularity heuristic.
    """
    candidate = Path(path)
    segments: dict[str, None] = {}
    links: list[tuple[str, str]] = []
    path_names: list[str] = []
    path_references: list[str] = []
    for raw_line in candidate.read_text().splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        fields = raw_line.split("\t")
        record = fields[0]
        if record == "S":
            name = fields[1] if len(fields) > 1 else ""
            if not name:
                raise OrganelleInputError(
                    code="qc.input_contract_violation",
                    message="GFA segment record is missing a name",
                    details={"path": str(candidate)},
                )
            if name in segments:
                raise OrganelleInputError(
                    code="qc.input_contract_violation",
                    message=f"GFA contains a duplicate segment: {name}",
                    details={"segment": name, "path": str(candidate)},
                )
            segments[name] = None
        elif record == "L":
            if len(fields) < 5:
                raise OrganelleInputError(
                    code="qc.input_contract_violation",
                    message="GFA link record is malformed",
                    details={"path": str(candidate)},
                )
            left, right = fields[1], fields[3]
            links.append((left, right))
        elif record == "P":
            if len(fields) < 3:
                raise OrganelleInputError(
                    code="qc.input_contract_violation",
                    message="GFA path record is malformed",
                    details={"path": str(candidate)},
                )
            path_names.append(fields[1])
            for ref in fields[2].split(","):
                segment_name = ref[:-1] if ref and ref[-1] in "+-" else ref
                if segment_name:
                    path_references.append(segment_name)

    # GFA records are not required to define segments before links/paths.
    # Validate referential integrity only after the complete segment table is known.
    for left, right in links:
        _require_known_segment(left, segments, candidate, "link")
        _require_known_segment(right, segments, candidate, "link")
    for segment_name in path_references:
        _require_known_segment(segment_name, segments, candidate, "path")

    degree: dict[str, int] = {name: 0 for name in segments}
    parallel: dict[frozenset[str], int] = {}
    parent: dict[str, str] = {name: name for name in segments}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for left, right in links:
        degree[left] += 1
        degree[right] += 1
        union(left, right)
        if left != right:
            pair = frozenset((left, right))
            parallel[pair] = parallel.get(pair, 0) + 1

    component_count = len({find(name) for name in segments})
    branch_count = sum(1 for value in degree.values() if value > 2)
    tip_count = sum(1 for value in degree.values() if value == 1)
    parallel_edge_count = sum(1 for count in parallel.values() if count > 1)

    return GfaSummary(
        segment_count=len(segments),
        edge_count=len(links),
        path_count=len(path_names),
        component_count=component_count,
        branch_count=branch_count,
        parallel_edge_count=parallel_edge_count,
        tip_count=tip_count,
        path_names=tuple(path_names),
    )


def _require_known_segment(
    name: str,
    segments: dict[str, None],
    candidate: Path,
    context: Literal["link", "path"],
) -> None:
    if name not in segments:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message=f"GFA {context} references an unknown segment: {name}",
            details={"segment": name, "context": context, "path": str(candidate)},
        )


# ---------------------------------------------------------------------------
# Evidence resolution
# ---------------------------------------------------------------------------


def resolve_assembly_evidence(result: OrganelleResult) -> ResolvedAssemblyEvidence:
    """Resolve and integrity-check the input evidence for one QC run.

    Enforces ``operation_id == "assembly.assemble"``, a non-failed status, the
    presence of exactly one ``assembly_run_manifest``, ``assembly_run_record``,
    and ``primary_genome_manifest`` artifact, recomputed artifact digests for
    every declared manifest input, ``taxon_group == "plant"``, a manifest
    ``input_payload`` (else ``qc.unsupported_technology``), a Result/manifest
    run-identity match, and a complete Result↔manifest↔Genome identity chain.
    Read libraries are built only from the payload; managed HMM/database
    artifacts are verified but never represented as libraries. Any violation
    raises a typed :class:`OrganelleInputError` with an approved QC error code.
    """
    if result.operation_id != "assembly.assemble":
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message=(
                f"assembly QC consumes only assembly.assemble results; got {result.operation_id!r}"
            ),
            details={"operation_id": result.operation_id},
        )
    if result.status == "failed":
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="assembly QC cannot consume a failed assembly result",
            details={"status": result.status},
        )

    manifest_artifact = _require_unique_artifact(result, "assembly_run_manifest")
    record_artifact = _require_unique_artifact(result, "assembly_run_record")
    genome_artifact = _require_unique_artifact(result, "primary_genome_manifest")

    canonical_path = _reverify_artifact(manifest_artifact)
    record_path = _reverify_artifact(record_artifact)
    genome_path = _reverify_artifact(genome_artifact)

    manifest = AssemblyRunManifest.model_validate_json(record_path.read_bytes())
    if canonical_path.read_bytes() != manifest.canonical_bytes():
        raise OrganelleInputError(
            code="qc.artifact_digest_mismatch",
            message="canonical assembly run manifest diverges from its loadable record",
            details={"path": str(canonical_path)},
        )

    taxon_group = (
        manifest.routing_evidence.taxon_group if manifest.routing_evidence is not None else None
    )
    if taxon_group != "plant":
        raise OrganelleInputError(
            code="qc.unsupported_taxon",
            message="assembly QC v1 supports only plant organelle assemblies",
            details={"taxon_group": taxon_group},
        )

    if result.provenance is None or result.provenance.run_manifest_id != manifest.run_manifest_id:
        claimed = result.provenance.run_manifest_id if result.provenance is not None else None
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="result provenance does not match the assembly run manifest identity",
            details={
                "claimed_run_manifest_id": claimed,
                "manifest_run_manifest_id": manifest.run_manifest_id,
            },
        )

    payload = manifest.input_payload
    if payload is None:
        # Without the manifest-authoritative payload QC cannot obtain
        # technology/layout/quality_state, so it fails closed rather than guess.
        raise OrganelleInputError(
            code="qc.unsupported_technology",
            message="assembly run manifest does not record an input payload",
            details={"run_manifest_id": manifest.run_manifest_id},
        )

    # Recompute SHA256/size for every declared manifest input artifact before
    # trusting any of them, including managed HMM/database resources.
    input_artifacts_by_role: dict[str, ArtifactRef] = {}
    for item in manifest.input_artifacts:
        _reverify_artifact(item.artifact)
        input_artifacts_by_role[item.role] = item.artifact

    _verify_provenance_agreement(result, manifest)

    primary_genome = OrganelleGenome.model_validate_json(genome_path.read_bytes())
    primary_output = next(
        (output for output in manifest.outputs if output.role == manifest.primary_sequence_role),
        None,
    )
    if primary_output is None:
        raise OrganelleInputError(
            code="qc.artifact_missing",
            message="manifest does not declare the primary sequence output",
            details={"primary_sequence_role": manifest.primary_sequence_role},
        )
    # The manifest primary-sequence output is the authority: recompute its digest
    # and return its resolved path. The Genome sequence must match this content
    # identity, so a missing/tampered output URI with copied hash metadata cannot
    # pass by way of the Genome sequence alone.
    primary_fasta_path = _reverify_artifact(primary_output.artifact)
    _verify_genome_identity(
        primary_genome, result, manifest, manifest_artifact, primary_output.artifact
    )

    input_libraries = _build_input_libraries(payload, input_artifacts_by_role)

    graph_output = next(
        (output for output in manifest.outputs if output.role == "assembly_graph"),
        None,
    )
    assembly_graph_path: Path | None = None
    if graph_output is not None:
        assembly_graph_path = _reverify_artifact(graph_output.artifact)

    return ResolvedAssemblyEvidence(
        source_result=result,
        run_manifest=manifest,
        primary_genome=primary_genome,
        primary_fasta_path=primary_fasta_path,
        assembly_graph_path=assembly_graph_path,
        input_libraries=input_libraries,
    )


def _require_unique_artifact(result: OrganelleResult, kind: str) -> ArtifactRef:
    matches = tuple(artifact for artifact in result.artifacts if artifact.kind == kind)
    if len(matches) == 0:
        raise OrganelleInputError(
            code="qc.artifact_missing",
            message=f"assembly result is missing a {kind} artifact",
            details={"artifact_kind": kind},
        )
    if len(matches) > 1:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message=f"assembly result has multiple {kind} artifacts",
            details={"artifact_kind": kind, "count": len(matches)},
        )
    return matches[0]


def _reverify_artifact(artifact: ArtifactRef) -> Path:
    """Recompute the artifact digest and return its resolved path."""
    path = Path(artifact.uri)
    if not path.is_file():
        raise OrganelleInputError(
            code="qc.artifact_missing",
            message=f"artifact file is missing: {path}",
            details={"uri": artifact.uri, "artifact_kind": artifact.kind},
        )
    actual = ArtifactRef.from_path(
        path,
        kind=artifact.kind,
        format=artifact.format,
        media_type=artifact.media_type,
    )
    if actual.sha256 != artifact.sha256 or actual.size_bytes != artifact.size_bytes:
        raise OrganelleInputError(
            code="qc.artifact_digest_mismatch",
            message=f"artifact digest mismatch: {path}",
            details={
                "uri": artifact.uri,
                "artifact_kind": artifact.kind,
                "expected_sha256": artifact.sha256,
                "actual_sha256": actual.sha256,
            },
        )
    return path


def _build_input_libraries(
    payload: AssemblyInputPayload, artifacts_by_role: dict[str, ArtifactRef]
) -> tuple[InputLibraryEvidence, ...]:
    """Build read-library evidence from the payload, resolving roles exactly.

    Managed HMM/database artifacts are never represented here: only the
    payload's ``short_libraries`` and ``long_libraries`` produce evidence.
    """
    libraries: list[InputLibraryEvidence] = []
    for library in payload.short_libraries:
        read1 = _require_input_artifact(library.read1_artifact, artifacts_by_role)
        read2 = (
            _require_input_artifact(library.read2_artifact, artifacts_by_role)
            if library.read2_artifact is not None
            else None
        )
        libraries.append(
            InputLibraryEvidence(
                role=library.read1_artifact,
                technology=library.technology,
                layout=library.layout,
                quality_state="",
                read1=read1,
                read2=read2,
                reads=None,
                read_length=library.read_length,
                insert_size=library.insert_size,
            )
        )
    for library in payload.long_libraries:
        reads = _require_input_artifact(library.reads_artifact, artifacts_by_role)
        libraries.append(
            InputLibraryEvidence(
                role=library.reads_artifact,
                technology=library.technology,
                layout="",
                quality_state=library.quality_state,
                read1=None,
                read2=None,
                reads=reads,
                read_length=None,
                insert_size=None,
            )
        )
    return tuple(libraries)


def _require_input_artifact(role: str, artifacts_by_role: dict[str, ArtifactRef]) -> ArtifactRef:
    artifact = artifacts_by_role.get(role)
    if artifact is None:
        raise OrganelleInputError(
            code="qc.artifact_missing",
            message="payload library references an undeclared input artifact role",
            details={"role": role},
        )
    return artifact


def _verify_provenance_agreement(result: OrganelleResult, manifest: AssemblyRunManifest) -> None:
    """Require Result provenance to agree with the Result/manifest where the
    released assembly fields are populated."""
    provenance = result.provenance
    if provenance is None:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="assembly result has no provenance",
            details={},
        )
    if provenance.operation_id != result.operation_id:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="result provenance operation_id does not match the result",
            details={"provenance_operation_id": provenance.operation_id},
        )
    if result.operation_version != manifest.contract_version:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="result operation_version does not match the manifest contract version",
            details={"result_operation_version": result.operation_version},
        )
    if provenance.operation_version != manifest.contract_version:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="result provenance operation_version does not match the manifest contract version",
            details={"provenance_operation_version": provenance.operation_version},
        )
    if provenance.input_object_ids != (manifest.input_data_id,):
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="result provenance input_object_ids must be exactly the manifest input data identity",
            details={
                "manifest_input_data_id": manifest.input_data_id,
                "provenance_input_object_ids": list(provenance.input_object_ids),
            },
        )
    if not provenance.actual_backend or provenance.actual_backend != manifest.selected_backend:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="result provenance actual_backend must equal the manifest selected backend",
            details={
                "provenance_actual_backend": provenance.actual_backend,
                "manifest_selected_backend": manifest.selected_backend,
            },
        )


def _verify_genome_identity(
    genome: OrganelleGenome,
    result: OrganelleResult,
    manifest: AssemblyRunManifest,
    manifest_artifact: ArtifactRef,
    primary_sequence_artifact: ArtifactRef,
) -> None:
    """Verify the complete Result↔manifest↔Genome identity chain."""
    if result.scope != manifest.organelle or genome.organelle != manifest.organelle:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="result scope and genome organelle must match the manifest organelle",
            details={
                "result_scope": result.scope,
                "genome_organelle": genome.organelle,
                "manifest_organelle": manifest.organelle,
            },
        )
    if len(genome.source_manifests) != 1:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="primary genome must reference exactly one assembly run manifest",
            details={"source_manifest_count": len(genome.source_manifests)},
        )
    source = genome.source_manifests[0]
    if (
        source.kind != "assembly_run_manifest"
        or source.sha256 != manifest_artifact.sha256
        or source.size_bytes != manifest_artifact.size_bytes
    ):
        raise OrganelleInputError(
            code="qc.artifact_digest_mismatch",
            message="genome source manifest does not match the canonical assembly run manifest",
            details={"expected_sha256": manifest_artifact.sha256, "actual_sha256": source.sha256},
        )
    sequence = genome.sequence
    if sequence is None:
        raise OrganelleInputError(
            code="qc.artifact_missing",
            message="primary genome manifest has no sequence artifact",
            details={"genome_object_id": genome.object_id},
        )
    if (
        sequence.sha256 != primary_sequence_artifact.sha256
        or sequence.size_bytes != primary_sequence_artifact.size_bytes
    ):
        raise OrganelleInputError(
            code="qc.artifact_digest_mismatch",
            message="genome sequence does not match the manifest primary sequence output",
            details={
                "genome_sequence_sha256": sequence.sha256,
                "output_sha256": primary_sequence_artifact.sha256,
            },
        )
    # Recompute the Genome sequence digest too, so a missing/tampered Genome
    # sequence URI with copied hash metadata fails even when the manifest output
    # is valid.
    _reverify_artifact(sequence)
    # The direct assembly lineage is exactly one v1 assembly.assemble record.
    if len(genome.lineage) != 1:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="primary genome must carry exactly one assembly lineage record",
            details={"lineage_count": len(genome.lineage)},
        )
    lineage = genome.lineage[0]
    if lineage.operation_id != "assembly.assemble":
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="genome lineage operation_id is not assembly.assemble",
            details={"lineage_operation_id": lineage.operation_id},
        )
    if lineage.parent_object_ids != (manifest.input_data_id,):
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="genome lineage parent_object_ids must be exactly the manifest input data identity",
            details={
                "manifest_input_data_id": manifest.input_data_id,
                "lineage_parent_object_ids": list(lineage.parent_object_ids),
            },
        )
    provenance = result.provenance
    if provenance is None:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="assembly result has no provenance",
            details={},
        )
    if lineage.operation_version != manifest.contract_version:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="genome lineage operation_version does not match the manifest contract version",
            details={"lineage_operation_version": lineage.operation_version},
        )
    if lineage.parameters_hash != provenance.parameters_hash:
        raise OrganelleInputError(
            code="qc.input_contract_violation",
            message="genome lineage parameter hash does not match the result provenance",
            details={"lineage_parameters_hash": lineage.parameters_hash},
        )
