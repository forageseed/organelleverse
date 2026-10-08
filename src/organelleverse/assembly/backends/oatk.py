"""Oatk PacBio HiFi assembly adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

from organelleverse.assembly.backends.base import (
    AdapterContext,
    AssemblyCommand,
    BackendOutput,
    ExpectedBackendResources,
    NormalizedAssemblyOutputs,
    PreparedBackendResources,
    RawAssemblyOutputs,
)
from organelleverse.assembly.backends.spec import AssemblyProfile
from organelleverse.assembly.contracts import OatkParameters
from organelleverse.assembly.environment_specs import AssemblyEnvironmentSpec
from organelleverse.assembly.environments import (
    EnvironmentManager,
    PreparedEnvironment,
)
from organelleverse.assembly.normalization import (
    normalize_fasta,
    normalize_gfa,
    parse_gene_markers,
    validate_fasta_against_gfa,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.external import MESSAGE_TAIL_LINES, run_external, tail_lines
from organelleverse.core.frozen import FrozenMap

if TYPE_CHECKING:
    from organelleverse.assembly.contracts import (
        AssemblyInputPayload,
        AssemblyRequest,
    )

__all__ = ["OatkAdapter", "OatkResourceProvider"]


_OUTPUT_ROLES = (
    "assembly_fasta",
    "assembly_graph",
    "gene_annotations",
    "backend_annotation",
    "backend_full_graph",
)

_NORMALIZED_NAMES = {
    "assembly_fasta": "assembly.fasta",
    "assembly_graph": "assembly.gfa",
    "gene_annotations": "gene_annotations.bed",
    "backend_annotation": "backend_annotation.txt",
    "backend_full_graph": "backend_full_graph.gfa",
}


def _unsupported(message: str, **details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.unsupported_data_profile", message=message, details=details
    )


def _output_incomplete(message: str, **details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.output_incomplete", message=message, details=details
    )


def _organelle_mismatch(**details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.organelle_mismatch",
        message=(
            "assembled sequence matches a different organelle than requested; "
            "Oatk's HMM seeding cannot guarantee the emitted contig's identity "
            "on genomes where the requested organelle does not close a circle"
        ),
        details=details,
    )


#: Profile files shipped beside the requested one in the managed Oatk database.
_SIBLING_PROFILE: dict[str, str] = {
    "mitochondrion": "embryophyta_pltd.fam",
    "plastid": "embryophyta_mito.fam",
}


class _ProfileScore(TypedDict):
    total: float
    best: float
    hits: int
    best_name: str


def _profile_score(nhmmscan: str | Path, profile: Path, fasta: Path) -> _ProfileScore:
    """Score one fam against a FASTA and summarize hit evidence.

    nhmmscan's default human output carries one ``Scores for complete hit``
    block per query with rows ``E-value score bias Model start end``; the
    tabular ``--tblout -`` stream is unreliable across HMMER builds. The
    summary uses the SUM of hit scores (robust for genome-level identity:
    a plastid scores dozens of strong pltd-fam hits against a handful of
    weak cross-organelle rRNA/gene hits) with the single best hit recorded
    for the error details. A single best hit alone is NOT sufficient — the
    conserved rRNAs score high on the wrong organelle too.
    """

    try:
        completed = run_external(
            [str(nhmmscan), str(profile), str(fasta)],
            timeout=1200,
            tool="nhmmscan",
            code="assembly.organelle_verification_failed",
            extra_details={"profile": profile.name},
        )
    except OrganelleExecutionError as error:
        message = "nhmmscan failed while verifying assembled sequence identity"
        details = cast(Mapping[str, Any], error.details)
        tail = tail_lines(str(details.get("stderr_tail", "")), MESSAGE_TAIL_LINES)
        raise OrganelleExecutionError(
            code=error.code,
            message=f"{message}: {tail}" if tail else message,
            details=details,
        ) from error
    in_scores = False
    total = 0.0
    best = -1.0
    best_name = ""
    hits = 0
    for line in completed.stdout.splitlines():
        if line.startswith("Scores for complete hit:"):
            in_scores = True
            continue
        if in_scores:
            if not line.strip() or line.lstrip().startswith(("E-value", "-------")):
                if line.lstrip().startswith("-------"):
                    continue
                if not line.strip():
                    in_scores = False
                continue
            fields = line.split()
            if len(fields) >= 4:
                try:
                    value = float(fields[1])
                except ValueError:
                    in_scores = False
                    continue
                total += value
                hits += 1
                if value > best:
                    best = value
                    best_name = fields[3]
    return {"total": round(total, 1), "best": round(best, 1), "hits": hits, "best_name": best_name}


def _verify_organelle_identity(context: AdapterContext, fasta: Path) -> None:
    """Refuse an assembled sequence that better matches the other organelle.

    Observed upstream behavior (Medicago sativa HiFi, ~75x organelle depth):
    an Oatk mitochondrion run seeded with ``embryophyta_mito.fam`` emitted the
    *plastid* contig as its circular result, because the multipartite mito
    graph did not close while the plastid component did. The released layer
    verifies the primary contig against BOTH organelle profiles and fails
    closed on mismatch rather than publishing a wrong-genome assembly.
    Verification is skipped only when the sibling profile is absent from the
    database directory.
    """

    requested = context.request.organelle
    requested_profile = Path(context.resources.artifacts["hmm_profiles"].uri)
    sibling = requested_profile.parent / _SIBLING_PROFILE[requested]
    if not sibling.is_file():
        return
    nhmmscan = context.environment.require_executable("nhmmscan")
    requested_summary = _profile_score(nhmmscan, requested_profile, fasta)
    sibling_summary = _profile_score(nhmmscan, sibling, fasta)
    if sibling_summary["total"] > requested_summary["total"]:
        raise _organelle_mismatch(
            requested_organelle=requested,
            detected_organelle=(
                "mitochondrion" if requested == "plastid" else "plastid"
            ),
            requested_profile=requested_profile.name,
            sibling_profile=sibling.name,
            requested_summary=requested_summary,
            sibling_summary=sibling_summary,
            contig=str(fasta.name),
        )


class OatkAdapter:
    """Adapter for the Oatk PacBio HiFi organelle assembler."""

    backend_id = "oatk"

    def preflight(self, context: AdapterContext) -> None:
        if context.route.selected_backend != self.backend_id:
            raise _unsupported(
                "Oatk adapter selected for a non-Oatk route",
                selected_backend=context.route.selected_backend,
            )
        if context.environment.backend_id != self.backend_id:
            raise _unsupported(
                "Oatk adapter requires an Oatk managed environment",
                environment_backend=context.environment.backend_id,
            )
        if context.route.profile is not AssemblyProfile.PACBIO_HIFI:
            raise _unsupported(
                "Oatk accepts only PacBio HiFi input",
                profile=context.route.profile.value,
            )
        if context.profile is None or context.profile.target != context.request.organelle:
            raise _unsupported(
                "resolved profile target does not match the requested organelle",
                profile_target=context.profile.target if context.profile is not None else None,
                requested_organelle=context.request.organelle,
            )
        if not context.payload.long_libraries:
            raise _unsupported("Oatk requires at least one long-read HiFi library")
        if context.payload.short_libraries:
            raise _unsupported(
                "Oatk does not consume short-read libraries",
                short_library_count=len(context.payload.short_libraries),
            )
        for library in context.payload.long_libraries:
            if library.technology != "pacbio_hifi" or library.quality_state != "ccs":
                raise _unsupported(
                    "Oatk accepts only pacbio_hifi + ccs libraries",
                    technology=library.technology,
                    quality_state=library.quality_state,
                )
        if context.payload.contig_inputs:
            raise _unsupported("Oatk does not accept preassembled contig input")
        unsupported_auxiliary = sorted(
            name
            for name, value in context.payload.auxiliary.model_dump(mode="python").items()
            if name != "hmm_profiles_artifact" and value is not None
        )
        if unsupported_auxiliary:
            raise _unsupported(
                "Oatk does not consume the declared auxiliary inputs",
                unsupported_auxiliary=unsupported_auxiliary,
            )
        provided = context.request.backend_parameters
        if provided is not None and provided.backend != self.backend_id:
            raise _unsupported(
                "backend_parameters discriminator does not match Oatk",
                discriminator=provided.backend,
            )

    def build_command(self, context: AdapterContext) -> AssemblyCommand:
        self.preflight(context)
        organelle = context.request.organelle
        coverage = _effective_coverage(context)
        threads = context.request.threads

        stable = [
            "oatk",
            "-c",
            str(coverage),
            "-t",
            str(threads),
            "--nhmmscan",
            "nhmmscan",
            "-m" if organelle == "mitochondrion" else "-p",
            "role://artifact/hmm_profiles",
            "-o",
            "role://workspace/assembly",
        ]
        resolved = [
            str(context.environment.require_executable("oatk")),
            "-c",
            str(coverage),
            "-t",
            str(threads),
            "--nhmmscan",
            str(context.environment.require_executable("nhmmscan")),
            "-m" if organelle == "mitochondrion" else "-p",
            str(context.resources.artifacts["hmm_profiles"].uri),
            "-o",
            str(context.workspace / "backend" / "oatk" / "oatk"),
        ]
        for role in _ordered_hifi_roles(context):
            stable.append(f"role://artifact/{role}")
            artifact = context.input_artifacts[role]
            _verify_artifact_hash(artifact)
            resolved.append(str(Path(artifact.uri)))

        return AssemblyCommand(
            stable_argv=tuple(stable),
            resolved_argv=tuple(resolved),
        )

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs:
        suffix = "mito" if context.request.organelle == "mitochondrion" else "pltd"
        prefix = context.workspace / "backend" / "oatk"
        role_paths = {
            "assembly_fasta": prefix / f"oatk.{suffix}.ctg.fasta",
            "assembly_graph": prefix / f"oatk.{suffix}.gfa",
            "gene_annotations": prefix / f"oatk.{suffix}.ctg.bed",
            "backend_annotation": prefix / f"oatk.annot_{suffix}.txt",
            "backend_full_graph": prefix / "oatk.utg.final.gfa",
        }
        missing: list[str] = []
        outputs: list[BackendOutput] = []
        for role in _OUTPUT_ROLES:
            path = role_paths[role]
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(role)
            outputs.append(
                BackendOutput(
                    role=role,
                    path=path,
                    format=path.suffix.lstrip(".") or "bin",
                )
            )
        if missing:
            raise _output_incomplete(
                "Oatk did not produce every required target file",
                missing_roles=sorted(missing),
                organelle=context.request.organelle,
            )
        _verify_organelle_identity(context, role_paths["assembly_fasta"])
        return RawAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="assembly_fasta",
            primary_graph_role="assembly_graph",
        )

    def normalize(
        self,
        context: AdapterContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        destinations: dict[str, Path] = {}
        for role in _OUTPUT_ROLES:
            destinations[role] = output_dir / _NORMALIZED_NAMES[role]

        fasta_stats = normalize_fasta(
            raw.require("assembly_fasta").path, destinations["assembly_fasta"]
        )
        gfa_stats = normalize_gfa(
            raw.require("assembly_graph").path, destinations["assembly_graph"]
        )
        validate_fasta_against_gfa(fasta_stats, gfa_stats)
        _copy_file(raw.require("gene_annotations").path, destinations["gene_annotations"])
        _copy_file(raw.require("backend_annotation").path, destinations["backend_annotation"])
        _copy_file(raw.require("backend_full_graph").path, destinations["backend_full_graph"])

        # Raw/normalized parsed sequence equality is enforced by normalize_fasta/
        # normalize_gfa preserving content; re-verify the normalized hashes match.
        normalized_fasta = normalize_fasta(
            destinations["assembly_fasta"], output_dir / ".verify.fasta"
        )
        if normalized_fasta.sequences != fasta_stats.sequences:
            raise OrganelleExecutionError(
                code="assembly.validation_failed",
                message="normalized FASTA sequence content diverged from raw",
            )
        (output_dir / ".verify.fasta").unlink(missing_ok=True)

        markers = parse_gene_markers(destinations["gene_annotations"])
        output_records = tuple(
            BackendOutput(
                role=role,
                path=destinations[role],
                format=destinations[role].suffix.lstrip(".") or "bin",
            )
            for role in _OUTPUT_ROLES
        )
        return NormalizedAssemblyOutputs(
            outputs=output_records,
            primary_sequence_role="assembly_fasta",
            primary_graph_role="assembly_graph",
            record_count=fasta_stats.record_count,
            total_bases=fasta_stats.total_bases,
            gene_markers=markers,
        )


def _effective_coverage(context: AdapterContext) -> int:
    params = context.effective_backend_parameters
    if "minimum_kmer_coverage" in params:
        value = params["minimum_kmer_coverage"]
        if isinstance(value, int):
            return value
    provided = context.request.backend_parameters
    if isinstance(provided, OatkParameters):
        return provided.minimum_kmer_coverage
    return 30


def _ordered_hifi_roles(context: AdapterContext) -> tuple[str, ...]:
    roles: list[str] = []
    for index in range(len(context.payload.long_libraries)):
        roles.append(f"long_{index}_reads")
    return tuple(roles)


def _verify_artifact_hash(artifact: ArtifactRef) -> None:
    path = Path(artifact.uri)
    if not path.is_file():
        raise OrganelleExecutionError(
            code="assembly.output_incomplete",
            message="declared HiFi read artifact is missing or unreadable",
            details={"uri": artifact.uri},
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != artifact.sha256:
        raise OrganelleExecutionError(
            code="assembly.output_incomplete",
            message="declared HiFi read artifact hash no longer matches its content",
            details={"uri": artifact.uri, "declared_sha256": artifact.sha256},
        )


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


class OatkResourceProvider:
    """Prepare and identify the HMM profile resource used by Oatk."""

    def expected(
        self,
        manager: EnvironmentManager,
        environment_spec: AssemblyEnvironmentSpec,
        request: AssemblyRequest,
        payload: AssemblyInputPayload,
    ) -> ExpectedBackendResources:
        override = _profile_override_from_data(payload, request.data)
        component = manager.expected_profile_identity(
            environment_spec,
            organelle=request.organelle,
            override=override,
        )
        return ExpectedBackendResources(
            components=(component,),
            artifact_roles=("hmm_profiles",),
            model_hashes=FrozenMap({})
            if override is None
            else FrozenMap({"hmm_profile": component.sha256}),
            database_hashes=FrozenMap({"oatkdb": component.sha256})
            if override is None
            else FrozenMap({}),
        )

    def prepare(
        self,
        manager: EnvironmentManager,
        environment_spec: AssemblyEnvironmentSpec,
        request: AssemblyRequest,
        payload: AssemblyInputPayload,
        environment: PreparedEnvironment,
    ) -> PreparedBackendResources:
        override = _profile_override_from_data(payload, request.data)
        profile = manager.prepare_profile(
            environment_spec,
            organelle=request.organelle,
            policy="require" if request.environment_source == "managed" else "ensure",
            override=override,
            environment=environment,
        )
        return PreparedBackendResources(
            components=(profile.component,),
            artifacts=FrozenMap.from_items({"hmm_profiles": profile.artifact}),
            model_hashes=FrozenMap({})
            if override is None
            else FrozenMap({"hmm_profile": profile.component.sha256}),
            database_hashes=FrozenMap({"oatkdb": profile.component.sha256})
            if override is None
            else FrozenMap({}),
        )


def _profile_override_from_data(
    payload: AssemblyInputPayload, data: OrganelleData
) -> ArtifactRef | None:
    """Resolve the user-supplied HMM profile ArtifactRef from the payload and data."""
    role = payload.auxiliary.hmm_profiles_artifact
    if role is None:
        return None
    artifact = data.artifacts.get(role)
    if artifact is None:
        return None
    return artifact
