"""Pure, deterministic compatibility routing for structured assembly inputs."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError, OrganelleNeedsInput
from organelleverse.core.input_requests import make_needs_input_request
from organelleverse.operations.spec import StrictSpecModel

from .backends import BACKENDS, AssemblyBackendSpec, AssemblyProfile, AuxiliaryRole
from .contracts import (
    AssemblyInputPayload,
    AssemblyRequest,
    GenomeSizeEvidence,
    validate_assembly_data,
)
from .read_stats import ReadStatistics, ReadStatisticsCache, stream_read_statistics

OrganelleType = Literal["mitochondrion", "plastid"]
TaxonGroup = Literal["plant", "animal", "fungi"]

#: The integer coverage multiplier that separates low-coverage PMAT routing from
#: high-coverage Oatk routing. Routing compares integers only, so the boundary
#: at exactly ``3 * genome_size_bp`` is unambiguous.
COVERAGE_THRESHOLD_MULTIPLIER = 3

_GENOME_SIZE_CHOICES: tuple[str, ...] = (
    "use_default_oatk",
    "provide_genome_size",
    "lookup_by_species",
)


class RejectedBackend(StrictSpecModel):
    backend_id: str
    reason_code: str
    missing: tuple[str, ...] = ()


class AssemblyRoute(StrictSpecModel):
    requested_method: str
    selected_backend: str
    profile: AssemblyProfile
    rule_id: str
    compatible_candidates: tuple[str, ...]
    rejected_candidates: tuple[RejectedBackend, ...] = ()


class RoutingContext(StrictSpecModel):
    """Backend-neutral routing context built before any side effect."""

    profile: AssemblyProfile
    taxon_group: TaxonGroup
    total_bases: int | None = None
    genome_size_bp: int | None = None
    threshold_multiplier: int | None = None
    genome_size_evidence: GenomeSizeEvidence | None = None
    read_statistics: tuple[ReadStatistics, ...] = ()


class AutoRouteRule(StrictSpecModel):
    """One ordered, declarative automatic-routing rule."""

    rule_id: str
    backend_id: str
    organelle: OrganelleType | None
    profiles: tuple[AssemblyProfile, ...]
    taxon_group: TaxonGroup | None = None
    coverage_predicate: Literal["lte_3x", "gt_3x"] | None = None


_LONG_PROFILES = {
    ("pacbio_hifi", "ccs"): AssemblyProfile.PACBIO_HIFI,
    ("pacbio_clr", "raw"): AssemblyProfile.PACBIO_CLR_RAW,
    ("pacbio_clr", "corrected"): AssemblyProfile.PACBIO_CLR_CORRECTED,
    ("ont", "raw"): AssemblyProfile.ONT_RAW,
    ("ont", "corrected"): AssemblyProfile.ONT_CORRECTED,
    ("ont", "hq"): AssemblyProfile.ONT_HQ,
    ("ont", "duplex"): AssemblyProfile.ONT_DUPLEX,
}

#: Noisy long reads together with one Illumina library (second plus third generation).
_HYBRID_PROFILES = {
    ("pacbio_clr", "raw"): AssemblyProfile.PACBIO_CLR_RAW_ILLUMINA,
    ("ont", "raw"): AssemblyProfile.ONT_RAW_ILLUMINA,
}

_AUXILIARY_FIELDS = {
    AuxiliaryRole.SEED_FASTA: "seed_fasta_artifact",
    AuxiliaryRole.REFERENCE_FASTA: "reference_fasta_artifact",
    AuxiliaryRole.REFERENCE_GENBANK: "reference_genbank_artifact",
    AuxiliaryRole.CHLOROPLAST_FASTA: "chloroplast_fasta_artifact",
    AuxiliaryRole.HMM_PROFILES: "hmm_profiles_artifact",
    AuxiliaryRole.CORRECTION_CONFIG: "correction_config_artifact",
    AuxiliaryRole.GENOME_SIZE: "genome_size",
    AuxiliaryRole.GENOME_RANGE: "genome_range",
    AuxiliaryRole.GENOME_SIZE_REPORT: "genome_size_report_artifact",
    AuxiliaryRole.CANU_EXECUTABLE: "canu_executable_artifact",
    AuxiliaryRole.NEXTDENOVO_EXECUTABLE: "nextdenovo_executable_artifact",
}

AUTO_RULES = (
    AutoRouteRule(
        rule_id="auto.mito_hifi_lte_3x_pmat",
        backend_id="pmat",
        organelle="mitochondrion",
        profiles=(AssemblyProfile.PACBIO_HIFI,),
        taxon_group="plant",
        coverage_predicate="lte_3x",
    ),
    AutoRouteRule(
        rule_id="auto.mito_hifi_gt_3x_oatk",
        backend_id="oatk",
        organelle="mitochondrion",
        profiles=(AssemblyProfile.PACBIO_HIFI,),
        taxon_group="plant",
        coverage_predicate="gt_3x",
    ),
    AutoRouteRule(
        rule_id="auto.hifi_oatk",
        backend_id="oatk",
        organelle=None,
        profiles=(AssemblyProfile.PACBIO_HIFI,),
    ),
    AutoRouteRule(
        rule_id="auto.raw_clr_pmat",
        backend_id="pmat",
        organelle=None,
        profiles=(AssemblyProfile.PACBIO_CLR_RAW,),
    ),
    AutoRouteRule(
        rule_id="auto.raw_ont_pmat",
        backend_id="pmat",
        organelle=None,
        profiles=(AssemblyProfile.ONT_RAW,),
    ),
    AutoRouteRule(
        rule_id="auto.illumina_pe_getorganelle",
        backend_id="getorganelle",
        organelle=None,
        profiles=(AssemblyProfile.ILLUMINA_PE,),
    ),
    AutoRouteRule(
        rule_id="auto.illumina_se_getorganelle",
        backend_id="getorganelle",
        organelle=None,
        profiles=(AssemblyProfile.ILLUMINA_SE,),
    ),
)


def _profile_error(message: str, payload: AssemblyInputPayload) -> OrganelleInputError:
    return OrganelleInputError(
        code="assembly.unsupported_data_profile",
        message=message,
        details={
            "short_layouts": sorted({item.layout for item in payload.short_libraries}),
            "long_technologies": sorted({item.technology for item in payload.long_libraries}),
            "quality_states": sorted({item.quality_state for item in payload.long_libraries}),
            "contig_inputs": len(payload.contig_inputs),
        },
    )


def classify_profile(data: OrganelleData) -> AssemblyProfile:
    """Classify a complete validated assembly input into one canonical profile."""
    payload = validate_assembly_data(data)
    has_reads = bool(payload.short_libraries or payload.long_libraries)
    if payload.contig_inputs:
        if has_reads:
            raise _profile_error("reads and contigs cannot be mixed", payload)
        raise _profile_error("preassembled contigs are unsupported", payload)

    technologies = {item.technology for item in payload.long_libraries}
    qualities = {item.quality_state for item in payload.long_libraries}
    layouts = {item.layout for item in payload.short_libraries}
    if len(technologies) > 1 or len(qualities) > 1:
        raise _profile_error("mixed long-read profiles are unsupported", payload)

    if technologies:
        technology = next(iter(technologies))
        quality = next(iter(qualities))
        if payload.short_libraries and technology != "pacbio_hifi":
            hybrid = _HYBRID_PROFILES.get((technology, quality))
            if (
                hybrid is None
                or len(payload.long_libraries) != 1
                or len(payload.short_libraries) != 1
            ):
                raise _profile_error(
                    "this long-read profile cannot be combined with short reads",
                    payload,
                )
            return hybrid
        return _LONG_PROFILES[(technology, quality)]

    if len(layouts) != 1:
        raise _profile_error("mixed Illumina layouts are unsupported", payload)
    layout = next(iter(layouts))
    return AssemblyProfile.ILLUMINA_PE if layout == "paired_end" else AssemblyProfile.ILLUMINA_SE


def compatible_backends(
    data: OrganelleData,
    organelle: OrganelleType,
) -> tuple[str, ...]:
    """Return all registry backends compatible with the classified profile."""
    profile = classify_profile(data)
    return tuple(
        spec.backend_id
        for spec in BACKENDS.values()
        if organelle in spec.organelles and profile in spec.profiles
    )


def _released_backend_ids() -> tuple[str, ...]:
    """Return the currently released runtime backend ids (local import)."""
    from organelleverse.assembly.backends.runtime import RUNTIMES

    return tuple(RUNTIMES)


def _genome_size_evidence(payload: AssemblyInputPayload) -> GenomeSizeEvidence | None:
    return payload.auxiliary.genome_size_evidence


def _is_plant_mito_hifi_auto(
    request: AssemblyRequest,
    profile: AssemblyProfile,
) -> bool:
    return (
        request.method == "auto"
        and request.taxon_group == "plant"
        and request.organelle == "mitochondrion"
        and profile is AssemblyProfile.PACBIO_HIFI
    )


def _genome_size_required_error(
    semantic_payload: Mapping[str, object],
    *,
    organelle: OrganelleType,
    taxon_group: TaxonGroup,
) -> OrganelleNeedsInput:
    needs_input = make_needs_input_request(
        semantic_payload=semantic_payload,
        field="auxiliary.genome_size",
        choices=_GENOME_SIZE_CHOICES,
    )
    return OrganelleNeedsInput(
        needs_input=needs_input,
        code="assembly.genome_size_required",
        message="nuclear genome size is required for coverage routing",
        details={
            "organelle": organelle,
            "profile": AssemblyProfile.PACBIO_HIFI.value,
            "taxon_group": taxon_group,
        },
        suggested_action={"choose": list(_GENOME_SIZE_CHOICES)},
    )


def prepare_routing_context(
    request: AssemblyRequest,
    *,
    stats_cache: ReadStatisticsCache | None = None,
    released_backend_ids: Collection[str] | None = None,
) -> RoutingContext:
    """Build the read-only routing context before any workspace side effect.

    The context carries the classified profile, the common taxon group, the
    traceable genome-size evidence, and streaming long-read statistics. It never
    creates a workspace, materializes an environment, or launches a backend.
    """
    payload = validate_assembly_data(request.data)
    profile = classify_profile(request.data)
    evidence = _genome_size_evidence(payload)
    genome_size_bp = evidence.genome_size_bp if evidence is not None else None

    # When PMAT is released and this is a plant-mitochondrial HiFi automatic
    # request, the genome size is required to choose between PMAT and Oatk. The
    # missing-input signal fires before any read parsing or cache write.
    released = (
        set(released_backend_ids)
        if released_backend_ids is not None
        else set(_released_backend_ids())
    )
    coverage_routing_active = "pmat" in released and _is_plant_mito_hifi_auto(request, profile)
    if coverage_routing_active and genome_size_bp is None:
        raise _genome_size_required_error(
            request.semantic_payload(),
            organelle=request.organelle,
            taxon_group=request.taxon_group,
        )

    read_statistics: list[ReadStatistics] = []
    for library in payload.long_libraries:
        artifact = request.data.artifacts[library.reads_artifact]
        read_statistics.append(stream_read_statistics(artifact, cache=stats_cache))
    total_bases = sum(stat.total_bases for stat in read_statistics) if read_statistics else None
    threshold_multiplier = (
        COVERAGE_THRESHOLD_MULTIPLIER
        if coverage_routing_active and genome_size_bp is not None
        else None
    )
    return RoutingContext(
        profile=profile,
        taxon_group=request.taxon_group,
        total_bases=total_bases,
        genome_size_bp=genome_size_bp,
        threshold_multiplier=threshold_multiplier,
        genome_size_evidence=evidence,
        read_statistics=tuple(read_statistics),
    )


def _missing_requirements(
    payload: AssemblyInputPayload,
    spec: AssemblyBackendSpec,
    organelle: OrganelleType,
    profile: AssemblyProfile,
) -> tuple[str, ...]:
    roles = list(spec.required_auxiliary)
    if organelle == "mitochondrion":
        roles.extend(spec.mitochondrial_auxiliary)
    if spec.backend_id == "pmat" and profile in {
        AssemblyProfile.PACBIO_CLR_RAW,
        AssemblyProfile.ONT_RAW,
    }:
        roles.append(AuxiliaryRole.CORRECTION_CONFIG)
    missing = {
        role.value for role in roles if getattr(payload.auxiliary, _AUXILIARY_FIELDS[role]) is None
    }
    for field_name in spec.required_short_fields:
        if any(getattr(library, field_name) is None for library in payload.short_libraries):
            missing.add(field_name)
    return tuple(sorted(missing))


def _raise_missing(
    spec: AssemblyBackendSpec,
    profile: AssemblyProfile,
    organelle: OrganelleType,
    missing: tuple[str, ...],
) -> None:
    if not missing:
        return
    raise OrganelleInputError(
        code="assembly.missing_auxiliary",
        message=f"{spec.backend_id} requires additional assembly inputs",
        details={
            "backend": spec.backend_id,
            "profile": profile.value,
            "organelle": organelle,
            "missing": list(missing),
        },
        suggested_action={"provide_auxiliary": list(missing)},
    )


def _rejected_candidates(
    payload: AssemblyInputPayload,
    profile: AssemblyProfile,
    organelle: OrganelleType,
    selected_backend: str,
    eligible_backend_ids: Collection[str],
) -> tuple[RejectedBackend, ...]:
    rejected: list[RejectedBackend] = []
    eligible = set(eligible_backend_ids)
    for spec in BACKENDS.values():
        if (
            spec.backend_id not in eligible
            or spec.backend_id == selected_backend
            or organelle not in spec.organelles
        ):
            continue
        if profile not in spec.profiles:
            rejected.append(
                RejectedBackend(
                    backend_id=spec.backend_id,
                    reason_code="assembly.unsupported_data_profile",
                )
            )
            continue
        missing = _missing_requirements(payload, spec, organelle, profile)
        rejected.append(
            RejectedBackend(
                backend_id=spec.backend_id,
                reason_code=("assembly.missing_auxiliary" if missing else "assembly.not_selected"),
                missing=missing,
            )
        )
    return tuple(rejected)


def _resolve_explicit(
    *,
    method: str,
    organelle: OrganelleType,
    profile: AssemblyProfile,
    payload: AssemblyInputPayload,
    candidates: tuple[str, ...],
    released: set[str] | None,
) -> AssemblyRoute:
    if method not in BACKENDS.ids():
        raise OrganelleInputError(
            code="assembly.unknown_backend",
            message=f"unknown assembly backend: {method}",
            details={"backend": method, "known_backends": list(BACKENDS)},
        )
    if released is not None and method not in released:
        raise OrganelleInputError(
            code="assembly.unknown_backend",
            message=f"assembly backend {method} is not released",
            details={"backend": method, "released_backends": sorted(released)},
        )
    spec = BACKENDS[method]
    if organelle not in spec.organelles or profile not in spec.profiles:
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message=f"{method} does not support the requested assembly profile",
            details={
                "backend": method,
                "organelle": organelle,
                "profile": profile.value,
                "compatible_candidates": list(candidates),
            },
        )
    _raise_missing(
        spec,
        profile,
        organelle,
        _missing_requirements(payload, spec, organelle, profile),
    )
    return AssemblyRoute(
        requested_method=method,
        selected_backend=method,
        profile=profile,
        rule_id="explicit.backend",
        compatible_candidates=candidates,
        rejected_candidates=_rejected_candidates(
            payload,
            profile,
            organelle,
            method,
            released if released is not None else BACKENDS.ids(),
        ),
    )


def _coverage_matches(
    rule: AutoRouteRule,
    context: RoutingContext | None,
) -> bool:
    """Return True when a coverage-predicate rule matches the context.

    Coverage rules apply only when PMAT is part of the decision, which requires
    a complete context (total bases and genome size). The comparison is integer
    only to keep the ``3 * genome_size`` boundary exact.
    """
    if rule.coverage_predicate is None:
        return True
    if context is None:
        return False
    if context.genome_size_bp is None or context.total_bases is None:
        return False
    is_low_coverage = context.total_bases <= COVERAGE_THRESHOLD_MULTIPLIER * context.genome_size_bp
    if rule.coverage_predicate == "lte_3x":
        return is_low_coverage
    return not is_low_coverage


def _routing_semantic_payload(
    data: OrganelleData,
    *,
    organelle: OrganelleType,
    profile: AssemblyProfile,
    taxon_group: TaxonGroup,
) -> Mapping[str, object]:
    """Build a destination-independent routing identity for needs-input.

    Direct callers that do not pass an :class:`AssemblyRequest` still get a
    deterministic, path-free payload: the input data identity, requested
    organelle, classified profile, and taxon group.
    """
    return {
        "input_data_id": data.object_id,
        "organelle": organelle,
        "profile": profile.value,
        "taxon_group": taxon_group,
        "requested_method": "auto",
    }


def _select_auto_rule(
    *,
    profile: AssemblyProfile,
    organelle: OrganelleType,
    taxon_group: TaxonGroup,
    context: RoutingContext | None,
    released: set[str] | None,
    data: OrganelleData,
) -> AutoRouteRule | None:
    eligible = released if released is not None else set(BACKENDS.ids())
    coverage_path_active = (
        released is not None
        and "pmat" in released
        and taxon_group == "plant"
        and organelle == "mitochondrion"
        and profile is AssemblyProfile.PACBIO_HIFI
    )
    for rule in AUTO_RULES:
        if rule.backend_id not in eligible:
            continue
        if rule.organelle is not None and rule.organelle != organelle:
            continue
        if rule.taxon_group is not None and rule.taxon_group != taxon_group:
            continue
        if profile not in rule.profiles:
            continue
        if rule.coverage_predicate is not None:
            # The PMAT-vs-Oatk coverage decision is meaningful only when PMAT is
            # released. Otherwise the generic HiFi rule selects Oatk and no
            # genome size is requested.
            if not coverage_path_active:
                continue
            if context is None or context.genome_size_bp is None:
                raise _genome_size_required_error(
                    _routing_semantic_payload(
                        data, organelle=organelle, profile=profile, taxon_group=taxon_group
                    ),
                    organelle=organelle,
                    taxon_group=taxon_group,
                )
            if not _coverage_matches(rule, context):
                continue
        return rule
    return None


def resolve_backend(
    data: OrganelleData,
    *,
    organelle: OrganelleType,
    method: str = "auto",
    context: RoutingContext | None = None,
    released_backend_ids: Collection[str] | None = None,
) -> AssemblyRoute:
    """Resolve one backend deterministically without probing availability.

    ``released_backend_ids`` restricts automatic and explicit routing to the
    supplied runtime set. When it is ``None`` the current released runtime
    registry is loaded locally, so public routing never advertises a static
    backend that cannot execute.
    ``context`` carries streaming statistics and genome-size evidence built by
    :func:`prepare_routing_context`; automatic coverage rules consult it to
    choose between PMAT and Oatk and to request a missing genome size.
    """
    if organelle not in {"mitochondrion", "plastid"}:
        raise OrganelleInputError(
            code="assembly.unsupported_organelle",
            message="assembly organelle must be mitochondrion or plastid",
            details={"organelle": organelle},
        )
    payload = validate_assembly_data(data)
    profile = classify_profile(data)
    released = (
        set(released_backend_ids)
        if released_backend_ids is not None
        else set(_released_backend_ids())
    )
    candidates = tuple(
        backend_id for backend_id in compatible_backends(data, organelle) if backend_id in released
    )
    taxon_group: TaxonGroup = context.taxon_group if context is not None else "plant"

    if context is not None and context.profile is not profile:
        raise OrganelleInputError(
            code="assembly.routing_context_mismatch",
            message="routing context profile does not match assembly data",
            details={
                "context_profile": context.profile.value,
                "data_profile": profile.value,
            },
        )

    if method != "auto":
        return _resolve_explicit(
            method=method,
            organelle=organelle,
            profile=profile,
            payload=payload,
            candidates=candidates,
            released=released,
        )

    selected_rule = _select_auto_rule(
        profile=profile,
        organelle=organelle,
        taxon_group=taxon_group,
        context=context,
        released=released,
        data=data,
    )
    if selected_rule is None:
        raise OrganelleInputError(
            code="assembly.unsupported_data_profile",
            message="no automatic assembly route supports this profile",
            details={
                "organelle": organelle,
                "profile": profile.value,
                "compatible_candidates": list(candidates),
            },
            suggested_action={"select_explicit_backend": list(candidates)},
        )
    backend_id = selected_rule.backend_id
    spec = BACKENDS[backend_id]
    _raise_missing(
        spec,
        profile,
        organelle,
        _missing_requirements(payload, spec, organelle, profile),
    )
    return AssemblyRoute(
        requested_method="auto",
        selected_backend=backend_id,
        profile=profile,
        rule_id=selected_rule.rule_id,
        compatible_candidates=candidates,
        rejected_candidates=_rejected_candidates(
            payload,
            profile,
            organelle,
            backend_id,
            released,
        ),
    )
