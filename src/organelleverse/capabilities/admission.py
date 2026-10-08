"""Pure eight-rule capability admission decisions."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import Field

from organelleverse.core.errors import OrganelleContractError, OrganelleError
from organelleverse.environments import locate_executable
from organelleverse.operations.spec import DependencyKind, StrictSpecModel

from .assets import resolve_asset_identity
from .hashing import hash_file, hash_fixture_dataset
from .index import CapabilityDiagnostic, CapabilityEntry, CapabilityIndex, CapabilityStatus
from .models import ImplementationKind
from .verification import (
    EnvironmentSnapshot,
    FastFileKey,
    ObservedDependency,
    PlatformIdentity,
    VerificationRecord,
    VerificationStore,
    compute_environment_key,
    current_platform_identity,
    default_verification_store,
)


class ReferenceIdentity(StrictSpecModel):
    capability_id: str
    bundle_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    environment_key: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evaluator_versions: tuple[str, ...]


class AdmissionDecision(StrictSpecModel):
    admitted: bool
    diagnostic: CapabilityDiagnostic | None = None


class AdmissionEnvironment(Protocol):
    def platform_identity(self) -> PlatformIdentity: ...

    def observe_environment(self, entry: CapabilityEntry) -> EnvironmentSnapshot: ...


class LocalAdmissionEnvironment:
    """Pure local file-identity observation: no import, subprocess, install, or network."""

    def platform_identity(self) -> PlatformIdentity:
        return current_platform_identity()

    def observe_environment(self, entry: CapabilityEntry) -> EnvironmentSnapshot:
        observed: dict[str, ObservedDependency] = {}
        probes = {probe.dependency: probe for probe in entry.bundle.probes}
        for dependency in entry.bundle.contract.dependencies:
            if dependency.kind is DependencyKind.EXECUTABLE:
                executable = locate_executable(dependency.name)
                if executable is None:
                    raise OrganelleError(
                        code="capability.environment_drift",
                        message=f"declared executable is unavailable: {dependency.name}",
                        details={"dependency": dependency.name},
                    )
                stat = executable.stat()
                probe = probes.get(dependency.name)
                observed[dependency.name] = ObservedDependency(
                    kind="executable",
                    content_hash=hash_file(executable),
                    realpath=str(executable),
                    fast_key=FastFileKey(size=stat.st_size, mtime_ns=stat.st_mtime_ns),
                    satisfied_requires=probe.requires if probe is not None else (),
                )
            elif (
                dependency.kind in {DependencyKind.MODEL, DependencyKind.DATABASE}
                and dependency.locator
            ):
                identity = resolve_asset_identity(dependency.locator)
                observed[dependency.locator] = ObservedDependency(
                    kind="asset",
                    content_hash=identity.content_hash,
                )
        return EnvironmentSnapshot(
            platform=self.platform_identity(),
            observed_environment=observed,
        )


def _reject(
    entry: CapabilityEntry, code: str, message: str, **details: object
) -> AdmissionDecision:
    origin = entry.origins
    diagnostic = CapabilityDiagnostic.model_validate(
        {
            "code": code,
            "message": message,
            "capability_id": entry.capability_id,
            "origins": origin,
            "details": details,
        }
    )
    return AdmissionDecision(admitted=False, diagnostic=diagnostic)


def evaluate_admission(
    entry: CapabilityEntry,
    *,
    record: VerificationRecord | None,
    current_environment: EnvironmentSnapshot,
    current_fixture_hash: str,
    reference_identities: dict[str, ReferenceIdentity],
) -> AdmissionDecision:
    """Apply rules 1-8 in stable order without importing or executing anything."""
    if entry.bundle.capability.implementation is ImplementationKind.COMPOSITE:
        return _reject(
            entry,
            "capability.implementation_not_supported",
            "composite capabilities are parsed and described but not executable in v1",
        )
    if record is None:
        return _reject(
            entry,
            "capability.verification_missing",
            "capability has no verification record",
        )
    current_identity = entry.execution_identity
    if (
        record.bundle_content_hash != entry.content_hash
        or record.execution_identity != current_identity
        or record.fixture_dataset_hash != current_fixture_hash
    ):
        return _reject(
            entry,
            "capability.verification_stale",
            "capability, code, execution, or fixture identity no longer matches verification",
            recorded_bundle_hash=record.bundle_content_hash,
            current_bundle_hash=entry.content_hash,
            recorded_code_hash=(
                record.execution_identity.code_tree_hash
                if record.execution_identity is not None
                else None
            ),
            current_code_hash=(
                current_identity.code_tree_hash if current_identity is not None else None
            ),
            recorded_execution_digest=(
                record.execution_identity.digest if record.execution_identity is not None else None
            ),
            current_execution_digest=(
                current_identity.digest if current_identity is not None else None
            ),
            recorded_fixture_hash=record.fixture_dataset_hash,
            current_fixture_hash=current_fixture_hash,
        )
    if record.protocol_version != "1.0":
        return _reject(
            entry,
            "capability.protocol_unsupported",
            "verification protocol is unsupported",
            protocol_version=record.protocol_version,
        )
    if record.platform != current_environment.platform:
        return _reject(
            entry,
            "capability.platform_mismatch",
            "verification record belongs to a different platform",
            recorded=record.platform.model_dump(mode="json"),
            current=current_environment.platform.model_dump(mode="json"),
        )
    current_environment_key = compute_environment_key(current_environment)
    if (
        record.environment_key != current_environment_key
        or record.observed_environment != current_environment.observed_environment
    ):
        return _reject(
            entry,
            "capability.environment_drift",
            "executable or asset identity no longer matches verification evidence",
            recorded_environment_key=record.environment_key,
            current_environment_key=current_environment_key,
        )
    for evidence in record.equivalence:
        reference = evidence.reference
        if reference is None:
            continue
        current_reference = reference_identities.get(reference.capability_id)
        if (
            current_reference is None
            or current_reference.bundle_content_hash != reference.bundle_content_hash
            or current_reference.environment_key != reference.environment_key
            or evidence.evaluator_version not in current_reference.evaluator_versions
        ):
            return _reject(
                entry,
                "capability.reference_drift",
                "reference capability or evaluator identity has changed",
                reference_capability_id=reference.capability_id,
            )
    failed = tuple(
        item.case
        for item in record.equivalence
        if item.verdict != "pass"
        or item.determinism_verdict == "fail"
        or item.rewrite_verdict == "fail"
    )
    if failed:
        return _reject(
            entry,
            "capability.equivalence_failed",
            "one or more verification fixtures failed equivalence, determinism, or rewrite checks",
            cases=list(failed),
        )
    return AdmissionDecision(admitted=True)


class _PreparedAdmission(StrictSpecModel):
    entry: CapabilityEntry
    record: VerificationRecord | None
    environment: EnvironmentSnapshot
    fixture_hash: str
    diagnostic: CapabilityDiagnostic | None = None
    schema_diagnostic: CapabilityDiagnostic | None = None


def _diagnostic_from_error(entry: CapabilityEntry, error: OrganelleError) -> CapabilityDiagnostic:
    payload = error.as_dict()
    return CapabilityDiagnostic.model_validate(
        {
            "code": error.code,
            "message": error.message,
            "capability_id": entry.capability_id,
            "origins": entry.origins,
            "details": payload["details"],
        }
    )


def _worker_schema_diagnostic(
    entry: CapabilityEntry,
    record: VerificationRecord,
) -> CapabilityDiagnostic | None:
    if entry.execution_identity is None:
        # Only a pure-core native capability ever reaches admission with no
        # ExecutionIdentity (discovery.py's origin.channel == "core"
        # exemption; a mixed-origin bundle is always required to carry a
        # real one - see _is_pure_core_native in verification.py). A core
        # capability has no bundle-local worker signature to regenerate a
        # schema from without importing the installed implementation, and
        # admission stays import-free by design (LocalAdmissionEnvironment
        # is "no import, subprocess, install, or network"), so there is no
        # honest no-import drift check to run here for it.
        return None
    if record.execution_identity != entry.execution_identity:
        return None
    from organelleverse.operations.python_binding import worker_parameter_schema

    try:
        regenerated = worker_parameter_schema(entry.bundle, record.worker_parameters)
    except OrganelleError as error:
        return _diagnostic_from_error(entry, error)
    try:
        regenerated_json = json.dumps(
            regenerated,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        recorded_json = json.dumps(
            record.parameter_schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        return _diagnostic_from_error(
            entry,
            OrganelleContractError(
                code="capability.schema_drift",
                message="verified worker schema is not canonical finite JSON",
                details={
                    "capability_id": entry.capability_id,
                    "reason": str(error),
                },
            ),
        )
    if regenerated_json == recorded_json:
        return None
    return _diagnostic_from_error(
        entry,
        OrganelleContractError(
            code="capability.schema_drift",
            message="frozen worker signature no longer regenerates the verified schema",
            details={"capability_id": entry.capability_id},
        ),
    )


def _prepare(
    entry: CapabilityEntry,
    store: VerificationStore,
    environment: AdmissionEnvironment,
) -> _PreparedAdmission:
    fixture_hash = hash_fixture_dataset(entry.bundle_root)
    records = store.records_for(entry.capability_id)
    placeholder = EnvironmentSnapshot(
        platform=environment.platform_identity(),
        observed_environment={},
    )
    if not records:
        return _PreparedAdmission(
            entry=entry,
            record=None,
            environment=placeholder,
            fixture_hash=fixture_hash,
        )
    matching_bundle = tuple(
        record for record in records if record.bundle_content_hash == entry.content_hash
    )
    if not matching_bundle:
        return _PreparedAdmission(
            entry=entry,
            record=records[0],
            environment=placeholder,
            fixture_hash=fixture_hash,
        )
    matching_fixture = tuple(
        record for record in matching_bundle if record.fixture_dataset_hash == fixture_hash
    )
    if not matching_fixture:
        return _PreparedAdmission(
            entry=entry,
            record=matching_bundle[0],
            environment=placeholder,
            fixture_hash=fixture_hash,
        )
    try:
        snapshot = environment.observe_environment(entry)
    except OrganelleError as error:
        return _PreparedAdmission(
            entry=entry,
            record=matching_fixture[0],
            environment=placeholder,
            fixture_hash=fixture_hash,
            diagnostic=_diagnostic_from_error(entry, error),
        )
    key = compute_environment_key(snapshot)
    exact = tuple(record for record in matching_fixture if record.environment_key == key)
    if exact:
        selected = exact[0]
    else:
        same_platform = tuple(
            record for record in matching_fixture if record.platform == snapshot.platform
        )
        selected = (same_platform or matching_fixture)[0]
    return _PreparedAdmission(
        entry=entry,
        record=selected,
        environment=snapshot,
        fixture_hash=fixture_hash,
        schema_diagnostic=_worker_schema_diagnostic(entry, selected),
    )


def _reference_identities(
    prepared: dict[str, _PreparedAdmission], admitted_ids: set[str]
) -> dict[str, ReferenceIdentity]:
    identities: dict[str, ReferenceIdentity] = {}
    for capability_id in sorted(admitted_ids):
        item = prepared[capability_id]
        record = item.record
        if record is None:
            continue
        identities[capability_id] = ReferenceIdentity(
            capability_id=capability_id,
            bundle_content_hash=item.entry.content_hash,
            environment_key=record.environment_key,
            evaluator_versions=tuple(
                sorted({evidence.evaluator_version for evidence in record.equivalence})
            ),
        )
    return identities


def admit_capabilities(
    index: CapabilityIndex,
    *,
    store: VerificationStore | None = None,
    environment: AdmissionEnvironment | None = None,
) -> CapabilityIndex:
    """Apply verification-record admission to every unique discovered bundle."""
    selected_store = store or default_verification_store()
    selected_environment = environment or LocalAdmissionEnvironment()
    prepared = {
        entry.capability_id: _prepare(entry, selected_store, selected_environment)
        for entry in index.entries
    }
    admitted_ids = set(prepared)
    decisions: dict[str, AdmissionDecision] = {}
    for _ in range(len(prepared) + 1):
        references = _reference_identities(prepared, admitted_ids)
        decisions = {}
        next_admitted: set[str] = set()
        for capability_id, item in prepared.items():
            if item.diagnostic is not None:
                decision = AdmissionDecision(admitted=False, diagnostic=item.diagnostic)
            else:
                decision = evaluate_admission(
                    item.entry,
                    record=item.record,
                    current_environment=item.environment,
                    current_fixture_hash=item.fixture_hash,
                    reference_identities=references,
                )
                if decision.admitted and item.schema_diagnostic is not None:
                    decision = AdmissionDecision(
                        admitted=False,
                        diagnostic=item.schema_diagnostic,
                    )
            decisions[capability_id] = decision
            if decision.admitted:
                next_admitted.add(capability_id)
        if next_admitted == admitted_ids:
            break
        admitted_ids = next_admitted
    entries: list[CapabilityEntry] = []
    diagnostics = list(index.diagnostics())
    for capability_id in sorted(prepared):
        item = prepared[capability_id]
        decision = decisions[capability_id]
        status = (
            CapabilityStatus.ADMITTED
            if decision.admitted
            else (
                CapabilityStatus.UNSUPPORTED
                if decision.diagnostic is not None
                and decision.diagnostic.code == "capability.implementation_not_supported"
                else CapabilityStatus.REJECTED
            )
        )
        record = item.record
        entries.append(
            item.entry.model_copy(
                update={
                    "status": status,
                    "diagnostic": decision.diagnostic,
                    "parameter_schema": (record.parameter_schema if record is not None else None),
                    "verification_environment_key": (
                        record.environment_key if record is not None else None
                    ),
                    "worker_parameters": (record.worker_parameters if record is not None else None),
                }
            )
        )
        if decision.diagnostic is not None:
            diagnostics.append(decision.diagnostic)
    return CapabilityIndex(
        entries=tuple(entries),
        diagnostic_items=tuple(diagnostics),
        conflict_items=index.conflicts(),
    )


__all__ = [
    "AdmissionDecision",
    "AdmissionEnvironment",
    "LocalAdmissionEnvironment",
    "ReferenceIdentity",
    "admit_capabilities",
    "evaluate_admission",
]
