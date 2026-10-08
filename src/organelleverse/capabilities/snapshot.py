"""Deterministic, read-only capability admission snapshots."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import Field

from organelleverse.operations.spec import StrictSpecModel

from .admission import AdmissionEnvironment, LocalAdmissionEnvironment, admit_capabilities
from .index import CapabilityIndex, CapabilityStatus
from .trust import TrustStore
from .verification import PlatformIdentity, VerificationStore


class CapabilitySnapshotCounts(StrictSpecModel):
    """Unambiguous counts for discovery, admission, and Agent visibility."""

    discovered: int
    admitted: int
    not_admitted: int
    agent_visible: int
    admitted_but_untrusted: int
    unsupported: int
    conflicted_ids: int
    discovery_diagnostics: int


class CapabilitySnapshotEntry(StrictSpecModel):
    """One candidate's final admission and effective trust state."""

    capability_id: str
    status: CapabilityStatus
    reason_code: str | None = None
    origin_channels: tuple[str, ...]
    trust_state: Literal["not_required", "trusted", "untrusted"]
    agent_visible: bool
    verification_environment_key: str | None = None


class CapabilityAdmissionSnapshot(StrictSpecModel):
    """A time-free snapshot suitable for release evidence and figure counts."""

    schema_version: Literal["organelleverse.capability-admission-snapshot.v1"] = Field(
        default="organelleverse.capability-admission-snapshot.v1",
        alias="schema",
    )
    platform: PlatformIdentity
    counts: CapabilitySnapshotCounts
    not_admitted_by_reason: dict[str, int]
    agent_hidden_by_reason: dict[str, int]
    capabilities: tuple[CapabilitySnapshotEntry, ...]


def _trust_state(
    *,
    admitted: bool,
    pure_core: bool,
    execution_digest: str | None,
    trust_store: TrustStore,
) -> Literal["not_required", "trusted", "untrusted"]:
    if pure_core:
        return "not_required"
    if not admitted:
        return "untrusted"
    if execution_digest is not None and trust_store.is_trusted(execution_digest):
        return "trusted"
    return "untrusted"


def build_admission_snapshot(
    candidates: CapabilityIndex,
    *,
    verification_store: VerificationStore,
    trust_store: TrustStore,
    environment: AdmissionEnvironment | None = None,
) -> CapabilityAdmissionSnapshot:
    """Evaluate one candidate index without modifying verification or trust stores."""
    selected_environment = environment or LocalAdmissionEnvironment()
    admitted_index = admit_capabilities(
        candidates,
        store=verification_store,
        environment=selected_environment,
    )
    entries: list[CapabilitySnapshotEntry] = []
    withheld: Counter[str] = Counter()
    hidden: Counter[str] = Counter()
    admitted_count = 0
    visible_count = 0
    unsupported_count = 0

    for entry in admitted_index.entries:
        admitted = entry.status is CapabilityStatus.ADMITTED
        if admitted:
            admitted_count += 1
        if entry.status is CapabilityStatus.UNSUPPORTED:
            unsupported_count += 1
        reason_code = None if entry.diagnostic is None else entry.diagnostic.code
        if not admitted:
            withheld[reason_code or "capability.admission_unknown"] += 1

        pure_core = all(origin.channel == "core" for origin in entry.origins)
        execution_digest = (
            None if entry.execution_identity is None else entry.execution_identity.digest
        )
        trust_state = _trust_state(
            admitted=admitted,
            pure_core=pure_core,
            execution_digest=execution_digest,
            trust_store=trust_store,
        )
        agent_visible = admitted and trust_state in {"not_required", "trusted"}
        if agent_visible:
            visible_count += 1
        elif admitted:
            hidden["capability.untrusted"] += 1

        entries.append(
            CapabilitySnapshotEntry(
                capability_id=entry.capability_id,
                status=entry.status,
                reason_code=reason_code,
                origin_channels=tuple(sorted({origin.channel for origin in entry.origins})),
                trust_state=trust_state,
                agent_visible=agent_visible,
                verification_environment_key=entry.verification_environment_key,
            )
        )

    discovered_count = len(entries)
    return CapabilityAdmissionSnapshot(
        platform=selected_environment.platform_identity(),
        counts=CapabilitySnapshotCounts(
            discovered=discovered_count,
            admitted=admitted_count,
            not_admitted=discovered_count - admitted_count,
            agent_visible=visible_count,
            admitted_but_untrusted=admitted_count - visible_count,
            unsupported=unsupported_count,
            conflicted_ids=len(candidates.conflicts()),
            discovery_diagnostics=len(candidates.diagnostics()),
        ),
        not_admitted_by_reason=dict(sorted(withheld.items())),
        agent_hidden_by_reason=dict(sorted(hidden.items())),
        capabilities=tuple(entries),
    )


__all__ = [
    "CapabilityAdmissionSnapshot",
    "CapabilitySnapshotCounts",
    "CapabilitySnapshotEntry",
    "build_admission_snapshot",
]
