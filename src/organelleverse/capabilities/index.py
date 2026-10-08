"""Immutable discovery/admission index and structured terminal states."""

from __future__ import annotations

import importlib
import threading
from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

from pydantic import Field, model_validator
from pydantic import JsonValue as PydanticJsonValue

from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleInputError,
)
from organelleverse.operations.spec import StrictSpecModel

from .code_identity import ExecutionIdentity
from .models import BundleDocument
from .worker_contracts import WorkerParameter

if TYPE_CHECKING:
    from organelleverse.operations.python_binding import EnvironmentParameterProvider
    from organelleverse.operations.registry import BoundOperation
    from organelleverse.operations.spec import OperationSpec

    from .trust import TrustStore
    from .worker import BundleWorkerExecutor

_HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ORIGIN_PATTERN = r"^(core|local|project|package:[A-Za-z0-9._-]+)$"


class CapabilityStatus(StrEnum):
    """One unique bundle's current read-side state."""

    DISCOVERED = "discovered"
    ADMITTED = "admitted"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"


class CapabilityOrigin(StrictSpecModel):
    """One channel and concrete source that yielded a bundle."""

    channel: str = Field(pattern=_ORIGIN_PATTERN)
    source_path: str = Field(min_length=1)
    search_root: str | None = None


class CapabilityDiagnostic(StrictSpecModel):
    """A deterministic explanation for a candidate/channel that was not admitted."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    capability_id: str | None = None
    origins: tuple[CapabilityOrigin, ...] = ()
    details: dict[str, PydanticJsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def canonicalize(self) -> Self:
        ordered = tuple(sorted(self.origins, key=lambda item: (item.channel, item.source_path)))
        if ordered != self.origins:
            object.__setattr__(self, "origins", ordered)
        return self


class CapabilityEntry(StrictSpecModel):
    """One same-hash-deduplicated capability and all of its origins."""

    capability_id: str
    content_hash: str = Field(pattern=_HASH_PATTERN)
    bundle_root: Path
    bundle: BundleDocument
    origins: tuple[CapabilityOrigin, ...] = Field(min_length=1)
    status: CapabilityStatus = CapabilityStatus.DISCOVERED
    diagnostic: CapabilityDiagnostic | None = None
    parameter_schema: dict[str, PydanticJsonValue] | None = None
    verification_environment_key: str | None = Field(default=None, pattern=_HASH_PATTERN)
    execution_identity: ExecutionIdentity | None = None
    worker_parameters: tuple[WorkerParameter, ...] | None = None

    @model_validator(mode="after")
    def canonicalize(self) -> Self:
        ordered = tuple(sorted(self.origins, key=lambda item: (item.channel, item.source_path)))
        if ordered != self.origins:
            object.__setattr__(self, "origins", ordered)
        pure_core = all(origin.channel == "core" for origin in ordered)
        non_core_native = self.bundle.capability.implementation.value == "native" and not pure_core
        if non_core_native and self.execution_identity is None:
            raise ValueError("non-core native capability requires an execution identity")
        if self.execution_identity is not None:
            if self.execution_identity.capability_id != self.capability_id:
                raise ValueError("execution identity capability ID does not match entry")
            if self.execution_identity.bundle_content_hash != self.content_hash:
                raise ValueError("execution identity bundle hash does not match entry")
            if self.execution_identity.callable_locator != self.bundle.contract.callable_locator:
                raise ValueError("execution identity callable locator does not match entry")
        return self


class CapabilityConflictMember(StrictSpecModel):
    """One origin/hash member of a same-ID content conflict."""

    content_hash: str = Field(pattern=_HASH_PATTERN)
    origin_channel: str = Field(pattern=_ORIGIN_PATTERN)
    source_path: str = Field(min_length=1)


class CapabilityConflict(StrictSpecModel):
    """All mutually excluded variants for one capability ID."""

    capability_id: str
    members: tuple[CapabilityConflictMember, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def canonicalize(self) -> Self:
        ordered = tuple(
            sorted(
                self.members,
                key=lambda item: (item.origin_channel, item.source_path, item.content_hash),
            )
        )
        if ordered != self.members:
            object.__setattr__(self, "members", ordered)
        return self


class CapabilityIndex(StrictSpecModel):
    """Immutable view over admitted, rejected, unsupported, and conflicted capabilities."""

    entries: tuple[CapabilityEntry, ...] = ()
    diagnostic_items: tuple[CapabilityDiagnostic, ...] = ()
    conflict_items: tuple[CapabilityConflict, ...] = ()

    @model_validator(mode="after")
    def canonicalize(self) -> Self:
        entries = tuple(sorted(self.entries, key=lambda item: item.capability_id))
        diagnostics = tuple(
            sorted(
                self.diagnostic_items,
                key=lambda item: (
                    item.code,
                    item.capability_id or "",
                    item.origins[0].source_path if item.origins else "",
                ),
            )
        )
        conflicts = tuple(sorted(self.conflict_items, key=lambda item: item.capability_id))
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "diagnostic_items", diagnostics)
        object.__setattr__(self, "conflict_items", conflicts)
        return self

    def list(self) -> tuple[CapabilityEntry, ...]:
        """Return admitted capabilities only, sorted by stable ID."""
        return tuple(item for item in self.entries if item.status is CapabilityStatus.ADMITTED)

    def describe(self, capability_id: str) -> CapabilityEntry:
        """Describe one unique bundle, retaining rejected/unsupported states."""
        if self.is_conflicted(capability_id):
            conflict = next(
                item for item in self.conflict_items if item.capability_id == capability_id
            )
            raise OrganelleContractError(
                code="capability.conflict",
                message=f"conflicting capability bundles declare {capability_id}",
                details=conflict.model_dump(mode="json"),
            )
        for entry in self.entries:
            if entry.capability_id == capability_id:
                return entry
        raise OrganelleInputError(
            code="input.unknown_capability",
            message=f"unknown capability: {capability_id}",
            details={"capability_id": capability_id},
        )

    def diagnostics(self) -> tuple[CapabilityDiagnostic, ...]:
        """Return every structured non-admission or discovery diagnostic."""
        return self.diagnostic_items

    def conflicts(self) -> tuple[CapabilityConflict, ...]:
        """Return every same-ID/different-content conflict."""
        return self.conflict_items

    def is_conflicted(self, capability_id: str) -> bool:
        return any(item.capability_id == capability_id for item in self.conflict_items)

    def rejection(self, capability_id: str) -> CapabilityDiagnostic | None:
        for entry in self.entries:
            if entry.capability_id == capability_id and entry.status in {
                CapabilityStatus.REJECTED,
                CapabilityStatus.UNSUPPORTED,
            }:
                return entry.diagnostic
        return None

    def binding_source(
        self,
        *,
        trust_store: TrustStore | None = None,
        environment_provider: EnvironmentParameterProvider | None = None,
        executor: BundleWorkerExecutor | None = None,
    ) -> IndexBindingSource:
        """Create the stateful lazy-binding adapter consumed by OperationRegistry."""
        return IndexBindingSource(
            self,
            trust_store=trust_store,
            environment_provider=environment_provider,
            executor=executor,
        )


class IndexBindingSource:
    """Stateful lazy resolver around one immutable CapabilityIndex."""

    def __init__(
        self,
        index: CapabilityIndex,
        *,
        trust_store: TrustStore | None = None,
        environment_provider: EnvironmentParameterProvider | None = None,
        executor: BundleWorkerExecutor | None = None,
    ) -> None:
        self._index = index
        self._trust_store = trust_store
        self._environment_provider = environment_provider
        self._executor = executor
        self._bindings: dict[str, BoundOperation] = {}
        self._lock = threading.RLock()

    def list_specs(self) -> tuple[OperationSpec, ...]:
        return tuple(entry.bundle.contract for entry in self._index.list())

    def known_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *(entry.capability_id for entry in self._index.entries),
                    *(conflict.capability_id for conflict in self._index.conflicts()),
                }
            )
        )

    def describe_spec(self, operation_id: str) -> OperationSpec | None:
        if self._index.is_conflicted(operation_id):
            return None
        for entry in self._index.entries:
            if entry.capability_id == operation_id:
                return entry.bundle.contract
        return None

    def parameter_schema(self, operation_id: str) -> dict[str, object] | None:
        for entry in self._index.entries:
            if (
                entry.capability_id == operation_id
                and entry.status is CapabilityStatus.ADMITTED
                and entry.parameter_schema is not None
            ):
                return cast(dict[str, object], deepcopy(entry.parameter_schema))
        return None

    def is_conflicted(self, operation_id: str) -> bool:
        return self._index.is_conflicted(operation_id)

    def conflict_details(self, operation_id: str) -> dict[str, object]:
        for conflict in self._index.conflicts():
            if conflict.capability_id == operation_id:
                return conflict.model_dump(mode="json")
        return {"capability_id": operation_id}

    def rejection(self, operation_id: str) -> tuple[str, str, dict[str, object]] | None:
        diagnostic = self._index.rejection(operation_id)
        if diagnostic is None:
            return None
        details = cast(dict[str, object], diagnostic.model_dump(mode="json"))
        return diagnostic.code, diagnostic.message, details

    def resolve(self, operation_id: str) -> BoundOperation | None:
        with self._lock:
            existing = self._bindings.get(operation_id)
            if existing is not None:
                return existing
            entry = next(
                (
                    item
                    for item in self._index.entries
                    if item.capability_id == operation_id
                    and item.status is CapabilityStatus.ADMITTED
                ),
                None,
            )
            if entry is None:
                return None
            if entry.parameter_schema is None:
                raise OrganelleContractError(
                    code="capability.verification_missing",
                    message=f"admitted capability has no frozen parameter schema: {operation_id}",
                )
            if not all(origin.channel == "core" for origin in entry.origins):
                from .trust import TrustStore

                store = self._trust_store or TrustStore()
                identity = entry.execution_identity
                if identity is None:
                    raise OrganelleContractError(
                        code="capability.execution_provider_required",
                        message=f"capability has no content-addressed execution provider: {operation_id}",
                        details={"capability_id": operation_id},
                    )
                parameters = entry.worker_parameters
                if parameters is None:
                    raise OrganelleContractError(
                        code="capability.verification_missing",
                        message=(
                            "admitted bundle-local capability has no frozen worker signature: "
                            f"{operation_id}"
                        ),
                        details={"capability_id": operation_id},
                    )
                from organelleverse.operations.python_binding import bind_worker_capability

                from .worker import OneShotBundleWorkerExecutor, WorkerInvocationStrategy

                executor = self._executor or OneShotBundleWorkerExecutor()
                binding = bind_worker_capability(
                    entry.bundle,
                    parameters,
                    entry.parameter_schema,
                    invocation_strategy=WorkerInvocationStrategy(entry, executor, store),
                )
                self._bindings[operation_id] = binding
                return binding
            locator = entry.bundle.contract.callable_locator
            if locator is None:
                raise OrganelleContractError(
                    code="capability.implementation_not_supported",
                    message=f"capability has no Python callable in v1: {operation_id}",
                )
            module_name, attribute_name = locator.split(":", 1)
            module = importlib.import_module(module_name)
            implementation = getattr(module, attribute_name, None)
            if not callable(implementation):
                raise OrganelleContractError(
                    code="capability.binding_invalid",
                    message=f"callable locator does not resolve to a callable: {locator}",
                    details={"operation_id": operation_id, "callable_locator": locator},
                )
            from organelleverse.capabilities.data_contracts import resolve_core_data_contracts
            from organelleverse.operations.python_binding import bind_python_capability

            binding = bind_python_capability(
                entry.bundle,
                implementation,
                frozen_schema=entry.parameter_schema,
                environment_provider=self._environment_provider,
                data_contracts=resolve_core_data_contracts(entry),
            )
            self._bindings[operation_id] = binding
            return binding


__all__ = [
    "CapabilityConflict",
    "CapabilityConflictMember",
    "CapabilityDiagnostic",
    "CapabilityEntry",
    "CapabilityIndex",
    "CapabilityOrigin",
    "CapabilityStatus",
    "IndexBindingSource",
]
