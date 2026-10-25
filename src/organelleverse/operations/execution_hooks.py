"""Optional execution hooks supplied by a caller of the scientific API.

OrganelleVerse describes scientific requirements and runs locally by default.
An embedding application may install an executor; target selection, transport,
credentials and task records remain entirely with that application. Importing
this module never discovers an application, reads machine config or probes.
"""
from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from pydantic import Field

from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.operations.spec import SideEffect, StrictSpecModel


class ExecutionRequirement(StrictSpecModel):
    """Scientific execution needs, without connection or scheduling details."""

    reason: Literal['missing_dependencies', 'environment_unavailable', 'unsupported_platform']
    operation_id: str | None = None
    backend_id: str | None = None
    required_platform: str = 'linux-64'
    details: dict[str, Any] = Field(default_factory=dict)


class ExecutionDelegated(Exception):
    """Unwind local preparation so the embedding caller can execute the request.

    The opaque selection is meaningful only to the installed hook. OV neither
    validates its transport nor reads its attributes.
    """

    def __init__(self, selection: object, *, reason: str = '') -> None:
        super().__init__(reason or 'execution delegated to the calling application')
        self.selection = selection
        self.reason = reason


class ExecutionHooks(Protocol):
    def select(self, requirement: ExecutionRequirement) -> object | None: ...

    def execute(self, request: Mapping[str, object], *, registry: Any,
                granted: Collection[SideEffect], selection: object) -> dict[str, object]: ...


@dataclass(frozen=True)
class ExecutionPolicy:
    local_only: bool = False
    copy_staged_inputs: bool = False


_DEFAULT_HOOKS: ExecutionHooks | None = None
_HOOKS: ContextVar[ExecutionHooks | None] = ContextVar('ov_execution_hooks', default=None)
_HOOKS_SCOPED: ContextVar[bool] = ContextVar('ov_execution_hooks_scoped', default=False)
_POLICY: ContextVar[ExecutionPolicy] = ContextVar('ov_execution_policy', default=ExecutionPolicy())
_PROCESS_POLICY = ExecutionPolicy()
_UNSET = object()


def install_execution_hooks(hooks: ExecutionHooks | None) -> ExecutionHooks | None:
    """Install the caller-owned executor at application boot; return the prior one.

    A library-only process leaves this unset. Per-request execution_scope can
    override it without mutating another conversation's policy.
    """
    global _DEFAULT_HOOKS
    previous, _DEFAULT_HOOKS = _DEFAULT_HOOKS, hooks
    return previous


def set_process_execution_policy(policy: ExecutionPolicy) -> ExecutionPolicy:
    """Set a dedicated worker's policy, including its native capability threads.

    This is for a process that owns one admitted invocation. Application request
    handlers use execution_scope instead of changing this process-wide value.
    """
    global _PROCESS_POLICY
    previous, _PROCESS_POLICY = _PROCESS_POLICY, policy
    return previous


def local_execution_required() -> bool:
    return _PROCESS_POLICY.local_only or _POLICY.get().local_only


def staged_input_copy_required() -> bool:
    return _PROCESS_POLICY.copy_staged_inputs or _POLICY.get().copy_staged_inputs


def active_execution_hooks() -> ExecutionHooks | None:
    return _HOOKS.get() if _HOOKS_SCOPED.get() else _DEFAULT_HOOKS


@contextmanager
def execution_scope(*, hooks: ExecutionHooks | None | object = _UNSET,
                    local_only: bool | None = None,
                    copy_staged_inputs: bool | None = None) -> Iterator[None]:
    policy = _POLICY.get()
    changes = {}
    if local_only is not None:
        changes['local_only'] = local_only
    if copy_staged_inputs is not None:
        changes['copy_staged_inputs'] = copy_staged_inputs
    token = _POLICY.set(replace(policy, **changes))
    hook_token = scoped_token = None
    if hooks is not _UNSET:
        hook_token = _HOOKS.set(hooks)  # type: ignore[arg-type]
        scoped_token = _HOOKS_SCOPED.set(True)
    try:
        yield
    finally:
        _POLICY.reset(token)
        if hook_token is not None:
            _HOOKS.reset(hook_token)
            _HOOKS_SCOPED.reset(scoped_token)


def select_execution(requirement: ExecutionRequirement) -> object | None:
    hooks = active_execution_hooks()
    if local_execution_required() or hooks is None:
        return None
    return hooks.select(requirement)


def execute_delegated(request: Mapping[str, object], *, registry: Any,
                      granted: Collection[SideEffect], selection: object) -> dict[str, object]:
    hooks = active_execution_hooks()
    if local_execution_required() or hooks is None:
        raise OrganelleDependencyError(
            code='execution.host_required',
            message='this execution selection requires the caller that created it; no executor is installed',
        )
    return hooks.execute(request, registry=registry, granted=granted, selection=selection)
