"""Single governed orchestration boundary for execute-best calls."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import Field, JsonValue, StrictStr, model_validator

from organelleverse.capabilities.index import CapabilityEntry, CapabilityIndex, CapabilityStatus
from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.spec import SideEffect, StrictSpecModel
from organelleverse.optimization.auto import (
    AutoOptimizationContext,
    AutoOptimizationDecision,
    AutoOptimizationRequest,
    OperationInvocation,
    PluginInvocation,
    decide_auto_optimization,
)
from organelleverse.optimization.contracts_v3 import OptimizationProfileV3
from organelleverse.optimization.profiles_v3 import identity_from_capability_entry
from organelleverse.optimization.strategies import candidate_digest
from organelleverse.optimization.strategy_models import AdaptiveStudyRequest
from organelleverse.optimization.study_models import AdoptedStudyIdentity

from .adaptive_models import AdaptiveStudyRecord, FinalExecutionClaim, FinalExecutionLink, OpaqueId


class AdaptiveStudyPort(Protocol):
    def find_adopted(self, identity: AdoptedStudyIdentity) -> AdaptiveStudyRecord | None: ...

    def run_to_terminal(self, request: AdaptiveStudyRequest) -> AdaptiveStudyRecord: ...

    def record_final_execution(
        self, study_id: str, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord: ...

    def claim_final_execution(
        self, study_id: str, claim: FinalExecutionClaim
    ) -> AdaptiveStudyRecord: ...

    def complete_final_execution(
        self, study_id: str, claim_id: str, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord: ...


@dataclass(frozen=True)
class PreparedAdaptiveStudy:
    """A caller-supplied real evaluator runtime and its closed request."""

    engine: AdaptiveStudyPort
    request: AdaptiveStudyRequest


@dataclass(frozen=True)
class AutoOptimizationSnapshot:
    """Current admission/profile facts passed to the runtime provider."""

    index: CapabilityIndex
    profile: OptimizationProfileV3
    risk: Literal["low", "medium", "high"]


class FinalExecutionReceipt(StrictSpecModel):
    run_id: OpaqueId
    status: Literal["succeeded", "failed"]


class ExecuteBestResult(StrictSpecModel):
    decision: AutoOptimizationDecision
    parameter_digest: StrictStr | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    study_id: OpaqueId | None = None
    run_id: OpaqueId | None = None
    execution_status: Literal["succeeded", "failed"] | None = None

    @model_validator(mode="after")
    def _closed_execution(self) -> ExecuteBestResult:
        fields = (self.parameter_digest, self.run_id, self.execution_status)
        if self.decision.decision == "ask":
            if any(value is not None for value in fields):
                raise ValueError("ask decisions cannot carry execution receipts")
        elif self.parameter_digest is None or self.run_id is None:
            raise ValueError("executable decisions require a parameter digest and final receipt")
        return self


PreparedStudyProvider = Callable[
    [AutoOptimizationRequest, AutoOptimizationSnapshot], PreparedAdaptiveStudy | None
]
FinalExecutor = Callable[
    [str, OperationInvocation | PluginInvocation, dict[str, JsonValue]], FinalExecutionReceipt
]


class AutoOptimizationService:
    """Decide, optionally study, then execute one exact parameter selection."""

    def __init__(
        self,
        *,
        index_provider: Callable[[], CapabilityIndex],
        profile_provider: Callable[[], Mapping[str, OptimizationProfileV3]],
        prepared_study_provider: PreparedStudyProvider | None,
        final_executor: FinalExecutor,
        final_link_validator: Callable[[FinalExecutionLink], None] | None = None,
    ) -> None:
        self._index_provider = index_provider
        self._profile_provider = profile_provider
        self._prepared_study_provider = prepared_study_provider
        self._final_executor = final_executor
        self._final_link_validator = final_link_validator

    def profiles(self) -> Mapping[str, OptimizationProfileV3]:
        """Return the same live profile catalog used by execute-best decisions."""
        return self._profile_provider()

    def execute(
        self,
        request: AutoOptimizationRequest,
        *,
        risk_gate_satisfied: bool = False,
    ) -> ExecuteBestResult:
        index = self._index_provider()
        profile = self._profile_provider().get(request.capability_id)
        context, snapshot = _context(
            request,
            index,
            profile,
            risk_gate_satisfied=risk_gate_satisfied,
        )
        decision = decide_auto_optimization(context)
        if decision.decision == "ask":
            return ExecuteBestResult(decision=decision)
        if decision.approval_required and not risk_gate_satisfied:
            return ExecuteBestResult(decision=_ask(decision, "optimization.risk_approval_required"))
        if decision.decision == "baseline":
            assert context.contract is not None
            parameters = {
                **context.contract.baseline.parameters,
                **request.invocation.parameters,
            }
            return self._execute_direct(request, decision, parameters)

        provider = self._prepared_study_provider
        if provider is None or snapshot is None:
            return ExecuteBestResult(decision=_ask(decision, "optimization.runtime_unavailable"))
        prepared = provider(request, snapshot)
        if prepared is None:
            return ExecuteBestResult(decision=_ask(decision, "optimization.runtime_unavailable"))
        _validate_prepared(prepared.request, context)
        adopted = prepared.engine.find_adopted(AdoptedStudyIdentity.from_request(prepared.request))
        if adopted is not None:
            decision = decide_auto_optimization(
                context.model_copy(update={"adopted_study_id": adopted.study_id})
            )
            return self._execute_record(request, prepared, adopted, decision)

        terminal = prepared.engine.run_to_terminal(prepared.request)
        if terminal.execution_selection is None:
            raise OrganelleContractError(
                code="optimization.study_not_executable",
                message="adaptive study produced no verified execution selection",
                details={"study_id": terminal.study_id},
            )
        decision = decision.model_copy(update={"study_id": terminal.study_id})
        if terminal.status == "baseline_retained":
            decision = decision.model_copy(update={"fallback": _fallback_for(terminal)})
        return self._execute_record(request, prepared, terminal, decision)

    def _execute_direct(
        self,
        request: AutoOptimizationRequest,
        decision: AutoOptimizationDecision,
        parameters: dict[str, JsonValue],
    ) -> ExecuteBestResult:
        digest = candidate_digest(parameters)
        invocation = request.invocation.model_copy(update={"parameters": parameters})
        receipt = self._final_executor(request.capability_id, invocation, parameters)
        return ExecuteBestResult(
            decision=decision,
            parameter_digest=digest,
            run_id=receipt.run_id,
            execution_status=receipt.status,
        )

    def _execute_record(
        self,
        request: AutoOptimizationRequest,
        prepared: PreparedAdaptiveStudy,
        record: AdaptiveStudyRecord,
        decision: AutoOptimizationDecision,
    ) -> ExecuteBestResult:
        selection = record.execution_selection
        if selection is None:
            raise OrganelleContractError(
                code="optimization.study_not_executable",
                message="adaptive study has no verified execution selection",
                details={"study_id": record.study_id},
            )
        if candidate_digest(selection.parameters) != selection.parameter_digest:
            raise OrganelleContractError(
                code="optimization.adoption_tampered",
                message="execution selection no longer matches its canonical digest",
                details={"study_id": record.study_id},
            )
        existing = record.final_execution_link
        if existing is not None:
            if self._final_link_validator is not None:
                self._final_link_validator(existing)
            return ExecuteBestResult(
                decision=decision,
                parameter_digest=selection.parameter_digest,
                study_id=record.study_id,
                run_id=existing.run_receipt_id,
                execution_status=existing.execution_status,
            )
        claim_id = f"claim-{uuid4().hex}"
        claimed = prepared.engine.claim_final_execution(
            record.study_id,
            FinalExecutionClaim(
                claim_id=claim_id,
                executed_parameter_digest=selection.parameter_digest,
                claimed_at=datetime.now(UTC),
            ),
        )
        if claimed.final_execution_link is not None:
            linked = claimed.final_execution_link
            if self._final_link_validator is not None:
                self._final_link_validator(linked)
            return ExecuteBestResult(
                decision=decision,
                parameter_digest=selection.parameter_digest,
                study_id=record.study_id,
                run_id=linked.run_receipt_id,
                execution_status=linked.execution_status,
            )
        if (
            claimed.final_execution_claim is None
            or claimed.final_execution_claim.claim_id != claim_id
        ):
            raise OrganelleContractError(
                code="optimization.final_execution_in_progress",
                message="another caller owns the durable final execution claim",
                details={"study_id": record.study_id},
            )
        parameters = {**request.invocation.parameters, **selection.parameters}
        invocation = request.invocation.model_copy(update={"parameters": parameters})
        receipt = self._final_executor(request.capability_id, invocation, parameters)
        link = FinalExecutionLink(
            run_receipt_id=receipt.run_id,
            executed_parameter_digest=selection.parameter_digest,
            linked_at=datetime.now(UTC),
            execution_status=receipt.status,
        )
        if self._final_link_validator is not None:
            self._final_link_validator(link)
        prepared.engine.complete_final_execution(record.study_id, claim_id, link)
        return ExecuteBestResult(
            decision=decision,
            parameter_digest=selection.parameter_digest,
            study_id=record.study_id,
            run_id=receipt.run_id,
            execution_status=receipt.status,
        )


def _context(
    request: AutoOptimizationRequest,
    index: CapabilityIndex,
    profile: OptimizationProfileV3 | None,
    *,
    risk_gate_satisfied: bool,
) -> tuple[AutoOptimizationContext, AutoOptimizationSnapshot | None]:
    admitted = {
        entry.capability_id: entry
        for entry in index.list()
        if entry.status is CapabilityStatus.ADMITTED
    }
    if profile is None:
        context = AutoOptimizationContext(
            request=request,
            profile_status="not_applicable",
            contract=None,
            admitted=request.capability_id in admitted,
            identities_complete=False,
            risk=None,
            risk_gate_satisfied=risk_gate_satisfied,
            environment_requires_choice=False,
        )
        return context, None
    contract = profile.contract
    identities_complete = False
    risk: Literal["low", "medium", "high"] | None = None
    if contract is not None:
        required = (
            contract.target,
            contract.evaluator.identity,
            *(reference.identity for reference in contract.strategies),
        )
        resolved = tuple(_exact_entry(admitted, identity) for identity in required)
        identities_complete = all(entry is not None for entry in resolved)
        if identities_complete:
            risk = _risk(tuple(entry for entry in resolved if entry is not None))
    context = AutoOptimizationContext(
        request=request,
        profile_status=profile.status,
        contract=contract,
        admitted=request.capability_id in admitted,
        identities_complete=identities_complete,
        risk=risk,
        risk_gate_satisfied=risk_gate_satisfied,
        environment_requires_choice=False,
    )
    snapshot = None if risk is None else AutoOptimizationSnapshot(index, profile, risk)
    return context, snapshot


def _exact_entry(
    admitted: Mapping[str, CapabilityEntry], identity: object
) -> CapabilityEntry | None:
    capability_id = getattr(identity, "capability_id", None)
    entry = admitted.get(capability_id) if isinstance(capability_id, str) else None
    if entry is None:
        return None
    return entry if identity_from_capability_entry(entry) == identity else None


def _risk(entries: tuple[CapabilityEntry, ...]) -> Literal["low", "medium", "high"]:
    effects = {effect for entry in entries for effect in entry.bundle.contract.side_effects}
    if effects.intersection(
        {SideEffect.WRITE_FILES, SideEffect.SUBPROCESS, SideEffect.NETWORK, SideEffect.GPU}
    ):
        return "high"
    if SideEffect.READ_FILES in effects:
        return "medium"
    return "low"


def _validate_prepared(request: AdaptiveStudyRequest, context: AutoOptimizationContext) -> None:
    decision = decide_auto_optimization(context)
    expected_contract = context.contract
    if expected_contract is not None and context.request.budget is not None:
        expected_contract = expected_contract.model_copy(update={"budget": context.request.budget})
    if (
        expected_contract is None
        or request.contract != expected_contract
        or request.contract.target.capability_id != context.request.capability_id
        or request.input_digest != decision.invocation_digest
        or request.fixed_parameters_digest != decision.fixed_parameters_digest
    ):
        raise OrganelleContractError(
            code="optimization.prepared_study_mismatch",
            message="prepared adaptive study does not match the current execute-best request",
        )


def _ask(decision: AutoOptimizationDecision, reason: str) -> AutoOptimizationDecision:
    return decision.model_copy(
        update={
            "decision": "ask",
            "reason_code": reason,
            "study_id": None,
            "fallback": None,
        }
    )


def _fallback_for(
    record: AdaptiveStudyRecord,
) -> Literal[
    "validation_not_improved",
    "study_failed",
    "cancelled",
    "budget_exhausted",
    "no_valid_trial",
]:
    assert record.decision is not None
    reason = record.decision.reason_code
    if reason == "optimization.insufficient_improvement":
        return "validation_not_improved"
    if reason == "optimization.budget_exhausted":
        return "budget_exhausted"
    if reason in {"optimization.no_feasible_candidate", "optimization.validation_incomplete"}:
        return "no_valid_trial"
    if reason in {"optimization.cancelled", "optimization.study_cancelled"}:
        return "cancelled"
    return "study_failed"


__all__ = [
    "AutoOptimizationService",
    "AutoOptimizationSnapshot",
    "ExecuteBestResult",
    "FinalExecutionReceipt",
    "PreparedAdaptiveStudy",
]
