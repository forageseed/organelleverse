"""Pure governed decision policy for the execute-best entry point."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, StrictStr, model_validator

from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.spec import StrictSpecModel

from .contracts_v3 import OptimizationBudgetV3, OptimizationContractV3
from .models import _canonical_json  # pyright: ignore[reportPrivateUsage]

Digest = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
SafeText = Annotated[StrictStr, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]


class OperationInvocation(StrictSpecModel):
    kind: Literal["operation"] = "operation"
    input: JsonValue
    parameters: dict[StrictStr, JsonValue]


class PluginInvocation(StrictSpecModel):
    kind: Literal["plugin"] = "plugin"
    inputs: dict[StrictStr, StrictStr]
    parameters: dict[StrictStr, JsonValue]


Invocation = Annotated[OperationInvocation | PluginInvocation, Field(discriminator="kind")]


class AutoOptimizationRequest(StrictSpecModel):
    capability_id: SafeText
    invocation: Invocation
    mode: Literal["never", "always", "auto"] = "auto"
    budget: OptimizationBudgetV3 | None = None


class AutoOptimizationContext(StrictSpecModel):
    """Trusted snapshot inputs; none are caller-set governance overrides."""

    request: AutoOptimizationRequest
    profile_status: Literal["enabled", "eligible", "not_applicable"]
    contract: OptimizationContractV3 | None
    admitted: bool
    identities_complete: bool
    risk: Literal["low", "medium", "high"] | None
    risk_gate_satisfied: bool
    environment_requires_choice: bool
    adopted_study_id: SafeText | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.profile_status == "enabled":
            if self.contract is None:
                raise ValueError("enabled auto-optimization context requires a contract")
            if self.contract.target.capability_id != self.request.capability_id:
                raise ValueError("request capability must match the enabled contract target")
        elif self.contract is not None:
            raise ValueError("non-enabled auto-optimization context cannot carry a contract")
        return self


class AutoOptimizationDecision(StrictSpecModel):
    decision: Literal["reuse", "optimize", "baseline", "ask"]
    reason_code: SafeText
    capability_id: SafeText
    contract_digest: Digest | None
    benchmark_digest: Digest | None
    invocation_digest: Digest
    fixed_parameters_digest: Digest
    study_id: SafeText | None = None
    approval_required: bool
    fallback: (
        Literal[
            "validation_not_improved",
            "study_failed",
            "cancelled",
            "budget_exhausted",
            "no_valid_trial",
        ]
        | None
    ) = None


def decide_auto_optimization(context: AutoOptimizationContext) -> AutoOptimizationDecision:
    """Apply the frozen eight-rule table without I/O or heuristic scoring."""

    request = context.request
    contract = context.contract
    if not context.admitted:
        return _decision(context, "ask", "optimization.capability_not_admitted")
    if context.profile_status == "not_applicable":
        return _decision(context, "ask", "optimization.profile_not_applicable")
    if context.profile_status == "eligible":
        return _decision(context, "ask", "optimization.profile_eligible")
    assert contract is not None
    if not context.identities_complete:
        return _decision(context, "ask", "optimization.identity_incomplete")
    if context.risk is None:
        return _decision(context, "ask", "optimization.risk_identity_missing")
    effective_risk = _effective_risk(context)
    if not _surface_matches(request, contract):
        raise _error(
            "optimization.invocation_surface_mismatch",
            "invocation surface does not match the admitted target",
        )
    _require_narrowed_budget(request.budget, contract.budget)

    tuned = {parameter.name for parameter in contract.parameters}
    if request.mode != "always" and tuned.intersection(request.invocation.parameters):
        return _decision(context, "baseline", "optimization.explicit_parameters_preserved")
    if request.mode == "never":
        return _decision(context, "baseline", "optimization.mode_never")
    if context.adopted_study_id is not None:
        return _decision(
            context,
            "reuse",
            "optimization.adopted_study_reused",
            study_id=context.adopted_study_id,
        )
    if request.mode == "always" and context.risk_gate_satisfied:
        return _decision(context, "optimize", "optimization.mode_always")
    if (
        request.mode == "auto"
        and effective_risk == "low"
        and not context.environment_requires_choice
    ):
        return _decision(context, "optimize", "optimization.auto_low_risk")
    if request.mode == "auto" and context.environment_requires_choice:
        return _decision(context, "ask", "optimization.environment_choice_required")
    if request.mode == "auto" and effective_risk in {"medium", "high"}:
        return _decision(context, "ask", "optimization.risk_approval_required")
    if request.mode == "always" and not context.risk_gate_satisfied:
        return _decision(context, "ask", "optimization.risk_approval_required")
    return _decision(context, "baseline", "optimization.baseline_default")


def _decision(
    context: AutoOptimizationContext,
    decision: Literal["reuse", "optimize", "baseline", "ask"],
    reason: str,
    *,
    study_id: str | None = None,
) -> AutoOptimizationDecision:
    contract = context.contract
    invocation = context.request.invocation
    approval = context.environment_requires_choice or _effective_risk(context) in {
        "medium",
        "high",
    }
    tuned: set[str] = (
        set() if contract is None else {parameter.name for parameter in contract.parameters}
    )
    fixed = {name: value for name, value in invocation.parameters.items() if name not in tuned}
    invocation_input = invocation.input if invocation.kind == "operation" else invocation.inputs
    return AutoOptimizationDecision(
        decision=decision,
        reason_code=reason,
        capability_id=context.request.capability_id,
        contract_digest=None if contract is None else contract.digest,
        benchmark_digest=(
            None
            if contract is None
            else _canonical_digest(contract.benchmark.model_dump(mode="json"))
        ),
        invocation_digest=_canonical_digest(invocation_input),
        fixed_parameters_digest=_canonical_digest(fixed),
        study_id=study_id,
        approval_required=approval,
    )


def _surface_matches(request: AutoOptimizationRequest, contract: OptimizationContractV3) -> bool:
    expected = "plugin" if contract.target.surface == "plugin" else "operation"
    return request.invocation.kind == expected


def _effective_risk(
    context: AutoOptimizationContext,
) -> Literal["low", "medium", "high"] | None:
    if context.contract is not None and context.contract.adoption.approval == "human_required":
        return "high"
    return context.risk


def _require_narrowed_budget(
    requested: OptimizationBudgetV3 | None, declared: OptimizationBudgetV3
) -> None:
    if requested is None:
        return
    fields = (
        "max_trials",
        "parallelism",
        "max_wall_time_seconds",
        "max_cpu_time_seconds",
        "max_peak_memory_bytes",
    )
    if any(getattr(requested, field) > getattr(declared, field) for field in fields):
        raise _error(
            "optimization.budget_escalation",
            "requested optimization budget exceeds the admitted contract",
        )


def _error(code: str, message: str) -> OrganelleContractError:
    return OrganelleContractError(code=code, message=message)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        _canonical_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "AutoOptimizationContext",
    "AutoOptimizationDecision",
    "AutoOptimizationRequest",
    "OperationInvocation",
    "PluginInvocation",
    "decide_auto_optimization",
]
