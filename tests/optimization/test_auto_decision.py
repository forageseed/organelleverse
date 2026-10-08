"""Pure, deterministic policy tests for execute-best decisions."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import JsonValue, ValidationError

from organelleverse.core.errors import OrganelleContractError
from organelleverse.optimization.auto import (
    AutoOptimizationContext,
    AutoOptimizationRequest,
    OperationInvocation,
    PluginInvocation,
    decide_auto_optimization,
)
from organelleverse.optimization.contracts_v3 import OptimizationBudgetV3
from tests.optimization.test_contracts_v3 import contract


def request(
    *,
    mode: str = "auto",
    parameters: dict[str, JsonValue] | None = None,
    budget: OptimizationBudgetV3 | None = None,
) -> AutoOptimizationRequest:
    return AutoOptimizationRequest(
        capability_id="demo.target",
        mode=mode,  # type: ignore[arg-type]
        invocation=OperationInvocation(
            input={"artifact": "opaque-input"},
            parameters={} if parameters is None else parameters,
        ),
        budget=budget,
    )


def context(**updates: object) -> AutoOptimizationContext:
    enabled = contract()
    enabled = enabled.model_copy(
        update={"adoption": enabled.adoption.model_copy(update={"approval": "automatic"})}
    )
    payload: dict[str, object] = {
        "request": request(),
        "profile_status": "enabled",
        "contract": enabled,
        "admitted": True,
        "identities_complete": True,
        "risk": "low",
        "risk_gate_satisfied": True,
        "environment_requires_choice": False,
        "adopted_study_id": None,
    }
    payload.update(updates)
    return AutoOptimizationContext.model_validate(payload)


@pytest.mark.parametrize(
    ("updates", "decision", "reason"),
    [
        ({"admitted": False}, "ask", "optimization.capability_not_admitted"),
        (
            {"request": request(parameters={"iterations": 4})},
            "baseline",
            "optimization.explicit_parameters_preserved",
        ),
        (
            {"request": request(mode="never")},
            "baseline",
            "optimization.mode_never",
        ),
        (
            {"adopted_study_id": "study-exact"},
            "reuse",
            "optimization.adopted_study_reused",
        ),
        (
            {"request": request(mode="always")},
            "optimize",
            "optimization.mode_always",
        ),
        ({}, "optimize", "optimization.auto_low_risk"),
        (
            {"risk": "medium"},
            "ask",
            "optimization.risk_approval_required",
        ),
        (
            {"request": request(mode="always"), "risk_gate_satisfied": False},
            "ask",
            "optimization.risk_approval_required",
        ),
    ],
)
def test_decision_table_uses_first_matching_rule_deterministically(
    updates: dict[str, object], decision: str, reason: str
) -> None:
    first = decide_auto_optimization(context(**updates))
    second = decide_auto_optimization(context(**deepcopy(updates)))

    assert first.decision == decision
    assert first.reason_code == reason
    assert first.model_dump_json() == second.model_dump_json()


def test_eligible_not_applicable_and_missing_identity_ask_without_optimization() -> None:
    eligible = decide_auto_optimization(
        context(profile_status="eligible", contract=None, identities_complete=False)
    )
    not_applicable = decide_auto_optimization(
        context(profile_status="not_applicable", contract=None)
    )
    missing = decide_auto_optimization(context(identities_complete=False))

    assert (eligible.decision, eligible.reason_code) == (
        "ask",
        "optimization.profile_eligible",
    )
    assert (not_applicable.decision, not_applicable.reason_code) == (
        "ask",
        "optimization.profile_not_applicable",
    )
    assert (missing.decision, missing.reason_code) == (
        "ask",
        "optimization.identity_incomplete",
    )


def test_only_explicit_tuned_parameters_take_precedence() -> None:
    fixed_only = decide_auto_optimization(
        context(request=request(parameters={"output_format": "gfa"}))
    )
    explicit_tuned = decide_auto_optimization(
        context(request=request(parameters={"iterations": 4, "output_format": "gfa"}))
    )
    forced = decide_auto_optimization(
        context(request=request(mode="always", parameters={"iterations": 4}))
    )

    assert fixed_only.decision == "optimize"
    assert explicit_tuned.decision == "baseline"
    assert forced.decision == "optimize"


def test_exact_adopted_match_precedes_risk_but_not_explicit_or_never() -> None:
    reused = decide_auto_optimization(context(adopted_study_id="study-exact", risk="high"))
    explicit = decide_auto_optimization(
        context(
            request=request(parameters={"iterations": 4}),
            adopted_study_id="study-exact",
        )
    )
    never = decide_auto_optimization(
        context(request=request(mode="never"), adopted_study_id="study-exact")
    )

    assert reused.decision == "reuse"
    assert explicit.decision == "baseline"
    assert never.decision == "baseline"


def test_auto_risk_and_environment_choice_require_approval() -> None:
    for risk in ("medium", "high"):
        decision = decide_auto_optimization(context(risk=risk))  # type: ignore[arg-type]
        assert decision.decision == "ask"
        assert decision.approval_required
    environment = decide_auto_optimization(context(environment_requires_choice=True))
    assert environment.decision == "ask"
    assert environment.reason_code == "optimization.environment_choice_required"


def test_human_required_adoption_is_high_risk_even_if_snapshot_claims_low() -> None:
    human_required = contract()
    decision = decide_auto_optimization(context(contract=human_required, risk="low"))

    assert decision.decision == "ask"
    assert decision.reason_code == "optimization.risk_approval_required"
    assert decision.approval_required


def test_requested_budget_may_only_narrow_every_contract_ceiling() -> None:
    original = contract().budget
    narrowed = OptimizationBudgetV3(
        max_trials=4,
        parallelism=1,
        max_wall_time_seconds=30,
        max_cpu_time_seconds=60,
        max_peak_memory_bytes=512,
    )
    assert (
        decide_auto_optimization(context(request=request(budget=narrowed))).decision == "optimize"
    )

    fields = (
        "max_trials",
        "parallelism",
        "max_wall_time_seconds",
        "max_cpu_time_seconds",
        "max_peak_memory_bytes",
    )
    for field in fields:
        expanded = original.model_copy(update={field: getattr(original, field) + 1})
        with pytest.raises(OrganelleContractError) as failure:
            decide_auto_optimization(context(request=request(budget=expanded)))
        assert failure.value.code == "optimization.budget_escalation"


def test_reuse_digests_separate_input_fixed_and_tuned_parameters() -> None:
    first = decide_auto_optimization(
        context(request=request(mode="always", parameters={"iterations": 2, "format": "gfa"}))
    )
    tuned_changed = decide_auto_optimization(
        context(request=request(mode="always", parameters={"iterations": 8, "format": "gfa"}))
    )
    fixed_changed = decide_auto_optimization(
        context(request=request(mode="always", parameters={"iterations": 2, "format": "vg"}))
    )

    assert first.invocation_digest == tuned_changed.invocation_digest
    assert first.fixed_parameters_digest == tuned_changed.fixed_parameters_digest
    assert first.fixed_parameters_digest != fixed_changed.fixed_parameters_digest


@pytest.mark.parametrize(
    "field",
    ["approved", "approval", "risk", "score", "adopted_parameters", "contract_digest"],
)
def test_request_forbids_governance_and_adoption_overrides(field: str) -> None:
    payload = request().model_dump(mode="python")
    payload[field] = True
    with pytest.raises(ValidationError):
        AutoOptimizationRequest.model_validate(payload)


def test_invocation_is_a_closed_discriminated_union() -> None:
    plugin = AutoOptimizationRequest(
        capability_id="demo.target",
        invocation=PluginInvocation(inputs={"reads": "artifact:reads"}, parameters={}),
    )
    assert plugin.invocation.kind == "plugin"

    payload = request().model_dump(mode="python")
    payload["invocation"]["kind"] = "unknown"
    with pytest.raises(ValidationError):
        AutoOptimizationRequest.model_validate(payload)
