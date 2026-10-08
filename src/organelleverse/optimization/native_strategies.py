"""Admitted native capability adapters for the shared bounded search algorithms."""

from __future__ import annotations

from pydantic import Field

from organelleverse.core.frozen import thaw_json
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.spec import StrictSpecModel

from .strategies import strategy_for
from .strategy_models import (
    SearchObservation,
    StrategyBinding,
    StrategySpec,
    StrategyState,
    SuccessiveHalvingState,
)


class StrategyInput(StrictSpecModel):
    spec: StrategySpec
    binding: StrategyBinding
    state: StrategyState | SuccessiveHalvingState | None = None
    observations: tuple[SearchObservation, ...] = ()
    remaining_budget: int = Field(ge=0, le=256)


def _propose(result: OrganelleResult, kind: str) -> OrganelleResult:
    payload = StrategyInput.model_validate(thaw_json(result.metrics["optimization_strategy"]))
    if payload.binding.kind != kind:
        raise ValueError("Strategy capability and requested strategy kind must match")
    # The halving base is selected only from the built-in algorithms. The engine
    # separately requires admission and exact bindings for every configured kind.
    strategy = strategy_for(kind, base_resolver=lambda binding: strategy_for(binding.kind))
    state = payload.state or strategy.initialize(payload.spec, payload.binding)
    if state.spec != payload.spec or state.binding != payload.binding:
        raise ValueError("Persisted strategy state must match its closed specification")
    if payload.observations:
        state = strategy.observe(state, payload.observations)
    batch = strategy.propose(state, (), payload.remaining_budget)
    return OrganelleResult(
        operation_id="optimization." + kind,
        operation_version="1.0",
        scope=result.scope,
        status="ok",
        metrics={"optimization_proposals": batch.model_dump(mode="json")},
    )


def grid(result: OrganelleResult) -> OrganelleResult:
    return _propose(result, "grid")


def random(result: OrganelleResult) -> OrganelleResult:
    return _propose(result, "random")


def space_filling(result: OrganelleResult) -> OrganelleResult:
    return _propose(result, "space_filling")


def successive_halving(result: OrganelleResult) -> OrganelleResult:
    return _propose(result, "successive_halving")
