"""Typed, replayable successive-halving search transitions."""

from __future__ import annotations

import math
from collections.abc import Callable

from organelleverse.core.errors import OrganelleContractError

from .strategies import candidate_digest
from .strategy_models import (
    CandidateProposal,
    ProposalBatch,
    SearchObservation,
    SearchStrategy,
    StrategyBinding,
    StrategySpec,
    StrategyState,
    SuccessiveHalvingConfig,
    SuccessiveHalvingState,
)


class SuccessiveHalvingStrategy:
    """Advance complete cohorts across declared resource rungs."""

    def __init__(self, resolve_base: Callable[[StrategyBinding], SearchStrategy]) -> None:
        self._resolve_base = resolve_base

    def initialize(self, spec: StrategySpec, binding: StrategyBinding) -> SuccessiveHalvingState:
        config = spec.successive_halving
        if binding.kind != "successive_halving" or config is None:
            raise _proposal_error("successive-halving requires its exact binding and config")
        base_parameters = tuple(
            domain for domain in spec.parameters if domain.name != config.resource_parameter
        )
        if len(base_parameters) == len(spec.parameters) or not base_parameters:
            raise _proposal_error("resource parameter must be a controlled non-empty search axis")
        base_spec = StrategySpec(
            parameters=base_parameters,
            seed=spec.seed,
            total_design_size=spec.total_design_size,
        )
        base = self._resolve_base(config.base_strategy)
        if isinstance(base, SuccessiveHalvingStrategy):
            raise _proposal_error("successive-halving base strategy cannot be recursive")
        base_state = base.initialize(base_spec, config.base_strategy)
        cohort = base.propose(base_state, (), spec.total_design_size).proposals
        return SuccessiveHalvingState(
            binding=binding,
            spec=spec,
            config=config,
            cohort=cohort,
        )

    def propose(
        self,
        state: StrategyState | SuccessiveHalvingState,
        observations: tuple[SearchObservation, ...],
        remaining_budget: int,
    ) -> ProposalBatch:
        if not isinstance(state, SuccessiveHalvingState):
            raise _proposal_error("successive-halving requires typed durable state")
        if observations or state.phase == "awaiting":
            raise _proposal_error("successive-halving must observe the current complete rung first")
        if state.phase == "complete":
            return ProposalBatch(proposals=(), state=state, exhausted=True)
        rung = state.config.rungs[state.rung_index]
        assignments = tuple(self._assignment(item, state, rung) for item in state.cohort)
        if len(assignments) > remaining_budget:
            raise OrganelleContractError(
                code="optimization.budget_exhausted",
                message="remaining budget cannot cover the complete successive-halving rung",
            )
        updated = state.model_copy(update={"phase": "awaiting", "current_assignments": assignments})
        return ProposalBatch(proposals=assignments, state=updated)

    @staticmethod
    def _assignment(
        item: CandidateProposal, state: SuccessiveHalvingState, rung: int
    ) -> CandidateProposal:
        parameters = {**item.parameters, state.config.resource_parameter: rung}
        return CandidateProposal(
            parameters=parameters,
            candidate_digest=candidate_digest(parameters),
            base_candidate_digest=item.candidate_digest,
            resource_rung=rung,
        )

    def observe(
        self,
        state: StrategyState | SuccessiveHalvingState,
        observations: tuple[SearchObservation, ...],
    ) -> SuccessiveHalvingState:
        if not isinstance(state, SuccessiveHalvingState) or state.phase != "awaiting":
            raise _proposal_error("successive-halving observations require an awaiting rung")
        expected = {
            (item.candidate_digest, item.base_candidate_digest, item.resource_rung)
            for item in state.current_assignments
        }
        observed = {
            (item.candidate_digest, item.base_candidate_digest, item.resource_rung)
            for item in observations
        }
        if len(observations) != len(observed) or observed != expected:
            raise _proposal_error(
                "successive-halving requires exactly one observation per assignment"
            )
        promoted = promote_successive_halving(
            observations,
            config=state.config,
            objective_order=state.spec.objective_order,
        )
        exact_to_base = {item.candidate_digest: item.base_candidate_digest for item in observations}
        promoted_base_items: list[str] = []
        for exact_digest in promoted:
            base_digest = exact_to_base[exact_digest]
            if base_digest is None:
                raise _proposal_error("successive-halving observation lacks its base identity")
            promoted_base_items.append(base_digest)
        promoted_base = tuple(promoted_base_items)
        completed = (
            *state.completed_assignment_digests,
            *(item.candidate_digest for item in observations),
        )
        common = {
            "completed_assignment_digests": completed,
            "promoted": promoted_base,
        }
        if not promoted_base or state.rung_index + 1 == len(state.config.rungs):
            return state.model_copy(
                update={"phase": "complete", "current_assignments": (), **common}
            )
        cohort_by_digest = {item.candidate_digest: item for item in state.cohort}
        return state.model_copy(
            update={
                "phase": "ready",
                "rung_index": state.rung_index + 1,
                "cohort": tuple(cohort_by_digest[item] for item in promoted_base),
                "current_assignments": (),
                **common,
            }
        )


def promote_successive_halving(
    observations: tuple[SearchObservation, ...],
    *,
    config: SuccessiveHalvingConfig,
    objective_order: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    """Select the bounded Pareto cohort using only declared objective order."""

    feasible = [
        observation
        for observation in observations
        if observation.feasible
        and not observation.failed
        and all(name in observation.metrics for name, _ in objective_order)
    ]
    original_count = len(feasible)
    front = [
        candidate
        for candidate in feasible
        if not any(
            other is not candidate and _observation_dominates(other, candidate, objective_order)
            for other in feasible
        )
    ]

    def rank(observation: SearchObservation) -> tuple[float | str, ...]:
        values: list[float | str] = []
        for name, direction in objective_order:
            value = observation.metrics[name]
            values.append(-value if direction == "maximize" else value)
        values.append(observation.base_candidate_digest or observation.candidate_digest)
        return tuple(values)

    front.sort(key=rank)
    retained = min(math.ceil(original_count / config.eta), len(front))
    return tuple(item.candidate_digest for item in front[:retained])


def _observation_dominates(
    left: SearchObservation,
    right: SearchObservation,
    objective_order: tuple[tuple[str, str], ...],
) -> bool:
    no_worse = True
    better = False
    for name, direction in objective_order:
        left_value = left.metrics[name]
        right_value = right.metrics[name]
        if direction == "maximize":
            no_worse &= left_value >= right_value
            better |= left_value > right_value
        else:
            no_worse &= left_value <= right_value
            better |= left_value < right_value
    return no_worse and better


def _proposal_error(message: str) -> OrganelleContractError:
    return OrganelleContractError(
        code="optimization.strategy_proposal_invalid",
        message=message,
    )


__all__ = ["SuccessiveHalvingStrategy", "promote_successive_halving"]
