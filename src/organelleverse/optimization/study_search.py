"""Replayable strategy checkpoint orchestration for adaptive search."""

from __future__ import annotations

from typing import Literal

from pydantic import StrictStr

from organelleverse.core.errors import OrganelleContractError
from organelleverse.plugin_experiments.adaptive_models import (
    AdaptiveStudyRecord,
    CandidateAggregate,
)
from organelleverse.plugin_experiments.store import ExperimentStore

from .checkpoint_validation import validate_checkpoint_replay
from .evaluator_dispatch import evaluate_candidate
from .strategies import candidate_digest, validate_proposal_batch
from .strategy_models import (
    AdaptiveStudyRequest,
    CandidateProposal,
    SearchObservation,
    SearchStrategy,
    StrategySpec,
    StrategyState,
    SuccessiveHalvingState,
)
from .strategy_registry import StrategyRegistry
from .study_models import EvaluatorRunner, canonical_digest


def run_search(
    *,
    record: AdaptiveStudyRecord,
    request: AdaptiveStudyRequest,
    baseline: CandidateProposal,
    strategies: StrategyRegistry,
    runner: EvaluatorRunner,
    store: ExperimentStore,
) -> tuple[tuple[CandidateAggregate, ...], dict[str, CandidateProposal], AdaptiveStudyRecord]:
    strategy = strategies.resolve(request.strategy_binding)
    state, proposal_map, phase = _restore_checkpoint(record, request, strategy, baseline)
    aggregates = _persisted_search_aggregates(record, baseline)
    if state is None:
        prior_limit = _remaining_capacity(record, request)
        for proposal in _prior_proposals(request, baseline)[:prior_limit]:
            proposal_map[proposal.candidate_digest] = proposal
            aggregate, record = evaluate_candidate(
                record=record,
                request=request,
                proposal=proposal,
                split="search",
                repeat_count=request.contract.repeats.search_repeats,
                runner=runner,
                store=store,
            )
            _replace_aggregate(aggregates, aggregate)
        capacity = _remaining_capacity(record, request)
        if capacity < 1:
            if proposal_map:
                return tuple(aggregates), proposal_map, record
            raise _error("optimization.budget_exhausted", "no measured search budget remains")
        if request.successive_halving is not None:
            capacity = _halving_initial_capacity(
                capacity,
                len(request.successive_halving.rungs),
                request.successive_halving.eta,
            )
            if capacity < 1:
                raise _error(
                    "optimization.budget_exhausted",
                    "remaining budget cannot cover a complete successive-halving schedule",
                )
        else:
            capacity = min(256, capacity + len(proposal_map) + 1)
        state = strategy.initialize(_strategy_spec(request, capacity), request.strategy_binding)
        phase = "ready"
    while phase != "complete":
        if phase == "awaiting":
            pending = _pending_proposals(state, proposal_map)
        else:
            remaining = _remaining_capacity(record, request)
            batch = strategy.propose(state, (), remaining)
            validate_proposal_batch(
                request.contract.parameters,
                batch.proposals,
                remaining_budget=remaining,
                prior_state=state,
                binding=request.strategy_binding,
            )
            if not batch.proposals:
                phase = "complete"
                break
            state = batch.state
            pending = batch.proposals
            proposal_map.update((item.candidate_digest, item) for item in pending)
            phase = "awaiting"
            record = _save_checkpoint(record, state, proposal_map, phase, store)
        observations: list[SearchObservation] = []
        for proposal in pending:
            aggregate, record = evaluate_candidate(
                record=record,
                request=request,
                proposal=proposal,
                split="search",
                repeat_count=request.contract.repeats.search_repeats,
                runner=runner,
                store=store,
            )
            if proposal.candidate_digest != baseline.candidate_digest:
                _replace_aggregate(aggregates, aggregate)
            observations.append(_observation(proposal, aggregate))
        state = strategy.observe(state, tuple(observations))
        phase = state.phase if isinstance(state, SuccessiveHalvingState) else "ready"
        record = _save_checkpoint(record, state, proposal_map, phase, store)
    if isinstance(state, SuccessiveHalvingState) and aggregates:
        rungs = tuple(item.resource_rung for item in aggregates if item.resource_rung is not None)
        if rungs:
            full_rung = max(rungs)
            aggregates = [item for item in aggregates if item.resource_rung == full_rung]
    return tuple(aggregates), proposal_map, record


def _restore_checkpoint(
    record: AdaptiveStudyRecord,
    request: AdaptiveStudyRequest,
    strategy: SearchStrategy,
    baseline: CandidateProposal,
) -> tuple[
    StrategyState | SuccessiveHalvingState | None,
    dict[str, CandidateProposal],
    str,
]:
    payload = record.strategy_state
    raw_state = payload.get("strategy")
    raw_proposals = payload.get("candidate_parameters", {})
    raw_spec = payload.get("strategy_spec")
    raw_spec_digest = payload.get("strategy_spec_digest")
    raw_phase = payload.get("phase", "ready")
    if raw_state is None:
        if payload:
            raise _error("optimization.strategy_nondeterministic", "strategy checkpoint is invalid")
        return None, {}, "ready"
    if not isinstance(raw_state, dict) or not isinstance(raw_proposals, dict):
        raise _error("optimization.strategy_nondeterministic", "strategy checkpoint is invalid")
    try:
        state = (
            SuccessiveHalvingState.model_validate(raw_state)
            if request.strategy_binding.kind == "successive_halving"
            else StrategyState.model_validate(raw_state)
        )
        persisted_spec = StrategySpec.model_validate(raw_spec)
        proposals = {
            str(digest): CandidateProposal.model_validate(proposal)
            for digest, proposal in raw_proposals.items()
        }
    except (TypeError, ValueError) as error:
        raise _error(
            "optimization.strategy_nondeterministic", "strategy checkpoint is invalid"
        ) from error
    if state.binding != request.strategy_binding or state.spec != persisted_spec:
        raise _error(
            "optimization.strategy_nondeterministic", "strategy checkpoint changed identity"
        )
    if raw_spec_digest != canonical_digest(persisted_spec.model_dump(mode="json")):
        raise _error(
            "optimization.strategy_nondeterministic",
            "strategy checkpoint changed its frozen specification",
        )
    _validate_restored_spec(state.spec, request)
    try:
        proposals_valid = all(
            digest == proposal.candidate_digest
            and candidate_digest(proposal.parameters) == proposal.candidate_digest
            for digest, proposal in proposals.items()
        )
    except (TypeError, ValueError) as error:
        raise _error(
            "optimization.strategy_nondeterministic", "proposal checkpoint is invalid"
        ) from error
    if not proposals_valid:
        raise _error("optimization.strategy_nondeterministic", "proposal checkpoint is invalid")
    if not isinstance(raw_phase, str) or raw_phase not in {"ready", "awaiting", "complete"}:
        raise _error("optimization.strategy_nondeterministic", "strategy phase is invalid")
    if isinstance(state, SuccessiveHalvingState) and raw_phase != state.phase:
        raise _error("optimization.strategy_nondeterministic", "strategy phase changed identity")
    if isinstance(state, StrategyState) and raw_phase == "complete":
        raise _error("optimization.strategy_nondeterministic", "strategy phase changed identity")
    validate_checkpoint_replay(
        strategy=strategy,
        state=state,
        proposals=proposals,
        priors=_prior_proposals(request, baseline),
        record=record,
        request=request,
    )
    return state, proposals, raw_phase


def _pending_proposals(
    state: StrategyState | SuccessiveHalvingState,
    proposals: dict[str, CandidateProposal],
) -> tuple[CandidateProposal, ...]:
    if isinstance(state, SuccessiveHalvingState):
        if state.phase != "awaiting":
            raise _error("optimization.strategy_nondeterministic", "halving checkpoint is invalid")
        return state.current_assignments
    pending = tuple(proposals.values())
    if not pending:
        raise _error("optimization.strategy_nondeterministic", "core checkpoint lost proposals")
    return pending


def _save_checkpoint(
    record: AdaptiveStudyRecord,
    state: StrategyState | SuccessiveHalvingState,
    proposals: dict[str, CandidateProposal],
    phase: str,
    store: ExperimentStore,
) -> AdaptiveStudyRecord:
    checkpoint = {
        "strategy": state.model_dump(mode="json"),
        "strategy_spec": state.spec.model_dump(mode="json"),
        "strategy_spec_digest": canonical_digest(state.spec.model_dump(mode="json")),
        "candidate_parameters": {
            digest: proposal.model_dump(mode="json") for digest, proposal in proposals.items()
        },
        "phase": phase,
    }
    return store.update_study(record.model_copy(update={"strategy_state": checkpoint}))


def _strategy_spec(request: AdaptiveStudyRequest, capacity: int) -> StrategySpec:
    directions: dict[str, Literal["maximize", "minimize"]] = {
        item.name: item.direction for item in request.contract.objectives
    }
    order: tuple[tuple[StrictStr, Literal["maximize", "minimize"]], ...] = (
        tuple(
            (
                next(
                    item.metric_pointer for item in request.contract.objectives if item.name == name
                ),
                directions[name],
            )
            for name in request.objective_order
        )
        if request.successive_halving is not None
        else ()
    )
    return StrategySpec(
        parameters=request.contract.parameters,
        seed=request.contract.repeats.seed,
        total_design_size=capacity,
        successive_halving=request.successive_halving,
        objective_order=order,
    )


def _remaining_capacity(record: AdaptiveStudyRecord, request: AdaptiveStudyRequest) -> int:
    reserve = request.contract.repeats.validation_repeats * (1 + request.finalist_limit)
    runs = request.contract.budget.max_trials - record.budget.attempted_runs - reserve
    return max(0, runs // request.contract.repeats.search_repeats)


def _halving_initial_capacity(available: int, rung_count: int, eta: int) -> int:
    for initial in range(available, 0, -1):
        cohort = initial
        total = 0
        for _ in range(rung_count):
            total += cohort
            cohort = max(1, (cohort + eta - 1) // eta)
        if total <= available:
            return initial
    return 0


def _persisted_search_aggregates(
    record: AdaptiveStudyRecord, baseline: CandidateProposal
) -> list[CandidateAggregate]:
    return [
        item
        for item in record.aggregates
        if item.split == "search" and item.candidate_digest != baseline.candidate_digest
    ]


def _prior_proposals(
    request: AdaptiveStudyRequest, baseline: CandidateProposal
) -> tuple[CandidateProposal, ...]:
    proposals: list[CandidateProposal] = []
    seen = {baseline.candidate_digest}
    for prior in request.contract.priors:
        parameters = prior.parameters
        if request.successive_halving is not None:
            parameters = {
                **parameters,
                request.successive_halving.resource_parameter: request.successive_halving.rungs[0],
            }
        digest = candidate_digest(parameters)
        if digest in seen:
            continue
        seen.add(digest)
        base_digest: str | None = None
        resource_rung: int | None = None
        if request.successive_halving is not None:
            resource_name = request.successive_halving.resource_parameter
            rung = parameters[resource_name]
            if type(rung) is int:
                resource_rung = rung
                base_digest = candidate_digest(
                    {name: value for name, value in parameters.items() if name != resource_name}
                )
        proposals.append(
            CandidateProposal(
                parameters=parameters,
                candidate_digest=digest,
                base_candidate_digest=base_digest,
                resource_rung=resource_rung,
            )
        )
    return tuple(proposals)


def _validate_restored_spec(spec: StrategySpec, request: AdaptiveStudyRequest) -> None:
    expected = _strategy_spec(request, spec.total_design_size)
    if spec != expected:
        raise _error(
            "optimization.strategy_nondeterministic",
            "strategy checkpoint does not match the immutable request",
        )


def _replace_aggregate(aggregates: list[CandidateAggregate], aggregate: CandidateAggregate) -> None:
    aggregates[:] = [
        item
        for item in aggregates
        if (item.candidate_digest, item.resource_rung)
        != (aggregate.candidate_digest, aggregate.resource_rung)
    ]
    aggregates.append(aggregate)


def _observation(proposal: CandidateProposal, aggregate: CandidateAggregate) -> SearchObservation:
    return SearchObservation(
        candidate_digest=proposal.candidate_digest,
        base_candidate_digest=proposal.base_candidate_digest,
        resource_rung=proposal.resource_rung,
        metrics={name: metric.value for name, metric in aggregate.metrics.items()},
        feasible=aggregate.feasible,
        failed=not aggregate.feasible,
    )


def _error(code: str, message: str) -> OrganelleContractError:
    return OrganelleContractError(code=code, message=message)


__all__ = ["run_search"]
