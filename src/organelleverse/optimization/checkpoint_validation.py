"""Deterministic replay validation for durable strategy checkpoints."""

from __future__ import annotations

from organelleverse.core.errors import OrganelleContractError
from organelleverse.plugin_experiments.adaptive_models import (
    AdaptiveStudyRecord,
    CandidateAggregate,
)

from .strategies import candidate_digest, validate_proposal_batch
from .strategy_models import (
    AdaptiveStudyRequest,
    CandidateProposal,
    SearchObservation,
    SearchStrategy,
    StrategyState,
    SuccessiveHalvingState,
)


def validate_checkpoint_replay(
    *,
    strategy: SearchStrategy,
    state: StrategyState | SuccessiveHalvingState,
    proposals: dict[str, CandidateProposal],
    priors: tuple[CandidateProposal, ...],
    record: AdaptiveStudyRecord,
    request: AdaptiveStudyRequest,
) -> None:
    """Prove that restored state and proposals are deterministic strategy output."""

    try:
        initial = strategy.initialize(state.spec, request.strategy_binding)
        if isinstance(state, StrategyState):
            _validate_core_replay(
                strategy=strategy,
                initial=initial,
                state=state,
                proposals=proposals,
                priors=priors,
                record=record,
                request=request,
            )
        else:
            _validate_halving_replay(
                strategy=strategy,
                initial=initial,
                target=state,
                proposals=proposals,
                priors=priors,
                record=record,
                request=request,
            )
    except (OrganelleContractError, TypeError, ValueError, KeyError) as error:
        raise _invalid("strategy checkpoint cannot be replayed") from error


def _validate_core_replay(
    *,
    strategy: SearchStrategy,
    initial: StrategyState | SuccessiveHalvingState,
    state: StrategyState,
    proposals: dict[str, CandidateProposal],
    priors: tuple[CandidateProposal, ...],
    record: AdaptiveStudyRecord,
    request: AdaptiveStudyRequest,
) -> None:
    if not isinstance(initial, StrategyState) or state.step < 1:
        raise _invalid("core checkpoint state is not reachable")
    replayed = initial
    emitted: list[CandidateProposal] = []
    preexisting = {
        candidate_digest(request.contract.baseline.parameters),
        *(item.candidate_digest for item in priors),
    }
    reserve = request.contract.repeats.validation_repeats * (1 + request.finalist_limit)
    attempted = _search_attempt_count(record, preexisting)
    remaining = _remaining_search_candidates(request, attempted, reserve)
    for _ in range(state.spec.total_design_size):
        batch = strategy.propose(replayed, (), remaining)
        if not batch.proposals:
            raise _invalid("core checkpoint advanced through an empty proposal batch")
        validate_proposal_batch(
            request.contract.parameters,
            batch.proposals,
            remaining_budget=remaining,
            prior_state=replayed,
            binding=request.strategy_binding,
        )
        start = len(emitted)
        end = start + len(batch.proposals)
        if tuple(item.candidate_digest for item in batch.proposals) != state.emitted[start:end]:
            raise _invalid("core checkpoint is not a deterministic design prefix")
        emitted.extend(batch.proposals)
        replayed = batch.state
        if replayed == state:
            break
        if len(emitted) >= len(state.emitted):
            raise _invalid("core checkpoint state does not match its emitted design")
        attempted += _search_attempt_count(
            record,
            {item.candidate_digest for item in batch.proposals} - preexisting,
        )
        remaining = _remaining_search_candidates(request, attempted, reserve)
    if replayed != state or len(emitted) != len(state.emitted):
        raise _invalid("core checkpoint state is not exactly replayable")
    retained_priors = tuple(item for item in priors if item.candidate_digest in proposals)
    expected = _proposal_map(retained_priors, tuple(emitted))
    if proposals != expected:
        raise _invalid("core checkpoint proposal map changed its strategy output")


def _search_attempt_count(record: AdaptiveStudyRecord, digests: set[str]) -> int:
    return sum(
        1
        for attempt in record.attempts
        if attempt.identity.split == "search" and attempt.identity.candidate_digest in digests
    )


def _remaining_search_candidates(
    request: AdaptiveStudyRequest, attempted: int, reserve: int
) -> int:
    runs = request.contract.budget.max_trials - attempted - reserve
    return max(0, runs // request.contract.repeats.search_repeats)


def _validate_halving_replay(
    *,
    strategy: SearchStrategy,
    initial: StrategyState | SuccessiveHalvingState,
    target: SuccessiveHalvingState,
    proposals: dict[str, CandidateProposal],
    priors: tuple[CandidateProposal, ...],
    record: AdaptiveStudyRecord,
    request: AdaptiveStudyRequest,
) -> None:
    if not isinstance(initial, SuccessiveHalvingState):
        raise _invalid("halving checkpoint has the wrong initial state")
    replayed = initial
    emitted: list[CandidateProposal] = []
    for _ in range(len(target.config.rungs)):
        if replayed == target:
            _require_exact_map(proposals, priors, tuple(emitted))
            return
        batch = strategy.propose(replayed, (), target.spec.total_design_size)
        validate_proposal_batch(
            request.contract.parameters,
            batch.proposals,
            remaining_budget=len(batch.proposals),
            prior_state=replayed,
            binding=request.strategy_binding,
        )
        emitted.extend(batch.proposals)
        if batch.state == target:
            _require_exact_map(proposals, priors, tuple(emitted))
            return
        observations = tuple(
            _observation(item, _aggregate_for(record, item)) for item in batch.proposals
        )
        replayed = strategy.observe(batch.state, observations)
        if replayed == target:
            _require_exact_map(proposals, priors, tuple(emitted))
            return
    raise _invalid("halving checkpoint is not reachable from measured observations")


def _aggregate_for(record: AdaptiveStudyRecord, proposal: CandidateProposal) -> CandidateAggregate:
    matches = tuple(
        item
        for item in record.aggregates
        if item.split == "search"
        and item.candidate_digest == proposal.candidate_digest
        and item.resource_rung == proposal.resource_rung
    )
    if len(matches) != 1:
        raise _invalid("halving replay requires one exact measured aggregate")
    return matches[0]


def _observation(proposal: CandidateProposal, aggregate: CandidateAggregate) -> SearchObservation:
    return SearchObservation(
        candidate_digest=proposal.candidate_digest,
        base_candidate_digest=proposal.base_candidate_digest,
        resource_rung=proposal.resource_rung,
        metrics={name: metric.value for name, metric in aggregate.metrics.items()},
        feasible=aggregate.feasible,
        failed=not aggregate.feasible,
    )


def _proposal_map(
    priors: tuple[CandidateProposal, ...], emitted: tuple[CandidateProposal, ...]
) -> dict[str, CandidateProposal]:
    return {item.candidate_digest: item for item in (*priors, *emitted)}


def _require_exact_map(
    proposals: dict[str, CandidateProposal],
    priors: tuple[CandidateProposal, ...],
    emitted: tuple[CandidateProposal, ...],
) -> None:
    if proposals != _proposal_map(priors, emitted):
        raise _invalid("halving checkpoint proposal map changed its strategy output")


def _invalid(message: str) -> OrganelleContractError:
    return OrganelleContractError(
        code="optimization.strategy_nondeterministic",
        message=message,
    )


__all__ = ["validate_checkpoint_replay"]
