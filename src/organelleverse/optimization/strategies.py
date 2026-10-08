"""Deterministic bounded core search strategies."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable

from pydantic import JsonValue

from organelleverse.core.errors import OrganelleContractError

from .contracts_v3 import _validate_candidate  # pyright: ignore[reportPrivateUsage]
from .models import ParameterDomain, _canonical_json  # pyright: ignore[reportPrivateUsage]
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


def candidate_digest(parameters: dict[str, JsonValue]) -> str:
    payload = json.dumps(
        _canonical_json(parameters), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class CoreStrategy:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def initialize(self, spec: StrategySpec, binding: StrategyBinding) -> StrategyState:
        if binding.kind != self.kind:
            raise _proposal_error("strategy binding kind does not match its implementation")
        return StrategyState(kind=self.kind, binding=binding, spec=spec)  # type: ignore[arg-type]

    def propose(
        self,
        state: StrategyState | SuccessiveHalvingState,
        observations: tuple[SearchObservation, ...],
        remaining_budget: int,
    ) -> ProposalBatch:
        del observations
        if not isinstance(state, StrategyState) or state.binding.kind != self.kind:
            raise _proposal_error("strategy state does not match its implementation")
        candidates = _ordered_candidates(state.spec, self.kind)
        unseen = [item for item in candidates if candidate_digest(item) not in state.emitted]
        selected = unseen[: max(0, remaining_budget)]
        proposals = tuple(
            CandidateProposal(parameters=item, candidate_digest=candidate_digest(item))
            for item in selected
        )
        updated = state.model_copy(
            update={
                "step": state.step + 1,
                "emitted": (*state.emitted, *(item.candidate_digest for item in proposals)),
            }
        )
        return ProposalBatch(
            proposals=proposals,
            state=updated,
            exhausted=len(unseen) <= len(selected),
        )

    def observe(
        self,
        state: StrategyState | SuccessiveHalvingState,
        observations: tuple[SearchObservation, ...],
    ) -> StrategyState:
        del observations
        if not isinstance(state, StrategyState):
            raise _proposal_error("strategy state does not match its implementation")
        return state


def strategy_for(
    kind: str,
    *,
    base_resolver: Callable[[StrategyBinding], SearchStrategy] | None = None,
) -> SearchStrategy:
    if kind not in {"grid", "random", "space_filling", "successive_halving"}:
        raise ValueError(f"unsupported strategy kind: {kind}")
    if kind == "successive_halving":
        from .successive_halving import SuccessiveHalvingStrategy

        if base_resolver is None:
            raise ValueError("successive-halving requires an admitted base strategy resolver")
        return SuccessiveHalvingStrategy(base_resolver)
    return CoreStrategy(kind)


def validate_proposal_batch(
    parameters: tuple[ParameterDomain, ...],
    proposals: tuple[CandidateProposal, ...],
    *,
    remaining_budget: int,
    prior_state: StrategyState | SuccessiveHalvingState,
    binding: StrategyBinding,
) -> tuple[CandidateProposal, ...]:
    if (
        prior_state.binding != binding
        or prior_state.spec.parameters != parameters
        or prior_state.kind != binding.kind
    ):
        raise _proposal_error("proposal state/spec/binding does not match the request")
    if len(proposals) > remaining_budget:
        raise _proposal_error("proposal batch exceeds the remaining trial budget")
    seen: set[str] = set()
    previously_emitted = (
        set(prior_state.emitted)
        if isinstance(prior_state, StrategyState)
        else set(prior_state.completed_assignment_digests)
    )
    expected_halving: tuple[CandidateProposal, ...] | None = None
    if isinstance(prior_state, SuccessiveHalvingState):
        if prior_state.phase != "ready" or prior_state.rung_index >= len(prior_state.config.rungs):
            raise _proposal_error("successive-halving proposals require a ready prior state")
        rung = prior_state.config.rungs[prior_state.rung_index]
        expected_halving = tuple(
            CandidateProposal(
                parameters={
                    **item.parameters,
                    prior_state.config.resource_parameter: rung,
                },
                candidate_digest=candidate_digest(
                    {
                        **item.parameters,
                        prior_state.config.resource_parameter: rung,
                    }
                ),
                base_candidate_digest=item.candidate_digest,
                resource_rung=rung,
            )
            for item in prior_state.cohort
        )
        if proposals != expected_halving:
            raise _proposal_error(
                "successive-halving proposals must equal the complete current cohort"
            )
    for proposal in proposals:
        try:
            _validate_candidate(parameters, proposal.parameters, "strategy proposal")
        except ValueError as error:
            raise _proposal_error(str(error)) from error
        digest = candidate_digest(proposal.parameters)
        if isinstance(prior_state, StrategyState):
            valid_shape = proposal.base_candidate_digest is None and proposal.resource_rung is None
        else:
            rung = prior_state.config.rungs[prior_state.rung_index]
            base_parameters = {
                name: value
                for name, value in proposal.parameters.items()
                if name != prior_state.config.resource_parameter
            }
            valid_shape = (
                proposal.resource_rung == rung
                and proposal.parameters.get(prior_state.config.resource_parameter) == rung
                and proposal.base_candidate_digest == candidate_digest(base_parameters)
            )
        if (
            not valid_shape
            or proposal.candidate_digest != digest
            or digest in seen
            or digest in previously_emitted
        ):
            raise _proposal_error("proposal digest is non-canonical or duplicated")
        seen.add(digest)
    return proposals


def promote_successive_halving(
    observations: tuple[SearchObservation, ...],
    *,
    config: SuccessiveHalvingConfig,
    objective_order: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    from .successive_halving import promote_successive_halving as promote

    return promote(observations, config=config, objective_order=objective_order)


def _proposal_error(message: str) -> OrganelleContractError:
    return OrganelleContractError(
        code="optimization.strategy_proposal_invalid",
        message=message,
    )


def _ordered_candidates(spec: StrategySpec, kind: str) -> list[dict[str, JsonValue]]:
    if kind == "grid":
        size = math.prod(_finite_cardinality(domain) for domain in spec.parameters)
        return [
            _finite_candidate_at(spec.parameters, index)
            for index in range(min(spec.total_design_size, size))
        ]
    if kind == "space_filling":
        return _space_filling_design(spec)
    return _random_design(spec)


def _finite_candidate_at(
    parameters: tuple[ParameterDomain, ...], index: int
) -> dict[str, JsonValue]:
    positions: list[int] = []
    for domain in reversed(parameters):
        index, position = divmod(index, _finite_cardinality(domain))
        positions.append(position)
    return {
        domain.name: _finite_value_at(domain, position)
        for domain, position in zip(parameters, reversed(positions), strict=True)
    }


def _random_design(spec: StrategySpec) -> list[dict[str, JsonValue]]:
    rng = random.Random(spec.seed)
    finite_size = _finite_design_size(spec.parameters)
    if finite_size is not None:
        count = min(spec.total_design_size, finite_size)
        return [
            _finite_candidate_at(spec.parameters, index)
            for index in rng.sample(range(finite_size), count)
        ]
    candidates: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    for _ in range(spec.total_design_size * 20):
        candidate = {domain.name: _sample(domain, rng) for domain in spec.parameters}
        digest = candidate_digest(candidate)
        if digest not in seen:
            seen.add(digest)
            candidates.append(candidate)
            if len(candidates) == spec.total_design_size:
                break
    if len(candidates) != spec.total_design_size:
        raise _proposal_error("continuous random design exhausted its bounded retry budget")
    return candidates


def _space_filling_design(spec: StrategySpec) -> list[dict[str, JsonValue]]:
    size = spec.total_design_size
    rng = random.Random(spec.seed)
    columns: list[list[JsonValue]] = []
    for domain in spec.parameters:
        order = list(range(size))
        rng.shuffle(order)
        if domain.kind == "number" and not domain.values:
            assert domain.minimum is not None and domain.maximum is not None
            transformed_min = math.log(domain.minimum) if domain.logarithmic else domain.minimum
            transformed_max = math.log(domain.maximum) if domain.logarithmic else domain.maximum
            values: list[JsonValue] = []
            for stratum in order:
                fraction = (stratum + rng.random()) / size
                transformed = transformed_min + fraction * (transformed_max - transformed_min)
                values.append(math.exp(transformed) if domain.logarithmic else transformed)
            columns.append(values)
        else:
            cardinality = _finite_cardinality(domain)
            columns.append([_finite_value_at(domain, position % cardinality) for position in order])
    candidates: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    for index in range(size):
        candidate = {
            domain.name: columns[column][index] for column, domain in enumerate(spec.parameters)
        }
        digest = candidate_digest(candidate)
        if digest not in seen:
            seen.add(digest)
            candidates.append(candidate)
    finite_size = _finite_design_size(spec.parameters)
    target = min(size, finite_size) if finite_size is not None else size
    if len(candidates) < target and finite_size is not None:
        for index in range(finite_size):
            candidate = _finite_candidate_at(spec.parameters, index)
            digest = candidate_digest(candidate)
            if digest in seen:
                continue
            seen.add(digest)
            candidates.append(candidate)
            if len(candidates) == target:
                break
    if len(candidates) != target:
        raise _proposal_error("space-filling design could not produce its frozen candidate set")
    return candidates


def _finite_design_size(parameters: tuple[ParameterDomain, ...]) -> int | None:
    try:
        return math.prod(_finite_cardinality(domain) for domain in parameters)
    except ValueError:
        return None


def _sample(domain: ParameterDomain, rng: random.Random) -> JsonValue:
    if domain.kind == "number" and not domain.values:
        assert domain.minimum is not None and domain.maximum is not None
        if domain.logarithmic:
            return math.exp(rng.uniform(math.log(domain.minimum), math.log(domain.maximum)))
        return rng.uniform(domain.minimum, domain.maximum)
    return _finite_value_at(domain, rng.randrange(_finite_cardinality(domain)))


def _finite_cardinality(domain: ParameterDomain) -> int:
    if domain.kind == "integer":
        assert type(domain.minimum) is int
        assert type(domain.maximum) is int
        assert type(domain.step) is int
        return ((domain.maximum - domain.minimum) // domain.step) + 1
    if domain.kind in {"categorical", "number"} and domain.values:
        return len(domain.values)
    if domain.kind == "boolean":
        return 2
    raise ValueError(f"strategy requires a finite domain for {domain.name!r}")


def _finite_value_at(domain: ParameterDomain, index: int) -> JsonValue:
    if domain.kind == "integer":
        assert type(domain.minimum) is int
        assert type(domain.step) is int
        return domain.minimum + index * domain.step
    if domain.kind in {"categorical", "number"} and domain.values:
        return domain.values[index]
    if domain.kind == "boolean":
        return bool(index)
    raise ValueError(f"strategy requires a finite domain for {domain.name!r}")


__all__ = [
    "CoreStrategy",
    "candidate_digest",
    "promote_successive_halving",
    "strategy_for",
    "validate_proposal_batch",
]
