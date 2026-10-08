"""Pure deterministic candidate generation for governed optimization studies."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence

from pydantic import JsonValue

from .models import OptimizationContract, ParameterDomain


def grid_candidates(
    contract: OptimizationContract,
) -> tuple[dict[str, JsonValue], ...]:
    """Enumerate a finite grid in declaration order up to the hard trial budget."""

    if "grid" not in contract.strategies:
        raise ValueError("the optimization contract does not permit grid search")
    size = _finite_space_size(contract.parameters)
    if size is None:
        raise ValueError("grid search requires finite parameter domains")
    return tuple(
        _finite_candidate_at(contract.parameters, index)
        for index in range(min(size, contract.budget.max_trials))
    )


def random_candidates(
    contract: OptimizationContract,
) -> tuple[dict[str, JsonValue], ...]:
    """Sample bounded unique candidates using only the contract's seeded RNG."""

    if "random" not in contract.strategies:
        raise ValueError("the optimization contract does not permit random search")
    rng = random.Random(contract.seed)
    target = contract.budget.max_trials
    finite_size = _finite_space_size(contract.parameters)
    if finite_size is not None and finite_size <= target:
        candidates = list(_all_finite_candidates(contract.parameters))
        rng.shuffle(candidates)
        return tuple(candidates)
    candidates: list[dict[str, JsonValue]] = []
    seen: set[str] = set()
    for _ in range(contract.budget.max_trials * 20):
        candidate = {domain.name: _sample_domain(domain, rng) for domain in contract.parameters}
        identity = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if identity not in seen:
            seen.add(identity)
            candidates.append(candidate)
            if len(candidates) == target:
                break
    return tuple(candidates)


def optimization_request_schema(
    contract: OptimizationContract,
    *,
    inputs_schema: dict[str, JsonValue],
    fixed_parameters_schema: dict[str, JsonValue],
    candidate_schema: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Build the bounded conditional Agent schema for one admitted contract."""

    properties: dict[str, JsonValue] = {
        "inputs": inputs_schema,
        "fixed_parameters": fixed_parameters_schema,
        "strategy": {
            "type": "string",
            "enum": list(contract.strategies),
            "default": "explicit",
        },
        "seed": {"type": "integer", "default": contract.seed},
        "rationale": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2000,
            "pattern": r"\S",
        },
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": contract.budget.max_trials,
            "items": candidate_schema,
        },
    }
    branches: list[JsonValue] = []
    for strategy in contract.strategies:
        branch: dict[str, JsonValue] = {
            "properties": {"strategy": {"const": strategy}},
        }
        if strategy == "explicit":
            branch["required"] = ["candidates"]
            branch["not"] = {"required": ["rationale"]}
        elif strategy == "agent":
            branch["required"] = ["strategy", "candidates", "rationale"]
        else:
            branch["required"] = ["strategy"]
            branch["not"] = {
                "anyOf": [
                    {"required": ["candidates"]},
                    {"required": ["rationale"]},
                ]
            }
        branches.append(branch)
    return {
        "type": "object",
        "properties": properties,
        "required": ["inputs"],
        "oneOf": branches,
        "additionalProperties": False,
    }


def _finite_values(domain: ParameterDomain) -> Sequence[JsonValue]:
    if domain.kind == "integer":
        assert type(domain.minimum) is int
        assert type(domain.maximum) is int
        assert type(domain.step) is int
        return range(domain.minimum, domain.maximum + 1, domain.step)
    if domain.kind in {"number", "categorical"} and domain.values:
        return domain.values
    if domain.kind == "boolean":
        return (False, True)
    raise ValueError(f"parameter {domain.name!r} does not have a finite grid domain")


def _sample_domain(domain: ParameterDomain, rng: random.Random) -> JsonValue:
    if domain.kind == "integer":
        assert type(domain.minimum) is int
        assert type(domain.step) is int
        cardinality = _finite_cardinality(domain)
        assert cardinality is not None
        return domain.minimum + rng.randrange(cardinality) * domain.step
    if domain.kind in {"categorical", "boolean"} or domain.values:
        return rng.choice(_finite_values(domain))
    assert domain.kind == "number"
    assert domain.minimum is not None
    assert domain.maximum is not None
    if domain.logarithmic:
        return math.exp(rng.uniform(math.log(domain.minimum), math.log(domain.maximum)))
    return rng.uniform(domain.minimum, domain.maximum)


def _finite_space_size(parameters: tuple[ParameterDomain, ...]) -> int | None:
    size = 1
    for domain in parameters:
        cardinality = _finite_cardinality(domain)
        if cardinality is None:
            return None
        size *= cardinality
    return size


def _finite_cardinality(domain: ParameterDomain) -> int | None:
    if domain.kind == "integer":
        assert type(domain.minimum) is int
        assert type(domain.maximum) is int
        assert type(domain.step) is int
        return ((domain.maximum - domain.minimum) // domain.step) + 1
    if domain.kind in {"number", "categorical"} and domain.values:
        return len(domain.values)
    if domain.kind == "boolean":
        return 2
    return None


def _all_finite_candidates(
    parameters: tuple[ParameterDomain, ...],
) -> tuple[dict[str, JsonValue], ...]:
    size = _finite_space_size(parameters)
    assert size is not None
    return tuple(_finite_candidate_at(parameters, index) for index in range(size))


def _finite_candidate_at(
    parameters: tuple[ParameterDomain, ...], index: int
) -> dict[str, JsonValue]:
    positions: list[int] = []
    remaining = index
    for domain in reversed(parameters):
        cardinality = _finite_cardinality(domain)
        assert cardinality is not None
        remaining, position = divmod(remaining, cardinality)
        positions.append(position)
    return {
        domain.name: _finite_value_at(domain, position)
        for domain, position in zip(parameters, reversed(positions), strict=True)
    }


def _finite_value_at(domain: ParameterDomain, index: int) -> JsonValue:
    if domain.kind == "integer":
        assert type(domain.minimum) is int
        assert type(domain.step) is int
        return domain.minimum + index * domain.step
    if domain.kind in {"number", "categorical"} and domain.values:
        return domain.values[index]
    if domain.kind == "boolean":
        return bool(index)
    raise ValueError(f"parameter {domain.name!r} does not have a finite domain")


__all__ = [
    "grid_candidates",
    "optimization_request_schema",
    "random_candidates",
]
