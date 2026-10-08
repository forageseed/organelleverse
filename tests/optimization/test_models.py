"""Governed cross-module optimization contract boundaries."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from organelleverse.optimization import (
    ObjectiveSpec,
    OptimizationBudget,
    OptimizationContract,
    ParameterDomain,
)

_BUNDLE_HASH = "sha256:" + "a" * 64
_EXECUTION_IDENTITY = "sha256:" + "b" * 64


def _contract(**updates: object) -> OptimizationContract:
    payload: dict[str, object] = {
        "target_capability_id": "demo.optimize",
        "target_bundle_version": "1.2.3",
        "target_contract_version": "1.0",
        "bundle_content_hash": _BUNDLE_HASH,
        "parameters": (
            ParameterDomain(
                name="count",
                kind="integer",
                minimum=1,
                maximum=5,
                step=2,
            ),
        ),
        "objective": ObjectiveSpec(direction="maximize"),
        "strategies": ("explicit", "grid", "random", "agent"),
        "seed": 0,
        "budget": OptimizationBudget(max_trials=5, parallelism=2),
    }
    payload.update(updates)
    return OptimizationContract.model_validate(payload)


def test_valid_domains_round_trip_deterministically() -> None:
    contract = OptimizationContract(
        target_capability_id="pangenome.build_graph",
        target_bundle_version="1.2.3",
        target_contract_version="1.0",
        bundle_content_hash=_BUNDLE_HASH,
        execution_identity=_EXECUTION_IDENTITY,
        parameters=(
            ParameterDomain(
                name="segment_length",
                kind="integer",
                minimum=100,
                maximum=10_000,
                step=100,
            ),
            ParameterDomain(
                name="identity",
                kind="number",
                minimum=80.0,
                maximum=100.0,
                logarithmic=False,
            ),
            ParameterDomain(
                name="backend",
                kind="categorical",
                values=("minigraph", "pggb"),
            ),
            ParameterDomain(name="use_masking", kind="boolean"),
        ),
        objective=ObjectiveSpec(
            name="plugin_optimization_score",
            direction="maximize",
        ),
        strategies=("explicit", "random", "agent"),
        seed=7,
        budget=OptimizationBudget(max_trials=8, parallelism=2),
    )

    payload = contract.model_dump(mode="json")
    restored = OptimizationContract.model_validate(payload)

    assert restored == contract
    assert restored.schema_version == "organelleverse.optimization.contract.v2"
    assert restored.parameters[0].minimum == 100
    assert restored.parameters[2].values == ("minigraph", "pggb")
    assert restored.parameters[3].values == ()
    assert restored.digest == contract.digest
    assert restored.digest.startswith("sha256:")
    assert "digest" not in payload


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "x", "kind": "number"},
        {"name": "x", "kind": "number", "minimum": 1.0, "maximum": 1.0},
        {"name": "x", "kind": "number", "minimum": 2.0, "maximum": 1.0},
        {"name": "x", "kind": "number", "minimum": math.nan, "maximum": 1.0},
        {"name": "x", "kind": "number", "minimum": 0.0, "maximum": math.inf},
        {"name": "x", "kind": "number", "minimum": 0.0, "maximum": 1.0, "step": 0},
        {
            "name": "x",
            "kind": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": math.inf,
        },
        {
            "name": "x",
            "kind": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "step": math.nan,
        },
        {
            "name": "x",
            "kind": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "values": (0.5, 0.5),
        },
        {
            "name": "x",
            "kind": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "values": (-0.1, 0.5),
        },
        {
            "name": "x",
            "kind": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "values": (0.5, 1.1),
        },
        {
            "name": "x",
            "kind": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "logarithmic": True,
        },
        {"name": "x", "kind": "integer", "minimum": 1, "maximum": 5},
        {"name": "x", "kind": "integer", "minimum": 1.0, "maximum": 5, "step": 1},
        {"name": "x", "kind": "integer", "minimum": 1, "maximum": 5, "step": 1.5},
        {"name": "x", "kind": "categorical", "values": ("only",)},
        {"name": "x", "kind": "categorical", "values": ("same", "same")},
        {"name": "x", "kind": "categorical", "values": (1.0, math.nan)},
        {"name": "x", "kind": "boolean", "values": (False, True)},
        {
            "name": "x",
            "kind": "boolean",
            "minimum": 0,
            "maximum": 1,
        },
    ],
)
def test_invalid_or_unbounded_domains_fail_closed(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ParameterDomain.model_validate(payload)


@pytest.mark.parametrize(
    ("max_trials", "parallelism"),
    [(0, 1), (257, 1), (3, 0), (33, 33), (2, 3)],
)
def test_invalid_budgets_fail_closed(max_trials: int, parallelism: int) -> None:
    with pytest.raises(ValidationError):
        OptimizationBudget(max_trials=max_trials, parallelism=parallelism)


def test_duplicate_contract_definitions_fail_closed() -> None:
    domain = ParameterDomain(
        name="count",
        kind="integer",
        minimum=1,
        maximum=5,
        step=1,
    )

    with pytest.raises(ValidationError):
        _contract(parameters=(domain, domain))
    with pytest.raises(ValidationError):
        _contract(strategies=("explicit", "explicit"))


def test_grid_rejects_a_continuous_number_without_explicit_values() -> None:
    continuous = ParameterDomain(
        name="threshold",
        kind="number",
        minimum=0.0,
        maximum=1.0,
    )

    with pytest.raises(ValidationError):
        _contract(parameters=(continuous,), strategies=("grid",))


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "x", "kind": "number", "minimum": "0", "maximum": "1"},
        {"name": "x", "kind": "integer", "minimum": 1.0, "maximum": 5, "step": 1},
        {"name": "x", "kind": "integer", "minimum": True, "maximum": 5, "step": 1},
        {
            "name": "x",
            "kind": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "logarithmic": "false",
        },
    ],
)
def test_domain_control_fields_are_strictly_typed(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ParameterDomain.model_validate(payload)


@pytest.mark.parametrize(
    ("max_trials", "parallelism"),
    [("5", 2), (5.0, 2), (True, 1), (5, "2"), (5, 2.0), (5, False)],
)
def test_budget_fields_reject_coercible_values(
    max_trials: object,
    parallelism: object,
) -> None:
    with pytest.raises(ValidationError):
        OptimizationBudget.model_validate({"max_trials": max_trials, "parallelism": parallelism})


@pytest.mark.parametrize("seed", ["7", 7.0, True])
def test_contract_seed_rejects_coercible_values(seed: object) -> None:
    with pytest.raises(ValidationError):
        _contract(seed=seed)


def test_semantically_equal_number_lexemes_have_one_digest() -> None:
    integer_lexemes = ParameterDomain(
        name="threshold",
        kind="number",
        minimum=0,
        maximum=1,
    )
    float_lexemes = ParameterDomain(
        name="threshold",
        kind="number",
        minimum=0.0,
        maximum=1.0,
    )

    left = _contract(parameters=(integer_lexemes,), strategies=("random",))
    right = _contract(parameters=(float_lexemes,), strategies=("random",))

    assert left.digest == right.digest


def test_categorical_numeric_equivalents_are_duplicate_but_bool_is_distinct() -> None:
    with pytest.raises(ValidationError):
        ParameterDomain(name="category", kind="categorical", values=(1, 1.0))
    with pytest.raises(ValidationError):
        ParameterDomain(name="category", kind="categorical", values=(0, -0.0))

    domain = ParameterDomain(name="category", kind="categorical", values=(1, True))

    assert domain.values == (1, True)


def test_adjacent_large_integers_are_distinct_categorical_values() -> None:
    first = 123456789012345678901234567890
    second = first + 1

    domain = ParameterDomain(
        name="large_integer",
        kind="categorical",
        values=(first, second),
    )

    assert domain.values == (first, second)


def test_adjacent_large_integer_contracts_have_distinct_digests() -> None:
    first = 123456789012345678901234567890
    second = first + 1
    upper = second + 1
    first_domain = ParameterDomain(
        name="large_integer",
        kind="number",
        minimum=first,
        maximum=upper,
    )
    second_domain = ParameterDomain(
        name="large_integer",
        kind="number",
        minimum=second,
        maximum=upper,
    )

    first_contract = _contract(parameters=(first_domain,), strategies=("random",))
    second_contract = _contract(parameters=(second_domain,), strategies=("random",))

    assert first_contract.digest != second_contract.digest
