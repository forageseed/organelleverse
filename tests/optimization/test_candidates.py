"""Deterministic bounded candidate generation."""

from __future__ import annotations

from organelleverse.optimization import (
    ObjectiveSpec,
    OptimizationBudget,
    OptimizationContract,
    ParameterDomain,
    grid_candidates,
    random_candidates,
)
from organelleverse.optimization.candidates import _finite_cardinality


def _contract(
    parameters: tuple[ParameterDomain, ...],
    *,
    max_trials: int = 8,
    seed: int = 17,
    strategies: tuple[str, ...] = ("explicit", "grid", "random", "agent"),
) -> OptimizationContract:
    return OptimizationContract(
        target_capability_id="demo.optimize",
        target_bundle_version="1.0.0",
        target_contract_version="1.0",
        bundle_content_hash="sha256:" + "a" * 64,
        parameters=parameters,
        objective=ObjectiveSpec(direction="maximize"),
        strategies=strategies,  # type: ignore[arg-type]
        seed=seed,
        budget=OptimizationBudget(max_trials=max_trials, parallelism=2),
    )


def test_grid_preserves_declaration_order_and_rightmost_changes_fastest() -> None:
    contract = _contract(
        (
            ParameterDomain(name="count", kind="integer", minimum=1, maximum=6, step=2),
            ParameterDomain(name="mode", kind="categorical", values=("fast", "exact")),
            ParameterDomain(name="mask", kind="boolean"),
        ),
        max_trials=7,
    )

    assert grid_candidates(contract) == (
        {"count": 1, "mode": "fast", "mask": False},
        {"count": 1, "mode": "fast", "mask": True},
        {"count": 1, "mode": "exact", "mask": False},
        {"count": 1, "mode": "exact", "mask": True},
        {"count": 3, "mode": "fast", "mask": False},
        {"count": 3, "mode": "fast", "mask": True},
        {"count": 3, "mode": "exact", "mask": False},
    )


def test_seeded_random_is_reproducible_bounded_unique_and_exhausts_finite_space() -> None:
    finite = _contract(
        (
            ParameterDomain(name="count", kind="integer", minimum=1, maximum=3, step=2),
            ParameterDomain(name="mask", kind="boolean"),
        ),
        max_trials=8,
    )

    first = random_candidates(finite)
    second = random_candidates(finite)

    assert first == second
    assert len(first) == 4
    assert len({tuple(candidate.items()) for candidate in first}) == 4
    assert {candidate["count"] for candidate in first} == {1, 3}
    assert {candidate["mask"] for candidate in first} == {False, True}


def test_random_continuous_and_logarithmic_numbers_stay_within_closed_bounds() -> None:
    contract = _contract(
        (
            ParameterDomain(name="linear", kind="number", minimum=-2.0, maximum=3.0),
            ParameterDomain(
                name="logarithmic",
                kind="number",
                minimum=0.001,
                maximum=100.0,
                logarithmic=True,
            ),
        ),
        max_trials=8,
        strategies=("explicit", "random", "agent"),
    )

    candidates = random_candidates(contract)

    assert len(candidates) == 8
    assert len({tuple(candidate.items()) for candidate in candidates}) == 8
    assert all(-2.0 <= candidate["linear"] <= 3.0 for candidate in candidates)
    assert all(0.001 <= candidate["logarithmic"] <= 100.0 for candidate in candidates)


def test_huge_integer_domain_is_arithmetic_lazy_and_budget_bounded() -> None:
    domain = ParameterDomain(
        name="position",
        kind="integer",
        minimum=0,
        maximum=10**12,
        step=3,
    )
    contract = _contract((domain,), max_trials=256)

    assert _finite_cardinality(domain) == (10**12 // 3) + 1
    grid = grid_candidates(contract)
    sampled = random_candidates(contract)

    assert len(grid) == len(sampled) == 256
    assert grid[0] == {"position": 0}
    assert grid[-1] == {"position": 255 * 3}
    assert len({candidate["position"] for candidate in sampled}) == 256
    assert all(
        0 <= candidate["position"] <= 10**12 and candidate["position"] % 3 == 0
        for candidate in sampled
    )
