from __future__ import annotations

import random
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from organelleverse.core.errors import OrganelleContractError
from organelleverse.optimization.evaluation import CapabilityIdentityV3
from organelleverse.optimization.models import ParameterDomain
from organelleverse.optimization.profiles_v3 import identity_from_capability_entry
from organelleverse.optimization.strategies import (
    promote_successive_halving,
    strategy_for,
    validate_proposal_batch,
)
from organelleverse.optimization.strategy_models import (
    AdaptiveStudyRequest,
    EvaluatorBinding,
    OpaqueArtifactBinding,
    SearchObservation,
    StrategyBinding,
    StrategySpec,
    SuccessiveHalvingConfig,
    SuccessiveHalvingState,
)
from organelleverse.optimization.strategy_registry import StrategyRegistry


def identity(capability_id: str = "optimization.grid") -> CapabilityIdentityV3:
    return CapabilityIdentityV3(
        capability_id=capability_id,
        bundle_version="1.0.0",
        contract_version="1.0",
        bundle_content_hash="sha256:" + "a" * 64,
        execution_identity=None,
        implementation="native",
        surface="native",
    )


def spec(seed: int = 17) -> StrategySpec:
    return StrategySpec(
        parameters=(
            ParameterDomain(name="left", kind="integer", minimum=1, maximum=2, step=1),
            ParameterDomain(name="right", kind="categorical", values=("a", "b")),
        ),
        seed=seed,
        total_design_size=4,
    )


def core_binding(kind: str) -> StrategyBinding:
    return StrategyBinding(kind=kind, identity=identity(f"optimization.{kind}"))  # type: ignore[arg-type]


def proposals(kind: str, seed: int = 17) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    strategy = strategy_for(kind)
    state = strategy.initialize(spec(seed), core_binding(kind))
    batch = strategy.propose(state, (), 4)
    return tuple((proposal.candidate_digest, proposal.parameters) for proposal in batch.proposals)


def test_core_strategies_replay_and_grid_is_rightmost_fastest() -> None:
    for kind in ("grid", "random", "space_filling"):
        assert proposals(kind) == proposals(kind)

    assert tuple(parameters for _, parameters in proposals("grid")) == (
        {"left": 1, "right": "a"},
        {"left": 1, "right": "b"},
        {"left": 2, "right": "a"},
        {"left": 2, "right": "b"},
    )


def test_changed_seed_only_changes_seeded_strategies() -> None:
    assert proposals("grid", 17) == proposals("grid", 19)
    assert proposals("random", 17) != proposals("random", 19)
    assert proposals("space_filling", 17) != proposals("space_filling", 19)


def test_space_filling_freezes_design_across_chunked_replay() -> None:
    continuous = StrategySpec(
        parameters=(ParameterDomain(name="x", kind="number", minimum=0.0, maximum=1.0),),
        seed=23,
        total_design_size=4,
    )
    strategy = strategy_for("space_filling")
    state = strategy.initialize(continuous, core_binding("space_filling"))
    whole = strategy.propose(state, (), 4)
    first = strategy.propose(state, (), 2)
    second = strategy.propose(first.state, (), 2)

    assert first.proposals + second.proposals == whole.proposals
    values = (proposal.parameters["x"] for proposal in whole.proposals)
    strata = {min(int(float(value) * 4), 3) for value in values if isinstance(value, (int, float))}
    assert strata == {0, 1, 2, 3}


def test_grid_takes_prefix_of_huge_integer_domain_without_materializing_it() -> None:
    huge = StrategySpec(
        parameters=(
            ParameterDomain(name="iterations", kind="integer", minimum=0, maximum=10**12, step=1),
        ),
        seed=1,
        total_design_size=2,
    )
    strategy = strategy_for("grid")
    batch = strategy.propose(strategy.initialize(huge, core_binding("grid")), (), 2)
    assert tuple(item.parameters for item in batch.proposals) == (
        {"iterations": 0},
        {"iterations": 1},
    )


def test_registry_requires_exact_admitted_binding(tmp_path: Path) -> None:
    from tests.optimization.test_profiles_v3 import entry

    capability = entry("native", tmp_path, "optimization.grid")
    binding = StrategyBinding(kind="grid", identity=identity_from_capability_entry(capability))
    registry = StrategyRegistry()

    with pytest.raises(OrganelleContractError) as unadmitted:
        registry.register(
            binding,
            strategy_for("grid"),
            entry=capability.model_copy(update={"status": "rejected"}),
        )
    assert unadmitted.value.code == "optimization.strategy_not_admitted"

    registry.register(binding, strategy_for("grid"), entry=capability)
    other = entry("native", tmp_path, "strategy.other")
    other_binding = StrategyBinding(kind="grid", identity=identity_from_capability_entry(other))
    with pytest.raises(OrganelleContractError) as duplicate:
        registry.register(other_binding, strategy_for("grid"), entry=other)
    assert duplicate.value.code == "optimization.strategy_binding_mismatch"
    forged = binding.model_copy(
        update={
            "identity": identity().model_copy(update={"bundle_content_hash": "sha256:" + "b" * 64})
        }
    )
    with pytest.raises(OrganelleContractError) as mismatch:
        registry.resolve(forged)
    assert mismatch.value.code == "optimization.strategy_binding_mismatch"


@pytest.mark.parametrize(
    "case", ["duplicate", "outside", "over_budget", "cross_batch", "ordinary_rung"]
)
def test_untrusted_proposals_are_revalidated_before_execution(case: str) -> None:
    binding = core_binding("grid")
    state = strategy_for("grid").initialize(spec(), binding)
    batch = strategy_for("grid").propose(state, (), 2)
    proposals = batch.proposals
    remaining = 2
    validation_state = state
    if case == "duplicate":
        proposals = (proposals[0], proposals[0])
    elif case == "outside":
        proposals = (proposals[0].model_copy(update={"parameters": {"left": 9, "right": "a"}}),)
    else:
        if case == "over_budget":
            remaining = 1
        elif case == "cross_batch":
            validation_state = batch.state
        else:
            proposals = (
                proposals[0].model_copy(
                    update={
                        "base_candidate_digest": proposals[0].candidate_digest,
                        "resource_rung": 1,
                    }
                ),
            )

    with pytest.raises(OrganelleContractError) as raised:
        validate_proposal_batch(
            spec().parameters,
            proposals,
            remaining_budget=remaining,
            prior_state=validation_state,
            binding=binding,
        )
    assert raised.value.code == "optimization.strategy_proposal_invalid"


def test_successive_halving_promotes_ceil_feasible_fraction_with_digest_ties() -> None:
    config = SuccessiveHalvingConfig(
        resource_parameter="left",
        rungs=(1, 2),
        eta=2,
        base_strategy=StrategyBinding(kind="grid", identity=identity()),
    )
    observations = tuple(
        SearchObservation(
            candidate_digest="sha256:" + digit * 64,
            metrics={"accuracy": score},
            feasible=feasible,
            failed=failed,
        )
        for digit, score, feasible, failed in (
            ("a", 0.9, True, False),
            ("b", 0.9, True, False),
            ("c", 0.8, True, False),
            ("d", 1.0, False, False),
            ("e", 1.0, True, True),
        )
    )

    promoted = promote_successive_halving(
        observations,
        config=config,
        objective_order=(("accuracy", "maximize"),),
    )

    assert promoted == ("sha256:" + "a" * 64, "sha256:" + "b" * 64)


def test_successive_halving_filters_dominated_candidates_before_reduction() -> None:
    config = SuccessiveHalvingConfig(
        resource_parameter="left",
        rungs=(1, 2),
        eta=2,
        base_strategy=StrategyBinding(kind="grid", identity=identity()),
    )
    observations = tuple(
        SearchObservation(
            candidate_digest="sha256:" + digit * 64,
            metrics={"accuracy": accuracy, "runtime": runtime},
            feasible=True,
        )
        for digit, accuracy, runtime in (
            ("a", 10.0, 10.0),
            ("b", 9.0, 9.0),
            ("c", 8.0, 10.0),
        )
    )
    promoted = promote_successive_halving(
        observations,
        config=config,
        objective_order=(("accuracy", "maximize"), ("runtime", "minimize")),
    )
    assert promoted == ("sha256:" + "a" * 64, "sha256:" + "b" * 64)


def test_seeded_random_exhausts_small_finite_space_without_touching_global_rng() -> None:
    import random

    random.seed(91)
    expected = random.random()
    random.seed(91)
    strategy = strategy_for("random")
    state = strategy.initialize(spec(), core_binding("random"))
    first = strategy.propose(state, (), 4)
    second = strategy.propose(first.state, (), 4)
    assert len({item.candidate_digest for item in first.proposals}) == 4
    assert first.exhausted
    assert second.proposals == () and second.exhausted
    assert random.random() == expected


def test_seeded_random_samples_huge_finite_space_without_materializing_it() -> None:
    huge = StrategySpec(
        parameters=(
            ParameterDomain(name="iterations", kind="integer", minimum=0, maximum=10**12, step=1),
        ),
        seed=91,
        total_design_size=2,
    )
    strategy = strategy_for("random")
    batch = strategy.propose(strategy.initialize(huge, core_binding("random")), (), 2)
    assert len(batch.proposals) == 2
    assert len({item.candidate_digest for item in batch.proposals}) == 2
    assert batch.exhausted


def test_continuous_random_retry_exhaustion_fails_instead_of_claiming_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from organelleverse.optimization import strategies

    continuous = StrategySpec(
        parameters=(ParameterDomain(name="x", kind="number", minimum=0.0, maximum=1.0),),
        seed=1,
        total_design_size=2,
    )

    def collide(_domain: ParameterDomain, _rng: random.Random) -> JsonValue:
        return 0.5

    monkeypatch.setattr(strategies, "_sample", collide)
    strategy = strategy_for("random")
    with pytest.raises(OrganelleContractError) as raised:
        strategy.propose(strategy.initialize(continuous, core_binding("random")), (), 2)
    assert raised.value.code == "optimization.strategy_proposal_invalid"


def test_space_filling_finite_collisions_are_filled_without_replacement() -> None:
    finite = StrategySpec(
        parameters=(
            ParameterDomain(name="left", kind="boolean"),
            ParameterDomain(name="right", kind="boolean"),
        ),
        seed=4,
        total_design_size=4,
    )
    strategy = strategy_for("space_filling")
    batch = strategy.propose(strategy.initialize(finite, core_binding("space_filling")), (), 4)
    assert len(batch.proposals) == 4
    assert len({item.candidate_digest for item in batch.proposals}) == 4
    assert batch.exhausted


def test_successive_halving_has_typed_replayable_rung_state_and_stable_base_identity(
    tmp_path: Path,
) -> None:
    from tests.optimization.test_profiles_v3 import entry

    base_entry = entry("native", tmp_path, "optimization.grid")
    base_binding = StrategyBinding(kind="grid", identity=identity_from_capability_entry(base_entry))
    outer_binding = StrategyBinding(
        kind="successive_halving", identity=identity("optimization.successive_halving")
    )
    config = SuccessiveHalvingConfig(
        resource_parameter="left", rungs=(1, 2), eta=2, base_strategy=base_binding
    )
    halving_spec = StrategySpec(
        parameters=spec().parameters,
        seed=5,
        total_design_size=2,
        successive_halving=config,
        objective_order=(("accuracy", "maximize"),),
    )
    registry = StrategyRegistry()
    registry.register(base_binding, strategy_for("grid"), entry=base_entry)
    strategy = strategy_for("successive_halving", base_resolver=registry.resolve)
    state = strategy.initialize(halving_spec, outer_binding)
    assert isinstance(state, SuccessiveHalvingState)

    first = strategy.propose(state, (), 2)
    assert isinstance(first.state, SuccessiveHalvingState)
    assert (
        validate_proposal_batch(
            halving_spec.parameters,
            first.proposals,
            remaining_budget=2,
            prior_state=state,
            binding=outer_binding,
        )
        == first.proposals
    )
    assert {item.resource_rung for item in first.proposals} == {1}
    assert all(item.base_candidate_digest is not None for item in first.proposals)
    assert all(item.candidate_digest != item.base_candidate_digest for item in first.proposals)
    observations = tuple(
        SearchObservation(
            candidate_digest=item.candidate_digest,
            base_candidate_digest=item.base_candidate_digest,
            resource_rung=item.resource_rung,
            metrics={"accuracy": float(index)},
            feasible=True,
        )
        for index, item in enumerate(first.proposals)
    )
    promoted = strategy.observe(first.state, observations)
    assert isinstance(promoted, SuccessiveHalvingState)
    assert promoted.phase == "ready" and promoted.rung_index == 1

    second = strategy.propose(promoted, (), 1)
    assert len(second.proposals) == 1
    assert second.proposals[0].resource_rung == 2
    assert second.proposals[0].base_candidate_digest in {
        item.base_candidate_digest for item in first.proposals
    }
    assert second.proposals[0].candidate_digest not in {
        item.candidate_digest for item in first.proposals
    }

    for malicious, malicious_state in (
        (first.proposals[:1], state),
        (
            (
                first.proposals[0].model_copy(
                    update={"parameters": {"left": 1, "right": "forged"}}
                ),
                first.proposals[1],
            ),
            state,
        ),
        (first.proposals, first.state),
    ):
        with pytest.raises(OrganelleContractError) as rejected:
            validate_proposal_batch(
                halving_spec.parameters,
                malicious,
                remaining_budget=2,
                prior_state=malicious_state,
                binding=outer_binding,
            )
        assert rejected.value.code == "optimization.strategy_proposal_invalid"


def test_successive_halving_stops_when_no_candidate_is_feasible(tmp_path: Path) -> None:
    from tests.optimization.test_profiles_v3 import entry

    base_entry = entry("native", tmp_path, "optimization.grid")
    base_binding = StrategyBinding(kind="grid", identity=identity_from_capability_entry(base_entry))
    outer_binding = StrategyBinding(
        kind="successive_halving", identity=identity("optimization.successive_halving")
    )
    config = SuccessiveHalvingConfig(
        resource_parameter="left", rungs=(1, 2), eta=2, base_strategy=base_binding
    )
    halving_spec = StrategySpec(
        parameters=spec().parameters,
        seed=5,
        total_design_size=2,
        successive_halving=config,
        objective_order=(("accuracy", "maximize"),),
    )
    registry = StrategyRegistry()
    registry.register(base_binding, strategy_for("grid"), entry=base_entry)
    strategy = strategy_for("successive_halving", base_resolver=registry.resolve)
    state = strategy.initialize(halving_spec, outer_binding)
    first = strategy.propose(state, (), 2)
    observations = tuple(
        SearchObservation(
            candidate_digest=item.candidate_digest,
            base_candidate_digest=item.base_candidate_digest,
            resource_rung=item.resource_rung,
            metrics={"accuracy": 0.0},
            feasible=False,
        )
        for item in first.proposals
    )
    terminal = strategy.observe(first.state, observations)
    assert isinstance(terminal, SuccessiveHalvingState)
    assert terminal.phase == "complete"
    exhausted = strategy.propose(terminal, (), 2)
    assert exhausted.proposals == () and exhausted.exhausted


def test_successive_halving_rejects_an_unregistered_base_binding(tmp_path: Path) -> None:
    from tests.optimization.test_profiles_v3 import entry

    base_entry = entry("native", tmp_path, "optimization.grid")
    admitted = StrategyBinding(kind="grid", identity=identity_from_capability_entry(base_entry))
    forged = admitted.model_copy(
        update={
            "identity": admitted.identity.model_copy(
                update={"bundle_content_hash": "sha256:" + "f" * 64}
            )
        }
    )
    config = SuccessiveHalvingConfig(
        resource_parameter="left", rungs=(1, 2), eta=2, base_strategy=forged
    )
    halving_spec = StrategySpec(
        parameters=spec().parameters,
        seed=5,
        total_design_size=2,
        successive_halving=config,
        objective_order=(("accuracy", "maximize"),),
    )
    registry = StrategyRegistry()
    registry.register(admitted, strategy_for("grid"), entry=base_entry)
    strategy = strategy_for("successive_halving", base_resolver=registry.resolve)
    with pytest.raises(OrganelleContractError) as raised:
        strategy.initialize(
            halving_spec,
            StrategyBinding(
                kind="successive_halving", identity=identity("optimization.successive_halving")
            ),
        )
    assert raised.value.code == "optimization.strategy_binding_mismatch"


def test_adaptive_request_requires_exact_bindings_and_complete_objective_order() -> None:
    from tests.optimization.test_contracts_v3 import contract

    enabled = contract()
    binding = StrategyBinding(kind="grid", identity=enabled.strategies[0].identity)
    evaluator = EvaluatorBinding(
        identity=enabled.evaluator.identity,
        evaluator_digest="sha256:" + "e" * 64,
    )
    request = AdaptiveStudyRequest(
        study_id="study-demo",
        contract=enabled,
        strategy_binding=binding,
        evaluator_binding=evaluator,
        objective_order=("accuracy",),
        finalist_limit=1,
        input_digest="sha256:" + "1" * 64,
        fixed_parameters_digest="sha256:" + "2" * 64,
        input_artifact=OpaqueArtifactBinding(
            artifact_ref="artifact:strategy-inputs", content_digest="sha256:" + "1" * 64
        ),
        fixed_parameters_artifact=OpaqueArtifactBinding(
            artifact_ref="artifact:strategy-fixed", content_digest="sha256:" + "2" * 64
        ),
        environment_digest="sha256:" + "3" * 64,
    )
    assert request.strategy_binding == binding

    with pytest.raises(ValidationError):
        AdaptiveStudyRequest.model_validate({**request.model_dump(), "objective_order": ()})

    forged = binding.model_copy(update={"identity": identity("optimization.grid")})
    with pytest.raises(ValidationError):
        AdaptiveStudyRequest.model_validate({**request.model_dump(), "strategy_binding": forged})


def test_adaptive_request_requires_exact_successive_halving_base_binding() -> None:
    from organelleverse.optimization.contracts_v3 import StrategyReferenceV3
    from tests.optimization.test_contracts_v3 import contract

    base = contract()
    enabled = contract(
        strategies=(
            *base.strategies,
            StrategyReferenceV3(
                kind="successive_halving",
                identity=identity("optimization.successive_halving"),
            ),
        )
    )
    halving_reference = next(
        item for item in enabled.strategies if item.kind == "successive_halving"
    )
    grid_reference = next(item for item in enabled.strategies if item.kind == "grid")
    config = SuccessiveHalvingConfig(
        resource_parameter="iterations",
        rungs=(2, 4),
        eta=2,
        base_strategy=StrategyBinding(kind="grid", identity=grid_reference.identity),
    )
    payload = {
        "study_id": "study-halving",
        "contract": enabled,
        "strategy_binding": StrategyBinding(
            kind="successive_halving", identity=halving_reference.identity
        ),
        "evaluator_binding": EvaluatorBinding(
            identity=enabled.evaluator.identity,
            evaluator_digest="sha256:" + "e" * 64,
        ),
        "objective_order": ("accuracy",),
        "finalist_limit": 1,
        "input_digest": "sha256:" + "1" * 64,
        "fixed_parameters_digest": "sha256:" + "2" * 64,
        "input_artifact": OpaqueArtifactBinding(
            artifact_ref="artifact:halving-input", content_digest="sha256:" + "1" * 64
        ),
        "fixed_parameters_artifact": OpaqueArtifactBinding(
            artifact_ref="artifact:halving-fixed", content_digest="sha256:" + "2" * 64
        ),
        "environment_digest": "sha256:" + "3" * 64,
        "successive_halving": config,
    }
    assert AdaptiveStudyRequest.model_validate(payload).successive_halving == config

    forged = config.model_copy(
        update={
            "base_strategy": config.base_strategy.model_copy(
                update={"identity": identity("optimization.grid")}
            )
        }
    )
    with pytest.raises(ValidationError):
        AdaptiveStudyRequest.model_validate({**payload, "successive_halving": forged})
