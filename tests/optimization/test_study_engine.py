"""End-to-end contract tests for the governed adaptive study engine."""

from __future__ import annotations

import multiprocessing
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from organelleverse.core.errors import OrganelleContractError
from organelleverse.optimization.contracts_v3 import (
    BaselineCandidateV3,
    OptimizationBudgetV3,
    PriorCandidateV3,
    StrategyReferenceV3,
)
from organelleverse.optimization.profiles_v3 import identity_from_capability_entry
from organelleverse.optimization.strategies import candidate_digest, strategy_for
from organelleverse.optimization.strategy_models import (
    AdaptiveStudyRequest,
    CandidateProposal,
    EvaluatorBinding,
    OpaqueArtifactBinding,
    StrategyBinding,
    SuccessiveHalvingConfig,
)
from organelleverse.optimization.strategy_registry import StrategyRegistry
from organelleverse.optimization.study_engine import (
    AdaptiveStudyEngine,
    AdoptedStudyIdentity,
    EvaluationRequest,
)
from organelleverse.plugin_experiments.adaptive_models import FinalExecutionLink
from organelleverse.plugin_experiments.store import ExperimentStore
from tests.optimization.evaluator_fixtures import (
    MEMORY_LIMIT as _MEMORY_LIMIT,
)
from tests.optimization.evaluator_fixtures import (
    BackgroundSideEffectEvaluatorRunner,
    CpuBurnerEvaluatorRunner,
    CrashingEvaluatorRunner,
    FakeEvaluatorRunner,
    MemoryBurnerEvaluatorRunner,
    MismatchedEvaluatorRunner,
    SecretFailingEvaluatorRunner,
    UncooperativeEvaluatorRunner,
)
from tests.optimization.test_contracts_v3 import contract
from tests.optimization.test_profiles_v3 import entry

_PROJECT_ROOT = str(Path(__file__).parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_HASH_C = "sha256:" + "c" * 64
_HASH_D = "sha256:" + "d" * 64


def _request(
    tmp_path: Path,
    *,
    study_id: str = "study-engine",
    improve: bool = True,
) -> tuple[AdaptiveStudyRequest, StrategyRegistry, FakeEvaluatorRunner, ExperimentStore]:
    del improve
    strategy_entry = entry("native", tmp_path, "optimization.grid")
    strategy_identity = identity_from_capability_entry(strategy_entry)
    enabled = contract(
        strategies=(StrategyReferenceV3(kind="grid", identity=strategy_identity),),
        budget=OptimizationBudgetV3(
            max_trials=8,
            parallelism=1,
            max_wall_time_seconds=60,
            max_cpu_time_seconds=60,
            max_peak_memory_bytes=_MEMORY_LIMIT,
        ),
    )
    # These engine tests exercise dispatch, validation, and exact adoption.
    # Process-isolation overhead is sampled independently for baseline and
    # candidate workers, so a tight relative RSS/time adoption guard makes
    # otherwise identical fake evidence depend on host scheduling. Dedicated
    # contract/adoption tests cover the production ratio thresholds.
    enabled = enabled.model_copy(
        update={
            "adoption": enabled.adoption.model_copy(
                update={
                    "maximum_wall_time_ratio": 1_000_000_000.0,
                    "maximum_peak_memory_ratio": 1_000_000_000.0,
                }
            )
        }
    )
    strategy_binding = StrategyBinding(kind="grid", identity=strategy_identity)
    evaluator_binding = EvaluatorBinding(
        identity=enabled.evaluator.identity,
        evaluator_digest=_HASH_C,
    )
    registry = StrategyRegistry()
    registry.register(strategy_binding, strategy_for("grid"), entry=strategy_entry)
    request = AdaptiveStudyRequest(
        study_id=study_id,
        contract=enabled,
        strategy_binding=strategy_binding,
        evaluator_binding=evaluator_binding,
        objective_order=("accuracy",),
        finalist_limit=1,
        input_digest=_HASH_C,
        fixed_parameters_digest=_HASH_D,
        input_artifact=OpaqueArtifactBinding(
            artifact_ref="artifact:study-inputs", content_digest=_HASH_C
        ),
        fixed_parameters_artifact=OpaqueArtifactBinding(
            artifact_ref="artifact:study-fixed-parameters", content_digest=_HASH_D
        ),
        environment_digest="sha256:" + "e" * 64,
    )
    runner = FakeEvaluatorRunner(evaluator_binding)
    return request, registry, runner, ExperimentStore(tmp_path / "store")


def _engine(
    registry: StrategyRegistry,
    runner: FakeEvaluatorRunner,
    store: ExperimentStore,
) -> AdaptiveStudyEngine:
    return AdaptiveStudyEngine(
        strategy_registry=registry,
        evaluator_runner=runner,
        store=store,
    )


def test_package_root_exports_engine_constructor_chain(tmp_path: Path) -> None:
    from organelleverse.optimization import (
        AdaptiveStudyEngine as PublicAdaptiveStudyEngine,
    )
    from organelleverse.optimization import (
        AdaptiveStudyRequest as PublicAdaptiveStudyRequest,
    )
    from organelleverse.optimization import (
        AdoptedStudyIdentity as PublicAdoptedStudyIdentity,
    )
    from organelleverse.optimization import (
        EvaluationRequest as PublicEvaluationRequest,
    )
    from organelleverse.optimization import (
        EvaluatorBinding as PublicEvaluatorBinding,
    )
    from organelleverse.optimization import (
        EvaluatorRunner as PublicEvaluatorRunner,
    )
    from organelleverse.optimization import (
        StrategyBinding as PublicStrategyBinding,
    )
    from organelleverse.optimization import (
        StrategyRegistry as PublicStrategyRegistry,
    )
    from organelleverse.optimization import (
        SuccessiveHalvingConfig as PublicSuccessiveHalvingConfig,
    )

    assert PublicAdaptiveStudyEngine is AdaptiveStudyEngine
    assert PublicAdaptiveStudyRequest is AdaptiveStudyRequest
    assert PublicAdoptedStudyIdentity is AdoptedStudyIdentity
    assert PublicEvaluationRequest is EvaluationRequest
    assert PublicEvaluatorBinding is EvaluatorBinding
    assert PublicStrategyBinding is StrategyBinding
    assert PublicStrategyRegistry is StrategyRegistry
    assert PublicSuccessiveHalvingConfig is SuccessiveHalvingConfig
    assert PublicEvaluatorRunner.__name__ == "EvaluatorRunner"

    _, registry, runner, store = _request(tmp_path)
    engine = PublicAdaptiveStudyEngine(
        strategy_registry=registry,
        evaluator_runner=runner,
        store=store,
    )
    assert isinstance(engine, PublicAdaptiveStudyEngine)


def test_submit_run_lookup_and_write_once_final_link(tmp_path: Path) -> None:
    request, registry, runner, store = _request(tmp_path)
    engine = _engine(registry, runner, store)

    submitted = engine.submit(request)
    assert submitted.status == "created" and submitted.attempts == ()
    terminal = engine.run_to_terminal(request)

    assert terminal.status == "adopted", (terminal.decision, terminal.attempts[-1])
    assert terminal.execution_selection is not None
    assert terminal.execution_selection.source == "adopted"
    assert terminal.execution_selection.parameters == {"iterations": 6}
    assert terminal.execution_selection.parameter_digest == terminal.adopted_parameter_digest
    assert engine.get(request.study_id) == terminal
    adopted_identity = AdoptedStudyIdentity.from_request(request)
    assert engine.find_adopted(adopted_identity) == terminal
    assert (
        engine.find_adopted(adopted_identity.model_copy(update={"environment_digest": _HASH_D}))
        is None
    )

    assert terminal.adopted_parameter_digest is not None
    link = FinalExecutionLink(
        run_receipt_id="run-final",
        executed_parameter_digest=terminal.adopted_parameter_digest,
        linked_at=datetime.now(UTC),
    )
    linked = engine.record_final_execution(request.study_id, link)
    assert linked.final_execution_link == link
    assert engine.record_final_execution(request.study_id, link) == linked
    with pytest.raises(OrganelleContractError):
        engine.record_final_execution(
            request.study_id,
            link.model_copy(update={"run_receipt_id": "run-conflict"}),
        )


def test_adopted_lookup_fails_closed_on_duplicate_exact_identity(tmp_path: Path) -> None:
    request, registry, runner, store = _request(tmp_path, study_id="study-identity-a")
    engine = _engine(registry, runner, store)
    first = engine.run_to_terminal(request)
    assert first.status == "adopted", (
        first.decision,
        first.aggregates,
        first.attempts[-1],
    )
    duplicate = request.model_copy(update={"study_id": "study-identity-b"})
    second = engine.run_to_terminal(duplicate)
    assert second.status == "adopted"

    with pytest.raises(OrganelleContractError) as conflict:
        engine.find_adopted(AdoptedStudyIdentity.from_request(request))
    assert conflict.value.code == "optimization.adopted_identity_conflict"


def test_no_improvement_mechanically_retains_freshly_validated_baseline(
    tmp_path: Path,
) -> None:
    request, registry, _, store = _request(tmp_path, study_id="study-fallback")
    runner = FakeEvaluatorRunner(request.evaluator_binding, improve=False)
    terminal = _engine(registry, runner, store).run_to_terminal(request)

    assert terminal.status == "baseline_retained"
    assert terminal.decision is not None
    assert terminal.decision.reason_code == "optimization.insufficient_improvement"
    assert terminal.execution_selection is not None
    assert terminal.execution_selection.source == "verified_baseline"
    assert terminal.execution_selection.parameters == request.contract.baseline.parameters
    assert terminal.final_execution_link is None


def test_search_and_validation_use_disjoint_splits_and_fresh_run_ids(tmp_path: Path) -> None:
    request, registry, runner, store = _request(tmp_path, study_id="study-splits")
    terminal = _engine(registry, runner, store).run_to_terminal(request)

    search = [item for item in terminal.attempts if item.identity.split == "search"]
    validation = [item for item in terminal.attempts if item.identity.split == "validation"]
    assert search and validation
    assert {item.identity.split_id for item in search} == {
        request.contract.benchmark.search.split_id
    }
    assert {item.identity.split_content_hash for item in search} == {
        request.contract.benchmark.search.content_hash
    }
    assert {item.identity.split_id for item in validation} == {
        request.contract.benchmark.validation.split_id
    }
    assert {item.identity.split_content_hash for item in validation} == {
        request.contract.benchmark.validation.content_hash
    }
    assert {item.run_id for item in search}.isdisjoint(item.run_id for item in validation)


def test_minegraph_prior_is_only_a_measured_initial_candidate(tmp_path: Path) -> None:
    request, registry, _, store = _request(tmp_path, study_id="study-prior")
    prior = PriorCandidateV3(
        prior_id="minegraph-seed",
        source="minegraph",
        parameters={"iterations": 8},
        evidence_refs=("minegraph:estimate-1",),
    )
    seeded = request.model_copy(
        update={"contract": request.contract.model_copy(update={"priors": (prior,)})}
    )
    runner = FakeEvaluatorRunner(seeded.evaluator_binding)
    terminal = _engine(registry, runner, store).run_to_terminal(seeded)

    search_digests = {
        item.identity.candidate_digest
        for item in terminal.attempts
        if item.identity.split == "search"
    }
    assert candidate_digest({"iterations": 8}) in search_digests
    assert search_digests == {candidate_digest({"iterations": value}) for value in (2, 4, 6, 8)}
    assert terminal.decision is not None


def test_evaluator_binding_mismatch_fails_before_any_dispatch(tmp_path: Path) -> None:
    request, registry, _, store = _request(tmp_path, study_id="study-mismatch")
    wrong = FakeEvaluatorRunner(
        request.evaluator_binding.model_copy(update={"evaluator_digest": _HASH_D})
    )

    with pytest.raises(OrganelleContractError) as failure:
        _engine(registry, wrong, store).run_to_terminal(request)

    assert failure.value.code == "optimization.evaluator_mismatch"
    assert wrong.call_count.value == 0
    assert store.list_studies() == ()


@pytest.mark.parametrize(
    ("max_trials", "peak_memory", "reason"),
    [
        (1, 128, "optimization.budget_exhausted"),
        (8, _MEMORY_LIMIT + 1, "optimization.resource_constraint_failed"),
    ],
)
def test_trial_and_resource_limits_are_hard_gates(
    tmp_path: Path,
    max_trials: int,
    peak_memory: int,
    reason: str,
) -> None:
    request, registry, _, store = _request(
        tmp_path, study_id=f"study-gate-{max_trials}-{peak_memory}"
    )
    bounded = request.model_copy(
        update={
            "contract": request.contract.model_copy(
                update={
                    "budget": request.contract.budget.model_copy(update={"max_trials": max_trials})
                }
            )
        }
    )
    runner = FakeEvaluatorRunner(bounded.evaluator_binding, peak_memory_bytes=peak_memory)
    terminal = _engine(registry, runner, store).run_to_terminal(bounded)

    assert terminal.status == "failed"
    assert terminal.decision is not None and terminal.decision.reason_code == reason
    assert terminal.budget.attempted_runs <= max_trials
    assert all(item.peak_memory_bytes >= peak_memory for item in terminal.attempts)


def test_wall_time_grant_does_not_trust_an_uncooperative_runner(tmp_path: Path) -> None:
    request, registry, _, store = _request(tmp_path, study_id="study-wall-timeout")
    bounded = request.model_copy(
        update={
            "contract": request.contract.model_copy(
                update={
                    "budget": request.contract.budget.model_copy(
                        update={"max_wall_time_seconds": 1}
                    )
                }
            )
        }
    )
    runner = UncooperativeEvaluatorRunner(bounded.evaluator_binding)

    started = time.monotonic()
    terminal = _engine(registry, runner, store).run_to_terminal(bounded)

    assert time.monotonic() - started < 3.5
    assert terminal.status == "failed"
    assert terminal.decision is not None
    assert terminal.decision.reason_code == "optimization.resource_constraint_failed"
    assert terminal.attempts[-1].error is not None
    assert terminal.attempts[-1].error.code == "optimization.resource_constraint_failed"
    time.sleep(0.2)
    assert runner.delayed_side_effect.value == 0
    assert not any(child.name.startswith("run-") for child in multiprocessing.active_children())


@pytest.mark.parametrize(
    ("runner_type", "budget_update"),
    [
        (CpuBurnerEvaluatorRunner, {"max_cpu_time_seconds": 1}),
        (MemoryBurnerEvaluatorRunner, {"max_peak_memory_bytes": 8 * 1024 * 1024}),
    ],
)
def test_cpu_and_memory_grants_include_descendant_processes(
    tmp_path: Path,
    runner_type: type[FakeEvaluatorRunner],
    budget_update: dict[str, int],
) -> None:
    request, registry, _, store = _request(
        tmp_path, study_id=f"study-isolated-{runner_type.__name__}"
    )
    bounded = request.model_copy(
        update={
            "contract": request.contract.model_copy(
                update={"budget": request.contract.budget.model_copy(update=budget_update)}
            )
        }
    )
    runner = runner_type(bounded.evaluator_binding)

    terminal = _engine(registry, runner, store).run_to_terminal(bounded)

    assert terminal.status == "failed"
    assert terminal.decision is not None
    assert terminal.decision.reason_code == "optimization.resource_constraint_failed"
    assert runner.call_count.value == 1
    assert not any(child.name.startswith("run-") for child in multiprocessing.active_children())


def test_successful_attempt_kills_background_descendants_before_return(tmp_path: Path) -> None:
    request, registry, _, store = _request(tmp_path, study_id="study-background-descendant")
    side_effect = tmp_path / "escaped-side-effect"
    runner = BackgroundSideEffectEvaluatorRunner(request.evaluator_binding, side_effect)

    terminal = _engine(registry, runner, store).run_to_terminal(request)
    time.sleep(0.75)

    assert terminal.status == "adopted"
    assert not side_effect.exists()
    assert not any(child.name.startswith("run-") for child in multiprocessing.active_children())


@pytest.mark.parametrize(
    ("runner_type", "reason"),
    [
        (SecretFailingEvaluatorRunner, "optimization.evaluator_failed"),
        (MismatchedEvaluatorRunner, "optimization.evaluator_mismatch"),
    ],
)
def test_evaluator_failures_keep_stable_public_reasons_without_private_details(
    tmp_path: Path,
    runner_type: type[FakeEvaluatorRunner],
    reason: str,
) -> None:
    request, registry, _, store = _request(tmp_path, study_id=f"study-{reason.rsplit('.', 1)[1]}")
    runner = runner_type(request.evaluator_binding)

    terminal = _engine(registry, runner, store).run_to_terminal(request)

    assert terminal.status == "failed"
    assert terminal.decision is not None and terminal.decision.reason_code == reason
    assert terminal.attempts[-1].error is not None
    assert terminal.attempts[-1].error.code == reason
    assert "private-token" not in terminal.model_dump_json()


@pytest.mark.parametrize("resource", ["wall", "cpu"])
def test_wall_and_cpu_grants_stop_further_dispatch(tmp_path: Path, resource: str) -> None:
    request, registry, _, store = _request(tmp_path, study_id=f"study-{resource}-gate")
    budget_update = (
        {"max_wall_time_seconds": 2} if resource == "wall" else {"max_cpu_time_seconds": 1}
    )
    bounded = request.model_copy(
        update={
            "contract": request.contract.model_copy(
                update={"budget": request.contract.budget.model_copy(update=budget_update)}
            )
        }
    )
    runner = FakeEvaluatorRunner(
        bounded.evaluator_binding,
        wall_time_seconds=3.0 if resource == "wall" else 0.5,
        cpu_time_seconds=2.0 if resource == "cpu" else 0.5,
    )
    terminal = _engine(registry, runner, store).run_to_terminal(bounded)

    assert terminal.status == "failed"
    assert terminal.decision is not None
    assert terminal.decision.reason_code == "optimization.resource_constraint_failed"
    assert runner.call_count.value <= 1


def _halving_request(
    tmp_path: Path,
    *,
    study_id: str,
    max_trials: int = 8,
    prior_resource: int | None = None,
) -> tuple[AdaptiveStudyRequest, StrategyRegistry, FakeEvaluatorRunner, ExperimentStore]:
    grid_entry = entry("native", tmp_path, "optimization.grid")
    halving_entry = entry("native", tmp_path, "optimization.successive_halving")
    grid_binding = StrategyBinding(kind="grid", identity=identity_from_capability_entry(grid_entry))
    halving_binding = StrategyBinding(
        kind="successive_halving",
        identity=identity_from_capability_entry(halving_entry),
    )
    enabled = contract(
        parameters=(
            contract().parameters[0],
            contract().parameters[0].model_copy(update={"name": "resource"}),
        ),
        baseline=BaselineCandidateV3(parameters={"iterations": 2, "resource": 2}),
        strategies=(
            StrategyReferenceV3(kind="grid", identity=grid_binding.identity),
            StrategyReferenceV3(kind="successive_halving", identity=halving_binding.identity),
        ),
        priors=(
            ()
            if prior_resource is None
            else (
                PriorCandidateV3(
                    prior_id="minegraph-sh",
                    source="minegraph",
                    parameters={"iterations": 8, "resource": prior_resource},
                    evidence_refs=("minegraph:sh-prior",),
                ),
            )
        ),
        budget=OptimizationBudgetV3(
            max_trials=max_trials,
            parallelism=1,
            max_wall_time_seconds=60,
            max_cpu_time_seconds=60,
            max_peak_memory_bytes=_MEMORY_LIMIT,
        ),
    )
    config = SuccessiveHalvingConfig(
        resource_parameter="resource",
        rungs=(2, 8),
        eta=2,
        base_strategy=grid_binding,
    )
    request = AdaptiveStudyRequest(
        study_id=study_id,
        contract=enabled,
        strategy_binding=halving_binding,
        evaluator_binding=EvaluatorBinding(
            identity=enabled.evaluator.identity, evaluator_digest=_HASH_C
        ),
        objective_order=("accuracy",),
        finalist_limit=1,
        input_digest=_HASH_C,
        fixed_parameters_digest=_HASH_D,
        input_artifact=OpaqueArtifactBinding(
            artifact_ref="artifact:halving-inputs", content_digest=_HASH_C
        ),
        fixed_parameters_artifact=OpaqueArtifactBinding(
            artifact_ref="artifact:halving-fixed-parameters", content_digest=_HASH_D
        ),
        environment_digest="sha256:" + "e" * 64,
        successive_halving=config,
    )
    registry = StrategyRegistry()
    registry.register(grid_binding, strategy_for("grid"), entry=grid_entry)
    registry.register(
        halving_binding,
        strategy_for("successive_halving", base_resolver=registry.resolve),
        entry=halving_entry,
    )
    runner = FakeEvaluatorRunner(request.evaluator_binding)
    return request, registry, runner, ExperimentStore(tmp_path / f"store-{study_id}")


def test_successive_halving_validates_only_full_resource_finalists(tmp_path: Path) -> None:
    request, registry, runner, store = _halving_request(tmp_path, study_id="study-halving")
    terminal = _engine(registry, runner, store).run_to_terminal(request)

    validation = [item for item in terminal.attempts if item.identity.split == "validation"]
    assert validation
    assert {item.identity.resource_rung for item in validation} == {8}
    assert all(
        item.identity.parameter_digest == item.identity.candidate_digest for item in validation
    )


def test_successive_halving_does_not_start_when_a_complete_rung_cannot_fit(
    tmp_path: Path,
) -> None:
    request, registry, runner, store = _halving_request(
        tmp_path,
        study_id="study-halving-budget",
        max_trials=6,
    )
    terminal = _engine(registry, runner, store).run_to_terminal(request)

    assert terminal.status == "failed"
    assert terminal.decision is not None
    assert terminal.decision.reason_code == "optimization.budget_exhausted"
    assert runner.call_count.value == 1


def test_successive_halving_prior_is_rebased_to_a_declared_rung(tmp_path: Path) -> None:
    request, registry, _, store = _halving_request(
        tmp_path,
        study_id="study-halving-prior",
        prior_resource=4,
    )
    runner = FakeEvaluatorRunner(request.evaluator_binding)
    terminal = _engine(registry, runner, store).run_to_terminal(request)

    prior_digest = candidate_digest({"iterations": 8, "resource": 2})
    prior_attempts = [
        item
        for item in terminal.attempts
        if item.identity.split == "search" and item.identity.candidate_digest == prior_digest
    ]
    assert prior_attempts
    assert all(item.identity.resource_rung in {2, 8} for item in prior_attempts)


def test_restart_never_reruns_a_successful_logical_repeat(tmp_path: Path) -> None:
    request, registry, first_runner, store = _request(tmp_path, study_id="study-restart")
    first_engine = _engine(registry, first_runner, store)
    terminal = first_engine.run_to_terminal(request)
    successful = {
        (
            item.identity.split,
            item.identity.candidate_digest,
            item.identity.resource_rung,
            item.identity.logical_repeat_index,
        )
        for item in terminal.attempts
        if item.status == "succeeded"
    }

    reopened_runner = FakeEvaluatorRunner(request.evaluator_binding)
    reopened = _engine(registry, reopened_runner, ExperimentStore(tmp_path / "store"))
    assert reopened.run_to_terminal(request) == terminal
    assert reopened_runner.call_count.value == 0
    assert len(successful) == sum(item.status == "succeeded" for item in terminal.attempts)


def test_interrupted_pending_batch_resumes_without_rerunning_successes(tmp_path: Path) -> None:
    request, registry, _, store = _request(tmp_path, study_id="study-interrupted")
    crashing = CrashingEvaluatorRunner(request.evaluator_binding, crash_on_call=2)
    with pytest.raises(KeyboardInterrupt):
        _engine(registry, crashing, store).run_to_terminal(request)

    before = store.get_study(request.study_id)
    assert any(item.status == "running" for item in before.attempts)
    baseline_digest = before.attempts[0].identity.candidate_digest

    reopened_store = ExperimentStore(tmp_path / "store")
    recovered = reopened_store.get_study(request.study_id)
    assert recovered.status == "interrupted"
    assert any(item.status == "interrupted" for item in recovered.attempts)
    resumed_runner = FakeEvaluatorRunner(request.evaluator_binding)
    terminal = _engine(registry, resumed_runner, reopened_store).run_to_terminal(request)

    assert terminal.status in {"adopted", "baseline_retained"}
    newly_recorded = terminal.attempts[len(recovered.attempts) :]
    assert not any(
        item.identity.split == "search" and item.identity.candidate_digest == baseline_digest
        for item in newly_recorded
    )
    interrupted = next(item for item in terminal.attempts if item.status == "interrupted")
    retry = next(
        item
        for item in terminal.attempts
        if item.identity.logical_repeat_index == interrupted.identity.logical_repeat_index
        and item.identity.candidate_digest == interrupted.identity.candidate_digest
        and item.identity.attempt_no == interrupted.identity.attempt_no + 1
    )
    assert retry.run_id != interrupted.run_id


@pytest.mark.parametrize(
    "tamper",
    [
        "seed",
        "design_size",
        "step",
        "phase",
        "missing_state",
        "bad_proposal",
        "canonical_out_of_domain",
        "emitted_order",
    ],
)
def test_recovery_rejects_tampered_strategy_checkpoint(tmp_path: Path, tamper: str) -> None:
    request, registry, _, store = _request(tmp_path, study_id=f"study-checkpoint-{tamper}")
    crashing = CrashingEvaluatorRunner(request.evaluator_binding, crash_on_call=2)
    with pytest.raises(KeyboardInterrupt):
        _engine(registry, crashing, store).run_to_terminal(request)
    reopened = ExperimentStore(tmp_path / "store")
    record = reopened.get_study(request.study_id)
    checkpoint = dict(record.strategy_state)
    if tamper == "seed":
        state = dict(cast(dict[str, JsonValue], checkpoint["strategy"]))
        spec = dict(cast(dict[str, JsonValue], state["spec"]))
        spec["seed"] = cast(int, spec["seed"]) + 1
        state["spec"] = spec
        checkpoint["strategy"] = state
    elif tamper == "design_size":
        state = dict(cast(dict[str, JsonValue], checkpoint["strategy"]))
        spec = dict(cast(dict[str, JsonValue], state["spec"]))
        spec["total_design_size"] = cast(int, spec["total_design_size"]) + 1
        state["spec"] = spec
        checkpoint["strategy"] = state
        checkpoint["strategy_spec"] = spec
    elif tamper == "step":
        state = dict(cast(dict[str, JsonValue], checkpoint["strategy"]))
        state["step"] = cast(int, state["step"]) + 1
        checkpoint["strategy"] = state
    elif tamper == "phase":
        checkpoint["phase"] = "complete"
    elif tamper == "missing_state":
        checkpoint.pop("strategy")
    elif tamper == "bad_proposal":
        checkpoint["candidate_parameters"] = {"bad": {"parameters": "not-a-mapping"}}
    elif tamper == "canonical_out_of_domain":
        parameters = {"iterations": 999}
        digest = candidate_digest(parameters)
        checkpoint["candidate_parameters"] = {
            digest: CandidateProposal(
                parameters=parameters,
                candidate_digest=digest,
            ).model_dump(mode="json")
        }
    else:
        state = dict(cast(dict[str, JsonValue], checkpoint["strategy"]))
        state["emitted"] = list(reversed(cast(list[JsonValue], state["emitted"])))
        checkpoint["strategy"] = state
    reopened.update_study(record.model_copy(update={"strategy_state": checkpoint}))

    terminal = _engine(
        registry, FakeEvaluatorRunner(request.evaluator_binding), reopened
    ).run_to_terminal(request)
    assert terminal.status == "failed"
    assert terminal.decision is not None
    assert terminal.decision.reason_code == "optimization.strategy_nondeterministic"
