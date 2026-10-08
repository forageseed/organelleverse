"""Durable adaptive studies share the legacy experiment store safely."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.result import ErrorDetail
from organelleverse.optimization.evaluation import CapabilityIdentityV3
from organelleverse.optimization.strategies import candidate_digest
from organelleverse.plugin_experiments.adaptive_models import (
    AdaptiveStudyRecord,
    AdoptionDecision,
    AttemptIdentity,
    BudgetAccount,
    ExecutionSelection,
    FinalExecutionLink,
    RepeatEvidence,
)
from organelleverse.plugin_experiments.models import (
    ExperimentRecord,
    ExperimentRequest,
    ExperimentTrial,
)
from organelleverse.plugin_experiments.store import ExperimentStore

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64
_ADOPTED_PARAMETERS = {"iterations": 4}
_ADOPTED_DIGEST = candidate_digest(_ADOPTED_PARAMETERS)


def _evaluator_identity() -> CapabilityIdentityV3:
    return CapabilityIdentityV3(
        capability_id="demo.evaluator",
        bundle_version="1.0.0",
        contract_version="1.0",
        bundle_content_hash=_HASH_C,
        execution_identity=None,
        implementation="native",
        surface="native",
    )


def _study(*, study_id: str = "study-a1", status: str = "created") -> AdaptiveStudyRecord:
    return AdaptiveStudyRecord(
        study_id=study_id,
        status=status,  # type: ignore[arg-type]
        submitted_at=datetime.now(UTC),
        contract_digest=_HASH_A,
        evaluator_digest=_HASH_B,
        evaluator_identity=_evaluator_identity(),
    )


def _attempt(index: int, *, status: str = "succeeded", value: float = 1.0) -> RepeatEvidence:
    return RepeatEvidence(
        identity=AttemptIdentity(
            split="search",
            split_id="search-v1",
            split_content_hash=_HASH_C,
            candidate_digest=f"sha256:{index:064x}",
            parameter_digest=f"sha256:{index:064x}",
            logical_repeat_index=index,
            attempt_no=0,
            evaluator_digest=_HASH_B,
            evaluator_identity=_evaluator_identity(),
        ),
        status=status,  # type: ignore[arg-type]
        run_id=f"run-{index}",
        seed=index,
        dispatch_digest=_HASH_A,
        granted_wall_time_seconds=10.0,
        granted_cpu_time_seconds=10.0,
        granted_peak_memory_bytes=1024,
        metrics={"quality": value},
        wall_time_seconds=1.5,
        cpu_time_seconds=1.0,
        peak_memory_bytes=128,
    )


def _adopt(store: ExperimentStore) -> AdaptiveStudyRecord:
    record = store.create_study(_study())
    for status in (
        "baseline_search",
        "searching",
        "shortlist",
        "baseline_validation",
        "validating",
    ):
        record = store.update_study(record.model_copy(update={"status": status}))
    return store.update_study(
        record.model_copy(
            update={
                "status": "adopted",
                "adopted_parameter_digest": _ADOPTED_DIGEST,
                "execution_selection": ExecutionSelection(
                    source="adopted",
                    parameters=_ADOPTED_PARAMETERS,
                    parameter_digest=_ADOPTED_DIGEST,
                ),
                "decision": AdoptionDecision(
                    status="adopted",
                    reason_code="optimization.candidate_adopted",
                    candidate_digest=_ADOPTED_DIGEST,
                    parameter_digest=_ADOPTED_DIGEST,
                ),
                "completed_at": datetime.now(UTC),
            }
        )
    )


def test_adaptive_model_has_discriminator_and_rejects_unsafe_study_id() -> None:
    assert _study().record_kind == "adaptive_study.v1"
    with pytest.raises(ValidationError):
        _study(study_id="../escape")


def test_execution_selection_is_closed_and_content_bound() -> None:
    with pytest.raises(ValidationError):
        ExecutionSelection.model_validate(
            {
                "source": "adopted",
                "parameters": {"iterations": 4},
                "parameter_digest": _HASH_A,
            }
        )
    with pytest.raises(ValidationError):
        ExecutionSelection.model_validate(
            {
                "source": "trial",
                "parameters": {"iterations": 4},
                "parameter_digest": _ADOPTED_DIGEST,
            }
        )


def test_terminal_status_closes_execution_selection_source(tmp_path: Path) -> None:
    adopted = _adopt(ExperimentStore(tmp_path / "experiments"))
    payload = adopted.model_dump(mode="json")
    payload["execution_selection"] = {
        "source": "verified_baseline",
        "parameters": _ADOPTED_PARAMETERS,
        "parameter_digest": _ADOPTED_DIGEST,
    }

    with pytest.raises(ValidationError):
        AdaptiveStudyRecord.model_validate(payload)


def test_mixed_loader_round_trips_adaptive_study(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    store = ExperimentStore(root)
    study = store.create_study(_study())

    reopened = ExperimentStore(root)
    assert reopened.get_study(study.study_id) == study
    assert reopened.list_studies() == (study,)


def test_same_root_loader_keeps_legacy_and_adaptive_types_separate(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    store = ExperimentStore(root)
    legacy = ExperimentRecord(
        experiment_id="exp-legacy",
        capability_id="demo.experiment",
        status="queued",
        submitted_at=datetime.now(UTC),
        request=ExperimentRequest(
            capability_id="demo.experiment",
            inputs={},
            candidates=({"threshold": 0.5},),
        ),
        trials=(ExperimentTrial(index=0, parameters={"threshold": 0.5}, status="queued"),),
    )
    store.create(legacy)
    store.create_study(_study())

    reopened = ExperimentStore(root)
    assert reopened.get("exp-legacy").status == "failed"
    assert reopened.get_study("study-a1").status == "created"
    assert len(reopened.list()) == 1
    assert len(reopened.list_studies()) == 1


def test_legacy_create_cannot_overwrite_an_adaptive_study_id(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    store.create_study(_study())
    legacy = ExperimentRecord(
        experiment_id="study-a1",
        capability_id="demo.experiment",
        status="queued",
        submitted_at=datetime.now(UTC),
        request=ExperimentRequest(
            capability_id="demo.experiment",
            inputs={},
            candidates=({"threshold": 0.5},),
        ),
        trials=(ExperimentTrial(index=0, parameters={"threshold": 0.5}, status="queued"),),
    )
    with pytest.raises(OrganelleContractError) as captured:
        store.create(legacy)
    assert captured.value.code == "experiment.duplicate_id"
    assert store.get_study("study-a1") == _study().model_copy(
        update={"submitted_at": store.get_study("study-a1").submitted_at}
    )


def test_unknown_record_discriminator_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    root.mkdir()
    (root / "study-unknown.json").write_text(
        json.dumps({"record_kind": "adaptive_study.v999"}), encoding="utf-8"
    )
    with pytest.raises(OrganelleContractError) as captured:
        ExperimentStore(root)
    assert captured.value.code == "experiment.history_invalid"


def test_reopen_interrupts_only_inflight_attempt_and_keeps_study_resumable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiments"
    store = ExperimentStore(root)
    running = _attempt(0, status="running")
    created = store.create_study(_study())
    store.update_study(created.model_copy(update={"status": "baseline_search"}))
    store.update_study(store.get_study("study-a1").model_copy(update={"status": "searching"}))
    store.append_repeat_evidence("study-a1", running)

    reopened = ExperimentStore(root)
    recovered = reopened.get_study("study-a1")
    assert recovered.status == "interrupted"
    assert recovered.attempts[0].status == "interrupted"
    assert recovered.attempts[0].error is not None
    assert recovered.attempts[0].error.code == "experiment.run_interrupted"
    assert recovered.budget.attempted_runs == 1

    retry = _attempt(0).model_copy(
        update={
            "identity": running.identity.model_copy(update={"attempt_no": 1}),
            "run_id": "run-retry",
        }
    )
    resumed = reopened.append_repeat_evidence("study-a1", retry)
    assert [item.identity.attempt_no for item in resumed.attempts] == [0, 1]
    assert resumed.budget.attempted_runs == 2


def test_append_is_idempotent_and_conflicting_identity_fails(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    store.create_study(_study())
    evidence = _attempt(0)

    first = store.append_repeat_evidence("study-a1", evidence)
    second = store.append_repeat_evidence("study-a1", evidence)
    assert first == second
    assert second.budget == BudgetAccount(
        attempted_runs=1,
        wall_time_seconds=1.5,
        cpu_time_seconds=1.0,
        peak_memory_bytes=128,
    )

    conflicting = evidence.model_copy(update={"metrics": {"quality": 2.0}})
    with pytest.raises(OrganelleContractError) as captured:
        store.append_repeat_evidence("study-a1", conflicting)
    assert captured.value.code == "optimization.attempt_conflict"


def test_running_attempt_completes_atomically_without_double_counting(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    store.create_study(_study())
    running = _attempt(0, status="running").model_copy(
        update={"metrics": {}, "wall_time_seconds": 0.25, "cpu_time_seconds": 0.1}
    )
    begun = store.append_repeat_evidence("study-a1", running)
    assert begun.budget.attempted_runs == 1

    completed = _attempt(0).model_copy(update={"run_id": running.run_id, "seed": running.seed})
    finished = store.append_repeat_evidence("study-a1", completed)
    assert len(finished.attempts) == 1
    assert finished.attempts[0].status == "succeeded"
    assert finished.budget.attempted_runs == 1
    assert finished.budget.wall_time_seconds == 1.5
    assert finished.budget.cpu_time_seconds == 1.0

    rewrite = completed.model_copy(
        update={
            "status": "failed",
            "metrics": {},
            "error": ErrorDetail(code="synthetic.failure", message="must not replace terminal"),
        }
    )
    with pytest.raises(OrganelleContractError) as captured:
        store.append_repeat_evidence("study-a1", rewrite)
    assert captured.value.code == "optimization.attempt_conflict"


def test_generic_study_update_cannot_bypass_atomic_attempt_append(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    original = store.create_study(_study())
    bypass = original.model_copy(
        update={
            "attempts": (_attempt(0),),
            "budget": BudgetAccount(attempted_runs=1),
        }
    )
    with pytest.raises(OrganelleContractError) as captured:
        store.update_study(bypass)
    assert captured.value.code == "optimization.attempt_history_write_forbidden"
    assert store.get_study("study-a1") == original


def test_two_thread_appends_do_not_lose_different_repeats(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    store.create_study(_study())

    def append(item: RepeatEvidence) -> AdaptiveStudyRecord:
        return store.append_repeat_evidence("study-a1", item)

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(append, (_attempt(0), _attempt(1))))

    persisted = store.get_study("study-a1")
    assert {item.identity.logical_repeat_index for item in persisted.attempts} == {0, 1}
    assert persisted.budget.attempted_runs == 2
    assert persisted.budget.wall_time_seconds == 3.0


def test_run_id_cannot_be_reused_for_another_attempt(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    store.create_study(_study())
    first = _attempt(0)
    store.append_repeat_evidence("study-a1", first)
    reused = _attempt(1).model_copy(update={"run_id": first.run_id})
    with pytest.raises(OrganelleContractError) as captured:
        store.append_repeat_evidence("study-a1", reused)
    assert captured.value.code == "optimization.attempt_conflict"


def test_terminal_rejects_append_and_final_link_is_write_once_durable(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    store = ExperimentStore(root)
    _adopt(store)

    with pytest.raises(OrganelleContractError) as captured:
        store.append_repeat_evidence("study-a1", _attempt(0))
    assert captured.value.code == "optimization.study_terminal"

    link = FinalExecutionLink(
        run_receipt_id="run-final-1",
        executed_parameter_digest=_ADOPTED_DIGEST,
        linked_at=datetime.now(UTC),
    )
    linked = store.record_final_execution("study-a1", link)
    assert store.record_final_execution("study-a1", link) == linked
    assert ExperimentStore(root).get_study("study-a1").final_execution_link == link

    with pytest.raises(OrganelleContractError) as conflict:
        store.record_final_execution(
            "study-a1",
            link.model_copy(update={"run_receipt_id": "run-final-2"}),
        )
    assert conflict.value.code == "optimization.final_execution_conflict"


def test_legacy_adopted_record_reopens_but_is_not_executable(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    store = ExperimentStore(root)
    adopted = _adopt(store)
    payload = adopted.model_dump(mode="json")
    payload.pop("execution_selection")
    (root / "study-a1.json").write_text(json.dumps(payload), encoding="utf-8")

    reopened = ExperimentStore(root)
    legacy = reopened.get_study("study-a1")

    assert legacy.status == "adopted"
    assert legacy.execution_selection is None
    with pytest.raises(OrganelleContractError) as captured:
        reopened.record_final_execution(
            "study-a1",
            FinalExecutionLink(
                run_receipt_id="run-legacy",
                executed_parameter_digest=_ADOPTED_DIGEST,
                linked_at=datetime.now(UTC),
            ),
        )
    assert captured.value.code == "optimization.study_not_executable"


def test_legacy_adopted_with_final_link_reopens_but_is_not_executable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiments"
    store = ExperimentStore(root)
    adopted = _adopt(store)
    link = FinalExecutionLink(
        run_receipt_id="provider:run/123",
        executed_parameter_digest=_ADOPTED_DIGEST,
        linked_at=datetime.now(UTC),
    )
    linked = store.record_final_execution(adopted.study_id, link)
    payload = linked.model_dump(mode="json")
    payload.pop("execution_selection")
    (root / "study-a1.json").write_text(json.dumps(payload), encoding="utf-8")

    reopened = ExperimentStore(root)
    legacy = reopened.get_study("study-a1")

    assert legacy.final_execution_link == link
    assert legacy.execution_selection is None
    with pytest.raises(OrganelleContractError) as captured:
        reopened.record_final_execution("study-a1", link)
    assert captured.value.code == "optimization.study_not_executable"


def test_final_link_requires_matching_adopted_parameter_digest(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    _adopt(store)
    with pytest.raises(OrganelleContractError) as captured:
        store.record_final_execution(
            "study-a1",
            FinalExecutionLink(
                run_receipt_id="run-final",
                executed_parameter_digest=_HASH_A,
                linked_at=datetime.now(UTC),
            ),
        )
    assert captured.value.code == "optimization.final_execution_mismatch"


def test_verified_baseline_selection_can_link_exact_final_execution(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    store = ExperimentStore(root)
    record = store.create_study(_study())
    for status in (
        "baseline_search",
        "searching",
        "shortlist",
        "baseline_validation",
        "validating",
    ):
        record = store.update_study(record.model_copy(update={"status": status}))
    selection = ExecutionSelection(
        source="verified_baseline",
        parameters={"iterations": 2},
        parameter_digest=candidate_digest({"iterations": 2}),
    )
    record = store.update_study(
        record.model_copy(
            update={
                "status": "baseline_retained",
                "decision": AdoptionDecision(
                    status="baseline_retained",
                    reason_code="optimization.insufficient_improvement",
                ),
                "execution_selection": selection,
                "completed_at": datetime.now(UTC),
            }
        )
    )
    link = FinalExecutionLink(
        run_receipt_id="run-baseline",
        executed_parameter_digest=selection.parameter_digest,
        execution_status="failed",
        linked_at=datetime.now(UTC),
    )

    linked = store.record_final_execution(record.study_id, link)

    assert linked.execution_selection == selection
    reopened_link = ExperimentStore(root).get_study(record.study_id).final_execution_link
    assert reopened_link == link
    assert reopened_link is not None and reopened_link.execution_status == "failed"


def test_terminal_without_execution_selection_cannot_link(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    record = store.create_study(_study())
    record = store.update_study(record.model_copy(update={"status": "baseline_search"}))
    store.update_study(
        record.model_copy(
            update={
                "status": "baseline_retained",
                "decision": AdoptionDecision(
                    status="baseline_retained",
                    reason_code="optimization.validation_incomplete",
                ),
                "completed_at": datetime.now(UTC),
            }
        )
    )

    with pytest.raises(OrganelleContractError) as captured:
        store.record_final_execution(
            "study-a1",
            FinalExecutionLink(
                run_receipt_id="run-invalid",
                executed_parameter_digest=_HASH_A,
                linked_at=datetime.now(UTC),
            ),
        )

    assert captured.value.code == "optimization.study_not_executable"


def test_create_study_requires_pristine_created_record(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    dirty = _study().model_copy(
        update={"attempts": (_attempt(0),), "budget": BudgetAccount(attempted_runs=1)}
    )
    with pytest.raises(OrganelleContractError) as captured:
        store.create_study(dirty)
    assert captured.value.code == "optimization.study_not_pristine"


def test_attempt_numbers_are_monotone_per_logical_repeat(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    store.create_study(_study())
    first = _attempt(0, status="running")

    skipped = first.model_copy(
        update={"identity": first.identity.model_copy(update={"attempt_no": 1})}
    )
    with pytest.raises(OrganelleContractError) as captured:
        store.append_repeat_evidence("study-a1", skipped)
    assert captured.value.code == "optimization.attempt_sequence_invalid"

    store.append_repeat_evidence("study-a1", first)
    with pytest.raises(OrganelleContractError):
        store.append_repeat_evidence("study-a1", skipped)


def test_successful_logical_repeat_forbids_a_new_attempt_number(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    store.create_study(_study())
    succeeded = _attempt(0)
    store.append_repeat_evidence("study-a1", succeeded)
    retry = succeeded.model_copy(
        update={"identity": succeeded.identity.model_copy(update={"attempt_no": 1})}
    )
    with pytest.raises(OrganelleContractError) as captured:
        store.append_repeat_evidence("study-a1", retry)
    assert captured.value.code == "optimization.attempt_sequence_invalid"


def test_record_validator_closes_terminal_and_nonterminal_payloads() -> None:
    with pytest.raises(ValidationError):
        AdaptiveStudyRecord.model_validate(
            _study().model_dump(mode="json")
            | {
                "decision": {
                    "status": "baseline_retained",
                    "reason_code": "optimization.baseline_retained",
                }
            }
        )
    with pytest.raises(ValidationError):
        AdaptiveStudyRecord.model_validate(
            _study().model_dump(mode="json")
            | {"status": "baseline_retained", "completed_at": datetime.now(UTC)}
        )


def test_update_study_freezes_identity_and_enforces_state_transitions(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    original = store.create_study(_study())
    changed_digest = original.model_copy(update={"contract_digest": _HASH_C})
    with pytest.raises(OrganelleContractError) as frozen:
        store.update_study(changed_digest)
    assert frozen.value.code == "optimization.study_identity_mismatch"

    illegal = original.model_copy(update={"status": "validating"})
    with pytest.raises(OrganelleContractError) as transition:
        store.update_study(illegal)
    assert transition.value.code == "optimization.study_transition_invalid"


def test_loader_rejects_filename_payload_identity_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    root.mkdir()
    (root / "study-other.json").write_text(_study().model_dump_json(), encoding="utf-8")
    with pytest.raises(OrganelleContractError) as captured:
        ExperimentStore(root)
    assert captured.value.code == "experiment.history_invalid"


def test_legacy_create_rejects_unsafe_path_identity(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    unsafe = ExperimentRecord(
        experiment_id="../escape",
        capability_id="demo.experiment",
        status="queued",
        submitted_at=datetime.now(UTC),
        request=ExperimentRequest(
            capability_id="demo.experiment", inputs={}, candidates=({"x": 1},)
        ),
        trials=(ExperimentTrial(index=0, parameters={"x": 1}, status="queued"),),
    )
    with pytest.raises(OrganelleContractError) as captured:
        store.create(unsafe)
    assert captured.value.code == "experiment.invalid_id"
