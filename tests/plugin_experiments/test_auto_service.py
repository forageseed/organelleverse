"""Unified execute-best orchestration stays deterministic and fail closed."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from organelleverse.capabilities.index import CapabilityIndex
from organelleverse.core.errors import OrganelleContractError
from organelleverse.optimization.auto import AutoOptimizationRequest, OperationInvocation
from organelleverse.optimization.contracts_v3 import (
    OptimizationBudgetV3,
    OptimizationContractV3,
    OptimizationProfileV3,
    StrategyReferenceV3,
)
from organelleverse.optimization.evaluation import EvaluatorContractV3
from organelleverse.optimization.profiles_v3 import identity_from_capability_entry
from organelleverse.optimization.strategies import candidate_digest
from organelleverse.optimization.strategy_models import (
    AdaptiveStudyRequest,
    EvaluatorBinding,
    OpaqueArtifactBinding,
    StrategyBinding,
)
from organelleverse.optimization.study_models import canonical_digest
from organelleverse.plugin_experiments.adaptive_models import (
    AdaptiveStudyRecord,
    AdoptionDecision,
    ExecutionSelection,
    FinalExecutionClaim,
    FinalExecutionLink,
)
from organelleverse.plugin_experiments.auto_service import (
    AutoOptimizationService,
    FinalExecutionReceipt,
    PreparedAdaptiveStudy,
)
from organelleverse.plugin_experiments.store import ExperimentStore
from tests.optimization.test_contracts_v3 import contract, declaration_payload
from tests.optimization.test_profiles_v3 import entry

_HASH_C = "sha256:" + "c" * 64
_HASH_E = "sha256:" + "e" * 64


class _Engine:
    def __init__(self, terminal: AdaptiveStudyRecord, adopted: AdaptiveStudyRecord | None = None):
        self.terminal = terminal
        self.adopted = adopted
        self.find_calls = 0
        self.identities: list[object] = []
        self.run_calls = 0
        self.links: list[FinalExecutionLink] = []

    def find_adopted(self, identity: object) -> AdaptiveStudyRecord | None:
        self.find_calls += 1
        self.identities.append(identity)
        return self.adopted

    def run_to_terminal(self, request: AdaptiveStudyRequest) -> AdaptiveStudyRecord:
        del request
        self.run_calls += 1
        return self.terminal

    def record_final_execution(
        self, study_id: str, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord:
        assert study_id == self.terminal.study_id
        self.links.append(link)
        return self.terminal.model_copy(update={"final_execution_link": link})

    def claim_final_execution(
        self, study_id: str, claim: FinalExecutionClaim
    ) -> AdaptiveStudyRecord:
        assert study_id == self.terminal.study_id
        if self.terminal.final_execution_link is not None:
            return self.terminal
        if self.terminal.final_execution_claim is None:
            self.terminal = self.terminal.model_copy(update={"final_execution_claim": claim})
        return self.terminal

    def complete_final_execution(
        self, study_id: str, claim_id: str, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord:
        assert study_id == self.terminal.study_id
        assert self.terminal.final_execution_claim is not None
        assert self.terminal.final_execution_claim.claim_id == claim_id
        self.links.append(link)
        self.terminal = self.terminal.model_copy(
            update={"final_execution_claim": None, "final_execution_link": link}
        )
        return self.terminal


def _fixture(
    tmp_path: Path,
    *,
    risk: str = "low",
    approval: str = "automatic",
    fixed_parameters: dict[str, object] | None = None,
) -> tuple[CapabilityIndex, OptimizationProfileV3, AdaptiveStudyRequest]:
    target = entry("native", tmp_path, "demo.target")
    evaluator = entry("native", tmp_path, "demo.evaluator")
    strategy = entry("native", tmp_path, "strategy.grid")
    if risk in {"medium", "high"}:
        target = target.model_copy(
            update={
                "bundle": target.bundle.model_copy(
                    update={
                        "contract": target.bundle.contract.model_copy(
                            update={
                                "side_effects": (
                                    "read_files" if risk == "medium" else "subprocess",
                                )
                            }
                        )
                    }
                )
            }
        )
    target_identity = identity_from_capability_entry(target)
    evaluator_identity = identity_from_capability_entry(evaluator)
    strategy_identity = identity_from_capability_entry(strategy)
    payload = declaration_payload()
    payload.update(
        {
            "strategies": (StrategyReferenceV3(kind="grid", identity=strategy_identity),),
            "adoption": contract().adoption.model_copy(update={"approval": approval}),
        }
    )
    enabled = OptimizationContractV3(
        target=target_identity,
        evaluator=EvaluatorContractV3(identity=evaluator_identity),
        **payload,
    )
    profile = OptimizationProfileV3(status="enabled", target=target_identity, contract=enabled)
    invocation = OperationInvocation(
        kind="operation", input={"sample": "A"}, parameters=fixed_parameters or {}
    )
    from organelleverse.optimization.auto import _canonical_digest

    input_digest = _canonical_digest(invocation.input)
    fixed_digest = _canonical_digest(fixed_parameters or {})
    request = AdaptiveStudyRequest(
        study_id="study-auto-service",
        contract=enabled,
        strategy_binding=StrategyBinding(kind="grid", identity=strategy_identity),
        evaluator_binding=EvaluatorBinding(identity=evaluator_identity, evaluator_digest=_HASH_C),
        objective_order=tuple(item.name for item in enabled.objectives),
        finalist_limit=1,
        input_digest=input_digest,
        fixed_parameters_digest=fixed_digest,
        input_artifact=OpaqueArtifactBinding(
            artifact_ref="artifact:auto-input", content_digest=input_digest
        ),
        fixed_parameters_artifact=OpaqueArtifactBinding(
            artifact_ref="artifact:auto-fixed", content_digest=fixed_digest
        ),
        environment_digest=_HASH_E,
    )
    return CapabilityIndex(entries=(target, evaluator, strategy)), profile, request


def _record(
    request: AdaptiveStudyRequest,
    *,
    source: str = "adopted",
    reason_code: str = "optimization.test_terminal",
) -> AdaptiveStudyRecord:
    parameters = {"iterations": 4 if source == "adopted" else 2}
    digest = candidate_digest(parameters)
    status = "adopted" if source == "adopted" else "baseline_retained"
    return AdaptiveStudyRecord(
        study_id=request.study_id,
        status=status,
        submitted_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        contract_digest=request.contract.digest,
        evaluator_digest=request.evaluator_binding.evaluator_digest,
        evaluator_identity=request.evaluator_binding.identity,
        request=request.model_dump(mode="json"),
        decision=AdoptionDecision(
            status=status,
            reason_code=reason_code,
            candidate_digest=digest if status == "adopted" else None,
            parameter_digest=digest if status == "adopted" else None,
        ),
        adopted_parameter_digest=digest if status == "adopted" else None,
        execution_selection=ExecutionSelection(
            source=source, parameters=parameters, parameter_digest=digest
        ),
    )


def _request(
    *, parameters: dict[str, object] | None = None, mode: str = "auto"
) -> AutoOptimizationRequest:
    return AutoOptimizationRequest(
        capability_id="demo.target",
        invocation=OperationInvocation(
            kind="operation", input={"sample": "A"}, parameters=parameters or {}
        ),
        mode=mode,
    )


def test_missing_runtime_and_medium_risk_ask_without_side_effects(tmp_path: Path) -> None:
    index, profile, _ = _fixture(tmp_path)
    calls: list[object] = []
    service = AutoOptimizationService(
        index_provider=lambda: index,
        profile_provider=lambda: {"demo.target": profile},
        prepared_study_provider=None,
        final_executor=lambda capability_id, invocation, parameters: calls.append((invocation, parameters)),
    )

    unavailable = service.execute(_request())
    medium_index, medium_profile, _ = _fixture(tmp_path / "medium", risk="medium")
    medium = AutoOptimizationService(
        index_provider=lambda: medium_index,
        profile_provider=lambda: {"demo.target": medium_profile},
        prepared_study_provider=None,
        final_executor=lambda capability_id, invocation, parameters: calls.append((invocation, parameters)),
    ).execute(_request())

    assert unavailable.decision.decision == "ask"
    assert unavailable.decision.reason_code == "optimization.runtime_unavailable"
    assert medium.decision.decision == "ask"
    assert medium.decision.reason_code == "optimization.risk_approval_required"
    assert calls == []


@pytest.mark.parametrize(
    ("risk", "approval", "mode", "parameters"),
    [
        ("medium", "automatic", "never", {}),
        ("high", "automatic", "never", {}),
        ("medium", "automatic", "auto", {"iterations": 8}),
        ("low", "human_required", "never", {}),
    ],
)
def test_all_executable_decisions_require_the_service_boundary_risk_gate(
    tmp_path: Path,
    risk: str,
    approval: str,
    mode: str,
    parameters: dict[str, object],
) -> None:
    index, profile, _ = _fixture(tmp_path, risk=risk, approval=approval)
    prepared: list[object] = []
    executed: list[object] = []
    service = AutoOptimizationService(
        index_provider=lambda: index,
        profile_provider=lambda: {"demo.target": profile},
        prepared_study_provider=lambda request, snapshot: prepared.append((request, snapshot)),
        final_executor=lambda capability_id, invocation, selected: executed.append((invocation, selected)),
    )

    result = service.execute(_request(parameters=parameters, mode=mode))

    assert result.decision.decision == "ask"
    assert result.decision.reason_code == "optimization.risk_approval_required"
    assert prepared == []
    assert executed == []


def test_explicit_tuned_parameter_executes_direct_baseline_without_preparing(
    tmp_path: Path,
) -> None:
    index, profile, _ = _fixture(tmp_path)
    prepared: list[object] = []
    executed: list[tuple[object, dict[str, object]]] = []
    service = AutoOptimizationService(
        index_provider=lambda: index,
        profile_provider=lambda: {"demo.target": profile},
        prepared_study_provider=lambda request, snapshot: prepared.append((request, snapshot)),
        final_executor=lambda capability_id, invocation, parameters: (
            executed.append((invocation, parameters))
            or FinalExecutionReceipt(run_id="run-direct", status="succeeded")
        ),
    )

    result = service.execute(_request(parameters={"iterations": 8, "fixed": "kept"}))

    assert result.decision.decision == "baseline"
    assert result.run_id == "run-direct"
    assert len(executed) == 1
    invocation, parameters = executed[0]
    assert isinstance(invocation, OperationInvocation)
    assert invocation.parameters == {"iterations": 8, "fixed": "kept"}
    assert parameters == {"iterations": 8, "fixed": "kept"}
    assert prepared == []


def test_reuse_returns_existing_final_receipt_without_rerunning(tmp_path: Path) -> None:
    index, profile, study_request = _fixture(tmp_path)
    adopted = _record(study_request)
    link = FinalExecutionLink(
        run_receipt_id="run-existing",
        executed_parameter_digest=adopted.execution_selection.parameter_digest,  # type: ignore[union-attr]
        linked_at=datetime.now(UTC),
    )
    adopted = adopted.model_copy(update={"final_execution_link": link})
    engine = _Engine(adopted, adopted=adopted)
    executed: list[object] = []
    service = AutoOptimizationService(
        index_provider=lambda: index,
        profile_provider=lambda: {"demo.target": profile},
        prepared_study_provider=lambda request, snapshot: PreparedAdaptiveStudy(
            engine=engine, request=study_request
        ),
        final_executor=lambda capability_id, invocation, parameters: executed.append((invocation, parameters)),
    )

    result = service.execute(_request())

    assert result.decision.decision == "reuse"
    assert result.run_id == "run-existing"
    assert result.execution_status is None
    assert engine.run_calls == 0
    assert engine.links == []
    assert executed == []


def test_optimize_executes_typed_selection_once_and_records_link(tmp_path: Path) -> None:
    index, profile, study_request = _fixture(tmp_path, fixed_parameters={"fixed": "kept"})
    adopted = _record(study_request)
    engine = _Engine(adopted)
    executed: list[tuple[object, dict[str, object]]] = []
    service = AutoOptimizationService(
        index_provider=lambda: index,
        profile_provider=lambda: {"demo.target": profile},
        prepared_study_provider=lambda request, snapshot: PreparedAdaptiveStudy(
            engine=engine, request=study_request
        ),
        final_executor=lambda capability_id, invocation, parameters: (
            executed.append((invocation, parameters))
            or FinalExecutionReceipt(run_id="run-final", status="failed")
        ),
    )

    result = service.execute(_request(parameters={"fixed": "kept"}))

    assert result.decision.decision == "optimize"
    assert result.decision.study_id == study_request.study_id
    assert result.run_id == "run-final"
    assert result.parameter_digest == adopted.execution_selection.parameter_digest  # type: ignore[union-attr]
    invocation, parameters = executed[0]
    assert isinstance(invocation, OperationInvocation)
    assert invocation.input == {"sample": "A"}
    assert invocation.parameters == {"fixed": "kept", "iterations": 4}
    assert parameters == {"fixed": "kept", "iterations": 4}
    assert engine.run_calls == 1
    assert len(engine.links) == 1
    assert engine.links[0].run_receipt_id == "run-final"
    assert engine.links[0].execution_status == "failed"


@pytest.mark.parametrize(
    ("reason_code", "fallback"),
    [
        ("optimization.insufficient_improvement", "validation_not_improved"),
        ("optimization.budget_exhausted", "budget_exhausted"),
        ("optimization.no_feasible_candidate", "no_valid_trial"),
        ("optimization.study_cancelled", "cancelled"),
        ("optimization.evaluator_failed", "study_failed"),
    ],
)
def test_verified_baseline_preserves_durable_fallback_reason(
    tmp_path: Path, reason_code: str, fallback: str
) -> None:
    index, profile, study_request = _fixture(tmp_path)
    retained = _record(
        study_request,
        source="verified_baseline",
        reason_code=reason_code,
    )
    engine = _Engine(retained)
    service = AutoOptimizationService(
        index_provider=lambda: index,
        profile_provider=lambda: {"demo.target": profile},
        prepared_study_provider=lambda request, snapshot: PreparedAdaptiveStudy(
            engine=engine, request=study_request
        ),
        final_executor=lambda capability_id, invocation, parameters: FinalExecutionReceipt(
            run_id=f"run-{fallback}", status="succeeded"
        ),
    )

    result = service.execute(_request())

    assert result.decision.fallback == fallback
    assert result.run_id == f"run-{fallback}"
    assert len(engine.links) == 1


def test_budget_expansion_fails_before_prepare_and_narrowing_changes_reuse_identity(
    tmp_path: Path,
) -> None:
    index, profile, study_request = _fixture(tmp_path)
    adopted = _record(study_request)
    engine = _Engine(adopted, adopted=adopted)
    prepared: list[object] = []

    def provider(request: AutoOptimizationRequest, snapshot: object) -> PreparedAdaptiveStudy:
        prepared.append(snapshot)
        assert request.budget is not None
        narrowed_contract = study_request.contract.model_copy(update={"budget": request.budget})
        narrowed_request = study_request.model_copy(update={"contract": narrowed_contract})
        return PreparedAdaptiveStudy(engine=engine, request=narrowed_request)

    service = AutoOptimizationService(
        index_provider=lambda: index,
        profile_provider=lambda: {"demo.target": profile},
        prepared_study_provider=provider,
        final_executor=lambda capability_id, invocation, parameters: FinalExecutionReceipt(
            run_id="run-budget", status="succeeded"
        ),
    )
    declared = profile.contract.budget  # type: ignore[union-attr]
    expanded = declared.model_copy(update={"max_trials": declared.max_trials + 1})
    with pytest.raises(OrganelleContractError) as failure:
        service.execute(_request().model_copy(update={"budget": expanded}))
    assert failure.value.code == "optimization.budget_escalation"
    assert prepared == []

    narrowed = OptimizationBudgetV3(
        max_trials=4,
        parallelism=1,
        max_wall_time_seconds=30,
        max_cpu_time_seconds=60,
        max_peak_memory_bytes=512,
    )
    result = service.execute(_request().model_copy(update={"budget": narrowed}))

    assert result.decision.decision == "reuse"
    assert len(prepared) == 1
    assert engine.identities[0].budget_digest == canonical_digest(narrowed.model_dump(mode="json"))


class _StorePort:
    def __init__(self, store: ExperimentStore) -> None:
        self.store = store

    def find_adopted(self, identity: object) -> AdaptiveStudyRecord | None:
        del identity
        return next(
            (study for study in self.store.list_studies() if study.status == "adopted"), None
        )

    def run_to_terminal(self, request: AdaptiveStudyRequest) -> AdaptiveStudyRecord:
        return self.store.get_study(request.study_id)

    def record_final_execution(
        self, study_id: str, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord:
        return self.store.record_final_execution(study_id, link)

    def claim_final_execution(
        self, study_id: str, claim: FinalExecutionClaim
    ) -> AdaptiveStudyRecord:
        return self.store.claim_final_execution(study_id, claim)

    def complete_final_execution(
        self, study_id: str, claim_id: str, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord:
        return self.store.complete_final_execution(study_id, claim_id, link)


def _persist_adopted(store: ExperimentStore, terminal: AdaptiveStudyRecord) -> None:
    record = store.create_study(
        terminal.model_copy(
            update={
                "status": "created",
                "completed_at": None,
                "decision": None,
                "adopted_parameter_digest": None,
                "execution_selection": None,
            }
        )
    )
    for status in (
        "baseline_search",
        "searching",
        "shortlist",
        "baseline_validation",
        "validating",
    ):
        record = store.update_study(record.model_copy(update={"status": status}))
    store.update_study(terminal)


def test_two_services_share_one_durable_final_execution_claim(tmp_path: Path) -> None:
    index, profile, study_request = _fixture(tmp_path)
    root = tmp_path / "shared-studies"
    first_store = ExperimentStore(root)
    terminal = _record(study_request)
    _persist_adopted(first_store, terminal)
    second_store = ExperimentStore(root)
    started = threading.Event()
    release = threading.Event()
    execution_count = 0
    count_lock = threading.Lock()

    def execute(capability_id: str, invocation: object, parameters: object) -> FinalExecutionReceipt:
        nonlocal execution_count
        assert capability_id == "demo.target"
        del invocation, parameters
        with count_lock:
            execution_count += 1
        started.set()
        assert release.wait(timeout=5)
        return FinalExecutionReceipt(run_id="run-one", status="succeeded")

    def service(store: ExperimentStore) -> AutoOptimizationService:
        return AutoOptimizationService(
            index_provider=lambda: index,
            profile_provider=lambda: {"demo.target": profile},
            prepared_study_provider=lambda request, snapshot: PreparedAdaptiveStudy(
                engine=_StorePort(store), request=study_request
            ),
            final_executor=execute,
        )

    first = service(first_store)
    second = service(second_store)
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(first.execute, _request())
        assert started.wait(timeout=5)
        blocked = pool.submit(second.execute, _request())
        with pytest.raises(OrganelleContractError) as captured:
            blocked.result(timeout=5)
        assert captured.value.code == "optimization.final_execution_in_progress"
        release.set()
        completed = running.result(timeout=5)

    assert execution_count == 1
    assert completed.run_id == "run-one"
    reopened = ExperimentStore(root).get_study(study_request.study_id)
    assert reopened.final_execution_claim is None
    assert reopened.final_execution_link is not None
    assert reopened.final_execution_link.run_receipt_id == "run-one"
    reused = second.execute(_request())
    assert reused.run_id == "run-one"
    assert execution_count == 1


def test_executor_crash_leaves_a_durable_fail_closed_claim(tmp_path: Path) -> None:
    index, profile, study_request = _fixture(tmp_path)
    root = tmp_path / "crashed-studies"
    store = ExperimentStore(root)
    _persist_adopted(store, _record(study_request))

    def crash(capability_id: str, invocation: object, parameters: object) -> FinalExecutionReceipt:
        assert capability_id == "demo.target"
        del invocation, parameters
        raise RuntimeError("outcome unknown")

    first = AutoOptimizationService(
        index_provider=lambda: index,
        profile_provider=lambda: {"demo.target": profile},
        prepared_study_provider=lambda request, snapshot: PreparedAdaptiveStudy(
            engine=_StorePort(store), request=study_request
        ),
        final_executor=crash,
    )
    with pytest.raises(RuntimeError, match="outcome unknown"):
        first.execute(_request())

    reopened = ExperimentStore(root)
    assert reopened.get_study(study_request.study_id).final_execution_claim is not None
    retry_calls: list[object] = []
    second = AutoOptimizationService(
        index_provider=lambda: index,
        profile_provider=lambda: {"demo.target": profile},
        prepared_study_provider=lambda request, snapshot: PreparedAdaptiveStudy(
            engine=_StorePort(reopened), request=study_request
        ),
        final_executor=lambda capability_id, invocation, parameters: retry_calls.append((invocation, parameters)),
    )

    with pytest.raises(OrganelleContractError) as captured:
        second.execute(_request())
    assert captured.value.code == "optimization.final_execution_in_progress"
    assert retry_calls == []


def test_final_execution_receipts_reject_path_and_uri_values() -> None:
    for unsafe in ("/tmp/secret", "file://secret", "run with spaces"):
        with pytest.raises(ValueError):
            FinalExecutionReceipt(run_id=unsafe, status="succeeded")
