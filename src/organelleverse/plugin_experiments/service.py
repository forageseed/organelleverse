"""Governed optimization experiments with durable per-trial history.

The service materializes explicit, Agent, finite-grid or seeded-random candidates,
then invokes the same admitted Registry binding for every trial. Candidate order
is durable; the capability-declared objective direction selects the best finite
score, and ties select the lowest original candidate index.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic import JsonValue

from organelleverse.capabilities.index import CapabilityIndex, CapabilityStatus
from organelleverse.capabilities.models import PluginCapabilityBundle
from organelleverse.capabilities.plugin_descriptor import describe_plugin
from organelleverse.capabilities.trust import TrustStore
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleError,
    OrganelleInputError,
)
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.operations.registry import OperationRegistry
from organelleverse.operations.spec import PluginOptimization
from organelleverse.optimization import (
    contract_from_capability_entry,
    grid_candidates,
    random_candidates,
)
from organelleverse.plugin_protocol import PLUGIN_OPTIMIZATION_SCORE_METRIC

from .models import ExperimentRecord, ExperimentRequest, ExperimentTrial
from .store import ExperimentStore

__all__ = ["ExperimentService"]

_TERMINAL = frozenset({"succeeded", "failed"})


class ExperimentService:
    """Run governed optimization studies against admitted, trusted plugins."""

    def __init__(
        self,
        *,
        index_provider: Callable[[], CapabilityIndex],
        trust_store: TrustStore,
        store: ExperimentStore,
    ) -> None:
        self._index_provider = index_provider
        self._trust_store = trust_store
        self._store = store
        self._lock = threading.Lock()
        self._closed = False
        self._coordinator = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ov-experiment-coordinator"
        )
        self._trial_executors: set[ThreadPoolExecutor] = set()

    @property
    def store(self) -> ExperimentStore:
        """Read-only record access for projections (e.g. the Result DAG)."""
        return self._store

    # -- admission and request validation -----------------------------------

    def _validate(
        self, index: CapabilityIndex, request: ExperimentRequest
    ) -> tuple[PluginCapabilityBundle, PluginOptimization]:
        # describe() raises input.unknown_capability / capability.conflict.
        entry = index.describe(request.capability_id)
        if entry.status is not CapabilityStatus.ADMITTED:
            raise OrganelleContractError(
                code="capability.not_admitted",
                message="only admitted capabilities can run experiments",
                details={"capability_id": request.capability_id},
            )
        bundle = entry.bundle
        if not isinstance(bundle, PluginCapabilityBundle):
            raise OrganelleContractError(
                code="capability.plugin_required",
                message="experiments require a v2 plugin capability bundle",
                details={"capability_id": request.capability_id},
            )
        optimization = bundle.contract.optimization
        if optimization is None:
            raise OrganelleContractError(
                code="capability.optimization_not_declared",
                message="the plugin declares no optimization experiment",
                details={"capability_id": request.capability_id},
            )
        identity = entry.execution_identity
        if identity is None or not self._trust_store.is_trusted(identity.digest):
            raise OrganelleContractError(
                code="capability.untrusted",
                message="the plugin's current execution identity is not trusted",
                details={"capability_id": request.capability_id},
            )

        descriptor = describe_plugin(bundle)
        value_parameters = {field.name: field for field in descriptor.parameters}
        undeclared = sorted(set(optimization.parameters) - set(value_parameters))
        if undeclared:
            raise OrganelleContractError(
                code="capability.optimization_invalid",
                message="optimization parameters must name declared value parameters",
                details={"parameters": undeclared},
            )
        if len(request.candidates) > optimization.max_trials:
            raise OrganelleContractError(
                code="experiment.max_trials_exceeded",
                message="candidate count exceeds the declared max_trials bound",
                details={
                    "candidates": len(request.candidates),
                    "max_trials": optimization.max_trials,
                },
            )
        optimized = set(optimization.parameters)
        for name, value in request.fixed_parameters.items():
            if name in optimized or name not in value_parameters:
                raise OrganelleContractError(
                    code="experiment.fixed_parameter_invalid",
                    message=(
                        "fixed parameters may contain only non-optimized declared value parameters"
                    ),
                    details={"parameter": name},
                )
            self._check_schema(value_parameters[name].json_schema, value, name, fixed=True)
        for index_position, candidate in enumerate(request.candidates):
            names = set(candidate)
            if names != optimized:
                raise OrganelleContractError(
                    code="experiment.candidate_invalid",
                    message=(
                        "each candidate must contain every and only the declared "
                        "optimization parameters"
                    ),
                    details={
                        "candidate_index": index_position,
                        "missing": sorted(optimized - names),
                        "extra": sorted(names - optimized),
                    },
                )
            for name, value in candidate.items():
                self._check_schema(value_parameters[name].json_schema, value, name, fixed=False)
        return bundle, optimization

    @staticmethod
    def _check_schema(
        schema: dict[str, JsonValue], value: JsonValue, name: str, *, fixed: bool
    ) -> None:
        errors = sorted(
            Draft202012Validator(schema).iter_errors(value),  # pyright: ignore[reportUnknownMemberType]
            key=str,
        )
        if errors:
            raise OrganelleContractError(
                code=(
                    "experiment.fixed_parameter_invalid"
                    if fixed
                    else "experiment.candidate_invalid"
                ),
                message=f"value for {name!r} violates its declared JSON Schema",
                details={"parameter": name, "reason": str(errors[0])},
            )

    # -- submission and bounded execution ------------------------------------

    @staticmethod
    def _materialize_request(
        index: CapabilityIndex,
        request: ExperimentRequest,
    ) -> ExperimentRequest:
        entry = index.describe(request.capability_id)
        contract = contract_from_capability_entry(entry, seed=request.seed)
        if request.strategy not in contract.strategies:
            raise OrganelleContractError(
                code="experiment.strategy_not_allowed",
                message="the capability contract does not allow this optimization strategy",
                details={
                    "strategy": request.strategy,
                    "allowed": list(contract.strategies),
                },
            )
        if request.strategy in {"grid", "random"} and request.candidates:
            raise OrganelleContractError(
                code="experiment.candidates_not_allowed",
                message="automatic optimization strategies generate their own candidates",
                details={"strategy": request.strategy},
            )
        candidates = request.candidates
        if request.strategy == "grid":
            candidates = grid_candidates(contract)
        elif request.strategy == "random":
            candidates = random_candidates(contract)
        return ExperimentRequest.model_validate(
            request.model_dump(
                mode="json",
                exclude={"candidates", "contract_digest"},
            )
            | {
                "candidates": candidates,
                "contract_digest": contract.digest,
            }
        )

    def submit(self, request: ExperimentRequest) -> ExperimentRecord:
        with self._lock:
            if self._closed:
                raise OrganelleContractError(
                    code="experiment.service_closed",
                    message="the experiment service is closed",
                )
            index = self._index_provider()
            request = self._materialize_request(index, request)
            _bundle, optimization = self._validate(index, request)
            registry = OperationRegistry(
                capability_source=index.binding_source(trust_store=self._trust_store)
            )
            record = ExperimentRecord(
                experiment_id=f"exp-{uuid4().hex}",
                capability_id=request.capability_id,
                status="queued",
                submitted_at=datetime.now(UTC),
                request=request,
                trials=tuple(
                    ExperimentTrial(index=index, parameters=dict(candidate), status="queued")
                    for index, candidate in enumerate(request.candidates)
                ),
            )
            self._store.create(record)
            self._coordinator.submit(self._execute, record.experiment_id, registry, optimization)
            return record

    def _execute(
        self,
        experiment_id: str,
        registry: OperationRegistry,
        optimization: PluginOptimization,
    ) -> None:
        trial_executor = ThreadPoolExecutor(
            max_workers=optimization.parallelism,
            thread_name_prefix=f"ov-experiment-{experiment_id}",
        )
        with self._lock:
            self._trial_executors.add(trial_executor)
        try:
            record = self._store.get(experiment_id)
            record = self._store.update(record.model_copy(update={"status": "running"}))
            if record.status in {"succeeded", "failed"}:
                return
            futures = [
                trial_executor.submit(self._run_trial, experiment_id, registry, index)
                for index in range(len(record.trials))
            ]
            for future in futures:
                future.result()
            record = self._store.get(experiment_id)
            if record.status in {"succeeded", "failed"}:
                return
            successful = [trial for trial in record.trials if trial.status == "succeeded"]
            if successful:
                winner = min(
                    successful,
                    key=lambda trial: (
                        (
                            -cast(float, trial.score)
                            if optimization.direction == "maximize"
                            else cast(float, trial.score)
                        ),
                        trial.index,
                    ),
                )
                update: dict[str, object] = {
                    "status": "succeeded",
                    "best_trial_index": winner.index,
                    "completed_at": datetime.now(UTC),
                }
            else:
                update = {
                    "status": "failed",
                    "best_trial_index": None,
                    "completed_at": datetime.now(UTC),
                }
            self._store.update(record.model_copy(update=update))
        finally:
            with self._lock:
                self._trial_executors.discard(trial_executor)
            trial_executor.shutdown(wait=False, cancel_futures=True)

    def _run_trial(self, experiment_id: str, registry: OperationRegistry, index: int) -> None:
        record = self._store.get(experiment_id)
        trial = record.trials[index]
        self._persist_trial(experiment_id, trial.model_copy(update={"status": "running"}))
        request = record.request
        parameters = {
            **request.inputs,
            **request.fixed_parameters,
            **trial.parameters,
        }
        try:
            result = registry.require(request.capability_id).invoke(None, parameters)
        except OrganelleError as error:
            details = error.details
            self._persist_trial(
                experiment_id,
                trial.model_copy(
                    update={
                        "status": "failed",
                        "error": ErrorDetail.model_validate(
                            {
                                "code": error.code,
                                "message": error.message,
                                "details": dict(details) if isinstance(details, Mapping) else {},
                            }
                        ),
                    }
                ),
            )
            return
        except Exception as error:
            self._persist_trial(
                experiment_id,
                trial.model_copy(
                    update={
                        "status": "failed",
                        "error": ErrorDetail.model_validate(
                            {
                                "code": "experiment.trial_crashed",
                                "message": "a trial raised an unexpected exception",
                                "details": {"exception_type": type(error).__name__},
                            }
                        ),
                    }
                ),
            )
            return
        if not isinstance(result, OrganelleResult):
            self._persist_trial(
                experiment_id,
                trial.model_copy(
                    update={
                        "status": "failed",
                        "error": ErrorDetail(
                            code="experiment.result_invalid",
                            message="a plugin trial did not return an OrganelleResult",
                        ),
                    }
                ),
            )
            return
        if result.status in {"ok", "warning"}:
            score = result.metrics.get(PLUGIN_OPTIMIZATION_SCORE_METRIC)
            if (
                score is None
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                self._persist_trial(
                    experiment_id,
                    trial.model_copy(
                        update={
                            "status": "failed",
                            "result": result,
                            "error": ErrorDetail(
                                code="capability.plugin_score_missing",
                                message=(
                                    "a successful optimization trial carries no finite "
                                    "declared score"
                                ),
                            ),
                        }
                    ),
                )
                return
            self._persist_trial(
                experiment_id,
                trial.model_copy(
                    update={"status": "succeeded", "result": result, "score": float(score)}
                ),
            )
            return
        self._persist_trial(
            experiment_id,
            trial.model_copy(
                update={
                    "status": "failed",
                    "result": result,
                    "error": result.errors[0] if result.errors else None,
                }
            ),
        )

    def _persist_trial(self, experiment_id: str, trial: ExperimentTrial) -> None:
        self._store.update_trial(experiment_id, trial)

    # -- reads and lifecycle --------------------------------------------------

    def get(self, experiment_id: str) -> ExperimentRecord:
        return self._store.get(experiment_id)

    def contains(self, experiment_id: str) -> bool:
        """Return whether exact durable history contains this experiment."""
        try:
            self.get(experiment_id)
        except OrganelleInputError as error:
            if error.code == "experiment.unknown_experiment":
                return False
            raise
        return True

    def verifies(self, experiment_id: str) -> bool:
        """Return whether durable history contains this terminal experiment."""
        try:
            return self.get(experiment_id).status in _TERMINAL
        except OrganelleInputError as error:
            if error.code == "experiment.unknown_experiment":
                return False
            raise

    def list(self) -> tuple[ExperimentRecord, ...]:
        return self._store.list()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            trial_executors = tuple(self._trial_executors)
            self._trial_executors.clear()
            self._coordinator.shutdown(wait=False, cancel_futures=True)
            for executor in trial_executors:
                executor.shutdown(wait=False, cancel_futures=True)
            self._store.interrupt_nonterminal()
