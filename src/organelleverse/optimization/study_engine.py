"""Bounded evaluator-backed adaptive optimization orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.plugin_experiments.adaptive_models import (
    AdaptiveStudyRecord,
    AdoptionDecision,
    CandidateAggregate,
    ExecutionSelection,
    FinalExecutionClaim,
    FinalExecutionLink,
)
from organelleverse.plugin_experiments.store import ExperimentStore

from .evaluator_dispatch import evaluate_candidate
from .evidence import decide_adoption, pareto_front
from .strategies import candidate_digest
from .strategy_models import (
    AdaptiveStudyRequest,
    CandidateProposal,
)
from .strategy_registry import StrategyRegistry
from .study_models import AdoptedStudyIdentity, EvaluationRequest, EvaluatorRunner
from .study_search import run_search

TerminalStatus = Literal["adopted", "baseline_retained", "failed"]


class AdaptiveStudyEngine:
    def __init__(
        self,
        *,
        strategy_registry: StrategyRegistry,
        evaluator_runner: EvaluatorRunner,
        store: ExperimentStore,
    ) -> None:
        self._strategies = strategy_registry
        self._runner = evaluator_runner
        self._store = store

    def submit(self, request: AdaptiveStudyRequest) -> AdaptiveStudyRecord:
        self._require_evaluator(request)
        record = AdaptiveStudyRecord(
            study_id=request.study_id,
            status="created",
            submitted_at=datetime.now(UTC),
            contract_digest=request.contract.digest,
            evaluator_digest=request.evaluator_binding.evaluator_digest,
            evaluator_identity=request.evaluator_binding.identity,
            request=request.model_dump(mode="json"),
        )
        return self._store.create_study(record)

    def run_to_terminal(self, request: AdaptiveStudyRequest) -> AdaptiveStudyRecord:
        self._require_evaluator(request)
        try:
            record = self.get(request.study_id)
            self._require_same_request(record, request)
        except OrganelleInputError as error:
            if error.code != "optimization.unknown_study":
                raise
            record = self.submit(request)
        if record.status in {"adopted", "baseline_retained", "failed"}:
            return record
        try:
            return self._run(record, request)
        except OrganelleContractError as error:
            current = self.get(request.study_id)
            return self._terminate(current, "failed", error.code)

    def get(self, study_id: str) -> AdaptiveStudyRecord:
        return self._store.get_study(study_id)

    def find_adopted(self, identity: AdoptedStudyIdentity) -> AdaptiveStudyRecord | None:
        matches: list[AdaptiveStudyRecord] = []
        for record in self._store.list_studies():
            if record.status != "adopted":
                continue
            try:
                request = AdaptiveStudyRequest.model_validate(record.request)
            except ValueError as error:
                raise OrganelleContractError(
                    code="optimization.adopted_record_invalid",
                    message="an adopted study does not contain a valid reusable request",
                    details={"study_id": record.study_id},
                ) from error
            if AdoptedStudyIdentity.from_request(request) == identity:
                matches.append(record)
        if len(matches) > 1:
            raise _error(
                "optimization.adopted_identity_conflict",
                "multiple adopted studies have the same exact reusable identity",
            )
        return matches[0] if matches else None

    def record_final_execution(
        self, study_id: str, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord:
        return self._store.record_final_execution(study_id, link)

    def claim_final_execution(
        self, study_id: str, claim: FinalExecutionClaim
    ) -> AdaptiveStudyRecord:
        return self._store.claim_final_execution(study_id, claim)

    def complete_final_execution(
        self, study_id: str, claim_id: str, link: FinalExecutionLink
    ) -> AdaptiveStudyRecord:
        return self._store.complete_final_execution(study_id, claim_id, link)

    def _run(
        self, record: AdaptiveStudyRecord, request: AdaptiveStudyRequest
    ) -> AdaptiveStudyRecord:
        contract = request.contract
        baseline = CandidateProposal(
            parameters=contract.baseline.parameters,
            candidate_digest=candidate_digest(contract.baseline.parameters),
        )
        record = self._transition(record, "baseline_search")
        baseline_search, record = self._evaluate_candidate(
            record, request, baseline, "search", contract.repeats.search_repeats
        )
        if not baseline_search.feasible:
            reason = _failure_reason(
                (baseline_search,), default="optimization.validation_incomplete"
            )
            return self._terminate(record, "failed", reason)

        record = self._transition(record, "searching")
        search_aggregates, proposal_map, record = run_search(
            record=record,
            request=request,
            baseline=baseline,
            strategies=self._strategies,
            runner=self._runner,
            store=self._store,
        )
        front = pareto_front(
            search_aggregates,
            objectives=contract.objectives,
            constraints=contract.constraints,
        )
        if not front:
            return self._terminate(
                record,
                "failed",
                _failure_reason(
                    search_aggregates,
                    default="optimization.no_feasible_candidate",
                ),
            )
        # Ordinary search evaluates baseline and candidates at the same fidelity.
        # Halving finalists have a different resource rung from baseline_search;
        # comparing them here would confound parameters with compute allocation.
        if request.successive_halving is None:
            front = pareto_front(
                (baseline_search, *search_aggregates),
                objectives=contract.objectives,
                constraints=contract.constraints,
            )
        finalists = self._ordered_finalists(
            tuple(item for item in front if item.candidate_digest != baseline.candidate_digest),
            request,
        )[: request.finalist_limit]
        record = self._transition(record, "shortlist")
        record = record.model_copy(
            update={"pareto_candidate_digests": tuple(item.candidate_digest for item in front)}
        )
        record = self._store.update_study(record)

        record = self._transition(record, "baseline_validation")
        validation_baseline = baseline
        if request.successive_halving is not None:
            full_rung = request.successive_halving.rungs[-1]
            full_parameters = {
                **baseline.parameters,
                request.successive_halving.resource_parameter: full_rung,
            }
            validation_baseline = CandidateProposal(
                parameters=full_parameters,
                candidate_digest=candidate_digest(full_parameters),
                base_candidate_digest=baseline.candidate_digest,
                resource_rung=full_rung,
            )
        baseline_validation, record = self._evaluate_candidate(
            record,
            request,
            validation_baseline,
            "validation",
            contract.repeats.validation_repeats,
        )
        record = self._transition(record, "validating")
        validated: list[CandidateAggregate] = []
        validation_proposals: dict[str, CandidateProposal] = {}
        for finalist in finalists:
            proposal = proposal_map.get(finalist.candidate_digest)
            if proposal is None:
                raise _error("optimization.strategy_nondeterministic", "finalist proposal was lost")
            proposal = self._validation_proposal(proposal, request)
            validation_proposals[proposal.candidate_digest] = proposal
            aggregate, record = self._evaluate_candidate(
                record,
                request,
                proposal,
                "validation",
                contract.repeats.validation_repeats,
            )
            validated.append(aggregate)
        decision = decide_adoption(
            baseline_validation,
            tuple(validated),
            objectives=contract.objectives,
            constraints=contract.constraints,
            adoption=contract.adoption,
            objective_order=request.objective_order,
            validation_split=contract.benchmark.validation,
            evaluator_binding=request.evaluator_binding,
        )
        if not finalists and baseline_validation.feasible:
            decision = AdoptionDecision(
                status="baseline_retained",
                reason_code="optimization.baseline_dominates_search",
            )
        selection: ExecutionSelection | None = None
        if decision.status == "adopted" and decision.parameter_digest is not None:
            proposal = validation_proposals.get(decision.parameter_digest)
            if proposal is None:
                raise _error(
                    "optimization.adopted_record_invalid",
                    "adopted decision does not identify an evaluated proposal",
                )
            selection = ExecutionSelection(
                source="adopted",
                parameters=proposal.parameters,
                parameter_digest=decision.parameter_digest,
            )
        elif decision.status == "baseline_retained" and baseline_validation.feasible:
            selection = ExecutionSelection(
                source="verified_baseline",
                parameters=validation_baseline.parameters,
                parameter_digest=validation_baseline.candidate_digest,
            )
        elif decision.status == "baseline_retained":
            decision = decision.model_copy(update={"status": "failed"})
        return self._terminate(
            record,
            decision.status,
            decision.reason_code,
            decision,
            execution_selection=selection,
        )

    @staticmethod
    def _validation_proposal(
        proposal: CandidateProposal, request: AdaptiveStudyRequest
    ) -> CandidateProposal:
        config = request.successive_halving
        if config is None:
            return proposal
        parameters = {**proposal.parameters, config.resource_parameter: config.rungs[-1]}
        base_parameters = {
            name: value for name, value in parameters.items() if name != config.resource_parameter
        }
        return CandidateProposal(
            parameters=parameters,
            candidate_digest=candidate_digest(parameters),
            base_candidate_digest=candidate_digest(base_parameters),
            resource_rung=config.rungs[-1],
        )

    def _evaluate_candidate(
        self,
        record: AdaptiveStudyRecord,
        request: AdaptiveStudyRequest,
        proposal: CandidateProposal,
        split: Literal["search", "validation"],
        repeat_count: int,
    ) -> tuple[CandidateAggregate, AdaptiveStudyRecord]:
        return evaluate_candidate(
            record=record,
            request=request,
            proposal=proposal,
            split=split,
            repeat_count=repeat_count,
            runner=self._runner,
            store=self._store,
        )

    def _ordered_finalists(
        self, candidates: tuple[CandidateAggregate, ...], request: AdaptiveStudyRequest
    ) -> tuple[CandidateAggregate, ...]:
        objectives = {item.name: item for item in request.contract.objectives}

        def key(item: CandidateAggregate) -> tuple[float | str, ...]:
            values: list[float | str] = []
            for name in request.objective_order:
                objective = objectives[name]
                value = item.metrics[objective.metric_pointer].value
                values.append(-value if objective.direction == "maximize" else value)
            values.append(item.candidate_digest)
            return tuple(values)

        return tuple(sorted(candidates, key=key))

    def _require_evaluator(self, request: AdaptiveStudyRequest) -> None:
        if self._runner.binding != request.evaluator_binding:
            raise _error("optimization.evaluator_mismatch", "evaluator runner binding mismatch")

    @staticmethod
    def _require_same_request(record: AdaptiveStudyRecord, request: AdaptiveStudyRequest) -> None:
        if (
            record.contract_digest != request.contract.digest
            or record.evaluator_digest != request.evaluator_binding.evaluator_digest
            or record.evaluator_identity != request.evaluator_binding.identity
            or record.request != request.model_dump(mode="json")
        ):
            raise _error("optimization.study_identity_mismatch", "study request is immutable")

    def _transition(self, record: AdaptiveStudyRecord, status: str) -> AdaptiveStudyRecord:
        order = {
            "created": 0,
            "baseline_search": 1,
            "searching": 2,
            "shortlist": 3,
            "baseline_validation": 4,
            "validating": 5,
        }
        if record.status == "interrupted" or order.get(record.status, -1) <= order[status]:
            return self._store.update_study(record.model_copy(update={"status": status}))
        return record

    def _terminate(
        self,
        record: AdaptiveStudyRecord,
        status: TerminalStatus,
        reason: str,
        decision: AdoptionDecision | None = None,
        *,
        execution_selection: ExecutionSelection | None = None,
    ) -> AdaptiveStudyRecord:
        terminal = decision or AdoptionDecision(status=status, reason_code=reason)
        adopted = terminal.parameter_digest if terminal.status == "adopted" else None
        return self._store.update_study(
            record.model_copy(
                update={
                    "status": terminal.status,
                    "decision": terminal,
                    "adopted_parameter_digest": adopted,
                    "execution_selection": execution_selection,
                    "completed_at": datetime.now(UTC),
                }
            )
        )


def _error(code: str, message: str) -> OrganelleContractError:
    return OrganelleContractError(code=code, message=message)


def _failure_reason(aggregates: tuple[CandidateAggregate, ...], *, default: str) -> str:
    preferred = (
        "optimization.evaluator_mismatch",
        "optimization.evaluator_failed",
        "optimization.metric_invalid",
        "optimization.resource_constraint_failed",
        "optimization.budget_exhausted",
    )
    codes = {code for aggregate in aggregates for code in aggregate.failure_codes}
    return next((code for code in preferred if code in codes), default)


__all__ = ["AdaptiveStudyEngine", "AdoptedStudyIdentity", "EvaluationRequest", "EvaluatorRunner"]
