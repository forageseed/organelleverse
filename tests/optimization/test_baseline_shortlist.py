"""Baseline participates in equal-fidelity search, before held-out evaluation."""

from pathlib import Path

import pytest

from organelleverse.optimization.contracts_v3 import ObjectiveV3
from organelleverse.optimization.strategies import candidate_digest
from tests.optimization.evaluator_fixtures import ACCURACY, MEMORY, FakeEvaluatorRunner
from tests.optimization.test_study_engine import _engine, _request


class WorseCandidateRunner(FakeEvaluatorRunner):
    tradeoff = False
    fail_baseline_validation = False
    minimize = False

    def _result(self, request):
        result = super()._result(request)
        baseline = request.parameters["iterations"] == 2
        value = 0.8 if baseline else 0.7
        if self.minimize:
            value = -value
        memory = 0.5 if self.tradeoff and not baseline else 1.0
        if self.fail_baseline_validation and baseline and request.identity.split == "validation":
            memory = 3.0
        metrics = {ACCURACY: value, MEMORY: memory}
        if self.tradeoff:
            metrics["/metrics/optimization_evaluation/cost"] = memory
        return result.model_copy(update={"metrics": metrics})


@pytest.mark.parametrize("minimize", [False, True])
def test_dominated_candidates_skip_validation_but_baseline_is_verified(tmp_path: Path, minimize):
    request, registry, _, store = _request(tmp_path)
    if minimize:
        objective = request.contract.objectives[0].model_copy(update={"direction": "minimize"})
        request = request.model_copy(
            update={"contract": request.contract.model_copy(update={"objectives": (objective,)})}
        )
    runner = WorseCandidateRunner(request.evaluator_binding)
    runner.minimize = minimize
    engine = _engine(registry, runner, store)
    result = engine.run_to_terminal(request)
    baseline = candidate_digest(request.contract.baseline.parameters)
    assert result.status == "baseline_retained"
    assert result.decision.reason_code == "optimization.baseline_dominates_search"
    assert result.pareto_candidate_digests == (baseline,)
    heldout = [item for item in result.attempts if item.identity.split == "validation"]
    assert len(heldout) == request.contract.repeats.validation_repeats
    assert all(item.identity.parameter_digest == baseline for item in heldout)
    assert result.execution_selection.source == "verified_baseline"
    calls = runner.call_count.value
    assert engine.run_to_terminal(request) == result
    assert runner.call_count.value == calls


def test_tradeoff_candidate_is_not_discarded_by_baseline(tmp_path: Path):
    request, registry, _, store = _request(tmp_path)
    objective = ObjectiveV3(
        name="cost",
        metric_pointer="/metrics/optimization_evaluation/cost",
        direction="minimize",
        minimum_improvement=0.01,
        improvement_mode="absolute",
        aggregation="mean",
    )
    request = request.model_copy(
        update={
            "contract": request.contract.model_copy(
                update={"objectives": (*request.contract.objectives, objective)}
            ),
            "objective_order": ("accuracy", "cost"),
        }
    )
    runner = WorseCandidateRunner(request.evaluator_binding)
    runner.tradeoff = True
    result = _engine(registry, runner, store).run_to_terminal(request)
    baseline = candidate_digest(request.contract.baseline.parameters)
    assert baseline in result.pareto_candidate_digests
    assert any(
        item.identity.split == "validation" and item.identity.parameter_digest != baseline
        for item in result.attempts
    )
    assert result.decision.reason_code == "optimization.insufficient_improvement"


def test_dominating_search_baseline_cannot_bypass_failed_validation(tmp_path: Path):
    request, registry, _, store = _request(tmp_path)
    runner = WorseCandidateRunner(request.evaluator_binding)
    runner.fail_baseline_validation = True
    result = _engine(registry, runner, store).run_to_terminal(request)
    assert result.status == "failed"
    assert result.execution_selection is None
