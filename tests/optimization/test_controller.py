"""Governed optimization materialization in the existing experiment service."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.capabilities.index import CapabilityIndex
from organelleverse.capabilities.models import PluginCapabilityBundle
from organelleverse.core.errors import OrganelleContractError
from organelleverse.optimization import contract_from_capability_entry, random_candidates
from organelleverse.plugin_experiments import (
    ExperimentRequest,
    ExperimentService,
    ExperimentStore,
)
from tests.plugin_experiments.conftest import PluginEnvironment, wait_for_experiment


def _service(env: PluginEnvironment, tmp_path: Path) -> ExperimentService:
    return ExperimentService(
        index_provider=lambda: env.index,
        trust_store=env.trust_store,
        store=ExperimentStore(tmp_path / "experiments"),
    )


def test_agent_request_requires_bounded_rationale_and_candidate_payload() -> None:
    common = {
        "capability_id": "demo.experiment",
        "inputs": {"images": "/data/images"},
        "strategy": "agent",
        "candidates": ({"threshold": 0.5},),
    }

    with pytest.raises(ValidationError):
        ExperimentRequest.model_validate(common)
    with pytest.raises(ValidationError):
        ExperimentRequest.model_validate({**common, "rationale": " "})
    with pytest.raises(ValidationError):
        ExperimentRequest.model_validate({**common, "rationale": "x" * 2001})
    with pytest.raises(ValidationError):
        ExperimentRequest.model_validate(
            {
                **common,
                "rationale": "bounded search",
                "candidates": ({"threshold": "x" * 70_000},),
            }
        )


def test_invalid_or_over_budget_agent_candidates_fail_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    plugin_env = PluginEnvironment(tmp_path)
    service = _service(plugin_env, tmp_path)
    with pytest.raises(OrganelleContractError) as invalid:
        service.submit(
            ExperimentRequest(
                capability_id="demo.experiment",
                inputs={"images": str(plugin_env.images)},
                strategy="agent",
                rationale="Try a boundary candidate.",
                candidates=({"threshold": 1.5},),
            )
        )
    assert invalid.value.code == "experiment.candidate_invalid"
    assert service.list() == ()

    with pytest.raises(OrganelleContractError) as over_budget:
        service.submit(
            ExperimentRequest(
                capability_id="demo.experiment",
                inputs={"images": str(plugin_env.images)},
                strategy="agent",
                rationale="Compare four candidates.",
                candidates=tuple({"threshold": value} for value in (0.1, 0.2, 0.3, 0.4)),
            )
        )
    assert over_budget.value.code == "experiment.max_trials_exceeded"
    assert service.list() == ()


def test_random_submission_persists_exact_governed_request_in_existing_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    environment = PluginEnvironment(tmp_path)
    service = _service(environment, tmp_path)
    expected_contract = contract_from_capability_entry(environment.entry, seed=41)

    submitted = service.submit(
        ExperimentRequest(
            capability_id="demo.experiment",
            inputs={"images": str(environment.images)},
            strategy="random",
            seed=41,
        )
    )
    persisted = service.get(submitted.experiment_id)

    assert persisted.request.contract_digest == expected_contract.digest
    assert persisted.request.strategy == "random"
    assert persisted.request.seed == 41
    assert persisted.request.rationale is None
    assert persisted.request.candidates == random_candidates(expected_contract)
    assert len(persisted.trials) == len(persisted.request.candidates) == 3


def test_agent_submission_persists_rationale_and_exact_candidate_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    environment = PluginEnvironment(tmp_path)
    service = _service(environment, tmp_path)
    candidates = ({"threshold": 0.8}, {"threshold": 0.2})

    submitted = service.submit(
        ExperimentRequest(
            capability_id="demo.experiment",
            inputs={"images": str(environment.images)},
            strategy="agent",
            seed=9,
            rationale="Probe the high and low portions of the closed domain.",
            candidates=candidates,
        )
    )

    assert submitted.request.rationale == ("Probe the high and low portions of the closed domain.")
    assert submitted.request.candidates == candidates
    assert [trial.parameters for trial in submitted.trials] == list(candidates)
    assert (
        submitted.request.contract_digest
        == contract_from_capability_entry(environment.entry, seed=9).digest
    )


def test_minimize_selects_smallest_original_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    environment = PluginEnvironment(tmp_path)
    payload = environment.entry.bundle.model_dump(mode="json")
    payload["contract"]["optimization"]["direction"] = "minimize"
    bundle = PluginCapabilityBundle.model_validate(payload)
    environment.entry = environment.entry.model_copy(update={"bundle": bundle})
    environment.index = CapabilityIndex(entries=(environment.entry,))
    service = _service(environment, tmp_path)

    submitted = service.submit(
        ExperimentRequest(
            capability_id="demo.experiment",
            inputs={"images": str(environment.images)},
            candidates=({"threshold": 0.8}, {"threshold": 0.2}),
        )
    )
    completed = wait_for_experiment(service, submitted.experiment_id)

    assert [trial.score for trial in completed.trials] == [0.8, 0.2]
    assert completed.best_trial_index == 1


def test_direction_is_projected_into_the_contract_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    environment = PluginEnvironment(tmp_path)
    maximizing = contract_from_capability_entry(environment.entry, seed=3)
    payload = environment.entry.bundle.model_dump(mode="json")
    payload["contract"]["optimization"]["direction"] = "minimize"
    minimizing_bundle = PluginCapabilityBundle.model_validate(payload)
    minimizing_entry = environment.entry.model_copy(update={"bundle": minimizing_bundle})

    minimizing = contract_from_capability_entry(minimizing_entry, seed=3)

    assert maximizing.objective.direction == "maximize"
    assert minimizing.objective.direction == "minimize"
    assert minimizing.digest != maximizing.digest
