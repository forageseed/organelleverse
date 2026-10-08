"""Experiment service: explicit candidates, durable trials, deterministic winner."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from organelleverse.capabilities.index import CapabilityIndex, CapabilityStatus
from organelleverse.core.errors import OrganelleContractError, OrganelleError
from organelleverse.plugin_experiments import (
    ExperimentRecord,
    ExperimentRequest,
    ExperimentService,
    ExperimentStore,
    ExperimentTrial,
)
from tests.plugin_experiments.conftest import (
    TRIAL_PLUGIN,
    PluginEnvironment,
    admit,
    build_plugin_entry,
    wait_for_experiment,
)


def _service(env: PluginEnvironment, tmp_path: Path) -> ExperimentService:
    return ExperimentService(
        index_provider=lambda: env.index,
        trust_store=env.trust_store,
        store=ExperimentStore(tmp_path / "experiments"),
    )


def _request(images: Path, candidates: tuple[dict[str, object], ...], **overrides: object) -> ExperimentRequest:
    capability_id = overrides.pop("capability_id", "demo.experiment")
    return ExperimentRequest(
        capability_id=capability_id,  # type: ignore[arg-type]
        inputs={"images": str(images)},
        candidates=candidates,  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )


def test_explicit_candidates_run_in_order_and_highest_score_wins(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    record = service.submit(
        _request(
            plugin_env.images,
            ({"threshold": 0.25}, {"threshold": 0.50}, {"threshold": 0.75}),
        )
    )

    completed = wait_for_experiment(service, record.experiment_id)

    assert completed.status == "succeeded"
    assert [trial.index for trial in completed.trials] == [0, 1, 2]
    assert [trial.status for trial in completed.trials] == ["succeeded"] * 3
    assert [trial.score for trial in completed.trials] == [0.25, 0.5, 0.75]
    assert completed.best_trial_index == 2
    assert completed.completed_at is not None
    for trial in completed.trials:
        assert trial.result is not None
        assert trial.result.status == "ok"
        assert trial.result.metrics["plugin_optimization_score"] == trial.score


def test_trial_artifacts_are_the_same_l6_artifacts_as_direct_invocation(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    record = service.submit(_request(plugin_env.images, ({"threshold": 0.5},)))
    completed = wait_for_experiment(service, record.experiment_id)

    trial = completed.trials[0]
    assert trial.result is not None
    assert {artifact.kind for artifact in trial.result.artifacts} == {"masks"}


def test_equal_scores_select_the_lowest_candidate_index(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    record = service.submit(
        _request(plugin_env.images, ({"threshold": 0.5}, {"threshold": 0.5}))
    )
    completed = wait_for_experiment(service, record.experiment_id)
    assert completed.best_trial_index == 0


def test_unknown_capability_cannot_execute(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    with pytest.raises(OrganelleError) as captured:
        service.submit(_request(plugin_env.images, ({"threshold": 0.5},), capability_id="demo.missing"))
    assert captured.value.code == "input.unknown_capability"


def test_rejected_capability_cannot_execute(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    rejected = plugin_env.entry.model_copy(update={"status": CapabilityStatus.REJECTED})
    env_index = CapabilityIndex(entries=(rejected,))
    service = ExperimentService(
        index_provider=lambda: env_index,
        trust_store=plugin_env.trust_store,
        store=ExperimentStore(tmp_path / "experiments"),
    )
    with pytest.raises(OrganelleError) as captured:
        service.submit(_request(plugin_env.images, ({"threshold": 0.5},)))
    assert captured.value.code == "capability.not_admitted"


def test_conflicted_capability_cannot_execute(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    from organelleverse.capabilities.index import CapabilityConflict, CapabilityConflictMember

    other = build_plugin_entry(
        tmp_path / "bundle-b", source=TRIAL_PLUGIN + "\n# variant b\n"
    )
    conflict = CapabilityConflict(
        capability_id="demo.experiment",
        members=(
            CapabilityConflictMember(
                content_hash=plugin_env.entry.content_hash,
                origin_channel="local",
                source_path=str(tmp_path / "bundle" / "capability.toml"),
            ),
            CapabilityConflictMember(
                content_hash=other.content_hash,
                origin_channel="local",
                source_path=str(tmp_path / "bundle-b" / "capability.toml"),
            ),
        ),
    )
    conflicted = CapabilityIndex(
        entries=(plugin_env.entry, admit(other)), conflict_items=(conflict,)
    )
    service = ExperimentService(
        index_provider=lambda: conflicted,
        trust_store=plugin_env.trust_store,
        store=ExperimentStore(tmp_path / "experiments"),
    )
    with pytest.raises(OrganelleError):
        service.submit(_request(plugin_env.images, ({"threshold": 0.5},)))


def test_admitted_but_untrusted_capability_cannot_execute(tmp_path: Path) -> None:
    from organelleverse.capabilities.trust import TrustStore

    entry = admit(build_plugin_entry(tmp_path / "bundle"))
    service = ExperimentService(
        index_provider=lambda: CapabilityIndex(entries=(entry,)),
        trust_store=TrustStore(tmp_path / "trust.json"),  # empty: never trusted
        store=ExperimentStore(tmp_path / "experiments"),
    )
    images = tmp_path / "images"
    images.mkdir()
    with pytest.raises(OrganelleContractError) as captured:
        service.submit(_request(images, ({"threshold": 0.5},)))
    assert captured.value.code == "capability.untrusted"


def test_plugin_without_optimization_declaration_is_refused(tmp_path: Path) -> None:
    env = PluginEnvironment(tmp_path, optimization=False)
    service = _service(env, tmp_path)
    with pytest.raises(OrganelleContractError) as captured:
        service.submit(_request(env.images, ({"threshold": 0.5},)))
    assert captured.value.code == "capability.optimization_not_declared"


def test_candidate_count_above_max_trials_fails_before_any_trial(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    with pytest.raises(OrganelleContractError) as captured:
        service.submit(
            _request(
                plugin_env.images,
                (
                    {"threshold": 0.1},
                    {"threshold": 0.2},
                    {"threshold": 0.3},
                    {"threshold": 0.4},
                ),
            )
        )
    assert captured.value.code == "experiment.max_trials_exceeded"
    assert service.list() == ()


def test_candidate_must_contain_every_and_only_declared_parameters(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    with pytest.raises(OrganelleContractError) as missing:
        service.submit(_request(plugin_env.images, ({},)))
    assert missing.value.code == "experiment.candidate_invalid"
    with pytest.raises(OrganelleContractError) as extra:
        service.submit(_request(plugin_env.images, ({"threshold": 0.5, "probe_log": "x"},)))
    assert extra.value.code == "experiment.candidate_invalid"


def test_fixed_parameters_cannot_contain_an_optimized_or_path_parameter(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    with pytest.raises(OrganelleContractError) as optimized:
        service.submit(
            _request(
                plugin_env.images,
                ({"threshold": 0.5},),
                fixed_parameters={"threshold": 0.1},
            )
        )
    assert optimized.value.code == "experiment.fixed_parameter_invalid"
    with pytest.raises(OrganelleContractError) as path_param:
        service.submit(
            _request(
                plugin_env.images,
                ({"threshold": 0.5},),
                fixed_parameters={"masks_dir": "/tmp/out"},
            )
        )
    assert path_param.value.code == "experiment.fixed_parameter_invalid"


def test_candidate_values_outside_the_declared_schema_fail_before_invocation(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    with pytest.raises(OrganelleContractError) as captured:
        service.submit(_request(plugin_env.images, ({"threshold": 1.5},)))
    assert captured.value.code == "experiment.candidate_invalid"
    assert service.list() == ()


def test_one_failed_trial_remains_failed_while_others_continue(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    record = service.submit(
        _request(
            plugin_env.images,
            ({"threshold": 0.25}, {"threshold": 0.5}, {"threshold": 0.75}),
            fixed_parameters={"fail_at": 0.5},
        )
    )

    completed = wait_for_experiment(service, record.experiment_id)

    assert completed.status == "succeeded"
    statuses = [trial.status for trial in completed.trials]
    assert statuses == ["succeeded", "failed", "succeeded"]
    failed = completed.trials[1]
    assert failed.score is None
    assert failed.error is not None
    assert failed.error.code == "capability.plugin_execution_failed"
    assert failed.result is not None
    assert failed.result.status == "failed"
    assert completed.best_trial_index == 2


def test_all_failed_trials_make_the_experiment_failed(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    record = service.submit(
        _request(
            plugin_env.images,
            ({"threshold": 0.5},),
            fixed_parameters={"fail_at": 0.5},
        )
    )

    completed = wait_for_experiment(service, record.experiment_id)

    assert completed.status == "failed"
    assert completed.best_trial_index is None
    assert completed.trials[0].status == "failed"


def test_parallelism_never_exceeds_the_declared_bound(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    probe = tmp_path / "probe.log"
    service = _service(plugin_env, tmp_path)
    record = service.submit(
        _request(
            plugin_env.images,
            ({"threshold": 0.25}, {"threshold": 0.5}, {"threshold": 0.75}),
            fixed_parameters={"probe_log": str(probe)},
        )
    )
    completed = wait_for_experiment(service, record.experiment_id)
    assert completed.status == "succeeded"

    events: list[tuple[float, int]] = []
    for line in probe.read_text(encoding="utf-8").splitlines():
        kind, _, stamp = line.partition(" ")
        events.append((float(stamp), 1 if kind == "start" else -1))
    assert len(events) == 6
    # Replay the cross-process timeline (CLOCK_MONOTONIC is system-wide):
    # the peak depth must never exceed the declared parallelism bound of 2.
    depth = peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        depth += delta
        peak = max(peak, depth)
    assert peak <= 2


def test_submitted_history_is_durable_and_immutable(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    service = _service(plugin_env, tmp_path)
    record = service.submit(_request(plugin_env.images, ({"threshold": 0.5},)))
    completed = wait_for_experiment(service, record.experiment_id)

    reopened = ExperimentService(
        index_provider=lambda: plugin_env.index,
        trust_store=plugin_env.trust_store,
        store=ExperimentStore(tmp_path / "experiments"),
    )
    assert reopened.get(record.experiment_id) == completed
    assert [item.experiment_id for item in reopened.list()] == [record.experiment_id]


def test_close_shuts_down_without_retry(plugin_env: PluginEnvironment, tmp_path: Path) -> None:
    service = _service(plugin_env, tmp_path)
    service.close()
    with pytest.raises(OrganelleContractError) as captured:
        service.submit(_request(plugin_env.images, ({"threshold": 0.5},)))
    assert captured.value.code == "experiment.service_closed"


def test_parallel_trial_transitions_do_not_lose_a_terminal_outcome(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    class BarrierStore(ExperimentStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.barrier = threading.Barrier(2)

        def get(self, experiment_id: str):  # type: ignore[no-untyped-def,override]
            record = super().get(experiment_id)
            if threading.current_thread().name.startswith("persist-"):
                self.barrier.wait(timeout=5)
            return record

    store = BarrierStore(tmp_path / "experiments")
    service = ExperimentService(
        index_provider=lambda: plugin_env.index,
        trust_store=plugin_env.trust_store,
        store=store,
    )
    request = _request(plugin_env.images, ({"threshold": 0.25}, {"threshold": 0.75}))
    accepted = ExperimentRecord(
        experiment_id="exp-race",
        capability_id=request.capability_id,
        status="running",
        submitted_at=datetime.now(UTC),
        request=request,
        trials=(
            ExperimentTrial(index=0, parameters={"threshold": 0.25}, status="queued"),
            ExperimentTrial(index=1, parameters={"threshold": 0.75}, status="queued"),
        ),
    )
    store.create(accepted)
    updates = (
        ExperimentTrial(index=0, parameters={"threshold": 0.25}, status="succeeded", score=0.25),
        ExperimentTrial(index=1, parameters={"threshold": 0.75}, status="succeeded", score=0.75),
    )
    threads = [
        threading.Thread(  # pyright: ignore[reportPrivateUsage] - barrier probes the audited transition
            target=service._persist_trial,  # pyright: ignore[reportPrivateUsage]
            args=(accepted.experiment_id, trial),
            name=f"persist-{trial.index}",
        )
        for trial in updates
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert [trial.status for trial in store.get(accepted.experiment_id).trials] == ["succeeded", "succeeded"]


def test_close_terminalizes_accepted_trials_without_a_restart(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    probe = tmp_path / "probe.log"
    service = _service(plugin_env, tmp_path)
    accepted = service.submit(
        _request(
            plugin_env.images,
            ({"threshold": 0.25}, {"threshold": 0.5}, {"threshold": 0.75}),
            fixed_parameters={"probe_log": str(probe)},
        )
    )
    for _ in range(100):
        if any(trial.status == "running" for trial in service.get(accepted.experiment_id).trials):
            break
        time.sleep(0.02)
    else:
        raise AssertionError("an accepted trial did not start")

    service.close()
    closed = service.get(accepted.experiment_id)

    assert closed.status == "failed"
    assert all(trial.status in {"succeeded", "failed"} for trial in closed.trials)
    assert all(
        trial.status == "succeeded" or (trial.error is not None and trial.error.code == "experiment.run_interrupted")
        for trial in closed.trials
    )
    # Active workers cannot be cancelled. Let their cleanup finish before this
    # test's temporary managed-run root is removed.
    time.sleep(1)
