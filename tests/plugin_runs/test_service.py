"""One durable invocation path shared by GUI and Agent plugin runs."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import cast

import pytest

from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.operations import BoundOperation
from organelleverse.plugin_runs import PluginRunRequest, PluginRunService, PluginRunStore
from organelleverse.runtime import managed_run_path
from tests.plugin_experiments.conftest import TRIAL_PLUGIN, PluginEnvironment


@pytest.fixture
def plugin_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PluginEnvironment:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    return PluginEnvironment(tmp_path)


@pytest.fixture
def run_service(plugin_env: PluginEnvironment, tmp_path: Path) -> PluginRunService:
    return PluginRunService(
        index_provider=lambda: plugin_env.index,
        trust_store=plugin_env.trust_store,
        store=PluginRunStore(tmp_path / "runs"),
    )


def _request(plugin_env: PluginEnvironment) -> PluginRunRequest:
    return PluginRunRequest(
        capability_id="demo.experiment",
        inputs={"images": str(plugin_env.images)},
        parameters={"threshold": 0.25},
    )


def _wait(service: PluginRunService, run_id: str) -> object:
    for _ in range(200):
        record = service.get(run_id)
        if record.status in {"succeeded", "failed"}:
            return record
        time.sleep(0.02)
    raise AssertionError(f"run did not complete: {run_id}")


def test_gui_and_agent_use_one_run_service(
    plugin_env: PluginEnvironment, run_service: PluginRunService
) -> None:
    request = _request(plugin_env)
    gui = _wait(run_service, run_service.submit(request).run_id)
    agent = run_service.execute(request)

    assert gui.status == agent.status == "succeeded"  # type: ignore[attr-defined]
    assert gui.contract_identity == agent.contract_identity == plugin_env.entry.content_hash  # type: ignore[attr-defined]
    assert gui.l6_run_id is not None and agent.l6_run_id is not None  # type: ignore[attr-defined]
    assert gui.l6_run_id != gui.run_id and agent.l6_run_id != agent.run_id  # type: ignore[attr-defined]
    assert gui.result is not None and agent.result is not None  # type: ignore[attr-defined]
    assert gui.result.operation_id == agent.result.operation_id  # type: ignore[attr-defined]
    assert gui.l6_run_id == gui.result.provenance.run_id  # type: ignore[union-attr]
    assert agent.l6_run_id == agent.result.provenance.run_id  # type: ignore[union-attr]
    assert managed_run_path(gui.result.operation_id, gui.l6_run_id).is_dir()  # type: ignore[union-attr,arg-type]
    assert managed_run_path(agent.result.operation_id, agent.l6_run_id).is_dir()  # type: ignore[union-attr,arg-type]
    assert run_service.list() == (gui, agent)


@pytest.mark.parametrize(
    "run_request",
    [
        PluginRunRequest(
            capability_id="demo.experiment",
            inputs={},
            parameters={"images": "/tmp/images", "threshold": 0.25},
        ),
        PluginRunRequest(
            capability_id="demo.experiment",
            inputs={"images": "/tmp/images", "threshold": "0.25"},
            parameters={},
        ),
    ],
)
def test_request_field_direction_must_match_admitted_descriptor(
    run_service: PluginRunService, run_request: PluginRunRequest
) -> None:
    with pytest.raises(OrganelleInputError) as captured:
        run_service.execute(run_request)

    assert captured.value.code == "input.plugin_fields_invalid"


def test_terminal_context_is_browser_safe(
    plugin_env: PluginEnvironment, run_service: PluginRunService
) -> None:
    completed = run_service.execute(_request(plugin_env))
    context = run_service.context(completed.run_id)
    rendered = context.model_dump_json()

    assert context.result_id == completed.result.object_id  # type: ignore[union-attr]
    assert context.contract_identity == plugin_env.entry.content_hash
    assert context.artifacts[0].object_id == completed.result.artifacts[0].object_id  # type: ignore[union-attr]
    assert context.artifacts[0].artifact_id == "artifact-0000"
    assert "uri" not in rendered
    assert '"sha256":' not in rendered
    assert context.request.inputs == {"images": str(plugin_env.images)}
    assert "argv" not in rendered


def test_context_projects_paths_from_real_plugin_results_but_keeps_request_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = TRIAL_PLUGIN.replace(
        'summary={"threshold": threshold},',
        "summary={\n"
        '            "threshold": threshold,\n'
        '            "work_dir": str(context.work_dir),\n'
        '            "managed_reference": "managed://runs/demo/output",\n'
        '            "file_reference": "file:///tmp/worker-payload.json",\n'
        '            "nested": {\n'
        '                "argv": ["python", str(context.log_path)],\n'
        '                "worker_payload": {"path": str(context.work_dir)},\n'
        '                "staging_root": "private",\n'
        '                "plugin_root": "private",\n'
        '                "sample": "chloroplast",\n'
        "            },\n"
        "        },",
    ).replace(
        'raise RuntimeError("synthetic trial failure")',
        'raise RuntimeError("worker failed at " + str(context.work_dir) + " after staging")',
    )
    environment = PluginEnvironment(tmp_path, source=source)
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    service = PluginRunService(
        index_provider=lambda: environment.index,
        trust_store=environment.trust_store,
        store=PluginRunStore(tmp_path / "runs"),
    )
    request = _request(environment)

    succeeded = service.context(service.execute(request).run_id)
    failed = service.context(
        service.execute(
            request.model_copy(update={"parameters": {"threshold": 0.25, "fail_at": 0.25}})
        ).run_id
    )

    assert succeeded.request.inputs == {"images": str(environment.images)}
    assert succeeded.metrics["nested"] == {"sample": "chloroplast"}
    assert str(tmp_path) not in str(succeeded.metrics)
    assert "managed://" not in str(succeeded.metrics)
    assert "file://" not in str(succeeded.metrics)
    assert "argv" not in str(succeeded.metrics)
    assert "worker_payload" not in str(succeeded.metrics)
    assert "staging_root" not in str(succeeded.metrics)
    assert "plugin_root" not in str(succeeded.metrics)
    assert str(tmp_path) not in succeeded.summary_text
    assert failed.status == "failed"
    assert str(tmp_path) not in str(failed.diagnostics)
    assert failed.diagnostics[0].details == {"exception_type": "RuntimeError"}


def test_legacy_record_has_no_inferred_run_context(tmp_path: Path) -> None:
    import json

    from tests.plugin_runs.test_store import _legacy_payload

    root = tmp_path / "runs"
    root.mkdir()
    (root / "legacy-run.json").write_text(
        json.dumps(_legacy_payload(status="failed")) + "\n", encoding="utf-8"
    )
    environment = PluginEnvironment(tmp_path / "environment")
    service = PluginRunService(
        index_provider=lambda: environment.index,
        trust_store=environment.trust_store,
        store=PluginRunStore(root),
    )

    with pytest.raises(OrganelleInputError) as captured:
        service.context("legacy-run")

    assert captured.value.code == "run.context_unavailable"


def test_export_artifact_resolution_and_verification_are_exact(
    plugin_env: PluginEnvironment, run_service: PluginRunService, tmp_path: Path
) -> None:
    completed = run_service.execute(_request(plugin_env))
    artifact_id = completed.artifacts[0].artifact_id

    assert run_service.verifies(completed.run_id)
    assert not run_service.verifies("missing")
    assert run_service.artifact_path(completed.run_id, artifact_id).is_file()
    exported = run_service.export(completed.run_id, tmp_path / "exported")
    assert exported.result is not None
    assert all(Path(item.uri).is_relative_to(tmp_path / "exported") for item in exported.result.artifacts)


def test_terminal_result_round_trips_across_a_store_restart(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    root = tmp_path / "runs"
    service = PluginRunService(
        index_provider=lambda: plugin_env.index,
        trust_store=plugin_env.trust_store,
        store=PluginRunStore(root),
    )
    completed = service.execute(_request(plugin_env))

    assert PluginRunStore(root).get(completed.run_id) == completed


def test_unexpected_implementation_exception_is_recorded_once(
    plugin_env: PluginEnvironment, run_service: PluginRunService
) -> None:
    original_prepare = run_service._prepare
    calls = 0

    class CrashingOperation:
        def invoke(self, _input: object, _parameters: object) -> object:
            nonlocal calls
            calls += 1
            raise RuntimeError("unexpected implementation crash")

    def crashing_prepare(request: PluginRunRequest):  # type: ignore[no-untyped-def]
        record, _ = original_prepare(request)
        return record, cast(BoundOperation, CrashingOperation())

    run_service._prepare = crashing_prepare  # type: ignore[method-assign]

    failed = run_service.execute(_request(plugin_env))

    assert calls == 1
    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.error_code == "plugin_run.execution_failed"


def test_close_terminalizes_an_active_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = PluginEnvironment(tmp_path, parallelism=1)
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    service = PluginRunService(
        index_provider=lambda: environment.index,
        trust_store=environment.trust_store,
        store=PluginRunStore(tmp_path / "runs"),
    )
    accepted = service.submit(
        PluginRunRequest(
            capability_id="demo.experiment",
            inputs={"images": str(environment.images)},
            parameters={"threshold": 0.25, "probe_log": str(tmp_path / "probe.log")},
        )
    )
    for _ in range(200):
        if service.get(accepted.run_id).status == "running":
            break
        time.sleep(0.01)

    assert not service.verifies(accepted.run_id)
    with pytest.raises(OrganelleInputError) as captured:
        service.context(accepted.run_id)
    assert captured.value.code == "input.run_not_terminal"

    service.close()
    closed = service.get(accepted.run_id)

    assert closed.status == "failed"
    assert closed.error is not None
    assert closed.error.error_code == "plugin_run.run_interrupted"
    time.sleep(0.6)


def test_submit_and_close_have_one_atomic_acceptance_boundary(
    plugin_env: PluginEnvironment, run_service: PluginRunService
) -> None:
    prepared = threading.Event()
    release = threading.Event()
    original_prepare = run_service._prepare

    def paused_prepare(request: PluginRunRequest):  # type: ignore[no-untyped-def]
        prepared.set()
        assert release.wait(timeout=5)
        return original_prepare(request)

    run_service._prepare = paused_prepare  # type: ignore[method-assign]
    submitted: list[object] = []
    submitter = threading.Thread(target=lambda: submitted.append(run_service.submit(_request(plugin_env))))
    submitter.start()
    assert prepared.wait(timeout=5)
    closer = threading.Thread(target=run_service.close)
    closer.start()
    release.set()
    submitter.join(timeout=5)
    closer.join(timeout=5)

    assert len(submitted) == 1
    accepted = submitted[0]
    assert run_service.get(accepted.run_id).status in {"succeeded", "failed"}  # type: ignore[attr-defined]
    with pytest.raises(OrganelleContractError) as captured:
        run_service.submit(_request(plugin_env))
    assert captured.value.code == "plugin_run.service_closed"
    time.sleep(0.6)
