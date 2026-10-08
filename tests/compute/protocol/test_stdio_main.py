from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

from organelleverse.compute.protocol import (
    PROVIDER_TOOLS,
    InvocationPolicyProjection,
    PreparePolicyProjection,
    PrepareToolRequest,
    PrepareToolResponse,
    ProbeToolRequest,
    ProbeToolResponse,
    RunToolRequest,
    RunToolResponse,
    SubmitToolRequest,
    SubmitToolResponse,
    main,
)
from organelleverse.compute.worker import DurableWorkerBackend, _package_version
from organelleverse.operations.registry import registry
from organelleverse.operations.spec import SideEffect

from .test_worker_execution import _digest, _wait_terminal


def _server_argv() -> tuple[str, ...]:
    return (
        sys.executable,
        "-c",
        "from organelleverse.compute.protocol import main; raise SystemExit(main(['mcp', 'stdio']))",
    )


def _server_env(run_root: Path) -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[3] / "src"
    return {
        "PYTHONPATH": str(source_root),
        "ORGANELLEVERSE_WORKER_RUN_ROOT": str(run_root),
    }


async def _open_session(run_root: Path, action):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    argv = _server_argv()
    parameters = StdioServerParameters(
        command=argv[0], args=list(argv[1:]), env=_server_env(run_root)
    )
    with tempfile.TemporaryFile(mode="w+") as diagnostics:
        async with stdio_client(parameters, errlog=diagnostics) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                return await action(session)


def _response(result, model):
    assert result.is_error is False
    assert len(result.content) == 1
    return model.model_validate(json.loads(result.content[0].text))


def test_main_rejects_every_argv_except_mcp_stdio():
    assert main([]) == 2
    assert main(["mcp"]) == 2
    assert main(["worker", "run-id"]) == 2


def test_real_stdio_server_lists_exact_six_tools(tmp_path: Path):
    async def list_tools(session):
        return await session.list_tools()

    listed = asyncio.run(_open_session(tmp_path / "worker", list_tools))

    assert [tool.name for tool in listed.tools] == [name for name, _ in PROVIDER_TOOLS]


def test_stdio_control_process_restart_recovers_detached_run(tmp_path: Path):
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    run_root = tmp_path / "worker"

    async def prepare_and_submit(session):
        probe_result = await session.call_tool(
            "ov.compute.probe", ProbeToolRequest().model_dump(mode="json")
        )
        worker_catalog_digest = _response(probe_result, ProbeToolResponse).worker_catalog_digest
        prepared_result = await session.call_tool(
            "ov.compute.prepare",
            PrepareToolRequest(
                operation_id="io.read_long_reads",
                capability_contract_digest=_digest(b"stdio contract"),
                software_selector=f"organelleverse=={_package_version()}",
                policy=PreparePolicyProjection(),
            ).model_dump(mode="json"),
        )
        prepared = _response(prepared_result, PrepareToolResponse).prepared_target
        submitted_result = await session.call_tool(
            "ov.compute.submit",
            SubmitToolRequest(
                operation_id="io.read_long_reads",
                catalog_digest=worker_catalog_digest,
                input_snapshot=None,
                parameters_snapshot={
                    "reads": str(reads),
                    "technology": "pacbio_hifi",
                    "quality_state": "ccs",
                },
                policy=InvocationPolicyProjection(
                    side_effect_grants=frozenset({SideEffect.READ_FILES})
                ),
                target_digest=prepared.target_digest,
                prepared_environment_digest=prepared.environment_digest,
            ).model_dump(mode="json"),
        )
        submitted = _response(submitted_result, SubmitToolResponse).run
        backend = DurableWorkerBackend(registry=registry, run_root=run_root)
        deadline = asyncio.get_running_loop().time() + 15
        while True:
            record = backend.read_persistent_record(submitted.provider_run_id)
            if record.worker_ready and record.pid is not None:
                os.kill(record.pid, signal.SIGSTOP)
                return submitted, record.pid
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)

    submitted, worker_pid = asyncio.run(_open_session(run_root, prepare_and_submit))
    assert submitted.status.status in {"queued", "running"}

    # The first stdio server has exited here while its detached child is paused
    # and therefore provably still nonterminal.  A second server recovers that
    # same live run before it is allowed to finish.
    restarted_backend = DurableWorkerBackend(registry=registry, run_root=run_root)

    async def get_run(session):
        result = await session.call_tool(
            "ov.compute.get",
            RunToolRequest(provider_run_id=submitted.provider_run_id).model_dump(mode="json"),
        )
        return _response(result, RunToolResponse).run

    running = asyncio.run(_open_session(run_root, get_run))
    assert running.provider_run_id == submitted.provider_run_id
    assert running.status.status == "running"

    os.kill(worker_pid, signal.SIGCONT)
    completed = _wait_terminal(restarted_backend, submitted.provider_run_id)
    recovered = asyncio.run(_open_session(run_root, get_run))
    assert recovered == completed
    assert recovered.provider_run_id == submitted.provider_run_id


def test_main_uses_configured_private_run_root(tmp_path: Path, monkeypatch):
    configured = tmp_path / "configured"
    monkeypatch.setenv("ORGANELLEVERSE_WORKER_RUN_ROOT", str(configured))
    backend = DurableWorkerBackend(registry=registry, run_root=configured)
    assert backend.run_root == configured
    assert os.stat(configured).st_mode & 0o777 == 0o700
