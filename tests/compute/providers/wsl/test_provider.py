from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

from organelleverse.compute.protocol import (
    InvocationPolicyProjection,
    PreparePolicyProjection,
    PrepareToolRequest,
    ProbeToolResponse,
    SubmitToolRequest,
)
from organelleverse.compute.providers.wsl import WslComputeProvider
from organelleverse.compute.providers.wsl.config import WslTargetConfig
from organelleverse.compute.worker import _package_version
from organelleverse.operations.spec import SideEffect


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


class _LocalLauncher:
    def __init__(self) -> None:
        self.configs: list[WslTargetConfig] = []

    def mcp_argv(self, config: WslTargetConfig) -> tuple[str, ...]:
        self.configs.append(config)
        return (
            sys.executable,
            "-c",
            "from organelleverse.compute.protocol import main; raise SystemExit(main(['mcp', 'stdio']))",
        )


def test_wsl_compute_provider_uses_launcher_and_official_stdio_session(tmp_path: Path):
    launcher = _LocalLauncher()
    config = WslTargetConfig(target_id="wsl:Test", distribution="Test")
    source_root = Path(__file__).resolve().parents[4] / "src"
    provider = WslComputeProvider(
        config,
        launcher=launcher,
        env={
            "PYTHONPATH": str(source_root),
            "ORGANELLEVERSE_WORKER_RUN_ROOT": str(tmp_path / "worker"),
        },
    )

    response = provider.probe()

    assert isinstance(response, ProbeToolResponse)
    assert response.protocol_version == "1.0"
    assert launcher.configs == [config]


def test_wsl_compute_provider_is_a_lazy_package_export():
    import organelleverse.compute.providers.wsl as package

    assert package.WslComputeProvider is WslComputeProvider


def test_wsl_compute_provider_calls_all_six_fixed_tools_across_sessions(tmp_path: Path):
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    source_root = Path(__file__).resolve().parents[4] / "src"
    provider = WslComputeProvider(
        WslTargetConfig(target_id="wsl:Test", distribution="Test"),
        launcher=_LocalLauncher(),
        env={
            "PYTHONPATH": str(source_root),
            "ORGANELLEVERSE_WORKER_RUN_ROOT": str(tmp_path / "worker"),
        },
    )
    probe = provider.probe()
    prepared = provider.prepare(
        PrepareToolRequest(
            operation_id="io.read_long_reads",
            capability_contract_digest=_digest(b"wsl client contract"),
            software_selector=f"organelleverse=={_package_version()}",
            policy=PreparePolicyProjection(),
        )
    ).prepared_target
    submitted = provider.submit(
        SubmitToolRequest(
            operation_id="io.read_long_reads",
            catalog_digest=probe.worker_catalog_digest,
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
        )
    ).run
    deadline = time.monotonic() + 15
    current = provider.get(submitted.provider_run_id).run
    while current.status.status not in {"completed", "failed", "cancelled"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)
        current = provider.get(submitted.provider_run_id).run

    assert current.status.status == "completed"
    assert provider.artifacts(submitted.provider_run_id).manifest.provider_run_id == (
        submitted.provider_run_id
    )
    assert provider.cancel(submitted.provider_run_id).run == current
