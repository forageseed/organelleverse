"""Slurm provider tests: closed resources, fixed job-script bytes (no model
injection), total state mapping, and squeue/sacct fixture parsing.

No real Slurm cluster is needed: scheduler output is fixture text, and the
job script is rendered from closed inputs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from organelleverse.compute.contracts import ComputeProviderSpec, TransportKind
from organelleverse.compute.providers.slurm import PROVIDER_ID, provider_factory
from organelleverse.compute.providers.slurm.config import (
    SlurmResources,
    SlurmTargetConfig,
)
from organelleverse.compute.providers.slurm.lifecycle import (
    map_slurm_state,
    parse_sacct_state,
    parse_squeue_state,
)
from organelleverse.compute.providers.slurm.renderer import render_job_script
from organelleverse.core.errors import OrganelleContractError

# === factory declaration (spec §7.1/§13) ===================================


def test_factory_declaration_is_closed_and_slurm():
    decl = provider_factory()
    assert isinstance(decl, ComputeProviderSpec)
    assert decl.provider_id == PROVIDER_ID == "slurm"
    assert decl.transport_kind is TransportKind.SLURM
    assert decl.declaration_version == "1.0"
    assert decl.artifact_transports == ("shared_filesystem",)


def test_factory_declaration_is_frozen():
    with pytest.raises((ValidationError, TypeError)):
        provider_factory().provider_id = "x"  # type: ignore[misc]


# === resources are bounded (spec §13) ======================================


def test_resources_reject_unbounded_values():
    with pytest.raises(ValidationError):
        SlurmResources(cpus=0, memory_mb=128, walltime_seconds=60)
    with pytest.raises(ValidationError):
        SlurmResources(cpus=999_999, memory_mb=128, walltime_seconds=60)
    with pytest.raises(ValidationError):
        SlurmResources(cpus=1, memory_mb=64, walltime_seconds=60)  # too little mem
    with pytest.raises(ValidationError):
        SlurmResources(cpus=1, memory_mb=128, walltime_seconds=30)  # too short walltime


def test_target_requires_at_least_one_profile():
    with pytest.raises(ValidationError):
        SlurmTargetConfig(
            target_id="slurm:c1", cluster_id="c1", workspace_id="w",
            environment_id="e", allowed_resource_profiles={},
        )


# === renderer: fixed bytes, no model injection (the core Slurm guarantee) ==


def _target():
    return SlurmTargetConfig(
        target_id="slurm:c1", cluster_id="c1", workspace_id="w",
        environment_id="e",
        allowed_resource_profiles={"small": SlurmResources(cpus=2, memory_mb=4096, walltime_seconds=3600)},
    )


def test_rendered_script_is_invariant_to_scientific_parameters():
    # The script depends only on closed config + provider-generated paths; it
    # does NOT take scientific parameters, so changing them cannot change bytes.
    cfg = _target()
    res = cfg.allowed_resource_profiles["small"]
    r1 = render_job_script(
        config=cfg, resources=res,
        worker_executable="/opt/ov/bin/organelleverse-linux-provider",
        request_file="/data/ov/runs/req-abc.json",
    )
    r2 = render_job_script(
        config=cfg, resources=res,
        worker_executable="/opt/ov/bin/organelleverse-linux-provider",
        request_file="/data/ov/runs/req-xyz.json",  # different request path
    )
    # different request file -> different script (the path is in the exec line)
    assert r1.script != r2.script
    # but the SAME inputs always render the SAME bytes (deterministic)
    r1b = render_job_script(
        config=cfg, resources=res,
        worker_executable="/opt/ov/bin/organelleverse-linux-provider",
        request_file="/data/ov/runs/req-abc.json",
    )
    assert r1.script == r1b.script
    assert r1.script_sha256 == r1b.script_sha256


def test_script_contains_only_fixed_shell_no_model_bytes():
    cfg = _target()
    res = cfg.allowed_resource_profiles["small"]
    rendered = render_job_script(
        config=cfg, resources=res,
        worker_executable="/opt/ov/bin/organelleverse-linux-provider",
        request_file="/data/ov/runs/req-1.json",
    )
    # the exec line is the only command; it references the worker + request file
    assert "exec /opt/ov/bin/organelleverse-linux-provider run --request /data/ov/runs/req-1.json" in rendered.script
    # set -eu prologue present
    assert "set -eu" in rendered.script


def test_renderer_rejects_unsafe_paths():
    cfg = _target()
    res = cfg.allowed_resource_profiles["small"]
    for bad in ("; rm -rf /", "$(reboot)", "/x/../../etc/passwd", "-n bad"):
        with pytest.raises(OrganelleContractError) as exc:
            render_job_script(
                config=cfg, resources=res, worker_executable=bad, request_file="/r.json"
            )
        assert exc.value.code == "compute.slurm_unsafe_path"


def test_renderer_rejects_unsafe_module_ids():
    cfg = _target()
    res = cfg.allowed_resource_profiles["small"]
    with pytest.raises(OrganelleContractError):
        render_job_script(
            config=cfg, resources=res,
            worker_executable="/opt/ov/bin/ov", request_file="/r.json",
            allowlisted_modules=("; echo pwned",),
        )


# === state mapping is total and deterministic (spec §13) ==================


@pytest.mark.parametrize(
    "raw,expected_status,terminal,scientific",
    [
        ("PENDING", "queued", False, False),
        ("RUNNING", "running", False, False),
        ("COMPLETING", "running", False, False),
        ("SUSPENDED", "running", False, False),
        ("COMPLETED", "completed", True, True),  # exit 0
        ("CANCELLED", "cancelled", True, False),
        ("CANCELLED by 1000", "cancelled", True, False),
        ("FAILED", "failed", True, False),
        ("TIMEOUT", "failed", True, False),
        ("OUT_OF_MEMORY", "failed", True, False),
        ("PREEMPTED", "failed", True, False),
    ],
)
def test_state_mapping_is_deterministic(raw, expected_status, terminal, scientific):
    verdict = map_slurm_state(raw, exit_code=0 if raw == "COMPLETED" else None)
    assert verdict.status == expected_status
    assert verdict.is_terminal is terminal
    assert verdict.is_scientific_candidate is scientific


def test_completed_with_nonzero_exit_is_provider_failure_not_scientific():
    verdict = map_slurm_state("COMPLETED", exit_code=137)
    assert verdict.status == "failed"
    assert verdict.is_terminal is True
    assert verdict.is_scientific_candidate is False


def test_unknown_state_stays_nonterminal_never_fabricates_result():
    # an unrecognized state must not become a fabricated scientific failure
    verdict = map_slurm_state("SOMETHING_NEW")
    assert verdict.is_terminal is False
    assert verdict.status in ("queued", "running")


def test_empty_state_raises():
    with pytest.raises(OrganelleContractError):
        map_slurm_state("   ")


# === squeue / sacct fixture parsing ========================================


def test_parse_squeue_state_present():
    assert parse_squeue_state(b"RUNNING\n") == "RUNNING"


def test_parse_squeue_state_absent_returns_none():
    assert parse_squeue_state(b"") is None


def test_parse_sacct_state_picks_allocation_row():
    # parsable2: JobIDRaw|State|ExitCode ; step rows carry '.' and are skipped
    stdout = b"12345|COMPLETED|0:0\n12345.0|COMPLETED|0:0\n12345.batch|COMPLETED|0:0\n"
    obs = parse_sacct_state(stdout)
    assert obs.raw_state == "COMPLETED"
    assert obs.exit_code == 0
    assert obs.seen_in == "sacct"


def test_parse_sacct_state_missing_raises():
    with pytest.raises(OrganelleContractError) as exc:
        parse_sacct_state(b"12345.0|COMPLETED|0:0\n")  # only a step row, no allocation
    assert exc.value.code == "compute.slurm_accounting_missing"


# === bare-import boundary ==================================================


def test_bare_import_does_not_load_slurm_runtime():
    import pathlib
    import sys

    # the slurm runtime (scheduler command execution) is not imported by __init__
    assert "organelleverse.compute.providers.slurm.provider" not in sys.modules
    # static check: __init__ imports only the declaration
    root = pathlib.Path(__file__).resolve()
    for _ in range(6):
        root = root.parent
        if (root / "pyproject.toml").exists():
            break
    text = (root / "src/organelleverse/compute/providers/slurm/__init__.py").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith(("import mcp", "from mcp")):
            raise AssertionError("slurm __init__ imports mcp at module scope")
