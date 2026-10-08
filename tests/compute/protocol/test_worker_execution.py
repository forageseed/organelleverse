from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest

from organelleverse.compute.protocol import (
    InvocationPolicyProjection,
    PreparePolicyProjection,
    PrepareToolRequest,
    RunToolRequest,
    SubmitToolRequest,
    operation_catalog_digest,
)
from organelleverse.compute.worker import (
    DurableWorkerBackend,
    _package_version,
    current_worker_compute_target_hint,
    invoke_admitted_operation,
)
from organelleverse.core.errors import OrganelleContractError, OrganelleExecutionError
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.operations.registry import OperationRegistry, registry
from organelleverse.operations.spec import (
    CoreKind,
    ExecutionMode,
    OperationSpec,
    OperationStage,
    SideEffect,
)


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _prepared(backend: DurableWorkerBackend):
    response = backend.prepare(
        PrepareToolRequest(
            operation_id="io.read_long_reads",
            capability_contract_digest=_digest(b"io.read_long_reads contract"),
            software_selector=f"organelleverse=={_package_version()}",
            policy=PreparePolicyProjection(),
        )
    )
    return response.prepared_target


def _request(backend: DurableWorkerBackend, reads: Path) -> SubmitToolRequest:
    prepared = _prepared(backend)
    return SubmitToolRequest(
        operation_id="io.read_long_reads",
        catalog_digest=operation_catalog_digest(registry),
        input_snapshot=None,
        parameters_snapshot={
            "reads": str(reads),
            "technology": "pacbio_hifi",
            "quality_state": "ccs",
        },
        policy=InvocationPolicyProjection(side_effect_grants=frozenset({SideEffect.READ_FILES})),
        target_digest=prepared.target_digest,
        prepared_environment_digest=prepared.environment_digest,
    )


def _wait_terminal(
    backend: DurableWorkerBackend,
    provider_run_id: str,
    *,
    timeout: float = 15,
):
    deadline = time.monotonic() + timeout
    request = RunToolRequest(provider_run_id=provider_run_id)
    while time.monotonic() < deadline:
        run = backend.get(request).run
        if run.status.status in {"completed", "failed", "cancelled"}:
            return run
        time.sleep(0.02)
    pytest.fail(f"provider run {provider_run_id} did not become terminal")


def test_detached_worker_executes_real_admitted_operation_and_publishes_manifest(
    tmp_path: Path,
):
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    backend = DurableWorkerBackend(registry=registry, run_root=tmp_path / "worker")

    submitted = backend.submit(_request(backend, reads)).run

    assert submitted.status.status in {"queued", "running"}
    completed = _wait_terminal(backend, submitted.provider_run_id)
    assert completed.status.status == "completed"
    assert completed.status.error is None
    assert completed.status.result is not None
    assert completed.status.result["ok"] is True
    assert completed.status.result["operation_id"] == "io.read_long_reads"

    manifest = backend.artifacts(RunToolRequest(provider_run_id=submitted.provider_run_id)).manifest
    assert manifest.provider_run_id == submitted.provider_run_id
    assert manifest.result_snapshot == completed.status.result
    assert len(manifest.artifacts) == 1
    artifact = manifest.artifacts[0]
    assert artifact.sha256 == hashlib.sha256(reads.read_bytes()).hexdigest()
    assert artifact.size_bytes == reads.stat().st_size
    assert (
        artifact.locator == f"sha256/{artifact.sha256[:2]}/{artifact.sha256}.{artifact.size_bytes}"
    )


def test_catalog_mismatch_refuses_before_creating_or_executing_a_run(tmp_path: Path):
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    root = tmp_path / "worker"
    backend = DurableWorkerBackend(registry=registry, run_root=root)
    request = _request(backend, reads).model_copy(
        update={"catalog_digest": _digest(b"different catalog")}
    )

    with pytest.raises(OrganelleContractError, match="catalog"):
        backend.submit(request)

    assert not (root / "runs").exists() or not tuple((root / "runs").iterdir())


def test_backend_revalidates_invocation_before_starting_worker(tmp_path: Path):
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\nIIII\n", encoding="utf-8")
    root = tmp_path / "worker"
    backend = DurableWorkerBackend(registry=registry, run_root=root)
    request = _request(backend, reads).model_copy(
        update={"parameters_snapshot": {"technology": "pacbio_hifi"}}
    )

    with pytest.raises(OrganelleContractError, match="invocation"):
        backend.submit(request)

    assert not (root / "runs").exists() or not tuple((root / "runs").iterdir())


def _result_spec(operation_id: str, *, side_effects: tuple[SideEffect, ...] = ()):
    return OperationSpec(
        operation_id=operation_id,
        contract_version="1.0",
        title="Worker semantics probe",
        description="A real in-memory admitted operation used to pin worker semantics.",
        keywords=("semantics", "test", "worker"),
        execution_mode=ExecutionMode.DURABLE,
        stage=OperationStage.ANALYZE,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.RESULT,
        callable_locator="tests.worker:probe",
        side_effects=side_effects,
    )


def _semantic_request(operation_id: str, *, grants: frozenset[SideEffect] = frozenset()):
    return SubmitToolRequest(
        operation_id=operation_id,
        catalog_digest=_digest(b"semantic-test-catalog"),
        input_snapshot=None,
        parameters_snapshot={},
        policy=InvocationPolicyProjection(side_effect_grants=grants),
        target_digest=_digest(b"semantic-test-target"),
        prepared_environment_digest=_digest(b"semantic-test-environment"),
    )


def test_scientific_failed_result_remains_successful_worker_invocation():
    local = OperationRegistry()

    def scientific_failure() -> OrganelleResult:
        return OrganelleResult(
            operation_id="test.scientific_failure",
            scope="none",
            status="failed",
            errors=(ErrorDetail(code="science.no_signal", message="No signal"),),
        )

    local.register(_result_spec("test.scientific_failure"), scientific_failure)

    response = invoke_admitted_operation(
        _semantic_request("test.scientific_failure"), registry=local
    )

    assert response["ok"] is True
    assert response["result"]["status"] == "failed"


def test_invoke_json_failure_is_a_provider_failure_not_scientific_truth():
    local = OperationRegistry()

    def permissioned_operation() -> OrganelleResult:
        return OrganelleResult(
            operation_id="test.permissioned",
            scope="none",
            status="ok",
        )

    local.register(
        _result_spec("test.permissioned", side_effects=(SideEffect.NETWORK,)),
        permissioned_operation,
    )

    with pytest.raises(OrganelleExecutionError) as raised:
        invoke_admitted_operation(_semantic_request("test.permissioned"), registry=local)

    assert raised.value.code == "compute.provider_invocation_failed"


def test_worker_invocation_forces_local_only_compute_target_policy():
    local = OperationRegistry()

    class _ExternalResolver:
        def resolve(self):
            raise AssertionError("worker attempted recursive external routing")

    def guarded_operation() -> OrganelleResult:
        hint = current_worker_compute_target_hint()
        if hint is None or hint.policy != "local_only":
            _ExternalResolver().resolve()
        return OrganelleResult(
            operation_id="test.recursion_guard",
            scope="none",
            status="ok",
        )

    local.register(_result_spec("test.recursion_guard"), guarded_operation)

    response = invoke_admitted_operation(_semantic_request("test.recursion_guard"), registry=local)

    assert response["ok"] is True
