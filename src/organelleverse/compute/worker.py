"""Durable detached execution for the fixed Linux Provider protocol.

The MCP process is only a control plane.  Each admitted invocation is written
to a private run directory and executed by a detached Python child whose argv
contains only the provider-generated run id.  State files are the source of
truth, so a later server process can recover the same run.
"""

from __future__ import annotations

import contextlib
import contextvars
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import NoReturn, cast
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic import Field, JsonValue, ValidationError

from organelleverse.compute.artifacts import RemoteArtifact, RemoteArtifactManifest
from organelleverse.compute.contracts import ComputeTargetHint
from organelleverse.compute.protocol import (
    ArtifactsToolResponse,
    PreparedComputeTarget,
    PrepareToolRequest,
    PrepareToolResponse,
    ProbeToolRequest,
    ProbeToolResponse,
    ProviderRunHandle,
    RunStatus,
    RunToolRequest,
    RunToolResponse,
    SubmitToolRequest,
    SubmitToolResponse,
    WorkerIdentity,
    WorkerResourceFacts,
    operation_catalog_digest,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.operations.adapters.json import invoke_json
from organelleverse.operations.registry import OperationRegistry
from organelleverse.operations.spec import StrictSpecModel

__all__ = [
    "DurableWorkerBackend",
    "PersistentRunRecord",
    "current_worker_compute_target_hint",
    "invoke_admitted_operation",
    "run_worker",
]

_RUN_ROOT_ENV = "ORGANELLEVERSE_WORKER_RUN_ROOT"

# Upper bound on the child's wait for the parent to publish the spawned pid.
# The parent writes the pid immediately after Popen returns; if it dies in
# between, an unbounded wait would leave the orphan spinning at 200 Hz
# forever.  30 s is orders of magnitude beyond the real publish latency.
_PID_PUBLISH_TIMEOUT_SECONDS = 30.0
_TRANSPORT = "wsl_content_cache"
_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_LOCAL_ONLY_HINT = ComputeTargetHint(policy="local_only")
_ACTIVE_COMPUTE_TARGET_HINT: contextvars.ContextVar[ComputeTargetHint | None] = (
    contextvars.ContextVar("organelleverse_worker_compute_target_hint", default=None)
)


class PersistentRunRecord(StrictSpecModel):
    """Private durable state for one provider run."""

    provider_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    operation_id: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
    status: RunStatus
    pid: int | None = Field(default=None, ge=1)
    expected_argv: tuple[str, ...]
    worker_ready: bool = False
    cancel_requested: bool = False
    events: tuple[dict[str, JsonValue], ...] = ()


class _PreparedRecord(StrictSpecModel):
    operation_id: str
    software_selector: str
    prepared_target: PreparedComputeTarget


class _CancellationRequested(Exception):
    pass


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(payload: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _package_version() -> str:
    try:
        return importlib.metadata.version("organelleverse")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.1"


def _linux_distribution() -> str:
    try:
        values: dict[str, str] = {}
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
        return values.get("PRETTY_NAME") or values.get("NAME") or "Linux"
    except OSError:
        return "Linux"


def _memory_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return max(0, pages * page_size)
    except (OSError, ValueError):
        return 0


def _gpu_count() -> int:
    return sum(
        1
        for candidate in Path("/dev").glob("nvidia[0-9]*")
        if candidate.name.removeprefix("nvidia").isdigit()
    )


def _provider_error(
    code: str,
    message: str,
    *,
    details: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    return {
        "kind": "provider_failure",
        "error_code": code,
        "message": message,
        "details": dict(details or {}),
        "retryable": True,
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _read_json(path: Path) -> object:
    with path.open("rb") as handle:
        return json.load(handle)


def _run_dir(run_root: Path, provider_run_id: str) -> Path:
    return run_root / "runs" / provider_run_id


@contextlib.contextmanager
def _run_lock(run_root: Path, provider_run_id: str) -> Generator[None, None, None]:
    run_dir = _run_dir(run_root, provider_run_id)
    if not run_dir.is_dir():
        raise OrganelleInputError(
            code="compute.unknown_provider_run",
            message="the provider run id is unknown",
            details={"provider_run_id": provider_run_id},
        )
    lock_path = run_dir / "state.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _state_path(run_root: Path, provider_run_id: str) -> Path:
    return _run_dir(run_root, provider_run_id) / "state.json"


def _request_path(run_root: Path, provider_run_id: str) -> Path:
    return _run_dir(run_root, provider_run_id) / "request.json"


def _manifest_path(run_root: Path, provider_run_id: str) -> Path:
    return _run_dir(run_root, provider_run_id) / "manifest.json"


def _load_state(run_root: Path, provider_run_id: str) -> PersistentRunRecord:
    return PersistentRunRecord.model_validate(_read_json(_state_path(run_root, provider_run_id)))


def _write_state(run_root: Path, record: PersistentRunRecord) -> None:
    _atomic_write_json(
        _state_path(run_root, record.provider_run_id), record.model_dump(mode="json")
    )
    _atomic_write_json(
        _run_dir(run_root, record.provider_run_id) / "events.json", list(record.events)
    )


def _event(name: str, revision: int) -> dict[str, JsonValue]:
    return {"event": name, "revision": revision}


def _handle(record: PersistentRunRecord) -> ProviderRunHandle:
    return ProviderRunHandle(provider_run_id=record.provider_run_id, status=record.status)


def _worker_argv(provider_run_id: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "organelleverse.compute.worker",
        "run",
        provider_run_id,
    )


def _process_owns_argv(pid: int, expected_argv: tuple[str, ...]) -> bool:
    try:
        stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if len(stat_fields) >= 3 and stat_fields[2] == "Z":
            return False
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    observed = tuple(part.decode(errors="surrogateescape") for part in raw.split(b"\0") if part)
    return observed == expected_argv


def current_worker_compute_target_hint() -> ComputeTargetHint | None:
    """Return the forced local-only target policy while a worker invokes L4.

    Honesty note (spec §8.3): :func:`invoke_admitted_operation` *sets* this
    hint, but no production routing layer *reads* it yet, so anti-recursion
    is not enforced end-to-end today.  The hook is exported so the external
    routing consumer (tracked for L5-03) binds against a stable seam instead
    of retrofitting one.  Until then, do not claim §8.3 enforcement.
    """

    return _ACTIVE_COMPUTE_TARGET_HINT.get()


def invoke_admitted_operation(
    request: SubmitToolRequest,
    *,
    registry: OperationRegistry,
) -> dict[str, JsonValue]:
    """Invoke one persisted request through L4 with external routing forbidden.

    A valid L1 ``OrganelleResult(status="failed")`` remains inside an ``ok=True``
    response.  Only ``invoke_json``'s ``ok=False`` boundary is promoted to a
    provider invocation failure.
    """

    invocation: dict[str, object] = {
        "operation_id": request.operation_id,
        "input": request.input_snapshot,
        "parameters": request.parameters_snapshot,
    }
    token = _ACTIVE_COMPUTE_TARGET_HINT.set(_LOCAL_ONLY_HINT)
    try:
        raw_response = invoke_json(
            invocation,
            registry=registry,
            granted_side_effects=request.policy.side_effect_grants,
        )
    finally:
        _ACTIVE_COMPUTE_TARGET_HINT.reset(token)
    response = cast(dict[str, JsonValue], raw_response)
    if response.get("ok") is not True:
        error = response.get("error")
        details = error if isinstance(error, dict) else {}
        raise OrganelleExecutionError(
            code="compute.provider_invocation_failed",
            message="the admitted worker invocation failed before producing scientific truth",
            details=details,
        )
    return response


class DurableWorkerBackend:
    """Disk-backed implementation of the fixed Linux provider backend seam."""

    def __init__(self, *, registry: OperationRegistry, run_root: str | Path) -> None:
        self.registry = registry
        self.run_root = Path(run_root).expanduser().absolute()
        self.run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.run_root, 0o700)

    def probe(self, request: ProbeToolRequest) -> ProbeToolResponse:
        del request
        return ProbeToolResponse(
            worker_identity=WorkerIdentity(
                linux_distribution=_linux_distribution(),
                kernel=platform.release(),
                architecture=platform.machine(),
                python_version=platform.python_version(),
                organelleverse_version=_package_version(),
            ),
            worker_catalog_digest=operation_catalog_digest(self.registry),
            resources=WorkerResourceFacts(
                cpu_count=os.cpu_count() or 1,
                memory_bytes=_memory_bytes(),
                disk_free_bytes=shutil.disk_usage(self.run_root).free,
                gpu_count=_gpu_count(),
            ),
            artifact_transports=(_TRANSPORT,),
        )

    def prepare(self, request: PrepareToolRequest) -> PrepareToolResponse:
        self.registry.describe(request.operation_id)
        installed_selector = f"organelleverse=={_package_version()}"
        if request.software_selector != installed_selector:
            raise OrganelleDependencyError(
                code="compute.software_selector_mismatch",
                message="the worker software identity does not match the requested selector",
                details={
                    "requested_selector": request.software_selector,
                    "installed_selector": installed_selector,
                },
            )
        for asset in request.required_assets:
            asset_path = self.run_root / "assets" / f"{asset.sha256}.{asset.size_bytes}"
            if not _file_matches(asset_path, asset.sha256, asset.size_bytes):
                raise OrganelleDependencyError(
                    code="compute.required_asset_unavailable",
                    message="a required prepared asset is unavailable or has the wrong identity",
                    details={"asset_id": asset.asset_id},
                )

        environment_payload = {
            "worker_identity": self.probe(ProbeToolRequest()).worker_identity.model_dump(
                mode="json"
            ),
            "python_executable": hashlib.sha256(os.fsencode(sys.executable)).hexdigest(),
            "software_selector": request.software_selector,
        }
        environment_digest = _digest(environment_payload)
        target_payload = {
            "operation_id": request.operation_id,
            "catalog_digest": operation_catalog_digest(self.registry),
            "capability_contract_digest": request.capability_contract_digest,
            "environment_digest": environment_digest,
            "required_assets": [asset.model_dump(mode="json") for asset in request.required_assets],
        }
        prepared = PreparedComputeTarget(
            target_digest=_digest(target_payload),
            environment_digest=environment_digest,
            capability_contract_digest=request.capability_contract_digest,
        )
        record = _PreparedRecord(
            operation_id=request.operation_id,
            software_selector=request.software_selector,
            prepared_target=prepared,
        )
        _atomic_write_json(
            self.run_root / "prepared" / f"{prepared.target_digest}.json",
            record.model_dump(mode="json"),
        )
        return PrepareToolResponse(prepared_target=prepared)

    def submit(self, request: SubmitToolRequest) -> SubmitToolResponse:
        worker_digest = operation_catalog_digest(self.registry)
        if request.catalog_digest != worker_digest:
            raise OrganelleContractError(
                code="compute.catalog_mismatch",
                message="submitted catalog digest does not match the admitted worker catalog",
                details={
                    "submitted_catalog_digest": request.catalog_digest,
                    "worker_catalog_digest": worker_digest,
                },
            )
        self.registry.describe(request.operation_id)
        schema = self.registry.invocation_schema(request.operation_id)
        invocation: dict[str, JsonValue] = {
            "operation_id": request.operation_id,
            "input": request.input_snapshot,
            "parameters": request.parameters_snapshot,
        }
        errors = sorted(
            Draft202012Validator(schema).iter_errors(invocation),  # pyright: ignore[reportUnknownMemberType]
            key=str,
        )
        if errors:
            raise OrganelleContractError(
                code="compute.invocation_mismatch",
                message="submitted invocation does not match the admitted worker schema",
                details={"reason": errors[0].message},
            )
        self._validate_prepared_target(request)
        self._validate_staged_artifacts(request)

        provider_run_id = f"run-{uuid4().hex}"
        argv = _worker_argv(provider_run_id)
        run_dir = _run_dir(self.run_root, provider_run_id)
        run_dir.mkdir(mode=0o700, parents=True)
        _atomic_write_json(
            _request_path(self.run_root, provider_run_id), request.model_dump(mode="json")
        )
        queued = PersistentRunRecord(
            provider_run_id=provider_run_id,
            operation_id=request.operation_id,
            status=RunStatus(status="queued", revision=1),
            expected_argv=argv,
            events=(_event("queued", 1),),
        )
        _write_state(self.run_root, queued)

        environment = os.environ.copy()
        environment[_RUN_ROOT_ENV] = str(self.run_root)
        # Pin the detached worker to the host's import root (the parent of the
        # installed ``organelleverse`` package). Without this the child resolves
        # ``organelleverse`` through the ambient environment — which may be a
        # different checkout or an editable install lacking this module — and
        # the catalog-digest guard then correctly refuses the run.
        import_root = str(Path(__file__).resolve().parents[2])
        pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            import_root + os.pathsep + pythonpath if pythonpath else import_root
        )
        try:
            child = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
                env=environment,
            )
        except OSError as error:
            failed = queued.model_copy(
                update={
                    "status": RunStatus(
                        status="failed",
                        revision=2,
                        error=_provider_error(
                            "compute.worker_start_failed",
                            "the detached provider worker could not be started",
                            details={"errno": error.errno, "reason": str(error)},
                        ),
                    ),
                    "events": (*queued.events, _event("provider_failed", 2)),
                }
            )
            _write_state(self.run_root, failed)
            return SubmitToolResponse(run=_handle(failed))

        with _run_lock(self.run_root, provider_run_id):
            current = _load_state(self.run_root, provider_run_id)
            running = current.model_copy(
                update={
                    "status": RunStatus(status="running", revision=2),
                    "pid": child.pid,
                    "events": (*current.events, _event("worker_started", 2)),
                }
            )
            _write_state(self.run_root, running)
        return SubmitToolResponse(run=_handle(running))

    def get(self, request: RunToolRequest) -> RunToolResponse:
        record = self.read_persistent_record(request.provider_run_id)
        return RunToolResponse(run=_handle(record))

    def cancel(self, request: RunToolRequest) -> RunToolResponse:
        provider_run_id = request.provider_run_id
        with _run_lock(self.run_root, provider_run_id):
            record = self._load_known_state(provider_run_id)
            if record.status.status in _TERMINAL:
                return RunToolResponse(run=_handle(record))
            revision = record.status.revision + 1
            requested = record.model_copy(
                update={
                    "status": RunStatus(status=record.status.status, revision=revision),
                    "cancel_requested": True,
                    "events": (*record.events, _event("cancellation_requested", revision)),
                }
            )
            _write_state(self.run_root, requested)
            pid = requested.pid
            ready = requested.worker_ready
            expected_argv = requested.expected_argv

        if ready and pid is not None:
            if not _process_owns_argv(pid, expected_argv):
                return RunToolResponse(run=_handle(self._mark_worker_lost(provider_run_id)))
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return RunToolResponse(run=_handle(self._mark_worker_lost(provider_run_id)))

        # Return the persisted nonterminal cancellation request.  Only the child
        # may acknowledge it by publishing the cancelled terminal state.
        return RunToolResponse(run=_handle(requested))

    def artifacts(self, request: RunToolRequest) -> ArtifactsToolResponse:
        record = self.read_persistent_record(request.provider_run_id)
        if record.status.status != "completed":
            raise OrganelleExecutionError(
                code="compute.artifacts_unavailable",
                message="provider artifacts are available only for completed runs",
                details={"provider_run_id": request.provider_run_id},
            )
        try:
            manifest = RemoteArtifactManifest.model_validate(
                _read_json(_manifest_path(self.run_root, request.provider_run_id))
            )
        except (OSError, ValueError, ValidationError) as error:
            raise OrganelleExecutionError(
                code="compute.manifest_invalid",
                message="the provider artifact manifest is missing or invalid",
                details={"provider_run_id": request.provider_run_id},
            ) from error
        for artifact in manifest.artifacts:
            if not _file_matches(
                self.run_root / "objects" / artifact.locator,
                artifact.sha256,
                artifact.size_bytes,
            ):
                raise OrganelleExecutionError(
                    code="compute.remote_artifact_mismatch",
                    message="a remote artifact does not match its manifest identity",
                    details={"object_id": artifact.object_id},
                )
        return ArtifactsToolResponse(manifest=manifest)

    def read_persistent_record(self, provider_run_id: str) -> PersistentRunRecord:
        with _run_lock(self.run_root, provider_run_id):
            record = self._load_known_state(provider_run_id)
            if record.status.status not in _TERMINAL and (
                record.pid is None or not _process_owns_argv(record.pid, record.expected_argv)
            ):
                record = self._worker_lost_record(record)
                _write_state(self.run_root, record)
            return record

    def _load_known_state(self, provider_run_id: str) -> PersistentRunRecord:
        run_dir = _run_dir(self.run_root, provider_run_id)
        if not run_dir.is_dir():
            raise OrganelleInputError(
                code="compute.unknown_provider_run",
                message="the provider run id is unknown",
                details={"provider_run_id": provider_run_id},
            )
        try:
            return _load_state(self.run_root, provider_run_id)
        except (OSError, ValueError, ValidationError):
            operation_id = "unknown.unknown"
            try:
                persisted = SubmitToolRequest.model_validate(
                    _read_json(_request_path(self.run_root, provider_run_id))
                )
                operation_id = persisted.operation_id
            except (OSError, ValueError, ValidationError):
                pass
            failed = PersistentRunRecord(
                provider_run_id=provider_run_id,
                operation_id=operation_id,
                status=RunStatus(
                    status="failed",
                    revision=1,
                    error=_provider_error(
                        "compute.worker_state_invalid",
                        "the persistent provider run state is missing or invalid",
                    ),
                ),
                expected_argv=_worker_argv(provider_run_id),
                events=(_event("provider_failed", 1),),
            )
            _write_state(self.run_root, failed)
            return failed

    def _worker_lost_record(self, record: PersistentRunRecord) -> PersistentRunRecord:
        revision = record.status.revision + 1
        return record.model_copy(
            update={
                "status": RunStatus(
                    status="failed",
                    revision=revision,
                    error=_provider_error(
                        "compute.worker_lost",
                        "the detached provider worker exited without terminal state",
                    ),
                ),
                "events": (*record.events, _event("provider_failed", revision)),
            }
        )

    def _mark_worker_lost(self, provider_run_id: str) -> PersistentRunRecord:
        with _run_lock(self.run_root, provider_run_id):
            record = self._load_known_state(provider_run_id)
            if record.status.status in _TERMINAL:
                return record
            failed = self._worker_lost_record(record)
            _write_state(self.run_root, failed)
            return failed

    def _validate_prepared_target(self, request: SubmitToolRequest) -> None:
        prepared_path = self.run_root / "prepared" / f"{request.target_digest}.json"
        try:
            prepared = _PreparedRecord.model_validate(_read_json(prepared_path))
        except (OSError, ValueError, ValidationError) as error:
            raise OrganelleContractError(
                code="compute.target_not_prepared",
                message="the submitted target has not been prepared by this worker",
            ) from error
        if (
            prepared.operation_id != request.operation_id
            or prepared.prepared_target.target_digest != request.target_digest
            or prepared.prepared_target.environment_digest != request.prepared_environment_digest
        ):
            raise OrganelleContractError(
                code="compute.prepared_target_mismatch",
                message="the submitted operation or environment does not match the prepared target",
            )

    def _validate_staged_artifacts(self, request: SubmitToolRequest) -> None:
        for raw in request.staged_artifacts:
            from organelleverse.compute.artifacts import StagedArtifact

            try:
                artifact = StagedArtifact.model_validate(raw)
            except ValidationError as error:
                raise OrganelleContractError(
                    code="compute.staged_artifact_invalid",
                    message="a submitted staged artifact identity is invalid",
                ) from error
            if not _file_matches(
                self.run_root / "objects" / artifact.locator,
                artifact.sha256,
                artifact.size_bytes,
            ):
                raise OrganelleContractError(
                    code="compute.staged_artifact_mismatch",
                    message="a staged artifact does not match the bytes available to the worker",
                    details={"object_id": artifact.object_id},
                )


def _file_matches(path: Path, expected_sha256: str, expected_size: int) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != expected_size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected_sha256
    except OSError:
        return False


def _artifact_refs(result: Mapping[str, object]) -> tuple[ArtifactRef, ...]:
    kind = result.get("kind")
    values: list[object] = []
    if kind == "result":
        raw = result.get("artifacts", [])
        if isinstance(raw, list):
            values.extend(cast(list[object], raw))
    elif kind == "data":
        raw = result.get("artifacts", {})
        if isinstance(raw, Mapping):
            values.extend(cast(Mapping[str, object], raw).values())
    elif kind == "genome":
        values.extend(
            item for item in (result.get("sequence"), result.get("annotation")) if item is not None
        )
        manifests = result.get("source_manifests", [])
        if isinstance(manifests, list):
            values.extend(cast(list[object], manifests))
    return tuple(ArtifactRef.model_validate(item) for item in values)


def _publish_artifact(run_root: Path, artifact: ArtifactRef) -> RemoteArtifact:
    locator = f"sha256/{artifact.sha256[:2]}/{artifact.sha256}.{artifact.size_bytes}"
    destination = run_root / "objects" / locator
    if not _file_matches(destination, artifact.sha256, artifact.size_bytes):
        source = artifact.resolve()
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                os.chmod(temporary, 0o600)
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if digest.hexdigest() != artifact.sha256 or size != artifact.size_bytes:
                raise OrganelleExecutionError(
                    code="compute.output_artifact_mismatch",
                    message="an operation output artifact changed before provider publication",
                    details={"object_id": artifact.object_id},
                )
            os.replace(temporary, destination)
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
    return RemoteArtifact(
        object_id=artifact.object_id,
        kind=artifact.kind,
        format=artifact.format,
        media_type=artifact.media_type,
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
        transport=_TRANSPORT,
        locator=locator,
    )


def _manifest(
    run_root: Path,
    provider_run_id: str,
    request: SubmitToolRequest,
    response: dict[str, JsonValue],
) -> RemoteArtifactManifest:
    result = response.get("result")
    if not isinstance(result, Mapping):
        raise OrganelleExecutionError(
            code="compute.worker_result_invalid",
            message="a successful worker response has no strict L1 result",
        )
    artifacts = tuple(_publish_artifact(run_root, ref) for ref in _artifact_refs(result))
    fields = {
        "provider_run_id": provider_run_id,
        "operation_id": request.operation_id,
        "target_digest": request.target_digest,
        "prepared_environment_digest": request.prepared_environment_digest,
        "result_snapshot": response,
        "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
    }
    return RemoteArtifactManifest(
        provider_run_id=provider_run_id,
        operation_id=request.operation_id,
        target_digest=request.target_digest,
        prepared_environment_digest=request.prepared_environment_digest,
        result_snapshot=response,
        artifacts=artifacts,
        manifest_id="sha256:" + hashlib.sha256(_canonical_json_bytes(fields)).hexdigest(),
    )


def _cancel_signal(signum: int, frame: object) -> NoReturn:
    del signum, frame
    raise _CancellationRequested


def run_worker(
    provider_run_id: str,
    *,
    run_root: str | Path,
    registry: OperationRegistry,
) -> int:
    """Execute one already-persisted provider run in the detached child."""

    root = Path(run_root).expanduser().absolute()
    signal.signal(signal.SIGTERM, _cancel_signal)
    try:
        request = SubmitToolRequest.model_validate(_read_json(_request_path(root, provider_run_id)))
        # The parent must publish the spawned PID before the child is allowed
        # to advance lifecycle state.  This single ordering invariant prevents
        # a fast operation from racing past cancellation/process ownership.
        # The wait is bounded: if the parent died between Popen and the pid
        # write, an unbounded wait would spin this orphan at 200 Hz forever.
        # On timeout the child exits without a terminal state, which any
        # surviving parent observes as compute.worker_lost.
        deadline = time.monotonic() + _PID_PUBLISH_TIMEOUT_SECONDS
        while True:
            with _run_lock(root, provider_run_id):
                record = _load_state(root, provider_run_id)
                if record.pid is not None:
                    break
            if time.monotonic() >= deadline:
                return 2
            time.sleep(0.005)
        with _run_lock(root, provider_run_id):
            record = _load_state(root, provider_run_id)
            revision = record.status.revision + 1
            ready = record.model_copy(
                update={
                    "worker_ready": True,
                    "events": (*record.events, _event("worker_ready", revision)),
                    "status": RunStatus(status="running", revision=revision),
                }
            )
            _write_state(root, ready)
            if ready.cancel_requested:
                cancelled = ready.model_copy(
                    update={
                        "status": RunStatus(status="cancelled", revision=revision + 1),
                        "events": (*ready.events, _event("cancelled", revision + 1)),
                    }
                )
                _write_state(root, cancelled)
                return 0

        if request.catalog_digest != operation_catalog_digest(registry):
            raise OrganelleContractError(
                code="compute.catalog_mismatch",
                message="persisted catalog digest no longer matches the worker catalog",
            )
        registry.describe(request.operation_id)
        schema = registry.invocation_schema(request.operation_id)
        invocation: dict[str, JsonValue] = {
            "operation_id": request.operation_id,
            "input": request.input_snapshot,
            "parameters": request.parameters_snapshot,
        }
        errors = list(
            Draft202012Validator(schema).iter_errors(invocation)  # pyright: ignore[reportUnknownMemberType]
        )
        if errors:
            raise OrganelleContractError(
                code="compute.invocation_mismatch",
                message="persisted invocation no longer matches the worker schema",
            )

        response = invoke_admitted_operation(request, registry=registry)

        manifest = _manifest(root, provider_run_id, request, response)
        _atomic_write_json(_run_dir(root, provider_run_id) / "response.json", response)
        _atomic_write_json(_manifest_path(root, provider_run_id), manifest.model_dump(mode="json"))
        with _run_lock(root, provider_run_id):
            current = _load_state(root, provider_run_id)
            revision = current.status.revision + 1
            completed = current.model_copy(
                update={
                    "status": RunStatus(status="completed", revision=revision, result=response),
                    "events": (*current.events, _event("completed", revision)),
                }
            )
            _write_state(root, completed)
        return 0
    except _CancellationRequested:
        with _run_lock(root, provider_run_id):
            current = _load_state(root, provider_run_id)
            if current.status.status in _TERMINAL:
                # SIGTERM can be delivered at any bytecode boundary — including
                # after _write_state(completed) above.  A published terminal
                # state is authoritative (mirror _mark_worker_lost); never
                # overwrite it with cancelled.
                return 0
            revision = current.status.revision + 1
            cancelled = current.model_copy(
                update={
                    "status": RunStatus(status="cancelled", revision=revision),
                    "events": (*current.events, _event("cancelled", revision)),
                }
            )
            _write_state(root, cancelled)
        return 0
    except BaseException as error:
        try:
            with _run_lock(root, provider_run_id):
                current = _load_state(root, provider_run_id)
                if current.status.status in _TERMINAL:
                    # A terminal state is already authoritative (e.g. the run
                    # completed and a late exception raced process exit);
                    # never overwrite it with a crash record.
                    return 0
                revision = current.status.revision + 1
                if isinstance(error, (OrganelleContractError, OrganelleExecutionError)):
                    code = error.code
                    message = error.message
                else:
                    code = "compute.worker_crashed"
                    message = "the detached provider worker did not produce a usable result"
                failed = current.model_copy(
                    update={
                        "status": RunStatus(
                            status="failed",
                            revision=revision,
                            error=_provider_error(code, message),
                        ),
                        "events": (*current.events, _event("provider_failed", revision)),
                    }
                )
                _write_state(root, failed)
        except BaseException:
            pass
        return 1


def _main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2 or arguments[0] != "run":
        return 2
    run_root = os.environ.get(_RUN_ROOT_ENV)
    if not run_root:
        return 2
    from organelleverse.operations.registry import registry

    return run_worker(arguments[1], run_root=run_root, registry=registry)


if __name__ == "__main__":
    raise SystemExit(_main())
