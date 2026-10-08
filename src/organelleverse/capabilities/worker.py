"""One-shot isolated-interpreter launcher for verified bundle-local Python."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, JsonValue, ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleExecutionError,
    OrganelleInputError,
    OrganellePermissionError,
)
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.provenance import (
    InlineMaterializationRecord,
    PathTranslationRecord,
)
from organelleverse.core.result import OrganelleResult
from organelleverse.environments.path_translation import (
    ExecutionContext,
    translate_path,
)
from organelleverse.operations.registry import CoreObject
from organelleverse.operations.spec import (
    ArgumentMode,
    InlineParameterValue,
    ParameterCodec,
    PathParameterValue,
    ResultCodec,
)
from organelleverse.runtime import (
    create_staged_run,
    managed_run_path,
    publish_staged_result,
    remove_managed_entry,
)

from .code_identity import ExecutionIdentity, inspect_bundle_code
from .codecs import (
    CapabilityExecutionContext,
    encode_capability_result,
    encode_plugin_worker_failure,
    encode_plugin_worker_result,
    normalize_worker_core_object,
)
from .hashing import hash_bundle
from .index import CapabilityEntry
from .inline_materialize import decode_inline_content, materialize_inline_parameter
from .trust import TrustStore
from .worker_contracts import (
    WorkerInspection,
    WorkerMode,
    WorkerRequest,
    WorkerResponse,
    WorkerResult,
)

MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 10_000
_MAX_STDOUT_BYTES = MAX_FRAME_BYTES + 9
_READ_CHUNK = 64 * 1024
_WINDOWS = os.name == "nt"
_WORKER_MAIN = Path(__file__).with_name("_worker_main.py").resolve()
_LOGGER = logging.getLogger(__name__)


class BundleWorkerExecutor(Protocol):
    def inspect(self, entry: CapabilityEntry) -> WorkerInspection: ...

    def invoke(
        self,
        entry: CapabilityEntry,
        *,
        input: CoreObject | None,
        parameters: Mapping[str, object],
        run_id: str,
        staging_root: Path,
    ) -> WorkerResult: ...


def _frame_error(
    code: str,
    message: str,
    **details: object,
) -> OrganelleExecutionError:
    return OrganelleExecutionError(code=code, message=message, details=details)


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _validate_json_structure(value: object) -> None:
    remaining = MAX_JSON_NODES
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        remaining -= 1
        if remaining < 0 or depth > MAX_JSON_DEPTH:
            raise ValueError("JSON structure exceeds the worker depth or node bound")
        if item is None or isinstance(item, (bool, str, int)):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            continue
        if isinstance(item, Mapping):
            mapping = cast(Mapping[object, object], item)
            if not all(isinstance(key, str) for key in mapping):
                raise ValueError("JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in mapping.values())
            continue
        if isinstance(item, (list, tuple)):
            stack.extend(
                (child, depth + 1) for child in cast(list[object] | tuple[object, ...], item)
            )
            continue
        raise ValueError(f"{type(item).__name__} is not JSON-compatible")


def _canonical_json(value: object) -> bytes:
    _validate_json_structure(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _encode_frame(value: object) -> bytes:
    try:
        content = _canonical_json(value)
    except (RecursionError, TypeError, ValueError) as error:
        raise OrganelleContractError(
            code="capability.worker_frame_invalid",
            message="worker request cannot be encoded as bounded canonical JSON",
            details={"reason": str(error)},
        ) from error
    if len(content) > MAX_FRAME_BYTES:
        raise OrganelleContractError(
            code="capability.worker_frame_too_large",
            message="worker request frame exceeds the configured bound",
            details={
                "max_bytes": MAX_FRAME_BYTES,
                "observed_bytes": len(content),
            },
        )
    return struct.pack(">Q", len(content)) + content


def _decode_frame(frame: bytes) -> object:
    if len(frame) < 8:
        raise _frame_error(
            "capability.worker_frame_invalid",
            "worker response is missing its length prefix",
        )
    size = struct.unpack(">Q", frame[:8])[0]
    if size > MAX_FRAME_BYTES:
        raise _frame_error(
            "capability.worker_frame_too_large",
            "worker response frame exceeds the configured bound",
            max_bytes=MAX_FRAME_BYTES,
            observed_bytes=size,
        )
    if len(frame) != size + 8:
        raise _frame_error(
            "capability.worker_frame_invalid",
            "worker response must contain exactly one frame followed by EOF",
            declared_bytes=size,
            observed_bytes=max(0, len(frame) - 8),
        )
    try:
        value = json.loads(
            frame[8:].decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        _validate_json_structure(value)
        if _canonical_json(value) != frame[8:]:
            raise ValueError("worker response JSON is not in canonical wire form")
        return value
    except (RecursionError, UnicodeDecodeError, ValueError) as error:
        raise _frame_error(
            "capability.worker_frame_invalid",
            "worker response is not bounded canonical JSON",
            reason=str(error),
        ) from error


def _validate_response(request: WorkerRequest, response: WorkerResponse) -> None:
    if response.protocol != request.protocol:
        raise _frame_error(
            "capability.worker_protocol_mismatch",
            "worker response protocol does not match the request",
        )
    if response.request_id != request.request_id:
        raise _frame_error(
            "capability.worker_request_mismatch",
            "worker response request ID does not match the request",
        )
    if response.operation_id != request.operation_id:
        raise _frame_error(
            "capability.worker_operation_mismatch",
            "worker response operation ID does not match the request",
        )
    if response.mode != request.mode:
        raise _frame_error(
            "capability.worker_mode_mismatch",
            "worker response mode does not match the request",
        )
    if response.execution_identity != request.execution_identity:
        raise _frame_error(
            "capability.worker_identity_mismatch",
            "worker response execution identity does not match the request",
            expected_digest=request.execution_identity.digest,
            actual_digest=response.execution_identity.digest,
        )


class _BoundedDrain:
    def __init__(self, stream: BinaryIO, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self.content = bytearray()
        self.total = 0
        self.error: OSError | None = None

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read(_READ_CHUNK)
                if not chunk:
                    return
                self.total += len(chunk)
                remaining = self._limit - len(self.content)
                if remaining > 0:
                    self.content.extend(chunk[:remaining])
        except OSError as error:
            self.error = error
        finally:
            self._stream.close()


def _minimal_environment() -> dict[str, str]:
    environment = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
    if "PATH" in os.environ:
        environment["PATH"] = os.defpath
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.pid <= 0:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    _kill_process_group(process)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _close_pipe(stream: BinaryIO | None) -> None:
    if stream is None:
        return
    with suppress(OSError):
        stream.close()


class _RequestWriter:
    def __init__(self, stream: BinaryIO, frame: bytes) -> None:
        self._stream = stream
        self._frame = frame
        self.error: OSError | None = None

    def run(self) -> None:
        try:
            self._stream.write(self._frame)
            self._stream.flush()
        except BrokenPipeError:
            pass
        except OSError as error:
            self.error = error
        finally:
            self._stream.close()


class OneShotBundleWorkerExecutor:
    """Launch one fresh isolated interpreter for every inspect/invoke request."""

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("worker timeout must be positive")
        self.timeout_seconds = timeout_seconds

    def inspect(self, entry: CapabilityEntry) -> WorkerInspection:
        with tempfile.TemporaryDirectory(prefix="organelleverse-worker-staging-") as staging:
            request = self._request(
                entry,
                mode="inspect",
                input=None,
                parameters={},
                run_id=uuid4().hex,
                staging_root=Path(staging),
            )
            response = self._exchange(request)
            scratch = Path(staging)
            try:
                residue = sorted(path.name for path in scratch.iterdir())
            except OSError as error:
                raise OrganelleContractError(
                    code="capability.inspect_side_effect",
                    message="worker inspection scratch could not be verified empty",
                    details={"reason": str(error)},
                ) from error
            if residue:
                raise OrganelleContractError(
                    code="capability.inspect_side_effect",
                    message="worker inspection must not create files or directories",
                    details={"entries": residue},
                )
        return WorkerInspection(
            request_id=response.request_id,
            execution_identity=response.execution_identity,
            parameters=response.parameters,
        )

    def invoke(
        self,
        entry: CapabilityEntry,
        *,
        input: CoreObject | None,
        parameters: Mapping[str, object],
        run_id: str,
        staging_root: Path,
    ) -> WorkerResult:
        input_json = None if input is None else cast(JsonValue, input.model_dump(mode="json"))
        request = self._request(
            entry,
            mode="invoke",
            input=input_json,
            parameters=parameters,
            run_id=run_id,
            staging_root=staging_root,
        )
        response = self._exchange(request)
        return WorkerResult(
            request_id=response.request_id,
            execution_identity=response.execution_identity,
            value=response.value,
            artifact_paths=response.artifact_paths,
        )

    def _request(
        self,
        entry: CapabilityEntry,
        *,
        mode: WorkerMode,
        input: JsonValue | None,
        parameters: Mapping[str, object],
        run_id: str,
        staging_root: Path,
    ) -> WorkerRequest:
        identity = entry.execution_identity
        if identity is None:
            raise OrganelleContractError(
                code="capability.execution_provider_required",
                message="bundle-local worker requires an execution identity",
                details={"capability_id": entry.capability_id},
            )
        if mode == "invoke" and entry.worker_parameters is None:
            raise OrganelleContractError(
                code="capability.verification_missing",
                message="bundle-local invocation requires frozen worker signature tokens",
                details={"capability_id": entry.capability_id},
            )
        worker_parameters = entry.worker_parameters or ()
        try:
            json_parameters = cast(dict[str, JsonValue], dict(parameters))
            _validate_json_structure(json_parameters)
            return WorkerRequest(
                request_id=uuid4().hex,
                operation_id=entry.capability_id,
                mode=mode,
                execution_identity=identity,
                bundle_root=str(entry.bundle_root.resolve(strict=True)),
                contract=cast(dict[str, JsonValue], entry.bundle.contract.model_dump(mode="json")),
                worker_parameters=worker_parameters,
                input=input,
                parameters=json_parameters,
                run_id=run_id,
                staging_root=str(staging_root.resolve(strict=True)),
            )
        except (OSError, RecursionError, TypeError, ValueError, ValidationError) as error:
            raise OrganelleContractError(
                code="capability.worker_frame_invalid",
                message="worker request fields are invalid or non-JSON",
                details={"capability_id": entry.capability_id, "reason": str(error)},
            ) from error

    def _exchange(self, request: WorkerRequest) -> WorkerResponse:
        if _WINDOWS:
            raise OrganelleContractError(
                code="capability.execution_provider_required",
                message=(
                    "controlled bundle workers require a Windows Job Object provider "
                    "for process-tree cleanup"
                ),
                details={"capability_id": request.operation_id},
            )
        # A private, per-run bytecode cache root. ``-B`` only stops the worker
        # from *writing* bytecode; a ``__pycache__`` already sitting next to a
        # bundle's sources -- which every pip-installed bundle has -- would
        # still be read and executed. Setting ``pycache_prefix`` makes CPython
        # ignore those in-tree caches outright, so only source that
        # ``code_tree_hash`` covers can execute. The directory is created and
        # removed here, never inside the bundle, and ``-B`` keeps it empty.
        with tempfile.TemporaryDirectory(prefix="organelleverse-worker-pycache-") as cache_root:
            return self._exchange_once(request, cache_root)

    def _exchange_once(self, request: WorkerRequest, pycache_prefix: str) -> WorkerResponse:
        frame = _encode_frame(request.model_dump(mode="json"))
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-E",
                    "-B",
                    "-X",
                    f"pycache_prefix={pycache_prefix}",
                    str(_WORKER_MAIN),
                ],
                shell=False,
                cwd=request.staging_root,
                env=_minimal_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
            )
        except OSError as error:
            raise _frame_error(
                "capability.worker_spawn_failed",
                "bundle-local worker process could not be started",
                reason=str(error),
            ) from error
        if process.stdin is None or process.stdout is None or process.stderr is None:
            _terminate_process_group(process)
            _close_pipe(cast(BinaryIO | None, process.stdin))
            _close_pipe(cast(BinaryIO | None, process.stdout))
            _close_pipe(cast(BinaryIO | None, process.stderr))
            raise _frame_error(
                "capability.worker_spawn_failed",
                "worker protocol pipes were not created",
            )
        stdout = _BoundedDrain(cast(BinaryIO, process.stdout), _MAX_STDOUT_BYTES)
        stderr = _BoundedDrain(cast(BinaryIO, process.stderr), MAX_STDERR_BYTES)
        request_writer = _RequestWriter(cast(BinaryIO, process.stdin), frame)
        stdout_thread = threading.Thread(target=stdout.run, daemon=True)
        stderr_thread = threading.Thread(target=stderr.run, daemon=True)
        request_thread = threading.Thread(target=request_writer.run, daemon=True)
        threads = (stdout_thread, stderr_thread, request_thread)
        started_threads: list[threading.Thread] = []
        timed_out = False
        thread_start_error: OSError | RuntimeError | None = None
        try:
            for thread in threads:
                thread.start()
                started_threads.append(thread)
            try:
                process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
        except (OSError, RuntimeError) as error:
            thread_start_error = error
        finally:
            _terminate_process_group(process)
            for thread in started_threads:
                thread.join(timeout=1.0)
            _close_pipe(cast(BinaryIO, process.stdin))
            _close_pipe(cast(BinaryIO, process.stdout))
            _close_pipe(cast(BinaryIO, process.stderr))
            for thread in started_threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
        if thread_start_error is not None:
            raise _frame_error(
                "capability.worker_thread_start_failed",
                "worker protocol thread could not be started",
                reason=str(thread_start_error),
            ) from thread_start_error
        if stdout_thread.is_alive() or stderr_thread.is_alive() or request_thread.is_alive():
            raise _frame_error(
                "capability.worker_cleanup_failed",
                "worker protocol pipes did not close after process-group cleanup",
            )
        if request_writer.error is not None:
            raise _frame_error(
                "capability.worker_protocol_invalid",
                "worker request frame could not be written",
                reason=str(request_writer.error),
            )
        if stdout.error is not None or stderr.error is not None:
            error = stdout.error or stderr.error
            raise _frame_error(
                "capability.worker_protocol_invalid",
                "worker output could not be read",
                reason=str(error),
            )
        stderr_text = bytes(stderr.content).decode("utf-8", errors="replace")
        stderr_details = {
            "stderr": stderr_text,
            "stderr_truncated": stderr.total > MAX_STDERR_BYTES,
        }
        if timed_out:
            raise _frame_error(
                "capability.worker_timeout",
                "bundle-local worker exceeded its execution timeout",
                timeout_seconds=self.timeout_seconds,
                **stderr_details,
            )
        returncode = process.returncode
        if returncode is None:
            raise _frame_error(
                "capability.worker_cleanup_failed",
                "bundle-local worker has no terminal return code",
            )
        if returncode < 0:
            raise _frame_error(
                "capability.worker_signal",
                "bundle-local worker exited from a signal",
                signal=-returncode,
                **stderr_details,
            )
        if returncode != 0:
            raise _frame_error(
                "capability.worker_nonzero_exit",
                "bundle-local worker exited without a valid response",
                returncode=returncode,
                **stderr_details,
            )
        if stdout.total > _MAX_STDOUT_BYTES:
            raise _frame_error(
                "capability.worker_frame_too_large",
                "worker stdout exceeds the single bounded response frame",
                max_bytes=_MAX_STDOUT_BYTES,
                observed_bytes=stdout.total,
            )
        payload = _decode_frame(bytes(stdout.content))
        try:
            response = WorkerResponse.model_validate(payload)
        except ValidationError as error:
            raise _frame_error(
                "capability.worker_protocol_invalid",
                "worker response does not satisfy the closed protocol envelope",
                validation_errors=error.errors(include_url=False),
            ) from error
        _validate_response(request, response)
        if response.status == "error":
            assert response.error is not None
            error_type = (
                OrganelleContractError
                if response.error.code
                in {
                    "capability.binding_invalid",
                    "capability.execution_provider_required",
                    "capability.plugin_result_invalid",
                    "capability.plugin_score_invalid",
                    "capability.plugin_score_reserved",
                    "capability.plugin_signature_invalid",
                    "capability.schema_drift",
                }
                else OrganelleExecutionError
            )
            raise error_type(
                code=response.error.code,
                message=response.error.message,
                details=response.error.details,
            )
        return response


def _canonical_parameter_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical_parameter_value(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return _canonical_parameter_value(value.value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise TypeError("worker parameter mappings require string keys")
        return {cast(str, key): _canonical_parameter_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        return [
            _canonical_parameter_value(item)
            for item in cast(list[object] | tuple[object, ...], value)
        ]
    _validate_json_structure(value)
    return value


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameters_hash(parameters: Mapping[str, object]) -> str:
    import hashlib

    canonical = _canonical_json(
        {name: _canonical_parameter_value(value) for name, value in parameters.items()}
    )
    return hashlib.sha256(canonical).hexdigest()


def _load_mount_map() -> dict[str, str] | None:
    """Load an optional host/container mount map from the environment.

    ``ORG_VERSE_MOUNT_MAP`` is a JSON object mapping host prefixes to container
    prefixes, e.g. ``{"C:\\\\data": "/data", "/input/data": "/data"}``.
    """
    raw = os.environ.get("ORG_VERSE_MOUNT_MAP")
    if not raw:
        return None
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError as error:
        raise OrganelleContractError(
            code="capability.mount_map_invalid",
            message="ORG_VERSE_MOUNT_MAP is not valid JSON",
            details={"reason": str(error)},
        ) from error
    if not isinstance(parsed, dict):
        raise OrganelleContractError(
            code="capability.mount_map_invalid",
            message="ORG_VERSE_MOUNT_MAP must be a JSON object",
            details={"actual_type": type(parsed).__name__},
        )
    parsed_map = cast(dict[object, object], parsed)
    if not parsed_map or not all(
        isinstance(host, str) and isinstance(container, str)
        for host, container in parsed_map.items()
    ):
        raise OrganelleContractError(
            code="capability.mount_map_invalid",
            message="ORG_VERSE_MOUNT_MAP must map string host paths to string container paths",
            details={"entry_count": len(parsed_map)},
        )
    return cast(dict[str, str], parsed_map)


def _host_path_for_value(
    value: PathParameterValue,
    *,
    mount_map: dict[str, str],
    capability_id: str,
) -> str:
    """Return the host-accessible path used to read and hash the file."""
    if value.context is ExecutionContext.HOST:
        return value.value
    translated = translate_path(
        value.value,
        source_context=value.context,
        target_context=ExecutionContext.HOST,
        mount_map=mount_map,
    )
    return translated.translated_value


def _decode_inline_json(
    value: InlineParameterValue,
    *,
    parameter_name: str,
    capability_id: str,
    inline_max_bytes: int,
) -> tuple[JsonValue, int]:
    """Decode an inline value and parse it as bounded JSON for a JSON codec."""
    content = decode_inline_content(
        value,
        parameter_name=parameter_name,
        capability_id=capability_id,
        inline_max_bytes=inline_max_bytes,
    )
    try:
        parsed = cast(
            JsonValue,
            json.loads(
                content.decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            ),
        )
        _validate_json_structure(parsed)
        return parsed, len(content)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise OrganelleInputError(
            code="input.inline_json_invalid",
            message="inline content is not bounded canonical JSON",
            details={"format": value.format, "reason": str(error)},
        ) from error


def _validate_inline_json_schema(
    value: JsonValue,
    *,
    binding: object,
    capability_id: str,
    parameter_name: str,
) -> None:
    from jsonschema import Draft202012Validator

    schema = getattr(binding, "json_schema", {})
    errors = sorted(
        Draft202012Validator(schema).iter_errors(  # pyright: ignore[reportUnknownMemberType]
            value
        ),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        raise OrganelleContractError(
            code="capability.parameter_schema_invalid",
            message=f"parameter {parameter_name!r} does not satisfy its declared json_schema",
            details={
                "operation_id": capability_id,
                "parameter": parameter_name,
                "errors": [
                    {"path": list(error.absolute_path), "message": error.message}
                    for error in errors
                ],
            },
        )


def _worker_parameter_error(
    capability_id: str,
    error: TypeError | ValueError | RecursionError,
) -> OrganelleContractError:
    return OrganelleContractError(
        code="capability.worker_frame_invalid",
        message="worker parameters are not bounded canonical JSON",
        details={"capability_id": capability_id, "reason": str(error)},
    )


def _resolved_worker_path(
    value: str,
    *,
    capability_id: str,
    parameter: str,
) -> Path:
    candidate = Path(value)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise OrganelleInputError(
            code="input.missing_artifact",
            message=f"artifact does not exist: {candidate}",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "path": str(candidate),
            },
        ) from error
    except (OSError, RuntimeError) as error:
        raise OrganelleInputError(
            code="input.unreadable_artifact",
            message=f"artifact could not be resolved: {candidate}",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "path": str(candidate),
                "reason": str(error),
            },
        ) from error
    if not resolved.is_file():
        raise OrganelleInputError(
            code="input.missing_artifact",
            message=f"artifact does not exist: {candidate}",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "path": str(candidate),
            },
        )
    return resolved


def _resolved_worker_directory(
    value: str,
    *,
    capability_id: str,
    parameter: str,
) -> Path:
    candidate = Path(value)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise OrganelleInputError(
            code="input.missing_artifact",
            message=f"directory input does not exist: {candidate}",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "path": str(candidate),
            },
        ) from error
    except (OSError, RuntimeError) as error:
        raise OrganelleInputError(
            code="input.unreadable_artifact",
            message=f"directory input could not be resolved: {candidate}",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "path": str(candidate),
                "reason": str(error),
            },
        ) from error
    if not resolved.is_dir():
        raise OrganelleInputError(
            code="input.missing_artifact",
            message=f"directory input does not exist: {candidate}",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "path": str(candidate),
            },
        )
    return resolved


def _reject_pre_encode_publication(
    value: object,
    entry: CapabilityEntry,
) -> None:
    binding = entry.bundle.contract.binding
    if binding.result_codec is not ResultCodec.LEGACY_RESULT or not isinstance(value, Mapping):
        return
    output_paths = cast(Mapping[object, object], value).get("output_paths")
    if (
        isinstance(output_paths, Sequence)
        and not isinstance(output_paths, (str, bytes))
        and len(cast(Sequence[object], output_paths)) > 0
    ):
        raise OrganelleContractError(
            code="capability.execution_provider_required",
            message="worker path publication requires the Task 3 provider",
            details={"capability_id": entry.capability_id},
        )


def _cleanup_owned_staging(staging: Path) -> None:
    if not _WINDOWS and staging.is_dir() and not staging.is_symlink():
        with suppress(OSError):
            for current, directory_names, _file_names in os.walk(
                staging, topdown=True, followlinks=False
            ):
                root = Path(current)
                root.chmod(0o700)
                directory_names[:] = [
                    name for name in directory_names if not (root / name).is_symlink()
                ]
    try:
        remove_managed_entry(staging)
    except (OSError, OrganelleInputError) as error:
        with suppress(BaseException):
            _LOGGER.warning(
                "controlled-worker staging cleanup was incomplete: %s: %s",
                staging,
                error,
            )


def _verified_input_artifact(
    artifact: ArtifactRef,
    *,
    capability_id: str,
    parameter: str,
) -> ArtifactRef:
    resolved = _resolved_worker_path(
        artifact.uri,
        capability_id=capability_id,
        parameter=parameter,
    )
    try:
        current = ArtifactRef.from_path(
            resolved,
            kind=artifact.kind,
            format=artifact.format,
            media_type=artifact.media_type,
        )
    except OSError as error:
        raise OrganelleInputError(
            code="input.unreadable_artifact",
            message=f"core input artifact could not be read: {resolved}",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "path": str(resolved),
                "reason": str(error),
            },
        ) from error
    if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
        raise OrganelleInputError(
            code="input.artifact_digest_mismatch",
            message="core input artifact bytes do not match their declared identity",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "path": str(resolved),
                "artifact_id": artifact.object_id,
                "declared_sha256": artifact.sha256,
                "actual_sha256": current.sha256,
                "declared_size_bytes": artifact.size_bytes,
                "actual_size_bytes": current.size_bytes,
            },
        )
    return current


def _verified_worker_input(
    input: CoreObject | None,
    *,
    capability_id: str,
) -> tuple[CoreObject | None, tuple[str, ...]]:
    if input is None:
        return None, ()
    if isinstance(input, OrganelleResult):
        verified = tuple(
            _verified_input_artifact(
                artifact,
                capability_id=capability_id,
                parameter=f"input.artifacts[{index}]",
            )
            for index, artifact in enumerate(input.artifacts)
        )
        normalized: CoreObject = input.model_copy(update={"artifacts": verified})
    elif isinstance(input, OrganelleData):
        verified_items = {
            name: _verified_input_artifact(
                artifact,
                capability_id=capability_id,
                parameter=f"input.artifacts.{name}",
            )
            for name, artifact in input.artifacts.items()
        }
        verified = tuple(verified_items.values())
        normalized = input.model_copy(update={"artifacts": FrozenMap.from_items(verified_items)})
    else:
        assert isinstance(input, OrganelleGenome)
        sequence = (
            _verified_input_artifact(
                input.sequence,
                capability_id=capability_id,
                parameter="input.sequence",
            )
            if input.sequence is not None
            else None
        )
        annotation = (
            _verified_input_artifact(
                input.annotation,
                capability_id=capability_id,
                parameter="input.annotation",
            )
            if input.annotation is not None
            else None
        )
        manifests = tuple(
            _verified_input_artifact(
                artifact,
                capability_id=capability_id,
                parameter=f"input.source_manifests[{index}]",
            )
            for index, artifact in enumerate(input.source_manifests)
        )
        verified = tuple(
            artifact for artifact in (sequence, annotation, *manifests) if artifact is not None
        )
        normalized = input.model_copy(
            update={
                "sequence": sequence,
                "annotation": annotation,
                "source_manifests": manifests,
            }
        )
    if normalized.object_id != input.object_id:
        raise OrganelleContractError(
            code="capability.input_identity_changed",
            message="resolving worker input artifact locations changed L1 identity",
            details={"capability_id": capability_id, "input_object_id": input.object_id},
        )
    return normalized, tuple(artifact.sha256 for artifact in verified)


def _snapshot_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    capability_id: str,
    parameter: str,
) -> Path:
    import hashlib

    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        destination.chmod(0o600 if _WINDOWS else 0o400)
    except OSError as error:
        raise OrganelleInputError(
            code="input.snapshot_failed",
            message="worker input artifact could not be snapshotted",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "source": str(source),
                "destination": str(destination),
                "reason": str(error),
            },
        ) from error
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256 or size_bytes != expected_size_bytes:
        raise OrganelleInputError(
            code="input.artifact_digest_mismatch",
            message="worker input artifact changed while its snapshot was created",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "path": str(source),
                "declared_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
                "declared_size_bytes": expected_size_bytes,
                "actual_size_bytes": size_bytes,
            },
        )
    return destination


def _snapshot_directory(
    source: Path,
    destination: Path,
    *,
    expected_manifest_hash: str,
    capability_id: str,
    parameter: str,
) -> Path:
    """Copy and freeze a content-addressed plugin directory input."""

    try:
        shutil.copytree(source, destination, symlinks=False)
        actual_manifest_hash = _directory_manifest_hash(destination)
        if actual_manifest_hash != expected_manifest_hash:
            raise OrganelleInputError(
                code="input.artifact_digest_mismatch",
                message="worker directory input changed while its snapshot was created",
                details={
                    "operation_id": capability_id,
                    "parameter": parameter,
                    "source": str(source),
                    "expected_manifest_hash": expected_manifest_hash,
                    "actual_manifest_hash": actual_manifest_hash,
                },
            )
        if not _WINDOWS:
            for current, directory_names, file_names in os.walk(destination):
                root = Path(current)
                for name in file_names:
                    (root / name).chmod(0o400)
                for name in directory_names:
                    (root / name).chmod(0o500)
            destination.chmod(0o500)
    except (OSError, shutil.Error) as error:
        raise OrganelleInputError(
            code="input.snapshot_failed",
            message="worker directory input could not be snapshotted",
            details={
                "operation_id": capability_id,
                "parameter": parameter,
                "source": str(source),
                "destination": str(destination),
                "reason": str(error),
            },
        ) from error
    return destination


def _directory_manifest_evidence(root: Path) -> tuple[str, tuple[str, ...]]:
    """Return a tree manifest hash plus every member-file content hash."""

    entries: list[dict[str, object]] = []
    try:
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                path = current_path / name
                if path.is_symlink():
                    raise OrganelleInputError(
                        code="input.unsafe_directory_artifact",
                        message="worker directory inputs may not contain symbolic links",
                        details={"path": str(path)},
                    )
            for name in file_names:
                path = current_path / name
                if path.is_symlink() or not path.is_file():
                    raise OrganelleInputError(
                        code="input.unsafe_directory_artifact",
                        message="worker directory inputs must contain regular files only",
                        details={"path": str(path)},
                    )
                entries.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
    except OSError as error:
        raise OrganelleInputError(
            code="input.unreadable_artifact",
            message="worker directory input could not be hashed",
            details={"path": str(root), "reason": str(error)},
        ) from error
    manifest_hash = hashlib.sha256(_canonical_json(entries)).hexdigest()
    member_hashes = tuple(cast(str, entry["sha256"]) for entry in entries)
    return manifest_hash, member_hashes


def _directory_manifest_hash(root: Path) -> str:
    return _directory_manifest_evidence(root)[0]


def _snapshot_artifact(
    artifact: ArtifactRef,
    snapshot_root: Path,
    *,
    index: int,
    capability_id: str,
    parameter: str,
) -> tuple[ArtifactRef, tuple[Path, str, int]]:
    source = Path(artifact.uri)
    destination = snapshot_root / _snapshot_name("core", index, source)
    _snapshot_file(
        source,
        destination,
        expected_sha256=artifact.sha256,
        expected_size_bytes=artifact.size_bytes,
        capability_id=capability_id,
        parameter=parameter,
    )
    return artifact.model_copy(update={"uri": str(destination)}), (
        destination,
        artifact.sha256,
        artifact.size_bytes,
    )


def _snapshot_name(prefix: str, index: int, source: Path) -> str:
    """Keep a useful suffix without inheriting an unbounded source basename."""

    suffix = source.suffix[:32]
    return f"{prefix}-{index:04d}{suffix}"


def _snapshot_worker_input(
    input: CoreObject | None,
    snapshot_root: Path,
    *,
    capability_id: str,
) -> tuple[CoreObject | None, tuple[tuple[Path, str, int], ...]]:
    if input is None:
        return None, ()
    records: list[tuple[Path, str, int]] = []

    def snapshot(artifact: ArtifactRef, parameter: str) -> ArtifactRef:
        copied, record = _snapshot_artifact(
            artifact,
            snapshot_root,
            index=len(records),
            capability_id=capability_id,
            parameter=parameter,
        )
        records.append(record)
        return copied

    if isinstance(input, OrganelleResult):
        normalized: CoreObject = input.model_copy(
            update={
                "artifacts": tuple(
                    snapshot(artifact, f"input.artifacts[{index}]")
                    for index, artifact in enumerate(input.artifacts)
                )
            }
        )
    elif isinstance(input, OrganelleData):
        normalized = input.model_copy(
            update={
                "artifacts": FrozenMap.from_items(
                    {
                        name: snapshot(artifact, f"input.artifacts.{name}")
                        for name, artifact in input.artifacts.items()
                    }
                )
            }
        )
    else:
        assert isinstance(input, OrganelleGenome)
        sequence = (
            snapshot(input.sequence, "input.sequence") if input.sequence is not None else None
        )
        annotation = (
            snapshot(input.annotation, "input.annotation") if input.annotation is not None else None
        )
        manifests = tuple(
            snapshot(artifact, f"input.source_manifests[{index}]")
            for index, artifact in enumerate(input.source_manifests)
        )
        normalized = input.model_copy(
            update={
                "sequence": sequence,
                "annotation": annotation,
                "source_manifests": manifests,
            }
        )
    if normalized.object_id != input.object_id:
        raise OrganelleContractError(
            code="capability.input_identity_changed",
            message="snapshotting worker input artifact locations changed L1 identity",
            details={"capability_id": capability_id, "input_object_id": input.object_id},
        )
    return normalized, tuple(records)


def _verify_input_snapshots(
    records: tuple[tuple[Path, str, int], ...],
    *,
    capability_id: str,
    directory_records: tuple[tuple[Path, str], ...] = (),
) -> None:
    for path, expected_sha256, expected_size_bytes in records:
        try:
            actual_sha256 = _sha256_file(path)
            actual_size_bytes = path.stat().st_size
        except OSError as error:
            raise OrganelleContractError(
                code="capability.input_snapshot_modified",
                message="worker input snapshot disappeared or became unreadable",
                details={
                    "capability_id": capability_id,
                    "path": str(path),
                    "reason": str(error),
                },
            ) from error
        if actual_sha256 != expected_sha256 or actual_size_bytes != expected_size_bytes:
            raise OrganelleContractError(
                code="capability.input_snapshot_modified",
                message="worker input snapshot changed during execution",
                details={
                    "capability_id": capability_id,
                    "path": str(path),
                    "expected_sha256": expected_sha256,
                    "actual_sha256": actual_sha256,
                    "expected_size_bytes": expected_size_bytes,
                    "actual_size_bytes": actual_size_bytes,
                },
            )
    for path, expected_manifest_hash in directory_records:
        try:
            actual_manifest_hash = _directory_manifest_hash(path)
        except OrganelleInputError as error:
            raise OrganelleContractError(
                code="capability.input_snapshot_modified",
                message="worker directory input snapshot disappeared or became unreadable",
                details={
                    "capability_id": capability_id,
                    "path": str(path),
                    "reason": error.message,
                },
            ) from error
        if actual_manifest_hash != expected_manifest_hash:
            raise OrganelleContractError(
                code="capability.input_snapshot_modified",
                message="worker directory input snapshot changed during execution",
                details={
                    "capability_id": capability_id,
                    "path": str(path),
                    "expected_manifest_hash": expected_manifest_hash,
                    "actual_manifest_hash": actual_manifest_hash,
                },
            )


def _require_current_execution_identity(
    entry: CapabilityEntry,
    expected: ExecutionIdentity,
    locator: str,
) -> None:
    current_bundle_hash = hash_bundle(entry.bundle_root)
    if current_bundle_hash != entry.content_hash:
        raise OrganelleContractError(
            code="capability.execution_identity_drift",
            message="capability bundle content changed after admission",
            details={
                "capability_id": entry.capability_id,
                "expected_bundle_hash": entry.content_hash,
                "current_bundle_hash": current_bundle_hash,
            },
        )
    current = inspect_bundle_code(
        entry.bundle_root,
        capability_id=entry.capability_id,
        bundle_content_hash=current_bundle_hash,
        callable_locator=locator,
    )
    if current != expected:
        raise OrganelleContractError(
            code="capability.execution_identity_drift",
            message="capability execution identity changed after admission",
            details={
                "capability_id": entry.capability_id,
                "expected_execution_digest": expected.digest,
                "current_execution_digest": current.digest,
            },
        )


class WorkerInvocationStrategy:
    """Cached Registry strategy with fresh identity and trust checks per dispatch."""

    def __init__(
        self,
        entry: CapabilityEntry,
        executor: BundleWorkerExecutor,
        trust_store: TrustStore,
        *,
        mount_map: dict[str, str] | None = None,
    ) -> None:
        self._entry = entry
        self._executor = executor
        self._trust_store = trust_store
        self._mount_map = mount_map if mount_map is not None else _load_mount_map() or {}

    def invoke(
        self,
        input: CoreObject | None,
        parameters: Mapping[str, object],
    ) -> CoreObject:
        entry = self._entry
        expected = entry.execution_identity
        if expected is None:
            raise OrganelleContractError(
                code="capability.execution_provider_required",
                message="bundle-local strategy has no execution identity",
                details={"capability_id": entry.capability_id},
            )
        locator = entry.bundle.contract.callable_locator
        assert locator is not None
        target_context = getattr(
            entry.bundle.contract, "execution_context", ExecutionContext.HOST
        )
        if target_context is not ExecutionContext.HOST:
            raise OrganelleExecutionError(
                code="capability.execution_context_unavailable",
                message=(
                    f"controlled worker execution context {target_context.value!r} "
                    "is not available"
                ),
                details={
                    "capability_id": entry.capability_id,
                    "requested_context": target_context.value,
                    "available_contexts": [ExecutionContext.HOST.value],
                },
            )

        worker_parameters: dict[str, object] = {}
        artifact_hashes: list[str] = []
        publication_input_hashes: list[str] = []
        path_inputs: dict[str, tuple[Path, str, int]] = {}
        directory_inputs: dict[str, tuple[Path, str]] = {}
        inline_inputs: dict[str, InlineParameterValue] = {}
        path_translations: list[PathTranslationRecord] = []
        inline_materializations: list[InlineMaterializationRecord] = []
        bindings = {item.name: item for item in entry.bundle.contract.binding.parameters}
        try:
            for name, value in parameters.items():
                binding = bindings[name]
                if isinstance(value, PathParameterValue):
                    accepted = getattr(binding, "accepts", ("path",))
                    if "path" not in accepted:
                        raise OrganelleContractError(
                            code="capability.path_not_accepted",
                            message=f"parameter {name!r} does not accept path input",
                            details={
                                "capability_id": entry.capability_id,
                                "parameter": name,
                                "accepts": accepted,
                            },
                        )
                    if binding.codec not in {ParameterCodec.PATH, ParameterCodec.DIRECTORY}:
                        raise OrganelleContractError(
                            code="capability.path_codec_unsupported",
                            message=(
                                f"path input for parameter {name!r} is not supported "
                                f"with codec {binding.codec.value!r}"
                            ),
                            details={
                                "capability_id": entry.capability_id,
                                "parameter": name,
                                "codec": binding.codec.value,
                            },
                        )
                    if value.context != target_context:
                        translated = translate_path(
                            value.value,
                            source_context=value.context,
                            target_context=target_context,
                            mount_map=self._mount_map,
                        )
                        worker_path = translated.translated_value
                        path_translations.append(
                            PathTranslationRecord(
                                parameter=name,
                                original_value=value.value,
                                source_context=value.context.value,
                                target_context=target_context.value,
                                translated_value=translated.translated_value,
                            )
                        )
                    else:
                        worker_path = value.value
                    if binding.codec is ParameterCodec.DIRECTORY:
                        host_path = _host_path_for_value(
                            value,
                            mount_map=self._mount_map,
                            capability_id=entry.capability_id,
                        )
                        resolved_directory = _resolved_worker_directory(
                            host_path,
                            capability_id=entry.capability_id,
                            parameter=name,
                        )
                        directory_manifest_hash, member_hashes = _directory_manifest_evidence(
                            resolved_directory
                        )
                        directory_inputs[name] = (
                            resolved_directory,
                            directory_manifest_hash,
                        )
                        artifact_hashes.append(directory_manifest_hash)
                        publication_input_hashes.extend(member_hashes)
                        worker_parameters[name] = worker_path
                    else:
                        host_path = _host_path_for_value(
                            value,
                            mount_map=self._mount_map,
                            capability_id=entry.capability_id,
                        )
                        resolved = _resolved_worker_path(
                            host_path,
                            capability_id=entry.capability_id,
                            parameter=name,
                        )
                        try:
                            artifact_sha256 = _sha256_file(resolved)
                            artifact_size_bytes = resolved.stat().st_size
                        except OSError as error:
                            raise OrganelleInputError(
                                code="input.unreadable_artifact",
                                message=f"artifact could not be read: {resolved}",
                                details={
                                    "operation_id": entry.capability_id,
                                    "parameter": name,
                                    "path": str(resolved),
                                    "reason": str(error),
                                },
                            ) from error
                        artifact_hashes.append(artifact_sha256)
                        publication_input_hashes.append(artifact_sha256)
                        path_inputs[name] = (resolved, artifact_sha256, artifact_size_bytes)
                        worker_parameters[name] = worker_path
                elif isinstance(value, InlineParameterValue):
                    accepted = getattr(binding, "accepts", ("path",))
                    if "inline" not in accepted:
                        raise OrganelleContractError(
                            code="capability.inline_not_accepted",
                            message=f"parameter {name!r} does not accept inline content",
                            details={
                                "capability_id": entry.capability_id,
                                "parameter": name,
                                "accepts": accepted,
                            },
                        )
                    inline_inputs[name] = value
                elif (
                    entry.bundle.contract.binding.argument_mode is ArgumentMode.PLUGIN_PROTOCOL
                    and binding.codec is ParameterCodec.DIRECTORY
                    and binding.path_role == "input"
                ):
                    accepted = getattr(binding, "accepts", ("path",))
                    if "path" not in accepted:
                        raise OrganelleContractError(
                            code="capability.path_not_accepted",
                            message=f"parameter {name!r} does not accept path input",
                            details={
                                "capability_id": entry.capability_id,
                                "parameter": name,
                                "accepts": accepted,
                            },
                        )
                    if not isinstance(value, str):
                        raise TypeError("validated directory parameter is not a string")
                    resolved_directory = _resolved_worker_directory(
                        value,
                        capability_id=entry.capability_id,
                        parameter=name,
                    )
                    directory_manifest_hash, member_hashes = _directory_manifest_evidence(
                        resolved_directory
                    )
                    directory_inputs[name] = (
                        resolved_directory,
                        directory_manifest_hash,
                    )
                    artifact_hashes.append(directory_manifest_hash)
                    publication_input_hashes.extend(member_hashes)
                    worker_parameters[name] = str(resolved_directory)
                elif binding.codec is ParameterCodec.PATH and value is not None:
                    accepted = getattr(binding, "accepts", ("path",))
                    if "path" not in accepted:
                        raise OrganelleContractError(
                            code="capability.path_not_accepted",
                            message=f"parameter {name!r} does not accept path input",
                            details={
                                "capability_id": entry.capability_id,
                                "parameter": name,
                                "accepts": accepted,
                            },
                        )
                    if not isinstance(value, str):
                        raise TypeError("validated path parameter is not a string")
                    source_context = ExecutionContext.HOST
                    if source_context != target_context:
                        translated = translate_path(
                            value,
                            source_context=source_context,
                            target_context=target_context,
                            mount_map=self._mount_map,
                        )
                        worker_parameters[name] = translated.translated_value
                        path_translations.append(
                            PathTranslationRecord(
                                parameter=name,
                                original_value=value,
                                source_context=source_context.value,
                                target_context=target_context.value,
                                translated_value=translated.translated_value,
                            )
                        )
                    else:
                        worker_parameters[name] = value
                    resolved = _resolved_worker_path(
                        value,
                        capability_id=entry.capability_id,
                        parameter=name,
                    )
                    try:
                        artifact_sha256 = _sha256_file(resolved)
                        artifact_size_bytes = resolved.stat().st_size
                    except OSError as error:
                        raise OrganelleInputError(
                            code="input.unreadable_artifact",
                            message=f"artifact could not be read: {resolved}",
                            details={
                                "operation_id": entry.capability_id,
                                "parameter": name,
                                "path": str(resolved),
                                "reason": str(error),
                            },
                        ) from error
                    artifact_hashes.append(artifact_sha256)
                    publication_input_hashes.append(artifact_sha256)
                    path_inputs[name] = (resolved, artifact_sha256, artifact_size_bytes)
                else:
                    worker_parameters[name] = _canonical_parameter_value(value)

            parameters_digest = _parameters_hash(parameters)
        except (RecursionError, TypeError, ValueError) as error:
            raise _worker_parameter_error(entry.capability_id, error) from error
        _require_current_execution_identity(entry, expected, locator)
        if not self._trust_store.is_trusted(expected.digest):
            raise OrganellePermissionError(
                code="capability.untrusted",
                message=f"capability bundle is not trusted: {entry.capability_id}",
                details={
                    "capability_id": entry.capability_id,
                    "bundle_content_hash": entry.content_hash,
                    "execution_digest": expected.digest,
                },
            )

        verified_input, core_artifact_hashes = _verified_worker_input(
            input,
            capability_id=entry.capability_id,
        )
        request_run_id = uuid4().hex
        publication_run_id = uuid4().hex
        while publication_run_id == request_run_id:  # pragma: no cover - UUID collision guard
            publication_run_id = uuid4().hex
        staging = create_staged_run(entry.capability_id, request_run_id)
        snapshot_root = staging.with_name(f".inputs-{request_run_id}")
        completed = managed_run_path(entry.capability_id, publication_run_id)
        input_object_ids = (input.object_id,) if input is not None else ()
        provenance_input_hashes = frozenset((*artifact_hashes, *core_artifact_hashes))
        trusted_input_hashes = frozenset(
            (*publication_input_hashes, *core_artifact_hashes)
        )
        snapshot_owned = False
        context: CapabilityExecutionContext | None = None
        snapshot_records: list[tuple[Path, str, int]] = []
        directory_snapshot_records: list[tuple[Path, str]] = []
        try:
            try:
                snapshot_root.mkdir(mode=0o700)
                snapshot_owned = True
            except FileExistsError as error:
                raise OrganelleInputError(
                    code="capability.input_snapshot_conflict",
                    message="controlled worker input snapshot path already exists",
                    details={
                        "capability_id": entry.capability_id,
                        "path": str(snapshot_root),
                    },
                ) from error
            except OSError as error:
                raise OrganelleExecutionError(
                    code="capability.input_snapshot_create_failed",
                    message="controlled worker input snapshot root could not be created",
                    details={
                        "capability_id": entry.capability_id,
                        "path": str(snapshot_root),
                        "reason": str(error),
                    },
                ) from error
            if inline_inputs:
                inline_raw_dir = snapshot_root / ".inline-raw"
                try:
                    inline_raw_dir.mkdir(parents=True, exist_ok=True)
                except OSError as error:
                    raise OrganelleExecutionError(
                        code="capability.inline_snapshot_create_failed",
                        message="controlled worker inline staging directory could not be created",
                        details={
                            "capability_id": entry.capability_id,
                            "path": str(inline_raw_dir),
                            "reason": str(error),
                        },
                    ) from error
                for name, value in sorted(inline_inputs.items()):
                    binding = bindings[name]
                    inline_max_bytes = getattr(binding, "inline_max_bytes", 1_048_576)
                    if binding.codec is ParameterCodec.JSON:
                        parsed, size_bytes = _decode_inline_json(
                            value,
                            parameter_name=name,
                            capability_id=entry.capability_id,
                            inline_max_bytes=inline_max_bytes,
                        )
                        _validate_inline_json_schema(
                            parsed,
                            binding=binding,
                            capability_id=entry.capability_id,
                            parameter_name=name,
                        )
                        artifact_hashes.append(value.sha256)
                        publication_input_hashes.append(value.sha256)
                        worker_parameters[name] = parsed
                        inline_materializations.append(
                            InlineMaterializationRecord(
                                parameter=name,
                                format=value.format,
                                sha256=value.sha256,
                                size_bytes=size_bytes,
                            )
                        )
                    elif binding.codec is ParameterCodec.PATH:
                        raw_path = materialize_inline_parameter(
                            value,
                            inline_raw_dir,
                            parameter_name=name,
                            capability_id=entry.capability_id,
                            inline_max_bytes=inline_max_bytes,
                        )
                        resolved = _resolved_worker_path(
                            str(raw_path),
                            capability_id=entry.capability_id,
                            parameter=name,
                        )
                        try:
                            artifact_sha256 = _sha256_file(resolved)
                            artifact_size_bytes = resolved.stat().st_size
                        except OSError as error:
                            raise OrganelleInputError(
                                code="input.unreadable_artifact",
                                message=f"inline artifact could not be read: {resolved}",
                                details={
                                    "operation_id": entry.capability_id,
                                    "parameter": name,
                                    "path": str(resolved),
                                    "reason": str(error),
                                },
                            ) from error
                        artifact_hashes.append(artifact_sha256)
                        publication_input_hashes.append(artifact_sha256)
                        path_inputs[name] = (resolved, artifact_sha256, artifact_size_bytes)
                        inline_materializations.append(
                            InlineMaterializationRecord(
                                parameter=name,
                                format=value.format,
                                sha256=value.sha256,
                                size_bytes=artifact_size_bytes,
                            )
                        )
                    else:
                        raise OrganelleContractError(
                            code="capability.inline_codec_unsupported",
                            message=(
                                f"inline content for parameter {name!r} is not supported "
                                f"with codec {binding.codec.value!r}"
                            ),
                            details={
                                "capability_id": entry.capability_id,
                                "parameter": name,
                                "codec": binding.codec.value,
                            },
                        )
            # Inline PATH values only become verified artifacts during the
            # bounded materialization step above. Freeze the publication and
            # provenance trust set after that step so their hashes cannot be
            # omitted from either channel.
            provenance_input_hashes = frozenset((*artifact_hashes, *core_artifact_hashes))
            trusted_input_hashes = frozenset(
                (*publication_input_hashes, *core_artifact_hashes)
            )
            for index, name in enumerate(sorted(path_inputs)):
                source, expected_sha256, expected_size_bytes = path_inputs[name]
                destination = snapshot_root / _snapshot_name("path", index, source)
                _snapshot_file(
                    source,
                    destination,
                    expected_sha256=expected_sha256,
                    expected_size_bytes=expected_size_bytes,
                    capability_id=entry.capability_id,
                    parameter=name,
                )
                worker_parameters[name] = str(destination)
                snapshot_records.append((destination, expected_sha256, expected_size_bytes))
            for index, name in enumerate(sorted(directory_inputs)):
                source, expected_manifest_hash = directory_inputs[name]
                destination = snapshot_root / f"directory-{index:04d}"
                _snapshot_directory(
                    source,
                    destination,
                    expected_manifest_hash=expected_manifest_hash,
                    capability_id=entry.capability_id,
                    parameter=name,
                )
                worker_parameters[name] = str(destination)
                directory_snapshot_records.append((destination, expected_manifest_hash))
            worker_input, core_snapshot_records = _snapshot_worker_input(
                verified_input,
                snapshot_root,
                capability_id=entry.capability_id,
            )
            snapshot_records.extend(core_snapshot_records)
            context = CapabilityExecutionContext(
                operation_id=entry.capability_id,
                operation_version=entry.bundle.contract.contract_version,
                callable_locator=locator,
                run_id=publication_run_id,
                input_object_ids=input_object_ids,
                input_artifact_hashes=tuple(sorted(provenance_input_hashes)),
                parameters_hash=parameters_digest,
                path_translations=tuple(path_translations),
                inline_materializations=tuple(inline_materializations),
            )
            _require_current_execution_identity(entry, expected, locator)
            worker_result = self._executor.invoke(
                entry,
                input=worker_input,
                parameters=worker_parameters,
                run_id=request_run_id,
                staging_root=staging,
            )
            _verify_input_snapshots(
                tuple(snapshot_records),
                capability_id=entry.capability_id,
                directory_records=tuple(directory_snapshot_records),
            )
            _require_current_execution_identity(entry, expected, locator)
            if worker_result.execution_identity != expected:
                raise OrganelleContractError(
                    code="capability.worker_identity_mismatch",
                    message="worker invocation identity does not match the trusted bundle",
                    details={"capability_id": entry.capability_id},
                )
            if entry.bundle.contract.binding.argument_mode is ArgumentMode.PLUGIN_PROTOCOL:
                encoded = encode_plugin_worker_result(worker_result.value, context, staging)
            else:
                if worker_result.artifact_paths:
                    raise OrganelleContractError(
                        code="capability.worker_protocol_invalid",
                        message="embedded ArtifactRef objects are the sole artifact declaration channel",
                        details={
                            "capability_id": entry.capability_id,
                            "artifact_paths": list(worker_result.artifact_paths),
                        },
                    )
                _reject_pre_encode_publication(worker_result.value, entry)
                encoded = encode_capability_result(
                    worker_result.value,
                    entry.bundle.contract.binding,
                    context,
                    output_kind=entry.bundle.contract.output_kind,
                )
            normalized = normalize_worker_core_object(encoded, context, input=input)
            if trusted_input_hashes:
                return publish_staged_result(
                    normalized,
                    staging,
                    completed,
                    trusted_input_hashes=trusted_input_hashes,
                )
            return publish_staged_result(normalized, staging, completed)
        except OrganelleExecutionError as error:
            if (
                context is None
                or entry.bundle.contract.binding.argument_mode is not ArgumentMode.PLUGIN_PROTOCOL
                or error.code != "capability.plugin_execution_failed"
            ):
                raise
            _verify_input_snapshots(
                tuple(snapshot_records),
                capability_id=entry.capability_id,
                directory_records=tuple(directory_snapshot_records),
            )
            _require_current_execution_identity(entry, expected, locator)
            failed = encode_plugin_worker_failure(error, context, staging)
            if trusted_input_hashes:
                return publish_staged_result(
                    failed,
                    staging,
                    completed,
                    trusted_input_hashes=trusted_input_hashes,
                )
            return publish_staged_result(failed, staging, completed)
        finally:
            _cleanup_owned_staging(staging)
            if snapshot_owned:
                _cleanup_owned_staging(snapshot_root)


__all__ = [
    "MAX_FRAME_BYTES",
    "MAX_STDERR_BYTES",
    "BundleWorkerExecutor",
    "OneShotBundleWorkerExecutor",
    "WorkerInvocationStrategy",
]
