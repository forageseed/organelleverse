"""One admitted Registry invocation path for GUI and Agent plugin runs."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from pydantic import JsonValue

from organelleverse import runtime
from organelleverse.capabilities.index import CapabilityIndex, CapabilityStatus
from organelleverse.capabilities.models import PluginCapabilityBundle
from organelleverse.capabilities.plugin_descriptor import PluginDescriptor, describe_plugin
from organelleverse.capabilities.trust import TrustStore
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.core.result import ErrorDetail, Finding, OrganelleResult
from organelleverse.operations import BoundOperation, OperationRegistry

from .models import (
    PluginRunArtifact,
    PluginRunEvent,
    PluginRunRecord,
    PluginRunRequest,
    RunContext,
    RunContextArtifact,
    RunContextDiagnostic,
    RunError,
)
from .projection import browser_safe_diagnostic_details, browser_safe_json_object
from .store import PluginRunStore

__all__ = ["PluginRunService"]

_TERMINAL = frozenset({"succeeded", "failed"})


def _error(code: str, capability_id: str) -> OrganelleContractError:
    messages = {
        "capability.not_admitted": "only admitted capabilities can run as plugins",
        "capability.plugin_required": "normal plugin runs require a v2 plugin capability bundle",
    }
    return OrganelleContractError(
        code=code,
        message=messages[code],
        details={"capability_id": capability_id},
    )


def _validate_request_fields(
    request: PluginRunRequest, descriptor: PluginDescriptor
) -> None:
    input_fields = {field.name: field for field in descriptor.inputs}
    parameter_fields = {field.name: field for field in descriptor.parameters}
    supplied_inputs = set(request.inputs)
    supplied_parameters = set(request.parameters)
    invalid_inputs = sorted(supplied_inputs - set(input_fields))
    invalid_parameters = sorted(supplied_parameters - set(parameter_fields))
    missing_inputs = sorted(
        name for name, field in input_fields.items() if field.required and name not in supplied_inputs
    )
    missing_parameters = sorted(
        name
        for name, field in parameter_fields.items()
        if field.required and name not in supplied_parameters
    )
    if invalid_inputs or invalid_parameters or missing_inputs or missing_parameters:
        raise OrganelleInputError(
            code="input.plugin_fields_invalid",
            message="plugin request fields do not match the admitted descriptor",
            details={
                "capability_id": request.capability_id,
                "invalid_inputs": invalid_inputs,
                "invalid_parameters": invalid_parameters,
                "missing_inputs": missing_inputs,
                "missing_parameters": missing_parameters,
            },
        )


class PluginRunService:
    """Own durable normal plugin execution and its browser-safe projection."""

    def __init__(
        self,
        *,
        index_provider: Callable[[], CapabilityIndex],
        trust_store: TrustStore,
        store: PluginRunStore,
    ) -> None:
        self._index_provider = index_provider
        self._trust_store = trust_store
        self._store = store
        self._lock = threading.Lock()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ov-plugin-run")

    def _emit(
        self,
        run_id: str,
        kind: Literal["stage", "tool", "log"],
        *,
        stage: str | None = None,
        percent: int | None = None,
        message: str | None = None,
    ) -> None:
        sequence = len(self._store.events(run_id))
        self._store.append_event(
            PluginRunEvent(
                event_id=f"ev-{uuid4().hex[:12]}",
                run_id=run_id,
                sequence=sequence,
                kind=kind,
                stage=stage,
                percent=percent,
                message=message,
                created_at=datetime.now(UTC),
            )
        )

    def _emit_stage(self, run_id: str, stage: str, message: str | None = "") -> None:
        self._emit(run_id, "stage", stage=stage, message=message or None)

    def events(self, run_id: str) -> tuple[PluginRunEvent, ...]:
        return self._store.events(run_id)

    def _prepare(self, request: PluginRunRequest) -> tuple[PluginRunRecord, BoundOperation]:
        index = self._index_provider()
        entry = index.describe(request.capability_id)
        if entry.status is not CapabilityStatus.ADMITTED:
            raise _error("capability.not_admitted", request.capability_id)
        if not isinstance(entry.bundle, PluginCapabilityBundle):
            raise _error("capability.plugin_required", request.capability_id)
        descriptor = describe_plugin(entry.bundle)
        _validate_request_fields(request, descriptor)
        registry = OperationRegistry(
            capability_source=index.binding_source(trust_store=self._trust_store)
        )
        operation = registry.require(request.capability_id)
        return (
            PluginRunRecord(
                run_id=f"run-{uuid4().hex}",
                l6_run_id=None,
                capability_id=request.capability_id,
                contract_identity=entry.content_hash,
                status="queued",
                submitted_at=datetime.now(UTC),
                request=request,
                descriptor=descriptor,
            ),
            operation,
        )

    def submit(self, request: PluginRunRequest) -> PluginRunRecord:
        with self._lock:
            self._require_open()
            record, operation = self._prepare(request)
            self._store.create(record)
            self._emit_stage(record.run_id, "queued")
            self._executor.submit(self._invoke, record, operation)
            return record

    def execute(self, request: PluginRunRequest) -> PluginRunRecord:
        with self._lock:
            self._require_open()
            record, operation = self._prepare(request)
            self._store.create(record)
            self._emit_stage(record.run_id, "queued")
        return self._invoke(record, operation)

    def _require_open(self) -> None:
        if self._closed:
            raise OrganelleContractError(
                code="plugin_run.service_closed",
                message="the plugin-run service is closed",
            )

    def _invoke(self, record: PluginRunRecord, operation: BoundOperation) -> PluginRunRecord:
        running = self._store.update(record.model_copy(update={"status": "running"}))
        if running.status in _TERMINAL:
            return running
        self._emit_stage(record.run_id, "running")
        try:
            result = operation.invoke(
                None, {**record.request.inputs, **record.request.parameters}
            )
            if not isinstance(result, OrganelleResult):
                raise OrganelleContractError(
                    code="contract.result_required",
                    message="plugin operation did not return an OrganelleResult",
                    details={"capability_id": record.capability_id},
                )
            if result.provenance is None or result.provenance.run_id is None:
                raise OrganelleContractError(
                    code="contract.result_provenance_required",
                    message="plugin result has no L6 execution run identity",
                    details={"capability_id": record.capability_id},
                )
            status: Literal["succeeded", "failed"] = (
                "failed" if result.status == "failed" else "succeeded"
            )
            completed = record.model_copy(
                update={
                    "l6_run_id": result.provenance.run_id,
                    "status": status,
                    "completed_at": datetime.now(UTC),
                    "result": result,
                    "summary": _summary(result),
                    "artifacts": _artifacts(result, record.descriptor),
                    "error": _result_error(result),
                }
            )
        except OrganelleError as error:
            completed = record.model_copy(
                update={
                    "status": "failed",
                    "completed_at": datetime.now(UTC),
                    "error": RunError.model_validate(error.as_dict()),
                }
            )
        except Exception as error:
            completed = record.model_copy(
                update={
                    "status": "failed",
                    "completed_at": datetime.now(UTC),
                    "error": RunError.model_validate(
                        OrganelleExecutionError(
                            code="plugin_run.execution_failed",
                            message="plugin execution failed unexpectedly",
                            details={
                                "exception_type": type(error).__name__,
                                # The runner's own words (missing binary, bad
                                # input…) — without them the failure is a
                                # riddle to the person watching the run.
                                "message": str(error)[:400] or type(error).__name__,
                            },
                        ).as_dict()
                    ),
                }
            )
        terminal = self._store.update(completed)
        failure_message = ""
        if terminal.status == "failed" and terminal.error is not None:
            details = terminal.error.details
            # Prefer the runner's own words over the generic wrapper message.
            detail_message = details.get("message")
            if detail_message:
                failure_message = str(detail_message)
            failure_message = failure_message or terminal.error.message
        self._emit_stage(
            record.run_id,
            terminal.status,
            message=failure_message or None,
        )
        return terminal

    def get(self, run_id: str) -> PluginRunRecord:
        return self._store.get(run_id)

    def list(self) -> tuple[PluginRunRecord, ...]:
        return self._store.list()

    def context(self, run_id: str) -> RunContext:
        record = self.get(run_id)
        if record.status not in _TERMINAL:
            raise OrganelleInputError(
                code="input.run_not_terminal",
                message=f"plugin run is not terminal: {run_id}",
                details={"run_id": run_id, "status": record.status},
            )
        if record.contract_identity is None:
            raise OrganelleInputError(
                code="run.context_unavailable",
                message="legacy plugin-run history has no admitted contract identity",
                details={"run_id": run_id},
            )
        result = record.result
        metrics = (
            {}
            if result is None
            else cast(
                dict[str, JsonValue],
                browser_safe_json_object(
                    cast(Mapping[str, object], result.model_dump(mode="json")["metrics"])
                ),
            )
        )
        return RunContext(
            run_id=record.run_id,
            l6_run_id=record.l6_run_id,
            capability_id=record.capability_id,
            contract_identity=record.contract_identity,
            status=cast(Literal["succeeded", "failed"], record.status),
            request=record.request,
            result_id=None if result is None else result.object_id,
            summary_text="" if result is None else _safe_summary(result.summary_text),
            metrics=metrics,
            findings=() if result is None else tuple(_finding(item) for item in result.findings),
            flags=() if result is None else result.flags,
            artifacts=_artifact_contexts(record),
            diagnostics=_diagnostics(record),
            provenance_links=_provenance_links(result),
        )

    def export(self, run_id: str, destination: Path) -> PluginRunRecord:
        record = self.get(run_id)
        if record.status not in _TERMINAL or record.result is None:
            raise OrganelleInputError(
                code="input.run_not_exportable",
                message=f"plugin run is not exportable: {run_id}",
                details={"run_id": run_id, "status": record.status},
            )
        result = runtime.promote_result(record.result, destination)
        return record.model_copy(
            update={"result": result, "artifacts": _artifacts(result, record.descriptor)}
        )

    def artifact_path(self, run_id: str, artifact_id: str) -> Path:
        record = self.get(run_id)
        if record.result is not None:
            for displayed, artifact in zip(
                record.artifacts, record.result.artifacts, strict=True
            ):
                if displayed.artifact_id == artifact_id:
                    return Path(artifact.uri)
        raise OrganelleInputError(
            code="input.unknown_artifact",
            message=f"unknown artifact: {artifact_id}",
            details={"run_id": run_id, "artifact_id": artifact_id},
        )

    def verifies(self, run_id: str) -> bool:
        try:
            return self.get(run_id).status in _TERMINAL
        except OrganelleInputError:
            return False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._store.interrupt_nonterminal()
            self._executor.shutdown(wait=False, cancel_futures=True)


def _summary(result: OrganelleResult) -> dict[str, JsonValue]:
    payload = result.model_dump(mode="json")
    return {
        "summary_text": cast(JsonValue, payload["summary_text"]),
        "metrics": cast(JsonValue, payload["metrics"]),
    }


def _artifacts(
    result: OrganelleResult, descriptor: PluginDescriptor
) -> tuple[PluginRunArtifact, ...]:
    outputs = {output.name for output in descriptor.outputs}
    projected: list[PluginRunArtifact] = []
    for number, artifact in enumerate(result.artifacts):
        if artifact.kind == "log":
            output_name: str | None = None
            role: Literal["output", "log", "evidence"] = "log"
        elif artifact.kind == "partial_output":
            output_name = None
            role = "evidence"
        elif artifact.kind in outputs:
            output_name = artifact.kind
            role = "output"
        elif result.status == "failed":
            output_name = None
            role = "evidence"
        else:
            output_name = None
            role = "log"
        projected.append(
            PluginRunArtifact(
                artifact_id=f"artifact-{number:04d}",
                output_name=output_name,
                role=role,
                format=artifact.format,
                media_type=artifact.media_type,
                size_bytes=artifact.size_bytes,
            )
        )
    return tuple(projected)


def _result_error(result: OrganelleResult) -> RunError | None:
    if result.status != "failed":
        return None
    return _error_detail(result.errors[0])


def _error_detail(error: ErrorDetail) -> RunError:
    payload = error.model_dump(mode="json", exclude={"object_id", "kind", "code"})
    return RunError.model_validate(payload | {"error_code": error.code})


def _safe_summary(summary: str) -> str:
    projected = browser_safe_json_object({"summary": summary})
    value = projected.get("summary", "")
    return value if isinstance(value, str) else ""


def _finding(finding: Finding) -> dict[str, JsonValue]:
    payload = finding.model_dump(mode="json", exclude={"object_id"})
    return cast(dict[str, JsonValue], browser_safe_json_object(payload))


def _artifact_contexts(record: PluginRunRecord) -> tuple[RunContextArtifact, ...]:
    if record.result is None:
        return ()
    return tuple(
        RunContextArtifact(
            object_id=artifact.object_id,
            artifact_id=displayed.artifact_id,
            output_name=displayed.output_name,
            role=displayed.role,
            format=displayed.format,
            media_type=displayed.media_type,
            size_bytes=displayed.size_bytes,
        )
        for displayed, artifact in zip(
            record.artifacts, record.result.artifacts, strict=True
        )
    )


def _diagnostics(record: PluginRunRecord) -> tuple[RunContextDiagnostic, ...]:
    if record.result is not None and record.result.errors:
        errors = tuple(_error_detail(error) for error in record.result.errors)
    elif record.error is not None:
        errors = (record.error,)
    else:
        errors = ()
    return tuple(
        RunContextDiagnostic(
            error_code=error.error_code,
            message=_safe_summary(error.message),
            details=cast(
                dict[str, JsonValue], browser_safe_diagnostic_details(error.details)
            ),
            retryable=error.retryable,
            suggested_action=cast(
                dict[str, JsonValue], browser_safe_json_object(error.suggested_action)
            ),
        )
        for error in errors
    )


def _provenance_links(result: OrganelleResult | None) -> tuple[str, ...]:
    if result is None or result.provenance is None:
        return ()
    provenance = result.provenance
    links = [provenance.object_id]
    if provenance.run_manifest_id is not None:
        links.append(provenance.run_manifest_id)
    links.extend(provenance.upstream_run_manifest_ids)
    return tuple(links)
