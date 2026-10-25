"""In-process adapter that routes legacy Tool UI runs through PluginRunService.

This is a single synthetic v2 plugin capability.  Every non-image ``ToolDef`` is
submitted as one invocation of ``tools.adapter``; the adapter receives the tool
metadata and any uploaded files and produces a durable OrganelleResult
artifact, exactly like a real plugin.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from organelleverse.capabilities.index import (
    CapabilityEntry,
    CapabilityIndex,
    CapabilityOrigin,
    CapabilityStatus,
)
from organelleverse.capabilities.models import (
    AgentMetadata,
    GuiMetadata,
    PluginCapabilityBundle,
    PluginMetadata,
)
from organelleverse.core.errors import OrganelleInputError
from organelleverse.operations.python_binding import bind_python_capability
from organelleverse.operations.schemas import parameter_schema
from organelleverse.operations.spec import (
    AgentSurface,
    ArgumentMode,
    CoreKind,
    ExecutionContext,
    ExecutionMode,
    OperationStage,
    ParameterCodec,
    ParameterSource,
    PluginBindingSpec,
    PluginOperationSpec,
    PluginParameterBindingSpec,
    ProgressSource,
    ResultCodec,
)

_CAPABILITY_ID = "tools.adapter"
_BUNDLE_VERSION = "1.2.0"


def run_tool(
    tool_id: str,
    tool_name: str,
    tool_contract_version: str,
    tool_content_hash: str,
    params_json: str,
    files_json: str,
    output_path: Path,
    tool_badge: str = "read",
    staged_files: list[dict[str, str]] | None = None,
) -> tuple[Path, ...]:
    """Execute a legacy tool request and materialize a report artifact.

    ``params_json`` and ``files_json`` are JSON-encoded objects supplied by
    the host's tool run service.  ``files_json``
    maps frontend field names to the local filesystem paths where uploaded bytes
    were staged.  ``output_path`` is the destination the artifact codec publishes.

    The adapter also receives the pinned ``tool_contract_version`` and
    ``tool_content_hash`` so the durable request records the exact tool schema
    used for the run. ``tool_badge`` preserves the registry's cost contract.

    Built-in tools with real local runners (``motif`` and the ``genecode_*``
    suite) are dispatched through ``BUILTIN_TOOL_RUNNERS`` so the plugin path
    publishes the report and each returned output through the artifact codec.
    A missing executable runner fails clearly before any success report.
    """
    from organelleverse.operations.execution_hooks import staged_input_copy_required

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = json.loads(params_json) if params_json else {}
    files: dict[str, str] = json.loads(files_json) if files_json else {}
    staged_by_field = {item["field_name"]: item["path"] for item in (staged_files or [])}
    if staged_input_copy_required():
        import shutil

        for name, source in staged_by_field.items():
            if not name.isidentifier():
                raise ValueError("invalid input field name")
            suffix = "".join(Path(files.get(name, source)).suffixes)
            destination = output_path.parent / "inputs" / f"{name}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            files[name] = str(destination)
    else:
        files.update(staged_by_field)

    # Resolve staged file paths for built-in runners. Uploaded files always
    # win over whatever text the params carried for that field (the form
    # echoes the file name into params).
    for field_name, file_path in files.items():
        params[field_name] = file_path

    log: list[str] = []

    # Imported here, not at module scope: the release registry attaches this
    # adapter while listing operations, and the runners pull in scikit-image
    # and other analysis stacks that listing must not load.
    from organelleverse.operations.builtin_tool_runners import BUILTIN_TOOL_RUNNERS

    runner_fn = BUILTIN_TOOL_RUNNERS.get(tool_id)
    if runner_fn is not None:
        result = runner_fn(params, output_dir=output_path.parent, log=log.append)
        summary_text = str(result.get("summary_text", f"{tool_name} completed."))
        lines = [
            f"OrganelleVerse Tool Run: {tool_name}",
            f"Tool ID: {tool_id}",
            "",
            summary_text,
            "",
            "Output files:",
        ]
        produced = []
        for file_label, file_path in (result.get("output_files") or {}).items():
            lines.append(f"  {file_label}: {file_path}")
            path = Path(file_path)
            produced.extend(sorted(path.rglob("*")) if path.is_dir() else [path])
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return (
            output_path,
            *dict.fromkeys(path for path in produced if path.is_file() and path != output_path),
        )

    raise OrganelleInputError(
        code="input.tool_runner_unavailable",
        message=f"No executable runner is registered for tool {tool_id!r}.",
        details={"tool_id": tool_id},
    )


def _adapter_bundle() -> PluginCapabilityBundle:
    """Build the synthetic v2 plugin capability bundle for the tool adapter."""
    keywords = ("adapter", "legacy", "plugin", "tool", "ui")
    return PluginCapabilityBundle(
        schema_version="organelleverse.capability.v2",
        capability={
            "id": _CAPABILITY_ID,
            "bundle_version": _BUNDLE_VERSION,
            "implementation": "native",
        },
        contract=PluginOperationSpec(
            operation_id=_CAPABILITY_ID,
            contract_version="1.2",
            title="OrganelleVerse Tool Adapter",
            description=(
                "Routes legacy Tool UI invocations through the plugin capability "
                "system, producing a durable report artifact for every non-image tool."
            ),
            keywords=keywords,
            execution_mode=ExecutionMode.DURABLE,
            stage=OperationStage.ANALYZE,
            input_kind=CoreKind.NONE,
            output_kind=CoreKind.RESULT,
            callable_locator="organelleverse.operations.tools_adapter:run_tool",
            deterministic=False,
            deterministic_reason=(
                "routes to heterogeneous backend implementations, external tool versions, "
                "and user-provided inputs"
            ),
            idempotent=False,
            cacheable=False,
            binding=PluginBindingSpec(
                argument_mode=ArgumentMode.PLUGIN_PROTOCOL,
                result_codec=ResultCodec.ARTIFACT,
                parameters=(
                    PluginParameterBindingSpec(
                        name="staged_files",
                        codec=ParameterCodec.JSON,
                        source=ParameterSource.AGENT,
                        json_schema={
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field_name": {"type": "string"},
                                    "path": {"type": "string"},
                                },
                                "required": ["field_name", "path"],
                                "additionalProperties": False,
                            },
                            "default": [],
                        },
                    ),
                    PluginParameterBindingSpec(
                        name="output_path",
                        codec=ParameterCodec.PATH,
                        source=ParameterSource.AGENT,
                        path_role="output",
                        accepts=("path",),
                    ),
                    PluginParameterBindingSpec(
                        name="tool_id",
                        codec=ParameterCodec.JSON,
                        source=ParameterSource.AGENT,
                    ),
                    PluginParameterBindingSpec(
                        name="tool_name",
                        codec=ParameterCodec.JSON,
                        source=ParameterSource.AGENT,
                    ),
                    PluginParameterBindingSpec(
                        name="tool_contract_version",
                        codec=ParameterCodec.JSON,
                        source=ParameterSource.AGENT,
                    ),
                    PluginParameterBindingSpec(
                        name="tool_content_hash",
                        codec=ParameterCodec.JSON,
                        source=ParameterSource.AGENT,
                    ),
                    PluginParameterBindingSpec(
                        name="tool_badge",
                        codec=ParameterCodec.JSON,
                        source=ParameterSource.AGENT,
                    ),
                    PluginParameterBindingSpec(
                        name="params_json",
                        codec=ParameterCodec.JSON,
                        source=ParameterSource.AGENT,
                    ),
                    PluginParameterBindingSpec(
                        name="files_json",
                        codec=ParameterCodec.JSON,
                        source=ParameterSource.AGENT,
                    ),
                ),
            ),
            outputs=(),
            execution_context=ExecutionContext.HOST,
            progress_source=ProgressSource.STAGES,
        ),
        plugin=PluginMetadata(
            name="Tool Adapter",
            version=_BUNDLE_VERSION,
            author="OrganelleVerse",
            summary="Adapter surface for legacy Tool UI runs.",
            category="general",
            badge="read",
        ),
        agent=AgentMetadata(
            task_description="Execute a legacy Tool UI request through the plugin system.",
            prompt="",
            agent_surface=AgentSurface.GRANULAR,
        ),
        gui=GuiMetadata(),
        probes=(),
        data_contracts=(),
        fixtures=(),
        composite=None,
    )


def _bundle_content_hash(bundle: PluginCapabilityBundle) -> str:
    canonical = json.dumps(
        bundle.model_dump(mode="json", by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _parameter_schema(bundle: PluginCapabilityBundle) -> dict[str, Any]:
    bound = bind_python_capability(bundle, run_tool, frozen_schema=None)
    return parameter_schema(bound)


def tool_adapter_entry() -> CapabilityEntry:
    """Return a synthetic, pre-admitted core capability entry for the adapter."""
    bundle = _adapter_bundle()
    bundle_root = Path(__file__).parent
    return CapabilityEntry(
        capability_id=_CAPABILITY_ID,
        content_hash=_bundle_content_hash(bundle),
        bundle_root=bundle_root,
        bundle=bundle,
        origins=(
            CapabilityOrigin(
                channel="core",
                source_path=str(bundle_root),
                search_root=None,
            ),
        ),
        status=CapabilityStatus.ADMITTED,
        parameter_schema=_parameter_schema(bundle),
        execution_identity=None,
        worker_parameters=None,
    )


def index_with_tool_adapter(
    index_provider: Callable[[], CapabilityIndex],
) -> Callable[[], CapabilityIndex]:
    """Wrap an index provider so it always includes the synthetic tool adapter."""

    def provide() -> CapabilityIndex:
        index = index_provider()
        if any(entry.capability_id == _CAPABILITY_ID for entry in index.entries):
            return index
        entry = tool_adapter_entry()
        return CapabilityIndex.model_validate(
            {
                "entries": (*index.entries, entry),
                "diagnostic_items": index.diagnostic_items,
                "conflict_items": index.conflict_items,
            }
        )

    return provide
