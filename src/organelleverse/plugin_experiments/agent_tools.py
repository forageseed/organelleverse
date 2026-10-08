"""Framework-neutral Agent tools projected from admitted capabilities.

The catalog is deliberately only a schema and delegation layer.  It neither
discovers capabilities itself nor imports an Agent framework: callers supply
the admitted snapshot and the same experiment service used elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, cast

from jsonschema import Draft202012Validator
from pydantic import JsonValue

from organelleverse.capabilities.capability_registry import build_registry
from organelleverse.capabilities.index import CapabilityEntry, CapabilityIndex, CapabilityStatus
from organelleverse.capabilities.models import PluginCapabilityBundle
from organelleverse.capabilities.plugin_descriptor import (
    PluginDescriptor,
    PluginField,
    describe_plugin,
)
from organelleverse.capabilities.trust import TrustStore
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.operations.adapters.json import invoke_json
from organelleverse.operations.registry import OperationRegistry
from organelleverse.operations.spec import SideEffect, StrictSpecModel
from organelleverse.optimization import (
    AutoOptimizationRequest,
    OptimizationProfileV3,
    contract_from_capability_entry,
    identity_from_capability_entry,
    optimization_request_schema,
)
from organelleverse.plugin_experiments.models import ExperimentRecord, ExperimentRequest
from organelleverse.plugin_experiments.service import ExperimentService
from organelleverse.plugin_runs import PluginRunRequest, PluginRunService
from organelleverse.plugin_runs.projection import (
    browser_safe_diagnostic_details,
    browser_safe_json_object,
)

if TYPE_CHECKING:
    from organelleverse.plugin_experiments.auto_service import AutoOptimizationService

__all__ = ["CapabilityAgentTool", "CapabilityAgentToolCatalog"]


def _empty_profiles() -> Mapping[str, OptimizationProfileV3]:
    return {}


class CapabilityAgentTool(StrictSpecModel):
    """One closed Agent-facing capability action."""

    name: str
    capability_id: str
    action: Literal["run", "optimize", "execute_best"]
    invocation_style: Literal["operation", "plugin"]
    description: str
    input_schema: dict[str, JsonValue]


class CapabilityAgentToolCatalog:
    """Project one admitted index snapshot into callable Agent tool descriptors."""

    def __init__(
        self,
        *,
        index_provider: Callable[[], CapabilityIndex],
        trust_store: TrustStore,
        experiment_service: ExperimentService,
        run_service: PluginRunService,
        granted_side_effects: Collection[SideEffect],
        profile_provider: Callable[[], Mapping[str, OptimizationProfileV3]] | None = None,
        auto_service: AutoOptimizationService | None = None,
        risk_gate_satisfied: bool = False,
    ) -> None:
        self._index_provider = index_provider
        self._trust_store = trust_store
        self._experiment_service = experiment_service
        self._run_service = run_service
        self._granted_side_effects = granted_side_effects
        self._profile_provider = profile_provider or _empty_profiles
        self._auto_service = auto_service
        self._risk_gate_satisfied = risk_gate_satisfied
        self._issued_tools: dict[str, CapabilityAgentTool] = {}

    def list(self) -> tuple[CapabilityAgentTool, ...]:
        """Return every callable admitted operation from one index snapshot."""
        index = self._index_provider()
        tools = self._tools(index)
        self._issued_tools = {tool.name: tool for tool in tools}
        return tools

    def invoke(
        self, name: str, arguments: Mapping[str, object]
    ) -> dict[str, object] | ExperimentRecord:
        """Validate then delegate one named tool without decoding its name."""
        index = self._index_provider()
        tool = self._issued_tools.get(name)
        if tool is None:
            tools = self._tools(index)
            self._issued_tools = {item.name: item for item in tools}
            tool = self._issued_tools.get(name)
        if tool is None:
            raise OrganelleInputError(
                code="input.unknown_agent_tool",
                message=f"unknown capability Agent tool: {name}",
                details={"name": name},
            )
        Draft202012Validator(tool.input_schema).validate(dict(arguments))  # pyright: ignore[reportUnknownMemberType]
        if tool.action == "execute_best":
            if self._matching_auto_profile(index, tool.capability_id) is None:
                raise self._unknown_tool(name)
            request = AutoOptimizationRequest.model_validate(
                {"capability_id": tool.capability_id, **arguments}
            )
            if request.invocation.kind != tool.invocation_style:
                raise OrganelleContractError(
                    code="optimization.invocation_surface_mismatch",
                    message="execute-best invocation does not match the projected capability surface",
                )
            assert self._auto_service is not None
            result = self._auto_service.execute(
                request, risk_gate_satisfied=self._risk_gate_satisfied
            )
            return browser_safe_json_object(result.model_dump(mode="json"))
        if tool.action == "run":
            if tool.invocation_style == "plugin":
                record = self._run_service.execute(
                    PluginRunRequest.model_validate(
                        {"capability_id": tool.capability_id, **arguments}
                    )
                )
                return {
                    "ok": record.status == "succeeded",
                    "operation_id": record.capability_id,
                    "run_id": record.run_id,
                    "l6_run_id": record.l6_run_id,
                    "status": record.status,
                    "result": (
                        None
                        if record.result is None
                        else self._agent_result_payload(record.result.model_dump(mode="json"))
                    ),
                }
            registry = self._registry(index)
            response = invoke_json(
                {
                    "operation_id": tool.capability_id,
                    "input": arguments.get("input"),
                    "parameters": arguments.get("parameters", {}),
                },
                registry=registry,
                granted_side_effects=self._granted_side_effects,
            )
            raw_result = response.get("result")
            if isinstance(raw_result, Mapping):
                response["result"] = self._agent_result_payload(
                    cast(Mapping[str, object], raw_result)
                )
            return response
        return self._experiment_service.submit(
            ExperimentRequest.model_validate({"capability_id": tool.capability_id, **arguments})
        )

    def _tools(self, index: CapabilityIndex) -> tuple[CapabilityAgentTool, ...]:
        registry = self._registry(index)
        entries = {entry.capability_id: entry for entry in index.entries}
        profiles: Mapping[str, OptimizationProfileV3] = (
            self._profile_provider() if self._auto_service is not None else {}
        )
        shadowed = set(registry.shadowed_capability_ids())
        tools: list[CapabilityAgentTool] = []
        names: set[str] = set()
        for operation in registry.list():
            entry = (
                None if operation.operation_id in shadowed else entries.get(operation.operation_id)
            )
            if entry is not None and not self._runnable_by_agent(entry):
                continue
            descriptor = self._descriptor(entry)
            if descriptor is None:
                tool = CapabilityAgentTool(
                    name=self._name("run", operation.operation_id, names),
                    capability_id=operation.operation_id,
                    action="run",
                    invocation_style="operation",
                    description=operation.description,
                    input_schema=self._operation_schema(registry, operation.operation_id),
                )
                tools.append(tool)
                auto = self._auto_tool(entry, profiles, operation.operation_id, "operation", names)
                if auto is not None:
                    tools.append(auto)
                continue
            tools.append(
                CapabilityAgentTool(
                    name=self._name("run", operation.operation_id, names),
                    capability_id=operation.operation_id,
                    action="run",
                    invocation_style="plugin",
                    description=descriptor.agent_task_description,
                    input_schema=self._plugin_run_schema(descriptor),
                )
            )
            if descriptor.optimization is not None:
                assert entry is not None
                tools.append(
                    CapabilityAgentTool(
                        name=self._name("optimize", operation.operation_id, names),
                        capability_id=operation.operation_id,
                        action="optimize",
                        invocation_style="plugin",
                        description=descriptor.agent_task_description,
                        input_schema=self._optimization_schema(descriptor, entry),
                    )
                )
            auto = self._auto_tool(entry, profiles, operation.operation_id, "plugin", names)
            if auto is not None:
                tools.append(auto)
        return tuple(tools)

    def _registry(self, index: CapabilityIndex) -> OperationRegistry:
        """Build the one authoritative Registry from releases plus this snapshot."""
        return build_registry(index, trust_store=self._trust_store)

    @staticmethod
    def _agent_result_payload(result: Mapping[str, object]) -> dict[str, object]:
        """Retain scientific artifact identity while never exposing local URIs."""
        payload: dict[str, object] = {}
        raw_artifacts = result.get("artifacts")
        for name, value in result.items():
            if name == "artifacts":
                continue
            if name == "errors" and isinstance(value, list):
                payload[name] = [
                    CapabilityAgentToolCatalog._agent_error_payload(
                        cast(Mapping[object, object], item)
                    )
                    for item in cast(list[object], value)
                    if isinstance(item, Mapping)
                ]
                continue
            projected = browser_safe_json_object({name: value})
            if name in projected:
                payload[name] = projected[name]
        if isinstance(raw_artifacts, list):
            artifacts = cast(list[object], raw_artifacts)
            payload["artifacts"] = [
                CapabilityAgentToolCatalog._agent_artifact_payload(
                    cast(Mapping[object, object], raw_artifact)
                )
                for raw_artifact in artifacts
                if isinstance(raw_artifact, Mapping)
            ]
        elif isinstance(raw_artifacts, Mapping):
            payload["artifacts"] = {
                str(name): CapabilityAgentToolCatalog._agent_artifact_payload(
                    cast(Mapping[object, object], artifact)
                )
                if isinstance(artifact, Mapping)
                else artifact
                for name, artifact in cast(Mapping[object, object], raw_artifacts).items()
            }
        return payload

    @staticmethod
    def _agent_error_payload(error: Mapping[object, object]) -> dict[str, object]:
        """Project one canonical error without the worker's private reason."""

        projected = browser_safe_json_object(
            {str(name): value for name, value in error.items() if name != "details"}
        )
        details = error.get("details")
        if isinstance(details, Mapping):
            projected["details"] = browser_safe_diagnostic_details(
                cast(Mapping[str, object], details)
            )
        return projected

    @staticmethod
    def _agent_artifact_payload(artifact: Mapping[object, object]) -> dict[str, object]:
        """Expose only opaque ArtifactRef identity and portable metadata."""
        projected = {str(name): value for name, value in artifact.items()}
        try:
            artifact_id = ArtifactRef.model_validate(projected).object_id
        except ValueError:
            artifact_id = None
        projected.pop("uri", None)
        projected.pop("path", None)
        projected.pop("object_id", None)
        safe_artifact = browser_safe_json_object(projected)
        if artifact_id is not None:
            safe_artifact["artifact_id"] = artifact_id
        return safe_artifact

    def _runnable_by_agent(self, entry: CapabilityEntry) -> bool:
        """The Agent cannot grant trust, so hide untrusted non-core bindings."""
        if all(origin.channel == "core" for origin in entry.origins):
            return True
        identity = entry.execution_identity
        return identity is not None and self._trust_store.is_trusted(identity.digest)

    @staticmethod
    def _descriptor(entry: CapabilityEntry | None) -> PluginDescriptor | None:
        if entry is not None and isinstance(entry.bundle, PluginCapabilityBundle):
            return describe_plugin(entry.bundle)
        return None

    @staticmethod
    def _name(
        action: Literal["run", "optimize", "execute_best"],
        capability_id: str,
        names: set[str],
    ) -> str:
        name = f"{action}__{capability_id.replace('.', '__')}"
        if name in names:
            raise OrganelleContractError(
                code="capability.agent_tool_name_conflict",
                message=f"capability IDs map to the same Agent tool name: {name}",
                details={"tool_name": name, "capability_id": capability_id},
            )
        names.add(name)
        return name

    def _auto_tool(
        self,
        entry: CapabilityEntry | None,
        profiles: Mapping[str, OptimizationProfileV3],
        capability_id: str,
        invocation_style: Literal["operation", "plugin"],
        names: set[str],
    ) -> CapabilityAgentTool | None:
        if entry is None or self._auto_service is None:
            return None
        profile = profiles.get(capability_id)
        if not self._profile_matches(entry, profile):
            return None
        return CapabilityAgentTool(
            name=self._name("execute_best", capability_id, names),
            capability_id=capability_id,
            action="execute_best",
            invocation_style=invocation_style,
            description=f"Execute the governed best configuration for {capability_id}.",
            input_schema=self._execute_best_schema(invocation_style),
        )

    def _matching_auto_profile(
        self, index: CapabilityIndex, capability_id: str
    ) -> OptimizationProfileV3 | None:
        if self._auto_service is None:
            return None
        entry = next((item for item in index.entries if item.capability_id == capability_id), None)
        profile = self._profile_provider().get(capability_id)
        return (
            profile
            if entry is not None
            and self._runnable_by_agent(entry)
            and self._profile_matches(entry, profile)
            else None
        )

    @staticmethod
    def _profile_matches(entry: CapabilityEntry, profile: OptimizationProfileV3 | None) -> bool:
        return (
            entry.status is CapabilityStatus.ADMITTED
            and profile is not None
            and profile.status in {"enabled", "eligible"}
            and profile.target == identity_from_capability_entry(entry)
        )

    @staticmethod
    def _execute_best_schema(
        invocation_style: Literal["operation", "plugin"],
    ) -> dict[str, JsonValue]:
        schema = cast(dict[str, JsonValue], deepcopy(AutoOptimizationRequest.model_json_schema()))
        properties = cast(dict[str, JsonValue], schema["properties"])
        properties.pop("capability_id")
        required = cast(list[str], schema["required"])
        schema["required"] = [name for name in required if name != "capability_id"]
        invocation = cast(dict[str, JsonValue], properties["invocation"])
        reference = f"#/$defs/{'PluginInvocation' if invocation_style == 'plugin' else 'OperationInvocation'}"
        invocation["oneOf"] = [{"$ref": reference}]
        discriminator = cast(dict[str, JsonValue], invocation["discriminator"])
        discriminator["mapping"] = {invocation_style: reference}
        return schema

    @staticmethod
    def _unknown_tool(name: str) -> OrganelleInputError:
        return OrganelleInputError(
            code="input.unknown_agent_tool",
            message=f"unknown capability Agent tool: {name}",
            details={"name": name},
        )

    @staticmethod
    def _operation_schema(registry: OperationRegistry, operation_id: str) -> dict[str, JsonValue]:
        schema = cast(dict[str, JsonValue], deepcopy(registry.invocation_schema(operation_id)))
        properties = cast(dict[str, JsonValue], schema["properties"])
        properties.pop("operation_id")
        required = cast(list[str], schema["required"])
        schema["required"] = [name for name in required if name != "operation_id"]
        return schema

    @classmethod
    def _plugin_run_schema(cls, descriptor: PluginDescriptor) -> dict[str, JsonValue]:
        schema: dict[str, JsonValue] = {
            "type": "object",
            "properties": {
                "inputs": cls._fields_schema(descriptor.inputs),
                "parameters": cls._fields_schema(descriptor.parameters),
            },
            "required": ["inputs"],
            "additionalProperties": False,
        }
        if descriptor.agent_examples:
            schema["examples"] = list(descriptor.agent_examples)
        return schema

    @classmethod
    def _optimization_schema(
        cls,
        descriptor: PluginDescriptor,
        entry: CapabilityEntry,
    ) -> dict[str, JsonValue]:
        assert descriptor.optimization is not None
        optimized = set(descriptor.optimization.parameters)
        candidates = tuple(field for field in descriptor.parameters if field.name in optimized)
        fixed = tuple(field for field in descriptor.parameters if field.name not in optimized)
        return optimization_request_schema(
            contract_from_capability_entry(entry),
            inputs_schema=cls._fields_schema(descriptor.inputs),
            fixed_parameters_schema=cls._fields_schema(fixed),
            candidate_schema=cls._fields_schema(candidates, require_all=True),
        )

    @staticmethod
    def _fields_schema(
        fields: tuple[PluginField, ...], *, require_all: bool = False
    ) -> dict[str, JsonValue]:
        properties = {field.name: deepcopy(field.json_schema) for field in fields}
        required = [field.name for field in fields if require_all or field.required]
        schema: dict[str, JsonValue] = {
            "type": "object",
            "properties": cast(JsonValue, properties),
            "additionalProperties": False,
        }
        if required:
            schema["required"] = cast(JsonValue, required)
        return schema
