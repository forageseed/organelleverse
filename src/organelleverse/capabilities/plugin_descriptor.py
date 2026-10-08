"""Pure projection from a v2 plugin bundle to one GUI/Agent form descriptor.

The desktop run form and the Agent tool list both consume
:class:`PluginDescriptor`. It is derived entirely from the *validated* v2
contract: this module imports no plugin callable, writes no file, and never
verifies, trusts, or invokes a bundle. Field classification uses only
declared contract facts - ``codec``, ``path_role``, output references, and
the closed ``json_schema`` - never names or heuristics.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from organelleverse.operations.spec import (
    ParameterCodec,
    ParameterSource,
    PluginOperationSpec,
    PluginOptimization,
    PluginOutput,
    PluginParameterBindingSpec,
    StrictSpecModel,
)

from .models import PluginCapabilityBundle


class PluginField(StrictSpecModel):
    """One form field: a named input, output, or value parameter."""

    name: str = Field(min_length=1)
    codec: str
    direction: Literal["input", "output", "parameter"]
    required: bool
    json_schema: dict[str, JsonValue] = Field(default_factory=dict)
    description: str = ""
    label: str = ""
    placeholder: str = ""
    file_ext: tuple[str, ...] = ()
    accepts: tuple[Literal["path", "inline"], ...] = ()
    inline_max_bytes: int | None = Field(default=None, gt=0)


class PluginDescriptor(StrictSpecModel):
    """Everything a generated run form or Agent tool needs, in one document."""

    capability_id: str
    title: str
    summary: str
    inputs: tuple[PluginField, ...]
    outputs: tuple[PluginField, ...]
    parameters: tuple[PluginField, ...]
    agent_task_description: str
    agent_examples: tuple[str, ...]
    gui_page: str | None
    optimization: PluginOptimization | None
    icon: str = ""
    badge: Literal["job", "read"] = "read"
    category: Literal["mito", "chloro", "general", "image"] = "general"
    prompt: str = ""
    progress_source: Literal["stages", "tool", "none"] = "none"
    agent_surface: Literal["granular", "interactive"] = "granular"
    execution_context: Literal["host", "wsl", "container", "remote"] = "host"


def _required(parameter: PluginParameterBindingSpec) -> bool:
    """A parameter is required when its schema declares no default."""

    return "default" not in parameter.json_schema


def _schema_description(parameter: PluginParameterBindingSpec) -> str:
    description = parameter.json_schema.get("description", "")
    return description if isinstance(description, str) else ""


def _schema_label(parameter: PluginParameterBindingSpec) -> str:
    title = parameter.json_schema.get("title")
    if isinstance(title, str) and title.strip():
        return title
    return parameter.name


def _schema_placeholder(parameter: PluginParameterBindingSpec) -> str:
    for key in ("ui_placeholder", "ui:placeholder"):
        value = parameter.json_schema.get(key)
        if isinstance(value, str):
            return value
    return ""


def _schema_file_ext(parameter: PluginParameterBindingSpec) -> tuple[str, ...]:
    for key in ("ui_file_ext", "ui:fileExt"):
        value = parameter.json_schema.get(key)
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
    return ()


def _is_input(parameter: PluginParameterBindingSpec, output_parameters: set[str]) -> bool:
    return (parameter.codec is ParameterCodec.DIRECTORY and parameter.path_role == "input") or (
        parameter.source is ParameterSource.AGENT
        and parameter.codec is ParameterCodec.PATH
        and parameter.name not in output_parameters
    )


def _input_field(parameter: PluginParameterBindingSpec) -> PluginField:
    return PluginField(
        name=parameter.name,
        codec=parameter.codec.value,
        direction="input",
        required=_required(parameter),
        json_schema=parameter.json_schema,
        description=_schema_description(parameter),
        label=_schema_label(parameter),
        placeholder=_schema_placeholder(parameter),
        file_ext=_schema_file_ext(parameter),
        accepts=parameter.accepts,
        inline_max_bytes=(
            parameter.inline_max_bytes if "inline" in parameter.accepts else None
        ),
    )


def _output_field(
    output: PluginOutput, parameter: PluginParameterBindingSpec
) -> PluginField:
    return PluginField(
        name=output.name,
        codec=parameter.codec.value,
        direction="output",
        required=True,
        json_schema=parameter.json_schema,
        description=output.description,
        label=output.name,
        placeholder="",
        file_ext=(),
    )


def _parameter_field(parameter: PluginParameterBindingSpec) -> PluginField:
    return PluginField(
        name=parameter.name,
        codec=parameter.codec.value,
        direction="parameter",
        required=_required(parameter),
        json_schema=parameter.json_schema,
        description=_schema_description(parameter),
        label=_schema_label(parameter),
        placeholder=_schema_placeholder(parameter),
        file_ext=_schema_file_ext(parameter),
        accepts=parameter.accepts,
        inline_max_bytes=(
            parameter.inline_max_bytes if "inline" in parameter.accepts else None
        ),
    )


def describe_plugin(bundle: PluginCapabilityBundle) -> PluginDescriptor:
    """Project one validated v2 bundle into its GUI/Agent form descriptor.

    Pure: reads only the validated contract, preserves manifest declaration
    order, and performs no discovery, import, verification, or invocation.
    """

    contract: PluginOperationSpec = bundle.contract
    bindings = tuple(contract.binding.parameters)
    bindings_by_name = {parameter.name: parameter for parameter in bindings}
    output_parameters = {output.parameter for output in contract.outputs}
    return PluginDescriptor(
        capability_id=bundle.capability.id,
        title=contract.title,
        summary=bundle.plugin.summary,
        inputs=tuple(
            _input_field(parameter)
            for parameter in bindings
            if _is_input(parameter, output_parameters)
        ),
        outputs=tuple(
            _output_field(output, bindings_by_name[output.parameter])
            for output in contract.outputs
        ),
        parameters=tuple(
            _parameter_field(parameter)
            for parameter in bindings
            if not _is_input(parameter, output_parameters)
            and parameter.name not in output_parameters
        ),
        agent_task_description=bundle.agent.task_description,
        agent_examples=bundle.agent.examples,
        gui_page=bundle.gui.page,
        optimization=contract.optimization,
        icon=bundle.plugin.icon,
        badge=bundle.plugin.badge,
        category=bundle.plugin.category,
        prompt=bundle.agent.prompt,
        progress_source=contract.progress_source.value,
        agent_surface=bundle.agent.agent_surface.value,
        execution_context=contract.execution_context.value,
    )


__all__ = ["PluginDescriptor", "PluginField", "describe_plugin"]
