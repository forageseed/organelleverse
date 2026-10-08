"""Plugin Plan 01, Task 2: explicit v1/v2 parser dispatch and strict v2 models.

The v2 plugin contract (design: ``2026-08-11-scientific-plugin-ecosystem-design.md``)
adds plugin metadata, named outputs, an optimization declaration, and the
``plugin_protocol`` argument mode on top of the exact v1 OperationSpec
surface. These tests pin: schema dispatch happens *before* field-level
validation, v2 reuses the v1 parameter vocabulary (``codec``/``path_role``/
``source``/``json_schema``) rather than inventing a parallel one, every v2
cross-reference rule holds, and v1 documents keep parsing and serializing
byte-for-byte as before.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.spec import ArgumentMode

_V1_PAYLOAD = {
    "schema": "organelleverse.capability.v1",
    "capability": {"id": "demo.freeze", "bundle_version": "1.0.0", "implementation": "native"},
    "contract": {
        "operation_id": "demo.freeze",
        "contract_version": "1.0",
        "title": "Freeze probe",
        "description": "A v1 payload whose serialized document must never change.",
        "keywords": ["freeze", "probe", "v1"],
        "execution_mode": "inline",
        "stage": "analyze",
        "input_kind": "none",
        "output_kind": "result",
        "callable_locator": "tests.capabilities.fixtures_core_capability:run",
        "binding": {
            "argument_mode": "named_parameters",
            "parameters": [{"name": "value", "codec": "json"}],
        },
    },
}

_V1_EXPECTED_DOCUMENT = json.loads(
    '{"schema": "organelleverse.capability.v1", "capability": {"id": "demo.freeze",'
    ' "bundle_version": "1.0.0", "implementation": "native"}, "contract":'
    ' {"operation_id": "demo.freeze", "contract_version": "1.0", "title": "Freeze probe",'
    ' "description": "A v1 payload whose serialized document must never change.",'
    ' "keywords": ["freeze", "probe", "v1"], "execution_mode": "inline",'
    ' "stage": "analyze", "input_kind": "none", "output_kind": "result",'
    ' "input_sequence": false,'
    ' "organelle_types": [], "input_modalities": [], "output_modalities": [],'
    ' "callable_locator": "tests.capabilities.fixtures_core_capability:run",'
    ' "binding": {"argument_mode": "named_parameters", "core_input_parameter": null,'
    ' "parameters": [{"name": "value", "codec": "json", "source": "agent",'
    ' "json_schema": {}, "target_type_locator": "", "path_role": null}],'
    ' "result_codec": "canonical", "result_key": null}, "dependencies": [],'
    ' "side_effects": [], "deterministic": true, "deterministic_reason": "",'
    ' "rewrite": null, "idempotent": true, "cacheable": false,'
    ' "retry": {"max_attempts": 1, "retryable_error_codes": []},'
    ' "fallback": {"allowed": false, "allowed_backends": []}, "references": []},'
    ' "probe": [], "data_contract": [], "fixture": [], "composite": null}'
)


def v2_payload(**overrides: object) -> dict[str, object]:
    """A minimal valid v2 plugin manifest payload, overridable per test."""

    payload: dict[str, object] = {
        "schema": "organelleverse.capability.v2",
        "capability": {
            "id": "em.segment",
            "bundle_version": "1.0.0",
            "implementation": "native",
        },
        "contract": {
            "operation_id": "em.segment",
            "contract_version": "1.0",
            "title": "EM organelle segmentation",
            "description": "Segments mitochondria and chloroplasts in EM images.",
            "keywords": ["em", "organelle", "segmentation"],
            "execution_mode": "durable",
            "stage": "analyze",
            "input_kind": "none",
            "output_kind": "result",
            "callable_locator": "em_organelle.plugin:run",
            "execution_context": "host",
            "progress_source": "stages",
            "binding": {
                "argument_mode": "plugin_protocol",
                "parameters": [
                    {
                        "name": "images",
                        "codec": "directory",
                        "path_role": "input",
                        "json_schema": {
                            "type": "string",
                            "description": "Directory of electron-microscopy images.",
                        },
                    },
                    {"name": "labels", "codec": "directory", "path_role": "input"},
                    {
                        "name": "source_file",
                        "codec": "path",
                        "accepts": ["path", "inline"],
                        "inline_max_bytes": 1048576,
                        "json_schema": {
                            "type": "string",
                            "description": "Optional source file.",
                            "title": "Source File",
                            "ui_placeholder": "Path or pasted content.",
                            "ui_file_ext": [".txt", ".md"],
                        },
                    },
                    {
                        "name": "masks_dir",
                        "codec": "directory",
                        "path_role": "output",
                        "json_schema": {
                            "type": "string",
                            "description": "Predicted organelle masks destination.",
                        },
                    },
                    {
                        "name": "report_path",
                        "codec": "path",
                        "path_role": "output",
                        "json_schema": {
                            "type": "string",
                            "description": "Segmentation quality report destination.",
                        },
                    },
                    {
                        "name": "confidence_threshold",
                        "codec": "json",
                        "json_schema": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.5,
                        },
                    },
                ],
            },
            "outputs": [
                {
                    "name": "masks",
                    "kind": "directory",
                    "parameter": "masks_dir",
                    "description": "Predicted organelle masks.",
                },
                {
                    "name": "report",
                    "kind": "file",
                    "parameter": "report_path",
                    "description": "Segmentation quality and measurements report.",
                },
            ],
            "optimization": {
                "score_locator": "em_organelle.plugin:score",
                "parameters": ["confidence_threshold"],
                "max_trials": 12,
                "parallelism": 2,
            },
        },
        "plugin": {
            "name": "Electron-microscopy organelle segmentation",
            "version": "0.1.0",
            "author": "Example Laboratory",
            "summary": "Segments mitochondria and chloroplasts in EM images.",
            "tags": ["electron-microscopy", "organelle", "segmentation"],
            "icon": "🔬",
            "category": "image",
            "badge": "job",
        },
        "agent": {
            "task_description": "Segment organelles in electron-microscopy images.",
            "examples": ["Segment mitochondria in this image directory."],
            "prompt": "Segment the organelles in the provided EM image directory.",
            "agent_surface": "granular",
        },
        "gui": {"page": "ui:page"},
    }
    payload.update(overrides)
    return payload


# A minimal v2 manifest as a TOML document. Parser-level tests use this text;
# model-level tests use ``v2_payload`` dicts directly.
_V2_MANIFEST_TOML = """\
schema = "organelleverse.capability.v2"

[capability]
id = "em.segment"
bundle_version = "1.0.0"
implementation = "native"

[contract]
contract_version = "1.0"
title = "EM organelle segmentation"
description = "Segments mitochondria and chloroplasts in EM images."
keywords = ["em", "organelle", "segmentation"]
execution_mode = "durable"
stage = "analyze"
input_kind = "none"
output_kind = "result"
callable_locator = "em_organelle.plugin:run"

[contract.binding]
argument_mode = "plugin_protocol"

[[contract.binding.parameters]]
name = "masks_dir"
codec = "directory"
path_role = "output"

[[contract.outputs]]
name = "masks"
kind = "directory"
parameter = "masks_dir"
description = "Predicted organelle masks."

[plugin]
name = "Electron-microscopy organelle segmentation"
version = "0.1.0"
author = "Example Laboratory"
summary = "Segments mitochondria and chloroplasts in EM images."
tags = ["electron-microscopy", "organelle", "segmentation"]

[agent]
task_description = "Segment organelles in electron-microscopy images."
examples = ["Segment mitochondria in this image directory."]

[gui]
page = "ui:page"
"""


def _write_manifest(tmp_path: Path, text: str = _V2_MANIFEST_TOML, *, schema: str = "") -> Path:
    if schema:
        text = text.replace("organelleverse.capability.v2", schema, 1)
    path = tmp_path / "capability.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_parser_rejects_unsupported_schema_before_field_validation(tmp_path: Path) -> None:
    from organelleverse.capabilities.parser import parse_capability_bundle

    manifest = _write_manifest(tmp_path, schema="organelleverse.capability.v3")

    with pytest.raises(OrganelleContractError) as error:
        parse_capability_bundle(manifest)

    assert error.value.code == "capability.schema_unsupported"
    details = error.value.as_dict()["details"]
    assert isinstance(details, dict)
    assert details["schema"] == "organelleverse.capability.v3"


def test_parser_dispatches_v2_to_the_plugin_model(tmp_path: Path) -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle
    from organelleverse.capabilities.parser import parse_capability_bundle

    bundle = parse_capability_bundle(_write_manifest(tmp_path))

    assert isinstance(bundle, PluginCapabilityBundle)
    assert bundle.contract.binding.argument_mode is ArgumentMode.PLUGIN_PROTOCOL
    assert bundle.capability.id == "em.segment"


_V1_MANIFEST_TOML = """\
schema = "organelleverse.capability.v1"

[capability]
id = "demo.freeze"
bundle_version = "1.0.0"
implementation = "native"

[contract]
contract_version = "1.0"
title = "Freeze probe"
description = "A v1 payload whose serialized document must never change."
keywords = ["freeze", "probe", "v1"]
execution_mode = "inline"
stage = "analyze"
input_kind = "none"
output_kind = "result"
callable_locator = "tests.capabilities.fixtures_core_capability:run"

[contract.binding]
argument_mode = "named_parameters"

[[contract.binding.parameters]]
name = "value"
codec = "json"
"""


def test_parser_keeps_v1_on_the_exact_v1_model(tmp_path: Path) -> None:
    from organelleverse.capabilities.models import CapabilityBundle, PluginCapabilityBundle
    from organelleverse.capabilities.parser import parse_capability_bundle

    bundle = parse_capability_bundle(_write_manifest(tmp_path, _V1_MANIFEST_TOML))

    assert type(bundle) is CapabilityBundle
    assert not isinstance(bundle, PluginCapabilityBundle)


def test_v2_bundle_keeps_v1_parameter_vocabulary_and_adds_named_outputs() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle
    from organelleverse.operations.spec import PluginOutput

    bundle = PluginCapabilityBundle.model_validate(v2_payload())

    assert bundle.contract.binding.argument_mode is ArgumentMode.PLUGIN_PROTOCOL
    assert bundle.contract.outputs == (
        PluginOutput(
            name="masks",
            kind="directory",
            parameter="masks_dir",
            description="Predicted organelle masks.",
        ),
        PluginOutput(
            name="report",
            kind="file",
            parameter="report_path",
            description="Segmentation quality and measurements report.",
        ),
    )
    assert bundle.plugin.author == "Example Laboratory"
    assert bundle.gui.page == "ui:page"
    assert bundle.contract.optimization is not None
    assert bundle.contract.optimization.score_locator == "em_organelle.plugin:score"


def test_v2_output_parameter_must_be_a_declared_binding() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["outputs"][0]["parameter"] = "missing"  # type: ignore[index]

    with pytest.raises(ValidationError, match="output parameter"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_directory_output_requires_output_path_role() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["binding"]["parameters"][3]["path_role"] = "input"  # type: ignore[index]

    with pytest.raises(ValidationError, match="path_role"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_file_output_requires_path_codec_and_output_role() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    report = payload["contract"]["binding"]["parameters"][4]  # type: ignore[index]
    report["codec"] = "directory"  # type: ignore[index]
    report["path_role"] = "input"  # type: ignore[index]

    with pytest.raises(ValidationError, match="file output"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_output_transport_is_path_only() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["binding"]["parameters"][4]["accepts"] = [  # type: ignore[index]
        "path",
        "inline",
    ]

    with pytest.raises(ValidationError, match="path transport only"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_output_names_and_parameters_are_unique() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["outputs"] = [  # type: ignore[index]
        {"name": "masks", "kind": "directory", "parameter": "masks_dir", "description": "One."},
        {"name": "masks", "kind": "file", "parameter": "report_path", "description": "Two."},
    ]
    with pytest.raises(ValidationError, match="unique"):
        PluginCapabilityBundle.model_validate(payload)

    payload = v2_payload()
    payload["contract"]["outputs"] = [  # type: ignore[index]
        {"name": "masks", "kind": "directory", "parameter": "masks_dir", "description": "One."},
        {"name": "again", "kind": "directory", "parameter": "masks_dir", "description": "Two."},
    ]
    with pytest.raises(ValidationError, match="unique"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_every_output_role_directory_is_referenced_once() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["outputs"] = [  # type: ignore[index]
        {"name": "report", "kind": "file", "parameter": "report_path", "description": "Only."},
    ]
    with pytest.raises(ValidationError, match="referenced"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_optimization_names_must_be_agent_parameters() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["optimization"]["parameters"] = ["undeclared"]  # type: ignore[index]

    with pytest.raises(ValidationError, match="optimization"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_optimization_requires_positive_trial_budget() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["optimization"]["max_trials"] = 0  # type: ignore[index]

    with pytest.raises(ValidationError):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_permits_closed_json_schema_on_json_path_and_directory_parameters() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    bundle = PluginCapabilityBundle.model_validate(v2_payload())

    threshold = bundle.contract.binding.parameters[-1]
    assert threshold.codec.value == "json"
    assert threshold.json_schema["maximum"] == 1.0


def test_v1_still_rejects_json_schema_on_json_safe_parameters() -> None:
    from organelleverse.capabilities.models import CapabilityBundle

    payload = json.loads(json.dumps(_V1_PAYLOAD))
    payload["contract"]["binding"]["parameters"][0]["json_schema"] = {"type": "number"}

    with pytest.raises(ValidationError, match="never"):
        CapabilityBundle.model_validate(payload)


def test_v1_bundle_serialization_stays_byte_for_byte_unchanged() -> None:
    from organelleverse.capabilities.models import CapabilityBundle

    bundle = CapabilityBundle.model_validate(_V1_PAYLOAD)

    assert bundle.model_dump(mode="json", by_alias=True) == _V1_EXPECTED_DOCUMENT


def test_v2_plugin_metadata_defaults() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["plugin"].pop("icon", None)  # type: ignore[index]
    payload["plugin"].pop("category", None)  # type: ignore[index]
    payload["plugin"].pop("badge", None)  # type: ignore[index]

    bundle = PluginCapabilityBundle.model_validate(payload)

    assert bundle.plugin.icon == ""
    assert bundle.plugin.category == "general"
    assert bundle.plugin.badge == "read"


def test_v2_agent_metadata_defaults() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["agent"].pop("prompt", None)  # type: ignore[index]
    payload["agent"].pop("agent_surface", None)  # type: ignore[index]

    bundle = PluginCapabilityBundle.model_validate(payload)

    assert bundle.agent.prompt == ""
    assert bundle.agent.agent_surface.value == "granular"


def test_v2_operation_execution_and_progress_defaults() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"].pop("execution_context", None)  # type: ignore[index]
    payload["contract"].pop("progress_source", None)  # type: ignore[index]

    bundle = PluginCapabilityBundle.model_validate(payload)

    assert bundle.contract.execution_context.value == "host"
    assert bundle.contract.progress_source.value == "none"


def test_v2_binding_accepts_default_is_path() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    bundle = PluginCapabilityBundle.model_validate(v2_payload())
    source_file = bundle.contract.binding.parameters[2]
    assert source_file.codec.value == "path"
    assert source_file.accepts == ("path", "inline")
    assert source_file.inline_max_bytes == 1_048_576
    assert bundle.contract.binding.parameters[-1].accepts == ()


def test_v2_binding_accepts_are_canonicalized() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["binding"]["parameters"][2]["accepts"] = [  # type: ignore[index]
        "inline",
        "path",
        "inline",
    ]

    bundle = PluginCapabilityBundle.model_validate(payload)

    assert bundle.contract.binding.parameters[2].accepts == ("path", "inline")


def test_v2_binding_accepts_must_be_non_empty() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["binding"]["parameters"][2]["accepts"] = []  # type: ignore[index]

    with pytest.raises(ValidationError, match="at least one accepted"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_binding_accepts_only_path_or_inline() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["binding"]["parameters"][2]["accepts"] = ["path", "url"]  # type: ignore[index]

    with pytest.raises(ValidationError):
        PluginCapabilityBundle.model_validate(payload)


@pytest.mark.parametrize("codec", ["dataclass", "directory"])
def test_v2_binding_inline_requires_safe_codec(codec: str) -> None:
    from organelleverse.operations.spec import (
        ParameterCodec,
        PluginParameterBindingSpec,
    )

    with pytest.raises(ValidationError, match=r"input forms|inline content"):
        PluginParameterBindingSpec(
            name="x",
            codec=ParameterCodec(codec),
            target_type_locator="some:type" if codec == "dataclass" else "",
            path_role="input" if codec == "directory" else None,
            json_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            accepts=("path", "inline"),
        )


def test_v2_binding_inline_forbids_environment_source() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle
    from organelleverse.operations.spec import ParameterSource

    payload = v2_payload()
    payload["contract"]["binding"]["parameters"][2]["source"] = ParameterSource.ENVIRONMENT.value  # type: ignore[index]

    with pytest.raises(ValidationError, match="environment"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_binding_inline_max_bytes_must_be_positive() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["binding"]["parameters"][2]["inline_max_bytes"] = 0  # type: ignore[index]

    with pytest.raises(ValidationError, match="inline_max_bytes"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_agent_surface_must_be_valid() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["agent"]["agent_surface"] = "automatic"  # type: ignore[index]

    with pytest.raises(ValidationError, match="agent_surface"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_progress_source_must_be_valid() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["progress_source"] = "percent"  # type: ignore[index]

    with pytest.raises(ValidationError, match="progress_source"):
        PluginCapabilityBundle.model_validate(payload)


def test_v2_execution_context_must_be_valid() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle

    payload = v2_payload()
    payload["contract"]["execution_context"] = "vmware"  # type: ignore[index]

    with pytest.raises(ValidationError, match="execution_context"):
        PluginCapabilityBundle.model_validate(payload)
