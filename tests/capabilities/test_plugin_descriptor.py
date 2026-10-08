"""Plugin Plan 01, Task 3: project admitted v2 Bundles into one form descriptor.

The descriptor is the single projection the desktop form and the Agent tool
list both consume. It derives entirely from the validated v2 contract: no
plugin callable is imported, no file is written, no verification, trust, or
invocation happens.
"""

from __future__ import annotations

import importlib

import pytest

from tests.capabilities.test_plugin_v2_models import v2_payload


def _plugin_bundle():
    from organelleverse.capabilities.models import PluginCapabilityBundle

    return PluginCapabilityBundle.model_validate(v2_payload())


def test_descriptor_derives_three_categories_from_one_v2_bundle() -> None:
    from organelleverse.capabilities.plugin_descriptor import describe_plugin

    descriptor = describe_plugin(_plugin_bundle())

    assert [field.name for field in descriptor.inputs] == ["images", "labels", "source_file"]
    assert [field.name for field in descriptor.outputs] == ["masks", "report"]
    assert [field.name for field in descriptor.parameters] == ["confidence_threshold"]
    assert descriptor.outputs[0].codec == "directory"
    assert descriptor.outputs[0].direction == "output"
    assert descriptor.inputs[0].direction == "input"
    assert descriptor.parameters[0].direction == "parameter"
    assert descriptor.optimization is not None
    assert descriptor.gui_page == "ui:page"
    assert descriptor.agent_task_description.startswith("Segment organelles")


def test_descriptor_uses_binding_schema_without_parallel_parameter_data() -> None:
    from organelleverse.capabilities.plugin_descriptor import describe_plugin

    descriptor = describe_plugin(_plugin_bundle())

    assert descriptor.parameters[0].json_schema == {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "default": 0.5,
    }
    # A declared default makes a parameter optional in the generated form.
    assert descriptor.parameters[0].required is False
    assert descriptor.inputs[0].required is True


def test_descriptor_preserves_manifest_declaration_order() -> None:
    from organelleverse.capabilities.plugin_descriptor import describe_plugin

    descriptor = describe_plugin(_plugin_bundle())

    assert [field.name for field in descriptor.inputs] == ["images", "labels", "source_file"]
    assert [output.parameter for output in _plugin_bundle().contract.outputs] == [
        "masks_dir",
        "report_path",
    ]


def test_descriptor_has_no_discovery_or_execution_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from organelleverse.capabilities.plugin_descriptor import describe_plugin

    monkeypatch.setattr(importlib, "import_module", pytest.fail)

    assert describe_plugin(_plugin_bundle()).capability_id == "em.segment"


def test_descriptor_carries_ui_fields() -> None:
    from organelleverse.capabilities.plugin_descriptor import describe_plugin

    descriptor = describe_plugin(_plugin_bundle())

    assert descriptor.icon == "🔬"
    assert descriptor.badge == "job"
    assert descriptor.category == "image"
    assert descriptor.prompt.startswith("Segment the organelles")


def test_descriptor_field_carries_label_placeholder_and_file_ext() -> None:
    from organelleverse.capabilities.plugin_descriptor import describe_plugin

    descriptor = describe_plugin(_plugin_bundle())
    source = descriptor.inputs[2]

    assert source.name == "source_file"
    assert source.label == "Source File"
    assert source.placeholder == "Path or pasted content."
    assert source.file_ext == (".txt", ".md")
    assert source.accepts == ("path", "inline")
    assert source.inline_max_bytes == 1_048_576
    assert descriptor.inputs[0].accepts == ("path",)
    assert descriptor.inputs[0].inline_max_bytes is None
    assert descriptor.outputs[0].accepts == ()


def test_descriptor_field_label_fallback_to_name() -> None:
    from organelleverse.capabilities.models import PluginCapabilityBundle
    from organelleverse.capabilities.plugin_descriptor import describe_plugin

    payload = v2_payload()
    payload["contract"]["binding"]["parameters"][5]["json_schema"] = {  # type: ignore[index]
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "default": 0.5,
    }
    bundle = PluginCapabilityBundle.model_validate(payload)
    descriptor = describe_plugin(bundle)

    assert descriptor.parameters[0].label == "confidence_threshold"


def test_descriptor_progress_source_and_execution_context() -> None:
    from organelleverse.capabilities.plugin_descriptor import describe_plugin

    descriptor = describe_plugin(_plugin_bundle())

    assert descriptor.progress_source == "stages"
    assert descriptor.execution_context == "host"


def test_descriptor_agent_surface() -> None:
    from organelleverse.capabilities.plugin_descriptor import describe_plugin

    descriptor = describe_plugin(_plugin_bundle())

    assert descriptor.agent_surface == "granular"
