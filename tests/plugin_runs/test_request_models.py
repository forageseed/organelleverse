"""Tests for plugin-run request model (T-A1)."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import JsonValue, ValidationError

from organelleverse.operations.spec import (
    ExecutionContext,
    InlineParameterValue,
    PathParameterValue,
)
from organelleverse.plugin_runs.models import PluginRunRequest


def test_request_accepts_string_inputs() -> None:
    request = PluginRunRequest(capability_id="demo.tool", inputs={"path": "/data/x.fasta"})
    assert request.inputs == {"path": "/data/x.fasta"}


def test_request_accepts_path_parameter_value() -> None:
    value = PathParameterValue(value="/data/x.fasta", context=ExecutionContext.HOST)
    request = PluginRunRequest(capability_id="demo.tool", inputs={"path": value})
    assert request.inputs["path"] == value


def test_request_accepts_inline_parameter_value() -> None:
    content = ">sample\nACGT"
    value = InlineParameterValue(
        value=content,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        format="fasta",
    )
    request = PluginRunRequest(capability_id="demo.tool", inputs={"seq": value})
    assert request.inputs["seq"] == value


def test_request_accepts_inline_parameter_mapping() -> None:
    content = "0.75"
    request = PluginRunRequest.model_validate(
        {
            "capability_id": "demo.tool",
            "inputs": {},
            "parameters": {
                "threshold": {
                    "kind": "inline",
                    "value": content,
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "format": "json",
                    "encoding": "text",
                }
            },
        }
    )
    assert isinstance(request.parameters["threshold"], InlineParameterValue)


def test_scientific_json_kind_field_is_not_hijacked_as_transport() -> None:
    payload: dict[str, JsonValue] = {"kind": "inline", "score": 0.75}
    request = PluginRunRequest(
        capability_id="demo.tool", inputs={}, parameters={"payload": payload}
    )

    assert request.parameters["payload"] == payload


def test_request_rejects_overlapping_input_and_parameter_names() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        PluginRunRequest(
            capability_id="demo.tool",
            inputs={"sample": "/data/a"},
            parameters={"sample": "different"},
        )
