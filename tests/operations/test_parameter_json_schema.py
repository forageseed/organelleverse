"""``ParameterBindingSpec.json_schema`` must be a deeply, explicitly closed,
local-only Draft 2020-12 JSON Schema - not merely "not obviously open".

An ``Annotated[Any, Field(json_schema_extra=...)]`` field only ever produced a
documentation schema; nothing checked it was even valid, let alone closed.
These rules are what make the declared schema trustworthy enough to validate
real Agent-supplied values against at invocation time (see test_python_binding.py).
"""

from __future__ import annotations

import socket

import pytest
from pydantic import JsonValue, ValidationError

from organelleverse.operations.spec import ParameterBindingSpec, ParameterCodec


def _spec(schema: dict[str, JsonValue]) -> ParameterBindingSpec:
    return ParameterBindingSpec(
        name="payload",
        codec=ParameterCodec.DATACLASS,
        target_type_locator="tests.capabilities.fixtures_for_binding:Rectangle",
        json_schema=schema,
    )


# --- must be non-empty and contract-bearing --------------------------------------


def test_empty_schema_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec({})


def test_schema_with_no_contract_keys_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec({"title": "Payload"})


# --- object schemas must explicitly close additionalProperties -------------------


def test_object_schema_without_additional_properties_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec({"type": "object", "properties": {"width": {"type": "number"}}})


def test_object_schema_with_additional_properties_true_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(
            {
                "type": "object",
                "properties": {"width": {"type": "number"}},
                "additionalProperties": True,
            }
        )


def test_object_schema_with_additional_properties_false_is_accepted() -> None:
    spec = _spec(
        {
            "type": "object",
            "properties": {"width": {"type": "number"}, "height": {"type": "number"}},
            "required": ["width", "height"],
            "additionalProperties": False,
        }
    )
    assert spec.json_schema["type"] == "object"


def test_nested_open_object_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "outer": {"type": "number"},
                    "nested": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        # missing additionalProperties: false at this inner level
                    },
                },
            }
        )


def test_deeply_nested_closed_object_is_accepted() -> None:
    spec = _spec(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outer": {"type": "number"},
                "nested": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        }
    )
    properties = spec.json_schema["properties"]
    assert isinstance(properties, dict)
    assert "nested" in properties


def test_open_object_inside_a_list_items_schema_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(
            {
                "type": "array",
                "items": {"type": "object", "properties": {"x": {"type": "string"}}},
            }
        )


# --- schema must itself be a valid Draft 2020-12 document -------------------------


def test_syntactically_invalid_schema_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec({"type": "not-a-real-json-schema-type"})


# --- v1 forbids $ref entirely, local or remote (fail-closed) ---------------------
#
# A local $ref embedded via json_schema_extra is not correctly rebased when
# Pydantic later generates the FULL Agent parameter model's schema: its
# $defs live only in this one field's fragment, so parameter_model's own
# schema-generation walk crashes with a bare KeyError trying to resolve
# "#/$defs/..." against the wrong document (proven directly against
# bind_python_capability in test_python_binding.py). Rather than implement
# full $ref rebasing, v1 rejects every $ref outright, local or remote, so
# this class of bug cannot be constructed in the first place.


def test_remote_ref_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec({"$ref": "https://example.com/schema.json"})


def test_remote_ref_nested_inside_properties_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"width": {"$ref": "http://example.com/number.json"}},
            }
        )


def test_local_ref_is_also_rejected_in_v1() -> None:
    with pytest.raises(ValidationError):
        _spec(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"width": {"$ref": "#/$defs/dimension"}},
                "required": ["width"],
                "$defs": {"dimension": {"type": "number"}},
            }
        )


def test_local_ref_missing_its_target_is_also_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"width": {"$ref": "#/$defs/does_not_exist"}},
                "required": ["width"],
            }
        )


def test_rejecting_a_ref_never_opens_a_network_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden_connect(*args: object, **kwargs: object) -> object:
        raise AssertionError("validating a declared json_schema must never touch the network")

    monkeypatch.setattr(socket.socket, "connect", _forbidden_connect)
    with pytest.raises(ValidationError):
        _spec({"$ref": "https://example.com/schema.json"})
    with pytest.raises(ValidationError):
        _spec({"$ref": "#/$defs/dimension", "$defs": {"dimension": {"type": "number"}}})


# --- JSON/PATH codecs never consume json_schema: a non-empty one is rejected ------


def test_json_codec_rejects_a_non_empty_json_schema() -> None:
    with pytest.raises(ValidationError):
        ParameterBindingSpec(
            name="x",
            codec=ParameterCodec.JSON,
            json_schema={"type": "object", "additionalProperties": False},
        )


def test_path_codec_rejects_a_non_empty_json_schema() -> None:
    with pytest.raises(ValidationError):
        ParameterBindingSpec(
            name="path",
            codec=ParameterCodec.PATH,
            json_schema={"type": "object", "additionalProperties": False},
        )


def test_json_codec_with_no_json_schema_is_valid() -> None:
    spec = ParameterBindingSpec(name="x", codec=ParameterCodec.JSON)
    assert spec.json_schema == {}


# Legacy parameter codecs (legacy_genome/legacy_data/legacy_result) remain a
# valid ParameterBindingSpec declaration in v1 - a parsed bundle can still be
# inspected/listed. bind_python_capability is where v1's lack of an actual
# decoder must be rejected, at bind time, before any invocation is attempted;
# see test_python_binding.py::test_unimplemented_legacy_codec_is_rejected_at_bind_time.
