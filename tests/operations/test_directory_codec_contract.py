"""The ``directory`` parameter codec contract: a declared ``path_role``.

A directory parameter names a whole directory tree, not a single file, so the
PATH codec's single-file preflight (existence/readability checks, per-file
hashing) cannot apply. ``path_role`` is what tells the two cases apart:
``"input"`` for a pre-existing directory tree, ``"output"`` for a write
destination that may not exist yet. It is required iff the codec is
``directory`` and forbidden otherwise - mirroring ``target_type_locator``'s
``dataclass``-only rule. Like PATH, DIRECTORY derives its Agent schema as a
plain JSON string and never declares a ``json_schema``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.operations.spec import ParameterBindingSpec, ParameterCodec


def test_directory_codec_requires_a_path_role() -> None:
    with pytest.raises(ValidationError, match="path_role"):
        ParameterBindingSpec(name="data_root", codec=ParameterCodec.DIRECTORY)


def test_path_role_is_forbidden_for_non_path_codecs() -> None:
    # path_role lives on directory AND (since Ruling 1, Decision 004 approval
    # 2026-08-14) the path codec's output-file role; every other codec refuses it
    with pytest.raises(ValidationError, match="path_role"):
        ParameterBindingSpec(name="f", codec=ParameterCodec.JSON, path_role="input")
    ParameterBindingSpec(name="f", codec=ParameterCodec.PATH, path_role="input")  # allowed
    ParameterBindingSpec(name="f", codec=ParameterCodec.PATH, path_role="output")  # allowed


def test_directory_codec_declares_no_json_schema() -> None:
    with pytest.raises(ValidationError, match="json_schema"):
        ParameterBindingSpec(
            name="d",
            codec=ParameterCodec.DIRECTORY,
            path_role="input",
            json_schema={"type": "string"},
        )


_DIRECTORY_TOML = """\
schema = "organelleverse.capability.v1"

[capability]
id = "demo.directory_parameter"
bundle_version = "1.0.0"
implementation = "native"

[contract]
contract_version = "1.0"
title = "Demo directory parameter capability"
description = "Bundle document fixture for the directory codec contract."
keywords = ["codec", "demo", "directory"]
execution_mode = "inline"
stage = "analyze"
input_kind = "none"
output_kind = "result"
callable_locator = "demo.api:probe"

[contract.binding]
argument_mode = "named_parameters"

[[contract.binding.parameters]]
name = "data_root"
codec = "directory"
path_role = "input"
"""


def test_directory_bundle_document_parses(tmp_path: Path) -> None:
    path = tmp_path / "capability.toml"
    path.write_text(_DIRECTORY_TOML, encoding="utf-8")

    bundle = parse_capability_bundle(path)

    (parameter,) = bundle.contract.binding.parameters
    assert parameter.name == "data_root"
    assert parameter.codec is ParameterCodec.DIRECTORY
    assert parameter.path_role == "input"
