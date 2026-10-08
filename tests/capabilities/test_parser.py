"""``parse_capability_bundle``: a read-only ``capability.toml`` -> CapabilityBundle parser.

Discovery-safe by construction: parsing must never import the callable a
bundle names, run a subprocess, or touch the network. These are proven, not
assumed - see the "discovery safety" section below.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.core.errors import OrganelleContractError, OrganelleError
from organelleverse.core.frozen import FrozenMap


def _detail(error: OrganelleError, key: str) -> object:
    details = error.details
    assert isinstance(details, FrozenMap)
    return details[key]


_VALID_TOML = """\
schema = "organelleverse.capability.v1"

[capability]
id             = "demo.probe"
bundle_version = "1.0.0"
implementation = "external"

[contract]
contract_version  = "1.0"
title             = "Demo probe capability"
description       = "Parser fixture contract exercising the full bundle shape."
keywords          = ["demo", "fixture", "parser"]
execution_mode    = "inline"
stage             = "analyze"
input_kind        = "genome"
output_kind       = "result"
callable_locator  = "this.module.does.not:exist"

[[contract.dependencies]]
kind = "executable"
name = "blastn"

[[probe]]
dependency      = "blastn"
help_argv       = ["-help"]
requires        = ["-query", "-subject"]
version_argv    = ["-version"]
version_capture = "blast[^0-9]*([0-9][0-9.]*)"

[[fixture]]
case        = "arabidopsis_mito_min"
input       = { kind = "genome", path = "fixtures/arabidopsis_mito_min/input/mito.fasta" }
parameters  = { backend = "auto" }
expect      = "fixtures/arabidopsis_mito_min/expected.json"
equivalence = "exact"
"""


def _write(tmp_path: Path, text: str, name: str = "capability.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- happy path ------------------------------------------------------------------


def test_parses_a_well_formed_bundle(tmp_path: Path) -> None:
    path = _write(tmp_path, _VALID_TOML)
    bundle = parse_capability_bundle(path)

    assert bundle.capability.id == "demo.probe"
    assert bundle.contract.operation_id == "demo.probe"
    assert bundle.contract.callable_locator == "this.module.does.not:exist"
    assert len(bundle.probes) == 1
    assert bundle.probes[0].dependency == "blastn"
    assert len(bundle.fixtures) == 1
    assert bundle.fixtures[0].case == "arabidopsis_mito_min"


def test_operation_id_defaults_from_capability_id_when_omitted(tmp_path: Path) -> None:
    path = _write(tmp_path, _VALID_TOML)
    bundle = parse_capability_bundle(path)
    assert bundle.contract.operation_id == bundle.capability.id


def test_explicit_operation_id_matching_capability_id_is_accepted(tmp_path: Path) -> None:
    text = _VALID_TOML.replace(
        'contract_version  = "1.0"',
        'contract_version  = "1.0"\noperation_id      = "demo.probe"',
    )
    path = _write(tmp_path, text)
    bundle = parse_capability_bundle(path)
    assert bundle.contract.operation_id == "demo.probe"


def test_explicit_mismatched_operation_id_is_rejected(tmp_path: Path) -> None:
    text = _VALID_TOML.replace(
        'contract_version  = "1.0"',
        'contract_version  = "1.0"\noperation_id      = "other.op"',
    )
    path = _write(tmp_path, text)
    with pytest.raises(OrganelleContractError):
        parse_capability_bundle(path)


# --- relative paths are never rewritten -------------------------------------------


def test_fixture_paths_stay_exactly_as_written_relative_strings(tmp_path: Path) -> None:
    path = _write(tmp_path, _VALID_TOML)
    bundle = parse_capability_bundle(path)
    fixture = bundle.fixtures[0]
    assert fixture.input["path"] == "fixtures/arabidopsis_mito_min/input/mito.fasta"
    assert fixture.expect == "fixtures/arabidopsis_mito_min/expected.json"
    assert not Path(fixture.expect).is_absolute()


# --- structured, stable errors -----------------------------------------------------


def test_missing_file_raises_a_structured_contract_error_with_path(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist" / "capability.toml"
    with pytest.raises(OrganelleContractError) as excinfo:
        parse_capability_bundle(missing)
    assert _detail(excinfo.value, "bundle_path") == str(missing)
    assert excinfo.value.code


def test_invalid_toml_syntax_raises_a_structured_contract_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "this is not [valid toml")
    with pytest.raises(OrganelleContractError) as excinfo:
        parse_capability_bundle(path)
    assert _detail(excinfo.value, "bundle_path") == str(path)


def test_contract_validation_failure_raises_a_structured_contract_error(tmp_path: Path) -> None:
    text = _VALID_TOML.replace('implementation = "external"', 'implementation = "not-a-real-kind"')
    path = _write(tmp_path, text)
    with pytest.raises(OrganelleContractError) as excinfo:
        parse_capability_bundle(path)
    assert _detail(excinfo.value, "bundle_path") == str(path)


def test_unknown_top_level_field_fails_closed(tmp_path: Path) -> None:
    text = _VALID_TOML + '\n[extra]\nsomething = "unexpected"\n'
    path = _write(tmp_path, text)
    with pytest.raises(OrganelleContractError):
        parse_capability_bundle(path)


def test_unknown_probe_field_fails_closed(tmp_path: Path) -> None:
    text = _VALID_TOML.replace(
        'dependency      = "blastn"',
        'dependency      = "blastn"\nunexpected_field = true',
    )
    path = _write(tmp_path, text)
    with pytest.raises(OrganelleContractError):
        parse_capability_bundle(path)


def test_parser_errors_are_never_a_bare_pydantic_validation_error(tmp_path: Path) -> None:
    text = _VALID_TOML.replace('implementation = "external"', 'implementation = "not-a-real-kind"')
    path = _write(tmp_path, text)
    try:
        parse_capability_bundle(path)
    except ValidationError:
        pytest.fail("parser must translate ValidationError into OrganelleContractError")
    except OrganelleContractError:
        pass


# --- discovery safety --------------------------------------------------------------


def test_parser_never_imports_the_callable_locators_module(tmp_path: Path) -> None:
    path = _write(tmp_path, _VALID_TOML)
    import sys

    assert "this" not in sys.modules
    bundle = parse_capability_bundle(path)
    assert "this" not in sys.modules
    assert bundle.contract.callable_locator == "this.module.does.not:exist"


def test_parser_never_runs_a_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, _VALID_TOML)

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("parse_capability_bundle must never start a subprocess")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    parse_capability_bundle(path)


def test_parser_never_touches_the_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, _VALID_TOML)

    def _forbidden_connect(*args: object, **kwargs: object) -> object:
        raise AssertionError("parse_capability_bundle must never open a network connection")

    monkeypatch.setattr(socket.socket, "connect", _forbidden_connect)
    parse_capability_bundle(path)
