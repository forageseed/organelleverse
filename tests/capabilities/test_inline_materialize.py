"""Tests for inline parameter materialization (T-A1.1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from organelleverse.capabilities.inline_materialize import materialize_inline_parameter
from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.operations.spec import InlineParameterValue


def _inline(value: str, fmt: str = "fasta", encoding: str = "text") -> InlineParameterValue:
    content = (
        value.encode("utf-8")
        if encoding == "text"
        else __import__("base64").b64decode(value, validate=True)
    )
    return InlineParameterValue(
        value=value,
        sha256=hashlib.sha256(content).hexdigest(),
        format=fmt,
        encoding=encoding,  # type: ignore[arg-type]
    )


def test_materialize_text_inline(tmp_path: Path) -> None:
    payload = ">sample\nACGT"
    value = _inline(payload, fmt="fasta")
    path = materialize_inline_parameter(
        value,
        tmp_path,
        parameter_name="input_fasta",
        capability_id="demo.tool",
        inline_max_bytes=1024,
    )
    assert path.parent == tmp_path
    assert path.name == "inline-input_fasta.fasta"
    assert path.read_text() == payload


def test_materialize_base64_inline(tmp_path: Path) -> None:
    raw = b"\x89PNG\r\n\x1a\n"
    value = InlineParameterValue(
        value="iVBORw0KGgo=",
        sha256=hashlib.sha256(raw).hexdigest(),
        format="png",
        encoding="base64",
    )
    path = materialize_inline_parameter(
        value,
        tmp_path,
        parameter_name="image",
        capability_id="demo.tool",
        inline_max_bytes=1024,
    )
    assert path.read_bytes() == raw


def test_sha256_mismatch_fails(tmp_path: Path) -> None:
    value = InlineParameterValue(
        value="hello",
        sha256="0" * 64,
        format="txt",
        encoding="text",
    )
    with pytest.raises(OrganelleContractError, match="SHA-256 does not match"):
        materialize_inline_parameter(
            value,
            tmp_path,
            parameter_name="x",
            capability_id="demo.tool",
            inline_max_bytes=1024,
        )


def test_inline_max_bytes_enforced(tmp_path: Path) -> None:
    value = _inline("ab")
    with pytest.raises(OrganelleInputError, match="exceeds inline_max_bytes"):
        materialize_inline_parameter(
            value,
            tmp_path,
            parameter_name="x",
            capability_id="demo.tool",
            inline_max_bytes=1,
        )


def test_unknown_format_uses_extension(tmp_path: Path) -> None:
    value = _inline("data", fmt="custom")
    path = materialize_inline_parameter(
        value,
        tmp_path,
        parameter_name="x",
        capability_id="demo.tool",
        inline_max_bytes=1024,
    )
    assert path.suffix == ".custom"
