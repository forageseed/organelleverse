"""io.read_long_reads: the released producer of sequencing_reads data.

Registering it takes the catalog from 8 invocable operations to 13, because
assembly.assemble becomes invocable, result becomes producible, and the whole
consume tier follows.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from organelleverse import operations as op
from organelleverse.assembly.data_contract import validate_released_assembly_data
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.io_reads import LongReadLibrary, read_long_reads
from organelleverse.operations.adapters.json import invoke_json
from organelleverse.operations.registry import registry


def _mapping(value: object) -> dict[str, object]:
    """Narrow a JSON-Schema node for Pyright strict.

    ``isinstance(x, dict)`` alone narrows only to ``dict[Unknown, Unknown]``, so
    reading a key off it is a reportUnknownVariableType error. This is the repo
    idiom - see tests/operations/test_json_adapter.py.
    """
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _string_list(value: object) -> list[str]:
    """Narrow a JSON-Schema string array for Pyright strict."""
    assert isinstance(value, list)
    return cast(list[str], value)


_TECHNOLOGY = "ont"
_QUALITY = "raw"


@pytest.fixture
def reads(tmp_path: Path) -> Path:
    path = tmp_path / "long.fastq"
    path.write_text("@r1\nACGT\n+\n!!!!\n", encoding="utf-8")
    return path


def test_operation_is_registered_with_a_closed_three_parameter_schema() -> None:
    assert len(op.list()) == 20
    schema = op.parameter_schema("io.read_long_reads")
    expected = ["quality_state", "reads", "technology"]
    assert sorted(_mapping(schema["properties"])) == expected
    assert sorted(_string_list(schema["required"])) == expected


def test_literal_sets_cannot_drift_from_the_model() -> None:
    """The facade's Literal sets are copied from LongReadLibrary. If the model
    gains or loses a value and the facade does not, this fails."""
    properties = _mapping(op.parameter_schema("io.read_long_reads")["properties"])
    model_properties = _mapping(LongReadLibrary.model_json_schema()["properties"])
    for field in ("technology", "quality_state"):
        declared = _mapping(properties[field])
        expected = _mapping(model_properties[field])
        assert declared["enum"] == expected["enum"]


def test_direct_call_produces_released_acceptable_data(reads: Path) -> None:
    data = read_long_reads(reads, technology=_TECHNOLOGY, quality_state=_QUALITY)
    assert isinstance(data, OrganelleData)
    assert data.modality == "sequencing_reads"
    validate_released_assembly_data(data)


def test_registry_and_json_paths_agree_on_the_result(reads: Path) -> None:
    """The two paths take different input ENCODINGS and that is correct: the
    parameter model is strict, so registry.invoke needs a real Path while
    invoke_json takes the JSON string and decodes it. They must agree on the
    result, not on the encoding."""
    through_registry = op.invoke(
        "io.read_long_reads",
        input=None,
        parameters={"reads": reads, "technology": _TECHNOLOGY, "quality_state": _QUALITY},
    )
    response = invoke_json(
        {
            "operation_id": "io.read_long_reads",
            "input": None,
            "parameters": {
                "reads": str(reads),
                "technology": _TECHNOLOGY,
                "quality_state": _QUALITY,
            },
        },
        registry=registry,
        granted_side_effects=["read_files"],
    )
    assert response["ok"] is True
    assert isinstance(through_registry, OrganelleData)
    result = response["result"]
    assert isinstance(result, dict)
    assert result["object_id"] == through_registry.object_id


def test_registry_rejects_a_string_path(reads: Path) -> None:
    """Strict parameter validation. Do not 'fix' this to accept strings; the
    JSON adapter is what decodes them."""
    with pytest.raises(OrganelleParameterError) as excinfo:
        op.invoke(
            "io.read_long_reads",
            input=None,
            parameters={
                "reads": str(reads),
                "technology": _TECHNOLOGY,
                "quality_state": _QUALITY,
            },
        )
    assert excinfo.value.code == "parameter.invalid_operation_parameters"


def test_missing_file_raises_a_typed_input_error(tmp_path: Path) -> None:
    with pytest.raises(OrganelleInputError):
        read_long_reads(tmp_path / "absent.fastq", technology=_TECHNOLOGY, quality_state=_QUALITY)


def test_read_files_grant_is_required(reads: Path) -> None:
    response = invoke_json(
        {
            "operation_id": "io.read_long_reads",
            "input": None,
            "parameters": {
                "reads": str(reads),
                "technology": _TECHNOLOGY,
                "quality_state": _QUALITY,
            },
        },
        registry=registry,
        granted_side_effects=[],
    )
    error = response["error"]
    assert isinstance(error, dict)
    assert error["error_code"] == "permission.denied"


def test_the_operation_writes_nothing(reads: Path, tmp_path: Path) -> None:
    """read_files is the only declared side effect, so nothing may be written."""
    before = sorted(p.name for p in tmp_path.iterdir())
    read_long_reads(reads, technology=_TECHNOLOGY, quality_state=_QUALITY)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_a_non_fastq_file_is_refused(tmp_path: Path) -> None:
    """The defect this fixes: before verification, any readable file was accepted
    and labelled FASTQ with validated=True."""
    junk = tmp_path / "notes.txt"
    junk.write_text("hello world\nnot a sequence\n", encoding="utf-8")
    with pytest.raises(OrganelleInputError) as excinfo:
        read_long_reads(junk, technology=_TECHNOLOGY, quality_state=_QUALITY)
    assert excinfo.value.code == "input.invalid_fastq"
