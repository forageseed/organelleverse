"""bind_python_capability / encode_capability_result: the shared Python binding.

Python and Agent paths share one BoundOperation: the original callable is
never wrapped or reimplemented, only invoked through inspect.signature.bind
and then encoded through the declared ResultCodec.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from organelleverse.capabilities.models import CapabilityBundle, ImplementationKind
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleContractError, OrganelleError, OrganelleInputError
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleMetadata
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import ExecutionMode
from organelleverse.operations.python_binding import (
    bind_python_capability,
    validate_worker_contract,
)
from organelleverse.operations.spec import (
    ArgumentMode,
    ParameterBindingSpec,
    ParameterCodec,
    ParameterSource,
    PythonBindingSpec,
    ResultCodec,
)


def _bundle(**contract_overrides: object) -> CapabilityBundle:
    contract: dict[str, object] = {
        "operation_id": "demo.probe",
        "contract_version": "1.0",
        "title": "Demo probe capability",
        "description": "Test fixture contract used only to probe Python binding.",
        "keywords": ("demo", "fixture", "test"),
        "execution_mode": ExecutionMode.INLINE,
        "stage": "analyze",
        "input_kind": "none",
        "output_kind": "result",
        "callable_locator": "demo.api:probe",
    }
    contract.update(contract_overrides)
    return CapabilityBundle.model_validate(
        {
            "schema": "organelleverse.capability.v1",
            "capability": {
                "id": contract["operation_id"],
                "bundle_version": "1.0.0",
                "implementation": "native",
            },
            "contract": contract,
        }
    )


def test_named_parameters_binding_calls_the_real_function_and_encodes_json_metric() -> None:
    def add(*, x: int, y: int) -> dict[str, int]:
        return {"sum": x + y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
                ParameterBindingSpec(name="y", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="addition",
        )
    )

    bound = bind_python_capability(bundle, add, frozen_schema=None)
    direct = add(x=2, y=3)
    assert direct == {"sum": 5}

    agent = bound.invoke(None, {"x": 2, "y": 3})
    assert isinstance(agent, OrganelleResult)
    assert agent.status == "ok"
    assert agent.metrics["addition"] == {"sum": 5}
    assert agent.operation_id == "demo.probe"


def test_named_parameters_binding_rejects_a_parameter_the_implementation_does_not_have() -> None:
    def add(*, x: int) -> dict[str, int]:
        return {"x": x}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="not_a_real_parameter", codec=ParameterCodec.JSON),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )

    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, add, frozen_schema=None)


def test_named_parameters_binding_rejects_an_unsupported_annotation_without_a_declared_schema() -> None:
    def probe(*, payload: object) -> dict[str, str]:
        return {"payload": str(payload)}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(ParameterBindingSpec(name="payload", codec=ParameterCodec.JSON),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )

    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)


def test_agent_invocation_schema_omits_environment_sourced_parameters() -> None:
    def probe(*, query: str, workdir: str) -> dict[str, str]:
        return {"query": query, "workdir": workdir}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="query", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
                ParameterBindingSpec(
                    name="workdir", codec=ParameterCodec.JSON, source=ParameterSource.ENVIRONMENT
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )

    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    schema = bound.signature.parameter_model.model_json_schema()
    assert "query" in schema["properties"]
    assert "workdir" not in schema["properties"]


def test_environment_sourced_parameter_is_not_satisfiable_by_agent_input() -> None:
    def probe(*, query: str, workdir: str) -> dict[str, str]:
        return {"query": query, "workdir": workdir}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="query", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
                ParameterBindingSpec(
                    name="workdir", codec=ParameterCodec.JSON, source=ParameterSource.ENVIRONMENT
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)

    # A rejected extra field is an OrganelleParameterError, not specifically
    # OrganelleContractError: what matters is that it is rejected at all,
    # since workdir was excluded from the Agent-facing model entirely.
    with pytest.raises(OrganelleError):
        bound.invoke(None, {"query": "x", "workdir": "/should/not/be/agent-settable"})


def test_core_input_parameter_binds_the_converted_l1_input() -> None:
    def probe(*, genome: OrganelleData) -> dict[str, str]:
        return {"modality": genome.modality}

    bundle = _bundle(
        input_kind="data",
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            core_input_parameter="genome",
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        ),
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)

    data = OrganelleData(modality="organelle_records")
    result = bound.invoke(data, {})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"modality": "organelle_records"}


def test_canonical_core_binding_delegates_to_the_existing_signature_mechanism() -> None:
    def annotate(genome: OrganelleData, *, threshold: int = 1) -> OrganelleResult:
        return OrganelleResult(
            operation_id="demo.probe",
            scope="mitochondrion",
            status="ok",
            metrics={"threshold": threshold},  # pyright: ignore[reportArgumentType]
        )

    bundle = _bundle(input_kind="data")
    bound = bind_python_capability(bundle, annotate, frozen_schema=None)

    data = OrganelleData(modality="organelle_records")
    direct = annotate(data, threshold=2)
    agent = bound.invoke(data, {"threshold": 2})
    assert agent == direct


def _rectangle_bundle() -> CapabilityBundle:
    return _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="rectangle",
                    codec=ParameterCodec.DATACLASS,
                    source=ParameterSource.AGENT,
                    target_type_locator="tests.capabilities.fixtures_for_binding:Rectangle",
                    json_schema={
                        "type": "object",
                        "properties": {"width": {"type": "number"}, "height": {"type": "number"}},
                        "required": ["width", "height"],
                        "additionalProperties": False,
                    },
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )


def _rectangle_area(*, rectangle: object) -> dict[str, float]:
    from tests.capabilities.fixtures_for_binding import Rectangle

    assert isinstance(rectangle, Rectangle)
    return {"area": rectangle.width * rectangle.height}


def test_runtime_schema_validation_matches_the_documented_schema() -> None:
    bound = bind_python_capability(_rectangle_bundle(), _rectangle_area, frozen_schema=None)
    schema = bound.signature.parameter_model.model_json_schema()
    assert schema["properties"]["rectangle"]["additionalProperties"] is False
    assert set(schema["properties"]["rectangle"]["required"]) == {"width", "height"}

    result = bound.invoke(None, {"rectangle": {"width": 3, "height": 4}})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"area": 12}


def test_runtime_schema_validation_rejects_a_missing_required_field() -> None:
    bound = bind_python_capability(_rectangle_bundle(), _rectangle_area, frozen_schema=None)
    with pytest.raises(OrganelleContractError) as excinfo:
        bound.invoke(None, {"rectangle": {"width": 3}})
    assert excinfo.value.code == "capability.parameter_schema_invalid"


def test_runtime_schema_validation_rejects_an_extra_field() -> None:
    bound = bind_python_capability(_rectangle_bundle(), _rectangle_area, frozen_schema=None)
    with pytest.raises(OrganelleContractError) as excinfo:
        bound.invoke(None, {"rectangle": {"width": 3, "height": 4, "depth": 5}})
    assert excinfo.value.code == "capability.parameter_schema_invalid"


def test_runtime_schema_validation_rejects_a_wrong_type() -> None:
    bound = bind_python_capability(_rectangle_bundle(), _rectangle_area, frozen_schema=None)
    with pytest.raises(OrganelleContractError) as excinfo:
        bound.invoke(None, {"rectangle": {"width": "wide", "height": 4}})
    assert excinfo.value.code == "capability.parameter_schema_invalid"


def test_runtime_schema_validation_rejects_a_nested_open_object() -> None:
    """A nested-open-object schema is rejected the moment it is declared.

    Task 2's schema-level check (operations/spec.py) runs on every
    ParameterBindingSpec construction, so this never even reaches
    bind_python_capability - the earliest possible rejection point.
    """
    with pytest.raises(ValidationError):
        ParameterBindingSpec(
            name="payload",
            codec=ParameterCodec.NUMPY_ARTIFACT,
            source=ParameterSource.AGENT,
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "nested": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            },
        )


def test_json_and_path_codecs_are_not_routed_through_jsonschema_validation() -> None:
    """JSON/PATH parameters keep their existing Pydantic-level validation as-is."""

    def probe(*, x: int) -> dict[str, int]:
        return {"x": x}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(ParameterBindingSpec(name="x", codec=ParameterCodec.JSON),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    result = bound.invoke(None, {"x": 5})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"x": 5}


def test_dataclass_codec_decodes_a_json_object_into_the_target_type() -> None:
    def area(*, rectangle: object) -> dict[str, float]:
        from tests.capabilities.fixtures_for_binding import Rectangle

        assert isinstance(rectangle, Rectangle)
        return {"area": rectangle.width * rectangle.height}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="rectangle",
                    codec=ParameterCodec.DATACLASS,
                    source=ParameterSource.AGENT,
                    target_type_locator="tests.capabilities.fixtures_for_binding:Rectangle",
                    json_schema={
                        "type": "object",
                        "properties": {"width": {"type": "number"}, "height": {"type": "number"}},
                        "required": ["width", "height"],
                        "additionalProperties": False,
                    },
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, area, frozen_schema=None)

    result = bound.invoke(None, {"rectangle": {"width": 3, "height": 4}})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"area": 12}


def test_numpy_artifact_codec_decodes_a_json_list_into_an_array() -> None:
    import numpy as np

    def total(*, matrix: object) -> dict[str, float]:
        assert isinstance(matrix, np.ndarray)
        return {"total": float(matrix.sum())}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="matrix",
                    codec=ParameterCodec.NUMPY_ARTIFACT,
                    source=ParameterSource.AGENT,
                    json_schema={"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, total, frozen_schema=None)

    result = bound.invoke(None, {"matrix": [[1, 2], [3, 4]]})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"total": 10.0}


def test_named_parameters_binding_attaches_real_provenance() -> None:
    def add(*, x: int, y: int) -> dict[str, int]:
        return {"sum": x + y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
                ParameterBindingSpec(name="y", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="addition",
        )
    )
    bound = bind_python_capability(bundle, add, frozen_schema=None)

    result = bound.invoke(None, {"x": 2, "y": 3})
    assert isinstance(result, OrganelleResult)
    assert result.provenance is not None
    assert result.provenance.operation_id == "demo.probe"
    assert result.provenance.callable_locator == "demo.api:probe"
    assert len(result.provenance.parameters_hash) == 64


def test_parameters_hash_is_stable_for_equal_parameters_and_differs_otherwise() -> None:
    def add(*, x: int, y: int) -> dict[str, int]:
        return {"sum": x + y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
                ParameterBindingSpec(name="y", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="addition",
        )
    )
    bound = bind_python_capability(bundle, add, frozen_schema=None)

    first = bound.invoke(None, {"x": 2, "y": 3})
    second = bound.invoke(None, {"y": 3, "x": 2})  # same values, different call-site order
    third = bound.invoke(None, {"x": 2, "y": 4})
    assert isinstance(first, OrganelleResult)
    assert isinstance(second, OrganelleResult)
    assert isinstance(third, OrganelleResult)
    assert first.provenance is not None and second.provenance is not None and third.provenance is not None
    assert first.provenance.parameters_hash == second.provenance.parameters_hash
    assert first.provenance.parameters_hash != third.provenance.parameters_hash


def test_core_input_object_id_is_recorded_in_provenance() -> None:
    def probe(*, genome: OrganelleData) -> dict[str, str]:
        return {"modality": genome.modality}

    bundle = _bundle(
        input_kind="data",
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            core_input_parameter="genome",
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        ),
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)

    data = OrganelleData(modality="organelle_records")
    result = bound.invoke(data, {})
    assert isinstance(result, OrganelleResult)
    assert result.provenance is not None
    assert result.provenance.input_object_ids == (data.object_id,)


def test_input_artifact_hash_is_recorded_for_a_path_codec_parameter(tmp_path: Path) -> None:
    import hashlib

    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(">demo\nACGT\n", encoding="utf-8")
    expected_hash = hashlib.sha256(fasta_path.read_bytes()).hexdigest()

    def probe(*, fasta: str) -> dict[str, str]:
        return {"fasta": fasta}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)

    result = bound.invoke(None, {"fasta": str(fasta_path)})
    assert isinstance(result, OrganelleResult)
    assert result.provenance is not None
    assert result.provenance.input_artifact_hashes == (expected_hash,)


# --- PATH codec: Agent-facing str, safe conversion per implementation annotation --


def _path_bundle() -> CapabilityBundle:
    return _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )


def test_path_codec_agent_schema_is_a_json_string_when_implementation_annotates_path(
    tmp_path: Path,
) -> None:
    def probe(*, fasta: Path) -> dict[str, str]:
        return {"fasta": str(fasta)}

    bound = bind_python_capability(_path_bundle(), probe, frozen_schema=None)
    schema = bound.signature.parameter_model.model_json_schema()
    assert schema["properties"]["fasta"]["type"] == "string"


def test_path_codec_converts_to_a_real_path_object_when_implementation_annotates_path(
    tmp_path: Path,
) -> None:
    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(">demo\nACGT\n", encoding="utf-8")
    received: dict[str, object] = {}

    def probe(*, fasta: Path) -> dict[str, str]:
        received["fasta"] = fasta
        return {"fasta": str(fasta)}

    bound = bind_python_capability(_path_bundle(), probe, frozen_schema=None)
    result = bound.invoke(None, {"fasta": str(fasta_path)})
    assert isinstance(result, OrganelleResult)
    assert isinstance(received["fasta"], Path)
    assert not isinstance(received["fasta"], str)
    assert received["fasta"] == fasta_path


def test_path_codec_keeps_a_plain_str_when_implementation_annotates_str(tmp_path: Path) -> None:
    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(">demo\nACGT\n", encoding="utf-8")
    received: dict[str, object] = {}

    def probe(*, fasta: str) -> dict[str, str]:
        received["fasta"] = fasta
        return {"fasta": fasta}

    bound = bind_python_capability(_path_bundle(), probe, frozen_schema=None)
    bound.invoke(None, {"fasta": str(fasta_path)})
    assert type(received["fasta"]) is str


def test_path_codec_keeps_a_plain_str_for_a_str_or_path_union_annotation(tmp_path: Path) -> None:
    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(">demo\nACGT\n", encoding="utf-8")
    received: dict[str, object] = {}

    def probe(*, fasta: str | Path) -> dict[str, str]:
        received["fasta"] = fasta
        return {"fasta": str(fasta)}

    bound = bind_python_capability(_path_bundle(), probe, frozen_schema=None)
    bound.invoke(None, {"fasta": str(fasta_path)})
    assert type(received["fasta"]) is str


def test_path_codec_rejects_a_missing_file_before_invoking_the_scientific_function(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    def probe(*, fasta: Path) -> dict[str, str]:
        calls.append(fasta)
        return {"fasta": str(fasta)}

    bound = bind_python_capability(_path_bundle(), probe, frozen_schema=None)
    missing = tmp_path / "does-not-exist.fasta"
    with pytest.raises(OrganelleError):
        bound.invoke(None, {"fasta": str(missing)})
    assert calls == []


def test_path_codec_rejects_an_unreadable_file_before_invoking_the_scientific_function(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    def probe(*, fasta: Path) -> dict[str, str]:
        calls.append(fasta)
        return {"fasta": str(fasta)}

    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(">demo\nACGT\n", encoding="utf-8")
    fasta_path.chmod(0o000)
    try:
        bound = bind_python_capability(_path_bundle(), probe, frozen_schema=None)
        with pytest.raises(OrganelleError):
            bound.invoke(None, {"fasta": str(fasta_path)})
    finally:
        fasta_path.chmod(0o644)
    assert calls == []


def test_path_codec_never_breaks_parameters_hash_json_encoding(tmp_path: Path) -> None:
    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(">demo\nACGT\n", encoding="utf-8")

    def probe(*, fasta: Path) -> dict[str, str]:
        return {"fasta": str(fasta)}

    bound = bind_python_capability(_path_bundle(), probe, frozen_schema=None)
    result = bound.invoke(None, {"fasta": str(fasta_path)})
    assert isinstance(result, OrganelleResult)
    assert result.provenance is not None
    assert len(result.provenance.parameters_hash) == 64


def _detail(error: OrganelleError, key: str) -> object:
    details = error.details
    assert isinstance(details, FrozenMap)
    return details[key]


def _add_bundle() -> CapabilityBundle:
    return _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
                ParameterBindingSpec(name="y", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="sum",
        )
    )


def _add(*, x: int, y: int) -> dict[str, int]:
    return {"sum": x + y}


def test_frozen_schema_matching_the_live_schema_binds_successfully() -> None:
    bound = bind_python_capability(_add_bundle(), _add, frozen_schema=None)
    frozen = bound.signature.parameter_model.model_json_schema()

    rebound = bind_python_capability(_add_bundle(), _add, frozen_schema=frozen)
    result = rebound.invoke(None, {"x": 2, "y": 3})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["sum"] == {"sum": 5}


def test_frozen_schema_with_an_added_field_is_detected_as_drift() -> None:
    bound = bind_python_capability(_add_bundle(), _add, frozen_schema=None)
    frozen = bound.signature.parameter_model.model_json_schema()
    frozen["properties"] = {**frozen["properties"], "z": {"type": "integer"}}

    with pytest.raises(OrganelleContractError) as excinfo:
        bind_python_capability(_add_bundle(), _add, frozen_schema=frozen)
    assert excinfo.value.code == "capability.schema_drift"
    assert _detail(excinfo.value, "operation_id") == "demo.probe"
    details = excinfo.value.details
    assert isinstance(details, FrozenMap)
    assert "frozen_schema_hash" in details
    assert "live_schema_hash" in details


def test_frozen_schema_with_a_removed_field_is_detected_as_drift() -> None:
    bound = bind_python_capability(_add_bundle(), _add, frozen_schema=None)
    frozen = bound.signature.parameter_model.model_json_schema()
    properties = dict(frozen["properties"])
    del properties["y"]
    frozen["properties"] = properties
    frozen["required"] = [name for name in frozen["required"] if name != "y"]

    with pytest.raises(OrganelleContractError) as excinfo:
        bind_python_capability(_add_bundle(), _add, frozen_schema=frozen)
    assert excinfo.value.code == "capability.schema_drift"


def test_frozen_schema_with_a_changed_required_list_is_detected_as_drift() -> None:
    bound = bind_python_capability(_add_bundle(), _add, frozen_schema=None)
    frozen = bound.signature.parameter_model.model_json_schema()
    frozen["required"] = ["x"]  # y silently became optional

    with pytest.raises(OrganelleContractError) as excinfo:
        bind_python_capability(_add_bundle(), _add, frozen_schema=frozen)
    assert excinfo.value.code == "capability.schema_drift"


def test_frozen_schema_with_a_changed_enum_is_detected_as_drift() -> None:
    def choose(*, mode: Literal["fast", "slow"]) -> dict[str, str]:
        return {"mode": mode}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(ParameterBindingSpec(name="mode", codec=ParameterCodec.JSON),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, choose, frozen_schema=None)
    frozen = bound.signature.parameter_model.model_json_schema()
    frozen["properties"]["mode"]["enum"] = ["fast", "slow", "turbo"]

    with pytest.raises(OrganelleContractError) as excinfo:
        bind_python_capability(bundle, choose, frozen_schema=frozen)
    assert excinfo.value.code == "capability.schema_drift"


def test_frozen_schema_with_reordered_properties_is_not_flagged_as_drift() -> None:
    bound = bind_python_capability(_add_bundle(), _add, frozen_schema=None)
    frozen = bound.signature.parameter_model.model_json_schema()
    reordered = dict(frozen)
    reordered["properties"] = dict(reversed(list(frozen["properties"].items())))

    rebound = bind_python_capability(_add_bundle(), _add, frozen_schema=reordered)
    result = rebound.invoke(None, {"x": 1, "y": 1})
    assert isinstance(result, OrganelleResult)


def test_schema_drift_never_executes_the_original_callable() -> None:
    calls: list[tuple[int, int]] = []

    def counting_add(*, x: int, y: int) -> dict[str, int]:
        calls.append((x, y))
        return {"sum": x + y}

    bound = bind_python_capability(_add_bundle(), counting_add, frozen_schema=None)
    frozen = bound.signature.parameter_model.model_json_schema()
    frozen["properties"] = {**frozen["properties"], "z": {"type": "integer"}}

    with pytest.raises(OrganelleContractError):
        bind_python_capability(_add_bundle(), counting_add, frozen_schema=frozen)
    assert calls == []


def _executor_bundle() -> CapabilityBundle:
    return _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="query", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
                ParameterBindingSpec(
                    name="executor", codec=ParameterCodec.JSON, source=ParameterSource.ENVIRONMENT
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )


def test_environment_provider_injects_a_required_executor_parameter() -> None:
    sentinel_executor = object()
    received: dict[str, object] = {}

    def probe(*, query: str, executor: object) -> dict[str, str]:
        received["executor"] = executor
        return {"query": query}

    def provider(
        spec: object, environment_bindings: tuple[ParameterBindingSpec, ...]
    ) -> dict[str, object]:
        assert {binding.name for binding in environment_bindings} == {"executor"}
        return {"executor": sentinel_executor}

    bound = bind_python_capability(
        _executor_bundle(), probe, frozen_schema=None, environment_provider=provider
    )

    # Agent-facing schema never mentions the environment-sourced parameter.
    schema = bound.signature.parameter_model.model_json_schema()
    assert "executor" not in schema["properties"]

    result = bound.invoke(None, {"query": "x"})
    assert isinstance(result, OrganelleResult)
    assert received["executor"] is sentinel_executor  # exact same object, not a copy


def test_missing_environment_provider_fails_before_executing_the_scientific_function() -> None:
    received: dict[str, object] = {}

    def probe(*, query: str, executor: object) -> dict[str, str]:
        received["executor"] = executor
        return {"query": query}

    bound = bind_python_capability(_executor_bundle(), probe, frozen_schema=None)
    with pytest.raises(OrganelleContractError):
        bound.invoke(None, {"query": "x"})
    assert received == {}


def test_environment_provider_missing_a_required_parameter_is_rejected() -> None:
    def probe(*, query: str, executor: object) -> dict[str, str]:
        return {"query": query}

    def empty_provider(
        spec: object, environment_bindings: tuple[ParameterBindingSpec, ...]
    ) -> dict[str, object]:
        return {}

    bound = bind_python_capability(
        _executor_bundle(), probe, frozen_schema=None, environment_provider=empty_provider
    )
    with pytest.raises(OrganelleContractError) as excinfo:
        bound.invoke(None, {"query": "x"})
    assert excinfo.value.code == "capability.environment_parameter_missing"


def test_environment_provider_returning_an_undeclared_parameter_is_rejected() -> None:
    def probe(*, query: str, executor: object) -> dict[str, str]:
        return {"query": query}

    def over_providing(
        spec: object, environment_bindings: tuple[ParameterBindingSpec, ...]
    ) -> dict[str, object]:
        return {"executor": object(), "unexpected": object()}

    bound = bind_python_capability(
        _executor_bundle(), probe, frozen_schema=None, environment_provider=over_providing
    )
    with pytest.raises(OrganelleContractError):
        bound.invoke(None, {"query": "x"})


@pytest.mark.parametrize(
    "codec", [ParameterCodec.LEGACY_GENOME, ParameterCodec.LEGACY_DATA]
)
def test_unimplemented_legacy_codec_is_rejected_at_bind_time(codec: ParameterCodec) -> None:
    """v1 implements no legacy_genome/legacy_data decoder: reject at bind time.

    legacy_result left this set with its own decoder (Ruling 2, Decision 004
    approval 2026-08-14) - the agent-facing field IS the L1 OrganelleResult
    model; see tests/operations/test_legacy_result_codec.py."""
    calls: list[object] = []

    def probe(*, x: object) -> dict[str, str]:
        calls.append(x)
        return {"x": str(x)}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="x",
                    codec=codec,
                    source=ParameterSource.AGENT,
                    json_schema={"type": "object", "additionalProperties": False},
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)
    assert calls == []


# --- binding completeness: every uncovered, no-default parameter is rejected -----


def test_bind_rejects_an_uncovered_required_parameter() -> None:
    calls: list[object] = []

    def probe(*, x: int, y: int) -> dict[str, int]:
        calls.append((x, y))
        return {"x": x, "y": y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)
    assert calls == []


def test_bind_accepts_an_uncovered_parameter_that_has_a_default() -> None:
    def probe(*, x: int, y: int = 7) -> dict[str, int]:
        return {"x": x, "y": y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    result = bound.invoke(None, {"x": 1})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"x": 1, "y": 7}


def test_bind_rejects_a_nonexistent_core_input_parameter() -> None:
    def probe(*, y: int) -> dict[str, int]:
        return {"y": y}

    bundle = _bundle(
        input_kind="data",
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            core_input_parameter="genome_that_does_not_exist",
            parameters=(ParameterBindingSpec(name="y", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        ),
    )
    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)


def test_bind_rejects_a_core_input_parameter_that_is_also_a_named_binding() -> None:
    def probe(*, x: int) -> dict[str, int]:
        return {"x": x}

    bundle = _bundle(
        input_kind="data",
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            core_input_parameter="x",
            parameters=(ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        ),
    )
    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)


def test_bind_accepts_an_untargeted_var_keyword_parameter() -> None:
    """A **kwargs escape hatch nothing ever targets is harmless: it always
    receives zero items through named_parameters binding's keyword-only
    call. Matches a real released capability, plot_ogdraw_map."""

    def probe(*, y: int, **kwargs: object) -> dict[str, int]:
        assert kwargs == {}
        return {"y": y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(ParameterBindingSpec(name="y", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    result = bound.invoke(None, {"y": 1})
    assert isinstance(result, OrganelleResult)


def test_bind_accepts_an_untargeted_positional_only_parameter_with_a_default() -> None:
    def probe(x: int = 1, /, *, y: int = 2) -> dict[str, int]:
        return {"x": x, "y": y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    result = bound.invoke(None, {})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"x": 1, "y": 2}


def test_bind_rejects_an_untargeted_required_positional_only_parameter() -> None:
    """Untargeted is fine only when a default exists; without one it is
    still an uncovered required parameter, rejected like any other."""

    def probe(x: int, /, *, y: int = 2) -> dict[str, int]:
        return {"x": x, "y": y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)


def test_bind_rejects_an_agent_binding_targeting_a_var_positional_parameter_by_name() -> None:
    """*args's own listed name ("args") matches python_signature.parameters,
    so a careless agent binding could pass the existence check and only
    fail later with a bare TypeError from Signature.bind - reject it here."""

    def probe(*args: int, y: int = 1) -> dict[str, int]:
        return {"y": y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(ParameterBindingSpec(name="args", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)


def test_bind_rejects_an_agent_binding_targeting_a_var_keyword_parameter_by_name() -> None:
    def probe(*, y: int = 1, **kwargs: object) -> dict[str, int]:
        return {"y": y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="kwargs", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)


def test_bind_rejects_core_input_parameter_targeting_a_positional_only_parameter() -> None:
    def probe(genome: object, /, *, y: int = 1) -> dict[str, int]:
        return {"y": y}

    bundle = _bundle(
        input_kind="data",
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            core_input_parameter="genome",
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        ),
    )
    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)


def test_invocation_failures_never_leak_a_bare_type_error() -> None:
    """Every binding-completeness failure surfaces as a structured
    OrganelleError, never a bare inspect.Signature.bind TypeError."""

    def probe(*, x: int, y: int) -> dict[str, int]:
        return {"x": x, "y": y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    try:
        bind_python_capability(bundle, probe, frozen_schema=None)
        pytest.fail("expected bind_python_capability to reject the uncovered parameter")
    except TypeError:
        pytest.fail("a bare TypeError leaked out of bind_python_capability")
    except OrganelleError:
        pass


def test_bind_python_capability_rejects_a_composite_bundle() -> None:
    """composite bundles have no single Python callable to bind - v1 never
    executes them; bind_python_capability must refuse immediately."""
    calls: list[object] = []

    def probe() -> dict[str, str]:
        calls.append(object())
        return {}

    bundle = CapabilityBundle.model_validate(
        {
            "schema": "organelleverse.capability.v1",
            "capability": {
                "id": "demo.composite_probe",
                "bundle_version": "1.0.0",
                "implementation": "composite",
            },
            "contract": {
                "operation_id": "demo.composite_probe",
                "contract_version": "1.0",
                "title": "Demo composite capability",
                "description": "Test fixture contract used only to probe composite rejection.",
                "keywords": ("demo", "fixture", "test"),
                "execution_mode": ExecutionMode.INLINE,
                "stage": "analyze",
                "input_kind": "none",
                "output_kind": "result",
            },
            "composite": {
                "steps": [
                    {"id": "step_one", "invocation": {"capability_id": "demo.step_one"}}
                ],
            },
        }
    )
    assert bundle.capability.implementation is ImplementationKind.COMPOSITE

    with pytest.raises(OrganelleContractError) as excinfo:
        bind_python_capability(bundle, probe, frozen_schema=None)
    assert excinfo.value.code == "capability.implementation_not_supported"
    assert calls == []


def test_binding_never_mutates_the_original_signature() -> None:
    def probe(*, x: int) -> dict[str, int]:
        return {"x": x}

    original_signature = inspect.signature(probe)
    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(ParameterBindingSpec(name="x", codec=ParameterCodec.JSON),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bind_python_capability(bundle, probe, frozen_schema=None)
    assert inspect.signature(probe) == original_signature


# --- parameter provenance preflight: hashing runs before the callable, ever -------


def _metadata_bundle() -> CapabilityBundle:
    return _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="metadata", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )


_METADATA_PAYLOAD: dict[str, object] = {
    "species": "Arabidopsis thaliana",
    "genetic_code": 1,
    "assembly_type": "complete",
    "accession": "",
    "plastid_type": "",
    "source": "",
}


def test_operation_parameter_model_typed_parameter_invokes_successfully() -> None:
    """Reproduces the exact prior crash: a JSON-codec field typed
    OperationParameterModel (OrganelleMetadata) makes Pydantic construct a
    real model instance in agent_kwargs, which plain json.dumps cannot
    serialize. Must now succeed end-to-end, not crash after the callable ran."""

    def probe(*, metadata: OrganelleMetadata) -> dict[str, str]:
        return {"species": metadata.species}

    bound = bind_python_capability(_metadata_bundle(), probe, frozen_schema=None)
    result = bound.invoke(None, {"metadata": _METADATA_PAYLOAD})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"species": "Arabidopsis thaliana"}
    assert result.provenance is not None
    assert len(result.provenance.parameters_hash) == 64


def test_parameters_hash_is_stable_for_a_model_typed_parameter() -> None:
    def probe(*, metadata: OrganelleMetadata) -> dict[str, str]:
        return {"species": metadata.species}

    bound = bind_python_capability(_metadata_bundle(), probe, frozen_schema=None)
    first = bound.invoke(None, {"metadata": dict(_METADATA_PAYLOAD)})
    second = bound.invoke(None, {"metadata": dict(_METADATA_PAYLOAD)})
    assert isinstance(first, OrganelleResult)
    assert isinstance(second, OrganelleResult)
    assert first.provenance is not None and second.provenance is not None
    assert first.provenance.parameters_hash == second.provenance.parameters_hash

    changed_payload = {**_METADATA_PAYLOAD, "species": "Oryza sativa"}
    third = bound.invoke(None, {"metadata": changed_payload})
    assert isinstance(third, OrganelleResult)
    assert third.provenance is not None
    assert third.provenance.parameters_hash != first.provenance.parameters_hash


def test_provenance_preflight_rejects_a_non_finite_float_before_invoking_the_callable() -> None:
    """A DATACLASS/NUMPY_ARTIFACT-codec parameter's Agent-facing field is
    typed Any (json_schema_extra only), and jsonschema's own {"type":
    "number"} does not reject NaN/Infinity - so a non-finite float can reach
    agent_kwargs unless the provenance preflight itself checks for one."""
    calls: list[object] = []

    def area(*, rectangle: object) -> dict[str, float]:
        from tests.capabilities.fixtures_for_binding import Rectangle

        calls.append(rectangle)
        assert isinstance(rectangle, Rectangle)
        return {"area": rectangle.width * rectangle.height}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="rectangle",
                    codec=ParameterCodec.DATACLASS,
                    source=ParameterSource.AGENT,
                    target_type_locator="tests.capabilities.fixtures_for_binding:Rectangle",
                    json_schema={
                        "type": "object",
                        "properties": {"width": {"type": "number"}, "height": {"type": "number"}},
                        "required": ["width", "height"],
                        "additionalProperties": False,
                    },
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, area, frozen_schema=None)

    with pytest.raises(OrganelleContractError) as excinfo:
        bound.invoke(None, {"rectangle": {"width": float("nan"), "height": 4}})
    assert excinfo.value.code == "capability.parameter_provenance_invalid"
    assert calls == []


def test_provenance_preflight_rejects_infinity_before_invoking_the_callable() -> None:
    calls: list[object] = []

    def area(*, rectangle: object) -> dict[str, float]:
        from tests.capabilities.fixtures_for_binding import Rectangle

        calls.append(rectangle)
        assert isinstance(rectangle, Rectangle)
        return {"area": rectangle.width * rectangle.height}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="rectangle",
                    codec=ParameterCodec.DATACLASS,
                    source=ParameterSource.AGENT,
                    target_type_locator="tests.capabilities.fixtures_for_binding:Rectangle",
                    json_schema={
                        "type": "object",
                        "properties": {"width": {"type": "number"}, "height": {"type": "number"}},
                        "required": ["width", "height"],
                        "additionalProperties": False,
                    },
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, area, frozen_schema=None)

    with pytest.raises(OrganelleContractError) as excinfo:
        bound.invoke(None, {"rectangle": {"width": float("inf"), "height": 4}})
    assert excinfo.value.code == "capability.parameter_provenance_invalid"
    assert calls == []


def test_provenance_preflight_failure_never_leaks_a_bare_type_error() -> None:
    def area(*, rectangle: object) -> dict[str, float]:
        from tests.capabilities.fixtures_for_binding import Rectangle

        assert isinstance(rectangle, Rectangle)
        return {"area": rectangle.width * rectangle.height}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="rectangle",
                    codec=ParameterCodec.DATACLASS,
                    source=ParameterSource.AGENT,
                    target_type_locator="tests.capabilities.fixtures_for_binding:Rectangle",
                    json_schema={
                        "type": "object",
                        "properties": {"width": {"type": "number"}, "height": {"type": "number"}},
                        "required": ["width", "height"],
                        "additionalProperties": False,
                    },
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, area, frozen_schema=None)
    try:
        bound.invoke(None, {"rectangle": {"width": float("nan"), "height": 4}})
        pytest.fail("expected the non-finite float to be rejected")
    except TypeError:
        pytest.fail("a bare TypeError leaked out of bound.invoke")
    except OrganelleError:
        pass


def test_default_path_parameter_value_does_not_break_agent_schema_or_hash(tmp_path: Path) -> None:
    """A PATH-codec parameter with a *real Path* default, never submitted by
    the Agent, must not break the Agent schema (still a plain string type),
    the dynamically built parameter model, or provenance hashing. The
    default is a genuine ``Path`` instance - not ``str(Path(...))`` - since
    that is exactly what ``inspect.signature`` reports for a real
    ``def probe(*, fasta: Path = some_path)`` implementation; a test that
    only ever supplied an already-stringified default could never have
    caught the prior bug (Pydantic's own ``validate_default=True`` +
    ``strict=True`` rejecting a raw ``Path`` default against the ``str``
    Agent field)."""
    default_fasta = tmp_path / "default.fasta"
    default_fasta.write_text(">default\nACGT\n", encoding="utf-8")

    def probe(*, fasta: Path = default_fasta) -> dict[str, object]:
        return {"fasta": str(fasta)}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    schema = bound.signature.parameter_model.model_json_schema()
    assert schema["properties"]["fasta"]["type"] == "string"
    assert schema["properties"]["fasta"]["default"] == str(default_fasta)

    result = bound.invoke(None, {})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"fasta": str(default_fasta)}
    assert result.provenance is not None
    assert len(result.provenance.parameters_hash) == 64


# --- PATH codec: Path/str|Path defaults and Optional[...] support ---------------


def test_str_or_path_union_default_normalizes_a_real_path_object_to_a_string(tmp_path: Path) -> None:
    """``str | Path`` picks ``str`` as the conversion target (the
    implementation already accepts a raw string), but its *default* may
    still be a real ``Path`` object - that must normalize to a string before
    it ever reaches the Agent model, exactly like the ``Path``-only case."""
    default_fasta = tmp_path / "default.fasta"
    default_fasta.write_text(">default\nACGT\n", encoding="utf-8")

    def probe(*, fasta: str | Path = default_fasta) -> dict[str, object]:
        return {"fasta": str(fasta)}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    schema = bound.signature.parameter_model.model_json_schema()
    assert schema["properties"]["fasta"]["type"] == "string"
    assert schema["properties"]["fasta"]["default"] == str(default_fasta)

    result = bound.invoke(None, {})
    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"fasta": str(default_fasta)}


def test_optional_path_codec_agent_schema_allows_null() -> None:
    """A non-Optional PATH parameter's Agent field is a plain string; an
    Optional one (``str | Path | None`` etc.) must additionally allow
    ``null`` - never a self-contradictory ``"type": "string"`` schema
    carrying a ``None`` default, which is not itself a valid string."""

    def probe(*, fasta: Path | None = None) -> dict[str, object]:
        return {"fasta": None if fasta is None else str(fasta)}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    schema = bound.signature.parameter_model.model_json_schema()
    fasta_schema = schema["properties"]["fasta"]
    type_options = {entry.get("type") for entry in fasta_schema["anyOf"]}
    assert type_options == {"string", "null"}
    assert fasta_schema["default"] is None


def test_optional_path_codec_none_default_is_passed_through_without_a_file_check_or_hash() -> None:
    """None means "no artifact": no existence check, no content hash, and
    the implementation receives a real ``None`` - not a stringified
    sentinel, not a check against a nonexistent path."""
    received: dict[str, object] = {}

    def probe(*, fasta: Path | None = None) -> dict[str, object]:
        received["fasta"] = fasta
        return {"fasta": None if fasta is None else str(fasta)}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)

    result = bound.invoke(None, {})
    assert isinstance(result, OrganelleResult)
    assert received["fasta"] is None
    assert result.provenance is not None
    assert result.provenance.input_artifact_hashes == ()


def test_optional_path_codec_still_checks_and_hashes_a_submitted_path(tmp_path: Path) -> None:
    """Optionality only excuses a genuinely absent (None) value - a real,
    submitted path on an Optional PATH parameter is still existence-checked,
    hashed, and converted per the implementation's annotation exactly like a
    required one."""
    import hashlib

    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(">demo\nACGT\n", encoding="utf-8")
    expected_hash = hashlib.sha256(fasta_path.read_bytes()).hexdigest()
    received: dict[str, object] = {}

    def probe(*, fasta: Path | None = None) -> dict[str, object]:
        received["fasta"] = fasta
        return {"fasta": None if fasta is None else str(fasta)}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)

    result = bound.invoke(None, {"fasta": str(fasta_path)})
    assert isinstance(result, OrganelleResult)
    assert isinstance(received["fasta"], Path)
    assert received["fasta"] == fasta_path
    assert result.provenance is not None
    assert result.provenance.input_artifact_hashes == (expected_hash,)


def test_optional_path_codec_rejects_a_missing_file_even_though_none_is_allowed(
    tmp_path: Path,
) -> None:
    """Optional means "None is allowed", not "existence-checking is
    optional" - a caller who does submit a path still gets it verified
    before the scientific function ever runs."""
    calls: list[object] = []

    def probe(*, fasta: Path | None = None) -> dict[str, object]:
        calls.append(fasta)
        return {"fasta": None if fasta is None else str(fasta)}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    missing = tmp_path / "does-not-exist.fasta"
    with pytest.raises(OrganelleError):
        bound.invoke(None, {"fasta": str(missing)})
    assert calls == []


def test_str_or_path_or_none_union_annotation_is_supported(tmp_path: Path) -> None:
    """The full three-way union from the audit: ``str | Path | None``.
    Resolves to the ``str`` conversion target (str is present in the union)
    while still being Optional (None is also present)."""
    fasta_path = tmp_path / "genome.fasta"
    fasta_path.write_text(">demo\nACGT\n", encoding="utf-8")
    received: dict[str, object] = {}

    def probe(*, fasta: str | Path | None = None) -> dict[str, object]:
        received["fasta"] = fasta
        return {"fasta": fasta}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    schema = bound.signature.parameter_model.model_json_schema()
    type_options = {entry.get("type") for entry in schema["properties"]["fasta"]["anyOf"]}
    assert type_options == {"string", "null"}

    submitted = bound.invoke(None, {"fasta": str(fasta_path)})
    assert isinstance(submitted, OrganelleResult)
    assert type(received["fasta"]) is str
    assert received["fasta"] == str(fasta_path)

    omitted = bound.invoke(None, {})
    assert isinstance(omitted, OrganelleResult)
    assert received["fasta"] is None


def test_path_codec_rejects_a_none_default_when_the_annotation_is_not_optional() -> None:
    """``fasta: Path = None`` is internally inconsistent - the annotation
    never names ``None`` as acceptable, so a ``None`` default cannot be
    honored. Rejected at bind time, not left to silently coerce or crash
    later."""

    def probe(*, fasta: Path = None) -> dict[str, object]:  # noqa: RUF013 # type: ignore[assignment]
        return {"fasta": str(fasta)}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)


def test_path_codec_rejects_an_unsupported_default_type() -> None:
    """A PATH-codec default that is neither ``str``, ``Path``, nor (when
    Optional) ``None`` has no defined normalization - rejected at bind time
    rather than silently passed through as some other JSON-incompatible
    type."""

    def probe(*, fasta: Path = 123) -> dict[str, object]:  # type: ignore[assignment]
        return {"fasta": str(fasta)}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    with pytest.raises(OrganelleContractError):
        bind_python_capability(bundle, probe, frozen_schema=None)


# --- JSON codec: bare Path is not JSON-safe at the bundle codec wall ---


def test_json_codec_rejects_path_annotations() -> None:
    # bare Path, and Path nested in a list/union, are not JSON-safe for the
    # bundle codec wall: a Path-typed parameter must take the PATH codec
    # (containment + hashing), never the containment-free JSON codec.
    # allow_path=True (the default) is derive_operation_signature's registry
    # admission for the 20 released ops - deliberately unchanged, out of scope.
    from organelleverse.operations.signature import (
        _is_supported_json_annotation,  # pyright: ignore[reportPrivateUsage]
    )

    assert _is_supported_json_annotation(Path, allow_path=False) is False
    assert _is_supported_json_annotation(list[str | Path], allow_path=False) is False
    assert _is_supported_json_annotation(str | Path, allow_path=False) is False
    assert _is_supported_json_annotation(Path) is True  # registry admission unchanged


# --- PATH codec: list-of-paths parameters (multiple=True plan) ---


def _path_list_bundle() -> CapabilityBundle:
    return _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="fastas", codec=ParameterCodec.PATH, source=ParameterSource.AGENT
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )


def test_path_list_agent_field_is_a_list_of_json_strings() -> None:
    """The Agent-facing field for a ``list[str | Path]`` PATH parameter is
    ``list[str]`` - JSON has no path type, exactly like the scalar PATH codec."""

    def probe(*, fastas: list[str | Path]) -> dict[str, int]:
        return {"n": len(fastas)}

    bound = bind_python_capability(_path_list_bundle(), probe, frozen_schema=None)
    schema = bound.signature.parameter_model.model_json_schema()
    assert schema["properties"]["fastas"]["type"] == "array"
    assert schema["properties"]["fastas"]["items"] == {"type": "string"}


def test_path_list_invokes_with_real_files_and_hashes_every_element(tmp_path: Path) -> None:
    import hashlib

    first = tmp_path / "one.fasta"
    second = tmp_path / "two.fasta"
    first.write_text(">a\nACGT\n", encoding="utf-8")
    second.write_text(">b\nTGCA\n", encoding="utf-8")
    expected_hashes = (
        hashlib.sha256(first.read_bytes()).hexdigest(),
        hashlib.sha256(second.read_bytes()).hexdigest(),
    )
    received: dict[str, object] = {}

    def probe(*, fastas: list[str | Path]) -> dict[str, int]:
        received["fastas"] = fastas
        return {"n": len(fastas)}

    bound = bind_python_capability(_path_list_bundle(), probe, frozen_schema=None)
    result = bound.invoke(None, {"fastas": [str(first), str(second)]})

    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == {"n": 2}
    # the union names str, so the implementation receives the raw strings
    assert received["fastas"] == [str(first), str(second)]
    assert result.provenance is not None
    assert result.provenance.input_artifact_hashes == expected_hashes


def test_path_list_converts_to_path_objects_when_the_element_names_only_path(
    tmp_path: Path,
) -> None:
    first = tmp_path / "one.fasta"
    first.write_text(">a\nACGT\n", encoding="utf-8")
    received: dict[str, object] = {}

    def probe(*, fastas: list[Path]) -> dict[str, int]:
        received["fastas"] = fastas
        return {"n": len(fastas)}

    bound = bind_python_capability(_path_list_bundle(), probe, frozen_schema=None)
    bound.invoke(None, {"fastas": [str(first)]})

    value = received["fastas"]
    assert isinstance(value, list)
    assert all(isinstance(element, Path) and not isinstance(element, str) for element in value)
    assert value == [first]


def test_path_list_accepts_a_sequence_annotation(tmp_path: Path) -> None:
    """``Sequence[...]`` resolves to the same origin as ``list[...]`` (mirrors
    ``signature.py``'s ``_JSON_SEQUENCE_ORIGINS``), so it takes the same plan."""
    first = tmp_path / "one.fasta"
    first.write_text(">a\nACGT\n", encoding="utf-8")
    received: dict[str, object] = {}

    def probe(*, fastas: Sequence[str]) -> dict[str, int]:
        received["fastas"] = fastas
        return {"n": len(fastas)}

    bound = bind_python_capability(_path_list_bundle(), probe, frozen_schema=None)
    result = bound.invoke(None, {"fastas": [str(first)]})

    assert isinstance(result, OrganelleResult)
    assert received["fastas"] == [str(first)]


def test_path_list_rejects_a_missing_element_before_invoking_the_scientific_function(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    def probe(*, fastas: list[str | Path]) -> dict[str, int]:
        calls.append(fastas)
        return {"n": len(fastas)}

    present = tmp_path / "present.fasta"
    present.write_text(">a\nACGT\n", encoding="utf-8")
    missing = tmp_path / "does-not-exist.fasta"

    bound = bind_python_capability(_path_list_bundle(), probe, frozen_schema=None)
    with pytest.raises(OrganelleInputError) as excinfo:
        bound.invoke(None, {"fastas": [str(present), str(missing)]})
    assert excinfo.value.code == "input.missing_artifact"
    assert calls == []


def test_path_list_rejects_a_bare_dict_element_shape_at_bind_time() -> None:
    """``list[dict]`` declared PATH: a dict element is not a path-like
    annotation, so it is rejected at bind time, never deferred to invoke."""

    def probe(*, fastas: list[dict]) -> dict[str, int]:
        return {"n": len(fastas)}

    with pytest.raises(OrganelleContractError) as excinfo:
        bind_python_capability(_path_list_bundle(), probe, frozen_schema=None)
    assert excinfo.value.code == "capability.binding_invalid"


def test_path_list_rejects_a_none_element_at_bind_time() -> None:
    """A list element cannot be "no artifact": ``list[str | None]`` declared
    PATH is rejected at bind time."""

    def probe(*, fastas: list[str | None]) -> dict[str, int]:
        return {"n": len(fastas)}

    with pytest.raises(OrganelleContractError) as excinfo:
        bind_python_capability(_path_list_bundle(), probe, frozen_schema=None)
    assert excinfo.value.code == "capability.binding_invalid"


def test_path_list_rejects_a_default_at_bind_time() -> None:
    """No optional-list v1: a ``multiple`` plan with a non-``...`` default is
    a bind error in ``_normalize_path_default``."""

    def probe(*, fastas: list[str | Path] = ()) -> dict[str, int]:  # type: ignore[assignment]
        return {"n": len(fastas)}

    with pytest.raises(OrganelleContractError) as excinfo:
        bind_python_capability(_path_list_bundle(), probe, frozen_schema=None)
    assert excinfo.value.code == "capability.binding_invalid"


def test_parameters_hash_reflects_validated_and_defaulted_agent_parameters() -> None:
    """An agent-declared field with a default, omitted by the caller, still
    enters the hash at its resolved default - not silently absent."""

    def probe(*, x: int = 42) -> dict[str, int]:
        return {"x": x}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )
    bound = bind_python_capability(bundle, probe, frozen_schema=None)
    omitted = bound.invoke(None, {})
    explicit = bound.invoke(None, {"x": 42})
    assert isinstance(omitted, OrganelleResult)
    assert isinstance(explicit, OrganelleResult)
    assert omitted.provenance is not None and explicit.provenance is not None
    assert omitted.provenance.parameters_hash == explicit.provenance.parameters_hash


# --- L6 preflight: run_id + CapabilityExecutionContext construct before the callable ---


def test_scientific_callable_is_never_invoked_when_run_id_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_id generation is part of the preflight, not a detail folded into
    result encoding after the callable already ran - if it fails, the
    scientific callable must never have been invoked."""
    import organelleverse.operations.python_binding as python_binding_module

    calls: list[object] = []

    def add(*, x: int, y: int) -> dict[str, int]:
        calls.append((x, y))
        return {"sum": x + y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
                ParameterBindingSpec(name="y", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="addition",
        )
    )
    bound = bind_python_capability(bundle, add, frozen_schema=None)

    def failing_run_id() -> str:
        raise RuntimeError("run_id generation unavailable")

    monkeypatch.setattr(python_binding_module, "_new_run_id", failing_run_id)

    with pytest.raises(RuntimeError):
        bound.invoke(None, {"x": 2, "y": 3})
    assert calls == []


def test_scientific_callable_is_never_invoked_when_execution_context_fails_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run_id that fails CapabilityExecutionContext's own pattern
    validation proves the context is actually *validated*, not merely
    constructed, before the callable runs."""
    import organelleverse.operations.python_binding as python_binding_module

    calls: list[object] = []

    def add(*, x: int, y: int) -> dict[str, int]:
        calls.append((x, y))
        return {"sum": x + y}

    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(name="x", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
                ParameterBindingSpec(name="y", codec=ParameterCodec.JSON, source=ParameterSource.AGENT),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="addition",
        )
    )
    bound = bind_python_capability(bundle, add, frozen_schema=None)

    monkeypatch.setattr(python_binding_module, "_new_run_id", lambda: "Not A Valid Run Id!")

    with pytest.raises(ValidationError):
        bound.invoke(None, {"x": 2, "y": 3})
    assert calls == []


# --- worker contract: the codec whitelist stays {JSON, PATH} ---------------------
#
# The directory codec exists at the contract level (spec.py) but has no
# bundle-local controlled-worker decoder; the worker whitelist in
# python_binding._validate_worker_contract_spec must keep rejecting it with
# the existing diagnostic rather than growing a silent third case.


def test_worker_contract_rejects_a_directory_codec_parameter() -> None:
    bundle = _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="data_root",
                    codec=ParameterCodec.DIRECTORY,
                    path_role="input",
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="metrics",
        )
    )

    with pytest.raises(OrganelleContractError, match="JSON and path parameters only") as captured:
        validate_worker_contract(bundle)

    assert captured.value.code == "capability.execution_provider_required"


# --- DIRECTORY codec: input role - PATH parity plus subtree provenance --------
#
# An input-role directory parameter is the same grant as PATH, one subtree
# instead of one file: resolve(strict=True), is_dir, then a deterministic
# manifest hash of the whole tree enters input_artifact_hashes. An
# output-role directory parameter is an Agent-chosen *destination* and gets
# the designed containment rules instead (no parent traversal, no symlink
# components, a real directory, the managed run store off-limits) - and every
# rejection completes before the scientific callable is ever invoked.


def _directory_bundle(name: str, path_role: Literal["input", "output"]) -> CapabilityBundle:
    return _bundle(
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name=name,
                    codec=ParameterCodec.DIRECTORY,
                    source=ParameterSource.AGENT,
                    path_role=path_role,
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="value",
        )
    )


def _tree_manifest_hash(root: Path) -> str:
    """The deterministic tree manifest hash the DIRECTORY codec must produce."""
    manifest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        manifest.update(path.relative_to(root).as_posix().encode("utf-8"))
        manifest.update(b"\0")
        manifest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        manifest.update(b"\n")
    return manifest.hexdigest()


def test_directory_codec_agent_schema_is_a_json_string() -> None:
    def probe(*, data_root: Path) -> dict[str, str]:
        return {"root": str(data_root)}

    bound = bind_python_capability(
        _directory_bundle("data_root", "input"), probe, frozen_schema=None
    )
    schema = bound.signature.parameter_model.model_json_schema()
    assert schema["properties"]["data_root"]["type"] == "string"


def test_directory_input_rejects_a_missing_directory_before_invoking(tmp_path: Path) -> None:
    calls: list[object] = []

    def probe(*, data_root: Path) -> dict[str, str]:
        calls.append(data_root)
        return {"root": str(data_root)}

    bound = bind_python_capability(
        _directory_bundle("data_root", "input"), probe, frozen_schema=None
    )
    with pytest.raises(OrganelleInputError) as captured:
        bound.invoke(None, {"data_root": str(tmp_path / "does-not-exist")})
    assert captured.value.code == "input.missing_artifact"
    assert calls == []


def test_directory_input_rejects_a_regular_file_as_the_wrong_kind(tmp_path: Path) -> None:
    a_file = tmp_path / "a-file.txt"
    a_file.write_text("not a directory", encoding="utf-8")
    calls: list[object] = []

    def probe(*, data_root: Path) -> dict[str, str]:
        calls.append(data_root)
        return {"root": str(data_root)}

    bound = bind_python_capability(
        _directory_bundle("data_root", "input"), probe, frozen_schema=None
    )
    with pytest.raises(OrganelleInputError) as captured:
        bound.invoke(None, {"data_root": str(a_file)})
    assert captured.value.code == "input.missing_artifact"
    assert "expected a directory but found a non-directory" in captured.value.message
    assert calls == []


def test_directory_input_manifest_hash_tracks_the_tree_content(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "a.txt").write_text("alpha", encoding="utf-8")
    (tree / "sub" / "b.txt").write_text("beta", encoding="utf-8")

    def probe(*, data_root: Path) -> dict[str, str]:
        return {"root": str(data_root)}

    bound = bind_python_capability(
        _directory_bundle("data_root", "input"), probe, frozen_schema=None
    )

    first = bound.invoke(None, {"data_root": str(tree)})
    assert isinstance(first, OrganelleResult)
    assert first.provenance is not None
    assert first.provenance.input_artifact_hashes == (_tree_manifest_hash(tree.resolve()),)

    (tree / "sub" / "b.txt").write_text("gamma", encoding="utf-8")
    second = bound.invoke(None, {"data_root": str(tree)})
    assert second.provenance is not None
    assert second.provenance.input_artifact_hashes == (_tree_manifest_hash(tree.resolve()),)
    assert first.provenance.input_artifact_hashes != second.provenance.input_artifact_hashes


def test_directory_input_gives_a_str_annotated_implementation_the_raw_string(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    received: dict[str, object] = {}

    def probe(*, data_root: str) -> dict[str, str]:
        received["data_root"] = data_root
        return {"root": data_root}

    bound = bind_python_capability(
        _directory_bundle("data_root", "input"), probe, frozen_schema=None
    )
    bound.invoke(None, {"data_root": str(tree)})
    assert type(received["data_root"]) is str
    assert received["data_root"] == str(tree)


def test_directory_input_gives_a_path_annotated_implementation_a_resolved_path(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    received: dict[str, object] = {}

    def probe(*, data_root: Path) -> dict[str, str]:
        received["data_root"] = data_root
        return {"root": str(data_root)}

    bound = bind_python_capability(
        _directory_bundle("data_root", "input"), probe, frozen_schema=None
    )
    bound.invoke(None, {"data_root": str(tree)})
    assert isinstance(received["data_root"], Path)
    assert not isinstance(received["data_root"], str)
    assert received["data_root"] == tree.resolve()


# --- DIRECTORY codec: output role - containment for an Agent-chosen destination -


@pytest.mark.parametrize("destination", ["../escape", "sub/../escape"])
def test_directory_output_rejects_parent_traversal(destination: str, tmp_path: Path) -> None:
    calls: list[object] = []

    def probe(*, output_dir: Path) -> dict[str, str]:
        calls.append(output_dir)
        return {"output_dir": str(output_dir)}

    bound = bind_python_capability(
        _directory_bundle("output_dir", "output"), probe, frozen_schema=None
    )
    with pytest.raises(OrganelleInputError) as captured:
        bound.invoke(None, {"output_dir": destination})
    assert captured.value.code == "input.directory_destination_traversal"
    assert calls == []


def test_directory_output_rejects_an_empty_destination(tmp_path: Path) -> None:
    calls: list[object] = []

    def probe(*, output_dir: Path) -> dict[str, str]:
        calls.append(output_dir)
        return {"output_dir": str(output_dir)}

    bound = bind_python_capability(
        _directory_bundle("output_dir", "output"), probe, frozen_schema=None
    )
    with pytest.raises(OrganelleInputError) as captured:
        bound.invoke(None, {"output_dir": ""})
    assert captured.value.code == "input.directory_destination_traversal"
    assert calls == []


def test_directory_output_rejects_a_symlink_component_in_the_tail(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    # A dangling symlink: .exists() is False, so the ancestor walk stops above
    # it and the component-by-component walk must catch it instead.
    (real / "dangling").symlink_to(tmp_path / "no-such-target")
    destination = real / "dangling" / "new-subdir"
    calls: list[object] = []

    def probe(*, output_dir: Path) -> dict[str, str]:
        calls.append(output_dir)
        return {"output_dir": str(output_dir)}

    bound = bind_python_capability(
        _directory_bundle("output_dir", "output"), probe, frozen_schema=None
    )
    with pytest.raises(OrganelleInputError) as captured:
        bound.invoke(None, {"output_dir": str(destination)})
    assert captured.value.code == "input.directory_destination_symlink"
    assert calls == []


def test_directory_output_rejects_an_existing_destination_that_is_not_a_real_directory(
    tmp_path: Path,
) -> None:
    a_file = tmp_path / "a-file"
    a_file.write_text("payload", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target)
    calls: list[object] = []

    def probe(*, output_dir: Path) -> dict[str, str]:
        calls.append(output_dir)
        return {"output_dir": str(output_dir)}

    bound = bind_python_capability(
        _directory_bundle("output_dir", "output"), probe, frozen_schema=None
    )
    for destination in (a_file, linked):
        with pytest.raises(OrganelleInputError) as captured:
            bound.invoke(None, {"output_dir": str(destination)})
        assert captured.value.code == "input.directory_destination_not_directory"
    assert calls == []


def test_directory_output_refuses_the_managed_run_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import organelleverse.runtime as runtime_module

    # The codec lazy-imports managed_runs_root from the runtime module at call
    # time, so patching the module attribute is exactly what it resolves.
    monkeypatch.setattr(runtime_module, "managed_runs_root", lambda: tmp_path)
    calls: list[object] = []

    def probe(*, output_dir: Path) -> dict[str, str]:
        calls.append(output_dir)
        return {"output_dir": str(output_dir)}

    bound = bind_python_capability(
        _directory_bundle("output_dir", "output"), probe, frozen_schema=None
    )
    with pytest.raises(OrganelleInputError) as captured:
        bound.invoke(None, {"output_dir": str(tmp_path / "inside-the-run-store")})
    assert captured.value.code == "input.directory_destination_run_store"
    assert calls == []


def test_directory_output_accepts_a_not_yet_existing_destination(tmp_path: Path) -> None:
    received: dict[str, object] = {}

    def probe(*, output_dir: Path) -> dict[str, str]:
        received["output_dir"] = output_dir
        return {"output_dir": str(output_dir)}

    bound = bind_python_capability(
        _directory_bundle("output_dir", "output"), probe, frozen_schema=None
    )
    destination = tmp_path / "new-output"
    result = bound.invoke(None, {"output_dir": str(destination)})
    assert isinstance(result, OrganelleResult)
    assert isinstance(received["output_dir"], Path)
    assert not isinstance(received["output_dir"], str)
    assert received["output_dir"] == destination.resolve()
    # Validation never creates the destination: creation is the
    # implementation's own act, after validation.
    assert not destination.exists()


def test_directory_output_accepts_an_existing_real_directory(tmp_path: Path) -> None:
    received: dict[str, object] = {}

    def probe(*, output_dir: str) -> dict[str, str]:
        received["output_dir"] = output_dir
        return {"output_dir": output_dir}

    bound = bind_python_capability(
        _directory_bundle("output_dir", "output"), probe, frozen_schema=None
    )
    destination = tmp_path / "existing"
    destination.mkdir()
    result = bound.invoke(None, {"output_dir": str(destination)})
    assert isinstance(result, OrganelleResult)
    assert received["output_dir"] == str(destination)
