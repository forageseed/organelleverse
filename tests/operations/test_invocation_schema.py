from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import cast

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord, OrganelleData
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import (
    ErrorDetail,
    Finding,
    OperationSuggestion,
    OrganelleResult,
)
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
)


def _artifact(*, kind: str = "sequence", sha256: str = "a" * 64) -> ArtifactRef:
    return ArtifactRef(
        kind=kind,
        uri="artifact.dat",
        format="fasta",
        media_type="text/plain",
        sha256=sha256,
        size_bytes=8,
        validated=True,
    )


def _lineage() -> LineageRecord:
    return LineageRecord(
        parent_object_ids=("data:sha256:" + "1" * 64,),
        operation_id="assembly.assemble",
        operation_version="1.0",
        parameters_hash="2" * 64,
    )


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _sequence(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)


def _data() -> OrganelleData:
    return OrganelleData.model_validate(
        {
            "modality": "matrix",
            "artifacts": {"matrix": _artifact(kind="matrix")},
            "payload": {"samples": [{"name": "leaf", "weight": 1.5}]},
            "dimensions": {"samples": 1, "features": 2},
            "metadata": {"species": "Arabidopsis thaliana"},
            "lineage": (_lineage(),),
        }
    )


def _genome() -> OrganelleGenome:
    return OrganelleGenome(
        organelle="mitochondrion",
        sequence=_artifact(),
        metadata=OrganelleMetadata(species="Arabidopsis thaliana", source="test"),
        lineage=(_lineage(),),
        source_manifests=(
            _artifact(kind="assembly_run_manifest", sha256="b" * 64).evolve(
                format="json",
                media_type="application/json",
            ),
        ),
    )


def _result() -> OrganelleResult:
    provenance = ResultProvenance(
        operation_id="annotation.annotate",
        operation_version="1.0",
        package_version="0.0.1",
        git_commit="abc123",
        parameters_hash="c" * 64,
        software_versions=FrozenMap({"annotator": {"version": "1.0"}}),
        model_hashes=FrozenMap({"genes": "d" * 64}),
        database_hashes=FrozenMap({"reference": "e" * 64}),
    )
    return OrganelleResult(
        operation_id="annotation.annotate",
        scope="mitochondrion",
        status="warning",
        metrics=FrozenMap({"coverage": 12.5, "counts": [1, 2]}),
        findings=(Finding(code="coverage.low", value=12.5, confidence=0.75),),
        artifacts=(_artifact(),),
        provenance=provenance,
        errors=(
            ErrorDetail(
                code="annotation.partial",
                message="partial annotation",
                details=FrozenMap({"stage": "trna", "attempts": 1}),
                suggested_action=FrozenMap({"retry": True}),
            ),
        ),
        suggested_operations=(
            OperationSuggestion(
                operation_id="annotation.annotate",
                reason_code="retry.partial",
                parameter_changes=FrozenMap({"backend": "external"}),
            ),
        ),
    )


def _inspect_data(value: OrganelleData) -> OrganelleResult:
    return _result()


def _inspect_genome(value: OrganelleGenome) -> OrganelleResult:
    return _result()


def _inspect_result(value: OrganelleResult) -> OrganelleResult:
    return value


def _registry_for(kind: CoreKind) -> tuple[OperationRegistry, str]:
    registry = OperationRegistry()
    operation_id = {
        CoreKind.DATA: "analysis.inspect_data",
        CoreKind.GENOME: "analysis.inspect_genome",
        CoreKind.RESULT: "report.inspect_result",
    }[kind]
    stage = OperationStage.CONSUME if kind is CoreKind.RESULT else OperationStage.ANALYZE
    spec = OperationSpec(
        operation_id=operation_id,
        contract_version="1.0",
        title="Demo invocation-schema probe",
        description="Test fixture spec used only to probe invocation schema generation.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=stage,
        input_kind=kind,
        output_kind=CoreKind.RESULT,
        callable_locator=f"organelleverse.analysis.api:{operation_id.split('.')[1]}",
    )

    if kind is CoreKind.DATA:
        registry.register(spec, _inspect_data)
    elif kind is CoreKind.GENOME:
        registry.register(spec, _inspect_genome)
    else:
        registry.register(spec, _inspect_result)
    return registry, operation_id


def _validate_instance(schema: dict[str, object], instance: dict[str, object]) -> None:
    Draft202012Validator(schema).validate(instance)  # pyright: ignore[reportUnknownMemberType]


def _schema_errors(
    schema: dict[str, object], instance: dict[str, object]
) -> list[JsonSchemaValidationError]:
    generated = Draft202012Validator(schema).iter_errors(  # pyright: ignore[reportUnknownMemberType]
        instance  # pyright: ignore[reportArgumentType]
    )
    return cast(list[JsonSchemaValidationError], list(generated))


def _mutate_data_artifact(value: dict[str, object]) -> None:
    artifacts = _mapping(value["artifacts"])
    _mapping(artifacts["matrix"])["sha256"] = "bad"


def _mutate_data_dimensions(value: dict[str, object]) -> None:
    _mapping(value["dimensions"])["samples"] = -1


def _mutate_data_payload(value: dict[str, object]) -> None:
    _mapping(value["payload"])["bad"] = b"not-json"


def _mutate_data_lineage(value: dict[str, object]) -> None:
    lineage = _sequence(value["lineage"])
    _mapping(lineage[0])["operation_id"] = "bad"


def _mutate_result_finding(value: dict[str, object]) -> None:
    findings = _sequence(value["findings"])
    _mapping(findings[0])["confidence"] = 2


def _mutate_result_error(value: dict[str, object]) -> None:
    errors = _sequence(value["errors"])
    _mapping(errors[0])["details"] = []


def _mutate_result_provenance(value: dict[str, object]) -> None:
    _mapping(value["provenance"])["software_versions"] = []


def _mutate_result_artifact(value: dict[str, object]) -> None:
    artifacts = _sequence(value["artifacts"])
    _mapping(artifacts[0])["size_bytes"] = -1


def _mutate_result_suggestion(value: dict[str, object]) -> None:
    suggestions = _sequence(value["suggested_operations"])
    _mapping(suggestions[0])["parameter_changes"] = []


@pytest.mark.parametrize(
    ("kind", "value_factory"),
    [
        (CoreKind.DATA, _data),
        (CoreKind.GENOME, _genome),
        (CoreKind.RESULT, _result),
    ],
)
def test_invocation_schema_accepts_complete_serialized_core_inputs(
    kind: CoreKind,
    value_factory: Callable[[], OrganelleData | OrganelleGenome | OrganelleResult],
) -> None:
    registry, operation_id = _registry_for(kind)
    schema = registry.invocation_schema(operation_id)
    value = value_factory()

    Draft202012Validator.check_schema(schema)
    _validate_instance(
        schema,
        {
            "operation_id": operation_id,
            "input": value.model_dump(mode="json"),
            "parameters": {},
        },
    )


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (_mutate_data_artifact, "artifacts"),
        (_mutate_data_dimensions, "dimensions"),
        (_mutate_data_payload, "payload"),
        (_mutate_data_lineage, "lineage"),
    ],
)
def test_data_invocation_schema_rejects_malformed_serialized_nested_values(
    mutate: Callable[[dict[str, object]], None],
    expected_path: str,
) -> None:
    registry, operation_id = _registry_for(CoreKind.DATA)
    serialized = deepcopy(_data().model_dump(mode="json"))
    mutate(serialized)

    errors = _schema_errors(
        registry.invocation_schema(operation_id),
        {"operation_id": operation_id, "input": serialized, "parameters": {}},
    )

    assert errors
    assert expected_path in str(errors[0].absolute_path)


def test_genome_invocation_schema_rejects_unknown_nested_artifact_fields() -> None:
    registry, operation_id = _registry_for(CoreKind.GENOME)
    serialized = deepcopy(_genome().model_dump(mode="json"))
    source_manifests = _sequence(serialized["source_manifests"])
    _mapping(source_manifests[0])["unexpected"] = True

    errors = _schema_errors(
        registry.invocation_schema(operation_id),
        {"operation_id": operation_id, "input": serialized, "parameters": {}},
    )

    assert errors
    assert "source_manifests" in str(errors[0].absolute_path)


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (_mutate_result_finding, "findings"),
        (_mutate_result_error, "errors"),
        (_mutate_result_provenance, "provenance"),
        (_mutate_result_artifact, "artifacts"),
        (_mutate_result_suggestion, "suggested_operations"),
    ],
)
def test_result_invocation_schema_rejects_malformed_serialized_nested_values(
    mutate: Callable[[dict[str, object]], None],
    expected_path: str,
) -> None:
    registry, operation_id = _registry_for(CoreKind.RESULT)
    serialized = deepcopy(_result().model_dump(mode="json"))
    mutate(serialized)

    errors = _schema_errors(
        registry.invocation_schema(operation_id),
        {"operation_id": operation_id, "input": serialized, "parameters": {}},
    )

    assert errors
    assert expected_path in str(errors[0].absolute_path)


def test_frozen_json_recursive_schema_refs_do_not_alias() -> None:
    registry, operation_id = _registry_for(CoreKind.DATA)
    schema = registry.invocation_schema(operation_id)
    definitions = _mapping(schema["$defs"])
    frozen_definitions = [
        _mapping(definition)
        for name, definition in definitions.items()
        if name.endswith("__FrozenJson")
    ]
    assert len(frozen_definitions) == 1
    branches = _sequence(frozen_definitions[0]["anyOf"])
    array_branch = next(
        _mapping(branch) for branch in branches if _mapping(branch).get("type") == "array"
    )
    object_branch = next(
        _mapping(branch) for branch in branches if _mapping(branch).get("type") == "object"
    )
    array_reference = _mapping(array_branch["items"])
    object_reference = _mapping(object_branch["additionalProperties"])

    assert array_reference == object_reference
    assert array_reference is not object_reference
    array_reference["$ref"] = "#/$defs/mutated"
    assert object_reference["$ref"] != array_reference["$ref"]
