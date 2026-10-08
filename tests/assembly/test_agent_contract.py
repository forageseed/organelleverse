from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, cast, get_args, get_type_hints
from unittest.mock import patch

import pytest
from annotated_types import Ge, Le
from pydantic import ValidationError

from organelleverse.assembly.backends import BACKENDS
from organelleverse.assembly.contracts import (
    AssemblyMethod,
    AssemblyRequest,
    GetOrganelleParameters,
    HimtParameters,
    NovoplastyParameters,
    OatkParameters,
    OvasmParameters,
    PmatParameters,
    PtgaulParameters,
    TippoParameters,
    effective_backend_parameters,
    validate_assembly_data,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import CoreKind, OperationRegistry, OperationStage, SideEffect
from organelleverse.operations import list as list_operations
from organelleverse.operations.adapters import invoke_json


def _hifi_data(read_uri: str = "reads.fastq") -> OrganelleData:
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "long_reads": ArtifactRef(
                    kind="long_read",
                    uri=read_uri,
                    format="fastq",
                    media_type="application/x-fastq",
                    sha256="a" * 64,
                    size_bytes=1024,
                )
            },
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_reads",
                    }
                ],
            },
        }
    )


def _assembly_data(
    *,
    short_libraries: list[dict[str, object]] | None = None,
    long_libraries: list[dict[str, object]] | None = None,
    contig_inputs: list[dict[str, object]] | None = None,
    auxiliary: dict[str, object] | None = None,
) -> OrganelleData:
    artifacts: dict[str, ArtifactRef] = {}
    payload: dict[str, object] = {
        "contract_version": "organelleverse.assembly-input.v1",
        "short_libraries": short_libraries or [],
        "long_libraries": long_libraries or [],
        "contig_inputs": contig_inputs or [],
        "auxiliary": auxiliary or {},
    }
    roles: set[str] = set()
    for library in short_libraries or []:
        roles.add(cast(str, library["read1_artifact"]))
        read2 = library.get("read2_artifact")
        if isinstance(read2, str):
            roles.add(read2)
    for library in long_libraries or []:
        roles.add(cast(str, library["reads_artifact"]))
    for item in contig_inputs or []:
        roles.add(cast(str, item["fasta_artifact"]))
    for name, value in (auxiliary or {}).items():
        if name.endswith("_artifact") and isinstance(value, str):
            roles.add(value)
    for role in roles:
        artifacts[role] = ArtifactRef(
            kind="assembly_input",
            uri=f"{role}.dat",
            format="binary",
            media_type="application/octet-stream",
            sha256="a" * 64,
            size_bytes=1,
        )
    return OrganelleData.model_validate(
        {"modality": "sequencing_reads", "artifacts": artifacts, "payload": payload}
    )


def _candidate_registry() -> OperationRegistry:
    from organelleverse.assembly.api import assemble as canonical_assemble
    from organelleverse.assembly.data_contract import (
        RELEASED_ASSEMBLY_SEQUENCING_READS_DATA_CONTRACT,
    )
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    registry = OperationRegistry()
    registry.register(
        ASSEMBLE_SPEC,
        canonical_assemble,
        data_contracts=(RELEASED_ASSEMBLY_SEQUENCING_READS_DATA_CONTRACT,),
    )
    return registry


def _agent_request(*, parameters: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "operation_id": "assembly.assemble",
        "input": _hifi_data().model_dump(mode="json"),
        "parameters": parameters
        if parameters is not None
        else {"organelle": "mitochondrion", "method": "auto"},
    }


def _one_long_read_data(
    tmp_path: Path,
    *,
    technology: str,
    quality_state: str,
) -> OrganelleData:
    return _assembly_data(
        long_libraries=[
            {
                "technology": technology,
                "quality_state": quality_state,
                "reads_artifact": "long_reads",
            }
        ]
    )


def _make_himt(**fields: object) -> HimtParameters:
    return HimtParameters(**fields)  # type: ignore[call-arg]


def _string_literals(schema: object) -> set[str]:
    if isinstance(schema, Mapping):
        values: set[str] = set()
        mapping = cast(Mapping[str, object], schema)
        enum = mapping.get("enum")
        if isinstance(enum, list):
            values.update(item for item in cast(list[object], enum) if isinstance(item, str))
        const = mapping.get("const")
        if isinstance(const, str):
            values.add(const)
        for value in mapping.values():
            values.update(_string_literals(value))
        return values
    if isinstance(schema, list):
        values: set[str] = set()
        for item in cast(list[object], schema):
            values.update(_string_literals(item))
        return values
    return set()


def _assert_positive_nullable_integer(schema: dict[str, object]) -> None:
    variants = cast(list[dict[str, object]], schema["anyOf"])
    assert {cast(str, variant["type"]) for variant in variants} == {"integer", "null"}
    integer = next(variant for variant in variants if variant["type"] == "integer")
    assert integer["minimum"] == 1


def _dereference_backend_union(
    root_schema: dict[str, object],
    field_schema: object,
) -> list[dict[str, object]]:
    definitions = cast(dict[str, object], root_schema.get("$defs", {}))
    branches: list[object] = []
    if isinstance(field_schema, dict):
        node = cast(dict[str, object], field_schema)
        union = node.get("anyOf")
        if isinstance(union, list):
            branches.extend(cast(list[object], union))
        else:
            branches.append(node)
    result: list[dict[str, object]] = []
    for branch in branches:
        if isinstance(branch, dict):
            ref = cast(dict[str, object], branch).get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = definitions.get(ref.removeprefix("#/$defs/"))
                if isinstance(target, dict):
                    result.append(cast(dict[str, object], target))
                    continue
            result.append(cast(dict[str, object], branch))
    return result


def _backend_objects(backend_schema: list[dict[str, object]]) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for item in backend_schema:
        if item.get("type") == "object":
            objects.append(item)
    return objects


def _backend_discriminators(backend_schema: list[dict[str, object]]) -> set[str]:
    discriminators: set[str] = set()
    for obj in _backend_objects(backend_schema):
        properties = cast(dict[str, object], obj.get("properties", {}))
        backend = cast(dict[str, object], properties.get("backend", {}))
        const = backend.get("const")
        if isinstance(const, str):
            discriminators.add(const)
    return discriminators


def test_assembly_method_literal_matches_the_backend_registry() -> None:
    assert get_args(AssemblyMethod) == BACKENDS.ids()


def test_himt_parameters_cover_every_public_assemble_option() -> None:
    assert HimtParameters().model_dump(mode="json") == {
        "backend": "himt",
        "species": "plant",
        "kmer_length": 21,
        "head_number": 4,
        "extract_parallel": 2,
        "base_number": 3,
        "filter_depth": 0,
        "filter_percentage": 0.3,
        "proportion": 0.0,
        "accuracy": None,
        "no_flye_meta": False,
        "normalize_depth": 0,
    }


def test_himt_effective_accuracy_uses_declared_quality_state(tmp_path: Path) -> None:
    raw_ont = _one_long_read_data(tmp_path, technology="ont", quality_state="raw")
    corrected = _one_long_read_data(tmp_path, technology="ont", quality_state="corrected")
    assert (
        effective_backend_parameters("himt", None, payload=validate_assembly_data(raw_ont))[
            "accuracy"
        ]
        == 0.3
    )
    assert (
        effective_backend_parameters("himt", None, payload=validate_assembly_data(corrected))[
            "accuracy"
        ]
        == 0.8
    )
    assert (
        effective_backend_parameters(
            "himt", HimtParameters(accuracy=0.61), payload=validate_assembly_data(raw_ont)
        )["accuracy"]
        == 0.61
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kmer_length", 1),
        ("head_number", 1),
        ("extract_parallel", 1),
        ("base_number", 3),
        ("base_number", 4),
        ("filter_depth", 0),
        ("filter_percentage", 0.0),
        ("filter_percentage", 1.0),
        ("proportion", 0.0),
        ("proportion", 1.0),
        ("accuracy", 0.0),
        ("accuracy", 1.0),
        ("normalize_depth", 0),
        ("normalize_depth", -5),
    ],
)
def test_himt_parameters_accept_boundary_values(field: str, value: object) -> None:
    _make_himt(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kmer_length", 0),
        ("head_number", 0),
        ("extract_parallel", 0),
        ("base_number", 2),
        ("base_number", 5),
        ("filter_depth", -1),
        ("filter_percentage", -0.1),
        ("filter_percentage", 1.1),
        ("proportion", -0.1),
        ("proportion", 1.1),
        ("accuracy", -0.1),
        ("accuracy", 1.1),
    ],
)
def test_himt_parameters_reject_out_of_bounds_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _make_himt(**{field: value})


def test_himt_parameters_reject_wrong_types_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _make_himt(kmer_length="21")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _make_himt(unknown_flag=True)


def test_himt_parameters_reject_discriminator_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="backend"):
        AssemblyRequest(
            data=_hifi_data(),
            organelle="mitochondrion",
            method="himt",
            backend_parameters=OatkParameters(),
        )


def test_assembly_request_is_exact_frozen_and_resource_bounded(tmp_path: Path) -> None:
    request = AssemblyRequest(
        data=_hifi_data(),
        organelle="mitochondrion",
    )

    assert request.method == "auto"
    assert request.threads == 4
    assert request.memory_gb is None
    assert request.timeout_seconds is None
    assert request.environment_source == "auto"

    with pytest.raises(ValidationError) as extra:
        AssemblyRequest.model_validate(
            {"data": _hifi_data(), "organelle": "mitochondrion", "unexpected": True}
        )
    assert extra.value.errors()[0]["type"] == "extra_forbidden"

    with pytest.raises(ValidationError) as frozen:
        request.threads = 8
    assert frozen.value.errors()[0]["type"] == "frozen_instance"


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "unknown"},
        {"threads": 0},
        {"threads": 257},
        {"memory_gb": 0},
        {"timeout_seconds": 0},
        {"environment_source": "create"},
    ],
)
def test_assembly_request_rejects_invalid_parameters(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {"data": _hifi_data(), "organelle": "mitochondrion"}
    values.update(changes)

    with pytest.raises(ValidationError):
        AssemblyRequest.model_validate(values)


def test_assembly_request_requires_valid_data() -> None:
    # data + organelle suffice; the output destination is no longer a request field.
    request = AssemblyRequest.model_validate({"data": _hifi_data(), "organelle": "mitochondrion"})
    assert request.method == "auto"

    with pytest.raises(OrganelleInputError) as raised:
        AssemblyRequest.model_validate(
            {"data": OrganelleData(modality="alignment"), "organelle": "mitochondrion"}
        )
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_internal_anchor_has_the_exact_final_signature() -> None:
    from organelleverse.assembly.api import AssemblyEnvironmentHint
    from organelleverse.assembly.api import assemble as canonical_assemble

    signature = inspect.signature(canonical_assemble)
    assert tuple(signature.parameters) == (
        "data",
        "organelle",
        "method",
        "backend_parameters",
        "threads",
        "memory_gb",
        "timeout_seconds",
        "taxon_group",
        "environment_source",
        "backend_version",
        "environment_hint",
    )
    assert signature.parameters["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in tuple(signature.parameters)[1:]
    )
    assert signature.parameters["organelle"].default is inspect.Parameter.empty
    assert signature.parameters["method"].default == "auto"
    assert signature.parameters["backend_parameters"].default is None
    assert signature.parameters["threads"].default == 4
    assert signature.parameters["memory_gb"].default is None
    assert signature.parameters["timeout_seconds"].default is None
    assert signature.parameters["taxon_group"].default == "plant"
    assert signature.parameters["environment_source"].default == "auto"
    assert signature.parameters["backend_version"].default == "tested"
    assert signature.parameters["environment_hint"].default is None

    hints = get_type_hints(canonical_assemble, include_extras=True)
    assert {
        name: hint
        for name, hint in hints.items()
        if name not in {"memory_gb", "timeout_seconds", "environment_hint"}
    } == {
        "data": OrganelleData,
        "organelle": Literal["mitochondrion", "plastid"],
        "method": AssemblyMethod | Literal["auto"],
        "backend_parameters": (
            OatkParameters
            | HimtParameters
            | PmatParameters
            | GetOrganelleParameters
            | TippoParameters
            | PtgaulParameters
            | NovoplastyParameters
            | OvasmParameters
            | None
        ),
        "threads": Annotated[int, Ge(1), Le(256)],
        "taxon_group": Literal["plant", "animal", "fungi"],
        "environment_source": Literal["auto", "existing", "managed"],
        "backend_version": str,
        "return": OrganelleResult,
    }
    for name, expected in (
        ("memory_gb", Annotated[int, Ge(1)]),
        ("timeout_seconds", Annotated[int, Ge(1)]),
        ("environment_hint", AssemblyEnvironmentHint),
    ):
        nullable = get_args(hints[name])
        assert type(None) in nullable
        assert next(item for item in nullable if item is not type(None)) == expected


def test_assembly_contract_is_exact_and_released() -> None:
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    assert ASSEMBLE_SPEC.operation_id == "assembly.assemble"
    assert ASSEMBLE_SPEC.contract_version == "1.0"
    assert ASSEMBLE_SPEC.stage is OperationStage.ANALYZE
    assert ASSEMBLE_SPEC.input_kind is CoreKind.DATA
    assert ASSEMBLE_SPEC.output_kind is CoreKind.RESULT
    assert ASSEMBLE_SPEC.organelle_types == ("mitochondrion", "plastid")
    assert ASSEMBLE_SPEC.input_modalities == ("sequencing_reads",)
    assert ASSEMBLE_SPEC.callable_locator == "organelleverse.assembly.api:assemble"
    assert ASSEMBLE_SPEC.deterministic is False
    assert ASSEMBLE_SPEC.idempotent is True
    assert ASSEMBLE_SPEC.cacheable is False
    assert ASSEMBLE_SPEC.fallback.allowed is False
    assert ASSEMBLE_SPEC.side_effects == (
        SideEffect.READ_FILES,
        SideEffect.WRITE_FILES,
        SideEffect.SUBPROCESS,
        SideEffect.NETWORK,
    )
    assert "assembly.assemble" in {spec.operation_id for spec in list_operations()}


def test_pmat_runtime_and_graph_build_are_released_together() -> None:
    from organelleverse.assembly.backends.runtime import RUNTIMES

    released_operations = {spec.operation_id for spec in list_operations()}
    assert tuple(RUNTIMES) == (
        "oatk",
        "himt",
        "getorganelle",
        "pmat",
        "tippo",
        "ptgaul",
        "novoplasty",
        "ovasm",
    )
    assert "assembly.pmat_graph_build" in released_operations


def test_candidate_invocation_schema_freezes_every_parameter() -> None:
    schema = _candidate_registry().invocation_schema("assembly.assemble")
    root_properties = cast(dict[str, object], schema["properties"])
    parameters = cast(dict[str, object], root_properties["parameters"])
    properties = cast(dict[str, dict[str, object]], parameters["properties"])

    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["organelle"]
    assert _string_literals(properties["organelle"]) == {"mitochondrion", "plastid"}
    assert _string_literals(properties["method"]) == {
        "auto",
        "getorganelle",
        "himt",
        "oatk",
        "pmat",
        "tippo",
        "ptgaul",
        "novoplasty",
        "ovasm",
    }
    assert "output_dir" not in properties
    assert properties["threads"]["minimum"] == 1
    assert properties["threads"]["maximum"] == 256
    _assert_positive_nullable_integer(properties["memory_gb"])
    _assert_positive_nullable_integer(properties["timeout_seconds"])
    assert _string_literals(properties["taxon_group"]) == {"animal", "fungi", "plant"}
    assert _string_literals(properties["environment_source"]) == {"auto", "existing", "managed"}
    assert properties["backend_version"]["type"] == "string"
    assert "environment_hint" in properties
    backend_schema = _dereference_backend_union(schema, properties["backend_parameters"])
    assert _backend_discriminators(backend_schema) == {
        "getorganelle",
        "himt",
        "oatk",
        "pmat",
        "tippo",
        "ptgaul",
        "novoplasty",
        "ovasm",
    }
    assert "additionalProperties" not in json.dumps(backend_schema) or all(
        item.get("additionalProperties") is False for item in _backend_objects(backend_schema)
    )
    himt_schema = next(
        obj
        for obj in _backend_objects(backend_schema)
        if cast(
            dict[str, object], cast(dict[str, object], obj["properties"]).get("backend", {})
        ).get("const")
        == "himt"
    )
    himt_properties = cast(dict[str, dict[str, object]], himt_schema["properties"])
    assert himt_properties["backend"]["const"] == "himt"
    assert himt_properties["species"]["enum"] == ["plant", "animal"]
    assert himt_properties["kmer_length"]["minimum"] == 1
    assert himt_properties["kmer_length"]["default"] == 21
    assert himt_properties["head_number"]["minimum"] == 1
    assert himt_properties["head_number"]["default"] == 4
    assert himt_properties["extract_parallel"]["minimum"] == 1
    assert himt_properties["extract_parallel"]["default"] == 2
    assert himt_properties["base_number"]["minimum"] == 3
    assert himt_properties["base_number"]["maximum"] == 4
    assert himt_properties["base_number"]["default"] == 3
    assert himt_properties["filter_depth"]["minimum"] == 0
    assert himt_properties["filter_depth"]["default"] == 0
    assert himt_properties["filter_percentage"]["minimum"] == 0
    assert himt_properties["filter_percentage"]["maximum"] == 1
    assert himt_properties["filter_percentage"]["default"] == 0.3
    assert himt_properties["proportion"]["minimum"] == 0
    assert himt_properties["proportion"]["maximum"] == 1
    assert himt_properties["proportion"]["default"] == 0.0
    accuracy_variants = cast(
        list[dict[str, object]],
        himt_properties["accuracy"].get("anyOf", [himt_properties["accuracy"]]),
    )
    accuracy_number = next(
        variant for variant in accuracy_variants if variant.get("type") == "number"
    )
    assert accuracy_number["minimum"] == 0
    assert accuracy_number["maximum"] == 1
    assert himt_properties["no_flye_meta"]["default"] is False
    assert himt_properties["normalize_depth"]["default"] == 0
    oatk_schema = next(
        obj
        for obj in _backend_objects(backend_schema)
        if cast(
            dict[str, object], cast(dict[str, object], obj["properties"]).get("backend", {})
        ).get("const")
        == "oatk"
    )
    oatk_properties = cast(dict[str, dict[str, object]], oatk_schema["properties"])
    assert oatk_properties["backend"]["const"] == "oatk"
    assert oatk_properties["minimum_kmer_coverage"]["minimum"] == 1
    assert oatk_properties["minimum_kmer_coverage"]["default"] == 30
    assert "kmer_size" not in oatk_properties


def test_himt_parameters_reject_unknown_and_coerced_values_on_all_paths(
    tmp_path: Path,
) -> None:
    from organelleverse.assembly.api import assemble as canonical_assemble
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    data = _hifi_data()
    base = {"organelle": "mitochondrion", "method": "himt"}
    cases = [
        {**base, "backend_parameters": {"backend": "himt", "unknown": True}},
        {**base, "backend_parameters": {"backend": "himt", "kmer_length": "21"}},
    ]
    registry = _candidate_registry()
    side_effects = set(ASSEMBLE_SPEC.side_effects)

    for parameters in cases:
        with pytest.raises(ValidationError):
            canonical_assemble(data, **parameters)  # type: ignore[arg-type]
        with pytest.raises(OrganelleParameterError):
            registry.invoke("assembly.assemble", input=data, parameters=parameters)
        response = invoke_json(
            {
                "operation_id": "assembly.assemble",
                "input": data.model_dump(mode="json"),
                "parameters": {**parameters},
            },
            registry=registry,
            granted_side_effects=side_effects,
        )
        assert response["ok"] is False
        error = cast(dict[str, object], response["error"])
        assert error["error_code"] == "parameter.invalid_operation_parameters"


def test_oatk_release_contract_schema_exposes_only_the_supported_profile() -> None:
    from organelleverse.assembly.data_contract import oatk_assembly_data_json_schema

    schema = oatk_assembly_data_json_schema()
    definitions = cast(dict[str, object], schema["$defs"])
    payload = cast(dict[str, object], cast(dict[str, object], schema["properties"])["payload"])
    properties = cast(dict[str, dict[str, object]], payload["properties"])

    assert properties["short_libraries"]["maxItems"] == 0
    assert properties["long_libraries"]["minItems"] == 1
    assert properties["contig_inputs"]["maxItems"] == 0

    long_items = cast(dict[str, str], properties["long_libraries"]["items"])
    long_model = cast(dict[str, object], definitions[long_items["$ref"].removeprefix("#/$defs/")])
    long_properties = cast(dict[str, dict[str, object]], long_model["properties"])
    assert long_properties["technology"]["const"] == "pacbio_hifi"
    assert long_properties["quality_state"]["const"] == "ccs"

    auxiliary_ref = cast(str, properties["auxiliary"]["$ref"])
    auxiliary = cast(dict[str, object], definitions[auxiliary_ref.removeprefix("#/$defs/")])
    auxiliary_properties = cast(dict[str, dict[str, object]], auxiliary["properties"])
    assert auxiliary_properties["seed_fasta_artifact"]["type"] == "null"
    hmm_variants = cast(
        list[dict[str, object]], auxiliary_properties["hmm_profiles_artifact"]["anyOf"]
    )
    assert {variant["type"] for variant in hmm_variants} == {"string", "null"}


@pytest.mark.parametrize(
    "data",
    [
        _assembly_data(
            short_libraries=[
                {
                    "technology": "illumina",
                    "layout": "paired_end",
                    "read1_artifact": "r1",
                    "read2_artifact": "r2",
                    "read_length": 150,
                    "insert_size": 350,
                }
            ]
        ),
        _assembly_data(
            long_libraries=[
                {"technology": "pacbio_clr", "quality_state": "raw", "reads_artifact": "long_reads"}
            ]
        ),
        _assembly_data(
            short_libraries=[
                {
                    "technology": "illumina",
                    "layout": "single_end",
                    "read1_artifact": "r1",
                    "read_length": 150,
                }
            ],
            long_libraries=[
                {
                    "technology": "pacbio_hifi",
                    "quality_state": "ccs",
                    "reads_artifact": "long_reads",
                }
            ],
        ),
        _assembly_data(contig_inputs=[{"fasta_artifact": "contigs"}]),
        _assembly_data(
            long_libraries=[
                {
                    "technology": "pacbio_hifi",
                    "quality_state": "ccs",
                    "reads_artifact": "long_reads",
                }
            ],
            auxiliary={"reference_fasta_artifact": "reference"},
        ),
    ],
)
def test_oatk_release_contract_rejects_inputs_the_backend_would_ignore(
    data: OrganelleData,
) -> None:
    from organelleverse.assembly.data_contract import validate_oatk_assembly_data

    with pytest.raises(OrganelleInputError) as raised:
        validate_oatk_assembly_data(data)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_oatk_release_contract_accepts_hifi_and_optional_hmm_profile() -> None:
    from organelleverse.assembly.data_contract import validate_oatk_assembly_data

    data = _assembly_data(
        long_libraries=[
            {"technology": "pacbio_hifi", "quality_state": "ccs", "reads_artifact": "long_reads"}
        ],
        auxiliary={"hmm_profiles_artifact": "hmm_profiles"},
    )

    payload = validate_oatk_assembly_data(data)
    assert payload.long_libraries[0].technology == "pacbio_hifi"
    assert payload.auxiliary.hmm_profiles_artifact == "hmm_profiles"


def test_oatk_release_contract_preserves_common_genome_size_evidence() -> None:
    from organelleverse.assembly.data_contract import validate_oatk_assembly_data

    evidence = {"source": "user", "genome_size_bp": 125_000_000}
    data = _assembly_data(
        long_libraries=[
            {"technology": "pacbio_hifi", "quality_state": "ccs", "reads_artifact": "long_reads"}
        ],
        auxiliary={"genome_size": 125_000_000, "genome_size_evidence": evidence},
    )

    payload = validate_oatk_assembly_data(data)
    assert payload.auxiliary.genome_size == 125_000_000
    assert payload.auxiliary.genome_size_evidence is not None
    assert payload.auxiliary.genome_size_evidence.source == "user"


def test_oatk_parameters_are_strict_defaulted_and_semantic(tmp_path: Path) -> None:
    base = AssemblyRequest(
        data=_hifi_data(),
        organelle="mitochondrion",
    )
    explicit = base.model_copy(
        update={"backend_parameters": {"backend": "oatk", "minimum_kmer_coverage": 41}}
    )

    assert OatkParameters().minimum_kmer_coverage == 30
    assert explicit.backend_parameters == OatkParameters(minimum_kmer_coverage=41)
    assert base.resolved_semantic_hash("oatk") != explicit.resolved_semantic_hash("oatk")
    assert base.resolved_semantic_payload("oatk")["effective_backend_parameters"] == {
        "backend": "oatk",
        "kmer_size": 1001,
        "minimum_kmer_coverage": 30,
    }

    with pytest.raises(ValidationError, match="extra_forbidden"):
        OatkParameters.model_validate(
            {"backend": "oatk", "minimum_kmer_coverage": 30, "raw_flags": ["--all"]}
        )


def test_explicit_method_rejects_parameters_for_another_backend(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="backend_parameters must match explicit method"):
        AssemblyRequest(
            data=_hifi_data(),
            organelle="mitochondrion",
            method="pmat",
            backend_parameters=OatkParameters(),
        )


def test_canonical_api_validates_and_delegates_one_exact_request(tmp_path: Path) -> None:
    from organelleverse.assembly.api import assemble as canonical_assemble

    data = _hifi_data()
    expected = AssemblyRequest(
        data=data,
        organelle="mitochondrion",
        method="auto",
    )
    result = OrganelleResult(
        operation_id="assembly.assemble",
        scope="mitochondrion",
        status="ok",
    )
    with patch("organelleverse.assembly.service.execute_assembly", return_value=result) as execute:
        actual = canonical_assemble(
            data,
            organelle="mitochondrion",
            method="auto",
        )

    assert actual is result
    execute.assert_called_once_with(expected)


def test_canonical_api_rejects_unsupported_input_before_delegation(tmp_path: Path) -> None:
    from organelleverse.assembly.api import assemble as canonical_assemble

    # Contig-only input: no released backend accepts this (Oatk needs long HiFi,
    # HiMT needs long reads, GetOrganelle needs short reads).
    contig_data = _assembly_data(contig_inputs=[{"fasta_artifact": "c1"}])
    output_dir = tmp_path / "must-not-exist"
    with (
        patch("organelleverse.assembly.service.execute_assembly") as execute,
        pytest.raises(OrganelleInputError) as raised,
    ):
        canonical_assemble(
            contig_data,
            organelle="mitochondrion",
        )

    assert raised.value.code == "assembly.unsupported_data_profile"
    execute.assert_not_called()
    assert not output_dir.exists()


def test_canonical_api_rejects_python_type_coercion_before_delegation(
    tmp_path: Path,
) -> None:
    from organelleverse.assembly.api import assemble as canonical_assemble

    with (
        patch("organelleverse.assembly.service.execute_assembly") as execute,
        pytest.raises(ValidationError),
    ):
        canonical_assemble(
            _hifi_data(),
            organelle="mitochondrion",
            threads="4",  # type: ignore[arg-type]
        )

    execute.assert_not_called()


def test_direct_registry_and_agent_json_share_the_canonical_callable(tmp_path: Path) -> None:
    from organelleverse.assembly.api import assemble as canonical_assemble
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    registry = _candidate_registry()
    data = _hifi_data()
    parameters = {"organelle": "mitochondrion", "method": "auto"}
    result = OrganelleResult(
        operation_id="assembly.assemble",
        scope="mitochondrion",
        status="ok",
    )
    with patch("organelleverse.assembly.service.execute_assembly", return_value=result) as execute:
        direct = canonical_assemble(data, **parameters)  # type: ignore[arg-type]
        registered = registry.invoke(
            "assembly.assemble",
            input=data,
            parameters=parameters,
        )
        response = invoke_json(
            {
                "operation_id": "assembly.assemble",
                "input": data.model_dump(mode="json"),
                "parameters": {**parameters},
            },
            registry=registry,
            granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
        )

    assert direct is result
    assert registered == result
    assert response == {
        "ok": True,
        "operation_id": "assembly.assemble",
        "result": result.model_dump(mode="json"),
    }
    assert execute.call_count == 3


def test_agent_requires_every_declared_side_effect_before_delegation(tmp_path: Path) -> None:
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    grants = set(ASSEMBLE_SPEC.side_effects) - {SideEffect.NETWORK}
    with patch("organelleverse.assembly.service.execute_assembly") as execute:
        response = invoke_json(
            _agent_request(parameters={"organelle": "mitochondrion", "method": "auto"}),
            registry=_candidate_registry(),
            granted_side_effects=grants,
        )

    assert response["ok"] is False
    error = cast(dict[str, object], response["error"])
    assert error["error_code"] == "permission.denied"
    execute.assert_not_called()


def test_agent_invocation_returns_the_scientific_result() -> None:
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    result = OrganelleResult(
        operation_id="assembly.assemble",
        scope="mitochondrion",
        status="ok",
    )
    with patch("organelleverse.assembly.service.execute_assembly", return_value=result) as execute:
        response = invoke_json(
            _agent_request(),
            registry=_candidate_registry(),
            granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
        )

    assert response == {
        "ok": True,
        "operation_id": "assembly.assemble",
        "result": result.model_dump(mode="json"),
    }
    execute.assert_called_once()


@pytest.mark.parametrize(
    "parameters",
    [
        {"organelle": "mitochondrion", "method": "unknown"},
        {"organelle": "mitochondrion", "method": "auto", "unexpected": True},
    ],
)
def test_agent_rejects_invalid_parameters_before_the_anchor(
    parameters: dict[str, object],
) -> None:
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    response = invoke_json(
        _agent_request(parameters=parameters),
        registry=_candidate_registry(),
        granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
    )

    assert response["ok"] is False
    assert "result" not in response
    error = cast(dict[str, object], response["error"])
    assert error["error_code"] == "parameter.invalid_operation_parameters"


def test_himt_agent_paths_reach_canonical_callable_with_structured_parameters(
    tmp_path: Path,
) -> None:
    from organelleverse.assembly.api import assemble as canonical_assemble
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    data = _assembly_data(
        long_libraries=[
            {"technology": "pacbio_hifi", "quality_state": "ccs", "reads_artifact": "long_reads"}
        ]
    )
    expected = AssemblyRequest(
        data=data,
        organelle="mitochondrion",
        method="himt",
        backend_parameters=HimtParameters(backend="himt", kmer_length=31, accuracy=0.61),
        threads=2,
        environment_source="managed",
    )
    result = OrganelleResult(operation_id="assembly.assemble", scope="mitochondrion", status="ok")
    registry = _candidate_registry()
    with patch("organelleverse.assembly.service.execute_assembly", return_value=result) as execute:
        direct = canonical_assemble(
            data,
            organelle="mitochondrion",
            method="himt",
            backend_parameters=HimtParameters(backend="himt", kmer_length=31, accuracy=0.61),
            threads=2,
            environment_source="managed",
        )
        registered = registry.invoke(
            "assembly.assemble",
            input=data,
            parameters={
                "organelle": "mitochondrion",
                "method": "himt",
                "backend_parameters": HimtParameters(backend="himt", kmer_length=31, accuracy=0.61),
                "threads": 2,
                "environment_source": "managed",
            },
        )
        response = invoke_json(
            {
                "operation_id": "assembly.assemble",
                "input": data.model_dump(mode="json"),
                "parameters": {
                    "organelle": "mitochondrion",
                    "method": "himt",
                    "backend_parameters": {"backend": "himt", "kmer_length": 31, "accuracy": 0.61},
                    "threads": 2,
                    "environment_source": "managed",
                },
            },
            registry=registry,
            granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
        )

    assert direct is result
    assert registered == result
    assert response == {
        "ok": True,
        "operation_id": "assembly.assemble",
        "result": result.model_dump(mode="json"),
    }
    assert execute.call_count == 3
    for call in execute.call_args_list:
        assert call.args[0] == expected


@pytest.mark.parametrize(
    "invalid_parameters",
    [
        {
            "organelle": "mitochondrion",
            "method": "himt",
            "backend_parameters": {"backend": "oatk", "kmer_size": 1001},
        },
        {
            "organelle": "mitochondrion",
            "method": "himt",
            "backend_parameters": {"backend": "himt", "kmer_length": 31},
            "extra_field": True,
        },
    ],
)
def test_himt_agent_rejects_invalid_parameters_before_execution(
    invalid_parameters: dict[str, object],
) -> None:
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    data = _assembly_data(
        long_libraries=[
            {"technology": "pacbio_hifi", "quality_state": "ccs", "reads_artifact": "long_reads"}
        ]
    )
    with patch("organelleverse.assembly.service.execute_assembly") as execute:
        response = invoke_json(
            {
                "operation_id": "assembly.assemble",
                "input": data.model_dump(mode="json"),
                "parameters": invalid_parameters,
            },
            registry=_candidate_registry(),
            granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
        )
    assert response["ok"] is False
    error = cast(dict[str, object], response["error"])
    assert error["error_code"] == "parameter.invalid_operation_parameters"
    execute.assert_not_called()


def test_himt_agent_rejects_ignored_auxiliary_artifacts(tmp_path: Path) -> None:
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    # HiFi reads with a seed are a released combination since ovasm takes a custom seed, so
    # the combined data contract admits them. HiMT's own validator still refuses the seed it
    # would ignore, in the service, before any environment is touched: the files are real
    # because the service reads them on the way there.
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r\nACGT\n+\nIIII\n")
    seed = tmp_path / "seed.fasta"
    seed.write_text(">s\nACGT\n")
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "long_reads": ArtifactRef.from_path(
                    reads, kind="long_read", format="fastq", media_type="application/x-fastq"
                ),
                "seed": ArtifactRef.from_path(
                    seed, kind="assembly_input", format="fasta", media_type="application/x-fasta"
                ),
            },
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_reads",
                    }
                ],
                "auxiliary": {"seed_fasta_artifact": "seed"},
            },
        }
    )
    with patch("organelleverse.assembly.service.EnvironmentManager") as environments:
        response = invoke_json(
            {
                "operation_id": "assembly.assemble",
                "input": data.model_dump(mode="json"),
                "parameters": {"organelle": "mitochondrion", "method": "himt"},
            },
            registry=_candidate_registry(),
            granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
        )
    assert response["ok"] is False
    error = cast(dict[str, object], response["error"])
    assert error["error_code"] == "assembly.unsupported_data_profile"
    environments.assert_not_called()


def test_himt_agent_rejects_second_long_library(tmp_path: Path) -> None:
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    data = _assembly_data(
        long_libraries=[
            {"technology": "pacbio_hifi", "quality_state": "ccs", "reads_artifact": "long_a"},
            {"technology": "ont", "quality_state": "raw", "reads_artifact": "long_b"},
        ]
    )
    with patch("organelleverse.assembly.service.execute_assembly") as execute:
        response = invoke_json(
            {
                "operation_id": "assembly.assemble",
                "input": data.model_dump(mode="json"),
                "parameters": {"organelle": "mitochondrion", "method": "himt"},
            },
            registry=_candidate_registry(),
            granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
        )
    assert response["ok"] is False
    error = cast(dict[str, object], response["error"])
    assert error["error_code"] == "assembly.unsupported_data_profile"
    execute.assert_not_called()
