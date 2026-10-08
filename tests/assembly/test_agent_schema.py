from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from organelleverse.assembly.contracts import HimtParameters
from organelleverse.assembly.data_contract import (
    RELEASED_ASSEMBLY_SEQUENCING_READS_DATA_CONTRACT,
    SEQUENCING_READS_DATA_CONTRACT,
    getorganelle_assembly_data_json_schema,
    released_assembly_data_json_schema,
    validate_released_assembly_data,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleInputError, OrganelleParameterError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
    operation,
)
from organelleverse.operations.adapters import invoke_json


def _assembly_probe_spec() -> OperationSpec:
    return OperationSpec(
        operation_id="assembly.inspect_reads",
        contract_version="1.0",
        title="Inspect assembly sequencing reads",
        description="Test fixture spec used only to probe assembly agent-schema behavior.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.ANALYZE,
        input_kind=CoreKind.DATA,
        output_kind=CoreKind.RESULT,
        organelle_types=("mitochondrion", "plastid"),
        input_modalities=("sequencing_reads",),
        callable_locator="organelleverse.assembly.api:inspect_reads",
    )


def _dangling_reads_data() -> OrganelleData:
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [],
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "missing_reads",
                    }
                ],
                "contig_inputs": [],
                "auxiliary": {},
            },
        }
    )


def _valid_reads_data() -> OrganelleData:
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "long_reads": ArtifactRef(
                    kind="long_read",
                    uri="reads.fastq",
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


def _one_long_library_data(technology: str, quality_state: str) -> OrganelleData:
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "long_reads": ArtifactRef(
                    kind="long_read",
                    uri="reads.fastq",
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
                        "technology": technology,
                        "quality_state": quality_state,
                        "reads_artifact": "long_reads",
                    }
                ],
            },
        }
    )


def _released_registry() -> OperationRegistry:
    from organelleverse.assembly.api import assemble as canonical_assemble
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    registry = OperationRegistry()
    registry.register(
        ASSEMBLE_SPEC,
        canonical_assemble,
        data_contracts=(RELEASED_ASSEMBLY_SEQUENCING_READS_DATA_CONTRACT,),
    )
    return registry


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
    return [obj for obj in backend_schema if obj.get("type") == "object"]


def _backend_discriminators(backend_schema: list[dict[str, object]]) -> set[str]:
    discriminators: set[str] = set()
    for obj in _backend_objects(backend_schema):
        properties = cast(dict[str, object], obj.get("properties", {}))
        backend = cast(dict[str, object], properties.get("backend", {}))
        const = backend.get("const")
        if isinstance(const, str):
            discriminators.add(const)
    return discriminators


def test_released_assembly_data_schema_is_anyof_and_covers_profiles() -> None:
    schema = released_assembly_data_json_schema()
    encoded = json.dumps(schema, sort_keys=True)
    assert '"maxItems": 1' in encoded
    assert '"minItems": 1' in encoded
    for technology in ("pacbio_hifi", "pacbio_clr", "ont"):
        assert technology in encoded


def _getorganelle_reads_data(
    *, layout: Literal["paired_end", "single_end"], include_read2: bool
) -> OrganelleData:
    artifacts = {
        "read1": ArtifactRef(
            kind="short_read",
            uri="reads_1.fastq",
            format="fastq",
            media_type="application/x-fastq",
            sha256="1" * 64,
            size_bytes=1,
        )
    }
    library: dict[str, object] = {
        "technology": "illumina",
        "layout": layout,
        "read1_artifact": "read1",
        "read_length": 150,
    }
    if include_read2:
        artifacts["read2"] = ArtifactRef(
            kind="short_read",
            uri="reads_2.fastq",
            format="fastq",
            media_type="application/x-fastq",
            sha256="2" * 64,
            size_bytes=1,
        )
        library["read2_artifact"] = "read2"
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": artifacts,
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [library],
            },
        }
    )


def test_getorganelle_agent_schema_closes_each_short_read_layout() -> None:
    schema = getorganelle_assembly_data_json_schema()
    root_properties = cast(dict[str, object], schema["properties"])
    payload = cast(dict[str, object], root_properties["payload"])
    payload_properties = cast(dict[str, object], payload["properties"])
    short_libraries = cast(dict[str, object], payload_properties["short_libraries"])
    items = cast(dict[str, object], short_libraries["items"])

    assert items.get("anyOf") == [
        {"$ref": "#/$defs/payload___GetOrganellePairedEndLibrary"},
        {"$ref": "#/$defs/payload___GetOrganelleSingleEndLibrary"},
    ]
    definitions = cast(dict[str, object], schema["$defs"])
    branches = {
        name: cast(dict[str, object], definitions[f"payload___GetOrganelle{name}Library"])
        for name in ("PairedEnd", "SingleEnd")
    }
    paired_required = cast(list[str], branches["PairedEnd"]["required"])
    single_properties = cast(dict[str, object], branches["SingleEnd"]["properties"])
    assert "read2_artifact" in paired_required
    assert branches["SingleEnd"]["additionalProperties"] is False
    assert "read2_artifact" not in single_properties


@pytest.mark.parametrize(
    ("layout", "include_read2"),
    [("paired_end", False), ("single_end", True)],
)
def test_getorganelle_runtime_rejects_invalid_short_read_layout_pairs(
    layout: Literal["paired_end", "single_end"], include_read2: bool
) -> None:
    with pytest.raises(OrganelleInputError):
        validate_released_assembly_data(
            _getorganelle_reads_data(layout=layout, include_read2=include_read2)
        )


@pytest.mark.parametrize(
    ("layout", "include_read2"),
    [("paired_end", True), ("single_end", False)],
)
def test_getorganelle_runtime_decodes_valid_short_read_layouts(
    layout: Literal["paired_end", "single_end"], include_read2: bool
) -> None:
    payload = validate_released_assembly_data(
        _getorganelle_reads_data(layout=layout, include_read2=include_read2)
    )
    library = payload.short_libraries[0]
    assert library.layout == layout
    assert (library.read2_artifact is not None) is include_read2


@pytest.mark.parametrize(
    ("technology", "quality_state"),
    [
        ("pacbio_hifi", "ccs"),
        ("pacbio_clr", "raw"),
        ("pacbio_clr", "corrected"),
        ("ont", "raw"),
        ("ont", "corrected"),
        ("ont", "hq"),
        ("ont", "duplex"),
    ],
)
def test_released_contract_accepts_one_valid_long_library(
    technology: str, quality_state: str
) -> None:
    data = _one_long_library_data(technology, quality_state)
    validate_released_assembly_data(data)


def test_released_contract_accepts_same_profile_pmat_libraries() -> None:
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "a": ArtifactRef(
                    kind="long_read",
                    uri="a.fastq",
                    format="fastq",
                    media_type="application/x-fastq",
                    sha256="a" * 64,
                    size_bytes=1,
                ),
                "b": ArtifactRef(
                    kind="long_read",
                    uri="b.fastq",
                    format="fastq",
                    media_type="application/x-fastq",
                    sha256="b" * 64,
                    size_bytes=1,
                ),
            },
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "ont",
                        "quality_state": "raw",
                        "reads_artifact": "a",
                    },
                    {
                        "technology": "ont",
                        "quality_state": "raw",
                        "reads_artifact": "b",
                    },
                ],
            },
        }
    )
    payload = validate_released_assembly_data(data)
    assert len(payload.long_libraries) == 2


def test_released_contract_rejects_unsupported_combinations() -> None:
    # Short reads with an unsupported auxiliary role that GetOrganelle also rejects
    # (reference_fasta is not in GetOrganelle's allowed auxiliary set).
    short_unsupported_aux = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "r1": ArtifactRef(
                    kind="short_read",
                    uri="r1.fastq",
                    format="fastq",
                    media_type="application/x-fastq",
                    sha256="a" * 64,
                    size_bytes=1,
                ),
                "ref": ArtifactRef(
                    kind="assembly_input",
                    uri="ref.fasta",
                    format="fasta",
                    media_type="application/x-fasta",
                    sha256="b" * 64,
                    size_bytes=1,
                ),
            },
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [
                    {
                        "technology": "illumina",
                        "layout": "single_end",
                        "read1_artifact": "r1",
                        "read_length": 150,
                    }
                ],
                "auxiliary": {"reference_fasta_artifact": "ref"},
            },
        }
    )
    with pytest.raises(OrganelleInputError) as raised:
        validate_released_assembly_data(short_unsupported_aux)
    assert raised.value.code == "assembly.unsupported_data_profile"

    contig = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "c": ArtifactRef(
                    kind="assembly_input",
                    uri="c.fasta",
                    format="fasta",
                    media_type="application/x-fasta",
                    sha256="a" * 64,
                    size_bytes=1,
                )
            },
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "contig_inputs": [{"fasta_artifact": "c"}],
            },
        }
    )
    with pytest.raises(OrganelleInputError) as raised:
        validate_released_assembly_data(contig)
    assert raised.value.code == "assembly.unsupported_data_profile"

    raw_ont = _one_long_library_data("ont", "raw")
    with_aux = OrganelleData.model_validate(
        {
            **raw_ont.model_dump(mode="python", exclude={"object_id"}),
            "artifacts": {
                **raw_ont.model_dump(mode="python", exclude={"object_id"})["artifacts"],
                "seed_fasta": ArtifactRef(
                    kind="assembly_input",
                    uri="seed.fasta",
                    format="fasta",
                    media_type="application/x-fasta",
                    sha256="b" * 64,
                    size_bytes=1,
                ),
            },
            "payload": {
                **dict(raw_ont.payload),
                "auxiliary": {"seed_fasta_artifact": "seed_fasta"},
            },
        }
    )
    # raw long reads with a seed are a released combination: ovasm recruits with a custom
    # seed (the other backends still refuse a seed they would ignore)
    assert validate_released_assembly_data(with_aux).auxiliary.seed_fasta_artifact == "seed_fasta"

    # an auxiliary role no backend takes with raw long reads is still refused
    unused_aux = OrganelleData.model_validate(
        {
            **with_aux.model_dump(mode="python", exclude={"object_id"}),
            "payload": {
                **dict(raw_ont.payload),
                "auxiliary": {"chloroplast_fasta_artifact": "seed_fasta"},
            },
        }
    )
    with pytest.raises(OrganelleInputError) as raised:
        validate_released_assembly_data(unused_aux)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_released_contract_rejects_invalid_quality_pair_and_missing_artifact() -> None:
    invalid_quality = _one_long_library_data("pacbio_hifi", "raw")
    with pytest.raises(OrganelleInputError) as raised:
        validate_released_assembly_data(invalid_quality)
    assert raised.value.code == "assembly.unsupported_data_profile"

    missing = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "ont",
                        "quality_state": "raw",
                        "reads_artifact": "long_reads",
                    }
                ],
            },
        }
    )
    with pytest.raises(OrganelleInputError) as raised:
        validate_released_assembly_data(missing)
    assert raised.value.code == "assembly.missing_artifact_role"


def test_released_invocation_schema_exposes_combined_backend_union() -> None:
    schema = _released_registry().invocation_schema("assembly.assemble")
    root_properties = cast(dict[str, object], schema["properties"])
    parameters = cast(dict[str, object], root_properties["parameters"])
    properties = cast(dict[str, dict[str, object]], parameters["properties"])

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


def _registered_probe() -> tuple[OperationRegistry, list[str], Callable[..., OrganelleResult]]:
    calls: list[str] = []
    registry = OperationRegistry(data_contracts=(SEQUENCING_READS_DATA_CONTRACT,))
    spec = _assembly_probe_spec()

    @operation(spec=spec, registry=registry)
    def inspect(
        data: OrganelleData,
        *,
        output_dir: Path,
        method: Literal["auto", "oatk"] = "auto",
    ) -> OrganelleResult:
        calls.append(data.object_id)
        return OrganelleResult(
            operation_id="assembly.inspect_reads",
            scope="mitochondrion",
            status="ok",
        )

    return registry, calls, inspect


@pytest.mark.parametrize("path", ["direct", "registry", "agent"])
def test_every_invocation_path_uses_the_sequencing_validator(path: str) -> None:
    registry, calls, inspect = _registered_probe()
    invalid = _dangling_reads_data()
    if path == "direct":
        with pytest.raises(OrganelleInputError) as raised:
            inspect(invalid, output_dir=Path("out"))
        code = raised.value.code
    elif path == "registry":
        with pytest.raises(OrganelleInputError) as raised:
            registry.invoke(
                "assembly.inspect_reads",
                input=invalid,
                parameters={"output_dir": Path("out")},
            )
        code = raised.value.code
    else:
        response = invoke_json(
            {
                "operation_id": "assembly.inspect_reads",
                "input": invalid.model_dump(mode="json"),
                "parameters": {"output_dir": "out"},
            },
            registry=registry,
            granted_side_effects=set(),
        )
        assert response["ok"] is False
        code = cast(dict[str, object], response["error"])["error_code"]

    assert code == "assembly.missing_artifact_role"
    assert calls == []


@pytest.mark.parametrize("path", ["direct", "registry", "binding", "agent"])
def test_modality_preflight_precedes_parameter_validation_on_every_path(path: str) -> None:
    registry, calls, inspect = _registered_probe()
    binding = registry.require("assembly.inspect_reads")
    invalid = _dangling_reads_data()
    parameters: dict[str, object] = {"output_dir": Path("out"), "method": "unknown"}

    if path == "direct":
        with pytest.raises(OrganelleInputError) as raised:
            inspect(invalid, **parameters)
        code = raised.value.code
    elif path == "registry":
        with pytest.raises(OrganelleInputError) as raised:
            registry.invoke(
                "assembly.inspect_reads",
                input=invalid,
                parameters=parameters,
            )
        code = raised.value.code
    elif path == "binding":
        constructed = binding.signature.parameter_model.model_construct(
            _fields_set={"output_dir", "method"},
            output_dir=Path("out"),
            method="unknown",
        )
        with pytest.raises(OrganelleInputError) as raised:
            binding.invoke_validated(invalid, constructed)
        code = raised.value.code
    else:
        response = invoke_json(
            {
                "operation_id": "assembly.inspect_reads",
                "input": invalid.model_dump(mode="json"),
                "parameters": {"output_dir": "out", "method": "unknown"},
            },
            registry=registry,
            granted_side_effects=set(),
        )
        assert response["ok"] is False
        code = cast(dict[str, object], response["error"])["error_code"]

    assert code == "assembly.missing_artifact_role"
    assert calls == []


def test_invoke_validated_revalidates_forged_exact_parameter_model() -> None:
    registry, calls, _ = _registered_probe()
    binding = registry.require("assembly.inspect_reads")
    valid = binding.signature.parameter_model.model_validate(
        {"output_dir": Path("out"), "method": "auto"}
    )
    forged = valid.model_copy(update={"method": "unknown"})

    with pytest.raises(OrganelleParameterError) as raised:
        binding.invoke_validated(_valid_reads_data(), forged)

    assert raised.value.code == "parameter.invalid_operation_parameters"
    assert calls == []


def test_custom_decoder_revalidates_constructed_exact_parameter_model() -> None:
    registry, calls, _ = _registered_probe()
    binding = registry.require("assembly.inspect_reads")

    def forged_decoder(
        parameter_model: type[BaseModel],
        raw_parameters: Mapping[str, object],
    ) -> BaseModel:
        assert parameter_model is binding.signature.parameter_model
        assert raw_parameters == {}
        return parameter_model.model_construct(
            _fields_set={"output_dir", "method"},
            output_dir=Path("out"),
            method="unknown",
        )

    with pytest.raises(OrganelleParameterError) as raised:
        binding.invoke_with_parameter_decoder(
            _valid_reads_data(),
            {},
            parameter_decoder=forged_decoder,
        )

    assert raised.value.code == "parameter.invalid_operation_parameters"
    assert calls == []


def test_invoke_validated_rejects_forged_extra_parameter_field() -> None:
    registry, calls, _ = _registered_probe()
    binding = registry.require("assembly.inspect_reads")
    valid = binding.signature.parameter_model.model_validate(
        {"output_dir": Path("out"), "method": "auto"}
    )
    forged = valid.model_copy(update={"unexpected": True})
    assert forged.__dict__["unexpected"] is True
    assert "unexpected" in forged.__pydantic_fields_set__

    with pytest.raises(OrganelleParameterError) as raised:
        binding.invoke_validated(_valid_reads_data(), forged)

    assert raised.value.code == "parameter.invalid_operation_parameters"
    assert calls == []


def test_invoke_validated_rejects_forged_parameter_fields_set_entry() -> None:
    registry, calls, _ = _registered_probe()
    binding = registry.require("assembly.inspect_reads")
    forged = binding.signature.parameter_model.model_validate(
        {"output_dir": Path("out"), "method": "auto"}
    ).model_copy()
    forged.__pydantic_fields_set__.add("unexpected")
    assert "unexpected" not in forged.__dict__

    with pytest.raises(OrganelleParameterError) as raised:
        binding.invoke_validated(_valid_reads_data(), forged)

    assert raised.value.code == "parameter.invalid_operation_parameters"
    assert calls == []


def test_custom_decoder_rejects_forged_pydantic_extra_parameter_field() -> None:
    registry, calls, _ = _registered_probe()
    binding = registry.require("assembly.inspect_reads")

    def forged_decoder(
        parameter_model: type[BaseModel],
        raw_parameters: Mapping[str, object],
    ) -> BaseModel:
        assert parameter_model is binding.signature.parameter_model
        assert raw_parameters == {}
        forged = parameter_model.model_validate({"output_dir": Path("out"), "method": "auto"})
        object.__setattr__(forged, "__pydantic_extra__", {"unexpected": True})
        return forged

    with pytest.raises(OrganelleParameterError) as raised:
        binding.invoke_with_parameter_decoder(
            _valid_reads_data(),
            {},
            parameter_decoder=forged_decoder,
        )

    assert raised.value.code == "parameter.invalid_operation_parameters"
    assert calls == []


def test_complete_agent_invocation_schema_is_strict_and_chainable() -> None:
    registry, _, _ = _registered_probe()

    schema = registry.invocation_schema("assembly.inspect_reads")

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "assembly.inspect_reads invocation"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["operation_id", "input", "parameters"]
    properties = cast(dict[str, object], schema["properties"])
    operation_id = cast(dict[str, object], properties["operation_id"])
    assert operation_id == {"type": "string", "const": "assembly.inspect_reads"}

    input_schema = cast(dict[str, object], properties["input"])
    input_properties = cast(dict[str, object], input_schema["properties"])
    assert cast(dict[str, object], input_properties["modality"])["const"] == "sequencing_reads"
    assert "object_id" in input_properties
    assert "object_id" not in cast(list[str], input_schema["required"])

    parameters = cast(dict[str, object], properties["parameters"])
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["output_dir"]

    encoded = json.dumps(schema, sort_keys=True)
    for definition in (
        "short_libraries",
        "long_libraries",
        "contig_inputs",
        "auxiliary",
        "ShortLibraryContract",
        "LongLibraryContract",
        "ContigContract",
        "AuxiliaryContract",
    ):
        assert definition in encoded

    _assert_object_ids_optional(schema)
    _assert_internal_refs_resolve(schema)


def _assert_object_ids_optional(value: object) -> None:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        properties = mapping.get("properties")
        required = mapping.get("required", [])
        if isinstance(properties, Mapping) and "object_id" in properties:
            assert isinstance(required, list)
            assert "object_id" not in required
        for nested in mapping.values():
            _assert_object_ids_optional(nested)
    elif isinstance(value, list):
        for nested in cast(list[object], value):
            _assert_object_ids_optional(nested)


def _assert_internal_refs_resolve(schema: dict[str, object]) -> None:
    definitions = cast(dict[str, object], schema.get("$defs", {}))

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            mapping = cast(Mapping[str, object], value)
            reference = mapping.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                assert reference.removeprefix("#/$defs/") in definitions
            for nested in mapping.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in cast(list[object], value):
                walk(nested)

    walk(schema)


def test_himt_agent_json_preserves_structured_backend_parameters(
    tmp_path: Path,
) -> None:
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    registry = _released_registry()
    data = _valid_reads_data()
    result = OrganelleResult(operation_id="assembly.assemble", scope="mitochondrion", status="ok")
    with patch("organelleverse.assembly.service.execute_assembly", return_value=result) as execute:
        response = invoke_json(
            {
                "operation_id": "assembly.assemble",
                "input": data.model_dump(mode="json"),
                "parameters": {
                    "organelle": "mitochondrion",
                    "method": "himt",
                    "backend_parameters": {
                        "backend": "himt",
                        "kmer_length": 31,
                        "accuracy": 0.61,
                    },
                    "threads": 2,
                    "environment_source": "managed",
                },
            },
            registry=registry,
            granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
        )

    assert response == {
        "ok": True,
        "operation_id": "assembly.assemble",
        "result": result.model_dump(mode="json"),
    }
    execute.assert_called_once()
    request = execute.call_args[0][0]
    assert request.method == "himt"
    assert request.backend_parameters == HimtParameters(
        backend="himt", kmer_length=31, accuracy=0.61
    )


def test_himt_agent_json_rejects_oatk_parameters_for_himt_method(tmp_path: Path) -> None:
    from organelleverse.assembly.operations import ASSEMBLE_SPEC

    registry = _released_registry()
    data = _valid_reads_data()
    with patch("organelleverse.assembly.service.execute_assembly") as execute:
        response = invoke_json(
            {
                "operation_id": "assembly.assemble",
                "input": data.model_dump(mode="json"),
                "parameters": {
                    "organelle": "mitochondrion",
                    "method": "himt",
                    "backend_parameters": {"backend": "oatk", "kmer_size": 1001},
                },
            },
            registry=registry,
            granted_side_effects=set(ASSEMBLE_SPEC.side_effects),
        )
    assert response["ok"] is False
    error = cast(dict[str, object], response["error"])
    assert error["error_code"] == "parameter.invalid_operation_parameters"
    execute.assert_not_called()


def test_getorganelle_backend_parameters_schema_is_strict_and_closed() -> None:
    from pydantic import ValidationError

    from organelleverse.assembly.contracts import GetOrganelleParameters
    from organelleverse.operations.parameters import assert_admissible_parameter_model

    # The model is admissible as an Agent-facing structured parameter object.
    assert_admissible_parameter_model(GetOrganelleParameters)

    schema = GetOrganelleParameters.model_json_schema()
    assert schema["additionalProperties"] is False
    properties = cast(dict[str, object], schema["properties"])
    backend_properties = cast(dict[str, object], properties["backend"])
    assert backend_properties["const"] == "getorganelle"
    assert _backend_discriminators([cast(dict[str, object], schema)]) == {"getorganelle"}

    # No free-form argv, raw extra-args, or dependency path channel is exposed.
    forbidden = {"extra_args", "argv", "raw_args", "extra_argv", "dependency_path"}
    assert not (forbidden & set(properties))

    # Strict JSON decoding rejects string coercion into integer fields.
    with pytest.raises(ValidationError):
        GetOrganelleParameters.model_validate({"max_reads": "15"})
