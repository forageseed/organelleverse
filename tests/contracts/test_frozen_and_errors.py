import hashlib
import inspect
import json
import math
from collections.abc import Mapping
from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.base import StrictFrozenModel
from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleError,
    OrganelleInternalError,
    OrganelleNeedsInput,
)
from organelleverse.core.frozen import FrozenJson, FrozenMap, freeze_json, thaw_json
from organelleverse.core.input_requests import (
    NeedsInputRequest,
    make_needs_input_request,
)

NEEDS_INPUT_SCHEMA_VERSION = "organelleverse.needs-input.v1"


def _frozen_object(value: FrozenJson) -> FrozenMap[FrozenJson]:
    assert isinstance(value, FrozenMap)
    return value


def _dependency_error_with_payload(
    argument: str, payload: dict[str, object]
) -> OrganelleDependencyError:
    if argument == "details":
        return OrganelleDependencyError(
            code="dependency.missing_executable",
            message="blastn is required",
            details=payload,
        )
    return OrganelleDependencyError(
        code="dependency.missing_executable",
        message="blastn is required",
        suggested_action=payload,
    )


def test_freeze_json_normalizes_scalar_subclasses_and_mapping_keys() -> None:
    class MutableString(str):
        mutable: list[object]

        def __new__(cls, value: str) -> "MutableString":
            instance = super().__new__(cls, value)
            instance.mutable = []
            return instance

    scalar = MutableString("value")
    key = MutableString("key")
    frozen = freeze_json({key: scalar})

    frozen_object = _frozen_object(frozen)
    assert type(next(iter(frozen_object))) is str
    assert type(frozen_object["key"]) is str
    assert thaw_json(frozen) == {"key": "value"}


def test_freeze_json_recursively_prevents_mutation() -> None:
    frozen = freeze_json({"genes": ["cox1", {"score": 0.8}]})
    frozen_object = _frozen_object(frozen)
    genes = frozen_object["genes"]
    assert isinstance(genes, tuple)
    score = genes[1]
    assert isinstance(score, FrozenMap)
    assert score["score"] == 0.8
    with pytest.raises(TypeError):
        frozen_object["genes"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        score["score"] = 1.0  # type: ignore[index]


def test_frozen_map_exposes_no_mutable_backing_store() -> None:
    frozen = freeze_json({"genes": ["cox1"]})
    assert isinstance(frozen, FrozenMap)
    original_hash = hash(frozen)

    with pytest.raises(AttributeError):
        frozen._data["genes"] = ()  # type: ignore[attr-defined]

    assert frozen["genes"] == ("cox1",)
    assert tuple(frozen) == ("genes",)
    assert hash(frozen) == original_hash


def test_frozen_map_constructor_recursively_freezes_mutable_values() -> None:
    source = {"genes": ["cox1"]}
    frozen = FrozenMap.from_json(source)

    source["genes"].append("atp6")

    assert frozen["genes"] == ("cox1",)
    with pytest.raises(AttributeError):
        frozen["genes"].append("atp6")  # type: ignore[union-attr]


def test_frozen_map_from_items_canonicalizes_mutable_string_keys_and_values() -> None:
    class MutableHashString(str):
        hash_value: int

        def __new__(cls, value: str, hash_value: int) -> "MutableHashString":
            instance = super().__new__(cls, value)
            instance.hash_value = hash_value
            return instance

        def __hash__(self) -> int:
            return self.hash_value

    key = MutableHashString("alignment", 1)
    value = MutableHashString("ready", 2)
    frozen = FrozenMap.from_items({key: value})
    original_hash = hash(frozen)

    key.hash_value = 3
    value.hash_value = 4

    assert tuple(frozen) == ("alignment",)
    assert type(next(iter(frozen))) is str
    assert type(frozen["alignment"]) is str
    assert frozen["alignment"] == "ready"
    assert hash(frozen) == original_hash


def test_frozen_map_from_items_recursively_freezes_mutable_container_values() -> None:
    source = {"genes": ["cox1"]}
    json_typed_source = cast(Mapping[str, FrozenJson], source)

    frozen = FrozenMap.from_items(json_typed_source)
    source["genes"].append("atp6")

    assert frozen["genes"] == ("cox1",)


def test_frozen_map_from_items_preserves_artifact_ref_values() -> None:
    artifact = ArtifactRef(
        kind="alignment",
        uri="alignment.fa",
        format="fasta",
        sha256="a" * 64,
        size_bytes=4,
    )

    frozen = FrozenMap.from_items({"alignment": artifact})

    assert frozen["alignment"] is artifact
    assert hash(frozen) == hash(frozen)


def test_frozen_map_from_items_rejects_mutable_strict_frozen_model_subclasses() -> None:
    class MutableHashModel(StrictFrozenModel[Literal["mutable"]]):
        kind: Literal["mutable"] = "mutable"
        mutable: list[str]

        def __hash__(self) -> int:
            return 1

    model = MutableHashModel(mutable=["cox1"])

    with pytest.raises(OrganelleInternalError) as error:
        FrozenMap.from_items(cast(Any, {"unsafe": model}))

    assert error.value.code == "contract.non_frozen_value"


def test_freeze_json_rejects_non_json_values() -> None:
    with pytest.raises(OrganelleInternalError, match="not JSON-compatible"):
        freeze_json(object())
    with pytest.raises(OrganelleInternalError, match="finite"):
        freeze_json(math.nan)


def test_freeze_json_rejects_cyclic_mapping() -> None:
    value: dict[str, object] = {}
    value["self"] = value

    with pytest.raises(OrganelleInternalError) as error:
        freeze_json(value)

    assert error.value.code == "contract.cyclic_json_value"


def test_freeze_json_rejects_cyclic_list() -> None:
    value: list[object] = []
    value.append(value)

    with pytest.raises(OrganelleInternalError) as error:
        freeze_json(value)

    assert error.value.code == "contract.cyclic_json_value"


def test_freeze_json_rejects_cyclic_tuple() -> None:
    values: list[object] = []
    value = (values,)
    values.append(value)

    with pytest.raises(OrganelleInternalError) as error:
        freeze_json(value)

    assert error.value.code == "contract.cyclic_json_value"


def test_freeze_json_allows_shared_non_cyclic_values() -> None:
    shared = {"score": 0.8}

    assert thaw_json(freeze_json({"first": shared, "second": shared})) == {
        "first": {"score": 0.8},
        "second": {"score": 0.8},
    }


def test_thaw_json_restores_builtin_containers() -> None:
    original = {"a": [1, 2], "b": {"ok": True}}
    assert thaw_json(freeze_json(original)) == original


def test_structured_error_payload_is_agent_readable() -> None:
    error = OrganelleDependencyError(
        code="dependency.missing_executable",
        message="blastn is required",
        details={"executable": "blastn"},
        retryable=True,
        suggested_action={"operation_id": "annotation.annotate"},
    )
    assert str(error) == "blastn is required"
    payload = error.as_dict()
    assert payload["error_code"] == "dependency.missing_executable"
    assert payload["retryable"] is True
    assert payload["details"] == {"executable": "blastn"}


def test_structured_error_payloads_are_recursively_frozen_and_thawed() -> None:
    details = {"nested": [{"status": "ready"}]}
    suggested_action = {"next": ["retry"]}
    error = OrganelleDependencyError(
        code="dependency.missing_executable",
        message="blastn is required",
        details=details,
        suggested_action=suggested_action,
    )
    details["nested"][0]["status"] = "changed"
    suggested_action["next"].append("install")

    payload = error.as_dict()
    assert payload["details"] == {"nested": [{"status": "ready"}]}
    assert payload["suggested_action"] == {"next": ["retry"]}
    assert isinstance(payload["details"], dict)
    assert isinstance(payload["details"]["nested"], list)

    payload["details"]["nested"][0]["status"] = "changed"
    assert error.as_dict()["details"] == {"nested": [{"status": "ready"}]}


@pytest.mark.parametrize(
    "payload",
    [
        {"value": object()},
        {"value": math.nan},
        {"value": math.inf},
        {1: "value"},
    ],
    ids=["object", "nan", "infinity", "non-string-key"],
)
@pytest.mark.parametrize("argument", ["details", "suggested_action"])
def test_structured_error_rejects_non_json_payloads(argument: str, payload: object) -> None:
    with pytest.raises(OrganelleInternalError):
        _dependency_error_with_payload(argument, {"value": payload})


@pytest.mark.parametrize("argument", ["details", "suggested_action"])
def test_structured_error_rejects_cyclic_payloads(argument: str) -> None:
    payload: dict[str, object] = {}
    payload["self"] = payload

    with pytest.raises(OrganelleInternalError) as error:
        _dependency_error_with_payload(argument, payload)

    assert error.value.code == "contract.cyclic_json_value"


def _expected_request_id(
    *, semantic_payload: Mapping[str, object], field: str, choices: tuple[str, ...]
) -> str:
    canonical = json.dumps(
        {
            "schema_version": NEEDS_INPUT_SCHEMA_VERSION,
            "semantic_payload": dict(semantic_payload),
            "field": field,
            "choices": choices,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def test_needs_input_request_id_is_deterministic_and_destination_free() -> None:
    payload = {"organelle": "mitochondrion", "requested_method": "auto"}
    choices = ("use_default_oatk", "provide_genome_size")

    first = make_needs_input_request(
        semantic_payload=payload, field="auxiliary.genome_size", choices=choices
    )
    second = make_needs_input_request(
        semantic_payload=payload, field="auxiliary.genome_size", choices=choices
    )

    assert first.request_id == second.request_id
    assert first.request_id == _expected_request_id(
        semantic_payload=payload, field="auxiliary.genome_size", choices=choices
    )
    assert first.request_id.startswith("sha256:")
    # Destination paths are deliberately excluded: the factory takes only the
    # semantic invocation payload, so output_dir can never influence the identity.
    assert "output_dir" not in inspect.signature(make_needs_input_request).parameters


def test_needs_input_request_id_is_independent_of_payload_dict_ordering() -> None:
    ordered = make_needs_input_request(
        semantic_payload={"a": 1, "b": 2},
        field="auxiliary.genome_size",
        choices=("use_default_oatk", "provide_genome_size"),
    )
    reordered = make_needs_input_request(
        semantic_payload={"b": 2, "a": 1},
        field="auxiliary.genome_size",
        choices=("use_default_oatk", "provide_genome_size"),
    )
    assert ordered.request_id == reordered.request_id


def test_needs_input_request_id_changes_when_field_choices_or_payload_change() -> None:
    payload = {"organelle": "mitochondrion"}
    base = make_needs_input_request(
        semantic_payload=payload, field="auxiliary.genome_size", choices=("a", "b")
    )
    changed_field = make_needs_input_request(
        semantic_payload=payload, field="auxiliary.genome_size_bp", choices=("a", "b")
    )
    changed_choice = make_needs_input_request(
        semantic_payload=payload, field="auxiliary.genome_size", choices=("a", "c")
    )
    changed_choice_order = make_needs_input_request(
        semantic_payload=payload, field="auxiliary.genome_size", choices=("b", "a")
    )
    changed_payload = make_needs_input_request(
        semantic_payload={**payload, "organelle": "plastid"},
        field="auxiliary.genome_size",
        choices=("a", "b"),
    )
    for other in (changed_field, changed_choice, changed_choice_order, changed_payload):
        assert other.request_id != base.request_id


def test_needs_input_request_rejects_duplicate_choices() -> None:
    with pytest.raises(ValidationError, match="unique"):
        make_needs_input_request(
            semantic_payload={"a": 1},
            field="auxiliary.genome_size",
            choices=("use_default_oatk", "use_default_oatk"),
        )


@pytest.mark.parametrize(
    "choices",
    [
        (),
        ("",),
        ("a", ""),
    ],
)
def test_needs_input_request_rejects_empty_choices(choices: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        make_needs_input_request(
            semantic_payload={"a": 1},
            field="auxiliary.genome_size",
            choices=choices,
        )


def test_needs_input_request_is_frozen_extra_forbid_and_strict() -> None:
    request = make_needs_input_request(
        semantic_payload={"a": 1}, field="auxiliary.genome_size", choices=("a", "b")
    )

    with pytest.raises(ValidationError) as frozen:
        request.field = "other"  # type: ignore[misc]
    assert frozen.value.errors()[0]["type"] == "frozen_instance"

    dumped = request.model_dump(mode="python", round_trip=True)
    dumped["unexpected"] = True
    with pytest.raises(ValidationError) as extra:
        NeedsInputRequest.model_validate(dumped)
    assert extra.value.errors()[0]["type"] == "extra_forbidden"

    # Strict mode rejects an uncoerced non-string field.
    with pytest.raises(ValidationError):
        NeedsInputRequest.model_validate(
            {
                "schema_version": NEEDS_INPUT_SCHEMA_VERSION,
                "request_id": "sha256:" + "0" * 64,
                "field": 5,
                "choices": ("a",),
            }
        )


def test_needs_input_request_pinpoints_invalid_request_id_prefix() -> None:
    with pytest.raises(ValidationError) as raised:
        NeedsInputRequest.model_validate(
            {
                "schema_version": NEEDS_INPUT_SCHEMA_VERSION,
                "request_id": "md5:" + "0" * 64,
                "field": "auxiliary.genome_size",
                "choices": ("a",),
            }
        )
    assert raised.value.errors()[0]["loc"] == ("request_id",)


def test_needs_input_request_carries_closed_schema_version() -> None:
    request = make_needs_input_request(
        semantic_payload={"a": 1}, field="auxiliary.genome_size", choices=("a",)
    )
    assert request.schema_version == NEEDS_INPUT_SCHEMA_VERSION
    dumped = request.model_dump(mode="json")
    assert dumped["choices"] == ["a"]
    assert json.dumps(dumped, allow_nan=False)  # envelope stays JSON-safe


def test_needs_input_request_does_not_carry_semantic_payload_or_answers() -> None:
    request = make_needs_input_request(
        semantic_payload={"secret": "value"}, field="auxiliary.genome_size", choices=("a",)
    )
    dumped = request.model_dump(mode="json")
    # The request envelope is generic: it records what is missing, not the
    # invocation payload or any answer the caller may later supply.
    assert "semantic_payload" not in dumped
    assert "answer" not in dumped


def test_organelle_needs_input_carries_request_and_readable_error_payload() -> None:
    request = make_needs_input_request(
        semantic_payload={"organelle": "mitochondrion"},
        field="auxiliary.genome_size",
        choices=("use_default_oatk", "provide_genome_size", "lookup_by_species"),
    )
    error = OrganelleNeedsInput(
        code="assembly.genome_size_required",
        message="nuclear genome size is required for coverage routing",
        needs_input=request,
    )

    assert isinstance(error, OrganelleError)
    assert error.needs_input is request
    assert error.as_dict() == {
        "error_code": "assembly.genome_size_required",
        "message": "nuclear genome size is required for coverage routing",
        "details": {},
        "retryable": False,
        "suggested_action": {},
    }
    assert str(error) == "nuclear genome size is required for coverage routing"
