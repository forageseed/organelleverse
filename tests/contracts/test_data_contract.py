import json
from typing import cast

import pytest
from pydantic import ValidationError

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord, OrganelleData
from organelleverse.core.frozen import FrozenMap


def test_data_recursively_freezes_payload_and_metadata() -> None:
    data = OrganelleData.model_validate(
        {
            "modality": "erc",
            "payload": {"genes": ["cox1", "nad1"]},
            "dimensions": {"genes": 2},
            "metadata": {"species": ["A", "B"]},
        }
    )
    assert data.payload["genes"] == ("cox1", "nad1")
    with pytest.raises(TypeError):
        data.payload["genes"] += ("atp1",)  # type: ignore[index]


def test_data_rejects_arbitrary_runtime_objects() -> None:
    with pytest.raises(ValidationError, match="not JSON-compatible"):
        OrganelleData.model_validate({"modality": "erc", "payload": {"runtime": object()}})


def test_transform_lineage_is_explicit_and_non_mutating() -> None:
    raw = OrganelleData(modality="orthology")
    lineage = LineageRecord(
        parent_object_ids=(raw.object_id,),
        operation_id="coevolution.preprocess",
        operation_version="1.0",
        parameters_hash="a" * 64,
    )
    prepared = raw.evolve(modality="erc", lineage=(lineage,))
    assert prepared is not raw
    assert raw.lineage == ()
    assert prepared.lineage[0].parent_object_ids == (raw.object_id,)


@pytest.mark.parametrize(
    "artifact",
    [
        object(),
        {
            "kind": "alignment",
            "uri": "alignment.fa",
            "format": "fasta",
            "sha256": "a" * 64,
            "size_bytes": 4,
        },
    ],
)
def test_data_artifacts_require_artifact_refs(artifact: object) -> None:
    with pytest.raises(ValidationError, match="ArtifactRef"):
        OrganelleData(
            modality="erc",
            artifacts=cast(FrozenMap[ArtifactRef], {"alignment": artifact}),
        )


def test_data_revalidates_existing_artifact_refs() -> None:
    invalid = ArtifactRef.model_construct(
        kind="x",
        uri="",
        format="",
        sha256="bad",
        size_bytes=-1,
    )

    for validate in (
        lambda: OrganelleData(
            modality="erc",
            artifacts=cast(FrozenMap[ArtifactRef], {"alignment": invalid}),
        ),
        lambda: OrganelleData.model_validate(
            {"modality": "erc", "artifacts": {"alignment": invalid}}
        ),
    ):
        with pytest.raises(ValidationError):
            validate()


def test_data_serializes_frozen_contract_fields_as_normal_json() -> None:
    data = OrganelleData.model_validate(
        {
            "modality": "erc",
            "payload": {"genes": ["cox1"]},
            "dimensions": {"genes": 1},
            "metadata": {"source": {"study": "example"}},
        }
    )

    assert data.model_dump(mode="json")["payload"] == {"genes": ["cox1"]}
    assert json.loads(data.model_dump_json())["metadata"] == {"source": {"study": "example"}}


def test_data_json_dump_with_artifacts_round_trips_through_model_validate() -> None:
    artifact = ArtifactRef(
        kind="alignment",
        uri="alignment.fa",
        format="fasta",
        sha256="a" * 64,
        size_bytes=4,
    )
    data = OrganelleData.model_validate({"modality": "erc", "artifacts": {"alignment": artifact}})

    dumped = data.model_dump(mode="json", exclude={"object_id"})
    with pytest.raises(ValidationError, match="ArtifactRef"):
        OrganelleData(modality="erc", artifacts=dumped["artifacts"])

    restored = OrganelleData.model_validate(dumped)

    assert restored == data
    assert restored.artifacts["alignment"] == artifact


def test_data_rejects_forged_serialized_artifact_object_id() -> None:
    artifact = ArtifactRef(
        kind="alignment",
        uri="alignment.fa",
        format="fasta",
        sha256="a" * 64,
        size_bytes=4,
    )
    data = OrganelleData.model_validate({"modality": "erc", "artifacts": {"alignment": artifact}})
    serialized = data.model_dump()

    assert OrganelleData.model_validate(serialized) == data

    serialized["artifacts"]["alignment"]["object_id"] = "forged"
    with pytest.raises(ValidationError, match="object_id"):
        OrganelleData.model_validate(serialized)


def test_data_json_dump_with_artifacts_and_lineage_round_trips() -> None:
    artifact = ArtifactRef(
        kind="alignment",
        uri="alignment.fa",
        format="fasta",
        sha256="a" * 64,
        size_bytes=4,
    )
    parent = OrganelleData(modality="raw")
    lineage = LineageRecord(
        parent_object_ids=(parent.object_id,),
        operation_id="coevolution.preprocess",
        operation_version="1.0",
        parameters_hash="a" * 64,
    )
    data = OrganelleData.model_validate(
        {"modality": "erc", "artifacts": {"alignment": artifact}, "lineage": (lineage,)}
    )

    dumped = data.model_dump(mode="json", exclude={"object_id"})
    assert "object_id" not in dumped["lineage"][0]
    restored = OrganelleData.model_validate(dumped)

    assert restored == data
    assert restored.object_id == data.object_id
    assert restored.artifacts["alignment"] == artifact
    assert restored.lineage == data.lineage


def test_data_json_text_with_artifacts_and_lineage_round_trips() -> None:
    artifact = ArtifactRef(
        kind="alignment",
        uri="alignment.fa",
        format="fasta",
        sha256="a" * 64,
        size_bytes=4,
    )
    parent = OrganelleData(modality="raw")
    lineage = LineageRecord(
        parent_object_ids=(parent.object_id,),
        operation_id="coevolution.preprocess",
        operation_version="1.0",
        parameters_hash="a" * 64,
    )
    data = OrganelleData.model_validate(
        {"modality": "erc", "artifacts": {"alignment": artifact}, "lineage": (lineage,)}
    )

    dumped_json = data.model_dump_json(exclude={"object_id"})
    restored = OrganelleData.model_validate_json(dumped_json)

    assert restored == data
    assert restored.object_id == data.object_id
    assert restored.artifacts["alignment"] == artifact
    assert restored.lineage == data.lineage


class MutableHashStr(str):
    hash_value: int

    def __new__(cls, value: str, hash_value: int) -> "MutableHashStr":
        instance = super().__new__(cls, value)
        instance.hash_value = hash_value
        return instance

    def __hash__(self) -> int:
        return self.hash_value


def test_data_canonicalizes_artifact_keys_before_freezing() -> None:
    key = MutableHashStr("alignment", 1)
    artifact = ArtifactRef(
        kind="alignment",
        uri="alignment.fa",
        format="fasta",
        sha256="a" * 64,
        size_bytes=4,
    )
    data = OrganelleData.model_validate({"modality": "erc", "artifacts": {key: artifact}})
    artifacts_hash = hash(data.artifacts)
    data_hash = hash(data)
    object_id = data.object_id

    key.hash_value = 2

    assert type(next(iter(data.artifacts))) is str
    assert hash(data.artifacts) == artifacts_hash
    assert hash(data) == data_hash
    assert data.object_id == object_id


@pytest.mark.parametrize("dimensions", [{"genes": -1}, {"genes": True}, {"genes": 1.0}])
def test_data_rejects_invalid_dimension_counts(dimensions: object) -> None:
    with pytest.raises(ValidationError, match="non-negative integers"):
        OrganelleData.model_validate({"modality": "erc", "dimensions": dimensions})


def test_data_identity_uses_artifact_object_ids_not_uris() -> None:
    first = ArtifactRef(
        kind="alignment", uri="first.fa", format="fasta", sha256="a" * 64, size_bytes=4
    )
    second = ArtifactRef(
        kind="alignment", uri="second.fa", format="fasta", sha256="a" * 64, size_bytes=4
    )

    assert first.uri != second.uri
    assert (
        OrganelleData.model_validate(
            {"modality": "erc", "artifacts": {"alignment": first}}
        ).object_id
        == OrganelleData.model_validate(
            {"modality": "erc", "artifacts": {"alignment": second}}
        ).object_id
    )
