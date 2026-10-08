from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord, OrganelleData
from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import ErrorDetail, Finding, OperationSuggestion, OrganelleResult
from organelleverse.core.serialization import load_data, load_genome, load_result, save_contract


def _artifact() -> ArtifactRef:
    return ArtifactRef(
        kind="sequence",
        uri="genome.fa",
        format="fasta",
        sha256="a" * 64,
        size_bytes=8,
    )


def _genome() -> OrganelleGenome:
    parent = OrganelleData(modality="raw")
    return OrganelleGenome(
        organelle="mitochondrion",
        sequence=_artifact(),
        metadata=OrganelleMetadata(species="Example species", source="test"),
        lineage=(
            LineageRecord(
                parent_object_ids=(parent.object_id,),
                operation_id="assembly.assemble",
                operation_version="1.0",
                parameters_hash="b" * 64,
            ),
        ),
        source_manifests=(
            ArtifactRef(
                kind="assembly_run_manifest",
                uri="run.json",
                format="json",
                media_type="application/json",
                sha256="d" * 64,
                size_bytes=128,
            ),
        ),
    )


def _data() -> OrganelleData:
    parent = OrganelleData(modality="raw")
    return OrganelleData.model_validate(
        {
            "modality": "erc",
            "artifacts": {"alignment": _artifact()},
            "payload": {"object_id": "user-payload-id", "genes": ["cox1", "nad1"]},
            "dimensions": {"genes": 2},
            "metadata": {"source": {"object_id": "user-metadata-id"}},
            "lineage": (
                LineageRecord(
                    parent_object_ids=(parent.object_id,),
                    operation_id="coevolution.preprocess",
                    operation_version="1.0",
                    parameters_hash="b" * 64,
                ),
            ),
        }
    )


def _result() -> OrganelleResult:
    artifact = _artifact()
    provenance = ResultProvenance.model_validate(
        {
            "operation_id": "localization.predict",
            "operation_version": "1.0",
            "package_version": "0.1.0",
            "git_commit": "abc1234",
            "input_object_ids": ("genome:sha256:abc",),
            "parameters_hash": "c" * 64,
            "software_versions": {"object_id": "user-version-id"},
            "upstream_run_manifest_ids": ("assembly-run:sha256:" + "d" * 64,),
        }
    )
    return OrganelleResult.model_validate(
        {
            "operation_id": "quality_control.assess",
            "scope": "mitochondrion",
            "status": "warning",
            "metrics": {"object_id": "user-metric-id", "coverage": [42.0]},
            "findings": (Finding(code="quality.low_coverage", value=42.0),),
            "artifacts": (artifact,),
            "provenance": provenance,
            "errors": (
                ErrorDetail.model_validate(
                    {
                        "code": "quality.low_coverage",
                        "message": "coverage is below target",
                        "details": {"object_id": "user-detail-id"},
                    }
                ),
            ),
            "suggested_operations": (
                OperationSuggestion.model_validate(
                    {
                        "operation_id": "quality_control.assess",
                        "reason_code": "quality.low_coverage",
                        "parameter_changes": {"object_id": "user-suggestion-id"},
                    }
                ),
            ),
        }
    )


@pytest.mark.parametrize(
    ("contract", "loader"),
    [
        (_genome, load_genome),
        (_data, load_data),
        (_result, load_result),
    ],
)
def test_saved_contract_round_trips_exact_type_equality_and_identity(
    tmp_path: Path,
    contract: Callable[[], OrganelleGenome | OrganelleData | OrganelleResult],
    loader: Callable[[str | Path], OrganelleGenome | OrganelleData | OrganelleResult],
) -> None:
    original = contract()
    manifest = tmp_path / "contract.json"

    save_contract(original, manifest)
    restored = loader(manifest)

    assert type(restored) is type(original)
    assert restored == original
    assert restored.object_id == original.object_id
    assert json.loads(manifest.read_text(encoding="utf-8"))["kind"] == original.kind

    if isinstance(restored, OrganelleGenome):
        assert restored.source_manifests[0].sha256 == "d" * 64
    if isinstance(restored, OrganelleResult):
        assert restored.provenance is not None
        assert restored.provenance.upstream_run_manifest_ids == ("assembly-run:sha256:" + "d" * 64,)


def test_default_persistence_keeps_user_object_id_values_in_nested_json(tmp_path: Path) -> None:
    manifest = tmp_path / "result.json"
    original = _result()

    save_contract(original, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    restored = load_result(manifest)

    assert payload["metrics"]["object_id"] == "user-metric-id"
    assert payload["provenance"]["software_versions"]["object_id"] == "user-version-id"
    assert payload["errors"][0]["details"]["object_id"] == "user-detail-id"
    assert payload["suggested_operations"][0]["parameter_changes"]["object_id"] == (
        "user-suggestion-id"
    )
    assert restored.metrics["object_id"] == "user-metric-id"
    assert restored.provenance is not None
    assert restored.provenance.software_versions["object_id"] == "user-version-id"
    assert restored.errors[0].details["object_id"] == "user-detail-id"
    assert restored.suggested_operations[0].parameter_changes["object_id"] == "user-suggestion-id"


def test_saved_manifest_excludes_computed_object_ids_from_nested_contracts(tmp_path: Path) -> None:
    genome_manifest = tmp_path / "genome.json"
    data_manifest = tmp_path / "data.json"
    result_manifest = tmp_path / "result.json"

    save_contract(_genome(), genome_manifest)
    save_contract(_data(), data_manifest)
    save_contract(_result(), result_manifest)

    genome = json.loads(genome_manifest.read_text(encoding="utf-8"))
    data = json.loads(data_manifest.read_text(encoding="utf-8"))
    result = json.loads(result_manifest.read_text(encoding="utf-8"))

    assert "object_id" not in genome
    assert "object_id" not in genome["sequence"]
    assert "object_id" not in genome["metadata"]
    assert "object_id" not in genome["lineage"][0]
    assert "object_id" not in genome["source_manifests"][0]
    assert "object_id" not in data
    assert "object_id" not in data["artifacts"]["alignment"]
    assert "object_id" not in data["lineage"][0]
    assert "object_id" not in result
    assert "object_id" not in result["findings"][0]
    assert "object_id" not in result["artifacts"][0]
    assert "object_id" not in result["provenance"]
    assert "object_id" not in result["errors"][0]
    assert "object_id" not in result["suggested_operations"][0]


def _set_manifest_value(payload: dict[str, Any], path: tuple[str | int, ...], value: str) -> None:
    target: Any = payload
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("contract", "loader", "path"),
    [
        (_genome, load_genome, ("object_id",)),
        (_genome, load_genome, ("sequence", "object_id")),
        (_genome, load_genome, ("metadata", "object_id")),
        (_genome, load_genome, ("source_manifests", 0, "object_id")),
        (_data, load_data, ("artifacts", "alignment", "object_id")),
        (_data, load_data, ("lineage", 0, "object_id")),
        (_result, load_result, ("provenance", "object_id")),
        (_result, load_result, ("findings", 0, "object_id")),
        (_result, load_result, ("errors", 0, "object_id")),
        (_result, load_result, ("suggested_operations", 0, "object_id")),
    ],
)
def test_loader_rejects_forged_computed_object_id_at_every_contract_position(
    tmp_path: Path,
    contract: Callable[[], OrganelleGenome | OrganelleData | OrganelleResult],
    loader: Callable[[str | Path], OrganelleGenome | OrganelleData | OrganelleResult],
    path: tuple[str | int, ...],
) -> None:
    manifest = tmp_path / "contract.json"
    save_contract(contract(), manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    _set_manifest_value(payload, path, "forged:sha256:" + "f" * 64)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OrganelleContractError) as raised:
        loader(manifest)

    details = raised.value.as_dict()["details"]
    assert details["validation_errors"][0]["type"] == "extra_forbidden"
    assert details["validation_errors"][0]["loc"][-1] == "object_id"


def test_loader_wraps_json_readable_validation_errors_in_json_safe_contract_error(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "genome.json"
    save_contract(_genome(), manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sequence"] = None
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OrganelleContractError) as raised:
        load_genome(manifest)

    error = raised.value.as_dict()
    assert raised.value.code == "contract.invalid_persistence_manifest"
    assert error["details"]["validation_errors"][0]["loc"] == []
    assert error["details"]["validation_errors"][0]["message"] == (
        "Value error, OrganelleGenome requires sequence or annotation"
    )
    json.dumps(error, allow_nan=False)


def test_wrong_typed_loader_reports_expected_and_actual_kind(tmp_path: Path) -> None:
    manifest = tmp_path / "data.json"
    save_contract(_data(), manifest)

    with pytest.raises(OrganelleContractError) as raised:
        load_genome(manifest)

    assert raised.value.details == {"actual_kind": "data", "expected_kind": "genome"}


def test_loader_rejects_unknown_schema_version(tmp_path: Path) -> None:
    manifest = tmp_path / "genome.json"
    save_contract(_genome(), manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = "organelleverse.genome.v999"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OrganelleContractError, match="schema"):
        load_genome(manifest)


def test_loader_rejects_manifest_extra_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "genome.json"
    save_contract(_genome(), manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OrganelleContractError, match="invalid"):
        load_genome(manifest)


@pytest.mark.parametrize(
    "contents",
    ["{", "[1, 2, 3]"],
)
def test_loader_converts_read_and_json_failures_to_input_errors(
    tmp_path: Path, contents: str
) -> None:
    manifest = tmp_path / "invalid.json"
    manifest.write_text(contents, encoding="utf-8")

    with pytest.raises(OrganelleInputError):
        load_genome(manifest)

    with pytest.raises(OrganelleInputError):
        load_data(tmp_path / "missing.json")
