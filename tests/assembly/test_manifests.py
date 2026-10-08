from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from organelleverse.assembly.contracts import (
    AssemblyInputPayload,
    AssemblyRequest,
    LongLibraryContract,
    ShortLibraryContract,
)
from organelleverse.assembly.manifests import (
    AssemblyComponentIdentity,
    AssemblyEnvironmentIdentity,
    AssemblyRunManifest,
    AssemblyRunObservations,
    AssemblyRunParameters,
    AssemblyStageOutcome,
    ManifestArtifact,
    RoutingEvidence,
    assembly_run_id_from_artifact,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import LineageRecord, OrganelleData
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult


def _artifact(
    *,
    kind: str,
    uri: str,
    format: str,
    media_type: str,
    sha256: str,
    size_bytes: int,
    validated: bool = False,
) -> ArtifactRef:
    return ArtifactRef(
        kind=kind,
        uri=uri,
        format=format,
        media_type=media_type,
        sha256=sha256,
        size_bytes=size_bytes,
        validated=validated,
    )


def _data(
    *,
    read_uri: str = "reads/long.fastq",
    read_hash: str = "a" * 64,
    validated: bool = False,
) -> OrganelleData:
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "long_reads": _artifact(
                    kind="long_read",
                    uri=read_uri,
                    format="fastq",
                    media_type="application/x-fastq",
                    sha256=read_hash,
                    size_bytes=2048,
                    validated=validated,
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


def _request(
    tmp_path: Path,
    *,
    read_uri: str = "reads/long.fastq",
    output: str = "out",
    validated: bool = False,
) -> AssemblyRequest:
    return AssemblyRequest(
        data=_data(read_uri=read_uri, validated=validated),
        organelle="mitochondrion",
        method="oatk",
        threads=8,
        memory_gb=32,
        timeout_seconds=3600,
        environment_source="managed",
    )


def _manifest(
    *,
    input_uri: str = "reads/long.fastq",
    output_uri: str = "out/primary.fa",
) -> AssemblyRunManifest:
    data = _data(read_uri=input_uri)
    return AssemblyRunManifest(
        input_data_id=data.object_id,
        input_artifacts=(
            ManifestArtifact(role="long_reads", artifact=data.artifacts["long_reads"]),
        ),
        organelle="mitochondrion",
        parameters=AssemblyRunParameters(
            threads=8,
            memory_gb=32,
            timeout_seconds=3600,
            environment_source="managed",
            backend_parameters=cast(
                Any,
                {
                    "backend": "oatk",
                    "kmer_size": 1001,
                    "minimum_kmer_coverage": 30,
                },
            ),
        ),
        requested_method="oatk",
        selected_backend="oatk",
        route_reason_code="route.explicit_method",
        environment=AssemblyEnvironmentIdentity(
            carrier="conda",
            digest="sha256:" + "b" * 64,
            platform="linux-64",
        ),
        components=(
            AssemblyComponentIdentity(
                category="software",
                name="oatk",
                version="1.0",
                sha256="c" * 64,
            ),
            AssemblyComponentIdentity(
                category="database",
                name="oatkdb",
                version="2026.07",
                sha256="d" * 64,
            ),
        ),
        stable_argv=(
            "oatk",
            "mito",
            "-i",
            "role://artifact/long_reads",
            "-o",
            "role://workspace/assembly",
        ),
        stages=(
            AssemblyStageOutcome(stage="prepare", status="ok"),
            AssemblyStageOutcome(
                stage="assemble",
                status="ok",
                process_started=True,
                exit_code=0,
                termination="exit",
                output_roles=("primary_fasta", "assembly_graph"),
            ),
        ),
        process_exit_code=0,
        outputs=(
            ManifestArtifact(
                role="primary_fasta",
                artifact=_artifact(
                    kind="sequence",
                    uri=output_uri,
                    format="fasta",
                    media_type="text/x-fasta",
                    sha256="e" * 64,
                    size_bytes=4096,
                    validated=True,
                ),
            ),
            ManifestArtifact(
                role="assembly_graph",
                artifact=_artifact(
                    kind="assembly_graph",
                    uri="out/assembly.gfa",
                    format="gfa",
                    media_type="text/x-gfa",
                    sha256="f" * 64,
                    size_bytes=8192,
                    validated=True,
                ),
            ),
        ),
    )


def _observations(
    run_manifest_id: str,
    *,
    workspace: str = "/host/work",
) -> AssemblyRunObservations:
    started = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    return AssemblyRunObservations(
        run_manifest_id=run_manifest_id,
        started_at=started,
        finished_at=started + timedelta(seconds=12),
        duration_seconds=12,
        peak_memory_bytes=1_000_000,
        peak_cpu_percent=98.5,
        workspace_path=workspace,
        cache_path=f"{workspace}/cache",
        output_path=f"{workspace}/out",
        resolved_argv=("/opt/oatk/bin/oatk", "-i", f"{workspace}/reads.fastq"),
        stdout_path=f"{workspace}/stdout.log",
        stderr_path=f"{workspace}/stderr.log",
    )


def test_request_semantic_identity_ignores_artifact_and_output_paths(tmp_path: Path) -> None:
    first = _request(tmp_path, read_uri="host-a/reads.fastq", output="host-a/out")
    second = _request(
        tmp_path,
        read_uri="host-b/reads.fastq",
        output="host-b/out",
        validated=True,
    )

    assert first.data.object_id == second.data.object_id
    assert first.semantic_payload() == second.semantic_payload()
    assert first.semantic_hash == second.semantic_hash
    encoded = json.dumps(first.semantic_payload(), sort_keys=True)
    assert "host-a" not in encoded
    assert "validated" not in encoded
    assert (
        first.semantic_hash
        == hashlib.sha256(
            json.dumps(
                first.semantic_payload(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )


@pytest.mark.parametrize(
    ("model_field", "payload_field", "value"),
    [
        ("organelle", "organelle", "plastid"),
        ("method", "requested_method", "auto"),
        ("threads", "threads", 16),
        ("memory_gb", "memory_gb", 64),
        ("timeout_seconds", "timeout_seconds", 7200),
        ("environment_source", "environment_source", "existing"),
    ],
)
def test_every_request_execution_field_changes_semantic_hash(
    tmp_path: Path,
    model_field: str,
    payload_field: str,
    value: object,
) -> None:
    request = _request(tmp_path)
    changed = request.model_copy(update={model_field: value})

    assert changed.semantic_payload()[payload_field] != request.semantic_payload()[payload_field]
    assert changed.semantic_hash != request.semantic_hash


def test_request_input_data_identity_changes_semantic_hash_independently(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    changed_data = request.data.evolve(metadata={"sample": "different"})
    changed = request.model_copy(update={"data": changed_data})

    assert (
        changed.semantic_payload()["input_artifacts"]
        == request.semantic_payload()["input_artifacts"]
    )
    assert (
        changed.semantic_payload()["input_data_id"] != request.semantic_payload()["input_data_id"]
    )
    assert changed.semantic_hash != request.semantic_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "corrected_read"),
        ("format", "fq"),
        ("media_type", "application/octet-stream"),
        ("sha256", "1" * 64),
        ("size_bytes", 4096),
    ],
)
def test_every_request_artifact_content_field_changes_semantic_hash(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    request = _request(tmp_path)
    artifact = request.data.artifacts["long_reads"].model_copy(update={field: value})
    data = request.data.evolve(artifacts={"long_reads": artifact})
    changed = request.model_copy(update={"data": data})
    changed_artifacts = cast(
        dict[str, dict[str, object]], changed.semantic_payload()["input_artifacts"]
    )
    original_artifacts = cast(
        dict[str, dict[str, object]], request.semantic_payload()["input_artifacts"]
    )

    assert changed_artifacts["long_reads"][field] != original_artifacts["long_reads"][field]
    assert changed.semantic_hash != request.semantic_hash


def test_request_artifact_role_changes_semantic_hash(tmp_path: Path) -> None:
    request = _request(tmp_path)
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"renamed_reads": request.data.artifacts["long_reads"]},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "renamed_reads",
                    }
                ],
            },
        }
    )
    changed = request.model_copy(update={"data": data})
    changed_artifacts = cast(dict[str, object], changed.semantic_payload()["input_artifacts"])

    assert tuple(changed_artifacts) == ("renamed_reads",)
    assert changed.semantic_hash != request.semantic_hash


@pytest.mark.parametrize("volatile", ["artifact_uri", "artifact_validated"])
def test_each_request_volatile_field_is_independently_nonsemantic(
    tmp_path: Path,
    volatile: str,
) -> None:
    request = _request(tmp_path)
    update: dict[str, object] = (
        {"uri": "/different-host/reads.fastq"}
        if volatile == "artifact_uri"
        else {"validated": True}
    )
    artifact = request.data.artifacts["long_reads"].model_copy(update=update)
    data = request.data.evolve(artifacts={"long_reads": artifact})
    changed = request.model_copy(update={"data": data})

    assert changed.semantic_payload() == request.semantic_payload()
    assert changed.semantic_hash == request.semantic_hash


@pytest.mark.parametrize(
    "update",
    [
        {"threads": 0},
        {"method": "unknown"},
        {"environment_source": "create"},
    ],
)
def test_request_model_copy_revalidates_invalid_updates(
    tmp_path: Path,
    update: dict[str, object],
) -> None:
    request = _request(tmp_path)

    with pytest.raises(ValidationError):
        request.model_copy(update=update)


def test_request_model_copy_accepts_and_revalidates_a_valid_update(tmp_path: Path) -> None:
    request = _request(tmp_path)

    copied = request.model_copy(
        update={
            "method": "auto",
            "threads": 16,
            "environment_source": "existing",
        }
    )

    assert type(copied) is AssemblyRequest
    assert copied.method == "auto"
    assert copied.threads == 16
    assert copied.environment_source == "existing"
    assert copied.data == request.data


@pytest.mark.parametrize(
    "update",
    [
        {"threads": 0},
        {"method": "unknown"},
        {"environment_source": "create"},
    ],
)
def test_request_deprecated_copy_revalidates_invalid_updates(
    tmp_path: Path,
    update: dict[str, object],
) -> None:
    request = _request(tmp_path)

    with pytest.raises(ValidationError):
        request.copy(update=update)


def test_request_deprecated_copy_revalidates_sequencing_data(tmp_path: Path) -> None:
    request = _request(tmp_path)
    invalid_data = request.data.evolve(modality="annotation_table")

    with pytest.raises(OrganelleInputError, match="requires sequencing_reads"):
        request.copy(update={"data": invalid_data})


def test_request_deprecated_copy_revalidates_semantic_encoding(tmp_path: Path) -> None:
    request = _request(tmp_path)
    invalid_data = request.data.evolve(metadata={"sample": "\ud800"})

    with pytest.raises(ValidationError, match="canonical UTF-8 JSON"):
        request.copy(update={"data": invalid_data})


def test_request_deprecated_copy_accepts_a_valid_update_with_identity(tmp_path: Path) -> None:
    request = _request(tmp_path)

    copied = request.copy(
        update={
            "method": "auto",
            "threads": 16,
            "environment_source": "existing",
        }
    )

    assert type(copied) is AssemblyRequest
    assert copied.method == "auto"
    assert copied.threads == 16
    assert copied.environment_source == "existing"
    assert copied.semantic_hash != request.semantic_hash
    assert (
        copied.semantic_hash
        == hashlib.sha256(
            json.dumps(
                copied.semantic_payload(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )


def test_request_deprecated_copy_include_cannot_create_a_partial_model(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(ValidationError):
        request.copy(include={"threads"})


def test_request_deprecated_copy_exclude_cannot_create_a_partial_model(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(ValidationError):
        request.copy(exclude={"data"})


def test_request_deprecated_copy_deep_revalidates_and_preserves_identity(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    copied = request.copy(deep=True)

    assert type(copied) is AssemblyRequest
    assert copied is not request
    assert copied.data is not request.data
    assert copied.semantic_payload() == request.semantic_payload()
    assert copied.semantic_hash == request.semantic_hash


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
@pytest.mark.parametrize("semantic_field", ["artifact_format", "artifact_media_type", "metadata"])
def test_request_rejects_unencodable_semantic_identity(
    tmp_path: Path,
    semantic_field: str,
    surrogate: str,
) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValidationError, match=r"canonical UTF-8 JSON|valid string"):
        if semantic_field == "metadata":
            data = request.data.evolve(metadata={"sample": surrogate})
        else:
            artifact_field = semantic_field.removeprefix("artifact_")
            artifact = request.data.artifacts["long_reads"].model_copy(
                update={artifact_field: surrogate}
            )
            data = request.data.evolve(artifacts={"long_reads": artifact})
        request.model_copy(update={"data": data})


def test_request_valid_unicode_semantic_identity_is_deterministic(tmp_path: Path) -> None:
    request = _request(tmp_path)
    artifact = request.data.artifacts["long_reads"].model_copy(
        update={
            "format": "序列🧬",
            "media_type": "application/基因组+😀",
        }
    )
    data = request.data.evolve(
        artifacts={"long_reads": artifact},
        metadata={"样本": "叶绿体🌿"},
    )

    first = request.model_copy(update={"data": data})
    second = request.model_copy(update={"data": data})

    assert first.semantic_payload() == second.semantic_payload()
    assert first.semantic_hash == second.semantic_hash
    assert (
        first.semantic_hash
        == hashlib.sha256(
            json.dumps(
                first.semantic_payload(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )


def test_request_normal_json_round_trip_preserves_equality_hash_and_semantic_id(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    restored = AssemblyRequest.model_validate_json(request.model_dump_json())

    assert restored == request
    assert hash(restored) == hash(request)
    assert restored.semantic_hash == request.semantic_hash
    assert restored.data.object_id == request.data.object_id
    assert restored.data.artifacts["long_reads"].object_id == (
        request.data.artifacts["long_reads"].object_id
    )


@pytest.mark.parametrize("forged", ["data", "artifact"])
def test_request_json_round_trip_rejects_forged_nested_object_ids(
    tmp_path: Path,
    forged: str,
) -> None:
    request = _request(tmp_path)
    serialized = json.loads(request.model_dump_json())
    if forged == "data":
        serialized["data"]["object_id"] = "data:sha256:" + "0" * 64
    else:
        serialized["data"]["artifacts"]["long_reads"]["object_id"] = "long_read:sha256:" + "0" * 64

    with pytest.raises(ValidationError, match="object_id"):
        AssemblyRequest.model_validate_json(json.dumps(serialized))


def test_manifest_id_ignores_observations_and_artifact_uris() -> None:
    first = _manifest(input_uri="host-a/reads.fastq", output_uri="host-a/primary.fa")
    second = _manifest(input_uri="host-b/reads.fastq", output_uri="host-b/primary.fa")
    first_observations = _observations(first.run_manifest_id, workspace="/host-a/work")
    second_observations = _observations(second.run_manifest_id, workspace="/host-b/work")

    assert first.run_manifest_id == second.run_manifest_id
    assert first_observations != second_observations
    assert not hasattr(first_observations, "object_id")


def test_every_observation_field_is_excluded_from_manifest_identity() -> None:
    manifest = _manifest()
    observations = _observations(manifest.run_manifest_id)
    semantic_keys = set(manifest.semantic_payload())

    for field in type(observations).model_fields:
        if field not in {"schema_version", "run_manifest_id"}:
            assert field not in semantic_keys
    assert manifest.run_manifest_id not in manifest.canonical_bytes().decode("utf-8")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_data_id", "data:sha256:" + "1" * 64),
        ("organelle", "plastid"),
        ("requested_method", "auto"),
        ("route_reason_code", "route.compatible_profile"),
        ("stable_argv", ("oatk", "mito", "--threads=16")),
    ],
)
def test_every_manifest_top_level_fact_changes_run_id(field: str, value: object) -> None:
    base = _manifest()
    changed = base.model_copy(update={field: value})

    assert changed.semantic_payload()[field] != base.semantic_payload()[field]
    assert changed.run_manifest_id != base.run_manifest_id


def test_manifest_without_routing_evidence_has_none_by_default() -> None:
    manifest = _manifest()
    assert manifest.routing_evidence is None
    assert "routing_evidence" in manifest.semantic_payload()
    assert manifest.semantic_payload()["routing_evidence"] is None


def test_routing_evidence_record_changes_run_manifest_id() -> None:
    base = _manifest()
    evidence = RoutingEvidence(
        rule_id="auto.mito_hifi_lte_3x_pmat",
        taxon_group="plant",
        total_bases=3_000,
        genome_size_bp=1_000,
        threshold_multiplier=3,
        threshold_numerator=3_000,
        threshold_denominator=3_000,
        coverage_display="3.00x",
    )
    changed = base.model_copy(
        update={
            "route_reason_code": evidence.rule_id,
            "routing_evidence": evidence,
        }
    )
    assert changed.routing_evidence is not None
    assert changed.semantic_payload()["routing_evidence"] is not None
    assert base.semantic_payload()["routing_evidence"] is None
    assert changed.run_manifest_id != base.run_manifest_id


def test_routing_evidence_accession_changes_run_manifest_id() -> None:
    initial_evidence = RoutingEvidence(
        rule_id="route.explicit_method",
        taxon_group="plant",
        total_bases=3_000,
        genome_size_bp=1_000,
        coverage_display="3.00x",
    )
    base = _manifest().model_copy(update={"routing_evidence": initial_evidence})
    assert base.routing_evidence is not None
    changed = base.model_copy(
        update={
            "routing_evidence": base.routing_evidence.model_copy(
                update={"genome_size_accession": "GCF_000000000.1"}
            )
        }
    )
    assert changed.run_manifest_id != base.run_manifest_id


def test_manifest_rejects_mismatched_route_and_routing_evidence_rule() -> None:
    evidence = RoutingEvidence(
        rule_id="auto.hifi_oatk",
        taxon_group="plant",
    )
    with pytest.raises(ValidationError, match="routing evidence rule"):
        _manifest().model_copy(update={"routing_evidence": evidence})


@pytest.mark.parametrize(
    "update",
    [
        {"threshold_multiplier": 3},
        {
            "threshold_multiplier": 3,
            "threshold_numerator": 2_999,
            "threshold_denominator": 3_000,
        },
        {
            "threshold_multiplier": 3,
            "threshold_numerator": 3_000,
            "threshold_denominator": 2_999,
        },
        {"coverage_display": "approximately 3x"},
    ],
)
def test_routing_evidence_rejects_incoherent_derived_fields(
    update: dict[str, object],
) -> None:
    base: dict[str, object] = {
        "rule_id": "auto.mito_hifi_lte_3x_pmat",
        "taxon_group": "plant",
        "total_bases": 3_000,
        "genome_size_bp": 1_000,
        "coverage_display": "3.00x",
    }
    base.update(update)
    with pytest.raises(ValidationError):
        RoutingEvidence.model_validate(base)


def test_routing_evidence_accession_requires_a_genome_size() -> None:
    with pytest.raises(ValidationError):
        RoutingEvidence(
            rule_id="explicit.backend",
            taxon_group="plant",
            genome_size_accession="GCF_000000000.1",
        )


def test_explicit_requested_method_forbids_a_different_selected_backend() -> None:
    with pytest.raises(ValidationError, match="fallback"):
        _manifest().model_copy(update={"requested_method": "oatk", "selected_backend": "himt"})


def test_auto_requested_method_may_select_any_exact_assembly_backend() -> None:
    base = _manifest().model_copy(update={"requested_method": "auto"})
    manifest = base.model_copy(update={"requested_method": "auto", "selected_backend": "himt"})

    assert manifest.requested_method == "auto"
    assert manifest.selected_backend == "himt"
    assert manifest.run_manifest_id != base.run_manifest_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threads", 16),
        ("memory_gb", 64),
        ("timeout_seconds", 7200),
        ("environment_source", "existing"),
    ],
)
def test_every_manifest_parameter_fact_changes_run_id(field: str, value: object) -> None:
    base = _manifest()
    parameters = base.parameters.model_copy(update={field: value})
    changed = base.model_copy(update={"parameters": parameters})

    assert getattr(changed.parameters, field) != getattr(base.parameters, field)
    assert changed.run_manifest_id != base.run_manifest_id


def test_backend_parameter_fact_changes_run_manifest_id() -> None:
    base = _manifest()
    parameters = base.parameters.model_copy(
        update={
            "backend_parameters": {
                "backend": "oatk",
                "kmer_size": 1001,
                "minimum_kmer_coverage": 45,
            }
        }
    )
    changed = base.model_copy(update={"parameters": parameters})

    assert changed.run_manifest_id != base.run_manifest_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("carrier", "container"),
        ("digest", "sha256:" + "2" * 64),
        ("platform", "linux-aarch64"),
    ],
)
def test_every_manifest_environment_fact_changes_run_id(field: str, value: object) -> None:
    base = _manifest()
    environment = base.environment.model_copy(update={field: value})
    changed = base.model_copy(update={"environment": environment})

    assert getattr(changed.environment, field) != getattr(base.environment, field)
    assert changed.run_manifest_id != base.run_manifest_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "source"),
        ("name", "oatk-core"),
        ("version", "1.1"),
        ("sha256", "3" * 64),
        ("locator", "https://example.org/oatk/releases/v1"),
    ],
)
def test_every_manifest_component_fact_changes_run_id(field: str, value: object) -> None:
    base = _manifest()
    component = base.components[0].model_copy(update={field: value})
    changed = base.model_copy(update={"components": (component, base.components[1])})

    assert changed.run_manifest_id != base.run_manifest_id


def test_every_manifest_component_record_contributes_to_run_id() -> None:
    base = _manifest()
    database = base.components[1].model_copy(update={"sha256": "4" * 64})
    changed = base.model_copy(update={"components": (base.components[0], database)})

    assert changed.run_manifest_id != base.run_manifest_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "corrected_read"),
        ("format", "fq"),
        ("media_type", "application/octet-stream"),
        ("sha256", "5" * 64),
        ("size_bytes", 4096),
    ],
)
def test_every_manifest_input_artifact_fact_changes_run_id(field: str, value: object) -> None:
    base = _manifest()
    artifact = base.input_artifacts[0].artifact.model_copy(update={field: value})
    changed = base.model_copy(
        update={
            "input_artifacts": (ManifestArtifact(role="long_reads", artifact=artifact),),
        }
    )

    assert changed.run_manifest_id != base.run_manifest_id


def test_manifest_input_artifact_role_changes_run_id() -> None:
    base = _manifest()
    stable_argv = tuple(
        "role://artifact/corrected_reads" if token == "role://artifact/long_reads" else token
        for token in base.stable_argv
    )
    changed = base.model_copy(
        update={
            "input_artifacts": (
                ManifestArtifact(role="corrected_reads", artifact=base.input_artifacts[0].artifact),
            ),
            "stable_argv": stable_argv,
        }
    )

    assert changed.run_manifest_id != base.run_manifest_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "normalized_sequence"),
        ("format", "fna"),
        ("media_type", "application/octet-stream"),
        ("sha256", "6" * 64),
        ("size_bytes", 16384),
    ],
)
def test_every_manifest_output_artifact_fact_changes_run_id(field: str, value: object) -> None:
    base = _manifest()
    artifact = base.outputs[0].artifact.model_copy(update={field: value})
    changed = base.model_copy(
        update={
            "outputs": (
                ManifestArtifact(role="primary_fasta", artifact=artifact),
                base.outputs[1],
            )
        }
    )

    assert changed.run_manifest_id != base.run_manifest_id


def test_every_manifest_output_record_contributes_to_run_id() -> None:
    base = _manifest()
    graph = base.outputs[1].artifact.model_copy(update={"sha256": "7" * 64})
    changed = base.model_copy(
        update={
            "outputs": (
                base.outputs[0],
                ManifestArtifact(role="assembly_graph", artifact=graph),
            )
        }
    )

    assert changed.run_manifest_id != base.run_manifest_id


def test_manifest_output_role_changes_run_id_with_consistent_stage_links() -> None:
    base = _manifest()
    stage = base.stages[1].model_copy(
        update={"output_roles": ("primary_sequence", "assembly_graph")}
    )
    changed = base.model_copy(
        update={
            "stages": (base.stages[0], stage),
            "outputs": (
                ManifestArtifact(role="primary_sequence", artifact=base.outputs[0].artifact),
                base.outputs[1],
            ),
        }
    )

    assert changed.run_manifest_id != base.run_manifest_id


@pytest.mark.parametrize(
    "change",
    ["stage_name", "status", "output_order", "stage_order", "process_state", "exit_code"],
)
def test_every_manifest_stage_and_process_fact_changes_run_id(change: str) -> None:
    base = _manifest()
    if change == "stage_name":
        prepare = base.stages[0].model_copy(update={"stage": "preflight"})
        changed = base.model_copy(update={"stages": (prepare, base.stages[1])})
    elif change == "status":
        prepare = base.stages[0].model_copy(update={"status": "failed"})
        changed = base.model_copy(update={"stages": (prepare, base.stages[1])})
    elif change == "output_order":
        assemble = base.stages[1].model_copy(
            update={"output_roles": tuple(reversed(base.stages[1].output_roles))}
        )
        changed = base.model_copy(update={"stages": (base.stages[0], assemble)})
    elif change == "stage_order":
        changed = base.model_copy(update={"stages": tuple(reversed(base.stages))})
    elif change == "process_state":
        assemble = base.stages[1].model_copy(
            update={"process_started": False, "exit_code": None, "termination": None}
        )
        changed = base.model_copy(
            update={"stages": (base.stages[0], assemble), "process_exit_code": None}
        )
    else:
        assemble = base.stages[1].model_copy(update={"exit_code": 1})
        changed = base.model_copy(
            update={"stages": (base.stages[0], assemble), "process_exit_code": 1}
        )

    assert changed.run_manifest_id != base.run_manifest_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "organelleverse.assembly-run.v2"),
        ("operation_id", "assembly.other"),
        ("contract_version", "2.0"),
    ],
)
def test_manifest_constant_contract_facts_cannot_be_changed(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        _manifest().model_copy(update={field: value})


@pytest.mark.parametrize(
    "volatile",
    ["input_uri", "input_validated", "output_uri", "output_validated"],
)
def test_each_manifest_artifact_observation_is_independently_nonsemantic(volatile: str) -> None:
    base = _manifest()
    update: dict[str, object] = (
        {"uri": "/different-host/artifact"}
        if volatile.endswith("uri")
        else {"validated": volatile.startswith("input")}
    )
    if volatile.startswith("input"):
        artifact = base.input_artifacts[0].artifact.model_copy(update=update)
        changed = base.model_copy(
            update={"input_artifacts": (ManifestArtifact(role="long_reads", artifact=artifact),)}
        )
    else:
        artifact = base.outputs[0].artifact.model_copy(update=update)
        changed = base.model_copy(
            update={
                "outputs": (
                    ManifestArtifact(role="primary_fasta", artifact=artifact),
                    base.outputs[1],
                )
            }
        )

    assert changed.semantic_payload() == base.semantic_payload()
    assert changed.run_manifest_id == base.run_manifest_id


def test_nonsemantic_collection_order_does_not_change_manifest_id() -> None:
    base = _manifest()
    reordered = base.model_copy(
        update={
            "components": tuple(reversed(base.components)),
            "outputs": tuple(reversed(base.outputs)),
        }
    )

    assert reordered.semantic_payload() == base.semantic_payload()
    assert reordered.run_manifest_id == base.run_manifest_id


def test_input_artifact_tuple_order_does_not_change_manifest_id() -> None:
    base = _manifest()
    seed = ManifestArtifact(
        role="seed_fasta",
        artifact=_artifact(
            kind="sequence",
            uri="inputs/seed.fa",
            format="fasta",
            media_type="text/x-fasta",
            sha256="1" * 64,
            size_bytes=1024,
        ),
    )
    ordered = base.model_copy(update={"input_artifacts": (base.input_artifacts[0], seed)})
    reordered = ordered.model_copy(
        update={"input_artifacts": tuple(reversed(ordered.input_artifacts))}
    )

    assert reordered.input_artifacts != ordered.input_artifacts
    assert reordered.semantic_payload() == ordered.semantic_payload()
    assert reordered.run_manifest_id == ordered.run_manifest_id


def test_stable_argv_order_changes_manifest_id() -> None:
    base = _manifest()
    reordered_argv = (
        "oatk",
        "mito",
        "-o",
        "role://workspace/assembly",
        "-i",
        "role://artifact/long_reads",
    )
    reordered = base.model_copy(update={"stable_argv": reordered_argv})

    assert sorted(reordered.stable_argv) == sorted(base.stable_argv)
    assert reordered.run_manifest_id != base.run_manifest_id


def test_semantic_payload_is_stably_ordered_and_contains_only_content_identity() -> None:
    manifest = _manifest()

    payload = manifest.semantic_payload()
    inputs = cast(dict[str, dict[str, object]], payload["input_artifacts"])
    outputs = cast(dict[str, dict[str, object]], payload["outputs"])
    components = cast(list[dict[str, object]], payload["components"])

    assert tuple(inputs) == ("long_reads",)
    assert tuple(outputs) == ("assembly_graph", "primary_fasta")
    assert [(item["category"], item["name"]) for item in components] == [
        ("database", "oatkdb"),
        ("software", "oatk"),
    ]
    assert inputs["long_reads"] == {
        "kind": "long_read",
        "format": "fastq",
        "media_type": "application/x-fastq",
        "sha256": "a" * 64,
        "size_bytes": 2048,
    }
    assert payload["stable_argv"] == list(manifest.stable_argv)
    assert [item["stage"] for item in cast(list[dict[str, object]], payload["stages"])] == [
        "prepare",
        "assemble",
    ]
    assert "uri" not in json.dumps(payload)
    assert "validated" not in json.dumps(payload)
    assert manifest.canonical_bytes() == json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
@pytest.mark.parametrize(
    "semantic_field",
    ["stable_argv", "component_name", "component_version", "component_locator", "platform"],
)
def test_manifest_rejects_unencodable_semantic_text(
    semantic_field: str,
    surrogate: str,
) -> None:
    manifest = _manifest()
    with pytest.raises(ValidationError, match=r"canonical UTF-8 JSON|valid string|stable argv"):
        if semantic_field == "stable_argv":
            update: dict[str, object] = {"stable_argv": ("oatk", surrogate)}
        elif semantic_field == "platform":
            environment = manifest.environment.model_copy(update={"platform": surrogate})
            update = {"environment": environment}
        else:
            component_field = semantic_field.removeprefix("component_")
            component = manifest.components[0].model_copy(update={component_field: surrogate})
            update = {"components": (component, manifest.components[1])}
        manifest.model_copy(update=update)


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
@pytest.mark.parametrize("artifact_field", ["format", "media_type"])
def test_manifest_rejects_unencodable_nested_artifact_identity(
    artifact_field: str,
    surrogate: str,
) -> None:
    manifest = _manifest()
    with pytest.raises(ValidationError, match=r"canonical UTF-8 JSON|valid string"):
        artifact = manifest.input_artifacts[0].artifact.model_copy(
            update={artifact_field: surrogate}
        )
        manifest.model_copy(
            update={"input_artifacts": (ManifestArtifact(role="long_reads", artifact=artifact),)}
        )


def test_manifest_valid_unicode_semantic_identity_is_deterministic() -> None:
    manifest = _manifest()
    artifact = manifest.input_artifacts[0].artifact.model_copy(
        update={
            "format": "序列🧬",
            "media_type": "application/基因组+😀",
        }
    )
    component = manifest.components[0].model_copy(update={"name": "组装器🧬", "version": "版本😀"})
    environment = manifest.environment.model_copy(update={"platform": "平台-🐧"})

    first = manifest.model_copy(
        update={
            "input_artifacts": (ManifestArtifact(role="long_reads", artifact=artifact),),
            "components": (component, manifest.components[1]),
            "environment": environment,
            "stable_argv": ("组装器", "--模式=线粒体"),
        }
    )
    second = manifest.model_copy(
        update={
            "input_artifacts": (ManifestArtifact(role="long_reads", artifact=artifact),),
            "components": (component, manifest.components[1]),
            "environment": environment,
            "stable_argv": ("组装器", "--模式=线粒体"),
        }
    )

    assert first.semantic_payload() == second.semantic_payload()
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.run_manifest_id == second.run_manifest_id
    assert "组装器🧬" in first.canonical_bytes().decode("utf-8")


def test_surrogates_in_excluded_observation_fields_are_nonsemantic(
    tmp_path: Path,
) -> None:
    del tmp_path
    manifest = _manifest()
    observations = AssemblyRunObservations(
        run_manifest_id=manifest.run_manifest_id,
        workspace_path="host/work-\ud800",
        resolved_argv=("host/tool-\udfff",),
    )

    assert observations.workspace_path.endswith("\ud800")
    assert observations.resolved_argv == ("host/tool-\udfff",)
    assert "host/work" not in manifest.canonical_bytes().decode("utf-8")


def test_manifest_artifact_reconstructs_the_canonical_run_id() -> None:
    manifest = _manifest()

    artifact = manifest.as_artifact("run/assembly-run.json")

    assert artifact.kind == "assembly_run_manifest"
    assert artifact.format == "json"
    assert artifact.media_type == "application/json"
    assert artifact.sha256 == hashlib.sha256(manifest.canonical_bytes()).hexdigest()
    assert artifact.size_bytes == len(manifest.canonical_bytes())
    assert artifact.validated is True
    assert assembly_run_id_from_artifact(artifact) == manifest.run_manifest_id
    assert assembly_run_id_from_artifact(artifact) != artifact.object_id


def test_manifest_normal_json_round_trip_preserves_equality_hash_and_run_id() -> None:
    manifest = _manifest()

    restored = AssemblyRunManifest.model_validate_json(manifest.model_dump_json())

    assert restored == manifest
    assert hash(restored) == hash(manifest)
    assert restored.run_manifest_id == manifest.run_manifest_id
    assert restored.input_artifacts[0].artifact.object_id == (
        manifest.input_artifacts[0].artifact.object_id
    )


def test_manifest_json_round_trip_rejects_forged_nested_artifact_object_id() -> None:
    manifest = _manifest()
    serialized = json.loads(manifest.model_dump_json())
    serialized["input_artifacts"][0]["artifact"]["object_id"] = "long_read:sha256:" + "0" * 64

    with pytest.raises(ValidationError, match="object_id"):
        AssemblyRunManifest.model_validate_json(json.dumps(serialized))


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "run_manifest"},
        {"format": "yaml"},
        {"media_type": "text/plain"},
    ],
)
def test_run_id_reconstruction_rejects_nonassembly_manifest_artifacts(
    changes: dict[str, object],
) -> None:
    artifact = _manifest().as_artifact("run/assembly-run.json").model_copy(update=changes)

    with pytest.raises(ValueError, match="assembly run manifest"):
        assembly_run_id_from_artifact(artifact)


def test_manifest_evidence_graph_is_acyclic_and_excludes_volatile_facts(tmp_path: Path) -> None:
    request = _request(tmp_path, read_uri="/host/input/reads.fastq", output="host/output")
    manifest = _manifest(
        input_uri="/host/input/reads.fastq",
        output_uri="/host/output/primary.fa",
    )
    observations = _observations(manifest.run_manifest_id, workspace="/host/work")
    manifest_artifact = manifest.as_artifact("run/assembly-run.json")
    primary_fasta = manifest.outputs[0].artifact
    genome = OrganelleGenome(
        organelle="mitochondrion",
        sequence=primary_fasta,
        lineage=(
            LineageRecord(
                parent_object_ids=(request.data.object_id,),
                operation_id="assembly.assemble",
                operation_version="1.0",
                parameters_hash=request.semantic_hash,
            ),
        ),
        source_manifests=(manifest_artifact,),
    )
    result = OrganelleResult(
        operation_id="assembly.assemble",
        scope="mitochondrion",
        status="ok",
    )

    encoded = manifest.canonical_bytes().decode("utf-8")
    assert assembly_run_id_from_artifact(genome.source_manifests[0]) == manifest.run_manifest_id
    assert "genome" not in manifest.semantic_payload()
    for volatile in (
        "/host/input/reads.fastq",
        "/host/output/primary.fa",
        observations.started_at.isoformat() if observations.started_at is not None else "",
        observations.workspace_path,
        observations.resolved_argv[0],
        observations.stdout_path,
        observations.stderr_path,
        manifest.run_manifest_id,
        genome.object_id,
        result.object_id,
    ):
        assert volatile
        assert volatile not in encoded


@pytest.mark.parametrize(
    "locator",
    [
        "/opt/oatk",
        "./oatk",
        "../oatk",
        ".\\oatk",
        "..\\oatk",
        "~/oatk",
        "file:///opt/oatk",
        "FILE:C:\\Tools\\oatk.exe",
        "C:\\Tools\\oatk.exe",
        "C:/Tools/oatk.exe",
        "\\\\server\\share\\oatk.exe",
    ],
)
def test_component_identity_rejects_local_locators(locator: str) -> None:
    with pytest.raises(ValidationError, match="stable locator"):
        AssemblyComponentIdentity(category="software", name="oatk", locator=locator)


def test_component_identity_requires_one_stable_identifier() -> None:
    with pytest.raises(ValidationError, match="version, SHA256, or stable locator"):
        AssemblyComponentIdentity(category="software", name="oatk")


@pytest.mark.parametrize(
    "locator",
    [
        "https://example.org/oatk/releases/v1",
        "oci://registry.example.org/oatk@sha256:" + "a" * 64,
        "organelleverse.assembly.sources:oatk_v1",
    ],
)
def test_component_identity_allows_stable_nonlocal_locators(locator: str) -> None:
    component = AssemblyComponentIdentity(category="source", name="oatk", locator=locator)

    assert component.locator == locator


@pytest.mark.parametrize(
    "stable_argv",
    [
        (),
        ("",),
        ("/opt/oatk/bin/oatk",),
        ("oatk", "."),
        ("oatk", ".."),
        ("./oatk",),
        ("..\\oatk",),
        ("~/bin/oatk",),
        ("oatk", "-I/opt/oatkdb"),
        ("oatk", "-I:/opt/oatkdb"),
        ("oatk", "-I=/opt/oatkdb"),
        ("oatk", "-I./oatkdb"),
        ("oatk", "-I.\\oatkdb"),
        ("oatk", "-IC:\\oatkdb"),
        ("oatk", "-I:C:\\oatkdb"),
        ("oatk", "-I=C:\\oatkdb"),
        ("oatk", "-Ifile:///host/oatkdb"),
        ("oatk", "-I:file:///host/oatkdb"),
        ("oatk", "-I=file:///host/oatkdb"),
        ("oatk", "--input:/opt/oatkdb"),
        ("oatk", "--input:C:\\oatkdb"),
        ("oatk", "--input:file:///host/oatkdb"),
        ("oatk", "--input=/opt/oatkdb"),
        ("oatk", "--input=C:\\oatkdb"),
        ("oatk", "--input=file:///host/reads.fastq"),
        ("oatk", "--input=.\\host\\reads.fastq"),
        ("oatk", "C:\\host\\reads.fastq"),
        ("oatk", "\\\\server\\share\\reads.fastq"),
        ("oatk", "role://artifact/Bad-Role"),
        ("oatk", "-Irole://artifact/Bad-Role"),
        ("oatk", "role://artifact/reads/../../host"),
        ("oatk", "role://workspace/output:/opt/output"),
        ("oatk", "--input:role://artifact/reads/../../host"),
        ("oatk", "-Irole://workspace/output\\host"),
        ("oatk", "-Irole://artifact/oatkdb"),
        ("oatk", "--output=role://workspace/assembly"),
        ("oatk", "ratio=1/2"),
        ("oatk", "reads.fastq"),
        ("oatk", "--input=reads.fastq"),
        ("oatk", "reads%2Fhost.fastq"),
        ("oatk", "--input=reads%2Efastq"),
        ("oatk", "sample name"),
        ("oatk", "ratio=0.5"),
        ("oatk", "clock=1230"),
        ("oatk", "prefix=cache-db"),
        ("oatk", "role://artifact/missing"),
        ("oatk", "role://workspace/scratch"),
    ],
)
def test_manifest_rejects_empty_or_path_bearing_stable_argv(
    stable_argv: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError, match="stable argv"):
        _manifest().model_copy(update={"stable_argv": stable_argv})


@pytest.mark.parametrize(
    "token",
    [
        "-out=/opt/db",
        "-out:/opt/db",
        "-out/opt/db",
        "-DROOT=C:\\db",
        "-DROOT:C:\\db",
        "-DROOTC:\\db",
        "-out=file:///db",
        "-out:file:///db",
        "-outfile:///db",
        "-out=file:db",
        "-out=./db",
        "-out:../db",
        "-out.\\db",
        "-out=~/db",
        "-out:~\\db",
        "-out~/db",
        "-out=~",
        "-out=role://artifact/Bad-Role",
        "-out:role://artifact/Bad-Role",
        "-outrole://artifact/Bad-Role",
        "-profile=https://example.org/profile?root=/reference",
        "--profile=https://example.org/profile?root=/reference",
    ],
)
def test_manifest_rejects_path_syntax_in_single_dash_multichar_tokens(token: str) -> None:
    with pytest.raises(ValidationError, match="stable argv"):
        _manifest().model_copy(update={"stable_argv": ("oatk", token)})


@pytest.mark.parametrize(
    "token",
    [
        "C:db",
        "-I=C:db",
        "--input:C:db",
        "-DROOT=C:db",
        "prefix=C:db",
        "-IC:db",
        "-Ic:db",
        "-DROOTC:db",
        "--inputC:db",
    ],
)
def test_manifest_rejects_windows_drive_relative_stable_tokens(token: str) -> None:
    with pytest.raises(ValidationError, match="stable argv"):
        _manifest().model_copy(update={"stable_argv": ("oatk", token)})


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/profile?root=/reference",
        "https://example.org/a%20b?q=x%2Fy",
        "http://example.org:8080/a?b=c",
        "http://127.0.0.1:8080/a",
        "https://[2001:db8::1]/a?b=c#d",
        "HTTPS://example.org/path",
        "https://C:\\db",
        "https://example.org\\..\\host",
        "https://:",
        "https:///missing-host",
        "https://example.org:bad/path",
        "https://example.org:70000/path",
        "https://exa mple.org/path",
        "https://exa%mple.org/path",
        "https://exa%6Dple.org/path",
        "https://example.org/%ZZ",
        "https://example.org/?q=%G0",
        "https://example.org/#fragment%Q1",
        "https://_bad.example/path",
        "https://-bad.example/path",
        "https://bad-.example/path",
        "https://bad..example/path",
        "https://999.999.999.999/path",
        "https://[not-ip]/path",
    ],
)
def test_manifest_rejects_all_http_stable_urls(url: str) -> None:
    with pytest.raises(ValidationError, match="stable argv"):
        _manifest().model_copy(update={"stable_argv": ("oatk", url)})


@pytest.mark.parametrize(
    "token",
    [
        "clock=12:30",
        "prefix=cache:db",
        "namespace:value",
    ],
)
def test_manifest_rejects_colon_bearing_non_uri_literals(token: str) -> None:
    with pytest.raises(ValidationError, match="stable argv"):
        _manifest().model_copy(update={"stable_argv": ("oatk", token)})


@pytest.mark.parametrize(
    "token",
    [
        "\x00",
        "flag\nvalue",
        "flag\tvalue",
        "flag\x1fvalue",
        "flag\x7fvalue",
        "flag\x85value",
    ],
)
def test_manifest_rejects_control_characters_in_stable_tokens(token: str) -> None:
    with pytest.raises(ValidationError, match="stable argv"):
        _manifest().model_copy(update={"stable_argv": ("oatk", token)})


@pytest.mark.parametrize(
    "token",
    [
        "oatk",
        "-t8",
        "--mode=mito",
        "--threads=16",
        "--ratio=0.5",
        "--profile=cache-db",
        "8",
        "0.5",
        "-0.5",
        "+2",
        "mitochondrion",
        "线粒体",
        "role://artifact/long_reads",
        "role://artifact/primary_fasta",
        "role://workspace/assembly",
        "role://workspace/cache",
        "role://workspace/output",
        "role://workspace/temp",
    ],
)
def test_manifest_allows_roles_and_canonical_option_literals(token: str) -> None:
    manifest = _manifest().model_copy(update={"stable_argv": ("oatk", token)})

    assert manifest.stable_argv == ("oatk", token)


@pytest.mark.parametrize(
    "executable",
    ["oatk", "get_organelle_from_reads.py", "工具", "assembler-2.1+cpu"],
)
def test_manifest_allows_path_free_executable_basenames(executable: str) -> None:
    manifest = _manifest().model_copy(update={"stable_argv": (executable,)})

    assert manifest.stable_argv == (executable,)


@pytest.mark.parametrize(
    "executable",
    [".", "..", ".hidden", "tool.", "tool..bin", "tool%2Ebin", "tool:bin", "tool~bin"],
)
def test_manifest_rejects_noncanonical_executable_basenames(executable: str) -> None:
    with pytest.raises(ValidationError, match="stable argv"):
        _manifest().model_copy(update={"stable_argv": (executable,)})


@pytest.mark.parametrize(
    "change",
    ["input_roles", "output_roles", "components", "stages", "stage_outputs", "unknown_output"],
)
def test_manifest_rejects_duplicate_or_dangling_semantic_roles(change: str) -> None:
    manifest = _manifest()
    updates: dict[str, object]
    if change == "input_roles":
        updates = {"input_artifacts": (manifest.input_artifacts[0],) * 2}
    elif change == "output_roles":
        updates = {"outputs": (manifest.outputs[0],) * 2}
    elif change == "components":
        updates = {"components": (manifest.components[0],) * 2}
    elif change == "stages":
        updates = {"stages": (manifest.stages[0],) * 2}
    elif change == "stage_outputs":
        stage = manifest.stages[1].model_copy(
            update={"output_roles": ("primary_fasta", "primary_fasta")}
        )
        updates = {"stages": (manifest.stages[0], stage)}
    else:
        stage = manifest.stages[1].model_copy(update={"output_roles": ("missing_output",)})
        updates = {"stages": (manifest.stages[0], stage)}

    with pytest.raises(ValidationError):
        manifest.model_copy(update=updates)


@pytest.mark.parametrize(
    "values",
    [
        {"stage": "assemble", "status": "ok", "process_started": True},
        {"stage": "prepare", "status": "ok", "exit_code": 0},
        {
            "stage": "skipped_stage",
            "status": "skipped",
            "process_started": True,
            "exit_code": 0,
        },
    ],
)
def test_stage_process_state_and_exit_code_must_agree(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match=r"exit code|skipped|termination"):
        AssemblyStageOutcome.model_validate(values)


def test_started_timeout_stage_can_have_no_exit_code() -> None:
    outcome = AssemblyStageOutcome(
        stage="execute_backend",
        status="failed",
        process_started=True,
        termination="timeout",
        exit_code=None,
    )
    assert outcome.termination == "timeout"


def test_started_signal_stage_records_a_negative_exit_code() -> None:
    outcome = AssemblyStageOutcome(
        stage="execute_backend",
        status="failed",
        process_started=True,
        termination="signal",
        exit_code=-9,
    )
    assert outcome.termination == "signal"
    assert outcome.exit_code == -9


@pytest.mark.parametrize(
    "values",
    [
        {
            "stage": "execute_backend",
            "status": "failed",
            "process_started": True,
            "termination": "exit",
            "exit_code": None,
        },
        {
            "stage": "execute_backend",
            "status": "failed",
            "process_started": False,
            "termination": "exit",
            "exit_code": 1,
        },
        {
            "stage": "prepare",
            "status": "ok",
            "process_started": False,
            "termination": "timeout",
        },
    ],
)
def test_stage_termination_must_be_consistent_with_process_state(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match=r"termination|exit code"):
        AssemblyStageOutcome.model_validate(values)


def test_manifest_top_level_exit_code_matches_the_final_started_stage() -> None:
    manifest = _manifest()

    with pytest.raises(ValidationError, match="process exit code"):
        manifest.model_copy(update={"process_exit_code": None})
    with pytest.raises(ValidationError, match="process exit code"):
        manifest.model_copy(update={"process_exit_code": 1})

    unstarted = AssemblyStageOutcome(stage="prepare", status="ok")
    with pytest.raises(ValidationError, match="process exit code"):
        manifest.model_copy(update={"stages": (unstarted,), "process_exit_code": 0})


def test_observation_timestamps_require_matching_awareness_and_monotonic_order() -> None:
    run_id = _manifest().run_manifest_id
    aware = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    naive = aware.replace(tzinfo=None)

    with pytest.raises(ValidationError, match="matching timezone awareness"):
        AssemblyRunObservations(
            run_manifest_id=run_id,
            started_at=aware,
            finished_at=naive,
        )
    with pytest.raises(ValidationError, match="earlier than started_at"):
        AssemblyRunObservations(
            run_manifest_id=run_id,
            started_at=aware,
            finished_at=aware - timedelta(seconds=1),
        )


@pytest.mark.parametrize("field", ["duration_seconds", "peak_cpu_percent"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_observations_reject_nonfinite_resource_metrics(field: str, value: float) -> None:
    values: dict[str, object] = {"run_manifest_id": _manifest().run_manifest_id, field: value}

    with pytest.raises(ValidationError):
        AssemblyRunObservations.model_validate(values)


def test_observations_round_trip_finite_metrics_as_valid_json() -> None:
    observations = _observations(_manifest().run_manifest_id).model_copy(
        update={"duration_seconds": 12.125, "peak_cpu_percent": 98.25}
    )

    encoded = observations.model_dump_json()
    decoded = json.loads(encoded)

    assert decoded["duration_seconds"] == 12.125
    assert decoded["peak_cpu_percent"] == 98.25
    assert json.dumps(decoded, allow_nan=False)
    assert AssemblyRunObservations.model_validate_json(encoded) == observations


def test_manifest_contracts_are_frozen_strict_and_observations_have_no_identity() -> None:
    manifest = _manifest()
    observations = _observations(manifest.run_manifest_id)
    models: tuple[BaseModel, ...] = (
        manifest.input_artifacts[0],
        manifest.environment,
        manifest.components[0],
        manifest.stages[0],
        manifest.parameters,
        manifest,
        observations,
    )

    for model in models:
        assert model.model_config.get("frozen") is True
        assert model.model_config.get("extra") == "forbid"
    assert not hasattr(observations, "object_id")

    with pytest.raises(ValidationError) as extra:
        AssemblyRunObservations.model_validate(
            {"run_manifest_id": manifest.run_manifest_id, "unexpected": True}
        )
    assert extra.value.errors()[0]["type"] == "extra_forbidden"
    with pytest.raises(ValidationError) as frozen:
        observations.workspace_path = "/other/work"
    assert frozen.value.errors()[0]["type"] == "frozen_instance"


def test_manifest_primary_sequence_role_defaults_to_assembly_fasta() -> None:
    manifest = _manifest()
    assert manifest.primary_sequence_role == "assembly_fasta"


def test_manifest_primary_sequence_role_round_trips() -> None:
    manifest = _manifest().model_copy(update={"primary_sequence_role": "primary_fasta"})
    assert manifest.primary_sequence_role == "primary_fasta"
    restored = AssemblyRunManifest.model_validate_json(manifest.model_dump_json())
    assert restored.primary_sequence_role == "primary_fasta"


def _input_payload() -> AssemblyInputPayload:
    return AssemblyInputPayload(
        short_libraries=(
            ShortLibraryContract(
                technology="illumina",
                layout="paired_end",
                read1_artifact="pe_reads_r1",
                read2_artifact="pe_reads_r2",
                read_length=150,
                insert_size=300,
            ),
        ),
        long_libraries=(
            LongLibraryContract(
                technology="pacbio_hifi",
                quality_state="ccs",
                reads_artifact="long_reads",
            ),
        ),
    )


def test_manifest_input_payload_is_optional_and_absent_by_default() -> None:
    manifest = _manifest()
    assert manifest.input_payload is None
    # Backward compatibility: an absent payload adds no key to canonical identity.
    assert "input_payload" not in manifest.semantic_payload()
    assert "input_payload" not in manifest.canonical_bytes().decode("utf-8")


def test_manifest_input_payload_is_recorded_in_semantic_identity_when_present() -> None:
    base = _manifest()
    manifest = base.model_copy(update={"input_payload": _input_payload()})
    assert manifest.input_payload is not None
    assert "input_payload" in manifest.semantic_payload()
    assert manifest.semantic_payload()["input_payload"] is not None
    # Presence changes the canonical run id.
    assert manifest.run_manifest_id != base.run_manifest_id


def test_manifest_input_payload_round_trips() -> None:
    manifest = _manifest().model_copy(update={"input_payload": _input_payload()})
    restored = AssemblyRunManifest.model_validate_json(manifest.model_dump_json())
    assert restored.input_payload == manifest.input_payload
    assert restored.run_manifest_id == manifest.run_manifest_id


def test_manifest_input_payload_rejects_free_form_mapping() -> None:
    with pytest.raises(ValidationError):
        _manifest().model_copy(update={"input_payload": {"not": "a typed payload"}})


def test_manifest_input_payload_preserves_short_pe_and_long_library_details() -> None:
    manifest = _manifest().model_copy(update={"input_payload": _input_payload()})
    payload = manifest.input_payload
    assert payload is not None
    (short_library,) = payload.short_libraries
    assert short_library.technology == "illumina"
    assert short_library.layout == "paired_end"
    assert short_library.read1_artifact == "pe_reads_r1"
    assert short_library.read2_artifact == "pe_reads_r2"
    assert short_library.read_length == 150
    assert short_library.insert_size == 300
    (long_library,) = payload.long_libraries
    assert long_library.technology == "pacbio_hifi"
    assert long_library.quality_state == "ccs"
    assert long_library.reads_artifact == "long_reads"
