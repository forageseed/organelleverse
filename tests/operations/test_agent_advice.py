"""Real manifest conversion and JSON chaining with live readiness checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.core.serialization import save_contract
from organelleverse.operations import (
    CoreKind,
    DependencyKind,
    DependencySpec,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
    SideEffect,
)
from organelleverse.operations.adapters import invoke_json, invoke_next_json


def _spec(operation_id: str, **kwargs: object) -> OperationSpec:
    return OperationSpec(
        operation_id=operation_id,
        contract_version="1.0",
        title="Advice test",
        description="Exercise live operation advice and real artifact conversion.",
        keywords=("advice", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.ANALYZE,
        output_kind=CoreKind.RESULT,
        **kwargs,
    )


@pytest.fixture
def source(tmp_path: Path) -> OrganelleResult:
    sequence = tmp_path / "assembly.fasta"
    sequence.write_text(">molecule circular=true\nACGTACGT\n", encoding="utf-8")
    genome = OrganelleGenome(
        organelle="plastid",
        sequence=ArtifactRef.from_path(sequence, kind="sequence", format="fasta"),
    )
    manifest = save_contract(genome, tmp_path / "primary_genome.json")
    return OrganelleResult(
        operation_id="assembly.assemble",
        scope="plastid",
        status="ok",
        summary_text="Assembled one sequence.",
        artifacts=(ArtifactRef.from_path(manifest, kind="primary_genome_manifest", format="json"),),
    )


def _registry(
    source: OrganelleResult, *, required: bool = False, **extra: object
) -> OperationRegistry:
    registry = OperationRegistry()

    def assemble() -> OrganelleResult:
        return source

    def annotate(genome: OrganelleGenome, *, reference: str = "default") -> OrganelleResult:
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="ok",
            metrics={
                "species": genome.metadata.species,
                "reference": reference,
                "sequence_uri": genome.sequence.uri,
            },
        )

    def required_annotate(genome: OrganelleGenome, *, reference: str) -> OrganelleResult:
        return annotate(genome, reference=reference)

    registry.register(_spec("assembly.assemble", input_kind=CoreKind.NONE), assemble)
    registry.register(
        _spec(
            "annotation.annotate",
            input_kind=CoreKind.GENOME,
            organelle_types=("plastid",),
            result_artifact_inputs=("primary_genome_manifest",),
            side_effects=(SideEffect.READ_FILES,),
            **extra,
        ),
        required_annotate if required else annotate,
    )
    return registry


def _candidate(registry: OperationRegistry, source: OrganelleResult, **kwargs: object):
    return registry.agent_advice(source, **kwargs).next_operations[0]


def test_missing_metadata_explains_conversion_without_claiming_readiness(
    source: OrganelleResult,
) -> None:
    candidate = _candidate(_registry(source), source, granted_side_effects=("read_files",))
    assert not candidate.ready
    assert candidate.missing_metadata == ("species",)
    assert candidate.input_conversion.callable == "ov.io.read_assembly"
    assert candidate.input_conversion.artifact_id == source.artifacts[0].object_id
    assert candidate.input_conversion.artifact_uri == source.artifacts[0].uri


def test_complete_conditions_allow_real_json_chain(source: OrganelleResult) -> None:
    registry = _registry(source, required=True)
    candidate = _candidate(
        registry,
        source,
        metadata={"species": "Arabidopsis thaliana"},
        parameters={"annotation.annotate": {"reference": "curated"}},
        granted_side_effects=("read_files",),
    )
    assert candidate.ready
    response = invoke_next_json(
        {
            "operation_id": "annotation.annotate",
            "result": source.model_dump(mode="json"),
            "metadata": {"species": "Arabidopsis thaliana"},
            "parameters": {"reference": "curated"},
        },
        registry=registry,
        granted_side_effects={SideEffect.READ_FILES},
    )
    assert response["ok"] is True
    assert response["result"]["metrics"]["species"] == "Arabidopsis thaliana"
    assert response["result"]["metrics"]["reference"] == "curated"
    assert response["result"]["metrics"]["sequence_uri"].endswith("assembly.fasta")


def test_missing_target_parameters_do_not_break_source_success(source: OrganelleResult) -> None:
    registry = _registry(source, required=True)
    response = invoke_json(
        {"operation_id": "assembly.assemble", "input": None, "parameters": {}},
        registry=registry,
        granted_side_effects={SideEffect.READ_FILES},
    )
    assert response["ok"] is True
    assert response["result"] == source.model_dump(mode="json")
    candidate = response["agent_advice"]["next_operations"][0]
    assert candidate["missing_parameters"] == ["reference"]
    assert not candidate["ready"]


def test_catalog_changes_only_advice_not_result_identity(source: OrganelleResult) -> None:
    registry = OperationRegistry()
    receipt = source.model_dump(mode="json")
    assert registry.agent_advice(source).next_operations == ()
    registry = _registry(source)
    assert len(registry.agent_advice(source).next_operations) == 1
    assert source.model_dump(mode="json") == receipt
    assert (
        registry.invoke("assembly.assemble", input=None, parameters={}).object_id
        == receipt["object_id"]
    )


def test_failed_and_ambiguous_sources_are_not_ready(source: OrganelleResult) -> None:
    failed = source.evolve(
        status="failed", errors=(ErrorDetail(code="assembly.failed", message="failed"),)
    )
    duplicate = source.evolve(artifacts=source.artifacts + source.artifacts)
    for result, code in [(failed, "source.failed"), (duplicate, "input.ambiguous_artifact")]:
        candidate = _candidate(
            _registry(result),
            result,
            metadata={"species": "test"},
            granted_side_effects=("read_files",),
        )
        assert not candidate.ready
        assert code in {issue.code for issue in candidate.issues}


def test_missing_dependency_and_invalid_parameters_are_explicit(source: OrganelleResult) -> None:
    registry = _registry(
        source,
        dependencies=(
            DependencySpec(kind=DependencyKind.EXECUTABLE, name="ov-advice-nonexistent-backend"),
        ),
    )
    candidate = _candidate(
        registry,
        source,
        metadata={"species": "test"},
        parameters={"annotation.annotate": {"unknown": 1}},
        granted_side_effects=("read_files",),
    )
    assert not candidate.ready
    assert not candidate.dependencies.ready
    assert {"dependency.not_ready", "parameter.invalid"} <= {
        issue.code for issue in candidate.issues
    }


def test_no_read_grant_never_opens_source_paths(source: OrganelleResult, monkeypatch) -> None:
    import organelleverse.operations.advice as advice

    def fail_if_read(*args):
        raise AssertionError("read without permission")

    monkeypatch.setattr(advice, "_validate_assembly_input", fail_if_read)
    candidate = _candidate(_registry(source), source, metadata={"species": "test"})
    assert not candidate.ready
    assert candidate.missing_side_effects == (SideEffect.READ_FILES,)


@pytest.mark.parametrize("mutation", ["manifest_changed", "sequence_changed", "missing_file"])
def test_artifact_damage_blocks_execution_but_preserves_source(
    source: OrganelleResult, mutation: str
) -> None:
    manifest = source.artifacts[0].resolve()
    if mutation == "manifest_changed":
        manifest.write_text("{}", encoding="utf-8")
    elif mutation == "sequence_changed":
        (manifest.parent / "assembly.fasta").write_text(">changed\nAAAA\n", encoding="utf-8")
    else:
        manifest.unlink()
    registry = _registry(source)
    response = invoke_json(
        {
            "operation_id": "assembly.assemble",
            "input": None,
            "parameters": {},
            "advice_context": {"species": "test"},
        },
        registry=registry,
        granted_side_effects={SideEffect.READ_FILES},
    )
    assert response["ok"] is True
    assert response["result"]["object_id"] == source.object_id
    assert not response["agent_advice"]["next_operations"][0]["ready"]
    next_response = invoke_next_json(
        {
            "operation_id": "annotation.annotate",
            "result": source.model_dump(mode="json"),
            "metadata": {"species": "test"},
        },
        registry=registry,
        granted_side_effects={SideEffect.READ_FILES},
    )
    assert next_response["ok"] is False
    assert next_response["error"]["error_code"] == "input.next_operation_not_ready"


def test_explanations_keep_biological_uncertainty(source: OrganelleResult) -> None:
    source = source.evolve(flags=("missing_core_genes", "sequence_only_orf_candidates"))
    limitations = _registry(source).agent_advice(source).limitations
    assert "does not establish gene loss" in limitations[0].message
    assert "do not establish expression" in limitations[1].message


def test_stale_ready_advice_is_rechecked_before_execution(source: OrganelleResult) -> None:
    registry = _registry(source)
    assert _candidate(
        registry, source, metadata={"species": "test"}, granted_side_effects=("read_files",)
    ).ready
    source.artifacts[0].resolve().unlink()
    response = invoke_next_json(
        {
            "operation_id": "annotation.annotate",
            "result": source.model_dump(mode="json"),
            "metadata": {"species": "test"},
        },
        registry=registry,
        granted_side_effects={SideEffect.READ_FILES},
    )
    assert not response["ok"]


def test_non_json_parameter_is_reported_as_unready(source: OrganelleResult) -> None:
    candidate = _candidate(
        _registry(source),
        source,
        metadata={"species": "test"},
        parameters={"annotation.annotate": {"reference": object()}},
        granted_side_effects=("read_files",),
    )
    assert not candidate.ready
    assert "parameter.invalid" in {issue.code for issue in candidate.issues}


def test_real_release_catalog_exposes_annotation_conversion(source: OrganelleResult) -> None:
    from organelleverse.operations.catalog import load_release_catalog

    registry = OperationRegistry()
    load_release_catalog(registry)
    candidate = _candidate(
        registry,
        source,
        metadata={"species": "Arabidopsis thaliana"},
        granted_side_effects=("read_files", "write_files", "subprocess"),
    )
    assert candidate.operation_id == "annotation.annotate"
    assert candidate.input_conversion.callable == "ov.io.read_assembly"
    assert candidate.parameters["threads"] == 4
    assert candidate.dependencies.operation_id == "annotation.annotate"


def test_metadata_schema_error_does_not_claim_readiness(source: OrganelleResult) -> None:
    candidate = _candidate(
        _registry(source),
        source,
        metadata={"species": "test", "genetic_code": "invalid"},
        granted_side_effects=("read_files",),
    )
    assert not candidate.ready
    assert "input.metadata_invalid" in {issue.code for issue in candidate.issues}


def test_species_only_conversion_preserves_published_metadata(source: OrganelleResult) -> None:
    from organelleverse.core.serialization import load_genome
    from organelleverse.operations.advice import prepare_next_input

    manifest = source.artifacts[0].resolve()
    genome = load_genome(manifest)
    save_contract(
        genome.evolve(metadata=genome.metadata.evolve(accession="published-accession")), manifest
    )
    result = source.evolve(
        artifacts=(ArtifactRef.from_path(manifest, kind="primary_genome_manifest", format="json"),)
    )
    prepared = prepare_next_input(
        _registry(result),
        result,
        "annotation.annotate",
        metadata={"species": "Arabidopsis thaliana"},
        parameters={},
        granted_side_effects=("read_files",),
    )
    assert prepared.metadata.accession == "published-accession"
    assert prepared.metadata.species == "Arabidopsis thaliana"
    assert prepared.metadata.genetic_code == 11
