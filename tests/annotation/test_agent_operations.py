from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from organelleverse import operations
from organelleverse.annotation import api
from organelleverse.annotation.backends.base import AnnotationRequest, BackendRun
from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.annotation.service import run_annotation
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata, OrganelleType
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import OperationRegistry, SideEffect
from organelleverse.operations.adapters import invoke_json
from organelleverse.operations.dependencies import DependencyReport
from organelleverse.runtime import managed_runs_root


def _genome(tmp_path: Path) -> OrganelleGenome:
    fasta = tmp_path / "genome.fasta"
    fasta.write_text(">r1\nATGAAATAG\n")
    return OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
        metadata=OrganelleMetadata(species="Arabidopsis thaliana"),
    )


def _document(request: AnnotationRequest) -> AnnotationDocument:
    return AnnotationDocument(
        backend="mitochondrion",
        requested_stages=request.stages,
        completed_stages=request.stages,
        records=(
            AnnotationRecord(
                seqid="r1",
                name="r1",
                description="Agent fixture",
                sequence="ATGAAATAG",
                features=(
                    AnnotationFeature(
                        feature_id="r1:0:CDS",
                        seqid="r1",
                        type="CDS",
                        operator="single",
                        parts=(LocationPart(start=0, end=9, strand=1),),
                        qualifiers=(
                            FeatureQualifier(name="gene", values=("nad1",)),
                            FeatureQualifier(name="translation", values=("MK",)),
                        ),
                        parents=(),
                    ),
                ),
            ),
        ),
        source_metadata=FrozenMap(),
    )


@dataclass
class FakeBackend:
    name: str = "mitochondrion"
    organelle_types: tuple[OrganelleType, ...] = ("mitochondrion",)

    def run(
        self,
        genome: OrganelleGenome,
        request: AnnotationRequest,
        scratch: Path,
    ) -> BackendRun:
        del genome
        log = scratch / "backend.log"
        log.write_text("complete\n")
        return BackendRun(
            document=_document(request),
            commands=(),
            software_versions={"fixture": "1.0"},
            database_hashes={"fixture": "a" * 64},
            logs=(log,),
        )


def _request(genome: OrganelleGenome) -> dict[str, object]:
    return {
        "operation_id": "annotation.annotate",
        "input": genome.model_dump(mode="json"),
        "parameters": {
            "backend": "mitochondrion",
            "threads": 1,
            "call_trna": False,
            "call_rrna": False,
        },
    }


def _ready_dependencies(
    _registry: OperationRegistry,
    operation_id: str,
) -> DependencyReport:
    return DependencyReport(operation_id=operation_id, checks=(), ready=True)


def _error(response: dict[str, object]) -> dict[str, object]:
    value = response["error"]
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _result(response: dict[str, object]) -> dict[str, object]:
    value = response["result"]
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_default_catalog_contains_only_released_annotation_backends() -> None:
    specs = {spec.operation_id: spec for spec in operations.list()}

    assert set(specs) >= {
        "annotation.annotate",
        "annotation.extract",
        "annotation.write",
    }
    schema = operations.parameter_schema("annotation.annotate")
    properties = cast(dict[str, dict[str, object]], schema["properties"])
    assert properties["backend"]["enum"] == ["auto", "mitochondrion", "plastome"]
    encoded = json.dumps(schema)
    assert "native" not in encoded
    assert specs["annotation.annotate"].organelle_types == ("mitochondrion", "plastid")
    executable_dependencies = {
        dependency.name: dependency
        for dependency in specs["annotation.annotate"].dependencies
        if dependency.kind.value == "executable"
    }
    assert set(executable_dependencies) == {
        "blastn",
        "makeblastdb",
        "tblastn",
    }
    assert specs["annotation.annotate"].side_effects == (
        SideEffect.READ_FILES,
        SideEffect.WRITE_FILES,
        SideEffect.SUBPROCESS,
    )
    assert specs["annotation.extract"].side_effects == (
        SideEffect.READ_FILES,
        SideEffect.WRITE_FILES,
    )
    assert specs["annotation.write"].side_effects == (
        SideEffect.READ_FILES,
        SideEffect.WRITE_FILES,
    )


def test_direct_registry_and_agent_invocation_share_real_service_result_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genome = _genome(tmp_path)
    adapter = FakeBackend()

    def run_with_fixture(
        candidate: OrganelleGenome,
        request: AnnotationRequest,
    ) -> OrganelleResult:
        return run_annotation(
            candidate,
            request,
            adapters={"mitochondrion": adapter},
        )

    monkeypatch.setattr(api, "run_annotation", run_with_fixture)
    monkeypatch.setattr(OperationRegistry, "check_dependencies", _ready_dependencies)
    parameters: dict[str, object] = {
        "backend": "mitochondrion",
        "threads": 1,
        "call_trna": False,
        "call_rrna": False,
    }

    direct = api.annotate(
        genome,
        backend="mitochondrion",
        threads=1,
        call_trna=False,
        call_rrna=False,
    )
    registered = operations.invoke(
        "annotation.annotate",
        input=genome,
        parameters=parameters,
    )
    response = invoke_json(
        _request(genome),
        registry=operations.registry,
        granted_side_effects={"read_files", "write_files", "subprocess"},
    )

    assert response["ok"] is True
    assert isinstance(registered, OrganelleResult)
    assert registered.object_id == direct.object_id
    assert _result(response)["object_id"] == direct.object_id
    assert _result(response)["status"] == "ok"
    assert "annotation_planned" not in cast(list[str], _result(response)["flags"])


def test_released_annotate_publishes_into_the_managed_run_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genome = _genome(tmp_path)
    adapter = FakeBackend()

    def run_with_fixture(
        candidate: OrganelleGenome,
        request: AnnotationRequest,
    ) -> OrganelleResult:
        return run_annotation(
            candidate,
            request,
            adapters={"mitochondrion": adapter},
        )

    monkeypatch.setattr(api, "run_annotation", run_with_fixture)

    first = api.annotate(
        genome, backend="mitochondrion", threads=1, call_trna=False, call_rrna=False
    )
    second = api.annotate(
        genome, backend="mitochondrion", threads=1, call_trna=False, call_rrna=False
    )

    assert first.status == second.status == "ok"
    assert all(Path(item.uri).is_relative_to(managed_runs_root()) for item in first.artifacts)
    assert all(item.resolve().is_file() for item in first.artifacts)
    # Independent calls resolve to one content-addressed managed run: the
    # canonical annotation artifact URI is identical across both calls.
    first_annotation = next(item for item in first.artifacts if item.kind == "annotation")
    second_annotation = next(item for item in second.artifacts if item.kind == "annotation")
    assert first_annotation.uri == second_annotation.uri
    assert len(list((managed_runs_root() / "annotation.annotate").glob("sha256-*"))) == 1


def test_agent_denies_ungranted_side_effects_before_execution(tmp_path: Path) -> None:
    response = invoke_json(
        _request(_genome(tmp_path)),
        registry=operations.registry,
        granted_side_effects=set(),
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "permission.denied"
    details = cast(dict[str, object], _error(response)["details"])
    assert details["missing_side_effects"] == ["read_files", "subprocess", "write_files"]
    _annotation_root = managed_runs_root() / "annotation.annotate"
    assert not _annotation_root.exists() or not any(_annotation_root.glob("sha256-*"))


def test_agent_rejects_unknown_annotation_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OperationRegistry, "check_dependencies", _ready_dependencies)
    request = _request(_genome(tmp_path))
    parameters = cast(dict[str, object], request["parameters"])
    parameters["research_backend"] = "plastome"

    response = invoke_json(
        request,
        registry=operations.registry,
        granted_side_effects={"read_files", "write_files", "subprocess"},
    )

    assert response["ok"] is False
    assert _error(response)["error_code"] == "parameter.invalid_operation_parameters"
    assert "research_backend" in json.dumps(_error(response))
    _annotation_root = managed_runs_root() / "annotation.annotate"
    assert not _annotation_root.exists() or not any(_annotation_root.glob("sha256-*"))


def test_annotation_dependency_contract_excludes_external_trna_programs() -> None:
    spec = operations.describe("annotation.annotate")
    executable_names = {
        dependency.name for dependency in spec.dependencies if dependency.kind.value == "executable"
    }

    assert "tRNAscan-SE" not in executable_names
    assert "cmsearch" not in executable_names
