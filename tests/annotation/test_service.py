from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest

from organelleverse.annotation.backends.base import (
    AnnotationBackendName,
    AnnotationRequest,
    AnnotationStage,
    BackendRun,
)
from organelleverse.annotation.backends.mitochondrion import MitochondrionBackend
from organelleverse.annotation.execution import CommandEvidence, CommandRunner, ResolvedTool
from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.annotation.service import run_annotation, run_extraction
from organelleverse.annotation.writer import materialize_annotation, write_document_json
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata, OrganelleType


def annotation_document(
    *,
    requested: tuple[str, ...] = ("pcg", "trna", "rrna"),
    completed: tuple[str, ...] = ("pcg", "trna", "rrna"),
    include_trna: bool = True,
) -> AnnotationDocument:
    features = [
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
        )
    ]
    if include_trna:
        features.append(
            AnnotationFeature(
                feature_id="r1:1:tRNA",
                seqid="r1",
                type="tRNA",
                operator="single",
                parts=(LocationPart(start=0, end=3, strand=1),),
                qualifiers=(FeatureQualifier(name="gene", values=("trnM(cat)",)),),
                parents=(),
            )
        )
    features.append(
        AnnotationFeature(
            feature_id="r1:2:rRNA",
            seqid="r1",
            type="rRNA",
            operator="single",
            parts=(LocationPart(start=3, end=9, strand=1),),
            qualifiers=(FeatureQualifier(name="gene", values=("rrn5",)),),
            parents=(),
        )
    )
    return AnnotationDocument(
        backend="mitochondrion",
        requested_stages=requested,
        completed_stages=completed,
        records=(
            AnnotationRecord(
                seqid="r1",
                name="r1",
                description="service fixture",
                sequence="ATGAAATAG",
                features=tuple(features),
            ),
        ),
        source_metadata=FrozenMap({"topology": "linear"}),
    )


def genome_fixture(
    tmp_path: Path,
    *,
    organelle: OrganelleType = "mitochondrion",
    species: str = "Arabidopsis thaliana",
) -> OrganelleGenome:
    fasta = tmp_path / f"{organelle}.fasta"
    fasta.write_text(">r1\nATGAAATAG\n")
    return OrganelleGenome(
        organelle=organelle,
        sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
        metadata=OrganelleMetadata(species=species),
    )


def request(
    workspace: Path,
    *,
    backend: AnnotationBackendName = "mitochondrion",
    stages: tuple[AnnotationStage, ...] = ("pcg", "trna", "rrna"),
) -> AnnotationRequest:
    return AnnotationRequest(
        backend=backend,
        workspace=workspace,
        threads=2,
        stages=stages,
    )


@dataclass
class FakeBackend:
    document: AnnotationDocument
    error: Exception | None = None
    calls: int = 0
    name: str = "mitochondrion"
    organelle_types: tuple[OrganelleType, ...] = ("mitochondrion",)

    def run(
        self,
        genome: OrganelleGenome,
        request: AnnotationRequest,
        scratch: Path,
    ) -> BackendRun:
        del genome, request
        self.calls += 1
        if self.error is not None:
            raise self.error
        log = scratch / "backend.log"
        log.write_text("backend complete\n")
        return BackendRun(
            document=self.document,
            commands=(),
            software_versions={"fake": "1.0"},
            database_hashes={"fake_db": "a" * 64},
            logs=(log,),
        )


def test_service_runs_backend_and_persists_content_addressed_document(tmp_path: Path) -> None:
    genome = genome_fixture(tmp_path)
    adapter = FakeBackend(annotation_document())

    result = run_annotation(
        genome,
        request(tmp_path / "workspace"),
        adapters={"mitochondrion": adapter},
    )

    assert result.status == "ok"
    assert adapter.calls == 1
    assert "annotation_planned" not in result.flags
    annotation = next(item for item in result.artifacts if item.kind == "annotation")
    annotation_path = Path(annotation.uri)
    assert annotation.sha256 == sha256(annotation_path.read_bytes()).hexdigest()
    assert annotation_path.parent.name.startswith("sha256-")
    assert "features" not in result.metrics
    assert result.metrics["feature_count"] == 3
    assert result.provenance is not None
    assert result.provenance.actual_backend == "mitochondrion"
    assert not list((tmp_path / "workspace" / ".scratch").glob("annotation-*"))


def test_service_reports_rejected_cds_candidates_for_agents(tmp_path: Path) -> None:
    document = annotation_document().evolve(
        source_metadata=FrozenMap.from_json(
            {
                "rejected_cds_candidates": [
                    {
                        "gene_name": "nad5",
                        "start": 10,
                        "end": 18,
                        "issue_codes": ["premature_stop"],
                    }
                ],
                "missing_core_genes": ["nad5"],
            }
        )
    )

    result = run_annotation(
        genome_fixture(tmp_path),
        request(tmp_path / "workspace"),
        adapters={"mitochondrion": FakeBackend(document)},
    )

    assert result.status == "warning"
    assert result.metrics["rejected_cds_count"] == 1
    assert result.metrics["missing_core_genes"] == ("nad5",)
    rejected_value = result.metrics["rejected_cds_candidates"]
    assert isinstance(rejected_value, tuple)
    first_rejected = rejected_value[0]
    assert isinstance(first_rejected, FrozenMap)
    assert first_rejected["gene_name"] == "nad5"
    assert "cds_candidates_rejected" in result.flags
    assert "missing_core_genes" in result.flags
    assert "1 invalid CDS candidate rejected" in result.summary_text


def test_plastid_auto_fails_without_released_backend(tmp_path: Path) -> None:
    genome = genome_fixture(tmp_path, organelle="plastid")

    result = run_annotation(
        genome,
        request(tmp_path / "workspace", backend="auto"),
        adapters={"mitochondrion": FakeBackend(annotation_document())},
    )

    assert result.status == "failed"
    assert result.errors[0].code == "unsupported_annotation_scope"
    assert not (tmp_path / "workspace" / "runs").exists()


def test_missing_sequence_artifact_fails_before_backend(tmp_path: Path) -> None:
    annotation_path = tmp_path / "source.json"
    write_document_json(annotation_document(), annotation_path)
    genome = OrganelleGenome(
        organelle="mitochondrion",
        annotation=ArtifactRef.from_path(annotation_path, kind="annotation", format="json"),
        metadata=OrganelleMetadata(species="Arabidopsis thaliana"),
    )
    adapter = FakeBackend(annotation_document())

    result = run_annotation(
        genome,
        request(tmp_path / "workspace"),
        adapters={"mitochondrion": adapter},
    )

    assert result.status == "failed"
    assert result.errors[0].code == "input.missing_sequence_artifact"
    assert adapter.calls == 0
    assert not (tmp_path / "workspace" / "runs").exists()


def test_failed_preflight_preserves_dependency_error_and_no_final_artifact(
    tmp_path: Path,
) -> None:
    genome = genome_fixture(tmp_path)
    error = OrganelleDependencyError(
        code="dependency_missing",
        message="blastn is missing",
        details={"missing": ["blastn"]},
    )
    adapter = FakeBackend(annotation_document(), error=error)

    result = run_annotation(
        genome,
        request(tmp_path / "workspace"),
        adapters={"mitochondrion": adapter},
    )

    assert result.status == "failed"
    assert result.errors[0].code == "dependency_missing"
    assert result.errors[0].details["missing"] == ("blastn",)
    assert not (tmp_path / "workspace" / "runs").exists()


def test_backend_exception_becomes_stable_failure_without_final_artifact(tmp_path: Path) -> None:
    genome = genome_fixture(tmp_path)
    adapter = FakeBackend(annotation_document(), error=RuntimeError("backend exploded"))

    result = run_annotation(
        genome,
        request(tmp_path / "workspace"),
        adapters={"mitochondrion": adapter},
    )

    assert result.status == "failed"
    assert result.errors[0].code == "backend_execution_failed"
    assert result.errors[0].details["exception_type"] == "RuntimeError"
    assert not (tmp_path / "workspace" / "runs").exists()


def test_invalid_backend_output_is_not_treated_as_annotation(tmp_path: Path) -> None:
    class InvalidBackend(FakeBackend):
        def run(
            self,
            genome: OrganelleGenome,
            request: AnnotationRequest,
            scratch: Path,
        ) -> BackendRun:
            del genome, request, scratch
            self.calls += 1
            return cast(BackendRun, object())

    genome = genome_fixture(tmp_path)
    adapter = InvalidBackend(annotation_document())

    result = run_annotation(
        genome,
        request(tmp_path / "workspace"),
        adapters={"mitochondrion": adapter},
    )

    assert result.status == "failed"
    assert result.errors[0].code == "backend_output_invalid"
    assert not (tmp_path / "workspace" / "runs").exists()


def test_canonical_validation_failure_is_named_and_not_persisted(tmp_path: Path) -> None:
    document = annotation_document()
    first = document.records[0].features[0].evolve(parts=(LocationPart(start=0, end=10, strand=1),))
    record = document.records[0].evolve(features=(first, *document.records[0].features[1:]))
    adapter = FakeBackend(document.evolve(records=(record,)))

    result = run_annotation(
        genome_fixture(tmp_path),
        request(tmp_path / "workspace"),
        adapters={"mitochondrion": adapter},
    )

    assert result.status == "failed"
    assert result.errors[0].code == "annotation_validation_failed"
    issue_codes = result.errors[0].details["issue_codes"]
    assert isinstance(issue_codes, tuple)
    assert "coordinate_out_of_bounds" in issue_codes
    assert not (tmp_path / "workspace" / "runs").exists()


def test_requested_stage_mismatch_fails_closed(tmp_path: Path) -> None:
    document = annotation_document(
        requested=("pcg",),
        completed=("pcg",),
        include_trna=False,
    )

    result = run_annotation(
        genome_fixture(tmp_path),
        request(tmp_path / "workspace", stages=("pcg", "trna")),
        adapters={"mitochondrion": FakeBackend(document)},
    )

    assert result.status == "failed"
    assert result.errors[0].code == "annotation_validation_failed"
    issue_codes = result.errors[0].details["issue_codes"]
    assert isinstance(issue_codes, tuple)
    assert "requested_stage_incomplete" in issue_codes
    assert not (tmp_path / "workspace" / "runs").exists()


def test_idempotent_rerun_reuses_same_semantic_run(tmp_path: Path) -> None:
    genome = genome_fixture(tmp_path)
    adapter = FakeBackend(annotation_document())
    annotation_request = request(tmp_path / "workspace")

    first = run_annotation(genome, annotation_request, adapters={"mitochondrion": adapter})
    second = run_annotation(genome, annotation_request, adapters={"mitochondrion": adapter})

    first_annotation = next(item for item in first.artifacts if item.kind == "annotation")
    second_annotation = next(item for item in second.artifacts if item.kind == "annotation")
    assert first.status == second.status == "ok"
    assert first_annotation.object_id == second_annotation.object_id
    assert first_annotation.uri == second_annotation.uri
    assert len([p for p in (tmp_path / "workspace").iterdir() if p.name.startswith("sha256-")]) == 1


def test_semantic_manifest_ignores_transient_command_paths(tmp_path: Path) -> None:
    class ScratchPathBackend(FakeBackend):
        def run(
            self,
            genome: OrganelleGenome,
            request: AnnotationRequest,
            scratch: Path,
        ) -> BackendRun:
            del genome, request
            command = CommandEvidence(
                sequence=1,
                stage="pcg",
                argv=(
                    "/tools/blastn",
                    "-in",
                    str(scratch / "input.fasta"),
                    "-out",
                    str(scratch / "hits.tsv"),
                ),
                cwd=str(scratch),
                started_at="2026-07-13T00:00:00+00:00",
                duration_seconds=0.1,
                timeout_seconds=30,
                returncode=0,
                termination="exit",
            )
            return BackendRun(
                document=self.document,
                commands=(command,),
                software_versions={"blastn": "2.14.1+"},
                database_hashes={"fake_db": "a" * 64},
                logs=(),
            )

    genome = genome_fixture(tmp_path)
    adapter = ScratchPathBackend(annotation_document())

    first = run_annotation(
        genome,
        request(tmp_path / "workspace-a"),
        adapters={"mitochondrion": adapter},
    )
    second = run_annotation(
        genome,
        request(tmp_path / "workspace-b"),
        adapters={"mitochondrion": adapter},
    )

    assert first.status == second.status == "ok"
    assert first.provenance is not None
    assert second.provenance is not None
    assert first.provenance.run_manifest_id == second.provenance.run_manifest_id
    first_manifest = next(item for item in first.artifacts if item.kind == "annotation_manifest")
    second_manifest = next(item for item in second.artifacts if item.kind == "annotation_manifest")
    assert first_manifest.sha256 == second_manifest.sha256


@pytest.mark.parametrize("species", ["", "   "])
def test_released_mitochondrial_backend_requires_species(
    tmp_path: Path,
    species: str,
) -> None:
    result = run_annotation(
        genome_fixture(tmp_path, species=species),
        request(tmp_path / "workspace", stages=("pcg",)),
    )

    assert result.status == "failed"
    assert result.errors[0].code == "unsupported_annotation_scope"
    assert result.errors[0].suggested_action["provide_metadata"] == ("species",)


def test_released_pinaceae_trna_scope_fails_before_tool_preflight(tmp_path: Path) -> None:
    result = run_annotation(
        genome_fixture(tmp_path, species="Pinus taeda"),
        request(tmp_path / "workspace", stages=("pcg", "trna")),
    )

    assert result.status == "failed"
    assert result.errors[0].code == "unsupported_annotation_scope"
    assert result.errors[0].suggested_action["omit_stage"] == "trna"


def test_mitochondrial_adapter_injects_preflighted_tools_and_fixed_engines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from organelleverse.annotation.backends import mitochondrion as backend_module
    from organelleverse.annotation.mitochondrion.cds import RejectedCDSCandidate
    from organelleverse.annotation.mitochondrion.result import MitochondrialAnnotationStats

    database_root = tmp_path / "database"
    combined_hmm = database_root / "hmm" / "combined.hmm"
    gene_info = database_root / "gene_info"
    blast_refs = database_root / "blast_refs"
    exon_refs = database_root / "exon_refs"
    rrna_refs = database_root / "rrna_refs"
    for path in (
        combined_hmm,
        gene_info / "genes.json",
        blast_refs / "pcg.fa",
        exon_refs / "exon.fa",
        rrna_refs / "rrna.fa",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)

    class FakeDatabase:
        def __init__(self) -> None:
            self.combined_hmm = combined_hmm
            self.gene_info_dir = gene_info
            self.blast_ref_dir = blast_refs
            self.exon_ref_dir = exon_refs
            self.rrna_ref_dir = rrna_refs

        def verify(self) -> list[str]:
            return []

    captured: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            self.dependency_issues: list[str] = []
            captured.update(kwargs)

        def run_pipeline(self) -> MitochondrialAnnotationStats:
            output_dir = cast(Path, captured["output_dir"])
            paths = materialize_annotation(annotation_document(), output_dir / "published")
            return MitochondrialAnnotationStats(
                sample_name="fixture",
                output_paths=(paths["genbank"],),
                warnings=(),
                contig_count=1,
                cds_predicted=1,
                trna_predicted=1,
                rrna_predicted=1,
                missing_core_genes=(),
                invalid_cds=1,
                rejected_cds_candidates=(
                    RejectedCDSCandidate(
                        gene_name="nad5",
                        start=10,
                        end=18,
                        strand=1,
                        parts=((10, 18, 1),),
                        issue_codes=("premature_stop",),
                        issue_messages=("Internal stop codon at aa position 2",),
                    ),
                ),
            )

    requested_names: list[tuple[str, ...]] = []

    def fake_resolve(
        names: tuple[str, ...],
        *,
        runner: CommandRunner | None = None,
        version_timeout: int = 30,
        paths: Mapping[str, str] | None = None,
    ) -> tuple[ResolvedTool, ...]:
        del version_timeout
        requested_names.append(names)
        assert runner is not None
        assert runner.log_path is not None
        runner.log_path.write_text("")
        return tuple(
            ResolvedTool(
                name=name,
                path=(paths or {}).get(name, f"/tools/{name}"),
                version=f"{name} 1.0",
                version_argv=(f"/tools/{name}", "--version"),
            )
            for name in names
        )

    monkeypatch.setattr(backend_module, "DBManager", FakeDatabase)
    monkeypatch.setattr(backend_module, "MitochondrialAnnotationPipeline", FakePipeline)
    monkeypatch.setattr(backend_module, "resolve_required_tools", fake_resolve)
    # NCBI-only preflight: LOSAT unavailable (or ORG_VERSE_LOSAT_BIN=ncbi).
    monkeypatch.setattr(backend_module, "resolve_losat", lambda: None)
    annotation_request = request(tmp_path / "workspace")

    backend_run = MitochondrionBackend().run(
        genome_fixture(tmp_path),
        annotation_request,
        tmp_path / "scratch",
    )

    assert requested_names == [("blastn", "makeblastdb", "tblastn")]
    assert captured["trna_engine"] == "native"
    assert captured["rrna_engine"] == "pyhmmer"
    assert captured["call_pcg"] is True
    assert captured["call_trna"] is True
    assert captured["call_rrna"] is True
    assert captured["tool_paths"] == {
        "blastn": "/tools/blastn",
        "makeblastdb": "/tools/makeblastdb",
        "tblastn": "/tools/tblastn",
    }
    assert isinstance(captured["command_runner"], CommandRunner)
    assert backend_run.document.requested_stages == annotation_request.stages
    assert backend_run.document.completed_stages == annotation_request.stages
    assert len(backend_run.database_hashes["mitochondrion_trna_cm"]) == 64
    assert len(backend_run.database_hashes["mitochondrion_trna_hmm"]) == 64
    rejected = backend_run.document.source_metadata["rejected_cds_candidates"]
    assert isinstance(rejected, tuple)
    first_rejected = rejected[0]
    assert isinstance(first_rejected, FrozenMap)
    assert first_rejected["gene_name"] == "nad5"
    assert first_rejected["issue_codes"] == ("premature_stop",)
    assert backend_run.database_hashes["mitochondrion_hmm"]


def test_mitochondrial_adapter_checks_packaged_database_before_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from organelleverse.annotation.backends import mitochondrion as backend_module

    class BrokenDatabase:
        def verify(self) -> list[str]:
            return ["combined HMM missing"]

    def tools_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("tool preflight ran before database validation")

    monkeypatch.setattr(backend_module, "DBManager", BrokenDatabase)
    monkeypatch.setattr(backend_module, "resolve_required_tools", tools_must_not_run)

    with pytest.raises(OrganelleDependencyError) as raised:
        MitochondrionBackend().run(
            genome_fixture(tmp_path),
            request(tmp_path / "workspace", stages=("pcg",)),
            tmp_path / "scratch",
        )

    assert raised.value.code == "dependency_missing"
    assert raised.value.as_dict()["details"]["database_issues"] == ["combined HMM missing"]


def test_extraction_uses_canonical_annotation_without_rerunning_backend(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.json"
    write_document_json(annotation_document(), annotation_path)
    genome = OrganelleGenome(
        organelle="mitochondrion",
        annotation=ArtifactRef.from_path(annotation_path, kind="annotation", format="json"),
        metadata=OrganelleMetadata(species="Arabidopsis thaliana"),
    )

    result = run_extraction(genome, tmp_path / "extracted", ("CDS", "tRNA"))

    assert result.status == "ok"
    assert {artifact.kind for artifact in result.artifacts} == {
        "annotation_cds",
        "annotation_trna",
    }
    assert all(Path(artifact.uri).is_file() for artifact in result.artifacts)
    cds = next(artifact for artifact in result.artifacts if artifact.kind == "annotation_cds")
    assert Path(cds.uri).read_text().startswith(">nad1\nATGAAATAG")


def test_mitochondrial_preflight_prefers_losat_and_keeps_available_ncbi_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With LOSAT available, NCBI BLAST+ is optional: only tools present are resolved."""
    from organelleverse.annotation.backends import mitochondrion as backend_module

    requested: list[tuple[str, ...]] = []

    def fake_resolve(
        names: tuple[str, ...],
        *,
        runner: CommandRunner | None = None,
        version_timeout: int = 30,
        paths: Mapping[str, str] | None = None,
    ) -> tuple[ResolvedTool, ...]:
        del runner, version_timeout
        requested.append(names)
        return tuple(
            ResolvedTool(
                name=name,
                path=(paths or {}).get(name, f"/tools/{name}"),
                version=f"{name} 1.0",
                version_argv=(f"/tools/{name}", "--version"),
            )
            for name in names
        )

    monkeypatch.setattr(backend_module, "resolve_required_tools", fake_resolve)
    monkeypatch.setattr(backend_module, "resolve_losat", lambda: "/opt/losat/LOSAT")
    monkeypatch.setattr(
        backend_module.shutil, "which", lambda name: "/tools/blastn" if name == "blastn" else None
    )

    tool_paths, versions = backend_module._resolve_search_tools(
        CommandRunner(log_dir=tmp_path / "evidence")
    )

    assert requested == [("losat",), ("blastn",)]
    assert tool_paths == {"losat": "/opt/losat/LOSAT", "blastn": "/tools/blastn"}
    assert versions == {"losat": "losat 1.0", "blastn": "blastn 1.0"}
