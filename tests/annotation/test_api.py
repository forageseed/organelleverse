from __future__ import annotations

from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.annotation.backends.base import AnnotationRequest
from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.annotation.writer import write_document_json
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.runtime import managed_runs_root


def _v1_genome(tmp_path: Path) -> OrganelleGenome:
    fasta = tmp_path / "genome.fasta"
    fasta.write_text(">r1\nATGAAATAG\n")
    return OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
        metadata=OrganelleMetadata(species="Arabidopsis thaliana"),
    )


def _annotation_document() -> AnnotationDocument:
    return AnnotationDocument(
        backend="fixture",
        requested_stages=("pcg",),
        completed_stages=("pcg",),
        records=(
            AnnotationRecord(
                seqid="r1",
                name="r1",
                description="fixture",
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


def test_full_name_facade_exports_canonical_operations() -> None:
    import organelleverse.annotation as annotation_facade
    from organelleverse.annotation import api

    assert ov.annotation is annotation_facade
    assert annotation_facade.annotate is api.annotate
    assert annotation_facade.extract is api.extract
    assert annotation_facade.write is api.write


def test_canonical_annotate_builds_strict_released_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from organelleverse.annotation import api

    captured: list[AnnotationRequest] = []

    def fake_run(genome: OrganelleGenome, request: AnnotationRequest) -> OrganelleResult:
        captured.append(request)
        return OrganelleResult(
            operation_id="annotation.annotate",
            scope=genome.organelle,
            status="failed",
            errors=(ErrorDetail(code="fixture", message="fixture"),),
        )

    monkeypatch.setattr(api, "run_annotation", fake_run)

    result = api.annotate(
        _v1_genome(tmp_path),
        backend="mitochondrion",
        threads=3,
        call_trna=False,
        call_rrna=True,
    )

    assert result.status == "failed"
    assert captured == [
        AnnotationRequest(
            backend="mitochondrion",
            workspace=managed_runs_root() / "annotation.annotate",
            threads=3,
            stages=("pcg", "rrna"),
        )
    ]


def test_canonical_extract_reads_annotation_artifact(tmp_path: Path) -> None:
    from organelleverse.annotation import api

    source = tmp_path / "annotation.json"
    write_document_json(_annotation_document(), source)
    genome = OrganelleGenome(
        organelle="mitochondrion",
        annotation=ArtifactRef.from_path(source, kind="annotation", format="json"),
        metadata=OrganelleMetadata(species="Arabidopsis thaliana"),
    )

    result = api.extract(
        genome,
        feature_types=("CDS",),
    )

    assert result.operation_id == "annotation.extract"
    assert result.status == "ok"
    assert Path(result.artifacts[0].uri).read_text().startswith(">nad1\nATGAAATAG")
    assert Path(result.artifacts[0].uri).is_relative_to(managed_runs_root())


def test_failed_canonical_write_preserves_existing_destination(tmp_path: Path) -> None:
    from organelleverse.annotation import api

    output = tmp_path / "published"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep")
    failed = OrganelleResult(
        operation_id="annotation.annotate",
        scope="mitochondrion",
        status="failed",
        errors=(ErrorDetail(code="fixture", message="fixture"),),
    )

    with pytest.raises(ValueError, match="successful annotation result"):
        api.write(failed, output=output)

    assert marker.read_text() == "keep"
    assert sorted(path.name for path in output.iterdir()) == ["existing.txt"]
