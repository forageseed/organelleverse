"""Reusable canonical ``annotation.annotate`` Result fixture for annotation QC.

The fixture builds a real :class:`~organelleverse.annotation.models.AnnotationDocument`,
writes the canonical ``annotation.json`` and source ``run_manifest.json`` whose
semantic identity agrees with the document and Result provenance, and assembles an
:class:`~organelleverse.core.result.OrganelleResult` matching the contract that
``organelleverse.annotation.annotate`` emits. It is test-only infrastructure: it
does not import the annotation-QC contracts under test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.annotation.validation import validate_document
from organelleverse.annotation.writer import load_document_json, write_document_json
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import OrganelleResult

_SOFTWARE_VERSIONS = {"annotation-tool": "1.2.0", "organelleverse": "0.0.1"}
_DATABASE_HASHES = {
    "mitochondrion_hmm": hashlib.sha256(b"hmm").hexdigest(),
    "mitochondrion_gene_info": hashlib.sha256(b"gene-info").hexdigest(),
}
_ARGV: tuple[tuple[str, ...], ...] = (
    ("annotation-tool", "--input", "<path>/genome.fa", "--output", "<path>/annotation.gb"),
)
_PADDING = 80
_SLOT = 240


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _request_payload(stages: tuple[str, ...]) -> dict[str, object]:
    return {"backend": "mitochondrion", "threads": 1, "stages": list(stages)}


def _orf(length_codons: int) -> str:
    """A valid complete CDS ORF: ``ATG`` + Ala codons + ``TAA`` (no internal stop)."""

    if length_codons < 3:
        raise ValueError("ORF needs at least start, one sense, and stop codons")
    return "ATG" + "GCA" * (length_codons - 2) + "TAA"


def _qualifier(name: str, *values: str) -> FeatureQualifier:
    return FeatureQualifier(name=name, values=tuple(values))


_Slot: TypeAlias = tuple[str, str, str, int, int, Literal[-1, 1], tuple[FeatureQualifier, ...]]


def _feature(
    seqid: str,
    feature_id: str,
    feature_type: str,
    start: int,
    end: int,
    strand: Literal[-1, 1],
    qualifiers: tuple[FeatureQualifier, ...],
) -> AnnotationFeature:
    part = LocationPart(start=start, end=end, strand=strand)
    operator: Literal["single", "join", "order"] = "single"
    return AnnotationFeature(
        feature_id=feature_id,
        seqid=seqid,
        type=feature_type,
        operator=operator,
        parts=(part,),
        qualifiers=qualifiers,
        parents=(),
    )


def _build_record(
    seqid: str,
    *,
    genes: tuple[str, ...],
    duplicate_gene: str | None,
    partial_gene: str | None,
    pseudo_gene: str | None,
    transl_except_gene: str | None,
    invalid_extra: str | None,
    trna_genes: tuple[str, ...],
    rrna_genes: tuple[str, ...],
) -> AnnotationRecord:
    """Lay out features along one sequence and return the ``AnnotationRecord``."""

    slots: list[_Slot] = []
    cursor = _PADDING

    def claim(length: int) -> tuple[int, int]:
        nonlocal cursor
        start = cursor
        end = start + length
        cursor = end + _PADDING
        return start, end

    def add_cds(gene: str, suffix: str, qualifiers: tuple[FeatureQualifier, ...]) -> None:
        start, end = claim(36)
        slots.append((f"{gene}-{suffix}", "cds", gene, start, end, 1, qualifiers))

    for gene in genes:
        add_cds(gene, "cds", (_qualifier("gene", gene),))
    if duplicate_gene is not None:
        add_cds(duplicate_gene, "dup", (_qualifier("gene", duplicate_gene),))
    if partial_gene is not None:
        add_cds(partial_gene, "partial", (_qualifier("gene", partial_gene), _qualifier("partial")))
    if pseudo_gene is not None:
        add_cds(pseudo_gene, "pseudo", (_qualifier("gene", pseudo_gene), _qualifier("pseudo")))
    if transl_except_gene is not None:
        add_cds(
            transl_except_gene,
            "transl_except",
            (
                _qualifier("gene", transl_except_gene),
                _qualifier("transl_except", "123..125:aa:Sec"),
            ),
        )
    if invalid_extra is not None:
        # Intentionally out-of-bounds CDS to force a structural validation error.
        start, _end = claim(36)
        slots.append(
            (
                f"{invalid_extra}-invalid",
                "cds",
                invalid_extra,
                start,
                start + 36 + 10_000,
                1,
                (_qualifier("gene", invalid_extra),),
            )
        )
    for gene in trna_genes:
        start, end = claim(48)
        slots.append((f"{gene}-trna", "trna", gene, start, end, 1, (_qualifier("gene", gene),)))
    for gene in rrna_genes:
        start, end = claim(120)
        slots.append((f"{gene}-rrna", "rrna", gene, start, end, 1, (_qualifier("gene", gene),)))

    total_length = cursor + _PADDING
    sequence_chars: list[str] = ["A"] * total_length
    for _feature_id, ftype, _gene, start, end, _strand, _qualifiers in slots:
        if ftype != "cds" or end > total_length:
            continue
        orf = _orf((end - start) // 3)
        sequence_chars[start:end] = list(orf)
    sequence = "".join(sequence_chars)

    features = tuple(
        _feature(seqid, feature_id, ftype, start, end, strand, qualifiers)
        for feature_id, ftype, _gene, start, end, strand, qualifiers in slots
    )
    return AnnotationRecord(
        seqid=seqid,
        name=seqid,
        description=f"{seqid} canonical annotation fixture",
        sequence=sequence,
        features=features,
    )


def _rejected_candidate(gene: str) -> dict[str, Any]:
    start = 5
    end = 41
    return {
        "gene_name": gene,
        "start": start,
        "end": end,
        "strand": 1,
        "parts": ({"start": start, "end": end, "strand": 1},),
        "issue_codes": ("premature_stop",),
        "issue_messages": ("CDS contains a premature stop codon",),
    }


def annotation_result_fixture(
    root: Path,
    *,
    genes: tuple[str, ...] = ("atp1", "cob", "cox1"),
    rejected_gene: str | None = None,
    duplicate_gene: str | None = None,
    partial_gene: str | None = None,
    pseudo_gene: str | None = None,
    transl_except_gene: str | None = None,
    invalid_extra: str | None = None,
    trna_genes: tuple[str, ...] = ("trnC",),
    rrna_genes: tuple[str, ...] = ("rrn18",),
    stages: tuple[str, ...] = ("pcg", "trna", "rrna"),
    missing_core_genes: tuple[str, ...] = (),
) -> OrganelleResult:
    """Return one valid mitochondrial ``annotation.annotate`` Result.

    The Result carries exactly one ``annotation`` and one ``annotation_manifest``
    artifact whose on-disk content agrees with the document, manifest, metrics,
    and provenance. Optional knobs inject review signals (rejected candidates,
    duplicates, partial/pseudo features, translation exceptions, or one invalid
    CDS) for evaluation tests.
    """

    run_dir = root / "sha256-fixture"
    run_dir.mkdir(parents=True, exist_ok=True)
    seqid = "MT-FIXTURE"

    record = _build_record(
        seqid,
        genes=genes,
        duplicate_gene=duplicate_gene,
        partial_gene=partial_gene,
        pseudo_gene=pseudo_gene,
        transl_except_gene=transl_except_gene,
        invalid_extra=invalid_extra,
        trna_genes=trna_genes,
        rrna_genes=rrna_genes,
    )

    rejected_candidates: tuple[dict[str, Any], ...] = ()
    if rejected_gene is not None:
        rejected_candidates = (_rejected_candidate(rejected_gene),)

    document = AnnotationDocument(
        backend="mitochondrion",
        requested_stages=stages,
        completed_stages=stages,
        records=(record,),
        source_metadata=FrozenMap(
            {
                "missing_core_genes": missing_core_genes,
                "rejected_cds_candidates": rejected_candidates,
            }
        ),
    )

    annotation_path = run_dir / "annotation.json"
    write_document_json(document, annotation_path)
    document = load_document_json(annotation_path)

    validation = validate_document(document, stages)
    feature_counts: dict[str, int] = {
        key: cast(int, value) for key, value in validation.feature_counts.items()
    }
    feature_count = sum(feature_counts.values())

    input_sequence = "ATGC" * 256
    input_path = root / "input.fa"
    input_path.write_text(f">{seqid}-input\n{input_sequence}\n")
    input_artifact_hash = hashlib.sha256(input_sequence.encode("utf-8")).hexdigest()
    input_object_id = f"genome:sha256:{input_artifact_hash}"

    parameters_hash = _sha256_json(_request_payload(stages))
    semantic_payload: dict[str, object] = {
        "schema_version": "organelleverse.annotation-run.v1",
        "operation_id": "annotation.annotate",
        "input_object_id": input_object_id,
        "input_artifact_hash": input_artifact_hash,
        "parameters_hash": parameters_hash,
        "backend": "mitochondrion",
        "annotation_object_id": document.object_id,
        "software_versions": dict(sorted(_SOFTWARE_VERSIONS.items())),
        "database_hashes": dict(sorted(_DATABASE_HASHES.items())),
        "argv": [list(item) for item in _ARGV],
    }
    run_manifest_id = f"annotation-run:sha256:{_sha256_json(semantic_payload)}"

    command_path = run_dir / "commands.jsonl"
    command_path.write_text(
        json.dumps({"argv": list(_ARGV[0])}, sort_keys=True, allow_nan=False) + "\n"
    )
    manifest_record: dict[str, object] = {
        **semantic_payload,
        "run_manifest_id": run_manifest_id,
        "commands_file": command_path.name,
        "logs": [],
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_record, sort_keys=True, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n"
    )

    annotation_artifact = ArtifactRef.from_path(
        annotation_path,
        kind="annotation",
        format="json",
        media_type="application/json",
    )
    manifest_artifact = ArtifactRef.from_path(
        manifest_path,
        kind="annotation_manifest",
        format="json",
        media_type="application/json",
    )

    rejected_count = len(rejected_candidates)
    summary = f"Annotated {feature_count} features with mitochondrion."
    status = "warning" if rejected_count or missing_core_genes else "ok"
    provenance = ResultProvenance(
        operation_id="annotation.annotate",
        operation_version="1.0",
        package_version=_SOFTWARE_VERSIONS["organelleverse"],
        git_commit="",
        input_object_ids=(input_object_id,),
        input_artifact_hashes=(input_artifact_hash,),
        parameters_hash=parameters_hash,
        requested_backend="mitochondrion",
        actual_backend="mitochondrion",
        attempted_backends=("mitochondrion",),
        software_versions=FrozenMap(dict(_SOFTWARE_VERSIONS)),
        database_hashes=FrozenMap(dict(_DATABASE_HASHES)),
        argv=tuple(argument for command in _ARGV for argument in command),
        run_manifest_id=run_manifest_id,
    )
    metrics: dict[str, object] = {
        "backend": "mitochondrion",
        "requested_stages": stages,
        "completed_stages": document.completed_stages,
        "feature_count": feature_count,
        "feature_counts": feature_counts,
        "command_count": len(_ARGV),
        "rejected_cds_count": rejected_count,
        "rejected_cds_candidates": rejected_candidates,
        "missing_core_genes": missing_core_genes,
    }
    return OrganelleResult(
        operation_id="annotation.annotate",
        scope="mitochondrion",
        status=status,
        summary_text=summary,
        metrics=FrozenMap(metrics),
        flags=("annotation_validated", "annotation_persisted"),
        artifacts=(annotation_artifact, manifest_artifact),
        provenance=provenance,
    )


__all__ = ["annotation_result_fixture"]
