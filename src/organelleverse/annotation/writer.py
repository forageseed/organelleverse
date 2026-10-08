"""Canonical serialization and atomic annotation materialization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Iterable
from contextlib import suppress
from pathlib import Path
from typing import cast
from uuid import uuid4

from Bio.Seq import Seq
from Bio.SeqFeature import (
    AfterPosition,
    BeforePosition,
    CompoundLocation,
    ExactPosition,
    SeqFeature,
    SimpleLocation,
    UncertainPosition,
)
from Bio.SeqRecord import SeqRecord

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.result import OrganelleResult

from .genbank import extract_feature_records, parse_genbank
from .gff3 import document_to_gff3
from .models import AnnotationDocument, AnnotationFeature, LocationPart
from .splice_layout import MAX_CIS_INTRON, parts_are_trans_spliced
from .validation import validate_document

_OUTPUT_NAMES = {
    "json": "annotation.json",
    "genbank": "annotation.gb",
    "gff3": "annotation.gff3",
    "tbl": "annotation.tbl",
    "unedited_tbl": "annotation.unedited.tbl",
    "fsa": "genome.fsa",
    "cds_fasta": "cds.fasta",
    "protein_fasta": "proteins.fasta",
    "trna_fasta": "trna.fasta",
    "rrna_fasta": "rrna.fasta",
    "manifest": "manifest.json",
}

#: PMGA-style concatenated view of a multi-contig document, written under
#: ``concatenated/`` so downstream tools that want one record keep working.
_CONCATENATED_DIR = "concatenated"
_CONCATENATED_NAMES = {
    "concat_genbank": "annotation.gb",
    "concat_gff3": "annotation.gff3",
    "concat_tbl": "annotation.tbl",
    "concat_unedited_tbl": "annotation.unedited.tbl",
    "concat_fasta": "genome.fsa",
    "concat_stat": "merged_stat.txt",
}


def write_document_json(document: AnnotationDocument, path: str | Path) -> Path:
    """Write deterministic canonical JSON with a terminal newline."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        document.model_dump(mode="json"),
        sort_keys=True,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
    )
    destination.write_text(text + "\n")
    return destination


def load_document_json(path: str | Path) -> AnnotationDocument:
    """Load canonical JSON and verify all serialized object identities."""

    return AnnotationDocument.model_validate_json(Path(path).read_bytes())


def _position(value: int, status: str) -> int:
    if status == "before":
        return cast(int, BeforePosition(value))
    if status == "after":
        return cast(int, AfterPosition(value))
    if status == "unknown":
        return cast(int, UncertainPosition(value))
    return cast(int, ExactPosition(value))


def _simple_location(part: LocationPart) -> SimpleLocation:
    return SimpleLocation(
        _position(part.start, part.start_status),
        _position(part.end, part.end_status),
        strand=part.strand,
    )


_SPLICED_TYPES = frozenset({"gene", "CDS", "mRNA", "tRNA", "rRNA"})


def _seq_feature(feature: AnnotationFeature, length: int) -> SeqFeature:
    locations = [_simple_location(part) for part in feature.parts]
    location: SimpleLocation | CompoundLocation
    if feature.operator == "single":
        location = locations[0]
    else:
        location = CompoundLocation(locations, operator=feature.operator)
    qualifiers = {qualifier.name: list(qualifier.values) for qualifier in feature.qualifiers}
    if (
        feature.type in _SPLICED_TYPES
        and "trans_splicing" not in qualifiers
        and parts_are_trans_spliced(
            [(part.start, part.end, part.strand) for part in feature.parts],
            length,
            MAX_CIS_INTRON,
        )
    ):
        qualifiers["trans_splicing"] = [""]
    if "trans_splicing" in qualifiers:
        # a flag qualifier: Biopython writes None as a bare /trans_splicing
        qualifiers["trans_splicing"] = [None]
    return SeqFeature(location=location, type=feature.type, qualifiers=qualifiers)


def _seq_records(document: AnnotationDocument) -> list[SeqRecord]:
    topology_value = document.source_metadata.get("topology", "linear")
    topology = topology_value if isinstance(topology_value, str) else "linear"
    per_record = dict(document.source_metadata.get("record_topology") or {})
    records: list[SeqRecord] = []
    for record in document.records:
        seq_record = SeqRecord(
            Seq(record.sequence),
            id=record.seqid,
            name=record.name or record.seqid,
            description=record.description,
        )
        seq_record.annotations["molecule_type"] = "DNA"
        seq_record.annotations["topology"] = per_record.get(record.seqid, topology)
        seq_record.features = [
            _seq_feature(feature, len(record.sequence)) for feature in record.features
        ]
        records.append(seq_record)
    return records


def _write_genbank(document: AnnotationDocument, path: Path) -> None:
    from Bio.SeqIO import write as bio_write  # pyright: ignore[reportUnknownVariableType]

    write_records = cast(
        Callable[[Iterable[SeqRecord], str | Path, str], int],
        bio_write,
    )
    count = write_records(_seq_records(document), path, "genbank")
    if count != len(document.records):
        raise ValueError("GenBank writer did not emit every annotation record")


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    lines = [line for name, sequence in records for line in (f">{name}", sequence)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_signatures(document: AnnotationDocument) -> set[tuple[object, ...]]:
    return {
        (
            feature.type,
            tuple((part.start, part.end, part.strand) for part in feature.parts),
            feature.qualifier_values("gene"),
            feature.qualifier_values("product"),
        )
        for record in document.records
        for feature in record.features
    }


def _write_manifest(
    document: AnnotationDocument, paths: dict[str, Path], directory: Path
) -> None:
    files = {
        key: {
            "name": path.relative_to(directory).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in sorted(paths.items())
        if key != "manifest"
    }
    payload = {
        "schema_version": "organelleverse.annotation-manifest.v1",
        "annotation_object_id": document.object_id,
        "files": files,
    }
    paths["manifest"].write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
    )


def _validate_materialized(document: AnnotationDocument, paths: dict[str, Path]) -> None:
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("materialized annotation is missing a required file")
    if load_document_json(paths["json"]) != document:
        raise ValueError("canonical JSON did not round-trip")
    reparsed = parse_genbank(paths["genbank"])
    if _feature_signatures(reparsed) != _feature_signatures(document):
        raise ValueError("GenBank output did not preserve normalized features")
    gff_lines = paths["gff3"].read_text().splitlines()
    if not gff_lines or gff_lines[0] != "##gff-version 3":
        raise ValueError("GFF3 output is malformed")
    record_seqids = {record.seqid for record in document.records}
    fsa_headers = {
        line[1:].split()[0]
        for line in paths["fsa"].read_text().splitlines()
        if line.startswith(">")
    }
    if fsa_headers != record_seqids:
        raise ValueError("submission FASTA is malformed")
    for tbl_key in ("tbl", "unedited_tbl"):
        tbl_lines = paths[tbl_key].read_text().splitlines()
        headers = [line for line in tbl_lines if line.startswith(">Feature ")]
        if len(headers) != len(document.records):
            raise ValueError("TBL output is malformed")
        if {header.removeprefix(">Feature ") for header in headers} != record_seqids:
            raise ValueError("TBL output is malformed")
    manifest_value: object = json.loads(paths["manifest"].read_text())
    if not isinstance(manifest_value, dict):
        raise ValueError("annotation manifest is malformed")
    manifest = cast(dict[str, object], manifest_value)
    if manifest.get("annotation_object_id") != document.object_id:
        raise ValueError("annotation manifest is malformed")


def _write_concatenated(document: AnnotationDocument, directory: Path) -> dict[str, Path]:
    """Materialize the 200-N concatenated view of a multi-contig document."""
    from .contigs import merge_document_by_contigs, merged_stat_lines
    from .mitochondrion.tbl import render_tbl_document
    from .table2asn import write_submission_fasta

    directory.mkdir()
    merged = merge_document_by_contigs(document)
    paths = {key: directory / name for key, name in _CONCATENATED_NAMES.items()}
    _write_genbank(merged, paths["concat_genbank"])
    paths["concat_gff3"].write_text("\n".join(document_to_gff3(merged)) + "\n")
    render_tbl_document(merged, paths["concat_tbl"])
    render_tbl_document(merged, paths["concat_unedited_tbl"], unedited=True)
    write_submission_fasta(merged, paths["concat_fasta"])
    paths["concat_stat"].write_text("\n".join(merged_stat_lines(merged)) + "\n")
    record = merged.records[0]
    fsa_headers = {
        line[1:].split()[0]
        for line in paths["concat_fasta"].read_text().splitlines()
        if line.startswith(">")
    }
    if fsa_headers != {record.seqid}:
        raise ValueError("concatenated submission FASTA is malformed")
    for tbl_key in ("concat_tbl", "concat_unedited_tbl"):
        headers = [
            line.removeprefix(">Feature ")
            for line in paths[tbl_key].read_text().splitlines()
            if line.startswith(">Feature ")
        ]
        if headers != [record.seqid]:
            raise ValueError("concatenated TBL output is malformed")
    return paths


def _write_materialized_files(
    document: AnnotationDocument,
    directory: Path,
) -> dict[str, Path]:
    directory.mkdir()
    paths = {key: directory / name for key, name in _OUTPUT_NAMES.items()}
    write_document_json(document, paths["json"])
    _write_genbank(document, paths["genbank"])
    paths["gff3"].write_text("\n".join(document_to_gff3(document)) + "\n")
    from .mitochondrion.tbl import render_tbl_document
    from .table2asn import write_submission_fasta

    render_tbl_document(document, paths["tbl"])
    render_tbl_document(document, paths["unedited_tbl"], unedited=True)
    write_submission_fasta(document, paths["fsa"])
    _write_fasta(paths["cds_fasta"], extract_feature_records(document, "CDS"))
    _write_fasta(paths["protein_fasta"], extract_feature_records(document, "Protein"))
    _write_fasta(paths["trna_fasta"], extract_feature_records(document, "tRNA"))
    _write_fasta(paths["rrna_fasta"], extract_feature_records(document, "rRNA"))
    if len(document.records) > 1:
        paths.update(_write_concatenated(document, directory / _CONCATENATED_DIR))
    _write_manifest(document, paths, directory)
    _validate_materialized(document, paths)
    return paths


def materialize_annotation(
    document: AnnotationDocument,
    output: str | Path,
) -> dict[str, Path]:
    """Atomically replace an output directory after complete validation."""

    report = validate_document(document, document.requested_stages)
    if not report.valid:
        codes = ",".join(issue.code for issue in report.errors)
        raise ValueError(f"annotation_validation_failed: {codes}")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    scratch = destination.parent / f".{destination.name}.t2a-{uuid4().hex}"
    backup = destination.parent / f".{destination.name}.backup-{uuid4().hex}"
    destination_moved = False
    extra: dict[str, Path] = {}
    try:
        _write_materialized_files(document, temporary)
        # Submission validation is best-effort: a missing binary or a broken
        # run must never block materialization, only the report is affected.
        try:
            from .table2asn import (
                run_table2asn_validation,
                write_validation_report,
            )

            validation = run_table2asn_validation(document, scratch)
            if validation is not None:
                write_validation_report(validation, temporary / "table2asn_validation.json")
                extra["table2asn_validation"] = destination / "table2asn_validation.json"
        except Exception:  # noqa: BLE001 - reported as an absent artifact, not a failure
            pass
        finally:
            if scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)
        if destination.exists():
            os.replace(destination, backup)
            destination_moved = True
        os.replace(temporary, destination)
    except Exception:
        if destination_moved and not destination.exists() and backup.exists():
            os.replace(backup, destination)
        if temporary.exists():
            shutil.rmtree(temporary)
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        raise
    if backup.exists():
        with suppress(OSError):
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
    concatenated = (
        {
            key: destination / _CONCATENATED_DIR / name
            for key, name in _CONCATENATED_NAMES.items()
        }
        if len(document.records) > 1
        else {}
    )
    return {
        **{key: destination / name for key, name in _OUTPUT_NAMES.items()},
        **extra,
        **concatenated,
    }


def _materialize_extraction_result(
    result: OrganelleResult,
    output: str | Path,
) -> OrganelleResult:
    sources = tuple((artifact, artifact.resolve()) for artifact in result.artifacts)
    if not sources:
        raise ValueError("annotation.write requires extraction artifacts")
    names = [path.name for _artifact, path in sources]
    if len(set(names)) != len(names):
        raise ValueError("annotation extraction artifact names must be unique")
    for artifact, path in sources:
        if not path.is_file() or _sha256(path) != artifact.sha256:
            raise ValueError("annotation extraction artifact changed after computation")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    backup = destination.parent / f".{destination.name}.backup-{uuid4().hex}"
    destination_moved = False
    try:
        temporary.mkdir()
        for _artifact, source in sources:
            shutil.copy2(source, temporary / source.name)
        if destination.exists():
            os.replace(destination, backup)
            destination_moved = True
        os.replace(temporary, destination)
    except Exception:
        if destination_moved and not destination.exists() and backup.exists():
            os.replace(backup, destination)
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    if backup.exists():
        with suppress(OSError):
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()

    artifacts = tuple(
        ArtifactRef.from_path(
            destination / source.name,
            kind=artifact.kind,
            format=artifact.format,
            media_type=artifact.media_type,
        )
        for artifact, source in sources
    )
    return OrganelleResult(
        operation_id="annotation.write",
        scope=result.scope,
        status=result.status,
        summary_text=f"Materialized {len(artifacts)} extracted annotation files.",
        metrics=FrozenMap(
            {
                "file_count": len(artifacts),
                "source_result_object_id": result.object_id,
                "source_result_status": result.status,
            }
        ),
        flags=tuple(dict.fromkeys(("annotation_materialized", *result.flags))),
        artifacts=artifacts,
        provenance=result.provenance,
    )


def materialize_result(result: OrganelleResult, output: str | Path) -> OrganelleResult:
    """Materialize an annotation artifact without rerunning annotation."""

    if result.status == "failed":
        raise ValueError("annotation.write requires a successful annotation result")
    if result.operation_id == "annotation.extract":
        return _materialize_extraction_result(result, output)
    source = next(
        (
            artifact
            for artifact in result.artifacts
            if artifact.kind == "annotation" and artifact.format == "json"
        ),
        None,
    )
    if source is None:
        raise ValueError("annotation.write requires a canonical annotation JSON artifact")
    document = load_document_json(source.resolve())
    paths = materialize_annotation(document, output)
    formats = {
        "json": ("annotation_json", "json", "application/json"),
        "genbank": ("annotation_genbank", "genbank", "text/plain"),
        "gff3": ("annotation_gff3", "gff3", "text/plain"),
        "tbl": ("annotation_tbl", "tbl", "text/plain"),
        "unedited_tbl": ("annotation_tbl_unedited", "tbl", "text/plain"),
        "fsa": ("annotation_fasta", "fasta", "text/plain"),
        "cds_fasta": ("annotation_cds", "fasta", "text/plain"),
        "protein_fasta": ("annotation_protein", "fasta", "text/plain"),
        "trna_fasta": ("annotation_trna", "fasta", "text/plain"),
        "rrna_fasta": ("annotation_rrna", "fasta", "text/plain"),
        "manifest": ("annotation_manifest", "json", "application/json"),
    }
    if len(document.records) > 1:
        formats.update(
            {
                "concat_genbank": ("annotation_concatenated_genbank", "genbank", "text/plain"),
                "concat_gff3": ("annotation_concatenated_gff3", "gff3", "text/plain"),
                "concat_tbl": ("annotation_concatenated_tbl", "tbl", "text/plain"),
                "concat_unedited_tbl": (
                    "annotation_concatenated_tbl_unedited",
                    "tbl",
                    "text/plain",
                ),
                "concat_fasta": ("annotation_concatenated_fasta", "fasta", "text/plain"),
                "concat_stat": ("annotation_concatenated_stat", "txt", "text/plain"),
            }
        )
    artifacts = tuple(
        ArtifactRef.from_path(
            paths[key],
            kind=kind,
            format=format_name,
            media_type=media_type,
        )
        for key, (kind, format_name, media_type) in formats.items()
    )
    if "table2asn_validation" in paths:
        artifacts = (
            *artifacts,
            ArtifactRef.from_path(
                paths["table2asn_validation"],
                kind="annotation_table2asn_validation",
                format="json",
                media_type="application/json",
            ),
        )
    return OrganelleResult(
        operation_id="annotation.write",
        scope=result.scope,
        status=result.status,
        summary_text=f"Materialized {len(artifacts)} annotation files.",
        metrics=FrozenMap(
            {
                "file_count": len(artifacts),
                "source_result_object_id": result.object_id,
                "source_result_status": result.status,
            }
        ),
        flags=tuple(dict.fromkeys(("annotation_materialized", *result.flags))),
        artifacts=artifacts,
        provenance=result.provenance,
    )
