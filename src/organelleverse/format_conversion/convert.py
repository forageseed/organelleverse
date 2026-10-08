"""GenBank format conversion. Self-contained (Biopython optional).

convert(): read a GenBank file and emit GFF3 / FASTA / tabular.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from ..annotation.genbank import parse_genbank
from ..annotation.models import AnnotationDocument
from ..core.frozen import FrozenMap, thaw_json
from ..core.genome import OrganelleGenome
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult
from .convert_core import document_to_gff3

_OPERATION_ID = "format_conversion.convert"
_OPERATION_VERSION = "1.0"
_METHOD = "organelleverse"


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _parameters_hash(parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        parameters,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance(genome: OrganelleGenome, parameters: dict[str, Any]) -> ResultProvenance:
    return ResultProvenance(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=(genome.object_id,),
        input_artifact_hashes=(
            (genome.annotation.sha256,) if genome.annotation is not None else ()
        ),
        parameters_hash=_parameters_hash(parameters),
        actual_backend=_METHOD,
        attempted_backends=(_METHOD,),
    )


def convert(
    genome: OrganelleGenome,
    *,
    formats: tuple[str, ...] = ("gff3", "fasta"),
) -> OrganelleResult:
    """Convert a GenBank-annotated genome to GFF3 and/or FASTA (content also inline in metrics).

    The converted text is embedded in ``metrics["files"]`` (full FASTA and
    GFF3 - large for whole genomes); persist through ``write_conversion``
    rather than copying metrics around.

    Requires ``genome.annotation`` (a GenBank artifact).
    """
    parameters = {"formats": list(formats)}
    if genome.annotation is None:
        return OrganelleResult(
            operation_id=_OPERATION_ID,
            operation_version=_OPERATION_VERSION,
            scope=genome.organelle,
            status="failed",
            summary_text="convert() needs a GenBank annotation artifact.",
            provenance=_provenance(genome, parameters),
            errors=(
                ErrorDetail(
                    code="input.missing_annotation_artifact",
                    message="convert() needs a GenBank annotation artifact.",
                    details={"genome_object_id": genome.object_id},
                ),
            ),
        )
    document = parse_genbank(genome.annotation.resolve())
    rendered = _render_conversion(document, formats)
    return OrganelleResult(
        operation_id=_OPERATION_ID,
        operation_version=_OPERATION_VERSION,
        scope=genome.organelle,
        status="ok",
        summary_text=(
            f"Prepared {len(document.records)} records for {', '.join(formats)} conversion."
        ),
        metrics=FrozenMap.from_json(
            {
                "records": len(document.records),
                "formats": list(formats),
                "files": rendered,
            }
        ),
        findings=(
            Finding(
                code="format_conversion.records", metric="records", value=len(document.records)
            ),
        ),
        flags=("conversion_ready",),
        artifacts=(),
        provenance=_provenance(genome, parameters),
    )


def write_conversion(
    result_or_genome: OrganelleResult | OrganelleGenome,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("gff3", "fasta"),
) -> dict[str, Path]:
    """Write converted GFF3/FASTA files from ``convert()`` or a genome."""
    result = (
        convert(result_or_genome, formats=formats)
        if isinstance(result_or_genome, OrganelleGenome)
        else result_or_genome
    )
    if result.status != "ok":
        raise ValueError(f"cannot write conversion output from {result.status!r} result")
    rendered = cast(dict[str, Any], thaw_json(result.metrics)).get("files")
    if not isinstance(rendered, dict):
        raise ValueError("write_conversion() requires a result produced by convert().")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for fmt, text in rendered.items():
        filename = "converted.gff3" if fmt == "gff3" else "converted.fasta"
        path = out / filename
        path.write_text(str(text))
        paths[str(fmt)] = path
    return paths


def _render_conversion(
    document: AnnotationDocument,
    formats: tuple[str, ...],
) -> dict[str, str]:
    rendered: dict[str, str] = {}
    if "gff3" in formats:
        lines = document_to_gff3(document)
        rendered["gff3"] = "\n".join(lines) + "\n"
    if "fasta" in formats:
        rendered["fasta"] = (
            "\n".join(f">{record.seqid}\n{record.sequence}" for record in document.records) + "\n"
        )
    return rendered
