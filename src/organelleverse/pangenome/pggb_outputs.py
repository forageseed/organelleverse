"""Capture real PGGB side products; validation levels are explicit and bounded."""

from __future__ import annotations

import gzip
from pathlib import Path

from ..core.artifacts import ArtifactRef
from .graph_formats import _check_encoding

_FORMATS = {
    "og": "pangenome_graph_index",
    "vcf": "pangenome_variants",
    "maf": "pangenome_alignment",
    "paf": "pangenome_alignment",
}


def related_artifacts(directory: Path) -> tuple[tuple[ArtifactRef, ...], list[dict]]:
    """Discover only backend-produced OG/VCF/MAF/PAF, including gzip text files.

    OG receives a format-header check, not an asserted graph equivalence check.
    Text checks validate headers/mandatory columns and gzip transport, not the
    biological validity of alignments or variant calls. An empty PAF is valid.
    """
    artifacts = []
    validations = []
    root = directory.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        compressed = path.suffix == ".gz"
        fmt = Path(path.stem).suffix[1:] if compressed else path.suffix[1:]
        if fmt not in _FORMATS or (fmt == "og" and compressed):
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"PGGB related artifact escapes backend output: {path.name}")
        if fmt == "og":
            _check_encoding(path, "og")
            level = "odgi_format_header"
        else:
            _validate_text(path, fmt, compressed)
            level = "text_header_and_mandatory_columns"
        artifacts.append(
            ArtifactRef.from_path(
                path,
                kind=_FORMATS[fmt],
                format=fmt,
                media_type="application/gzip"
                if compressed
                else ("application/octet-stream" if fmt == "og" else "text/plain"),
            )
        )
        validations.append(
            {
                "file": str(path.relative_to(root)),
                "format": fmt,
                "compression": "gzip" if compressed else None,
                "validation": level,
            }
        )
    return tuple(artifacts), validations


def _validate_text(path: Path, fmt: str, compressed: bool) -> None:
    opener = gzip.open if compressed else open
    signature = False
    columns = False
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split("\t")
            if fmt == "vcf":
                signature |= line.startswith("##fileformat=VCFv")
                if line.startswith("#CHROM\t"):
                    columns = fields[:8] == [
                        "#CHROM",
                        "POS",
                        "ID",
                        "REF",
                        "ALT",
                        "QUAL",
                        "FILTER",
                        "INFO",
                    ]
                elif not line.startswith("#") and (
                    not columns or len(fields) < 8 or int(fields[1]) < 1
                ):
                    raise ValueError(f"Invalid VCF record in {path.name}:{number}")
            elif fmt == "maf":
                signature |= line.startswith("##maf version=")
                if line.startswith("s ") or line.startswith("s\t"):
                    row = line.split()
                    if len(row) != 7 or row[4] not in {"+", "-"}:
                        raise ValueError(f"Invalid MAF sequence row in {path.name}:{number}")
                    start, size, total = map(int, (row[2], row[3], row[5]))
                    if (
                        start < 0
                        or size < 0
                        or start + size > total
                        or len(row[6].replace("-", "")) != size
                    ):
                        raise ValueError(f"Invalid MAF coordinates in {path.name}:{number}")
            elif fmt == "paf":
                if line.startswith("#"):
                    continue
                if len(fields) < 12 or fields[4] not in {"+", "-"}:
                    raise ValueError(f"Invalid PAF record in {path.name}:{number}")
                qlen, qs, qe, tlen, ts, te, matches, block, mapq = map(
                    int,
                    (
                        fields[1],
                        fields[2],
                        fields[3],
                        fields[6],
                        fields[7],
                        fields[8],
                        fields[9],
                        fields[10],
                        fields[11],
                    ),
                )
                if not (
                    0 <= qs <= qe <= qlen
                    and 0 <= ts <= te <= tlen
                    and 0 <= matches <= block
                    and 0 <= mapq <= 255
                ):
                    raise ValueError(f"Invalid PAF coordinates in {path.name}:{number}")
    if fmt == "vcf" and not (signature and columns):
        raise ValueError(f"Missing VCF header in {path.name}")
    if fmt == "maf" and not signature:
        raise ValueError(f"Missing MAF header in {path.name}")
