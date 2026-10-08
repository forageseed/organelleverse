"""Export what was fetched: metadata as CSV, sequences as FASTA.

Both are streamed. A whole-corpus pull is 58,082 records and the GenBank flat
file for it is gigabytes; nothing here holds a corpus in memory.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "genbank_to_fasta",
    "merge_metadata_csv",
    "write_metadata_csv",
]

# The columns worth having in front of a human, in a sensible reading order.
_PREFERRED = (
    "accession",
    "organism",
    "taxid",
    "length",
    "gene_count",
    "protein_count",
    "ambiguous_count",
    "collection_date",
    "geo_location",
    "isolate",
    "cultivar",
    "create_date",
    "update_date",
    "status",
    "submitter_institution",
    "submitter_country",
    "title",
)


def _columns(records: Sequence[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    for record in records:
        seen.update(record)
    ordered = [c for c in _PREFERRED if c in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def write_metadata_csv(records: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write the metadata table (gget's ``save_metadata_to_csv``)."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    columns = _columns(records)
    with file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(records, key=lambda r: str(r.get("accession", ""))):
            writer.writerow(record)
    return file


def merge_metadata_csv(first: str | Path, second: str | Path, out: str | Path) -> Path:
    """Union two metadata tables on accession; the second wins on collision."""
    rows: dict[str, dict[str, Any]] = {}
    for source in (first, second):
        file = Path(source)
        if not file.is_file():
            continue
        with file.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                accession = (row.get("accession") or "").strip()
                if accession:
                    rows[accession] = {**rows.get(accession, {}), **row}
    return write_metadata_csv([rows[k] for k in sorted(rows)], out)


def _genbank_records(handle: TextIO) -> Iterator[tuple[str, str]]:
    """Yield ``(accession, sequence)`` from a GenBank flat file, streaming.

    Deliberately not Biopython: this module must stay import-light, and the two
    fields needed here (VERSION and ORIGIN) are unambiguous in the flat format.
    """
    accession = ""
    chunks: list[str] = []
    in_origin = False

    for line in handle:
        if line.startswith("VERSION"):
            parts = line.split()
            accession = parts[1] if len(parts) > 1 else ""
        elif line.startswith("ORIGIN"):
            in_origin = True
            chunks = []
        elif line.startswith("//"):
            if accession and chunks:
                yield accession, "".join(chunks)
            accession, chunks, in_origin = "", [], False
        elif in_origin:
            # "        1 atgcatgcat gcatgcatgc" -> strip the coordinate and spaces
            chunks.append("".join(ch for ch in line if ch.isalpha()))


def genbank_to_fasta(
    genbank_path: str | Path,
    fasta_path: str | Path,
    *,
    accessions: Iterable[str] | None = None,
    line_width: int = 60,
) -> int:
    """Stream a GenBank flat file out as FASTA. Returns the record count.

    ``accessions`` restricts the output to a subset without re-downloading —
    gget's ``_stream_copy_fasta`` with an accession set.
    """
    wanted = {a.strip() for a in accessions} if accessions is not None else None
    source = Path(genbank_path)
    target = Path(fasta_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with (
        source.open(encoding="utf-8", errors="replace") as reader,
        target.open("w", encoding="utf-8") as writer,
    ):
        for accession, sequence in _genbank_records(reader):
            if wanted is not None and accession not in wanted:
                continue
            writer.write(f">{accession}\n")
            for start in range(0, len(sequence), line_width):
                writer.write(sequence[start : start + line_width].upper() + "\n")
            written += 1
    return written
