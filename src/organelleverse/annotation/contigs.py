"""Map annotations between per-contig records and the 200-N merged sequence.

The mitochondrial pipeline annotates contigs joined by 200 N, so it works in
merged coordinates. Two directions live here:

- :func:`split_document_by_contigs` replaces the single merged record with one
  record per input contig, in contig-local coordinates. A feature whose parts
  fall on different contigs (a trans-spliced gene such as nad5 with exons on
  two contigs) cannot be one GenBank join, so each contig receives the parts it
  holds. A CDS piece keeps the reading frame it has in the whole gene
  (``codon_start``), is partial at the end where the gene continues on another
  contig, and carries a note naming those contigs.
- :func:`merge_document_by_contigs` concatenates per-contig records back into
  one ``merged_N_contigs`` record (PMGA-style concatenation): coordinates are
  shifted onto the joined sequence and every feature stays one entry per
  contig. :func:`merged_stat_lines` renders the contig table that accompanies
  such a concatenation.
"""

from __future__ import annotations

import re
from pathlib import Path

from Bio import SeqIO

from .models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)

#: Spacer ``fasta._merge_contigs`` puts between contigs.
GAP = 200

_PIECE_DROPPED = frozenset({"translation", "codon_start", "partial"})


def contig_layout(fasta_path: Path | str) -> list[tuple[str, int, int]]:
    """``(original_id, merged_start, length)`` per contig, 0-based; empty for one contig."""
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if len(records) < 2:
        return []
    layout: list[tuple[str, int, int]] = []
    cursor = 0
    for record in records:
        layout.append((record.id, cursor, len(record.seq)))
        cursor += len(record.seq) + GAP
    return layout


def merged_layout(records: tuple[AnnotationRecord, ...]) -> list[tuple[str, int, int]]:
    """``(seqid, merged_start, length)`` per record with GAP spacers, 0-based."""
    layout: list[tuple[str, int, int]] = []
    cursor = 0
    for index, record in enumerate(records):
        length = len(record.sequence)
        layout.append((record.seqid, cursor, length))
        cursor += length + (GAP if index < len(records) - 1 else 0)
    return layout


def _contig_map(layout: list[tuple[str, int, int]]) -> tuple[dict[str, object], ...]:
    return tuple(
        {"id": contig_id, "merged_start": start + 1, "merged_end": start + length, "length": length}
        for contig_id, start, length in layout
    )


def _locate(part: LocationPart, layout: list[tuple[str, int, int]]) -> int | None:
    """Index of the contig holding the part's first base; None inside a spacer."""
    for index, (_, start, length) in enumerate(layout):
        if start <= part.start < start + length:
            return index
    return None


def _shift(part: LocationPart, offset: int, length: int) -> LocationPart:
    """Contig-local part, clipped to the contig."""
    return part.evolve(start=part.start - offset, end=min(part.end - offset, length))


def _open_end(part: LocationPart, five_prime: bool) -> LocationPart:
    """Mark the 5' or 3' end of a part as running on (``<`` / ``>``)."""
    left = (part.strand == 1) == five_prime
    if left:
        return part.evolve(start_status="before")
    return part.evolve(end_status="after")


def _piece(
    feature: AnnotationFeature,
    parts: list[tuple[int, LocationPart]],
    prefix: int,
    total: int,
    here: str,
    others: list[str],
    contig_id: str,
    ordinal: int,
) -> AnnotationFeature:
    """The part of ``feature`` held by one contig, flagged as a trans-spliced piece."""
    indices = [i for i, _ in parts]
    located = [part for _, part in parts]
    open_5 = indices[0] != 0
    open_3 = indices[-1] != total - 1
    if open_5:
        located[0] = _open_end(located[0], five_prime=True)
    if open_3:
        located[-1] = _open_end(located[-1], five_prime=False)
    kept = [q for q in feature.qualifiers if q.name not in _PIECE_DROPPED | {"note"}]
    names = {q.name for q in kept}
    if "trans_splicing" not in names:
        kept.append(FeatureQualifier(name="trans_splicing", values=("",)))
    if feature.type == "CDS":
        if "exception" not in names:
            kept.append(FeatureQualifier(name="exception", values=("trans-splicing",)))
        original = int((feature.qualifier_values("codon_start") or ("1",))[0])
        kept.append(
            FeatureQualifier(name="codon_start", values=(str((original - 1 - prefix) % 3 + 1),))
        )
        if open_5 or open_3:
            kept.append(FeatureQualifier(name="partial", values=("",)))
    # Qualifiers are written by name, so the new note extends any existing ones.
    kept.append(
        FeatureQualifier(
            name="note",
            values=(
                *feature.qualifier_values("note"),
                f"trans-spliced across contigs: {len(parts)} of {total} parts are on {here}; "
                f"the others are on {', '.join(others)}",
            ),
        )
    )
    return feature.evolve(
        feature_id=f"{contig_id}:{ordinal}:{feature.type}",
        seqid=contig_id,
        operator="single" if len(located) == 1 else feature.operator,
        parts=tuple(located),
        qualifiers=tuple(sorted(kept, key=lambda q: q.name)),
    )


def split_document_by_contigs(
    document: AnnotationDocument, layout: list[tuple[str, int, int]]
) -> AnnotationDocument:
    """Replace the merged record with one record per contig in ``layout``."""
    if not layout or len(document.records) != 1:
        return document
    merged = document.records[0]
    buckets: list[list[AnnotationFeature]] = [[] for _ in layout]
    cross: list[dict[str, object]] = []
    for feature in merged.features:
        placed: dict[int, list[tuple[int, LocationPart]]] = {}
        prefixes: dict[int, int] = {}
        consumed = 0
        for position, part in enumerate(feature.parts):
            index = _locate(part, layout)
            if index is not None:
                _, offset, length = layout[index]
                placed.setdefault(index, []).append((position, _shift(part, offset, length)))
                prefixes.setdefault(index, consumed)
            consumed += part.end - part.start
        if not placed:
            continue
        for index, parts in placed.items():
            contig_id = layout[index][0]
            if len(placed) == 1:
                buckets[index].append(
                    feature.evolve(
                        feature_id=f"{contig_id}:{len(buckets[index])}:{feature.type}",
                        seqid=contig_id,
                        operator="single" if len(parts) == 1 else feature.operator,
                        parts=tuple(part for _, part in parts),
                    )
                )
            else:
                buckets[index].append(
                    _piece(
                        feature,
                        parts,
                        prefixes[index],
                        len(feature.parts),
                        contig_id,
                        [layout[i][0] for i in placed if i != index],
                        contig_id,
                        len(buckets[index]),
                    )
                )
        if len(placed) > 1:
            cross.append(
                {
                    "gene": (feature.qualifier_values("gene") or ("",))[0],
                    "type": feature.type,
                    "contigs": [layout[i][0] for i in placed],
                }
            )
    records = tuple(
        AnnotationRecord(
            seqid=contig_id,
            name=contig_id,
            description=merged.description,
            sequence=merged.sequence[start : start + length],
            features=tuple(buckets[index]),
        )
        for index, (contig_id, start, length) in enumerate(layout)
    )
    metadata = dict(document.source_metadata)
    metadata["contig_map"] = _contig_map(layout)
    metadata["cross_contig_features"] = tuple(cross)
    return document.evolve(records=records, source_metadata=metadata)


def merge_document_by_contigs(document: AnnotationDocument) -> AnnotationDocument:
    """Concatenate per-contig records back into one merged record with GAP spacers.

    The inverse of :func:`split_document_by_contigs` on coordinates only: each
    feature is shifted onto the merged sequence as-is, so a gene that was split
    across contigs stays one piece per contig rather than being re-joined into
    one location. A single-record document is returned unchanged.
    """
    if len(document.records) < 2:
        return document
    seqid = f"merged_{len(document.records)}_contigs"
    layout = merged_layout(document.records)
    features: list[AnnotationFeature] = []
    for record, (_, offset, _) in zip(document.records, layout, strict=True):
        for feature in record.features:
            features.append(
                feature.evolve(
                    feature_id=f"{seqid}:{len(features)}:{feature.type}",
                    seqid=seqid,
                    parts=tuple(
                        part.evolve(start=part.start + offset, end=part.end + offset)
                        for part in feature.parts
                    ),
                )
            )
    pieces: list[str] = []
    for index, record in enumerate(document.records):
        pieces.append(record.sequence)
        if index < len(document.records) - 1:
            pieces.append("N" * GAP)
    sequence = "".join(pieces)
    merged = AnnotationRecord(
        seqid=seqid,
        name=seqid,
        description=document.records[0].description,
        sequence=sequence,
        features=tuple(features),
    )
    metadata = dict(document.source_metadata)
    metadata["contig_map"] = _contig_map(layout)
    return document.evolve(records=(merged,), source_metadata=metadata)


def merged_stat_lines(document: AnnotationDocument) -> list[str]:
    """PMGA-style table of where each contig sits in the merged sequence.

    Accepts either the per-contig document (records are laid out with spacers)
    or its :func:`merge_document_by_contigs` product (the recorded
    ``contig_map`` is reused). Returns ``[]`` when there is nothing to merge.
    """
    if len(document.records) > 1:
        spans: list[tuple[str, int, int]] = merged_layout(document.records)
        entries = (
            {"id": cid, "merged_start": start + 1, "merged_end": start + length, "length": length}
            for cid, start, length in spans
        )
    else:
        recorded = document.source_metadata.get("contig_map")
        if not recorded:
            return []
        entries = recorded
    rows = ["contig_id\tlength\tmerged_start\tmerged_end"]
    for entry in entries:
        rows.append(
            f"{entry['id']}\t{entry['length']}\t{entry['merged_start']}\t{entry['merged_end']}"
        )
    return rows


_CIRCULAR = re.compile(r"circular=(true|false)")


def molecule_headers(fasta_path: Path | str) -> list[tuple[str, str, bool | None]]:
    """``(id, description, circular)`` per FASTA record.

    ``circular`` comes from the ``circular=true|false`` field that ``ovasm linearize``
    writes in every molecule header (``>linear.1 circular=true length=328872 path=...``);
    it is None for any other FASTA, whose headers carry no topology.
    """
    headers: list[tuple[str, str, bool | None]] = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        description = record.description.removeprefix(record.id).strip()
        found = _CIRCULAR.search(description)
        headers.append((record.id, description, None if found is None else found.group(1) == "true"))
    return headers


def apply_molecule_headers(
    document: AnnotationDocument, headers: list[tuple[str, str, bool | None]]
) -> AnnotationDocument:
    """Carry each ovasm molecule's topology and header onto its annotation record.

    Records pair with the FASTA records in order (a multi-contig document keeps the input
    order). Only headers that state a topology are applied, so an ordinary FASTA keeps the
    definition line the pipeline wrote. The topology goes to ``source_metadata
    ["record_topology"]`` (seqid -> ``circular``/``linear``), which the GenBank writer reads.
    """
    if len(document.records) != len(headers) or all(h[2] is None for h in headers):
        return document
    topology: dict[str, str] = {}
    records: list[AnnotationRecord] = []
    for record, (_, description, circular) in zip(document.records, headers, strict=True):
        if circular is None:
            records.append(record)
            continue
        topology[record.seqid] = "circular" if circular else "linear"
        records.append(record.evolve(description=description))
    metadata = dict(document.source_metadata)
    metadata["record_topology"] = topology
    return document.evolve(records=tuple(records), source_metadata=metadata)
