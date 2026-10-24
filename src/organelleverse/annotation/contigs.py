"""Map annotations between per-contig records and the 200-N merged sequence.

The mitochondrial pipeline annotates contigs joined by 200 N, so it works in
merged coordinates. Two directions live here:

- :func:`split_document_by_contigs` replaces the single merged record with one
  record per input contig, in contig-local coordinates. A feature whose parts
  fall on different contigs (a trans-spliced gene such as nad5 with exons on
  two contigs) cannot be one GenBank join, so each run of consecutive parts on
  one contig becomes a piece (a contig can hold two runs when the part between
  them lies on another contig). A CDS piece keeps the reading frame it has in the whole gene
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
        # Runs of consecutive parts on one contig. A contig can hold several runs
        # (nad5 exons 1-2 and 4-5 on one contig, exon 3 on another): each run is
        # its own piece, so every piece has one reading frame and open ends exactly
        # where the gene continues elsewhere.
        runs: list[tuple[int, int, list[tuple[int, LocationPart]]]] = []
        consumed = 0
        for position, part in enumerate(feature.parts):
            index = _locate(part, layout)
            if index is not None:
                _, offset, length = layout[index]
                located = (position, _shift(part, offset, length))
                if runs and runs[-1][0] == index and runs[-1][2][-1][0] == position - 1:
                    runs[-1][2].append(located)
                else:
                    runs.append((index, consumed, [located]))
            consumed += part.end - part.start
        if not runs:
            continue
        for run_no, (index, prefix, parts) in enumerate(runs):
            contig_id = layout[index][0]
            if len(runs) == 1:
                buckets[index].append(
                    feature.evolve(
                        feature_id=f"{contig_id}:{len(buckets[index])}:{feature.type}",
                        seqid=contig_id,
                        operator="single" if len(parts) == 1 else feature.operator,
                        parts=tuple(part for _, part in parts),
                    )
                )
            else:
                others = list(
                    dict.fromkeys(layout[i][0] for n, (i, _, _) in enumerate(runs) if n != run_no)
                )
                buckets[index].append(
                    _piece(
                        feature,
                        parts,
                        prefix,
                        len(feature.parts),
                        contig_id,
                        others,
                        contig_id,
                        len(buckets[index]),
                    )
                )
        if len(runs) > 1:
            cross.append(
                {
                    "gene": (feature.qualifier_values("gene") or ("",))[0],
                    "type": feature.type,
                    "contigs": list(dict.fromkeys(layout[i][0] for i, _, _ in runs)),
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


#: Longest copy of a circular molecule's start that is appended before the search.
_ORIGIN_FLANK = 15_000


def circular_flank(length: int) -> int:
    """Bases of a circular molecule's start to append: room for any gene across the origin."""
    return min(_ORIGIN_FLANK, length // 2)


def extend_circular_molecules(
    fasta_in: Path | str,
    fasta_out: Path | str,
    headers: list[tuple[str, str, bool | None]],
) -> list[int]:
    """Write ``fasta_out`` with each circular molecule followed by a copy of its first bases.

    The annotation search reads a linear sequence, so a gene that straddles the origin of a
    circular molecule is cut in two. With the start repeated at the end the gene is whole;
    :func:`fold_circular_flank` then turns it back into a join across the origin. Returns
    the flank length per record (0 for a molecule that is not circular).
    """
    records = list(SeqIO.parse(str(fasta_in), "fasta"))
    if len(records) != len(headers):
        raise ValueError("FASTA records and headers do not pair")
    flanks: list[int] = []
    with Path(fasta_out).open("w") as handle:
        for record, (_, _, circular) in zip(records, headers, strict=True):
            sequence = str(record.seq)
            flank = circular_flank(len(sequence)) if circular else 0
            flanks.append(flank)
            handle.write(f">{record.id}\n{sequence}{sequence[:flank]}\n")
    return flanks


def _wrap_part(part: LocationPart, length: int) -> list[LocationPart]:
    """The part in a circular molecule of ``length``, lower genomic piece first.

    A part that ends past ``length`` lies (wholly or partly) in the appended copy of the
    molecule start: the piece beyond ``length`` moves back to the start.
    """
    if part.end <= length:
        return [part]
    if part.start >= length:
        return [part.evolve(start=part.start - length, end=part.end - length)]
    return [
        part.evolve(end=length, end_status="exact"),
        part.evolve(start=0, end=part.end - length, start_status="exact"),
    ]


def _fold_feature(feature: AnnotationFeature, length: int) -> AnnotationFeature | None:
    """The feature inside a circular molecule, or None when it lies in the appended copy."""
    if min(p.start for p in feature.parts) >= length:
        return None
    if max(p.end for p in feature.parts) <= length:
        return feature
    parts: list[LocationPart] = []
    for part in feature.parts:
        pieces = _wrap_part(part, length)
        # biological order runs from the high coordinate to the low one on the minus strand
        parts.extend(pieces if part.strand == 1 else reversed(pieces))
    return feature.evolve(operator="join", parts=tuple(parts))


def _from_flank(feature: AnnotationFeature, length: int) -> AnnotationFeature | None:
    """A feature found only in the appended copy, moved back onto the molecule start.

    One that the end of the copy cut (an open end, or a ``partial`` qualifier) is an
    artifact of the extension, not a gene, and gives None.
    """
    if any(p.start_status != "exact" or p.end_status != "exact" for p in feature.parts):
        return None
    if any(q.name == "partial" for q in feature.qualifiers):
        return None
    return feature.evolve(
        parts=tuple(p.evolve(start=p.start - length, end=p.end - length) for p in feature.parts)
    )


def _covered_by(feature: AnnotationFeature, others: list[AnnotationFeature]) -> bool:
    """True when ``others`` holds the same gene over at least half of ``feature``."""
    return any(
        _name_key(feature) == _name_key(other) and _shared(feature, other) >= 0.5 * _covered(feature)
        for other in others
    )


def _name_key(feature: AnnotationFeature) -> tuple[str, tuple[str, ...]]:
    return feature.type, feature.qualifier_values("gene")


def _covered(feature: AnnotationFeature) -> int:
    return sum(p.end - p.start for p in feature.parts)


def _shared(a: AnnotationFeature, b: AnnotationFeature) -> int:
    total = 0
    for p in a.parts:
        for q in b.parts:
            total += max(0, min(p.end, q.end) - max(p.start, q.start))
    return total


def fold_circular_flank(
    document: AnnotationDocument, lengths: list[int | None]
) -> AnnotationDocument:
    """Fold the appended start copy of each circular record back onto the molecule.

    ``lengths`` gives the true length of each record, or None for a record that was not
    extended. A feature that runs past the end of the molecule becomes a join across the
    origin, and a copy of the same gene that the search left at the very start of the
    molecule (the cut remainder of a spanning gene) is dropped when the spanning gene already
    covers it. A feature wholly inside the copy moves back onto the molecule start, unless
    the original is already annotated there or the end of the copy cut it.
    """
    if len(document.records) != len(lengths) or all(n is None for n in lengths):
        return document
    records: list[AnnotationRecord] = []
    for record, length in zip(document.records, lengths, strict=True):
        if length is None or length >= len(record.sequence):
            records.append(record)
            continue
        folded: list[tuple[AnnotationFeature, str]] = []
        for original in record.features:
            if min(p.start for p in original.parts) >= length:
                moved = _from_flank(original, length)
                if moved is not None:
                    folded.append((moved, "flank"))
                continue
            feature = _fold_feature(original, length)
            if feature is not None:
                folded.append((feature, "wrapped" if feature is not original else "plain"))
        spanning = [feature for feature, kind in folded if kind == "wrapped"]
        anchors = [feature for feature, kind in folded if kind != "flank"]
        kept: list[AnnotationFeature] = []
        for feature, kind in folded:
            if kind == "flank":
                # the search may keep only the appended copy of a gene near the molecule
                # start; it stands in for the original unless that is already annotated
                if not _covered_by(feature, anchors):
                    kept.append(feature)
            elif kind == "wrapped" or not _covered_by(feature, spanning):
                kept.append(feature)
        records.append(
            record.evolve(sequence=record.sequence[:length], features=tuple(kept))
        )
    metadata = dict(document.source_metadata)
    if "contig_map" in metadata:
        metadata["contig_map"] = _contig_map(merged_layout(tuple(records)))
    return document.evolve(records=tuple(records), source_metadata=metadata)
