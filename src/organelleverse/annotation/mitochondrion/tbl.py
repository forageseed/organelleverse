"""NCBI five-column feature table, the format GenBank submissions are made in.

GFF and GenBank flatfile describe an annotation; the five-column table is what
``table2asn`` consumes to produce a submission. Without it the output of this
pipeline cannot be deposited, which is the difference between an annotation a
person can read and one a database will accept -- and PMGA emits it, so parity
requires it.

The format is deceptively plain and unforgiving in three places, each of which
silently produces a table that loads but is wrong:

* Coordinates are given 5' to 3' *as transcribed*, so a gene on the minus strand
  is written with its start greater than its end. Sorting the interval the usual
  way loses the strand entirely.
* Partial ends are marked by ``<`` and ``>`` on the coordinate, not by a
  qualifier, and they mean 5' and 3' *of the feature*, which for a minus-strand
  gene is the right and left ends of the interval respectively.
* A pseudogene carries ``/pseudo`` and no ``/product``; leaving the product in
  makes the validator reject the record.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import AnnotationDocument, AnnotationFeature
from .models.feature import rRNAAnnotation, tRNAAnnotation
from .models.gene import ExonRecord, GeneAnnotation, Strand
from .models.genome import GenomeSequence


def _intervals(ann: GeneAnnotation | tRNAAnnotation | rRNAAnnotation) -> list[tuple[int, int]]:
    """Exon intervals in transcription order, each written 5' to 3'."""
    exons = getattr(ann, "exons", None)
    spans = sorted((e.start, e.end) for e in exons) if exons else [(ann.start, ann.end)]
    minus = getattr(ann, "strand", Strand.PLUS) == Strand.MINUS
    if minus:
        return [(b, a) for a, b in reversed(spans)]
    return spans


def _mark_partial(spans: list[tuple[int, int]], five: bool, three: bool) -> list[tuple[str, str]]:
    """Apply ``<``/``>`` to the first and last coordinate of the feature.

    The marks belong to the feature's own 5' and 3' ends, which is why they are
    applied after the intervals are already in transcription order rather than to
    the lower and higher genome coordinates.
    """
    out = [(str(a), str(b)) for a, b in spans]
    if five and out:
        out[0] = (f"<{out[0][0]}", out[0][1])
    if three and out:
        out[-1] = (out[-1][0], f">{out[-1][1]}")
    return out


def _feature_block(
    kind: str, spans: list[tuple[str, str]], qualifiers: list[tuple[str, str]]
) -> list[str]:
    lines = [f"{spans[0][0]}\t{spans[0][1]}\t{kind}"]
    lines += [f"{a}\t{b}" for a, b in spans[1:]]
    lines += [f"\t\t\t{key}\t{value}".rstrip() for key, value in qualifiers]
    return lines


def _gene_qualifiers(name: str) -> list[tuple[str, str]]:
    return [("gene", name)]


def _annotation_lines(
    annotations: list[GeneAnnotation],
    trna_annotations: list[tRNAAnnotation],
    rrna_annotations: list[rRNAAnnotation],
    *,
    unedited: bool = False,
) -> list[str]:
    """Feature-block lines for one record, shared by both table renderers."""
    lines: list[str] = []

    for ann in sorted(annotations, key=lambda a: a.genomic_start):
        spans = _intervals(ann)
        marked = _mark_partial(spans, ann.is_partial_5prime, ann.is_partial_3prime)
        lines += _feature_block("gene", marked, _gene_qualifiers(ann.gene_name))

        if ann.is_pseudo:
            # a pseudogene declares /pseudo and no product; both is a validator error
            lines += _feature_block("CDS", marked, [("gene", ann.gene_name), ("pseudo", "")])
            continue

        quals = [("gene", ann.gene_name)]
        if ann.product:
            quals.append(("product", ann.product))
        quals.append(("transl_table", str(ann.transl_table)))
        if not unedited:
            for exc in ann.exceptions:
                quals.append(("exception", exc))
        for note in ann.notes:
            quals.append(("note", note))
        lines += _feature_block("CDS", marked, quals)

    for trna in sorted(trna_annotations, key=lambda a: a.genomic_start):
        spans = _mark_partial(_intervals(trna), False, False)
        name = getattr(trna, "gene_name", None) or getattr(trna, "name", "tRNA")
        product = getattr(trna, "product", "") or f"tRNA-{getattr(trna, 'amino_acid', 'Xxx')}"
        lines += _feature_block("gene", spans, _gene_qualifiers(name))
        lines += _feature_block("tRNA", spans, [("gene", name), ("product", product)])

    for rrna in sorted(rrna_annotations, key=lambda a: a.genomic_start):
        spans = _mark_partial(_intervals(rrna), False, False)
        name = getattr(rrna, "gene_name", None) or getattr(rrna, "name", "rRNA")
        product = getattr(rrna, "product", "") or name
        lines += _feature_block("gene", spans, _gene_qualifiers(name))
        lines += _feature_block("rRNA", spans, [("gene", name), ("product", product)])

    return lines


def render_tbl(
    annotations: list[GeneAnnotation],
    trna_annotations: list[tRNAAnnotation],
    rrna_annotations: list[rRNAAnnotation],
    genome: GenomeSequence,
    output_path: Path,
    *,
    unedited: bool = False,
) -> None:
    """Write the five-column feature table for ``table2asn``.

    ``unedited`` emits the table without RNA-editing exceptions, which is the
    pair of files PMGA produces (``.tbl`` and ``.unedited.tbl``): C-to-U editing
    creates initiators and terminators that are absent from the DNA, so a
    submission has to be able to state both what the genome says and what the
    transcript does.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f">Feature {genome.seqid}"]
    lines += _annotation_lines(annotations, trna_annotations, rrna_annotations, unedited=unedited)
    output_path.write_text("\n".join(lines) + "\n")


def _document_strand(feature: AnnotationFeature) -> Strand:
    return Strand(feature.parts[0].strand)


def _document_exons(feature: AnnotationFeature) -> list[ExonRecord]:
    """Location parts (zero-based, half-open) as 1-based inclusive exons.

    Parts are stored in biological (transcription) order; the renderer sorts
    and re-derives that order per strand, so cis-spliced features round-trip
    exactly.
    """
    strand = _document_strand(feature)
    return [
        ExonRecord(start=part.start + 1, end=part.end, strand=strand, number=number)
        for number, part in enumerate(feature.parts, 1)
    ]


def _document_partial(feature: AnnotationFeature) -> tuple[bool, bool]:
    """(5' partial, 3' partial) from the terminal parts' position statuses.

    Parts are in biological order, so the 5' end of the feature is the first
    part's left coordinate on the plus strand but its right coordinate on the
    minus strand; ``before``/``after`` are the document's ``<``/``>`` marks.
    """
    first, last = feature.parts[0], feature.parts[-1]
    if first.strand == 1:
        return first.start_status == "before", last.end_status == "after"
    return first.end_status == "after", last.start_status == "before"


def _first(values: tuple[str, ...], default: str) -> str:
    return values[0] if values else default


def _document_gene(feature: AnnotationFeature) -> GeneAnnotation:
    partial_5, partial_3 = _document_partial(feature)
    transl_table = feature.qualifier_values("transl_table")
    return GeneAnnotation(
        gene_name=_first(feature.qualifier_values("gene"), feature.feature_id),
        product=_first(feature.qualifier_values("product"), ""),
        exons=_document_exons(feature),
        strand=_document_strand(feature),
        notes=list(feature.qualifier_values("note")),
        exceptions=list(feature.qualifier_values("exception")),
        transl_table=int(transl_table[0]) if transl_table else 1,
        is_pseudo=any(qualifier.name == "pseudo" for qualifier in feature.qualifiers),
        is_partial_5prime=partial_5,
        is_partial_3prime=partial_3,
    )


def _document_trna(feature: AnnotationFeature) -> tRNAAnnotation:
    product = _first(feature.qualifier_values("product"), "")
    amino_acid = product.split("-", 1)[1] if product.startswith("tRNA-") else "Xxx"
    return tRNAAnnotation(
        gene_name=_first(feature.qualifier_values("gene"), feature.feature_id),
        product=product,
        exons=_document_exons(feature),
        strand=_document_strand(feature),
        anticodon=_first(feature.qualifier_values("anticodon"), ""),
        amino_acid=amino_acid,
    )


def _document_rrna(feature: AnnotationFeature) -> rRNAAnnotation:
    product = _first(feature.qualifier_values("product"), "")
    gene_name = _first(feature.qualifier_values("gene"), feature.feature_id)
    match = re.search(r"(\d+S)", product) or re.search(r"(\d+S)", gene_name)
    return rRNAAnnotation(
        gene_name=gene_name,
        product=product,
        exons=_document_exons(feature),
        strand=_document_strand(feature),
        rrna_type=match.group(1) if match else "",
    )


def render_tbl_document(
    document: AnnotationDocument,
    output_path: str | Path,
    *,
    unedited: bool = False,
) -> None:
    """Render the five-column table from a canonical annotation document.

    This is the capability write path's entry point: ``materialize_annotation``
    holds an :class:`AnnotationDocument`, not backend model objects, so the
    document's CDS/tRNA/rRNA features are adapted and then rendered by the same
    block builder as :func:`render_tbl` — the format traps (minus-strand 5'->3'
    order, partial-end marks, pseudogene qualifiers) live in exactly one place.
    Standalone ``gene`` and ``Protein`` features are skipped: the gene block is
    derived from each CDS/tRNA/rRNA feature, which is what carries product,
    transl_table, exception, and partial information. One ``>Feature`` section
    is emitted per record.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in document.records:
        genes: list[GeneAnnotation] = []
        trnas: list[tRNAAnnotation] = []
        rrnas: list[rRNAAnnotation] = []
        for feature in record.features:
            if feature.type == "CDS":
                genes.append(_document_gene(feature))
            elif feature.type == "tRNA":
                trnas.append(_document_trna(feature))
            elif feature.type == "rRNA":
                rrnas.append(_document_rrna(feature))
        lines.append(f">Feature {record.seqid}")
        lines += _annotation_lines(genes, trnas, rrnas, unedited=unedited)
    destination.write_text("\n".join(lines) + "\n")
