"""GenBank features → drawable loci, shared by the OGDraw map and the structure map.

Both maps read the same file the same way: CDS/tRNA/rRNA joins supply exons in
transcript order and supersede their parent ``gene`` span, every copy of a
duplicated gene (plastid IR copies, repeated mitochondrial tRNAs) is kept, and
each exon keeps its own strand. Trans-splicing comes from ``/trans_splicing`` on
the product or its gene when the file says so; otherwise it is inferred from the
exon layout (mixed strands, a backward jump, or a gap longer than any cis intron)
and the row records which test fired.
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..annotation.splice_layout import MAX_CIS_INTRON, trans_splicing_evidence
from ..core.errors import OrganelleInputError, OrganelleParameterError
from .ogdraw import Gene, classify_gene

PRODUCT_TYPES = ("CDS", "tRNA", "rRNA")

#: Human-readable reason for each ``trans_basis`` value, used in panel titles.
TRANS_BASIS_TEXT = {
    "qualifier": "/trans_splicing",
    "mixed_strand": "inferred: exons on both strands",
    "backward": "inferred: exon order runs backwards",
    "distance": f"inferred: exons > {MAX_CIS_INTRON // 1000} kb apart",
}


def _same_locus_id(feature, gene) -> bool:
    tag = feature.qualifiers.get("locus_tag")
    if tag:
        return tag == gene.qualifiers.get("locus_tag")
    return feature.qualifiers.get("gene") == gene.qualifiers.get("gene")


def parents(feature, genes):
    """Gene features of this product's locus, including duplicated genes and mixed strands.

    A gene matches when its parts hold every product part on the same strand. A
    gene written as one plain span over a split locus (older OrganelleVerse
    output and several other annotators write ``gene 92819..163357`` over a
    mixed-strand nad2 CDS) matches when that span holds every part.
    """
    parts = list(feature.location.parts)
    found = []
    for gene in genes:
        if not _same_locus_id(feature, gene):
            continue
        gene_parts = list(gene.location.parts)
        nested = all(
            any(p.start >= q.start and p.end <= q.end and p.strand == q.strand for q in gene_parts)
            for p in parts
        )
        spanning = len(gene_parts) == 1 and all(
            p.start >= gene_parts[0].start and p.end <= gene_parts[0].end for p in parts
        )
        if nested or spanning:
            found.append(gene)
    return found


def annotation_rows(record, organelle: str = "mito", *, strict: bool = True) -> list[dict[str, Any]]:
    """One row per drawable locus: products, plus genes no product covers.

    ``strict`` (structure maps) rejects remote or unstranded locations because
    the exon table must be exact. The plain OGDraw map passes ``False``: it
    draws an unstranded part on the plus strand and skips a remote part or a
    feature left with no local part.
    """
    genes = [f for f in record.features if f.type == "gene"]
    products = [f for f in record.features if f.type in PRODUCT_TYPES]
    covered = {id(g) for f in products for g in parents(f, genes)}
    length = len(record)
    rows = []
    for index, feature in enumerate(record.features):
        if feature.type not in PRODUCT_TYPES and not (
            feature.type == "gene" and id(feature) not in covered
        ):
            continue
        name = feature.qualifiers.get("gene", feature.qualifiers.get("locus_tag", [feature.type]))[
            0
        ]
        parts = list(feature.location.parts)
        if any(p.ref is not None or p.strand not in (-1, 1) for p in parts):
            if strict:
                raise OrganelleInputError(
                    code="visualization.structure.location",
                    message="Structure maps require local, stranded feature locations.",
                )
            from Bio.SeqFeature import SimpleLocation  # type: ignore

            parts = [
                SimpleLocation(p.start, p.end, strand=-1 if p.strand == -1 else 1)
                for p in parts
                if p.ref is None
            ]
            if not parts:
                continue
        basis = None
        if feature.type != "gene":
            if any("trans_splicing" in f.qualifiers for f in [feature, *parents(feature, genes)]):
                basis = "qualifier"
            else:
                basis = trans_splicing_evidence(
                    [(int(p.start), int(p.end), p.strand) for p in parts],
                    length,
                    max_cis_gap=MAX_CIS_INTRON,
                )
        # GenBank join order is biological order, including complement and trans-splicing.
        exons = [
            {"exon": i, "start": int(p.start) + 1, "end": int(p.end), "strand": p.strand}
            for i, p in enumerate(parts, 1)
        ]
        gaps = [
            ((b["start"] - a["end"] - 1) if a["strand"] == 1 else (a["start"] - b["end"] - 1))
            % length
            for a, b in pairwise(exons)
        ]
        # A contiguous join across the circular origin is not a splice junction.
        splicing = "trans" if basis else "cis" if any(gaps) else "none"
        if feature.type == "gene":
            splicing = "none"
        rows.append(
            {
                "feature_index": index,
                "gene": name,
                "feature_type": feature.type,
                "category": classify_gene(name, organelle),
                "splicing": splicing,
                "trans_basis": basis if splicing == "trans" else None,
                "exons": exons,
                "gaps": gaps if splicing == "cis" else [],
            }
        )
    return rows


def genes_from_rows(rows) -> list[Gene]:
    return [
        Gene(
            name=row["gene"],
            start=min(e["start"] for e in row["exons"]),
            end=max(e["end"] for e in row["exons"]),
            strand=row["exons"][0]["strand"],
            has_intron=row["splicing"] != "none",
            category=row["category"],
            exons=[(e["start"], e["end"]) for e in row["exons"]],
            exon_strands=[e["strand"] for e in row["exons"]],
            trans_spliced=row["splicing"] == "trans",
        )
        for row in rows
    ]


def resolve_ir_regions(record, organelle: str, show_ir: bool | None) -> dict[str, Any]:
    """LSC/IRb/SSC/IRa spans for a plastid record (0-based start, length).

    ``None`` draws the partition when a plastid record has one and skips it
    silently otherwise; ``True`` fails explicitly when it cannot be resolved;
    ``False`` never draws it.
    """
    if show_ir is False or (show_ir is None and organelle != "plastid"):
        return {}
    from ..comparative.orientation import _quadripartite

    if organelle != "plastid":
        raise OrganelleParameterError(
            code="visualization.structure.ir_organelle",
            message="IR partitioning requires a plastid record.",
        )
    regions = _quadripartite(str(record.seq).upper())
    if regions is None and show_ir:
        raise OrganelleInputError(
            code="visualization.structure.ir_unresolved",
            message="No quadripartite plastome structure detected.",
        )
    return regions or {}


#: Whole words only: "mitochondrialis" is part of a species name, "mitochondrion" is not.
_ORGANELLE_WORD = re.compile(
    r"\b(chloroplast|plastid|plastome|mitochondrion|mitochondrial|mitochondria|mitogenome|organelle)\b",
    re.IGNORECASE,
)


def record_display_name(record, fallback: str) -> str:
    """A species-style name for a record: /organism, else the DEFINITION's organism, else ``fallback``.

    OrganelleVerse writes ``ORGANISM .`` and a DEFINITION such as ``Amborella
    trichopoda chloroplast genome annotated by OrganelleVerse.``; the words before
    the organelle keyword are the species. A file stem such as ``annotation``
    names nothing, so it is only the last resort.
    """
    source = next((f for f in record.features if f.type == "source"), None)
    candidates = [
        *(source.qualifiers.get("organism", []) if source is not None else []),
        record.annotations.get("organism", ""),
    ]
    for value in candidates:
        value = str(value).strip().strip(".").strip()
        if value and value.lower() not in {"unknown", "unknown organism"}:
            return value.replace("_", " ")
    definition = str(getattr(record, "description", "") or "")
    match = _ORGANELLE_WORD.search(definition)
    if match and match.start() > 0:
        species = definition[: match.start()].strip(" ,.;:").replace("_", " ")
        if species:
            return species
    return fallback


def genbank_display_name(genbank_path) -> str:
    """Display name for one GenBank file (its record's species) or a count of several."""
    from Bio import SeqIO  # type: ignore

    paths = [Path(genbank_path)] if isinstance(genbank_path, (str, Path)) else [
        Path(p) for p in genbank_path
    ]
    if len(paths) != 1:
        return f"{len(paths)} GenBank files"
    try:
        records = list(SeqIO.parse(str(paths[0]), "genbank"))
    except (OSError, ValueError):
        return paths[0].stem
    if len(records) != 1:
        return paths[0].stem
    return record_display_name(records[0], paths[0].stem)
