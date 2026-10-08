"""Add sequence-only ORF candidate CDSs to the canonical annotation document."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .._bio import write_fasta
from ..core.errors import OrganelleInputError
from ..core.frozen import FrozenMap, thaw_json
from .models import AnnotationDocument, AnnotationFeature, FeatureQualifier, LocationPart
from .orfs import find_orfs


def _location(parts):
    return tuple((p.start, p.end, p.strand) for p in parts)


def add_orf_features(
    document: AnnotationDocument,
    *,
    scratch: Path,
    organelle: Literal["mitochondrion", "plastid"],
    genetic_code: int,
    min_aa: int,
    circular: bool,
) -> AnnotationDocument:
    """Preserve all existing features and suppress only identical CDS locations.

    Scan the exact canonical record sequences, so backend-renamed record IDs are
    respected. Retain other overlapping candidates, with overlap evidence rather
    than assigning an unvalidated gene/CMS identity. Use complete ATG-start ORFs.
    """
    fasta = write_fasta(
        scratch / "orf-input.fasta", [(r.seqid, r.sequence) for r in document.records]
    )
    result = find_orfs(
        fasta,
        organelle=organelle,
        genetic_code=genetic_code,
        min_aa=min_aa,
        circular=circular,
    )
    metrics = thaw_json(result.metrics)
    candidates = metrics["orfs"]
    ids = {f.feature_id for r in document.records for f in r.features}
    tags = {
        tag for r in document.records for f in r.features for tag in f.qualifier_values("locus_tag")
    }
    ordinal = 0
    duplicates = 0
    added = []
    records = []
    for record in document.records:
        existing = [f for f in record.features if f.type.casefold() == "cds"]
        locations = {_location(f.parts) for f in existing}
        features = list(record.features)
        for row in candidates:
            if row["sequence_id"] != record.seqid:
                continue
            direction = 1 if row["strand"] == "+" else -1
            parts = tuple(
                LocationPart(start=p["start"] - 1, end=p["end"], strand=direction)
                for p in row["parts"]
            )
            if _location(parts) in locations:
                duplicates += 1
                continue
            while True:
                ordinal += 1
                tag = f"OV_ORF_{ordinal}"
                feature_id = f"{record.seqid}:orf:{ordinal}"
                if feature_id not in ids and tag not in tags:
                    break
            ids.add(feature_id)
            tags.add(tag)
            overlaps = [
                f.feature_id
                for f in existing
                if any(a.start < b.end and b.start < a.end for a in parts for b in f.parts)
            ]
            note = "Sequence-only ORF candidate; expression, splicing, RNA editing and CMS causality untested"
            if overlaps:
                note += "; overlaps existing CDS: " + ", ".join(overlaps)
            feature = AnnotationFeature(
                feature_id=feature_id,
                seqid=record.seqid,
                type="CDS",
                operator="join" if len(parts) > 1 else "single",
                parts=parts,
                parents=(),
                qualifiers=(
                    FeatureQualifier(name="locus_tag", values=(tag,)),
                    FeatureQualifier(name="product", values=("hypothetical protein",)),
                    FeatureQualifier(name="translation", values=(row["protein_sequence"],)),
                    FeatureQualifier(name="transl_table", values=(str(genetic_code),)),
                    FeatureQualifier(name="codon_start", values=("1",)),
                    FeatureQualifier(
                        name="inference",
                        values=(f"ab initio prediction:orfipy:{metrics['backend_version']}",),
                    ),
                    FeatureQualifier(name="note", values=(note,)),
                ),
            )
            if feature.extract(record.sequence) != row["cds_sequence"]:
                raise OrganelleInputError(
                    code="annotation.orf_location_mismatch",
                    message="ORF CDS does not match canonical original-record coordinates",
                )
            features.append(feature)
            added.append(
                {
                    "feature_id": feature_id,
                    "locus_tag": tag,
                    "sequence_id": record.seqid,
                    "source_orf_id": row["id"],
                    "length_aa": row["length_aa"],
                    "wraps_origin": row["wraps_origin"],
                    "overlapping_cds": overlaps,
                }
            )
        records.append(record.evolve(features=tuple(features)))
    evidence = {
        "candidate_count": len(candidates),
        "added_count": len(added),
        "exact_existing_cds_count": duplicates,
        "added_features": added,
        "backend": metrics["backend"],
        "backend_version": metrics["backend_version"],
        "parameters": metrics["parameters"],
        "interpretation": "Sequence-only candidates, not confirmed genes or CMS determinants",
    }
    source_metadata = {**thaw_json(document.source_metadata), "orf_candidates": evidence}
    if circular:
        source_metadata["topology"] = "circular"
    return document.evolve(
        records=tuple(records),
        requested_stages=tuple(dict.fromkeys((*document.requested_stages, "orf"))),
        completed_stages=tuple(dict.fromkeys((*document.completed_stages, "orf"))),
        source_metadata=FrozenMap.from_json(source_metadata),
    )
