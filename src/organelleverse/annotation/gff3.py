"""GFF3 rendering for canonical annotation documents."""

from __future__ import annotations

from urllib.parse import quote

from .models import AnnotationDocument, AnnotationFeature


def _feature_name(feature: AnnotationFeature) -> str:
    for qualifier in ("gene", "locus_tag"):
        values = feature.qualifier_values(qualifier)
        if values and values[0]:
            return values[0]
    return feature.feature_id


def _part_phases(feature: AnnotationFeature) -> tuple[str, ...]:
    if feature.type.casefold() != "cds":
        return tuple("." for _ in feature.parts)
    codon_start = feature.qualifier_values("codon_start")
    initial_phase = int(codon_start[0]) - 1 if codon_start else 0
    if initial_phase not in {0, 1, 2}:
        raise ValueError(f"invalid codon_start for {feature.feature_id}")
    coding_bases = -initial_phase
    phases: list[str] = []
    for index, part in enumerate(feature.parts):
        phase = initial_phase if index == 0 else (3 - coding_bases % 3) % 3
        phases.append(str(phase))
        coding_bases += part.end - part.start
    return tuple(phases)


def document_to_gff3(document: AnnotationDocument) -> list[str]:
    """Render one GFF3 row per canonical location part."""

    lines = ["##gff-version 3"]
    for record in document.records:
        lines.append(f"##sequence-region {record.seqid} 1 {len(record.sequence)}")
        for feature in record.features:
            phases = _part_phases(feature)
            for part, phase in zip(feature.parts, phases, strict=True):
                attributes: list[str] = []
                # A discontinuous feature shares one ID across all of its parts.
                # Parent relationships come only from canonical explicit parents.
                attributes.append(f"ID={quote(feature.feature_id, safe='._:-')}")
                if feature.parents:
                    attributes.append(
                        "Parent="
                        + ",".join(quote(parent, safe="._:-") for parent in feature.parents)
                    )
                if any(
                    value.startswith("ab initio prediction:orfipy:")
                    for value in feature.qualifier_values("inference")
                ):
                    for key, name in (
                        ("Name", "locus_tag"),
                        ("product", "product"),
                        ("Note", "note"),
                        ("inference", "inference"),
                    ):
                        values = feature.qualifier_values(name)
                        if values:
                            attributes.append(f"{key}={quote(values[0], safe='._:-')}")
                strand = "-" if part.strand == -1 else "+"
                lines.append(
                    f"{record.seqid}\tOrganelleVerse\t{feature.type}\t"
                    f"{part.start + 1}\t{part.end}\t.\t{strand}\t{phase}\t" + ";".join(attributes)
                )
    return lines


__all__ = ["document_to_gff3"]
