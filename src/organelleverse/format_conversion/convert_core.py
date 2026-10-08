"""Typed cores for format conversion (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from urllib.parse import quote

from ..annotation.genbank import parse_genbank
from ..annotation.models import AnnotationDocument, AnnotationFeature


def _feature_name(feature: AnnotationFeature) -> str:
    for qualifier in ("gene", "locus_tag"):
        values = feature.qualifier_values(qualifier)
        if values and values[0]:
            return values[0]
    return feature.feature_id


def _gff_value(value: str) -> str:
    return quote(value, safe="._:-")


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
            root = _feature_name(feature)
            phases = _part_phases(feature)
            for part_index, (part, phase) in enumerate(
                zip(feature.parts, phases, strict=True),
                start=1,
            ):
                attributes: list[str] = []
                if len(feature.parts) == 1:
                    attributes.append(f"ID={_gff_value(feature.feature_id)}")
                else:
                    attributes.extend(
                        (
                            f"ID={_gff_value(feature.feature_id)}.part{part_index}",
                            f"Parent={_gff_value(root)}",
                        )
                    )
                if feature.parents:
                    attributes.append(
                        "Parent=" + ",".join(_gff_value(parent) for parent in feature.parents)
                    )
                strand = "-" if part.strand == -1 else "+"
                lines.append(
                    f"{record.seqid}\tOrganelleVerse\t{feature.type}\t"
                    f"{part.start + 1}\t{part.end}\t.\t{strand}\t{phase}\t" + ";".join(attributes)
                )
    return lines


def convert_genbank_to_gff3(genbank_path: str) -> list[str]:
    """Convert GenBank features to GFF3 lines. Returns list of strings."""

    return document_to_gff3(parse_genbank(genbank_path))


def convert_genbank_to_fasta(genbank_path: str) -> list[tuple[str, str]]:
    """Extract sequences from GenBank as FASTA pairs. Returns list."""

    document = parse_genbank(genbank_path)
    return [(record.seqid, record.sequence) for record in document.records]
