"""Faithful GenBank conversion and compound-aware sequence extraction."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, SupportsInt, cast

from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.frozen import FrozenMap

from .models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationOperator,
    LocationPart,
    PositionStatus,
)


class _BioLocationPart(Protocol):
    start: object
    end: object
    strand: int | None
    ref: str | None
    ref_db: str | None


class _BioLocation(Protocol):
    parts: Sequence[_BioLocationPart]
    operator: str | None


class _BioFeature(Protocol):
    location: _BioLocation | None
    type: str
    qualifiers: Mapping[str, Sequence[object]]


class _BioRecord(Protocol):
    id: str
    name: str
    description: str
    seq: object
    features: Sequence[SeqFeature]


@dataclass(frozen=True)
class ParsedLocation:
    """A parsed location expression before it is attached to a feature."""

    operator: LocationOperator
    parts: tuple[LocationPart, ...]


def _invalid_genbank(*, reason: str, path: Path | None = None) -> OrganelleInputError:
    details = {"reason": reason}
    if path is not None:
        details["path"] = str(path)
    return OrganelleInputError(
        code="input.invalid_genbank",
        message="GenBank input is not supported",
        details=details,
    )


class _LocationParser:
    def __init__(self, text: str) -> None:
        self.text = "".join(text.split())
        self.index = 0

    def parse(self) -> ParsedLocation:
        if not self.text:
            raise ValueError("location is empty")
        if ":" in self.text:
            raise ValueError("remote accession locations are not supported")
        if "^" in self.text:
            raise ValueError("between-base locations are not supported")
        location = self._location()
        if self.index != len(self.text):
            raise ValueError(f"unexpected location text at offset {self.index}")
        return location

    def _location(self) -> ParsedLocation:
        for operator in ("complement", "join", "order"):
            token = f"{operator}("
            if self.text.startswith(token, self.index):
                self.index += len(token)
                if operator == "complement":
                    nested = self._location()
                    self._consume(")")
                    parts = tuple(
                        LocationPart(
                            start=part.start,
                            end=part.end,
                            strand=-part.strand,
                            start_status=part.start_status,
                            end_status=part.end_status,
                        )
                        for part in reversed(nested.parts)
                    )
                    return ParsedLocation(operator=nested.operator, parts=parts)

                locations = [self._location()]
                while self._peek() == ",":
                    self.index += 1
                    locations.append(self._location())
                self._consume(")")
                parts = tuple(part for location in locations for part in location.parts)
                return ParsedLocation(operator=cast(LocationOperator, operator), parts=parts)

        start, start_status = self._fuzzy_integer()
        if self.text.startswith("..", self.index):
            self.index += 2
            end, end_status = self._fuzzy_integer()
        else:
            end, end_status = start, start_status
        if end < start:
            raise ValueError("location end precedes its start")
        return ParsedLocation(
            operator="single",
            parts=(
                LocationPart(
                    start=start - 1,
                    end=end,
                    strand=1,
                    start_status=start_status,
                    end_status=end_status,
                ),
            ),
        )

    def _fuzzy_integer(self) -> tuple[int, PositionStatus]:
        marker = self._peek()
        status: PositionStatus = "exact"
        if marker in {"<", ">"}:
            self.index += 1
            status = "before" if marker == "<" else "after"
        match = re.match(r"[0-9]+", self.text[self.index :])
        if match is None:
            raise ValueError(f"expected a positive location integer at offset {self.index}")
        self.index += len(match.group())
        value = int(match.group())
        if value < 1:
            raise ValueError("GenBank locations are one-based positive integers")
        return value, status

    def _peek(self) -> str:
        return self.text[self.index] if self.index < len(self.text) else ""

    def _consume(self, token: str) -> None:
        if not self.text.startswith(token, self.index):
            raise ValueError(f"expected {token!r} at offset {self.index}")
        self.index += len(token)


def parse_location_text(text: str) -> ParsedLocation:
    """Parse the supported deterministic subset of GenBank location syntax."""

    try:
        return _LocationParser(text).parse()
    except (TypeError, ValueError) as error:
        raise _invalid_genbank(reason=str(error)) from error


def _position_status(position: object) -> PositionStatus:
    position_type = type(position).__name__
    if position_type == "BeforePosition":
        return "before"
    if position_type == "AfterPosition":
        return "after"
    if position_type == "ExactPosition":
        return "exact"
    return "unknown"


def _position_value(position: object) -> int:
    try:
        return int(cast(SupportsInt, position))
    except TypeError as error:
        raise ValueError("location endpoint has no concrete coordinate") from error


def feature_from_biopython(seqid: str, index: int, feature: SeqFeature) -> AnnotationFeature | None:
    """Convert every Biopython location part without flattening the feature.

    Returns ``None`` for features whose location is entirely zero-length
    between-bases sites: they hold no sequence, so no AnnotationFeature can
    represent them.
    """

    feature_data = cast(_BioFeature, feature)
    location = feature_data.location
    if location is None:
        raise ValueError(f"feature {index} has no location")
    raw_parts = tuple(location.parts)
    if any(part.ref is not None or part.ref_db is not None for part in raw_parts):
        raise ValueError("remote accession locations are not supported")
    # Zero-length parts (GenBank between-bases sites, e.g. misc_difference
    # markers in real reference records) carry no sequence and cannot become
    # a LocationPart, which requires end > start.
    parts = tuple(
        LocationPart(
            start=_position_value(part.start),
            end=_position_value(part.end),
            strand=-1 if part.strand == -1 else 1,
            start_status=_position_status(part.start),
            end_status=_position_status(part.end),
        )
        for part in raw_parts
        if _position_value(part.end) > _position_value(part.start)
    )
    if not parts:
        return None
    operator = location.operator if isinstance(location, CompoundLocation) else None
    normalized_operator: LocationOperator = (
        "single" if len(parts) == 1 else cast(LocationOperator, operator or "join")
    )
    if normalized_operator not in {"single", "join", "order"}:
        raise ValueError(f"unsupported location operator: {normalized_operator}")
    qualifiers = tuple(
        FeatureQualifier(name=name, values=tuple(str(value) for value in values))
        for name, values in sorted(feature_data.qualifiers.items())
    )
    return AnnotationFeature(
        feature_id=f"{seqid}:{index}:{feature_data.type}",
        seqid=seqid,
        type=feature_data.type,
        operator=normalized_operator,
        parts=parts,
        qualifiers=qualifiers,
        parents=(),
    )


def _record_from_biopython(record: SeqRecord) -> AnnotationRecord:
    record_data = cast(_BioRecord, record)
    sequence = str(record_data.seq)
    if not sequence:
        raise ValueError(f"record {record_data.id!r} has no sequence")
    features = tuple(
        converted
        for index, feature in enumerate(record_data.features)
        if (converted := feature_from_biopython(record_data.id, index, feature)) is not None
    )
    return AnnotationRecord(
        seqid=record_data.id,
        name=record_data.name,
        description=record_data.description,
        sequence=sequence,
        features=features,
    )


def _completed_stages(records: tuple[AnnotationRecord, ...]) -> tuple[str, ...]:
    feature_types = {feature.type.casefold() for record in records for feature in record.features}
    stages: list[str] = []
    if "cds" in feature_types:
        stages.append("pcg")
    if "trna" in feature_types:
        stages.append("trna")
    if "rrna" in feature_types:
        stages.append("rrna")
    return tuple(stages)


def _document(
    records: tuple[AnnotationRecord, ...],
) -> AnnotationDocument:
    if not records:
        raise ValueError("GenBank input contains no records")
    record_ids = tuple(record.seqid for record in records)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("GenBank input contains duplicate record IDs")
    stages = _completed_stages(records)
    return AnnotationDocument(
        backend="genbank",
        requested_stages=stages,
        completed_stages=stages,
        records=records,
        source_metadata=FrozenMap({"format": "genbank"}),
    )


def parse_genbank(path: str | Path) -> AnnotationDocument:
    """Parse GenBank into the canonical document or fail without approximation."""

    source = Path(path)
    try:
        from Bio.SeqIO import parse as bio_parse  # pyright: ignore[reportUnknownVariableType]
    except ImportError:
        try:
            return _parse_minimal_fallback(source)
        except OrganelleInputError:
            raise
        except Exception as error:
            raise _invalid_genbank(reason=str(error), path=source) from error

    parse_records = cast(
        Callable[[str | Path, str], Iterable[SeqRecord]],
        bio_parse,
    )
    try:
        records = tuple(
            _record_from_biopython(record) for record in parse_records(source, "genbank")
        )
        return _document(records)
    except OrganelleInputError:
        raise
    except Exception as error:
        raise _invalid_genbank(reason=str(error), path=source) from error


@dataclass
class _FallbackFeature:
    type: str
    location: str
    qualifier_lines: list[str]


def _fallback_feature_entries(section: str) -> tuple[_FallbackFeature, ...]:
    entries: list[_FallbackFeature] = []
    current: _FallbackFeature | None = None
    reading_qualifiers = False
    for raw_line in section.splitlines():
        if raw_line.strip() == "Location/Qualifiers":
            continue
        padded = raw_line.ljust(21)
        key = padded[5:21].strip()
        body = padded[21:].strip()
        if key:
            if current is not None:
                entries.append(current)
            current = _FallbackFeature(type=key, location=body, qualifier_lines=[])
            reading_qualifiers = False
        elif current is not None and body.startswith("/"):
            reading_qualifiers = True
            current.qualifier_lines.append(body)
        elif current is not None and reading_qualifiers:
            current.qualifier_lines.append(body)
        elif current is not None:
            current.location += body
    if current is not None:
        entries.append(current)
    return tuple(entries)


def _fallback_qualifiers(lines: list[str]) -> tuple[FeatureQualifier, ...]:
    grouped: dict[str, list[str]] = {}
    current_name = ""
    for line in lines:
        if line.startswith("/"):
            token = line[1:]
            if "=" not in token:
                current_name = token.strip()
                grouped.setdefault(current_name, [])
                continue
            current_name, value = token.split("=", 1)
            grouped.setdefault(current_name, []).append(value.strip().strip('"'))
        elif current_name and grouped[current_name]:
            grouped[current_name][-1] += line.strip().strip('"')
    return tuple(
        FeatureQualifier(name=name, values=tuple(values))
        for name, values in sorted(grouped.items())
    )


def _parse_fallback_record(block: str) -> AnnotationRecord:
    locus_line = next((line for line in block.splitlines() if line.startswith("LOCUS")), "")
    locus_fields = locus_line.split()
    if len(locus_fields) < 2:
        raise ValueError("record has no LOCUS identifier")
    seqid = locus_fields[1]
    if "ORIGIN" not in block:
        raise ValueError(f"record {seqid!r} has no ORIGIN sequence")
    before_origin, origin = block.split("ORIGIN", 1)
    sequence = "".join(character for character in origin if character.isalpha()).upper()
    if not sequence:
        raise ValueError(f"record {seqid!r} has no sequence")
    definition = ""
    for line in block.splitlines():
        if line.startswith("DEFINITION"):
            definition = line[len("DEFINITION") :].strip()
            break
    entries: tuple[_FallbackFeature, ...] = ()
    if "FEATURES" in before_origin:
        entries = _fallback_feature_entries(before_origin.split("FEATURES", 1)[1])
    features = tuple(
        AnnotationFeature(
            feature_id=f"{seqid}:{index}:{entry.type}",
            seqid=seqid,
            type=entry.type,
            operator=parsed.operator,
            parts=parsed.parts,
            qualifiers=_fallback_qualifiers(entry.qualifier_lines),
            parents=(),
        )
        for index, entry in enumerate(entries)
        for parsed in (parse_location_text(entry.location),)
    )
    return AnnotationRecord(
        seqid=seqid,
        name=seqid,
        description=definition,
        sequence=sequence,
        features=features,
    )


def _parse_minimal_fallback(path: Path) -> AnnotationDocument:
    text = path.read_text()
    blocks = tuple(block.strip() for block in text.split("//") if block.strip())
    records = tuple(_parse_fallback_record(block) for block in blocks)
    return _document(records)


def _feature_label(feature: AnnotationFeature, fallback: str) -> str:
    for qualifier_name in ("gene", "locus_tag"):
        values = feature.qualifier_values(qualifier_name)
        if values and values[0]:
            return values[0]
    return fallback


def _translation(feature: AnnotationFeature, sequence: str) -> str | None:
    if feature.qualifier_values("pseudo") or feature.qualifier_values("pseudogene"):
        return None
    if any(qualifier.name in {"pseudo", "pseudogene"} for qualifier in feature.qualifiers):
        return None
    translation = feature.qualifier_values("translation")
    if feature.qualifier_values("transl_except"):
        if translation:
            return translation[0].replace(" ", "")
        raise _invalid_genbank(
            reason=f"feature {feature.feature_id} has transl_except without translation"
        )
    codon_start_values = feature.qualifier_values("codon_start")
    table_values = feature.qualifier_values("transl_table")
    try:
        start_offset = int(codon_start_values[0]) - 1 if codon_start_values else 0
        if start_offset not in {0, 1, 2}:
            raise ValueError("codon_start must be 1, 2, or 3")
        # Biopython accepts integer table IDs at runtime, while its stub says str.
        table = cast(str, int(table_values[0]) if table_values else 1)
        coding_sequence = feature.extract(sequence)[start_offset:]
        complete_length = len(coding_sequence) - len(coding_sequence) % 3
        if complete_length == 0:
            return ""
        protein = str(
            Seq(coding_sequence[:complete_length]).translate(
                table=table,
                cds=False,
                to_stop=False,
            )
        )
        return protein[:-1] if protein.endswith("*") else protein
    except (TypeError, ValueError) as error:
        raise _invalid_genbank(
            reason=f"cannot translate feature {feature.feature_id}: {error}"
        ) from error


def extract_feature_records(
    document: AnnotationDocument,
    feature_type: str,
) -> list[tuple[str, str]]:
    """Extract canonical feature sequences while preserving biological parts."""

    output: list[tuple[str, str]] = []
    requested_type = feature_type.casefold()
    target_type = "cds" if requested_type == "protein" else requested_type
    for record in document.records:
        for index, feature in enumerate(record.features):
            if feature.type.casefold() != target_type:
                continue
            label = _feature_label(feature, f"{feature_type}_{index}")
            if requested_type == "protein":
                protein = _translation(feature, record.sequence)
                if protein is not None:
                    output.append((label, protein))
            else:
                output.append((label, feature.extract(record.sequence)))
    return output
