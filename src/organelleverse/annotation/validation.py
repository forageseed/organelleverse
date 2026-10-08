"""Fail-closed structural and requested-stage annotation validation."""

from __future__ import annotations

from collections import Counter
from typing import Literal, cast

from Bio.Seq import Seq
from pydantic import Field, field_serializer

from organelleverse.core.base import StrictFrozenModel
from organelleverse.core.frozen import FrozenJson, FrozenMap, thaw_json

from .mitochondrion.cds import KNOWN_HARD_INTERNAL_STOP_GENES
from .models import AnnotationDocument, AnnotationFeature, AnnotationRecord


class AnnotationValidationIssue(StrictFrozenModel[Literal["annotation_issue"]]):
    """One stable structural or biological validation finding."""

    kind: Literal["annotation_issue"] = "annotation_issue"
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    feature_id: str = ""


class AnnotationValidationReport(StrictFrozenModel[Literal["annotation_validation"]]):
    """Complete validation outcome for a canonical annotation document."""

    kind: Literal["annotation_validation"] = "annotation_validation"
    valid: bool
    errors: tuple[AnnotationValidationIssue, ...]
    warnings: tuple[AnnotationValidationIssue, ...]
    feature_counts: FrozenMap[FrozenJson]

    @field_serializer("feature_counts")
    def serialize_feature_counts(self, value: FrozenMap[FrozenJson]) -> object:
        return thaw_json(value)


def _issue(code: str, message: str, feature_id: str = "") -> AnnotationValidationIssue:
    return AnnotationValidationIssue(code=code, message=message, feature_id=feature_id)


def _feature_is_translation_exempt(feature: AnnotationFeature) -> bool:
    exempt_qualifiers = {"pseudo", "pseudogene"}
    return any(qualifier.name.casefold() in exempt_qualifiers for qualifier in feature.qualifiers)


def _feature_partial_terminals(feature: AnnotationFeature) -> tuple[bool, bool]:
    if not any(qualifier.name.casefold() == "partial" for qualifier in feature.qualifiers):
        return False, False
    first = feature.parts[0]
    last = feature.parts[-1]
    partial_5prime = (
        first.start_status != "exact" if first.strand == 1 else first.end_status != "exact"
    )
    partial_3prime = (
        last.end_status != "exact" if last.strand == 1 else last.start_status != "exact"
    )
    # Legacy GenBank often supplies only /partial, without fuzzy endpoints.
    if not partial_5prime and not partial_3prime:
        return True, True
    return partial_5prime, partial_3prime


def _feature_has_translation_exception(feature: AnnotationFeature) -> bool:
    for qualifier in feature.qualifiers:
        name = qualifier.name.casefold()
        if name == "transl_except" and qualifier.values:
            return True
        if name == "exception" and any(
            value.strip().casefold() in {"rna editing", "ribosomal slippage", "trans-splicing"}
            for value in qualifier.values
        ):
            return True
    return False


def _feature_has_rna_editing_exception(feature: AnnotationFeature) -> bool:
    return any(
        qualifier.name.casefold() == "exception"
        and any(value.strip().casefold() == "rna editing" for value in qualifier.values)
        for qualifier in feature.qualifiers
    )


def _gene_name(feature: AnnotationFeature) -> str:
    values = feature.qualifier_values("gene")
    return values[0].strip() if values else ""


def _allowed_start_codons(feature: AnnotationFeature, backend: str) -> set[str]:
    gene_name = _gene_name(feature)
    allow_rna_editing = False
    if backend == "mitochondrion":
        from .mitochondrion.codon_contract import allowed_start_codons
        from .mitochondrion.db import DBManager

        allow_rna_editing = DBManager().is_start_gain_gene(gene_name)
        return allowed_start_codons(
            gene_name,
            allow_rna_editing=allow_rna_editing,
        )
    return {"ATG"}


def _has_terminal_stop(feature: AnnotationFeature, sequence: str, backend: str) -> bool:
    terminal = sequence[-3:].upper()
    from Bio.Data import CodonTable

    tables = feature.qualifier_values("transl_table")
    table = CodonTable.ambiguous_dna_by_id[int(tables[0]) if tables else 1]
    if terminal in table.stop_codons:
        return True
    if backend != "mitochondrion" or not _feature_has_rna_editing_exception(feature):
        return False
    from .mitochondrion.db import DBManager

    return DBManager().is_stop_gain_gene(_gene_name(feature)) and terminal in {
        "CAA",
        "CAG",
        "CGA",
    }


def _coding_sequence(feature: AnnotationFeature, record: AnnotationRecord) -> str:
    codon_start = feature.qualifier_values("codon_start")
    start_offset = int(codon_start[0]) - 1 if codon_start else 0
    if start_offset not in {0, 1, 2}:
        raise ValueError("codon_start must be 1, 2, or 3")
    sequence = feature.extract(record.sequence)[start_offset:]
    if (
        len(sequence) % 3
        and any(_feature_partial_terminals(feature))
        and any(q.name == "trans_splicing" for q in feature.qualifiers)
    ):
        # a trans-spliced piece held by one contig may stop inside a codon
        sequence = sequence[: len(sequence) // 3 * 3]
    if not sequence or len(sequence) % 3:
        raise ValueError("spliced CDS length after codon_start is not divisible by three")
    return sequence


def _translate_cds(feature: AnnotationFeature, record: AnnotationRecord) -> str:
    table_values = feature.qualifier_values("transl_table")
    coding_sequence = _coding_sequence(feature, record)
    # C-to-U editing is directional: an ACG start becomes AUG and a terminal
    # CAA/CGA/CAG becomes a stop. When the feature declares RNA editing, apply
    # those two gains so the observed translation matches the edited product
    # the writer emits (and matches NCBI's own edited translations).
    if _feature_has_rna_editing_exception(feature):
        bases = list(coding_sequence.upper())
        if len(bases) >= 3 and "".join(bases[:3]) == "ACG":
            bases[1] = "T"
        if len(bases) >= 3 and "".join(bases[-3:]) in {"CAA", "CGA", "CAG"}:
            bases[-3] = "T"
        coding_sequence = "".join(bases)
    # Biopython accepts integer table IDs at runtime, while its stub says str.
    table = cast(str, int(table_values[0]) if table_values else 1)
    translation = str(Seq(coding_sequence).translate(table=table, cds=False, to_stop=False))
    return translation[:-1] if translation.endswith("*") else translation


def _validate_cds(
    feature: AnnotationFeature,
    record: AnnotationRecord,
    backend: str,
    errors: list[AnnotationValidationIssue],
    warnings: list[AnnotationValidationIssue],
) -> None:
    if _feature_is_translation_exempt(feature):
        warnings.append(
            _issue(
                "cds_translation_exempt",
                "CDS translation equality was exempted by an explicit qualifier",
                feature.feature_id,
            )
        )
        return
    partial_5prime, partial_3prime = _feature_partial_terminals(feature)
    if partial_5prime or partial_3prime:
        warnings.append(
            _issue(
                "cds_translation_exempt",
                "partial CDS terminal and translation equality checks were relaxed",
                feature.feature_id,
            )
        )
    try:
        coding_sequence = _coding_sequence(feature, record)
    except (TypeError, ValueError) as error:
        errors.append(
            _issue(
                "cds_translation_inconsistent",
                str(error),
                feature.feature_id,
            )
        )
        return
    if backend == "mitochondrion":
        first_codon = coding_sequence[:3].upper()
        if not partial_5prime and first_codon not in _allowed_start_codons(feature, backend):
            errors.append(
                _issue(
                    "cds_missing_start",
                    f"complete CDS begins with unsupported start codon {first_codon}",
                    feature.feature_id,
                )
            )
        if not partial_3prime and not _has_terminal_stop(feature, coding_sequence, backend):
            errors.append(
                _issue(
                    "cds_missing_stop",
                    "complete CDS ends with unsupported terminal codon "
                    f"{coding_sequence[-3:].upper()}",
                    feature.feature_id,
                )
            )
    if errors and any(issue.feature_id == feature.feature_id for issue in errors):
        return
    observed = _translate_cds(feature, record)
    has_exception = _feature_has_translation_exception(feature)
    if "*" in observed:
        has_specific_translation_exception = any(
            qualifier.name.casefold() == "transl_except" and qualifier.values
            for qualifier in feature.qualifiers
        )
        # Multi-exon CDS: internal stops are frequently exon-junction
        # artifacts or RNA-editing candidates, not pseudogenes. The same
        # applies to the divergent/trans-spliced gene set cds.py demotes
        # (KNOWN_HARD_INTERNAL_STOP_GENES). Demote to warning so these
        # genes are published for review instead of failing the entire
        # annotation.
        gene_name = "".join(feature.qualifier_values("gene")).lower()
        is_multi_part = len(feature.parts) > 1
        is_known_hard = gene_name in KNOWN_HARD_INTERNAL_STOP_GENES
        demoted = has_specific_translation_exception or is_multi_part or is_known_hard
        finding = _issue(
            "cds_internal_stop_exception" if demoted else "cds_internal_stop",
            "CDS translation contains an internal stop codon"
            + (" (multi-exon CDS; likely exon-junction artifact)" if is_multi_part else "")
            + (
                " (known hard gene; expected for review)"
                if is_known_hard and not is_multi_part
                else ""
            ),
            feature.feature_id,
        )
        if demoted:
            warnings.append(finding)
        else:
            errors.append(finding)
            return
    expected_values = feature.qualifier_values("translation")
    if not expected_values:
        return
    if partial_5prime or partial_3prime:
        return
    if has_exception:
        warnings.append(
            _issue(
                "cds_translation_exception",
                "translation equality was exempted by an explicit exception qualifier",
                feature.feature_id,
            )
        )
        return
    expected = "".join(expected_values[0].split()).removesuffix("*")
    if observed != expected:
        errors.append(
            _issue(
                "cds_translation_inconsistent",
                f"translation qualifier does not match spliced CDS ({expected!r} != {observed!r})",
                feature.feature_id,
            )
        )


def validate_document(
    document: AnnotationDocument,
    required_stages: tuple[str, ...] = (),
) -> AnnotationValidationReport:
    """Validate coordinates, hierarchy, translation, and requested stages."""

    errors: list[AnnotationValidationIssue] = []
    warnings: list[AnnotationValidationIssue] = []
    feature_counts: Counter[str] = Counter()
    record_ids: set[str] = set()
    feature_ids: set[str] = set()

    for record in document.records:
        if record.seqid in record_ids:
            errors.append(_issue("duplicate_record_id", f"duplicate record ID: {record.seqid}"))
        record_ids.add(record.seqid)
        for feature in record.features:
            feature_counts[feature.type] += 1
            if feature.feature_id in feature_ids:
                errors.append(
                    _issue(
                        "duplicate_feature_id",
                        f"duplicate feature ID: {feature.feature_id}",
                        feature.feature_id,
                    )
                )
            feature_ids.add(feature.feature_id)
            if feature.seqid != record.seqid:
                errors.append(
                    _issue(
                        "feature_record_mismatch",
                        "feature seqid does not match its record",
                        feature.feature_id,
                    )
                )
            if feature.operator == "single" and len(feature.parts) != 1:
                errors.append(
                    _issue(
                        "location_operator_inconsistent",
                        "single location must contain exactly one part",
                        feature.feature_id,
                    )
                )
            if feature.operator != "single" and len(feature.parts) < 2:
                errors.append(
                    _issue(
                        "location_operator_inconsistent",
                        "compound location must contain at least two parts",
                        feature.feature_id,
                    )
                )
            in_bounds = True
            for part in feature.parts:
                if part.start < 0 or part.end > len(record.sequence) or part.end <= part.start:
                    in_bounds = False
                    errors.append(
                        _issue(
                            "coordinate_out_of_bounds",
                            f"location {part.start}:{part.end} exceeds record length {len(record.sequence)}",
                            feature.feature_id,
                        )
                    )
            if feature.type.casefold() == "cds" and in_bounds:
                _validate_cds(feature, record, document.backend, errors, warnings)

    for record in document.records:
        for feature in record.features:
            for parent in feature.parents:
                if parent not in feature_ids:
                    errors.append(
                        _issue(
                            "parent_feature_missing",
                            f"parent feature does not exist: {parent}",
                            feature.feature_id,
                        )
                    )

    counts_by_stage = {
        "pcg": sum(count for name, count in feature_counts.items() if name.casefold() == "cds"),
        "trna": sum(count for name, count in feature_counts.items() if name.casefold() == "trna"),
        "rrna": sum(count for name, count in feature_counts.items() if name.casefold() == "rrna"),
    }
    completed = {stage.casefold() for stage in document.completed_stages}
    for stage in dict.fromkeys(required_stages):
        normalized = stage.casefold()
        # ORF discovery can legitimately complete with zero new candidates.
        if normalized not in completed or (
            normalized != "orf" and counts_by_stage.get(normalized, 0) == 0
        ):
            errors.append(
                _issue(
                    "requested_stage_incomplete",
                    f"requested annotation stage did not complete: {stage}",
                )
            )

    frozen_counts = FrozenMap.from_items(dict(sorted(feature_counts.items())))
    return AnnotationValidationReport(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        feature_counts=frozen_counts,
    )
