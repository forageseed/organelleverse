from collections.abc import Callable

import pytest

from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.annotation.validation import validate_document
from organelleverse.core.frozen import FrozenMap


def valid_document() -> AnnotationDocument:
    cds = AnnotationFeature(
        feature_id="r1:0:CDS",
        seqid="r1",
        type="CDS",
        operator="single",
        parts=(LocationPart(start=0, end=9, strand=1),),
        qualifiers=(
            FeatureQualifier(name="gene", values=("nad1",)),
            FeatureQualifier(name="translation", values=("MK",)),
        ),
        parents=(),
    )
    trna = AnnotationFeature(
        feature_id="r1:1:tRNA",
        seqid="r1",
        type="tRNA",
        operator="single",
        parts=(LocationPart(start=0, end=3, strand=1),),
        qualifiers=(FeatureQualifier(name="gene", values=("trnM-CAU",)),),
        parents=(),
    )
    return AnnotationDocument(
        backend="fixture",
        requested_stages=("pcg", "trna"),
        completed_stages=("pcg", "trna"),
        records=(
            AnnotationRecord(
                seqid="r1",
                name="r1",
                description="validation fixture",
                sequence="ATGAAATAG",
                features=(cds, trna),
            ),
        ),
        source_metadata=FrozenMap(),
    )


def _replace_first_feature(
    document: AnnotationDocument,
    feature: AnnotationFeature,
) -> AnnotationDocument:
    record = document.records[0].evolve(features=(feature, *document.records[0].features[1:]))
    return document.evolve(records=(record,))


def outside_sequence(document: AnnotationDocument) -> AnnotationDocument:
    feature = (
        document.records[0].features[0].evolve(parts=(LocationPart(start=0, end=10, strand=1),))
    )
    return _replace_first_feature(document, feature)


def duplicate_feature_id(document: AnnotationDocument) -> AnnotationDocument:
    record = document.records[0]
    duplicate = record.features[1].evolve(feature_id=record.features[0].feature_id)
    unsafe_record = AnnotationRecord.model_construct(
        kind="annotation_record",
        seqid=record.seqid,
        name=record.name,
        description=record.description,
        sequence=record.sequence,
        features=(record.features[0], duplicate),
    )
    return AnnotationDocument.model_construct(
        schema_version="organelleverse.annotation.v1",
        kind="annotation_document",
        backend=document.backend,
        requested_stages=document.requested_stages,
        completed_stages=document.completed_stages,
        records=(unsafe_record,),
        source_metadata=document.source_metadata,
    )


def missing_requested_trna(document: AnnotationDocument) -> AnnotationDocument:
    record = document.records[0].evolve(features=(document.records[0].features[0],))
    return document.evolve(completed_stages=("pcg",), records=(record,))


def bad_cds_translation(document: AnnotationDocument) -> AnnotationDocument:
    feature = (
        document.records[0]
        .features[0]
        .evolve(
            qualifiers=(
                FeatureQualifier(name="gene", values=("nad1",)),
                FeatureQualifier(name="translation", values=("ZZ",)),
            )
        )
    )
    return _replace_first_feature(document, feature)


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    [
        (outside_sequence, "coordinate_out_of_bounds"),
        (duplicate_feature_id, "duplicate_feature_id"),
        (missing_requested_trna, "requested_stage_incomplete"),
        (bad_cds_translation, "cds_translation_inconsistent"),
    ],
)
def test_invalid_document_fails_closed(
    mutator: Callable[[AnnotationDocument], AnnotationDocument],
    error_code: str,
) -> None:
    report = validate_document(mutator(valid_document()), required_stages=("pcg", "trna"))

    assert not report.valid
    assert error_code in {issue.code for issue in report.errors}


def test_valid_document_reports_frozen_feature_counts() -> None:
    report = validate_document(valid_document(), required_stages=("pcg", "trna"))

    assert report.valid
    assert not report.errors
    assert report.feature_counts["CDS"] == 1
    assert report.feature_counts["tRNA"] == 1
    assert '"feature_counts":{"CDS":1,"tRNA":1}' in report.model_dump_json()


def test_pseudo_cds_translation_exemption_is_named_warning() -> None:
    document = valid_document()
    feature = (
        document.records[0]
        .features[0]
        .evolve(
            qualifiers=(
                FeatureQualifier(name="gene", values=("nad1",)),
                FeatureQualifier(name="pseudo"),
                FeatureQualifier(name="translation", values=("WRONG",)),
            )
        )
    )

    report = validate_document(_replace_first_feature(document, feature), ("pcg", "trna"))

    assert report.valid
    assert "cds_translation_exempt" in {issue.code for issue in report.warnings}


def test_generic_exception_cannot_exempt_non_triplet_cds() -> None:
    document = valid_document()
    feature = (
        document.records[0]
        .features[0]
        .evolve(
            parts=(LocationPart(start=0, end=8, strand=1),),
            qualifiers=(
                FeatureQualifier(name="gene", values=("nad1",)),
                FeatureQualifier(name="exception", values=("RNA editing",)),
            ),
        )
    )

    report = validate_document(_replace_first_feature(document, feature), ("pcg", "trna"))

    assert report.valid is False
    assert "cds_translation_inconsistent" in {issue.code for issue in report.errors}


def test_unqualified_internal_stop_fails_closed() -> None:
    document = valid_document()
    feature = (
        document.records[0]
        .features[0]
        .evolve(qualifiers=(FeatureQualifier(name="gene", values=("atp1",)),))
    )
    record = document.records[0].evolve(
        sequence="ATGTAATAG",
        features=(feature, document.records[0].features[1]),
    )

    report = validate_document(document.evolve(records=(record,)), ("pcg", "trna"))

    assert report.valid is False
    assert "cds_internal_stop" in {issue.code for issue in report.errors}


def test_known_hard_gene_internal_stop_is_demoted_to_warning() -> None:
    """Trans-spliced/divergent genes (nad1, rpl2, ...) keep internal stops
    by design; the canonical validator must not fail the whole document."""
    document = valid_document()
    feature = (
        document.records[0]
        .features[0]
        .evolve(qualifiers=(FeatureQualifier(name="gene", values=("nad1",)),))
    )
    record = document.records[0].evolve(
        sequence="ATGTAATAG",
        features=(feature, document.records[0].features[1]),
    )

    report = validate_document(document.evolve(records=(record,)), ("pcg", "trna"))

    assert report.valid is True
    codes = {issue.code for issue in report.warnings}
    assert "cds_internal_stop_exception" in codes


def test_complete_canonical_cds_requires_start_and_stop_codons() -> None:
    document = valid_document().evolve(backend="mitochondrion")
    feature = (
        document.records[0]
        .features[0]
        .evolve(qualifiers=(FeatureQualifier(name="gene", values=("nad1",)),))
    )
    missing_start_record = document.records[0].evolve(
        sequence="AAAAAATAA" + document.records[0].sequence[9:],
        features=(feature, *document.records[0].features[1:]),
    )
    missing_stop_record = document.records[0].evolve(
        sequence="ATGAAAAAA" + document.records[0].sequence[9:],
        features=(feature, *document.records[0].features[1:]),
    )

    missing_start = validate_document(
        document.evolve(records=(missing_start_record,)), ("pcg", "trna")
    )
    missing_stop = validate_document(
        document.evolve(records=(missing_stop_record,)), ("pcg", "trna")
    )

    assert "cds_missing_start" in {issue.code for issue in missing_start.errors}
    assert "cds_missing_stop" in {issue.code for issue in missing_stop.errors}


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [("ATGTAAA", "cds_translation_inconsistent"), ("ATGTAATAG", "cds_internal_stop")],
)
def test_partial_cds_does_not_hide_frame_or_internal_stop(
    sequence: str,
    expected: str,
) -> None:
    document = valid_document().evolve(backend="mitochondrion")
    feature = (
        document.records[0]
        .features[0]
        .evolve(
            parts=(LocationPart(start=0, end=len(sequence), strand=1),),
            qualifiers=(
                FeatureQualifier(name="gene", values=("atp1",)),
                FeatureQualifier(name="partial"),
            ),
        )
    )
    record = document.records[0].evolve(
        sequence=sequence,
        features=(feature, document.records[0].features[1]),
    )

    report = validate_document(document.evolve(records=(record,)), ("pcg", "trna"))

    assert report.valid is False
    assert expected in {issue.code for issue in report.errors}


def test_rna_editing_exception_does_not_make_tgg_a_terminal_stop() -> None:
    document = valid_document().evolve(backend="mitochondrion")
    feature = (
        document.records[0]
        .features[0]
        .evolve(
            qualifiers=(
                FeatureQualifier(name="gene", values=("atp6",)),
                FeatureQualifier(name="exception", values=("RNA editing",)),
            )
        )
    )
    record = document.records[0].evolve(
        sequence="ATGAAATGG",
        features=(feature, document.records[0].features[1]),
    )

    report = validate_document(document.evolve(records=(record,)), ("pcg", "trna"))

    assert "cds_missing_stop" in {issue.code for issue in report.errors}
