import pytest
from pydantic import ValidationError

from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.core.frozen import FrozenMap


def _feature(*, feature_id: str = "r1:0:CDS") -> AnnotationFeature:
    return AnnotationFeature(
        feature_id=feature_id,
        seqid="r1",
        type="CDS",
        operator="single",
        parts=(LocationPart(start=0, end=3, strand=1),),
        qualifiers=(FeatureQualifier(name="gene", values=("nad5",)),),
        parents=(),
    )


def test_compound_feature_extracts_parts_without_intervening_sequence() -> None:
    feature = AnnotationFeature(
        feature_id="r1:0:CDS",
        seqid="r1",
        type="CDS",
        operator="join",
        parts=(
            LocationPart(start=0, end=3, strand=1),
            LocationPart(start=9, end=12, strand=1),
        ),
        qualifiers=(),
        parents=(),
    )

    assert feature.extract("AAACCCGGGTTT") == "AAATTT"
    assert [feature.genomic_position(i) for i in range(6)] == [0, 1, 2, 9, 10, 11]


def test_reverse_compound_uses_biological_part_order() -> None:
    feature = AnnotationFeature(
        feature_id="r1:1:CDS",
        seqid="r1",
        type="CDS",
        operator="join",
        parts=(
            LocationPart(start=9, end=12, strand=-1),
            LocationPart(start=0, end=3, strand=-1),
        ),
        qualifiers=(),
        parents=(),
    )

    assert feature.extract("AAACCCGGGTTT") == "AAATTT"
    assert [feature.genomic_position(i) for i in range(6)] == [11, 10, 9, 2, 1, 0]


def test_location_part_rejects_empty_or_reverse_interval() -> None:
    with pytest.raises(ValidationError, match="end"):
        LocationPart(start=3, end=3, strand=1)


def test_single_location_requires_exactly_one_part() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        AnnotationFeature(
            feature_id="r1:0:CDS",
            seqid="r1",
            type="CDS",
            operator="single",
            parts=(
                LocationPart(start=0, end=3, strand=1),
                LocationPart(start=9, end=12, strand=1),
            ),
            qualifiers=(),
            parents=(),
        )


def test_feature_rejects_offsets_outside_spliced_sequence() -> None:
    feature = _feature()

    with pytest.raises(IndexError, match="non-negative"):
        feature.genomic_position(-1)
    with pytest.raises(IndexError, match="exceeds"):
        feature.genomic_position(3)


def test_record_rejects_duplicate_feature_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate feature ID"):
        AnnotationRecord(
            seqid="r1",
            name="record one",
            description="fixture",
            sequence="AAA",
            features=(_feature(), _feature()),
        )


def test_document_rejects_duplicate_record_ids() -> None:
    record = AnnotationRecord(
        seqid="r1",
        name="record one",
        description="fixture",
        sequence="AAA",
        features=(_feature(),),
    )

    with pytest.raises(ValidationError, match="duplicate record ID"):
        AnnotationDocument(
            backend="fixture",
            requested_stages=("pcg",),
            completed_stages=("pcg",),
            records=(record, record),
            source_metadata=FrozenMap({"sha256": "abc123"}),
        )


def test_document_json_round_trip_preserves_immutable_data() -> None:
    document = AnnotationDocument(
        backend="fixture",
        requested_stages=("pcg", "trna"),
        completed_stages=("pcg",),
        records=(
            AnnotationRecord(
                seqid="r1",
                name="record one",
                description="fixture",
                sequence="AAA",
                features=(_feature(),),
            ),
        ),
        source_metadata=FrozenMap({"path": "fixture.gb", "circular": True}),
    )

    restored = AnnotationDocument.model_validate_json(document.model_dump_json())

    assert restored == document
    assert restored.schema_version == "organelleverse.annotation.v1"
    assert restored.source_metadata["path"] == "fixture.gb"
