import builtins
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import pytest
from Bio.SeqIO import read as bio_read  # pyright: ignore[reportUnknownVariableType]
from Bio.SeqRecord import SeqRecord

from organelleverse.annotation.genbank import (
    extract_feature_records,
    parse_genbank,
    parse_location_text,
)
from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.frozen import FrozenMap, thaw_json


class _ExpectedFeature(Protocol):
    type: str
    qualifiers: Mapping[str, Sequence[str]]

    def extract(self, parent_sequence: object) -> object: ...


class _ExpectedRecord(Protocol):
    seq: object
    features: Sequence[_ExpectedFeature]


def test_arabidopsis_joined_nad5_matches_biopython() -> None:
    path = Path("tests/data/Arabidopsis_thaliana.gb")
    read_record = cast(Callable[[str | Path, str], SeqRecord], bio_read)
    biopython_record = cast(_ExpectedRecord, read_record(path, "genbank"))
    expected_feature = next(
        feature
        for feature in biopython_record.features
        if feature.type == "CDS" and feature.qualifiers.get("gene") == ["nad5"]
    )
    expected = str(expected_feature.extract(biopython_record.seq))

    document = parse_genbank(path)
    nad5 = next(
        sequence for name, sequence in extract_feature_records(document, "CDS") if name == "nad5"
    )

    assert len(expected) == 2010
    assert nad5 == expected
    assert len(nad5) != 333312


def test_minimal_location_parser_matches_biopython_order() -> None:
    location = parse_location_text("complement(join(1..3,10..12))")

    assert [(part.start, part.end, part.strand) for part in location.parts] == [
        (9, 12, -1),
        (0, 3, -1),
    ]
    assert location.operator == "join"


def test_minimal_location_parser_preserves_supported_topology() -> None:
    ordered = parse_location_text("order(1..3,\n 10..12)")
    fuzzy = parse_location_text("<1..>10")
    point = parse_location_text("7")

    assert ordered.operator == "order"
    assert [(part.start, part.end, part.strand) for part in ordered.parts] == [
        (0, 3, 1),
        (9, 12, 1),
    ]
    assert fuzzy.parts[0].start_status == "before"
    assert fuzzy.parts[0].end_status == "after"
    assert (point.parts[0].start, point.parts[0].end) == (6, 7)


@pytest.mark.parametrize("text", ["JX000001.1:1..20", "1^2", "join(1..3,)"])
def test_minimal_location_parser_rejects_unsupported_syntax(text: str) -> None:
    with pytest.raises(OrganelleInputError) as caught:
        parse_location_text(text)

    assert caught.value.code == "input.invalid_genbank"


def test_parse_genbank_rejects_empty_input(tmp_path: Path) -> None:
    path = tmp_path / "empty.gb"
    path.write_text("")

    with pytest.raises(OrganelleInputError) as caught:
        parse_genbank(path)

    assert caught.value.code == "input.invalid_genbank"
    details = cast(dict[str, object], thaw_json(caught.value.details))
    assert details["path"] == str(path)


def test_parse_genbank_preserves_qualifier_value_lists(tmp_path: Path) -> None:
    path = tmp_path / "qualifiers.gb"
    path.write_text(
        """\
LOCUS       R1                        12 bp    DNA     linear   PLN 01-JAN-2025
FEATURES             Location/Qualifiers
     CDS             join(1..3,10..12)
                     /gene="nad5"
                     /note="first"
                     /note="second"
ORIGIN
        1 aaacccgggttt
//
"""
    )

    document = parse_genbank(path)
    feature = document.records[0].features[0]

    assert feature.qualifier_values("note") == ("first", "second")
    assert feature.extract(document.records[0].sequence) == "AAATTT"


def test_parse_genbank_identity_does_not_depend_on_transient_source_path(tmp_path: Path) -> None:
    source = Path("tests/data/Arabidopsis_thaliana.gb")
    copied = tmp_path / "copied.gb"
    shutil.copy2(source, copied)

    original = parse_genbank(source)
    relocated = parse_genbank(copied)

    assert original.object_id == relocated.object_id
    assert "path" not in original.source_metadata
    assert "parser" not in original.source_metadata


def test_builtin_parser_preserves_the_same_supported_compound_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "builtin.gb"
    path.write_text(
        """\
LOCUS       R1                        12 bp    DNA     linear   PLN 01-JAN-2025
FEATURES             Location/Qualifiers
     CDS             complement(join(1..3,
                     10..12))
                     /gene="nad5"
                     /note="first"
                     /note="second"
ORIGIN
        1 aaacccgggttt
//
"""
    )

    original_import = builtins.__import__

    def import_without_biopython_seqio(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "Bio.SeqIO":
            raise ImportError("forced fallback")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_biopython_seqio)
    document = parse_genbank(path)
    feature = document.records[0].features[0]

    assert [(part.start, part.end, part.strand) for part in feature.parts] == [
        (9, 12, -1),
        (0, 3, -1),
    ]
    assert feature.qualifier_values("note") == ("first", "second")
    assert feature.extract(document.records[0].sequence) == "AAATTT"


def test_protein_extraction_uses_spliced_codon_start_and_translation_exceptions() -> None:
    translated = AnnotationFeature(
        feature_id="r1:0:CDS",
        seqid="r1",
        type="CDS",
        operator="join",
        parts=(
            LocationPart(start=0, end=4, strand=1),
            LocationPart(start=10, end=16, strand=1),
        ),
        qualifiers=(
            FeatureQualifier(name="gene", values=("nad5",)),
            FeatureQualifier(name="codon_start", values=("2",)),
            FeatureQualifier(name="transl_table", values=("1",)),
        ),
        parents=(),
    )
    exception = AnnotationFeature(
        feature_id="r1:1:CDS",
        seqid="r1",
        type="CDS",
        operator="single",
        parts=(LocationPart(start=16, end=25, strand=1),),
        qualifiers=(
            FeatureQualifier(name="gene", values=("ccmFc",)),
            FeatureQualifier(name="transl_except", values=("(pos:19..21,aa:Sec)",)),
            FeatureQualifier(name="translation", values=("MU",)),
        ),
        parents=(),
    )
    pseudo = AnnotationFeature(
        feature_id="r1:2:CDS",
        seqid="r1",
        type="CDS",
        operator="single",
        parts=(LocationPart(start=25, end=28, strand=1),),
        qualifiers=(
            FeatureQualifier(name="gene", values=("pseudo",)),
            FeatureQualifier(name="pseudo"),
        ),
        parents=(),
    )
    document = AnnotationDocument(
        backend="fixture",
        requested_stages=(),
        completed_stages=(),
        records=(
            AnnotationRecord(
                seqid="r1",
                name="r1",
                description="translation fixture",
                sequence="AATGCCCCCCAAATAGATGTGATAAAAAAAAA",
                features=(translated, exception, pseudo),
            ),
        ),
        source_metadata=FrozenMap(),
    )

    assert extract_feature_records(document, "Protein") == [("nad5", "MK"), ("ccmFc", "MU")]
