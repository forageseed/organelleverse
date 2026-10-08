"""Tests for the table2asn submission validator and its report parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.annotation.models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from organelleverse.annotation.table2asn import (
    find_table2asn,
    parse_val_text,
    run_table2asn_validation,
    write_submission_fasta,
)
from organelleverse.core.frozen import FrozenMap

_SAMPLE_VAL = """Some preamble line that is not a finding.
Error: valid [SEQ_FEAT.InternalStop] Internal stop codon FEATURE: CDS: hypothetical protein <1>  [lcl|r1:5-85]
Warning: valid [SEQ_FEAT.PartialProblem] Should be of type 5' partial FEATURE: CDS: hypothetical protein <1>
Error: valid [SEQ_DESCR.NoPubFound] No publications anywhere on this entire record. BIOSEQ: lcl|r1
Info: valid [GENERIC.MissingPubRequirement] No submission citation anywhere on this entire record. BIOSEQ: lcl|r1
"""


def _document(sequence: str, parts, gene: str) -> AnnotationDocument:
    cds = AnnotationFeature(
        feature_id="r1:0:CDS",
        seqid="r1",
        type="CDS",
        operator="single",
        parts=parts,
        qualifiers=(
            FeatureQualifier(name="gene", values=(gene,)),
            FeatureQualifier(name="product", values=("hypothetical protein",)),
            FeatureQualifier(name="transl_table", values=("11",)),
        ),
        parents=(),
    )
    return AnnotationDocument(
        backend="fixture",
        requested_stages=("pcg",),
        completed_stages=("pcg",),
        records=(
            AnnotationRecord(
                seqid="r1",
                name="r1",
                description="table2asn fixture",
                sequence=sequence,
                features=(cds,),
            ),
        ),
        source_metadata=FrozenMap({"topology": "linear", "species": "Nicotiana tabacum"}),
    )


def test_parse_val_text_splits_severity_code_and_message():
    findings = parse_val_text(_SAMPLE_VAL)
    assert [f.code for f in findings] == [
        "SEQ_DESCR.NoPubFound",
        "SEQ_FEAT.InternalStop",
        "SEQ_FEAT.PartialProblem",
        "GENERIC.MissingPubRequirement",
    ]
    feature_level = [f for f in findings if f.feature_level]
    assert [f.code for f in feature_level] == [
        "SEQ_FEAT.InternalStop",
        "SEQ_FEAT.PartialProblem",
    ]


def test_find_table2asn_honors_the_opt_out(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORG_VERSE_TABLE2ASN_BIN", "none")
    assert find_table2asn() is None
    monkeypatch.setenv("ORG_VERSE_TABLE2ASN_BIN", "off")
    assert find_table2asn() is None


def test_submission_fasta_carries_the_organism(tmp_path: Path):
    sequence = "ATGAAATAA" + "A" * 141
    document = _document(
        sequence,
        (LocationPart(start=0, end=9, strand=1),),
        "testgene",
    )
    path = write_submission_fasta(document, tmp_path / "genome.fsa")
    text = path.read_text()
    assert text.splitlines()[0] == ">r1 [organism=Nicotiana tabacum] [moltype=genomic DNA]"
    assert "".join(text.splitlines()[1:]) == sequence


@pytest.mark.parametrize(
    "sequence,parts,expect_feature_errors",
    [
        # In-frame ATG...TAA, no internal stop: submission-clean CDS.
        (
            "ATG" + "AACGTTGCAAGC" * 7 + "TAA",
            (LocationPart(start=0, end=90, strand=1),),
            False,
        ),
        # Same length but an in-frame TAA in the middle: internal stop.
        (
            "ATG" + "AACGTTGCAAGC" * 3 + "TAA" + "AACGTTGCAAGC" * 3 + "TAG",
            (LocationPart(start=0, end=90, strand=1),),
            True,
        ),
    ],
)
def test_run_table2asn_validation_reports_feature_errors(
    tmp_path: Path, sequence: str, parts, expect_feature_errors: bool
):
    binary = find_table2asn()
    if binary is None:
        pytest.skip("table2asn binary not provisioned")
    document = _document(sequence, parts, "check1")
    report = run_table2asn_validation(document, tmp_path / "work")
    assert report is not None
    assert report["schema_version"] == "organelleverse.table2asn-validation.v1"
    assert report["returncode"] == 0
    feature_codes = {f["code"] for f in report["feature_findings"]}
    if expect_feature_errors:
        assert report["feature_error_count"] >= 1
        assert any("InternalStop" in code or "NoStop" in code for code in feature_codes)
    else:
        assert report["feature_error_count"] == 0
