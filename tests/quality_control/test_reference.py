"""Reference breadth must include inversions without silently conflating IR hits."""

import builtins
import random
from types import SimpleNamespace

import pytest
from Bio.Seq import reverse_complement as rc

from organelleverse._bio import write_fasta
from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError
from organelleverse.qc import compare_assembly_to_reference
from organelleverse.quality_control.reference import _compare


def _dna(length, seed):
    return "".join(random.Random(seed).choices("ACGT", k=length))


def _plastome():
    lsc = "AAAAA" + _dna(8990, 1) + "AAAAA"
    ir = _dna(2000, 2)
    ssc = "AAAAA" + _dna(3990, 3) + "AAAAA"
    return lsc, ir, ssc


def test_both_strands_secondary_union_and_circular_coordinates():
    def hit(start, end, strand, primary):
        return SimpleNamespace(
            q_st=0,
            q_en=end - start,
            r_st=start,
            r_en=end,
            strand=strand,
            is_primary=primary,
            mlen=end - start,
            blen=end - start,
        )

    aligner = SimpleNamespace(map=lambda seq: [hit(0, 7, 1, True), hit(5, 10, -1, False)])
    result = _compare(aligner, "ACGTACGTAC", 10, {"IRa": {"start": 8, "length": 10}})
    assert result["reference_intervals"] == [[0, 10]]
    assert result["reference_coverage"] == 1
    assert result["alignment_identity"] == 1
    assert result["inverted_blocks"][0]["is_primary"] is False
    assert result["inverted_blocks"][0]["query_intervals"] == [[8, 10], [0, 3]]


def test_generic_reverse_and_no_match(tmp_path):
    ref = _dna(5000, 80)
    path = write_fasta(tmp_path / "q.fa", [("reverse", rc(ref)), ("unrelated", _dna(5000, 81))])
    result = compare_assembly_to_reference(path, write_fasta(tmp_path / "r.fa", [("r", ref)]))
    reverse, unrelated = result.metrics["comparisons"]
    assert result.status == "warning"
    assert reverse["raw"]["reference_coverage"] == 1
    assert reverse["raw"]["alignment_identity"] == 1
    assert reverse["raw"]["inverted_blocks"]
    assert reverse["normalized"] is None
    assert unrelated["raw"]["reference_coverage"] == 0
    assert unrelated["raw"]["alignment_identity"] is None


def test_plastome_candidates_replay_and_reference_rotation(tmp_path):
    lsc, ir, ssc = _plastome()
    original = lsc + ir + ssc + rc(ir)
    isomer = lsc + ir + rc(ssc) + rc(ir)
    inputs = [
        ("same", original),
        ("ssc", isomer),
        ("reverse", rc(original)),
        ("rotated", isomer[15500:] + isomer[:15500]),
    ]
    ref = original[123:] + original[:123]
    result = compare_assembly_to_reference(
        write_fasta(tmp_path / "q.fa", inputs),
        write_fasta(tmp_path / "r.fa", [("r", ref)]),
        organelle="plastid",
    )
    assert result.status == "ok"
    assert result.metrics["reference_normalization"]["rotation"] != 0
    for row in result.metrics["comparisons"]:
        assert row["raw"]["reference_coverage"] > 0.99
        assert row["normalized"]["reference_coverage"] > 0.99
        assert row["normalized"]["alignment_identity"] == 1
        seq = dict(inputs)[row["sample"]]
        for op in row["normalization"]["operations"]:
            if op["operation"] == "rotate_left":
                seq = seq[op["bases"] :] + seq[: op["bases"]]
            else:
                seq = seq[: op["start"]] + rc(seq[op["start"] : op["end"]]) + seq[op["end"] :]
        assert seq == original
    ssc_row = result.metrics["comparisons"][1]
    assert any(b["region"] == "SSC" for b in ssc_row["raw"]["inverted_blocks"])
    assert not any(b["region"] == "SSC" for b in ssc_row["normalized"]["inverted_blocks"])


def test_unresolved_plastid_not_fabricated(tmp_path):
    lsc, ir, ssc = _plastome()
    ref = write_fasta(tmp_path / "r.fa", [("r", lsc + ir + ssc + rc(ir))])
    query = write_fasta(tmp_path / "q.fa", [("irless", lsc)])
    result = compare_assembly_to_reference(query, ref, organelle="plastid")
    assert result.status == "failed"
    assert result.metrics["unresolved"][0]["reason"] == "quadripartite_not_identified"
    row = result.metrics["comparisons"][0]
    assert row["raw"] is None and row["normalized"] is None
    assert row["whole_sequence_raw"]["reference_coverage"] > 0
    result = compare_assembly_to_reference(query, query, organelle="plastid")
    assert result.status == "failed"


def test_missing_mappy_has_install_hint(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "mappy":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(OrganelleDependencyError, match=r"organelleverse\[align\]"):
        compare_assembly_to_reference(tmp_path / "q.fa", tmp_path / "r.fa")


@pytest.mark.parametrize("text", ["", ">a\nAC-G\n", ">a\nACGT\n>a\nACGT\n"])
def test_invalid_fasta(tmp_path, text):
    path = tmp_path / "input.fa"
    path.write_text(text)
    with pytest.raises(OrganelleInputError):
        compare_assembly_to_reference(path, path)
