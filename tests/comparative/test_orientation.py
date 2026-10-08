"""Orientation invariance, explicit failures and replayable transformations."""

import builtins
import random

import pytest
from Bio.Seq import reverse_complement as rc

from organelleverse._bio import write_fasta
from organelleverse.comparative import normalize_plastome_orientation
from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError


def plastome():
    rng = random.Random(113)
    lsc, ir, ssc = ["".join(rng.choices("ACGT", k=n)) for n in (9000, 2000, 4000)]
    # Five mismatches at each junction give the existing detector exact endpoints.
    lsc = "AAAAA" + lsc[5:-5] + "AAAAA"
    ssc = "AAAAA" + ssc[5:-5] + "AAAAA"
    return lsc, ir, ssc


def test_origin_strands_and_operation_replay(tmp_path):
    lsc, ir, ssc = plastome()
    original = lsc + ir + ssc + rc(ir)
    samples = [
        ("ref", original),
        ("ssc", lsc + ir + rc(ssc) + rc(ir)),
        ("reverse", rc(original)),
        ("both", rc(lsc + ir + rc(ssc) + rc(ir))),
    ]
    for offset in (200, 9500, 11500, 15500):
        samples.append((f"rotated_{offset}", original[offset:] + original[:offset]))
    path = write_fasta(tmp_path / "input.fa", samples)
    ref = write_fasta(tmp_path / "ref.fa", [("ref", original)])
    result = normalize_plastome_orientation(path, reference_fasta=ref)
    assert result.status == "ok"
    assert {seq for _, seq in result.metrics["sequences"]} == {original}
    for sample, (name, normalized) in zip(
        result.metrics["samples"], result.metrics["sequences"], strict=True
    ):
        seq = dict(samples)[name]
        for op in sample["operations"]:
            if op["operation"] == "rotate_left":
                seq = seq[op["bases"] :] + seq[: op["bases"]]
            else:
                seq = seq[: op["start"]] + rc(seq[op["start"] : op["end"]]) + seq[op["end"] :]
        assert seq == normalized
    again = normalize_plastome_orientation(
        write_fasta(tmp_path / "again.fa", result.metrics["sequences"]), reference_fasta=ref
    )
    assert again.metrics["sequences"] == result.metrics["sequences"]
    assert all(
        not s["ssc_reverse_complemented"] and not s["lsc_reverse_complemented"]
        for s in again.metrics["samples"]
    )


def test_majority_not_first_and_tie(tmp_path):
    lsc, ir, ssc = plastome()
    a, b = lsc + ir + ssc + rc(ir), rc(lsc) + ir + rc(ssc) + rc(ir)
    path = write_fasta(tmp_path / "input.fa", [("a", a), ("b", b), ("c", b)])
    result = normalize_plastome_orientation(path)
    assert result.metrics["target_strands"] == {"LSC": -1, "SSC": -1}
    assert result.metrics["samples"][0]["ssc_reverse_complemented"]
    assert result.metrics["samples"][0]["lsc_reverse_complemented"]
    tie = normalize_plastome_orientation(write_fasta(tmp_path / "tie.fa", [("a", a), ("b", b)]))
    assert tie.metrics["target_strands"] == {"LSC": 1, "SSC": 1}


def test_irless_and_unrelated_are_reported(tmp_path):
    lsc, ir, ssc = plastome()
    other = random.Random(22)
    unrelated = (
        "".join(other.choices("ACGT", k=9000))
        + ir
        + "".join(other.choices("ACGT", k=4000))
        + rc(ir)
    )
    path = write_fasta(
        tmp_path / "input.fa",
        [("ok", lsc + ir + ssc + rc(ir)), ("irless", lsc), ("unrelated", unrelated)],
    )
    result = normalize_plastome_orientation(path)
    assert result.status == "warning"
    assert len(result.metrics["sequences"]) == 1
    assert {s["reason"] for s in result.metrics["unresolved"]} == {
        "quadripartite_not_identified",
        "orientation_anchors_unresolved",
    }
    assert (
        normalize_plastome_orientation(write_fasta(tmp_path / "none.fa", [("none", lsc)])).status
        == "failed"
    )


def test_missing_optional_dependency(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "mappy":
            raise ImportError("absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(OrganelleDependencyError, match="organelleverse\\[align\\]"):
        normalize_plastome_orientation(tmp_path / "input.fa")


def test_rejects_alignment_input(tmp_path):
    path = write_fasta(tmp_path / "input.fa", [("a", "ACGT-N")])
    with pytest.raises(OrganelleInputError, match="ungapped"):
        normalize_plastome_orientation(path)
