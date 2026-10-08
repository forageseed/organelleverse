"""Scientific invariants for sequence rearrangements and their evidence."""

import builtins
import random

import pytest
from Bio.Seq import reverse_complement as rc

from organelleverse._bio import write_fasta
from organelleverse.comparative import detect_structural_variants
from organelleverse.comparative.structural import _backbone, _classify
from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError


def block(id, r, q, size=1000, strand=1, cigar=None):
    return dict(
        id=id,
        reference_start=r,
        reference_end=r + size,
        query_start=q,
        query_end=q + size,
        strand=strand,
        cigar=cigar or [[size, "M"]],
    )


def test_backbone_is_maximum_weight_not_longest_single_block():
    blocks = [block("a", 0, 0), block("b", 1000, 1000), block("c", 0, 2000, 1500)]
    assert _backbone(blocks) == {"a", "b"}


def test_classification_and_two_coordinate_axes():
    blocks = [
        block("a", 0, 0, 5000),
        block("b", 5000, 7000, 2000, -1),
        block("c", 7000, 5000, 1000),
        block("d", 9000, 9000, 5000),
        block("copy", 0, 14000, 5000),
    ]
    events, _ = _classify(blocks, 100)
    by_type = {e["type"]: e for e in events}
    assert set(by_type) == {"inversion", "duplication"}  # c lies on the backbone
    assert by_type["inversion"]["query_start"] == 7000
    assert by_type["duplication"]["supporting_blocks"] == ["a", "copy"]
    trans = [
        block("a", 0, 0, 5000),
        block("b", 5000, 10000, 1000),
        block("c", 6000, 5000, 5000),
        block("d", 11000, 11000, 5000),
    ]
    events, _ = _classify(trans, 100)
    assert [e["supporting_blocks"] for e in events if e["type"] == "translocation"] == [["b"]]


def test_reverse_cigar_indels_have_correct_query_coordinates():
    b = block("a", 100, 200, strand=-1, cigar=[[300, "M"], [100, "I"], [100, "D"], [600, "M"]])
    events, _ = _classify([b], 100)
    ins = next(e for e in events if e["type"] == "insertion")
    deletion = next(e for e in events if e["type"] == "deletion")
    assert (ins["reference_start"], ins["reference_end"], ins["query_start"], ins["query_end"]) == (
        400,
        400,
        800,
        900,
    )
    assert (
        deletion["reference_start"],
        deletion["reference_end"],
        deletion["query_start"],
        deletion["query_end"],
    ) == (400, 500, 800, 800)


@pytest.fixture
def sequences(tmp_path):
    rng = random.Random(177)
    seq = "".join(rng.choices("ACGT", k=22000))
    ref = write_fasta(tmp_path / "ref.fa", [("ref", seq)])

    def run(query, **kwargs):
        q = write_fasta(tmp_path / "query.fa", [("query", query)])
        return detect_structural_variants(ref, q, **kwargs)

    return seq, run


def test_circular_origin_and_global_reverse_are_not_variants(sequences):
    seq, run = sequences
    for query in (seq, seq[5000:] + seq[:5000], rc(seq[5000:] + seq[:5000])):
        result = run(query)
        assert result.metrics["counts"] == {}
        assert result.metrics["coverage"]["query"]["uncovered_bases"] == 0


def test_real_mappy_inversion_and_plot_contract(sequences):
    from organelleverse.visualization import summarize_synteny_matrix

    seq, run = sequences
    result = run(seq[:7000] + rc(seq[7000:11000]) + seq[11000:], topology="linear")
    inversions = [v for v in result.metrics["variants"] if v["type"] == "inversion"]
    assert len(inversions) == 1
    assert abs(inversions[0]["reference_start"] - 7000) < 10
    assert abs(inversions[0]["reference_end"] - 11000) < 10
    assert summarize_synteny_matrix(result)


@pytest.mark.parametrize("reverse", [False, True])
def test_plastome_ssc_isomer_and_rotation(tmp_path, reverse):
    from tests.comparative.test_orientation import plastome

    lsc, ir, ssc = plastome()
    seq = lsc + ir + ssc + rc(ir)
    query = lsc + ir + rc(ssc) + rc(ir)
    if reverse:
        query = rc(query)
    ref = write_fasta(tmp_path / "ref.fa", [("ref", seq)])
    q = write_fasta(tmp_path / "q.fa", [("q", query[1200:] + query[:1200])])
    result = detect_structural_variants(ref, q, plastome=True)
    assert result.metrics["variants"] == ()
    assert result.metrics["orientation"]["ssc"]["reversed"]


def test_no_homology_fails(sequences):
    _, run = sequences
    with pytest.raises(OrganelleInputError, match="No shared anchors"):
        run("A" * 22000)


def test_optional_dependency_error(monkeypatch):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "mappy":
            raise ImportError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(OrganelleDependencyError, match="organelleverse\\[align\\]"):
        detect_structural_variants("ref.fa", "query.fa")


def test_partial_repeat_does_not_hide_an_inverted_unique_extension():
    anchors = [block("a", 0, 0), block("b", 900, 2000, 2000, -1)]
    events, _ = _classify(anchors, 100)
    assert {e["type"] for e in events} == {"duplication", "inversion"}
    assert next(e for e in events if e["type"] == "duplication")["reference_length"] == 100
