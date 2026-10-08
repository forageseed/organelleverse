import pytest

from organelleverse.pangenome.reference_binding import (
    audit_reference_alleles,
    exact_circular_transform,
)


@pytest.mark.parametrize(
    "source,target,strand,offset,positions",
    [
        ("AACGAT", "GATAAC", "+", 3, [3, 4, 5, 0, 1, 2]),
        ("AACGAT", "CGTTAT", "-", 2, [3, 2, 1, 0, 5, 4]),
    ],
)
def test_rotation_maps_every_base_in_both_orientations(source, target, strand, offset, positions):
    binding = exact_circular_transform(source, target)
    assert (binding.strand, binding.offset) == (strand, offset)
    assert [binding.source_position(i) for i in range(len(target))] == positions
    reconstructed = "".join(source[binding.source_position(i)] for i in range(len(target)))
    if strand == "-":
        reconstructed = reconstructed.translate(str.maketrans("ACGT", "TGCA"))
    assert reconstructed == target
    with pytest.raises(ValueError, match="outside"):
        binding.source_position(len(target))


@pytest.mark.parametrize(
    "source,target,reason",
    [
        ("AACGAT", "AACGATT", "equal-length"),
        ("AACGAT", "AACGAC", "differ beyond"),
        ("ACACAC", "CACACA", "ambiguous"),
        ("ACGT", "ACGT", "ambiguous"),
        ("AACGNT", "AACGNT", "A/C/G/T"),
    ],
)
def test_changed_or_ambiguous_reference_never_creates_a_coordinate_map(source, target, reason):
    with pytest.raises(ValueError, match=reason):
        exact_circular_transform(source, target)


def test_audit_preserves_errors_and_does_not_wrap_or_accept_boolean_positions():
    audit = audit_reference_alleles(
        "AACGAT",
        [
            {"pos": 1, "ref": "aa"},
            {"pos": 2, "ref": "C"},
            {"pos": 6, "ref": "TA"},
            {"pos": True, "ref": "A"},
            {"pos": 1, "ref": ""},
            {"pos": 1, "ref": "N"},
        ],
    )
    assert audit["status"] == "failed"
    assert audit["counts"] == {
        "matched": 1,
        "mismatched": 1,
        "outside_reference": 1,
        "unsupported_literal_reference_allele": 3,
    }
    assert len(audit["rows"]) == 6
    assert audit["rows"][1]["observed_allele"] == "A"
    assert audit["rows"][2]["observed_allele"] is None


def test_audit_pass_requires_nonempty_complete_agreement():
    assert audit_reference_alleles("AACGAT", [])["status"] == "failed"
    assert audit_reference_alleles("AACGAT", [{"pos": 3, "ref": "CGAT"}])["status"] == "passed"
