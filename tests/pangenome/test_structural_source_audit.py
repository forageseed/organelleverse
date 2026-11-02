import pytest

from scripts.pangenome_benchmarks.audit_structural_table import audit, directed_interval


def test_literal_inclusive_coordinates_and_reverse_strand():
    assert directed_interval("AACCGGTT", 2, 5) == "ACCG"
    assert directed_interval("AACCGGTT", 5, 2) == "CGGT"
    with pytest.raises(ValueError):
        directed_interval("AACCGGTT", 0, 5)


def test_unequal_repeat_endpoints_are_not_repaired_or_reference_selected():
    row = dict(
        annotation="A vs. Col-0",
        chord_type="repeat",
        seq_1="Col-0",
        seq_2="Col-0",
        start_1=1,
        end_1=3,
        start_2=4,
        end_2=5,
        sheet_row=3,
    )
    result = audit([row], {"one": "AAAAAA", "two": "CCCCCC"})
    assert len(result["literal_col0_repeat_checks"]) == 2
    assert all(
        r["first_length"] == 3
        and r["second_length"] == 2
        and not r["exact_directed_sequence_match"]
        for r in result["literal_col0_repeat_checks"]
    )
