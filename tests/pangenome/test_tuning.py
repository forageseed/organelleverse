from __future__ import annotations

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.pangenome.tuning import (
    RepeatInterval,
    estimate_identity,
    estimate_segment_length,
    parse_mash_distances,
    parse_repeatmasker_output,
)


def test_mash_identity_uses_pinned_formula_and_clamps() -> None:
    assert estimate_identity((0.0, 0.041), margin=2) == 94
    assert estimate_identity((0.75,), margin=0, minimum=50, maximum=100) == 50
    assert estimate_identity((0.0,), margin=0, minimum=50, maximum=99) == 99


def test_repeat_estimator_merges_only_within_query_and_category() -> None:
    intervals = (
        RepeatInterval("plastome", 10, 30, "interspersed"),
        RepeatInterval("plastome", 31, 50, "interspersed"),
        RepeatInterval("plastome", 40, 90, "simple"),
        RepeatInterval("mitome", 1, 100, "rna"),
    )

    estimate = estimate_segment_length(
        intervals,
        multiplier=1.2,
        round_to=10,
        no_repeat_fallback=5000,
        include_rna=False,
    )

    assert estimate.segment_length == 70
    assert estimate.longest_span == 51
    assert estimate.fallback_used is False
    assert estimate.category_spans == {"interspersed": 41, "simple": 51}


def test_strict_mash_parser_rejects_malformed_and_non_finite_values() -> None:
    assert parse_mash_distances("a.msh\tb.msh\t0.041\t0\t100/1000\n") == (0.041,)
    for output in (
        "a.msh\tb.msh\tnan\t0\t100/1000\n",
        "a.msh\tb.msh\t0.1\n",
        "",
    ):
        with pytest.raises(OrganelleInputError, match="Mash"):
            parse_mash_distances(output)


@pytest.mark.parametrize(
    "output",
    (
        "a.msh\tb.msh\t0.1\tBROKEN\t100/1000\n",
        "a.msh\tb.msh\t0.1\t1.1\t100/1000\n",
        "a.msh\tb.msh\t0.1\t0\tBROKEN\n",
        "a.msh\tb.msh\t0.1\t0\t1001/1000\n",
        "a.msh\tb.msh\t0.1\t0\t1/0\n",
    ),
)
def test_strict_mash_parser_validates_p_value_and_shared_hash_count(output: str) -> None:
    with pytest.raises(OrganelleInputError, match="Mash"):
        parse_mash_distances(output)


def test_repeatmasker_parser_normalizes_coordinates_and_rejects_bad_rows() -> None:
    output = """
 SW   perc perc perc  query       position in query     matching repeat
score  div. del. ins. sequence    begin end (left) strand repeat class/family
  463  1.3  0.0  0.0  plastome  50  10  (0)  +  r1  DNA/hAT  1  40  (0)  1
  201  0.0  0.0  0.0  plastome  60  70  (0)  +  r2  Simple_repeat  1  11  (0)  2
  100  0.0  0.0  0.0  plastome  80  90  (0)  +  r3  tRNA  1  11  (0)  3
"""
    assert parse_repeatmasker_output(output) == (
        RepeatInterval("plastome", 10, 50, "interspersed"),
        RepeatInterval("plastome", 60, 70, "simple"),
        RepeatInterval("plastome", 80, 90, "rna"),
    )

    with pytest.raises(OrganelleInputError, match="RepeatMasker"):
        parse_repeatmasker_output("463 1.3 0.0 0.0 plastome BAD 10 (0) + r1 DNA/hAT\n")


@pytest.mark.parametrize(
    "output",
    (
        "BROKEN 1.3 0.0 0.0 plastome 10 50 (0) + r1 DNA/hAT 1 41 (0) 1\n",
        "unexpected RepeatMasker prose that is not a header\n",
    ),
)
def test_repeatmasker_parser_only_skips_recognized_headers(output: str) -> None:
    with pytest.raises(OrganelleInputError, match="RepeatMasker"):
        parse_repeatmasker_output(output)


def test_repeatmasker_parser_allows_real_headers_and_separator() -> None:
    assert (
        parse_repeatmasker_output(
            "SW perc perc perc query position in query matching repeat\n"
            "score div. del. ins. sequence begin end (left) strand repeat class/family\n"
            "------------------------------\n"
        )
        == ()
    )


def test_estimators_fail_closed_and_no_repeat_fallback_is_explicit() -> None:
    with pytest.raises(OrganelleInputError):
        estimate_identity(())
    with pytest.raises(OrganelleInputError):
        estimate_identity((float("inf"),))
    with pytest.raises(OrganelleInputError):
        estimate_segment_length((), no_repeat_fallback=0)

    estimate = estimate_segment_length((), no_repeat_fallback=4321)
    assert estimate.segment_length == 4321
    assert estimate.longest_span == 0
    assert estimate.fallback_used is True


def test_repeatmasker_accepts_standalone_real_no_hits_report():
    assert parse_repeatmasker_output(
        "There were no repetitive sequences detected in /input/pansn.fa\n"
    ) == ()


@pytest.mark.parametrize("suffix", ["", "pansn.fa\nSW perc perc perc query"])
def test_repeatmasker_no_hits_report_requires_filename_and_no_table(suffix):
    with pytest.raises(OrganelleInputError, match="RepeatMasker"):
        parse_repeatmasker_output("There were no repetitive sequences detected in " + suffix)
