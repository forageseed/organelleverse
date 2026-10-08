"""Gate behaviour for the per-gene tuned tables (``_tuning.tuned``).

The 2026-08-04 solver spec establishes that per-gene constant tables inflate
accuracy on their fitting set and act as systematic error elsewhere, so they
are off by default and re-enabled only via ``ORG_VERSE_MITO_TUNING``. These
tests pin both postures at the merge boundary of the solver migration.
"""

from __future__ import annotations

import pytest

from organelleverse.annotation.mitochondrion import boundary, pcg


@pytest.fixture
def full_tuning(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORG_VERSE_MITO_TUNING", "full")


class TestDefaultOff:
    """Default level ``none``: no per-gene table influences the result."""

    def test_min_score_falls_back_to_config(self) -> None:
        config = pcg.PCGConfig()
        assert "rps1" in pcg.PER_GENE_MIN_SCORES  # table entry exists but is gated
        assert pcg._get_min_score("rps1", config) == config.min_score

    def test_presence_rules_do_not_filter(self) -> None:
        keep, reason = pcg._should_filter_by_presence_rules("rps19", 10.0, 10, "")
        assert keep is False
        assert reason == ""

    def test_length_validation_is_neutral(self) -> None:
        assert pcg._length_validation("cox1", 10000) == (0.0, [])

    def test_gene_span_accepts_any_span(self) -> None:
        assert pcg._validate_gene_span("cox1", 1, 10_000_000) is True

    def test_variable_pcg_list_inactive(self) -> None:
        assert pcg._is_variable_pcg("rps3") is False

    def test_search_range_uses_default(self) -> None:
        assert boundary._get_gene_search_range("atp1", 250) == 250


class TestFullTuningRestored:
    """``ORG_VERSE_MITO_TUNING=full`` reproduces the pre-2026-08-03 behaviour."""

    def test_min_score_uses_table(self, full_tuning) -> None:
        config = pcg.PCGConfig()
        assert pcg._get_min_score("rps1", config) == pcg.PER_GENE_MIN_SCORES["rps1"]

    def test_presence_rules_filter(self, full_tuning) -> None:
        keep, reason = pcg._should_filter_by_presence_rules("rps19", 10.0, 10, "")
        assert keep is True
        assert "rps19" in reason

    def test_variable_pcg_list_active(self, full_tuning) -> None:
        assert pcg._is_variable_pcg("rps3") is True

    def test_search_range_conservative(self, full_tuning) -> None:
        assert boundary._get_gene_search_range("atp1", 250) == 100
