"""Progress and the optional time budget of the genome-scale tRNA search."""

import logging
import time
from pathlib import Path

import pytest

from organelleverse.annotation.cmsearch import genome
from organelleverse.annotation.cmsearch.models import CMHit


def _stub_search(monkeypatch, windows, delay=0.0, seen=None):
    monkeypatch.setattr(genome, "_nhmmer_windows", lambda *_: windows)

    def cm_search(sequence, model, *, min_bits):
        if seen is not None:
            seen.append(len(sequence))
        if delay:
            time.sleep(delay)
        return [CMHit(start=1, end=len(sequence), strand=1, score=50.0, cm_from=1, cm_to=70)]

    monkeypatch.setattr(genome, "cm_search", cm_search)


def _run(sequence="ACGT" * 100):
    return genome.find_trnas(sequence, cm_model=object(), hmm_path=Path("unused"), pad=0)


def test_every_window_is_aligned_without_a_budget(monkeypatch, caplog):
    monkeypatch.delenv(genome.BUDGET_ENV, raising=False)
    seen = []
    _stub_search(monkeypatch, [(1, 70), (101, 170), (201, 270)], seen=seen)
    with caplog.at_level(logging.INFO, logger=genome.__name__):
        hits = _run()
    assert len(seen) == 3 and len(hits) == 3
    assert any("3 candidate windows" in r.message for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_a_zero_budget_skips_every_window_and_says_so(monkeypatch, caplog):
    monkeypatch.setenv(genome.BUDGET_ENV, "0")
    seen = []
    _stub_search(monkeypatch, [(1, 70), (101, 170)], seen=seen)
    with caplog.at_level(logging.INFO, logger=genome.__name__):
        hits = _run()
    assert seen == [] and hits == []
    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    assert "2 of 2 candidate windows were not aligned" in warning.message


def test_the_budget_stops_between_windows(monkeypatch, caplog):
    monkeypatch.setenv(genome.BUDGET_ENV, "0.05")
    seen = []
    _stub_search(monkeypatch, [(i * 100 + 1, i * 100 + 70) for i in range(6)], delay=0.06, seen=seen)
    with caplog.at_level(logging.INFO, logger=genome.__name__):
        hits = _run("ACGT" * 200)
    assert 1 <= len(seen) < 6 and len(hits) == len(seen)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("raw", ["", "soon"])
def test_an_unusable_budget_means_no_limit(monkeypatch, raw):
    monkeypatch.setenv(genome.BUDGET_ENV, raw)
    assert genome._deadline() is None
