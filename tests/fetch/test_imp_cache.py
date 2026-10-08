"""IMP seed-cache classification: self-consistency, not species verification.

See the design doc's "IMP cache cleaning" section for why a kept entry is
labeled ``confidence="heuristic_consistent"``, never ``"verified"``.
"""

from __future__ import annotations

import json

from organelleverse.fetch.imp_cache import (
    CLEANED_CACHE_PATH,
    DROPPED_CACHE_PATH,
    RAW_CACHE_PATH,
    classify_imp_cache,
    imp_id_formula,
    load_cleaned_cache,
)


def test_formula_is_genus_initial_plus_first_two_species_letters() -> None:
    assert imp_id_formula("Linum_usitatissimum") == "Lus"
    assert imp_id_formula("Artemisia_argyi") == "Aar"


def test_formula_rejects_a_name_without_two_parts() -> None:
    assert imp_id_formula("Linum") is None


def test_a_unique_matching_entry_is_kept_as_heuristic_consistent() -> None:
    kept, dropped = classify_imp_cache({"Linum_usitatissimum": "Lus1"})
    assert kept == {"Linum_usitatissimum": {"imp_id": "Lus1", "confidence": "heuristic_consistent"}}
    assert dropped == {}


def test_a_resolved_pair_keeps_the_matching_species_and_drops_the_other() -> None:
    kept, dropped = classify_imp_cache({"Linum_usitatissimum": "Lus1", "Luzula_sylvatica": "Lus1"})
    assert kept == {"Linum_usitatissimum": {"imp_id": "Lus1", "confidence": "heuristic_consistent"}}
    assert dropped == {"Luzula_sylvatica": {"imp_id": "Lus1", "reason": "resolved_pair_loser"}}


def test_a_collision_where_both_match_the_formula_drops_both() -> None:
    kept, dropped = classify_imp_cache({"Geum_urbanum": "Gur1", "Glycyrrhiza_uralensis": "Gur1"})
    assert kept == {}
    assert set(dropped) == {"Geum_urbanum", "Glycyrrhiza_uralensis"}
    assert all(entry["reason"] == "collision" for entry in dropped.values())


def test_a_collision_where_neither_matches_the_formula_drops_both() -> None:
    kept, dropped = classify_imp_cache({"Medicago_sativa": "Mes1", "Melastoma_sanguineum": "Mes1"})
    assert kept == {}
    assert set(dropped) == {"Medicago_sativa", "Melastoma_sanguineum"}


def test_a_standalone_mismatch_is_dropped_with_the_expected_prefix_recorded() -> None:
    kept, dropped = classify_imp_cache({"Matricaria_chamomilla": "Mac1"})
    assert kept == {}
    assert dropped == {
        "Matricaria_chamomilla": {
            "imp_id": "Mac1",
            "reason": "standalone_mismatch",
            "expected_prefix": "Mch",
        }
    }


def test_classifying_the_real_committed_raw_cache_reproduces_98_kept_23_dropped() -> None:
    with RAW_CACHE_PATH.open() as handle:
        raw = json.load(handle)
    kept, dropped = classify_imp_cache(raw)

    assert len(raw) == 121
    assert len(kept) == 98
    assert len(dropped) == 23
    assert len(kept) + len(dropped) == len(raw)

    # Named examples from the design doc, spot-checked directly.
    assert kept["Linum_usitatissimum"]["imp_id"] == "Lus1"
    assert "Luzula_sylvatica" not in kept
    assert dropped["Luzula_sylvatica"]["reason"] == "resolved_pair_loser"
    assert "Medicago_sativa" in dropped
    assert "Melastoma_sanguineum" in dropped
    assert all(entry["confidence"] == "heuristic_consistent" for entry in kept.values())


def test_the_committed_cleaned_and_dropped_files_match_a_fresh_classification() -> None:
    with RAW_CACHE_PATH.open() as handle:
        raw = json.load(handle)
    kept, dropped = classify_imp_cache(raw)

    with CLEANED_CACHE_PATH.open() as handle:
        committed_kept = json.load(handle)
    with DROPPED_CACHE_PATH.open() as handle:
        committed_dropped = json.load(handle)

    assert committed_kept == kept
    assert committed_dropped == dropped


def test_load_cleaned_cache_returns_the_committed_whitelist() -> None:
    cache = load_cleaned_cache()
    assert len(cache) == 98
    assert cache["Linum_usitatissimum"] == {"imp_id": "Lus1", "confidence": "heuristic_consistent"}
