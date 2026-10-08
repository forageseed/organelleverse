"""IMP seed-cache cleaning.

See docs/superpowers/specs/2026-07-19-nuclear-genome-alternate-sources-design.md
"IMP cache cleaning" for the full rationale. Short version: the prior
workflow's own probing script *creates* a cache entry by generating an ID
from ``Genus[0] + species[:2]`` and accepting the first HTTP 200 it finds —
so checking a cached entry against that same formula only proves the entry
is internally consistent with the process that made it, never that the ID
independently belongs to that species. 10 of the raw 121-entry cache's IDs
are claimed by two different species each (confirmed via ``sha256sum``
against the actual downloaded files: every colliding pair is byte-identical,
meaning at least one species in each pair got the wrong genome).

``classify_imp_cache`` is the one place this logic runs. The committed
``imp_id_mapping.cleaned.json``/``imp_id_mapping.dropped.json`` were produced
by running it once against the real raw cache; the regression test in
``tests/fetch/test_imp_cache.py`` re-runs it and asserts the committed files
still match, catching any future accidental corruption of the raw seed file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "CLEANED_CACHE_PATH",
    "DROPPED_CACHE_PATH",
    "RAW_CACHE_PATH",
    "classify_imp_cache",
    "imp_id_formula",
    "load_cleaned_cache",
]

_DATA_DIR = Path(__file__).parent / "data"
RAW_CACHE_PATH = _DATA_DIR / "imp_id_mapping.raw.json"
CLEANED_CACHE_PATH = _DATA_DIR / "imp_id_mapping.cleaned.json"
DROPPED_CACHE_PATH = _DATA_DIR / "imp_id_mapping.dropped.json"


def imp_id_formula(species: str) -> str | None:
    """``Genus[0] + species[:2]`` — the prior workflow's own ID-derivation rule.

    ``species`` is the cache's own key format: ``Genus_species`` (e.g.
    ``"Linum_usitatissimum"``).
    """
    parts = species.split("_")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return parts[0][0].upper() + parts[1][:2].lower()


def classify_imp_cache(
    raw: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Split ``{species: imp_id}`` into ``(kept, dropped)`` by formula self-consistency.

    - An ID claimed by exactly one species, matching the formula: kept.
    - An ID claimed by two species where exactly one matches the formula:
      the matching one is kept, the other dropped as ``"resolved_pair_loser"``.
    - An ID claimed by two-or-more species where the formula cannot pick a
      single winner (all match, or none match): every claimant is dropped as
      ``"collision"`` — the formula genuinely cannot resolve this case.
    - A species whose ID has no colliding partner but still does not match
      its own formula: dropped as ``"standalone_mismatch"``.

    Every kept entry is tagged ``confidence="heuristic_consistent"`` — see
    this module's docstring for why not ``"verified"``.
    """
    by_id: dict[str, list[str]] = {}
    for species, imp_id in raw.items():
        by_id.setdefault(imp_id, []).append(species)

    kept: dict[str, dict[str, Any]] = {}
    dropped: dict[str, dict[str, Any]] = {}

    for imp_id, species_list in by_id.items():
        prefix = imp_id[:-1]
        matches = [sp for sp in species_list if imp_id_formula(sp) == prefix]
        nonmatches = [sp for sp in species_list if imp_id_formula(sp) != prefix]

        if len(species_list) == 1:
            species = species_list[0]
            if matches:
                kept[species] = {"imp_id": imp_id, "confidence": "heuristic_consistent"}
            else:
                dropped[species] = {
                    "imp_id": imp_id,
                    "reason": "standalone_mismatch",
                    "expected_prefix": imp_id_formula(species),
                }
            continue

        if len(matches) == 1:
            winner = matches[0]
            kept[winner] = {"imp_id": imp_id, "confidence": "heuristic_consistent"}
            for loser in nonmatches:
                dropped[loser] = {"imp_id": imp_id, "reason": "resolved_pair_loser"}
            continue

        for species in species_list:
            dropped[species] = {"imp_id": imp_id, "reason": "collision"}

    return kept, dropped


def load_cleaned_cache() -> dict[str, dict[str, Any]]:
    """The committed, cleaned whitelist — every entry ``confidence="heuristic_consistent"``."""
    with CLEANED_CACHE_PATH.open() as handle:
        data: dict[str, dict[str, Any]] = json.load(handle)
    return data
