"""resolve_taxon: only well-formed binomials, no silent genus guessing.

See docs/superpowers/specs/2026-07-19-nuclear-genome-alternate-sources-design.md
"Known gaps" #4 — expanding a genus abbreviation risks a silently wrong guess,
so it is rejected rather than resolved.
"""

from __future__ import annotations

import pytest

from organelleverse.core.errors import OrganelleParameterError
from organelleverse.fetch.taxon import resolve_taxon


def test_a_clean_binomial_resolves_exactly() -> None:
    resolved = resolve_taxon("Malus domestica")
    assert resolved.genus == "Malus"
    assert resolved.species == "domestica"
    assert resolved.binomial == "Malus domestica"
    assert resolved.resolution == "exact"
    assert resolved.requested == "Malus domestica"


def test_extra_whitespace_is_normalized_but_still_exact() -> None:
    resolved = resolve_taxon("  Malus   domestica  ")
    assert resolved.binomial == "Malus domestica"
    assert resolved.resolution == "exact"


@pytest.mark.parametrize(
    "raw,binomial",
    [
        ("Brassica napus subsp. napus", "Brassica napus"),
        ("Zea mays var. mays", "Zea mays"),
        ("Camellia sinensis cv. assamica", "Camellia sinensis"),
    ],
)
def test_infraspecific_qualifiers_are_stripped_and_marked_as_fallback(
    raw: str, binomial: str
) -> None:
    resolved = resolve_taxon(raw)
    assert resolved.binomial == binomial
    assert resolved.resolution == "species_fallback"
    assert resolved.requested == raw


def test_a_genus_abbreviation_is_rejected_not_guessed() -> None:
    with pytest.raises(OrganelleParameterError) as raised:
        resolve_taxon("A. thaliana")
    assert raised.value.code == "input.unresolvable_taxon"


def test_a_bare_genus_is_rejected() -> None:
    with pytest.raises(OrganelleParameterError) as raised:
        resolve_taxon("Malus")
    assert raised.value.code == "input.unresolvable_taxon"


def test_blank_input_is_rejected() -> None:
    with pytest.raises(OrganelleParameterError) as raised:
        resolve_taxon("   ")
    assert raised.value.code == "input.unresolvable_taxon"


def test_species_is_lowercased_regardless_of_input_casing() -> None:
    resolved = resolve_taxon("Malus DOMESTICA")
    assert resolved.species == "domestica"
