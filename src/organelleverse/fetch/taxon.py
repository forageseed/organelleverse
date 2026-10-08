"""Taxon-string normalization for the four alternate nuclear sources.

GIR/IMP/PGD/TAIR each need a clean ``(genus, species)`` pair to build a URL or
FTP path — unlike NCBI's own query, which already accepts free text, tax IDs,
and genus-level queries. ``resolve_taxon`` lives here (not in ``nuclear.py``,
which dispatches to the four source modules and would create a circular
import if it also owned the shared taxon-parsing they all need) and is
re-exported from ``nuclear.py`` for callers who expect it there.

Only well-formed binomials are accepted. A genus abbreviation (``G.
species``) is rejected rather than expanded — guessing the genus wrong would
silently misroute a fetch, and there is no genus lookup here to check a guess
against. An infraspecific qualifier (``subsp.``/``var.``/``cv.``) is stripped
to reach a species-level binomial, but that fallback is recorded on
``ResolvedTaxon.resolution``, never applied silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..core.errors import OrganelleParameterError

__all__ = ["ResolvedTaxon", "resolve_taxon"]

_INFRASPECIFIC = re.compile(r"\s+(subsp\.|var\.|cv\.)\s+.+$", re.IGNORECASE)
_BINOMIAL = re.compile(r"^([A-Z][a-z]+)\s+([A-Za-z][A-Za-z-]+)$")


@dataclass(frozen=True)
class ResolvedTaxon:
    """A taxon string reduced to the ``(genus, species)`` pair a source needs.

    ``resolution`` is ``"exact"`` when the input was already a well-formed
    binomial, or ``"species_fallback"`` when an infraspecific qualifier was
    stripped to get there.
    """

    requested: str
    genus: str
    species: str
    resolution: Literal["exact", "species_fallback"]

    @property
    def binomial(self) -> str:
        return f"{self.genus} {self.species}"


def resolve_taxon(taxon: str) -> ResolvedTaxon:
    """Reduce ``taxon`` to a ``(genus, species)`` pair, or reject it.

    Rejects anything that is not a well-formed binomial after normalizing
    whitespace and stripping a trailing infraspecific qualifier — including a
    genus abbreviation, which this function will not guess.
    """
    requested = taxon.strip()
    if not requested:
        raise OrganelleParameterError(
            code="input.unresolvable_taxon",
            message="taxon must not be blank",
            details={"taxon": taxon},
        )

    normalized = re.sub(r"\s+", " ", requested)
    stripped = _INFRASPECIFIC.sub("", normalized).strip()
    resolution: Literal["exact", "species_fallback"] = (
        "exact" if stripped == normalized else "species_fallback"
    )

    match = _BINOMIAL.match(stripped)
    if not match:
        raise OrganelleParameterError(
            code="input.unresolvable_taxon",
            message=f"{requested!r} is not a well-formed binomial",
            details={"taxon": requested, "normalized": stripped},
        )
    genus, species = match.group(1), match.group(2).lower()
    return ResolvedTaxon(requested=requested, genus=genus, species=species, resolution=resolution)
