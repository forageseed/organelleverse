"""Runtime switch for the benchmark-tuned per-gene tables in this package.

The mitochondrial backend carries per-gene constant tables that were fitted to a
specific validation set (see ``FIXED_OFFSET_GENES``, ``PER_GENE_MIN_SCORES``,
``_should_filter_by_presence_rules`` and friends). They inflate accuracy on the
species they were fitted to and act as systematic error elsewhere, so their real
contribution has to be measurable before they can be removed.

``ORG_VERSE_MITO_TUNING`` selects how many of them are active:

``full``
    Everything on — the behaviour shipped before 2026-08-03.
``no_offsets``
    Coordinate compensation off: fixed per-gene offsets, per-gene start-codon
    sets, per-gene search ranges, special-gene handling.
``no_length_gates``
    ``no_offsets`` plus every per-gene expected-length table and span check.
``none`` (default)
    Also drops per-gene score thresholds, presence rules, and the variable-copy
    gene list. Nothing per-gene-tuned remains.

The levels are cumulative, so a run at ``none`` is the honest "no tuned tables"
baseline that a species outside the fitting set would actually see.

``none`` is the default because measuring the four levels on PMGA Ranunculus —
the very genome the tables were fitted to, so the most favourable case they get
— showed the tables buy 5 exactly-correct intervals (135 vs 130, +3.7%) while
raising mean boundary error by half (16.9 → 25.4 bp). Outside the fitting set
only the cost remains.
Set ``ORG_VERSE_MITO_TUNING=full`` to reproduce the old behaviour.
"""

from __future__ import annotations

import os

_ENV = "ORG_VERSE_MITO_TUNING"

#: Cumulative levels, weakest tuning last.
_LEVELS = ("full", "no_offsets", "no_length_gates", "none")

#: Which categories each level still applies.
_ACTIVE: dict[str, frozenset[str]] = {
    "full": frozenset({"offsets", "length_gates", "score_gates"}),
    "no_offsets": frozenset({"length_gates", "score_gates"}),
    "no_length_gates": frozenset({"score_gates"}),
    "none": frozenset(),
}


#: Level applied when the environment variable is unset or unrecognised.
DEFAULT_LEVEL = "none"


def tuning_level() -> str:
    """Return the configured level, falling back to :data:`DEFAULT_LEVEL`."""
    value = os.environ.get(_ENV, "").strip().lower()
    return value if value in _LEVELS else DEFAULT_LEVEL


def tuned(category: str) -> bool:
    """Return True when ``category``'s per-gene tables should still be applied.

    Args:
        category: ``offsets``, ``length_gates``, or ``score_gates``.
    """
    return category in _ACTIVE[tuning_level()]
