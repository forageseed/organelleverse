"""Species identification for gene-tree leaf labels (one rule for every stage).

Leaf labels carry a species and a gene, joined by ``__``, ``|``, ``@`` or a
single ``_`` (``Arabidopsis_thaliana__AT1G01010``, ``Ath|g1``, ``Ath_g1``).
Cutting at the first ``_`` is only right for codes without underscores; on
two-word species names it collapses congeners (``Arabidopsis_thaliana`` and
``Arabidopsis_lyrata`` both become ``Arabidopsis``) and silently assigns
distinct species edges the same value.

When the species set is known (species-tree or master-tree leaves) the label is
resolved against it by longest species-name prefix, which is unambiguous even
for ``Genus_species_GENE`` labels. Only when no species set is available does
the explicit-delimiter fallback apply.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

__all__ = ["species_resolver", "split_species_label"]

#: Delimiters that may follow a known species name in a leaf label.
_AFTER_SPECIES = ("__", "_", "|", "@")
#: Delimiters that are unambiguous without knowing the species set.
_EXPLICIT = ("__", "|", "@")


def split_species_label(label: str) -> str:
    """Species part of ``label`` when the species set is not known.

    The earliest explicit delimiter (``__``, ``|``, ``@``) wins, so a
    two-word species name survives. A label with none of them is cut at its
    first ``_`` (species codes such as ``Brsc_Bra001``), and a label with no
    delimiter at all is its own species.
    """
    cuts = [pos for sep in _EXPLICIT if (pos := label.find(sep)) > 0]
    if cuts:
        return label[: min(cuts)]
    head, sep, _ = label.partition("_")
    return head if sep and head else label


def species_resolver(
    known_species: Iterable[str] | None = None,
    species_map: Mapping[str, str] | None = None,
) -> Callable[[str], str]:
    """Build ``leaf label -> species`` for one analysis.

    Resolution order: an explicit ``species_map`` entry, an exact known
    species name, the longest known species name followed by a delimiter, and
    finally :func:`split_species_label`.
    """
    explicit = dict(species_map or {})
    known = {name for name in (known_species or ()) if name}
    by_length = sorted(known, key=len, reverse=True)
    cache: dict[str, str] = {}

    def resolve(label: str) -> str:
        hit = cache.get(label)
        if hit is not None:
            return hit
        if label in explicit:
            out = explicit[label]
        elif label in known:
            out = label
        else:
            out = next(
                (
                    name
                    for name in by_length
                    if label.startswith(name) and label[len(name) :].startswith(_AFTER_SPECIES)
                ),
                "",
            ) or split_species_label(label)
        cache[label] = out
        return out

    return resolve
