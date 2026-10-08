"""Typed cores for comparative (OmicVerse-style data-in/data-out)."""

from __future__ import annotations
from .compare import _gene_order, _gene_set, _GENE_ALIASES


def compute_gene_sets(genomes) -> list[set[str]]:
    """Get gene sets for each genome. Returns list of sets."""
    return [_gene_set(g) for g in genomes]


def compute_gene_intersection(genomes) -> set[str]:
    """Get shared genes across genomes. Returns set."""
    sets = [_gene_set(g) for g in genomes]
    return set.intersection(*sets) if sets else set()


def normalize_gene_names(names: list[str]) -> dict[str, str]:
    """Normalize gene-name variants. Returns {input: canonical}."""
    out = {}
    for name in names:
        key = name.strip().lower()
        canonical = key
        for canon, aliases in _GENE_ALIASES.items():
            if key in aliases or key == canon.lower():
                canonical = canon
                break
        out[name] = canonical
    return out


def compute_synteny_score(genomes) -> dict:
    """Compute synteny (gene-order Jaccard). Returns dict."""
    orders = [_gene_order(g) for g in genomes]
    if len(orders) < 2:
        return {"synteny_blocks": 0, "order_jaccard": 1.0}
    s1 = {(orders[0][i], orders[0][i + 1]) for i in range(len(orders[0]) - 1)}
    s2 = {(orders[1][i], orders[1][i + 1]) for i in range(len(orders[1]) - 1)}
    j = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0
    return {"synteny_blocks": len(s1 & s2), "order_jaccard": round(j, 4)}
