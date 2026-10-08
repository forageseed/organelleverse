"""Bipartition-based tree comparison (Robinson-Foulds distance).

Pure-Python Newick/NEXUS split extraction shared by :func:`compare_trees` and
the MrBayes ASDSF recomputation in :mod:`.mrbayes`. Trees are compared as
*unrooted* topologies: every internal edge is one bipartition of the leaf set,
trivial splits (one side < 2 taxa) are ignored, and each split is stored
canonically as the side that does not contain the alphabetically first taxon.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from ._results import findings, ok_result, provenance

__all__ = [
    "compare_trees",
    "newick_clades",
    "read_tree",
    "robinson_foulds",
    "tree_splits",
]

_COMMENT = re.compile(r"\[[^\]]*\]")


def newick_clades(newick: str) -> tuple[list[str], list[tuple[frozenset[str], str | None]]]:
    """Return ``(leaves, [(clade_leaf_set, internal_label), ...])`` of a Newick string.

    Internal labels (e.g. bootstrap or posterior support) are returned as the
    raw string, or ``None`` when absent. Branch lengths are ignored.
    """
    text = _COMMENT.sub("", newick).strip()
    if text.endswith(";"):
        text = text[:-1]
    leaves: list[str] = []
    clades: list[tuple[frozenset[str], str | None]] = []
    stack: list[list[str]] = []
    i = 0
    n = len(text)
    current: list[str] | None = None

    def read_token(start: int) -> tuple[str, int]:
        if start < n and text[start] == "'":
            end = start + 1
            while end < n and text[end] != "'":
                end += 1
            return text[start + 1 : end], end + 1
        end = start
        while end < n and text[end] not in ",():;":
            end += 1
        return text[start:end].strip(), end

    def skip_length(start: int) -> int:
        if start < n and text[start] == ":":
            start += 1
            while start < n and text[start] not in ",()":
                start += 1
        return start

    while i < n:
        char = text[i]
        if char == "(":
            stack.append([])
            i += 1
        elif char == ",":
            i += 1
        elif char == ")":
            if not stack:
                raise OrganelleInputError(
                    code="phylogeny.tree.bad_newick", message="unbalanced parentheses"
                )
            members = stack.pop()
            label, i = read_token(i + 1)
            i = skip_length(i)
            clades.append((frozenset(members), label or None))
            if stack:
                stack[-1].extend(members)
            else:
                current = members
        elif char.isspace():
            i += 1
        else:
            name, i = read_token(i)
            i = skip_length(i)
            if not name:
                raise OrganelleInputError(
                    code="phylogeny.tree.bad_newick", message="empty leaf name"
                )
            leaves.append(name)
            if stack:
                stack[-1].append(name)
    if stack:
        raise OrganelleInputError(code="phylogeny.tree.bad_newick", message="unbalanced tree")
    del current
    if len(set(leaves)) != len(leaves):
        raise OrganelleInputError(
            code="phylogeny.tree.duplicate_leaves", message="tree has duplicate leaf names"
        )
    return leaves, clades


def tree_splits(newick: str) -> tuple[frozenset[str], dict[frozenset[str], str | None]]:
    """Return ``(taxa, {canonical_split: support_label})`` for an unrooted tree."""
    leaves, clades = newick_clades(newick)
    taxa = frozenset(leaves)
    anchor = min(taxa)
    splits: dict[frozenset[str], str | None] = {}
    for members, label in clades:
        side = taxa - members if anchor in members else members
        if len(side) < 2 or len(taxa) - len(side) < 2:
            continue
        if side not in splits or splits[side] is None:
            splits[side] = label
    return taxa, splits


def read_tree(path: str | Path) -> str:
    """Read the first tree of a Newick or NEXUS file as a Newick string.

    NEXUS ``translate`` tables (as written by MrBayes/BEAST) are applied.
    """
    text = Path(path).read_text()
    if not text.lstrip().upper().startswith("#NEXUS"):
        stripped = _COMMENT.sub("", text).strip()
        end = stripped.find(";")
        return stripped[: end + 1] if end >= 0 else stripped + ";"
    body = _COMMENT.sub("", text)
    translate: dict[str, str] = {}
    match = re.search(r"(?is)\btranslate\b(.*?);", body)
    if match:
        for item in match.group(1).split(","):
            parts = item.split()
            if len(parts) >= 2:
                translate[parts[0]] = parts[1].strip("'")
    match = re.search(r"(?is)\btree\s+[^=]+=\s*(\(.*?;)", body)
    if not match:
        raise OrganelleInputError(
            code="phylogeny.tree.no_tree", message=f"no tree statement in {path}"
        )
    newick = match.group(1)
    if translate:
        newick = re.sub(
            r"(?<=[(,])\s*([^(),:;\s]+)",
            lambda m: translate.get(m.group(1), m.group(1)),
            newick,
        )
    return newick


def robinson_foulds(newick_a: str, newick_b: str) -> dict[str, Any]:
    """Unrooted Robinson-Foulds distance between two trees on the same taxa.

    ``rf`` counts bipartitions present in exactly one tree; ``max_rf`` is
    ``|splits_a| + |splits_b|`` (``2(n-3)`` for two fully resolved trees) and
    ``normalized_rf = rf / max_rf``.
    """
    taxa_a, splits_a = tree_splits(newick_a)
    taxa_b, splits_b = tree_splits(newick_b)
    if taxa_a != taxa_b:
        raise OrganelleInputError(
            code="phylogeny.tree.taxa_mismatch",
            message="trees must have identical leaf sets",
            details={
                "only_in_a": sorted(taxa_a - taxa_b),
                "only_in_b": sorted(taxa_b - taxa_a),
            },
        )
    only_a = set(splits_a) - set(splits_b)
    only_b = set(splits_b) - set(splits_a)
    rf = len(only_a) + len(only_b)
    max_rf = len(splits_a) + len(splits_b)
    return {
        "n_taxa": len(taxa_a),
        "rf": rf,
        "max_rf": max_rf,
        "normalized_rf": (rf / max_rf) if max_rf else 0.0,
        "n_splits_a": len(splits_a),
        "n_splits_b": len(splits_b),
        "n_shared_splits": len(set(splits_a) & set(splits_b)),
        "splits_only_in_a": [
            {"taxa": sorted(s), "support": splits_a[s]} for s in sorted(only_a, key=sorted)
        ],
        "splits_only_in_b": [
            {"taxa": sorted(s), "support": splits_b[s]} for s in sorted(only_b, key=sorted)
        ],
    }


def compare_trees(tree_a: str | Path, tree_b: str | Path) -> OrganelleResult:
    """Compare two tree files (Newick or NEXUS) by unrooted Robinson-Foulds distance.

    Conflicting bipartitions are listed with their support labels (bootstrap
    or posterior probability) so poorly supported conflicts can be told apart
    from strongly supported ones.
    """
    stats = robinson_foulds(read_tree(tree_a), read_tree(tree_b))
    return ok_result(
        "compare_trees",
        metrics=stats,
        result_findings=findings(
            ("rf", stats["rf"]), ("normalized_rf", round(stats["normalized_rf"], 6))
        ),
        flags=("identical_topology",) if stats["rf"] == 0 else ("topology_differs",),
        summary_text=(
            f"RF = {stats['rf']} / {stats['max_rf']} (normalized {stats['normalized_rf']:.3f}) "
            f"on {stats['n_taxa']} taxa."
        ),
        result_provenance=provenance(
            "compare_trees",
            method="bipartition_rf",
            parameters={"tree_a": str(tree_a), "tree_b": str(tree_b)},
        ),
    )
