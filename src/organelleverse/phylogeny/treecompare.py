"""Tree comparison: Robinson-Foulds distance and cophenetic correlation.

Pure-Python Newick/NEXUS split extraction shared by :func:`compare_trees` and
the MrBayes ASDSF recomputation in :mod:`.mrbayes`. Trees are compared as
*unrooted* topologies: every internal edge is one bipartition of the leaf set,
trivial splits (one side < 2 taxa) are ignored, and each split is stored
canonically as the side that does not contain the alphabetically first taxon.

Branch lengths feed two matrices: patristic (tip-to-tip path) distances, whose
Pearson correlation between two trees is the cophenetic correlation, and the
shared root-to-MRCA path lengths that are the Brownian-motion phylogenetic
covariance used by :mod:`.pglmm`.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from organelleverse._public_validation import validated_public

from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from ..inputs import Tree
from ..sequences import as_path
from ._results import findings, ok_result, provenance

__all__ = [
    "RootedTree",
    "compare_trees",
    "cophenetic_correlation",
    "newick_clades",
    "parse_rooted_tree",
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


def tree_splits(
    newick: str, keep: Collection[str] | None = None
) -> tuple[frozenset[str], dict[frozenset[str], str | None]]:
    """Return ``(taxa, {canonical_split: support_label})`` for an unrooted tree.

    With ``keep``, the tree is pruned to those taxa first: each split is
    restricted to the kept leaves and the ones that become trivial vanish.
    """
    leaves, clades = newick_clades(newick)
    taxa = frozenset(leaves) if keep is None else frozenset(leaves) & frozenset(keep)
    if not taxa:
        raise OrganelleInputError(
            code="phylogeny.tree.taxa_mismatch", message="no taxa left after pruning"
        )
    anchor = min(taxa)
    splits: dict[frozenset[str], str | None] = {}
    for members, label in clades:
        members = members & taxa
        side = taxa - members if anchor in members else members
        if len(side) < 2 or len(taxa) - len(side) < 2:
            continue
        if side not in splits or splits[side] is None:
            splits[side] = label
    return taxa, splits


@dataclass(frozen=True)
class RootedTree:
    """A Newick tree as parent-pointer arrays, root first (node 0).

    ``length[i]`` is the branch above node ``i`` (0 when absent); ``depth[i]``
    is the root-to-node path length. ``has_lengths`` is False when the Newick
    carries no branch lengths at all.
    """

    names: tuple[str | None, ...]
    parent: tuple[int, ...]
    length: tuple[float, ...]
    depth: tuple[float, ...]
    leaves: tuple[int, ...]
    has_lengths: bool

    @property
    def leaf_names(self) -> list[str]:
        return [str(self.names[i]) for i in self.leaves]

    def shared_path_matrix(self, taxa: list[str]) -> np.ndarray:
        """Root-to-MRCA path length for every pair of ``taxa`` (diagonal: root-to-tip).

        Under Brownian motion this is the phylogenetic covariance matrix. The
        tree is pruned to ``taxa`` and re-rooted at their MRCA, as ``ape::drop.tip``
        does, so lengths above the retained taxa's MRCA are not counted.
        """
        index = {str(self.names[node]): node for node in self.leaves}
        missing = [name for name in taxa if name not in index]
        if missing:
            raise OrganelleInputError(
                code="phylogeny.tree.taxa_mismatch",
                message="taxa missing from the tree",
                details={"missing": missing},
            )
        position = {index[name]: k for k, name in enumerate(taxa)}
        children: dict[int, list[int]] = {}
        for node, up in enumerate(self.parent):
            if up >= 0:
                children.setdefault(up, []).append(node)
        below: dict[int, list[int]] = {}
        matrix = np.zeros((len(taxa), len(taxa)))
        mrca_depth = None
        for node in reversed(range(len(self.parent))):
            if node in position:
                below[node] = [position[node]]
                continue
            groups = [below.pop(child) for child in children.get(node, ()) if child in below]
            groups = [group for group in groups if group]
            if not groups:
                continue
            for a in range(len(groups)):
                for b in range(a + 1, len(groups)):
                    matrix[np.ix_(groups[a], groups[b])] = self.depth[node]
                    matrix[np.ix_(groups[b], groups[a])] = self.depth[node]
            merged = [k for group in groups for k in group]
            below[node] = merged
            if len(merged) == len(taxa) and mrca_depth is None:
                mrca_depth = self.depth[node]
        for name, k in zip(taxa, range(len(taxa)), strict=True):
            matrix[k, k] = self.depth[index[name]]
        if mrca_depth is None:  # a single taxon
            mrca_depth = 0.0
        return matrix - mrca_depth

    def patristic_matrix(self, taxa: list[str]) -> np.ndarray:
        """Tip-to-tip path length for every pair of ``taxa`` (unaffected by pruning)."""
        shared = self.shared_path_matrix(taxa)
        tips = np.diag(shared)
        return tips[:, None] + tips[None, :] - 2.0 * shared


def parse_rooted_tree(newick: str) -> RootedTree:
    """Parse a Newick string into a :class:`RootedTree`, keeping branch lengths."""
    text = _COMMENT.sub("", newick).strip()
    if text.endswith(";"):
        text = text[:-1]
    names: list[str | None] = [None]
    parent: list[int] = [-1]
    length: list[float] = [0.0]
    leaves: list[int] = []
    seen_length = False
    stack = [0]
    current = 0  # the node a following label/length belongs to
    i, n = 0, len(text)

    def read_label(start: int) -> tuple[str, int]:
        if start < n and text[start] == "'":
            end = text.find("'", start + 1)
            end = n if end < 0 else end
            return text[start + 1 : end], end + 1
        end = start
        while end < n and text[end] not in ",():;":
            end += 1
        return text[start:end].strip(), end

    def read_length(start: int, node: int) -> int:
        nonlocal seen_length
        if start < n and text[start] == ":":
            end = start + 1
            while end < n and text[end] not in ",()":
                end += 1
            raw = text[start + 1 : end].strip()
            try:
                length[node] = float(raw)
            except ValueError as error:
                raise OrganelleInputError(
                    code="phylogeny.tree.bad_newick", message=f"bad branch length {raw!r}"
                ) from error
            seen_length = True
            return end
        return start

    if not text.startswith("("):
        raise OrganelleInputError(code="phylogeny.tree.bad_newick", message="tree must start with '('")
    i = 1
    while i < n:
        char = text[i]
        if char == "(":
            names.append(None)
            parent.append(stack[-1])
            length.append(0.0)
            stack.append(len(names) - 1)
            i += 1
        elif char == ",":
            i += 1
        elif char == ")":
            if not stack:
                raise OrganelleInputError(
                    code="phylogeny.tree.bad_newick", message="unbalanced parentheses"
                )
            current = stack.pop()
            label, i = read_label(i + 1)
            names[current] = label or None
            i = read_length(i, current)
        elif char.isspace():
            i += 1
        else:
            if not stack:
                raise OrganelleInputError(
                    code="phylogeny.tree.bad_newick", message="text after the root"
                )
            label, i = read_label(i)
            if not label:
                raise OrganelleInputError(
                    code="phylogeny.tree.bad_newick", message="empty leaf name"
                )
            names.append(label)
            parent.append(stack[-1])
            length.append(0.0)
            leaves.append(len(names) - 1)
            i = read_length(i, len(names) - 1)
    if stack:
        raise OrganelleInputError(code="phylogeny.tree.bad_newick", message="unbalanced tree")
    leaf_names = [names[k] for k in leaves]
    if len(set(leaf_names)) != len(leaf_names):
        raise OrganelleInputError(
            code="phylogeny.tree.duplicate_leaves", message="tree has duplicate leaf names"
        )
    if any(value < 0 for value in length):
        raise OrganelleInputError(
            code="phylogeny.tree.negative_branch", message="tree has negative branch lengths"
        )
    depth = [0.0] * len(names)
    for node in range(1, len(names)):  # parents always precede children
        depth[node] = depth[parent[node]] + length[node]
    return RootedTree(
        names=tuple(names),
        parent=tuple(parent),
        length=tuple(length),
        depth=tuple(depth),
        leaves=tuple(leaves),
        has_lengths=seen_length,
    )


def cophenetic_correlation(
    newick_a: str, newick_b: str, taxa: list[str] | None = None
) -> float | None:
    """Pearson correlation of the two trees' patristic distances over ``taxa``.

    ``taxa`` defaults to the shared leaves. Returns None when either tree lacks
    branch lengths or fewer than three taxa are shared.
    """
    tree_a, tree_b = parse_rooted_tree(newick_a), parse_rooted_tree(newick_b)
    if not (tree_a.has_lengths and tree_b.has_lengths):
        return None
    if taxa is None:
        taxa = sorted(set(tree_a.leaf_names) & set(tree_b.leaf_names))
    if len(taxa) < 3:
        return None
    upper = np.triu_indices(len(taxa), k=1)
    x = tree_a.patristic_matrix(taxa)[upper]
    y = tree_b.patristic_matrix(taxa)[upper]
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


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


def robinson_foulds(
    newick_a: str, newick_b: str, *, prune_to_shared: bool = False
) -> dict[str, Any]:
    """Unrooted Robinson-Foulds distance between two trees on the same taxa.

    ``rf`` counts bipartitions present in exactly one tree; ``max_rf`` is
    ``|splits_a| + |splits_b|`` (``2(n-3)`` for two fully resolved trees) and
    ``normalized_rf = rf / max_rf``; ``rf_similarity = 1 - normalized_rf``.
    ``prune_to_shared`` first prunes both trees to the taxa they share.
    """
    keep: frozenset[str] | None = None
    dropped_a: list[str] = []
    dropped_b: list[str] = []
    if prune_to_shared:
        leaves_a = frozenset(newick_clades(newick_a)[0])
        leaves_b = frozenset(newick_clades(newick_b)[0])
        keep = leaves_a & leaves_b
        if len(keep) < 4:
            raise OrganelleInputError(
                code="phylogeny.tree.taxa_mismatch",
                message=f"trees share {len(keep)} taxa; at least 4 are needed",
            )
        dropped_a, dropped_b = sorted(leaves_a - keep), sorted(leaves_b - keep)
    taxa_a, splits_a = tree_splits(newick_a, keep)
    taxa_b, splits_b = tree_splits(newick_b, keep)
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
    normalized = (rf / max_rf) if max_rf else 0.0
    return {
        "n_taxa": len(taxa_a),
        "rf": rf,
        "max_rf": max_rf,
        "normalized_rf": normalized,
        "rf_similarity": 1.0 - normalized,
        "pruned_from_a": dropped_a,
        "pruned_from_b": dropped_b,
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


@validated_public('phylogeny.compare_trees')
def compare_trees(
    tree_a: str | Path | Tree,
    tree_b: str | Path | Tree,
    *,
    prune_to_shared: bool = False,
) -> OrganelleResult:
    """Compare two tree files (Newick or NEXUS) by topology and branch lengths.

    Topology: unrooted Robinson-Foulds distance and ``rf_similarity``
    (1 - normalized RF). Conflicting bipartitions are listed with their support
    labels (bootstrap or posterior probability) so poorly supported conflicts
    can be told apart from strongly supported ones.

    Branch lengths: ``cophenetic_correlation``, the Pearson correlation of the
    two trees' tip-to-tip path lengths (None when either tree has no branch
    lengths). Scale-free, so trees in substitutions and in Ma compare directly.

    ``prune_to_shared`` compares only the taxa both trees contain (e.g. a
    mitochondrial against a nuclear tree with a different sampling); otherwise
    the leaf sets must match.
    """
    tree_a = as_path(tree_a)
    tree_b = as_path(tree_b)
    newick_a, newick_b = read_tree(tree_a), read_tree(tree_b)
    stats = robinson_foulds(newick_a, newick_b, prune_to_shared=prune_to_shared)
    shared = sorted(
        set(newick_clades(newick_a)[0]) & set(newick_clades(newick_b)[0])
    )
    cophenetic = cophenetic_correlation(newick_a, newick_b, shared)
    stats["cophenetic_correlation"] = cophenetic
    flags = ["identical_topology" if stats["rf"] == 0 else "topology_differs"]
    if cophenetic is None:
        flags.append("no_branch_lengths")
    if stats["pruned_from_a"] or stats["pruned_from_b"]:
        flags.append("pruned_to_shared_taxa")
    cophenetic_text = "" if cophenetic is None else f"; cophenetic r = {cophenetic:.3f}"
    return ok_result(
        "compare_trees",
        metrics=stats,
        result_findings=findings(
            ("rf", stats["rf"]),
            ("normalized_rf", round(stats["normalized_rf"], 6)),
            ("rf_similarity", round(stats["rf_similarity"], 6)),
            ("cophenetic_correlation", None if cophenetic is None else round(cophenetic, 6)),
        ),
        flags=tuple(flags),
        summary_text=(
            f"RF = {stats['rf']} / {stats['max_rf']} (normalized {stats['normalized_rf']:.3f}, "
            f"similarity {stats['rf_similarity']:.3f}){cophenetic_text} on {stats['n_taxa']} taxa."
        ),
        result_provenance=provenance(
            "compare_trees",
            method="bipartition_rf+cophenetic",
            parameters={
                "tree_a": str(tree_a),
                "tree_b": str(tree_b),
                "prune_to_shared": prune_to_shared,
            },
        ),
    )
