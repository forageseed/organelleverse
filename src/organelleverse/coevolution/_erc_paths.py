"""Node-pair paths used by ERC 2.0 (TreeTools PathLengths semantics).

The layout contains every ancestor/descendant pair. Fitting and pairwise ERC
use only the independent edges of the tree pruned to the relevant taxa, as in
ERC_functions.R:coreGetResiduals and ERC.R:computeERC. Input roots must agree.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ._species import species_resolver
from .coevolution import TreeNode, parse_newick


@dataclass
class PathLayout:
    tree: TreeNode
    nodes: list[TreeNode]
    clades: dict[TreeNode, frozenset[str]]
    paths: list[tuple[TreeNode, TreeNode]]

    def edge_indices(self, species: set[str]) -> list[int]:
        """Independent edges after pruning, suppressing unary nodes."""
        retained = {
            n for n in self.nodes
            if (n.is_leaf and n.name in species)
            or sum(bool(self.clades[c] & species) for c in n.children) >= 2
        }
        lookup = {pair: i for i, pair in enumerate(self.paths)}
        indices = []
        for child in self.nodes:
            if child not in retained:
                continue
            parent = child.parent
            while parent is not None and parent not in retained:
                parent = parent.parent
            if parent is not None:
                indices.append(lookup[(parent, child)])
        return indices


def layout(tree: TreeNode) -> PathLayout:
    nodes = list(tree.traverse())
    clades = {n: frozenset(leaf.name for leaf in n.leaves()) for n in nodes}
    # A deterministic order; identity is the endpoint pair, not list position.
    paths = []
    for child in nodes:
        ancestor = child.parent
        while ancestor is not None:
            paths.append((ancestor, child))
            ancestor = ancestor.parent
    return PathLayout(tree, nodes, clades, paths)


def project(
    gene_trees: Mapping[str, str], species_tree_newick: str
) -> tuple[dict[str, list[float | None]], PathLayout, dict[str, set[str]]]:
    master = layout(parse_newick(species_tree_newick))
    master_species = set(master.clades[master.tree])
    if len(master_species) != len(master.tree.leaves()):
        raise ValueError("ERC master tree must have unique species names")
    resolve = species_resolver(master_species)
    out: dict[str, list[float | None]] = {}
    species_by_gene: dict[str, set[str]] = {}
    for gene, newick in gene_trees.items():
        tree = parse_newick(newick)
        if any(n.branch_length < 0 for n in tree.traverse()):
            raise ValueError(f'Gene {gene} has a negative evolutionary branch length')
        for leaf in tree.leaves():
            leaf.name = resolve(leaf.name)
        present = {leaf.name for leaf in tree.leaves()}
        if len(present) != len(tree.leaves()):
            raise ValueError(f"ERC 2.0 requires one copy per species: {gene}")
        if not present <= master_species:
            raise ValueError(f"Gene {gene} has species absent from the master tree")
        gl = layout(tree)
        # Retain true branch points of the induced master topology. A clade
        # with a single surviving child is suppressed, never mapped twice.
        kept = {
            n for n in master.nodes
            if (n.is_leaf and n.name in present)
            or sum(bool(master.clades[c] & present) for c in n.children) >= 2
        }
        master_by_clade = {master.clades[n] & present: n for n in kept}
        gene_by_clade = {clade: n for n, clade in gl.clades.items()}
        if set(gene_by_clade) != set(master_by_clade):
            raise ValueError(f"Gene {gene} topology/root does not match the induced master tree")
        mapped = {master_by_clade[clade]: n for clade, n in gene_by_clade.items()}
        depths = {n: n.get_distance_to_root() for n in gl.nodes}
        out[gene] = [
            depths[mapped[child]] - depths[mapped[parent]]
            if parent in mapped and child in mapped else None
            for parent, child in master.paths
        ]
        species_by_gene[gene] = present
    return out, master, species_by_gene


def lowess(x: Sequence[float], y: Sequence[float], fraction: float = .7) -> tuple[list[float], list[float]]:
    """Cleveland robust locally weighted linear regression, two reweightings.

    R stats::lowess uses the same tricube neighbourhood, bisquare robust
    weights and linear interpolation between evaluations separated by
    delta=.01*range(x), its default acceleration parameter.
    """
    import numpy as np

    order = np.argsort(x, kind="stable")
    xx, yy = np.asarray(x)[order], np.asarray(y)[order]
    n = len(xx)
    if n < 2:
        raise ValueError("ERC variance fitting needs at least two finite residuals")
    span = max(2, min(n, int(fraction * n + 1e-7)))
    delta = .01 * float(xx[-1] - xx[0])
    knots = [0]
    while knots[-1] < n - 1:
        current = knots[-1]
        end = int(np.searchsorted(xx, xx[current] + delta, side='right')) - 1
        following = max(current + 1, end)
        while following < n - 1 and xx[following] == xx[current]:
            following += 1
        knots.append(following)
    robust = np.ones(n)
    fitted = np.empty(n)
    for iteration in range(3):
        for i in knots:
            dist = np.abs(xx - xx[i])
            bandwidth = np.partition(dist, span - 1)[span - 1]
            if bandwidth == 0:
                weights = (dist == 0).astype(float) * robust
            else:
                u = np.minimum(dist / bandwidth, 1)
                weights = (1 - u ** 3) ** 3 * robust
            total = weights.sum()
            if total == 0:
                fitted[i] = yy[i]
                continue
            weights /= total
            center_x = np.dot(weights, xx)
            center_y = np.dot(weights, yy)
            variance = np.dot(weights, (xx - center_x) ** 2)
            slope = (np.dot(weights, (xx - center_x) * (yy - center_y)) / variance
                     if np.sqrt(variance) > .001 * (xx[-1] - xx[0]) else 0)
            fitted[i] = center_y + slope * (xx[i] - center_x)
        fitted = np.interp(xx, xx[knots], fitted[knots])
        if iteration < 2:
            residual = np.abs(yy - fitted)
            scale = 6 * float(np.median(residual))
            if scale < 1e-7 * float(np.mean(residual)) or scale == 0:
                break
            u = np.minimum(residual / scale, 1)
            # The official LOWESS bisquare rule has exact unit/zero-weight
            # plateaus at .001 and .999 of six MADs, respectively. Omitting
            # those plateaus changes the subsequent weighted fit.
            robust = np.where(u <= .001, 1., np.where(u > .999, 0., (1 - u ** 2) ** 2))
    # approxfun's default ties=mean, rule=2 (constant extrapolation).
    unique, first, counts = np.unique(xx, return_index=True, return_counts=True)
    means = [float(np.mean(fitted[i:i + count])) for i, count in zip(first, counts, strict=True)]
    return unique.tolist(), means
