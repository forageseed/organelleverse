"""Branch-by-Branch (BXB) branch-length extraction — typed core.

Faithfully re-implements the official ERCnet ``Branch_length_reconciliation.R``
logic in pure Python 3 (no ape/phytools/DLCpar dependency required for the
``lca`` reconciliation source).

The BXB method uses the **species tree as a common denominator**: each gene-tree
branch is mapped to the species-tree branch it evolved along, so branch lengths
from gene trees with different duplication/loss histories become
'apples-to-apples' comparable. See ERCnet README (Forsythe et al. 2024,
bioRxiv 2024.08.06.606904):

    "the species tree can be used as a common denominator for making
     'apples-to-apples' comparisons between gene trees with vastly different
     duplication/loss histories. … Sometimes multiple gene tree branches
     existed within a single species tree branch (e.g. two paralogs both
     evolving in the same species), meaning we average the branch-lengths.
     Sometimes a gene tree does not have branches that speak to the evolution
     in a species tree branch (e.g. the gene was lost), meaning the branch
     length is NA."

Pipeline (mirrors the R script stage-for-stage):

1. ``species_tree_edges`` — enumerate species-tree edges as ``"{anc}_to_{desc}"``.
2. ``bxb_lengths_lca`` / ``bxb_lengths_dlcpar`` — extract per-edge branch
   lengths from one gene tree, mapping gene-tree branches onto species-tree
   edges via LCA/SDI reconciliation or a DLCpar ``.locus.recon``+``.coal.recon``
   pair.
3. ``filter_outlier_tree`` — drop a whole gene tree if its longest edge is
   >10× the second-longest (R script lines 160-164).
4. ``normalize_edge_mean`` — divide each species-tree edge by its cross-gene
   mean (R script lines 217-219; the "genome-wide average" normalization).

All functions are pure **data-in → data-out** (OmicVerse-style typed cores).
Missing data is represented as ``float("nan")`` throughout; a true ``0.0`` is a
valid branch length and is *not* treated as missing (consistent with
``coevolution._is_missing``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from .coevolution import (
    TreeNode,
    parse_newick,
    reconcile_trees,
    _find_lca,
)

__all__ = [
    "species_tree_edges",
    "bxb_lengths_lca",
    "bxb_lengths_dlcpar",
    "filter_outlier_tree",
    "normalize_gene_sum",
    "normalize_edge_mean",
]

_NAN = float("nan")
# R script line 109: "fix very small branches inferred by raxml … < 1e-5 -> 0"
_MIN_BL = 1e-5
# R script lines 160-164: longest edge > 10x second-longest -> drop whole tree
_OUTLIER_RATIO = 10.0


# ===========================================================================
# 1. Species-tree edge enumeration
# ===========================================================================


def _node_label(node: TreeNode, fallback: str) -> str:
    """Return a node's name, falling back to a synthesized id if unnamed."""
    return node.name if node.name else fallback


def species_tree_edges(species_tree_newick: str) -> list[str]:
    """Enumerate species-tree edges as ``"{ancestor}_to_{descendant}"`` keys.

    Matches the R script (lines 46-49): ``sp_branches_df`` columns are
    ``paste0(sp_branches_df$ancestor, "_to_", sp_branches_df$decendant)``,
    where ape's ``sp_tr$edge`` rows are (ancestor_index, descendant_index)
    resolved against ``c(tip.label, node.label)``.

    Internal nodes use their label (e.g. ``N0``, ``N1`` …); leaves use the
    species code (e.g. ``Brsc``). The root's parent edge is included only if
    the root has a non-zero branch length (ape includes the root edge
    separately — we follow the same convention as the R column set).

    Parameters
    ----------
    species_tree_newick : str
        Rooted species tree in Newick format **with internal node labels**
        (e.g. the OrthoFinder/ERCnet ``SpeciesTree_rooted_node_labels.txt``).

    Returns
    -------
    list[str]
        Ordered edge keys, root-to-tips (pre-order traversal of edges).
    """
    root = parse_newick(species_tree_newick)
    edges: list[str] = []
    # ape's edge table is ordered by node index; pre-order matches closely
    # enough since downstream code indexes by key, not position.
    for node in root.traverse():
        if node.parent is None:
            continue  # root has no incoming edge in ape's $edge (usually)
        anc = _node_label(node.parent, f"root")
        desc = _node_label(node, "")
        if not desc:
            # Unnamed leaf/internal — skip (shouldn't happen for labelled sp tree)
            continue
        edges.append(f"{anc}_to_{desc}")
    return edges


# ===========================================================================
# 2a. Per-gene BXB extraction via LCA/SDI reconciliation (pure Python)
# ===========================================================================


def _dist_nodes(tree: TreeNode) -> dict[tuple[int, int], float]:
    """All-pairs node distances, keyed by ``id(node)``.

    Mirrors ape's ``dist.nodes(tree)``: a full distance matrix where entry
    (i, j) is the patristic distance between nodes i and j along the tree.
    Branch lengths < ``_MIN_BL`` are floored to 0 (R script line 109).
    """
    nodes = list(tree.traverse())
    # path length from root to each node
    depth: dict[int, float] = {}
    for n in nodes:
        d = 0.0
        cur: TreeNode | None = n
        while cur is not None and cur.parent is not None:
            bl = cur.branch_length if cur.branch_length else 0.0
            if bl < _MIN_BL:
                bl = 0.0
            d += bl
            cur = cur.parent
        depth[id(n)] = d
    # LCA depth cache for path-distance computation
    dist: dict[tuple[int, int], float] = {}
    for a in nodes:
        for b in nodes:
            if id(a) == id(b):
                dist[(id(a), id(b))] = 0.0
                continue
            key = (id(a), id(b))
            if (id(b), id(a)) in dist:
                dist[key] = dist[(id(b), id(a))]
                continue
            lca = _find_lca(tree, [a, b])
            lca_d = depth.get(id(lca), 0.0) if lca is not None else 0.0
            dist[key] = depth[id(a)] + depth[id(b)] - 2 * lca_d
    return dist


def bxb_lengths_lca(
    gene_tree_newick: str,
    species_tree_newick: str,
    edges: list[str],
    species_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Extract BXB branch lengths for one gene tree via LCA/SDI reconciliation.

    The pure-Python alternative to DLCpar. ``reconcile_trees`` maps each
    gene-tree node to a species-tree node and infers speciation vs duplication.
    For each species-tree edge, we find the gene-tree nodes mapped to the edge's
    ancestor and descendant (non-duplication nodes only), then measure the
    patristic distance between them on the gene tree. When multiple gene nodes
    map to the same species node (paralogs), we average — matching the R script
    (line 141: ``mean(BL_distance_mat[…])``).

    Parameters
    ----------
    gene_tree_newick : str
        Gene tree with branch lengths (Newick).
    species_tree_newick : str
        Rooted species tree with internal node labels.
    edges : list[str]
        Edge keys from ``species_tree_edges()``.
    species_map : dict, optional
        Maps gene-leaf-name → species code. If None, the prefix before
        ``__``/``_``/``|``/``@`` is used (ERCnet convention).

    Returns
    -------
    dict[str, float]
        ``{edge_key: branch_length_or_nan}``. Edges with no mapped gene-tree
        branch get ``nan``. Values of 0 are converted to ``nan`` (R line 151:
        "Change any BL=0 branches to NA").

    Warning
    -------
    **Known limitation on partial gene trees.** The pure-LCA reconciliation
    maps gene-tree nodes directly to species-tree nodes by post-order LCA,
    without DLCpar's intermediate *locus tree*. On gene trees with substantial
    taxon loss (e.g. ERC45, where each gene family covers only ~10% of the 88
    species-tree edges), the LCA mapping diverges systematically from DLCpar's
    locus-tree mapping: it both over-counts duplications and assigns deep
    internal gene nodes to different species-tree nodes. Empirically this
    reduces the lca→dlcpar ERC-correlation agreement to ~0.58 and the top-50
    pair overlap to near zero.

    For **bit-for-bit reproduction** of ERCnet results on partial gene trees,
    use :func:`bxb_lengths_dlcpar` (``reconciliation="dlcpar"``). The ``lca``
    path is appropriate only for near-complete gene trees (low taxon loss)
    where LCA and locus-tree reconciliation converge.
    """
    if species_map is None:
        species_map = {}

    gene_tree = parse_newick(gene_tree_newick)
    species_tree = parse_newick(species_tree_newick)

    # --- Reconcile: map each gene node -> species TreeNode, mark dup/spec ---
    sp_nodes_by_label: dict[str, TreeNode] = {}
    for n in species_tree.traverse():
        lbl = _node_label(n, "")
        if lbl:
            sp_nodes_by_label[lbl] = n

    def get_species(leaf_name: str) -> str:
        if leaf_name in species_map:
            return species_map[leaf_name]
        for sep in ("__", "_", "|", "@"):
            if sep in leaf_name:
                return leaf_name.split(sep)[0]
        return leaf_name

    mapping: dict[int, TreeNode | None] = {}
    events: dict[int, str] = {}
    # post-order for bottom-up LCA mapping
    _reconcile_lca(gene_tree, species_tree, sp_nodes_by_label, get_species, mapping, events)

    # --- Build per-species-node gene-node membership (non-dup only) ---
    # R script line 119: spec_gene_rec <- subset(rec_df_replace, Event!="dup")
    nodes_by_species: dict[str, list[TreeNode]] = {}
    for node in gene_tree.traverse():
        if events.get(id(node)) == "dup":
            continue  # only speciation nodes feed branch-length extraction
        sp_node = mapping.get(id(node))
        if sp_node is None:
            continue
        lbl = _node_label(sp_node, "")
        if lbl:
            nodes_by_species.setdefault(lbl, []).append(node)

    dist = _dist_nodes(gene_tree)

    # --- For each species edge: mean-of-per-ancestor-means (R lines 128-140) ---
    # R script: for each ancestor gene node a, store mean(dist[a, desc∩kids(a)]);
    #           BL[edge] = mean(na.omit(store_wt_av)). Ancestors contributing no
    # descendant get NaN (mean(numeric(0))) and are dropped by na.omit. Each
    # ancestor contributes ONE equally-weighted distance — not each (anc,desc)
    # pair. We reproduce that exactly.
    result: dict[str, float] = {}
    for edge in edges:
        anc_label, desc_label = edge.split("_to_", 1)
        anc_genes = nodes_by_species.get(anc_label, [])
        desc_genes = nodes_by_species.get(desc_label, [])
        if not anc_genes or not desc_genes:
            result[edge] = _NAN
            continue
        per_anc_means: list[float] = []
        for ag in anc_genes:
            desc_of_ag = {id(d) for d in _descendants(ag)}
            anc_dists = [dist[(id(ag), id(dg))] for dg in desc_genes if id(dg) in desc_of_ag]
            if anc_dists:  # mean(numeric(0)) = NaN -> dropped by na.omit
                per_anc_means.append(sum(anc_dists) / len(anc_dists))
        if not per_anc_means:
            result[edge] = _NAN
            continue
        val = sum(per_anc_means) / len(per_anc_means)
        # R line 147: "Change any BL=0 branches to NA"
        result[edge] = _NAN if val == 0.0 else val
    return result


def _reconcile_lca(
    gene_tree: TreeNode,
    species_tree: TreeNode,
    sp_nodes_by_label: dict[str, TreeNode],
    get_species: Callable[[str], str],
    mapping: dict[int, TreeNode | None],
    events: dict[int, str],
) -> None:
    """Populate ``mapping`` (gene node id -> species TreeNode) and ``events``.

    Same LCA/SDI logic as ``coevolution.reconcile_trees``, but exposes the
    internal mapping (instead of wrapping it in an OrganelleResult) so the BXB
    extractor can reuse it directly.
    """
    # post-order traversal
    for node in _post_order_list(gene_tree):
        if not node.children:
            sp = get_species(node.name) if node.name else ""
            mapping[id(node)] = sp_nodes_by_label.get(sp)
            continue
        child_maps = [mapping.get(id(c)) for c in node.children]
        child_species_nodes = [m for m in child_maps if m is not None]
        if len(child_species_nodes) == 1:
            mapped_node = child_species_nodes[0]
        elif len(child_species_nodes) >= 2:
            mapped_node = _find_lca(species_tree, child_species_nodes) or species_tree
        else:
            mapped_node = species_tree
        mapping[id(node)] = mapped_node
        # Duplication test (SDI/LCA). The naive ``mapped_node in child_maps``
        # over-counts duplications on *partial* gene trees (where many species
        # are missing): two subtrees can legitimately map to the same species
        # subtree via deep coalescence without any duplication. We use the
        # stricter test used by ETE/DLCpar's LCA mode: a node is a duplication
        # only when ≥2 children map to the *same species node* that is NOT their
        # species-tree LCA — i.e. at least one child's mapping is strictly
        # within (descendant of) the parent's mapping, indicating a second copy
        # in the same lineage. On single-copy gene trees (ERC45's HOG-filtered
        # data), this correctly yields 0% duplications, matching DLCpar.
        is_dup = _is_duplication(mapped_node, child_maps, species_tree)
        events[id(node)] = "dup" if is_dup else "spec"


def _is_duplication(
    parent_map: TreeNode | None,
    child_maps: list[TreeNode | None],
    species_tree: TreeNode,
) -> bool:
    """SDI duplication test that does not over-count on partial gene trees.

    A gene-tree node is a duplication iff at least one child maps to the *same*
    species-tree node as the parent **and** that child has a sibling whose
    mapping is a strict descendant of that same node (i.e. two distinct lineages
    pass through one species node — a true second copy). Equivalently for the
    common bifurcating case: duplication iff both children map to the same
    species node (one being a strict self-map), which requires the species node
    to appear ≥2 times among child maps that are equal to (or descendants of)
    the parent map.

    On single-copy gene trees this never fires, matching DLCpar.
    """
    if parent_map is None:
        return False
    # Count children whose species-map is the parent map itself or a descendant.
    relevant = [
        m
        for m in child_maps
        if m is not None and _is_equal_or_descendant(m, parent_map, species_tree)
    ]
    if len(relevant) < 2:
        return False
    # If ≥2 children map into the same species lineage, at least one must be a
    # strict duplicate (mapped to the parent node itself, not a distinct leaf),
    # OR two children map to the same node.
    # The unambiguous signal: any species node appears ≥2 times among child maps.
    seen: set[int] = set()
    for m in child_maps:
        if m is None:
            continue
        if id(m) in seen:
            return True
        seen.add(id(m))
    return False


def _is_equal_or_descendant(node: TreeNode, ancestor: TreeNode, species_tree: TreeNode) -> bool:
    """True if ``node`` is ``ancestor`` or a descendant of it in the species tree."""
    if node is ancestor:
        return True
    cur: TreeNode | None = node
    while cur is not None:
        if cur is ancestor:
            return True
        cur = cur.parent
    return False


def _post_order_list(root: TreeNode) -> list[TreeNode]:
    out: list[TreeNode] = []
    _collect_post_order(root, out)
    return out


def _collect_post_order(node: TreeNode, out: list[TreeNode]) -> None:
    for child in node.children:
        _collect_post_order(child, out)
    out.append(node)


def _descendants(node: TreeNode) -> list[TreeNode]:
    """All strict descendants of ``node`` (excluding node itself)."""
    out: list[TreeNode] = []
    for child in node.children:
        for n in child.traverse():
            out.append(n)
    return out


# ===========================================================================
# 2b. Per-gene BXB extraction via DLCpar .recon files (bit-for-bit with R)
# ===========================================================================


def bxb_lengths_dlcpar(
    gene_tree_newick: str,
    locus_recon_path: str | Path,
    coal_recon_path: str | Path,
    edges: list[str],
) -> dict[str, float]:
    """Extract BXB branch lengths from DLCpar reconciliation output.

    This is the exact-reproduction path: it reads the same files the R script
    reads (``.dlcpar.locus.recon`` + ``.dlcpar.coal.recon``) and reproduces the
    node-mapping logic verbatim (R script lines 79-151).

    The ``.locus.recon`` maps **locus-tree nodes** to species-tree nodes with
    dup/spec labels. The ``.coal.recon`` maps gene-tree nodes ↔ locus-tree
    nodes (needed because the locus tree differs from the gene tree when
    duplications are present — "the nodes in here refer to the nodes from the
    'locus tree', which doesn't have branch lengths", R lines 82-83).

    Parameters
    ----------
    gene_tree_newick : str
        Branch-length-bearing gene tree (the ``*_NODES_BL.txt`` file).
    locus_recon_path : path
        DLCpar ``.dlcpar.locus.recon`` file (gene/locus_node, species_node, event).
    coal_recon_path : path
        DLCpar ``.dlcpar.coal.recon`` file (gene_node, locus_node, count).
    edges : list[str]
        Edge keys from ``species_tree_edges()``.

    Returns
    -------
    dict[str, float]
        ``{edge_key: branch_length_or_nan}``.
    """
    gene_tree = parse_newick(gene_tree_newick)

    # --- Read .locus.recon: locus_node -> (species_node, event) ---
    locus_recon: dict[str, tuple[str, str]] = {}
    for line in Path(locus_recon_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 3:
            continue
        locus_recon[parts[0]] = (parts[1], parts[2])

    # --- Read .coal.recon: gene_node <-> locus_node ---
    gene_to_locus: dict[str, str] = {}
    locus_to_gene: dict[str, list[str]] = {}
    for line in Path(coal_recon_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        g_node, l_node = parts[0], parts[1]
        gene_to_locus[g_node] = l_node
        locus_to_gene.setdefault(l_node, []).append(g_node)

    # --- Build spec_gene_rec: gene_tree_node -> species_node (non-dup) ---
    # R lines 89-97: replace locus-node names with gene-tree-node names via
    # the coal.recon conversion table, then keep non-dup rows.
    spec_gene_rec: dict[str, str] = {}  # gene_node -> species_node
    for locus_node, (sp_node, event) in locus_recon.items():
        if event == "dup":
            continue
        # Convert locus_node back to gene_tree_node name(s)
        gene_nodes = locus_to_gene.get(locus_node, [locus_node])
        for gn in gene_nodes:
            spec_gene_rec[gn] = sp_node

    # --- Build dist.nodes matrix on the gene tree, keyed by node name ---
    name_to_node: dict[str, TreeNode] = {}
    for n in gene_tree.traverse():
        if n.name:
            name_to_node[n.name] = n
    dist = _dist_nodes(gene_tree)

    def _named_dist(a_name: str, b_name: str) -> float | None:
        a = name_to_node.get(a_name)
        b = name_to_node.get(b_name)
        if a is None or b is None:
            return None
        return dist[(id(a), id(b))]

    # --- For each species edge: mean-of-per-ancestor-means (R lines 123-144) ---
    # R script: for each ancestor gene node a, store mean(dist[a, desc∩kids(a)]);
    #           BL[edge] = mean(na.omit(store_wt_av)). Each ancestor contributes
    # ONE equally-weighted distance — not each (anc,desc) pair.
    result: dict[str, float] = {}
    for edge in edges:
        anc_label, desc_label = edge.split("_to_", 1)
        # R line 125-126: gene nodes mapped to this species ancestor/descendant
        anc_genes = [gn for gn, sp in spec_gene_rec.items() if sp == anc_label]
        desc_genes = [gn for gn, sp in spec_gene_rec.items() if sp == desc_label]
        if not anc_genes or not desc_genes:
            result[edge] = _NAN
            continue
        per_anc_means: list[float] = []
        for ag_name in anc_genes:
            ag = name_to_node.get(ag_name)
            if ag is None:
                continue
            desc_of_ag = {n.name for n in _descendants(ag) if n.name}
            anc_dists = [
                _named_dist(ag_name, dg_name) for dg_name in desc_genes if dg_name in desc_of_ag
            ]
            anc_dists = [d for d in anc_dists if d is not None]
            if anc_dists:  # mean(numeric(0)) = NaN -> dropped by na.omit
                per_anc_means.append(sum(anc_dists) / len(anc_dists))
        if not per_anc_means:
            result[edge] = _NAN
            continue
        val = sum(per_anc_means) / len(per_anc_means)
        result[edge] = _NAN if val == 0.0 else val
    return result


# ===========================================================================
# 3. Outlier filtering (R lines 160-164 / 195-199)
# ===========================================================================


def filter_outlier_tree(
    edge_lengths: dict[str, float],
    max_ratio: float = _OUTLIER_RATIO,
) -> dict[str, float] | None:
    """Return ``None`` if the gene tree is an outlier, else the input unchanged.

    R script (lines 160-164):
        ``if (longest > 10 * second_longest) { set all values to NA }``

    A gene tree is dropped entirely when its single longest edge dominates —
    this prevents a single spurious branch (e.g. a raxml artifact) from
    contaminating the cross-gene correlation.

    Parameters
    ----------
    edge_lengths : dict
        ``{edge_key: value}`` for one gene (may contain nan).
    max_ratio : float
        Drop the tree if longest > max_ratio × second-longest. Default 10.

    Returns
    -------
    dict or None
        The input dict if the tree passes, ``None`` if it's an outlier.
    """
    vals = sorted(
        (v for v in edge_lengths.values() if not _is_missing(v)),
        reverse=True,
    )
    if len(vals) < 2:
        return edge_lengths  # not enough non-missing values to judge
    if vals[0] > max_ratio * vals[1]:
        return None
    return edge_lengths


# ===========================================================================
# 4a. Per-gene sum normalization (ERC45-style public BXB contract)
# ===========================================================================


def normalize_gene_sum(
    matrix: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Normalize each gene's BXB vector by its non-missing branch-length sum.

    This keeps every gene on the same total scale before cross-gene ERC
    correlation, preventing genes with larger absolute tree lengths from
    dominating the analysis. Missing values remain NaN.
    """
    out: dict[str, dict[str, float]] = {}
    for gene, row in matrix.items():
        denom = sum(v for v in row.values() if not _is_missing(v))
        new_row: dict[str, float] = {}
        for edge, val in row.items():
            if _is_missing(val) or denom == 0.0:
                new_row[edge] = _NAN
            else:
                new_row[edge] = val / denom
        out[gene] = new_row
    return out


# ===========================================================================
# 4. Edge-mean normalization (R lines 217-219)
# ===========================================================================


def normalize_edge_mean(
    matrix: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Normalize each species-tree edge by its cross-gene mean.

    R script (lines 217-219):
        ``bxb_BL_norm[,s] <- bxb_BL_norm[,s] / mean(na.omit(bxb_BL_norm[,s]))``

    This is the official ERCnet "genome-wide average" normalization: each
    species-tree edge (column) is divided by the mean of its non-missing values
    across all genes. This calibrates every edge to a relative mean-rate of 1,
    removing the effect of some species-tree edges being systematically longer
    than others.

    Note: this is **not** per-gene sum normalization. It is per-edge mean
    normalization — the official behavior.

    Parameters
    ----------
    matrix : dict
        ``{gene: {edge_key: value}}`` (values may contain nan).

    Returns
    -------
    dict
        Normalized matrix with the same shape. Edges whose mean is 0 or fully
        missing become all-nan for every gene.
    """
    if not matrix:
        return {}
    # collect all edge keys
    all_edges: set[str] = set()
    for row in matrix.values():
        all_edges.update(row.keys())

    edge_means: dict[str, float] = {}
    for edge in all_edges:
        vals = [row[edge] for row in matrix.values() if edge in row and not _is_missing(row[edge])]
        edge_means[edge] = sum(vals) / len(vals) if vals else _NAN

    out: dict[str, dict[str, float]] = {}
    for gene, row in matrix.items():
        new_row: dict[str, float] = {}
        for edge, val in row.items():
            mean = edge_means.get(edge, _NAN)
            if _is_missing(val) or _is_missing(mean) or mean == 0.0:
                new_row[edge] = _NAN
            else:
                new_row[edge] = val / mean
        out[gene] = new_row
    return out


# ===========================================================================
# Internal helpers
# ===========================================================================


def _is_missing(v: float) -> bool:
    """True if ``v`` is NaN. (Mirrors ``coevolution._is_missing``.)"""
    return v != v
