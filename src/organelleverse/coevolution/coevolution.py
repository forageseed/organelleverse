"""Coevolution suite — ERC (evolutionary rate covariation), fully self-contained.

Absorbs and re-implements the ERCnet2 pipeline in pure Python 3 (no DLCpar /
Python2 dependency). The four functions model the ERCnet stages:

1. ``reconcile_trees()`` — gene-tree / species-tree reconciliation via a
   self-contained LCA (Last Common Ancestor) algorithm. Maps each gene-tree
   node to a species-tree node and infers speciation vs duplication events,
   producing DLCpar-compatible ``.locus.recon`` output. Replaces the original
   Python2/DLCpar step.

2. ``extract_branch_lengths()`` — R2T (root-to-tip) branch-length extraction
   from reconciled gene trees, producing a gene × branch matrix.

3. ``rate_covariation()`` — ERC2 residual-based correlation: transforms branch
   lengths → computes trimmed-mean master → regresses each gene against master
   (OLS or variance-weighted WLS, optional PCA correction) → all-by-all Pearson
   / Spearman correlations on residuals → Benjamini-Hochberg FDR.

4. ``coevolution_network()`` — builds a coevolution network from significant
   correlations; detects communities via connected components.

All tree operations use a self-contained Newick parser (no ETE3/Biopython
required). numpy/pandas are optional and used only when available for speed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path
from typing import Any

from scipy import stats as _scipy_stats

from ..core import (
    ArtifactRef,
    ErrorDetail,
    Finding,
    OrganelleResult,
    ResultProvenance,
    ResultScope,
)

__all__ = [
    "coevolution_network",
    "extract_branch_lengths",
    # back-compat aliases for the old API
    "phylogenomics",
    "rate_covariation",
    "reconcile_trees",
]


# ===========================================================================
# Canonical v1 result construction (I/O boundary only — no scientific logic)
# ===========================================================================

SUITE = "coevolution"
_OPERATION_VERSION = "1.0"
#: Every coevolution operation correlates organelle-encoded against
#: nuclear-encoded gene rates, so the canonical result scope is cytonuclear.
_SCOPE: ResultScope = "cytonuclear"


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _json_safe(value: Any) -> Any:
    """Coerce a compute payload into strict JSON for ``metrics``.

    The canonical contract stores metrics as frozen JSON, which has no NaN or
    infinity. Missing branch lengths (``float('nan')``) therefore serialize as
    JSON ``null``; :func:`_thaw_branch_lengths` restores them to NaN for
    downstream numeric stages so the arithmetic is unchanged.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _thaw_branch_lengths(
    payload: Any, order: Sequence[str] | None = None
) -> dict[str, dict[str, float]]:
    """Rebuild a mutable ``{gene: {branch: float}}`` matrix from frozen metrics.

    JSON ``null`` (a missing branch) becomes ``float('nan')`` again so the
    numeric stages see exactly the values the extractor computed. ``order``
    restores the extractor's gene order, which a ``FrozenMap`` (sorted by key)
    does not preserve.
    """
    matrix: dict[str, dict[str, float]] = {}
    if not isinstance(payload, Mapping):
        return matrix
    rows = {
        str(gene): {
            str(branch): (float("nan") if item is None else float(item))
            for branch, item in row.items()
        }
        for gene, row in payload.items()
        if isinstance(row, Mapping)
    }
    for gene in [str(item) for item in order or ()]:
        if gene in rows:
            matrix[gene] = rows[gene]
    for gene, row in rows.items():
        matrix.setdefault(gene, row)
    return matrix


def _parameters_hash(parameters: Mapping[str, Any] | None = None) -> str:
    payload = json.dumps(
        _json_safe(dict(parameters or {})),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provenance(
    op: str,
    *,
    method: str = "",
    argv: Sequence[str] = (),
    parameters: Mapping[str, Any] | None = None,
    random_seed: int | None = None,
) -> ResultProvenance:
    """Build canonical provenance; the legacy ``method`` maps to the backend."""
    return ResultProvenance(
        operation_id=f"{SUITE}.{op}",
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=_parameters_hash(parameters),
        attempted_backends=(method,) if method else (),
        actual_backend=method,
        argv=tuple(str(item) for item in argv),
        random_seed=random_seed,
    )


def _finding(code: str, *, metric: str = "", value: Any = None, unit: str = "") -> Finding:
    """Convert one legacy ``key_findings`` dict into a canonical Finding."""
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    elif isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    elif value is not None and not isinstance(value, (bool, int, float, str)):
        value = str(value)
    return Finding(code=code, metric=metric or code, value=value, unit=unit)


def _ok(
    op: str,
    *,
    scope: ResultScope = _SCOPE,
    artifacts: Sequence[ArtifactRef] = (),
    metrics: Mapping[str, Any] | None = None,
    findings: Sequence[Finding] = (),
    flags: Sequence[str] = (),
    anomalies: Sequence[str] = (),
    summary_text: str = "",
    provenance: ResultProvenance | None = None,
) -> OrganelleResult:
    """Successful canonical result.

    A canonical ``ok`` result cannot carry errors, so legacy non-fatal
    ``anomalies`` are surfaced as ``anomaly:<code>`` flags. This keeps the
    status (and therefore every ``status != "ok"`` control-flow branch)
    identical to the legacy behaviour.
    """
    return OrganelleResult(
        operation_id=f"{SUITE}.{op}",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="ok",
        summary_text=summary_text,
        metrics=_json_safe(dict(metrics or {})),
        findings=tuple(findings),
        flags=tuple(flags) + tuple(f"anomaly:{item}" for item in anomalies),
        artifacts=tuple(artifacts),
        provenance=provenance if provenance is not None else _provenance(op),
    )


def _failed(
    op: str,
    *,
    scope: ResultScope = _SCOPE,
    artifacts: Sequence[ArtifactRef] = (),
    metrics: Mapping[str, Any] | None = None,
    findings: Sequence[Finding] = (),
    flags: Sequence[str] = (),
    anomalies: Sequence[str] = (),
    summary_text: str = "",
    provenance: ResultProvenance | None = None,
) -> OrganelleResult:
    """Failed canonical result; legacy ``anomalies`` become structured errors."""
    codes = tuple(str(item) for item in anomalies) or ("coevolution_failed",)
    return OrganelleResult(
        operation_id=f"{SUITE}.{op}",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="failed",
        summary_text=summary_text,
        metrics=_json_safe(dict(metrics or {})),
        findings=tuple(findings),
        flags=tuple(flags),
        artifacts=tuple(artifacts),
        provenance=provenance if provenance is not None else _provenance(op),
        errors=tuple(
            ErrorDetail(code=code, message=summary_text or code, retryable=False) for code in codes
        ),
    )


def _rust_erc_kernel(accel: Any) -> Callable[..., list[dict]] | None:
    """Return the Rust ERC kernel when the accel facade actually exposes it.

    ``accel.HAS_RUST`` only reports that the compiled extension imported; the
    facade decides which kernels it re-exports. Probing for the symbol keeps the
    documented "Rust when available, pure Python otherwise" behaviour instead of
    raising ``AttributeError`` on a facade that omits ``erc_correlation``.
    """
    if not getattr(accel, "HAS_RUST", False):
        return None
    kernel = getattr(accel, "erc_correlation", None)
    return kernel if callable(kernel) else None


# ===========================================================================
# Newick parser (self-contained, no ETE3/Biopython)
# ===========================================================================


@dataclass
class TreeNode:
    """A minimal phylogenetic tree node."""

    name: str = ""
    branch_length: float = 0.0
    children: list[TreeNode] = field(default_factory=list)
    parent: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def traverse(self):
        """Yield all nodes (pre-order)."""
        yield self
        for child in self.children:
            yield from child.traverse()

    def leaves(self) -> list[TreeNode]:
        return [n for n in self.traverse() if n.is_leaf]

    def get_distance_to_root(self) -> float:
        d = 0.0
        node = self
        while node.parent is not None:
            d += node.branch_length
            node = node.parent
        return d


def _post_order(node: TreeNode):
    """Yield nodes post-order (children before parent) — needed for bottom-up reconciliation."""
    for child in node.children:
        yield from _post_order(child)
    yield node


def parse_newick(newick: str) -> TreeNode:
    """Parse a Newick string into a TreeNode tree. Self-contained."""
    s = newick.strip().rstrip(";")
    if not s:
        return TreeNode()
    root, _pos = _parse_newick_recursive(s, 0)
    return root


def _parse_newick_recursive(s: str, pos: int) -> tuple[TreeNode, int]:
    """Recursive descent Newick parser."""
    node = TreeNode()
    if pos < len(s) and s[pos] == "(":
        pos += 1  # skip '('
        while pos < len(s) and s[pos] != ")":
            child, pos = _parse_newick_recursive(s, pos)
            child.parent = node
            node.children.append(child)
            if pos < len(s) and s[pos] == ",":
                pos += 1  # skip comma
        pos += 1  # skip ')'
    # parse name and branch length
    name_chars: list[str] = []
    while pos < len(s) and s[pos] not in ",()":
        name_chars.append(s[pos])
        pos += 1
    raw = "".join(name_chars)
    if ":" in raw:
        name, bl_str = raw.split(":", 1)
        node.name = name.strip()
        try:
            node.branch_length = float(bl_str.strip())
        except ValueError:
            node.branch_length = 0.0
    else:
        node.name = raw.strip()
    return node, pos


# ===========================================================================
# 1. GT/ST reconciliation (LCA algorithm — replaces DLCpar/ETE3)
# ===========================================================================


def reconcile_trees(
    gene_tree_newick: str,
    species_tree_newick: str,
    species_map: dict[str, str] | None = None,
) -> OrganelleResult:
    """Reconcile a gene tree against a species tree using LCA mapping.

    Implements the SDI (Speciation-Duplication-Inference) LCA algorithm:
    each gene-tree node is mapped to the LCA of its children's species in the
    species tree. A node is a **duplication** if its mapping equals one of its
    children's mappings; otherwise it's a **speciation**.

    Parameters
    ----------
    gene_tree_newick : str
        Gene tree in Newick format (leaf names contain species codes).
    species_tree_newick : str
        Species tree in Newick format (leaf names are species codes).
    species_map : dict, optional
        Maps gene-leaf-name -> species-code. If None, species code is extracted
        as the prefix before ``_``, ``|``, or ``@``.

    Compute-only: returns the per-internal-node reconciliation rows in
    ``metrics["reconciliation_rows"]`` and writes no user file. Materialize the
    DLCpar-compatible ``.locus.recon`` artifact with
    :func:`organelleverse.coevolution.write` (which routes to
    :func:`_materialize_reconcile_trees`).
    """
    if species_map is None:
        species_map = {}
    try:
        gene_tree = parse_newick(gene_tree_newick)
        species_tree = parse_newick(species_tree_newick)
    except Exception as exc:
        return _failed(
            "reconcile_trees",
            summary_text=f"Newick parse error: {exc}",
            anomalies=["parse_error"],
        )

    sp_leaves = {n.name: n for n in species_tree.leaves() if n.name}

    def get_species(leaf_name: str) -> str:
        if leaf_name in species_map:
            return species_map[leaf_name]
        for sep in ("__", "_", "|", "@"):
            if sep in leaf_name:
                return leaf_name.split(sep)[0]
        return leaf_name

    # Map gene-tree nodes to species-tree NODE OBJECTS (by id) to preserve
    # internal-node identity even when internal nodes are unnamed.
    mapping: dict[int, TreeNode] = {}  # id(gene_node) -> species TreeNode
    events: dict[int, str] = {}

    for node in _post_order(gene_tree):
        if node.is_leaf:
            sp = get_species(node.name)
            mapping[id(node)] = sp_leaves.get(sp)  # may be None
        else:
            child_maps = [mapping.get(id(c)) for c in node.children]
            child_species_nodes = [m for m in child_maps if m is not None]
            if len(child_species_nodes) == 1:
                mapped_node = child_species_nodes[0]
            elif len(child_species_nodes) >= 2:
                mapped_node = _find_lca(species_tree, child_species_nodes) or species_tree
            else:
                mapped_node = species_tree
            mapping[id(node)] = mapped_node
            # duplication if mapped node equals one of the children's mapped nodes
            is_dup = mapped_node in child_maps
            events[id(node)] = "dup" if is_dup else "spec"

    dup_count = sum(1 for e in events.values() if e == "dup")
    spec_count = sum(1 for e in events.values() if e == "spec")

    # Build the DLCpar-compatible reconciliation rows (internal nodes only) as a
    # serializable payload; the materializer reconstructs the .locus.recon text.
    reconciliation_rows: list[dict[str, str]] = []
    node_counter = 0
    for node in gene_tree.traverse():
        if node.is_leaf:
            continue
        node_counter += 1
        nid = node.name or f"n{node_counter}"
        sp_node = mapping.get(id(node))
        if sp_node and sp_node.name:
            sp_name = sp_node.name
        elif sp_node:
            sp_name = f"s{id(sp_node)}"
        else:
            sp_name = "root"
        reconciliation_rows.append(
            {"node": nid, "species": sp_name, "event": events.get(id(node), "spec")}
        )

    ratio = dup_count / spec_count if spec_count else 0.0
    return _ok(
        "reconcile_trees",
        artifacts=(),
        metrics={
            "internal_nodes": dup_count + spec_count,
            "duplications": dup_count,
            "speciations": spec_count,
            "dup_spec_ratio": round(ratio, 4),
            "reconciliation_rows": reconciliation_rows,
        },
        findings=(
            _finding("duplications", value=dup_count),
            _finding("speciations", value=spec_count),
        ),
        flags=("duplication_detected",) if dup_count > 0 else (),
        summary_text=f"Reconciled gene tree: {dup_count} dup, {spec_count} spec "
        f"(dup/spec={ratio:.3f})."
        if spec_count
        else f"Reconciled: {dup_count} dup, {spec_count} spec.",
        provenance=_provenance(
            "reconcile_trees",
            method="lca_sdi",
            parameters={"species_map": sorted(species_map)},
        ),
    )


def _materialize_reconcile_trees(result: OrganelleResult, destination: str | Path) -> Path:
    """Materialize a ``reconcile_trees`` result as a DLCpar ``.locus.recon`` file.

    An explicit file destination (a path with a suffix) is written verbatim. A
    directory destination (no suffix) is joined with the default
    ``reconcile_trees.locus.recon`` filename.
    """
    rows = result.metrics.get("reconciliation_rows", ())
    path = Path(destination)
    if not path.suffix:
        path = path / "reconcile_trees.locus.recon"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{row['node']}\t{row['species']}\t{row['event']}" for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def _find_lca(root: TreeNode, target_leaves: list[TreeNode]) -> TreeNode | None:
    """Find the LCA of target nodes, including internal nodes."""
    if not target_leaves:
        return None

    paths: list[list[TreeNode]] = []
    for target in target_leaves:
        cur: TreeNode | None = target
        path: list[TreeNode] = []
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        if not path:
            return None
        paths.append(path)

    common = {id(n) for n in paths[0]}
    for path in paths[1:]:
        common &= {id(n) for n in path}
    for node in paths[0]:
        if id(node) in common:
            return node
    return root


# ===========================================================================
# 2. Branch-length extraction (R2T = root-to-tip)
# ===========================================================================


def extract_branch_lengths(
    gene_trees: dict[str, str],
    *,
    method: str = "R2T",
    species_tree: str | None = None,
    species_map: dict[str, str] | None = None,
    reconciliation: str = "lca",
    dlcpar_dir: str | Path | None = None,
    bxb_normalization: str = "edge_mean",
) -> OrganelleResult:
    """Extract branch-length vectors from multiple gene trees.

    Parameters
    ----------
    gene_trees : {gene_name: newick_string}
    method : "R2T" (root-to-tip, default) | "BXB" (branch-by-branch)
    species_tree : str, optional
        Required for ``method="BXB"``. The BXB output is indexed by species-tree
        edges, not raw gene-tree node names.
    species_map : dict, optional
        Optional gene-leaf-name -> species-code mapping for BXB (``reconciliation="lca"``).
    reconciliation : "lca" (default) | "dlcpar"
        How gene-tree nodes are mapped onto species-tree edges for BXB.

        * ``"lca"``  — pure-Python LCA/SDI reconciliation (no external deps).
          Suitable for near-complete gene trees with low taxon loss. **Known
          limitation:** on highly partial gene trees (e.g. ERC45, ~10% edge
          coverage per gene), LCA diverges from DLCpar's locus-tree mapping and
          produces systematically different ERC signals. For reproduction of
          published ERCnet results on such data, use ``"dlcpar"``.
        * ``"dlcpar"`` — reads official DLCpar ``.locus.recon`` + ``.coal.recon``
          files (bit-for-bit with ERCnet's R ``Branch_length_reconciliation.R``).
          Requires ``dlcpar_dir`` pointing at a DLCpar output directory whose
          files are named ``{gene_name}_NODES_BL.txt`` (+ ``.dlcpar.locus.recon``
          / ``.dlcpar.coal.recon`` siblings).
    dlcpar_dir : path, optional
        Directory of DLCpar output files. Required when ``reconciliation="dlcpar"``.
        Each ``gene_trees`` key must correspond to a ``{key}_NODES_BL.txt`` tree
        plus its ``.dlcpar.locus.recon`` and ``.dlcpar.coal.recon`` siblings.
    bxb_normalization : "edge_mean" (default) | "gene_sum" | "none"
        Normalization applied after BXB extraction.

        * ``"edge_mean"`` — official ERCnet: each species-tree edge divided by
          its cross-gene mean (R ``Branch_length_reconciliation.R`` lines 217-219).
          This is the "genome-wide average" normalization. **Default.**
        * ``"gene_sum"`` — each gene's vector divided by its non-missing sum
          (alternative per-gene scaling; not the official ERCnet behavior).
        * ``"none"`` — raw extracted branch lengths.
    """
    if not gene_trees:
        return _failed(
            "extract_branch_lengths",
            summary_text="extract_branch_lengths() requires >=1 gene tree.",
            anomalies=["no_gene_trees"],
        )

    method_key = method.upper()
    if method_key == "BXB":
        if not species_tree:
            return _failed(
                "extract_branch_lengths",
                summary_text="method='BXB' requires species_tree.",
                anomalies=("species_tree_required",),
                provenance=_provenance("extract_branch_lengths", method="BXB"),
            )
        from ._bxb_extract import (
            bxb_lengths_dlcpar,
            bxb_lengths_lca,
            filter_outlier_tree,
            normalize_edge_mean,
            normalize_gene_sum,
            species_tree_edges,
        )

        edges = species_tree_edges(species_tree)
        recon_key = reconciliation.lower()
        if recon_key not in ("lca", "dlcpar"):
            return _failed(
                "extract_branch_lengths",
                summary_text="reconciliation must be 'lca' or 'dlcpar'.",
                anomalies=("invalid_reconciliation",),
                provenance=_provenance("extract_branch_lengths", method="BXB"),
            )
        if recon_key == "dlcpar" and not dlcpar_dir:
            return _failed(
                "extract_branch_lengths",
                summary_text="reconciliation='dlcpar' requires dlcpar_dir.",
                anomalies=("dlcpar_dir_required",),
                provenance=_provenance("extract_branch_lengths", method="BXB"),
            )
        dlcpar_path = Path(dlcpar_dir) if dlcpar_dir else None

        bl_matrix: dict[str, dict[str, float]] = {}
        parse_errors = 0
        outliers = 0
        for gene_name, newick in gene_trees.items():
            try:
                if recon_key == "dlcpar":
                    # DLCpar files: {gene}_NODES_BL.txt + .dlcpar.locus.recon + .dlcpar.coal.recon
                    tree_file = dlcpar_path / f"{gene_name}_NODES_BL.txt"
                    locus_recon = dlcpar_path / f"{gene_name}_NODES_BL.txt.dlcpar.locus.recon"
                    coal_recon = dlcpar_path / f"{gene_name}_NODES_BL.txt.dlcpar.coal.recon"
                    if not (tree_file.exists() and locus_recon.exists() and coal_recon.exists()):
                        parse_errors += 1
                        continue
                    # Use the on-disk NODES_BL tree (authoritative branch lengths)
                    newick = tree_file.read_text().strip()
                    row = bxb_lengths_dlcpar(newick, locus_recon, coal_recon, edges)
                else:
                    row = bxb_lengths_lca(newick, species_tree, edges, species_map)
            except Exception:
                parse_errors += 1
                continue
            filtered = filter_outlier_tree(row)
            if filtered is None:
                outliers += 1
                row = {edge: float("nan") for edge in edges}
            else:
                row = filtered
            bl_matrix[gene_name] = {edge: row.get(edge, float("nan")) for edge in edges}

        norm_key = bxb_normalization.lower()
        if norm_key in ("edge_mean", "column_mean", "genome_wide_average"):
            bl_matrix = normalize_edge_mean(bl_matrix)
            norm_label = "edge_mean"
        elif norm_key in ("gene_sum", "sum", "per_gene"):
            bl_matrix = normalize_gene_sum(bl_matrix)
            norm_label = "gene_sum"
        elif norm_key in ("none", "raw"):
            norm_label = "none"
        else:
            return _failed(
                "extract_branch_lengths",
                summary_text=("bxb_normalization must be 'edge_mean', 'gene_sum', or 'none'."),
                anomalies=("invalid_bxb_normalization",),
                provenance=_provenance("extract_branch_lengths", method="BXB"),
            )

        non_missing = sum(1 for row in bl_matrix.values() for val in row.values() if val == val)
        n_species = _species_tree_leaf_count(species_tree)
        return _ok(
            "extract_branch_lengths",
            artifacts=(),
            metrics={
                "n_genes": len(bl_matrix),
                "n_branches": len(edges),
                "n_species": n_species,
                "method": "BXB",
                "reconciliation": recon_key,
                "normalization": norm_label,
                "branch_order": edges,
                # Frozen metrics sort their keys; keep the extraction order so
                # downstream stages and materialized tables stay row-stable.
                "gene_order": list(bl_matrix),
                "branch_lengths": bl_matrix,
                "non_missing_branch_lengths": non_missing,
                "parse_errors": parse_errors,
                "outlier_trees": outliers,
            },
            findings=(
                _finding("n_genes", value=len(bl_matrix)),
                _finding("n_branches", value=len(edges)),
                _finding("normalization", value=norm_label),
                _finding("reconciliation", value=recon_key),
            ),
            flags=("bxb_species_edge_matrix",),
            anomalies=("parse_errors",) if parse_errors else (),
            summary_text=(
                f"Extracted BXB branch lengths: {len(bl_matrix)} genes x "
                f"{len(edges)} species-tree branches "
                f"({recon_key}/{norm_label})."
            ),
            provenance=_provenance(
                "extract_branch_lengths",
                method=f"BXB/{recon_key}/{norm_label}",
                parameters={
                    "method": "BXB",
                    "reconciliation": recon_key,
                    "bxb_normalization": norm_label,
                    "dlcpar_dir": str(dlcpar_path) if dlcpar_path else None,
                },
            ),
        )

    bl_matrix: dict[str, dict[str, float]] = {}
    all_species: set[str] = set()
    for gene_name, newick in gene_trees.items():
        try:
            tree = parse_newick(newick)
        except Exception:
            continue
        row: dict[str, float] = {}
        for leaf in tree.leaves():
            sp = leaf.name.split("_")[0].split("|")[0].split("@")[0]
            row[sp] = leaf.get_distance_to_root()
        bl_matrix[gene_name] = row
        all_species.update(row.keys())

    return _ok(
        "extract_branch_lengths",
        artifacts=(),
        metrics={
            "n_genes": len(bl_matrix),
            "n_species": len(all_species),
            "branch_lengths": bl_matrix,
            "method": method,
        },
        findings=(
            _finding("n_genes", value=len(bl_matrix)),
            _finding("n_species", value=len(all_species)),
        ),
        flags=(),
        summary_text=f"Extracted branch lengths: {len(bl_matrix)} genes × "
        f"{len(all_species)} species ({method}).",
        provenance=_provenance(
            "extract_branch_lengths", method=method, parameters={"method": method}
        ),
    )


# ===========================================================================
# 3. ERC2 residual-based rate covariation (core algorithm)
# ===========================================================================


def rate_covariation(
    branch_lengths: dict[str, list[float] | dict[str, float]],
    *,
    method: str = "residual",
    transform: str = "sqrt",
    variance_weight: bool = False,
    min_overlap: int = 5,
    fdr_threshold: float = 0.05,
    compartment_map: dict[str, Any] | None = None,
    restrict: tuple[str, str] | None = None,
) -> OrganelleResult:
    """Compute pairwise ERC (evolutionary rate covariation).

    Implements the ERC2 residual-based method:
    1. Transform branch lengths (sqrt/log/none).
    2. Compute master branch lengths (trimmed mean across genes).
    3. Regress each gene against master → residuals (OLS or WLS).
    4. Compute all-by-all Pearson + Spearman correlations on residuals.
    5. Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    branch_lengths : {gene_name: [branch_length_per_species]} or {gene_name: {branch: value}}
    method : "residual" (ERC2, default) | "raw"
    transform : "sqrt" (default) | "log" | "none"
    variance_weight : bool — use WLS with variance-based weights
    min_overlap : int — minimum overlapping branches for correlation
    fdr_threshold : float — Benjamini-Hochberg significance threshold
    compartment_map : dict, optional
        Maps gene -> compartment string or metadata dict with ``compartment``.
    restrict : (compartment_a, compartment_b), optional
        If set, test only pairs whose canonical compartments match this pair
        (order-insensitive), e.g. ``("mito", "nuclear")``.
    """
    if restrict is not None and compartment_map is None:
        return _failed(
            "rate_covariation",
            summary_text="restrict requires compartment_map.",
            anomalies=("compartment_map_required",),
        )

    branch_vectors = _coerce_branch_length_vectors(branch_lengths)
    genes = list(branch_vectors)
    if len(genes) < 2:
        return _failed(
            "rate_covariation",
            summary_text="rate_covariation() requires >=2 genes.",
            anomalies=["too_few_genes"],
        )

    # --- Rust fast path (optional acceleration) ---
    from .. import accel

    rust_kernel = _rust_erc_kernel(accel)
    if rust_kernel is not None and not variance_weight and restrict is None:
        bl_list = [branch_vectors[g] for g in genes]
        rust_pairs = rust_kernel(genes, bl_list, method, transform, min_overlap, fdr_threshold)
        significant = [p for p in rust_pairs if p.get("significant")]
        mean_r = (sum(abs(p["r"]) for p in rust_pairs) / len(rust_pairs)) if rust_pairs else 0.0
        return _ok(
            "rate_covariation",
            artifacts=(),
            metrics={
                "pairs_tested": len(rust_pairs),
                "significant_pairs": len(significant),
                "mean_abs_r": round(mean_r, 4),
                "method": method,
                "transform": transform,
                "accelerated": True,
                "pair_scope": "all",
                "pairs": rust_pairs,
            },
            findings=tuple(
                _finding(
                    "erc_pair",
                    metric=f"{p['gene_a']}-{p['gene_b']}",
                    value=p["r"],
                )
                for p in significant[:5]
            )
            or (_finding("mean_abs_r", value=round(mean_r, 4)),),
            flags=("covariation_detected", "rust_accelerated")
            if significant
            else ("rust_accelerated",),
            summary_text=f"{len(rust_pairs)} pairs tested (Rust); {len(significant)} significant "
            f"(FDR<{fdr_threshold}); mean |r|={mean_r:.3f}.",
            provenance=_provenance(
                "rate_covariation",
                method=f"{method}_rust",
                parameters={
                    "method": method,
                    "transform": transform,
                    "min_overlap": min_overlap,
                    "fdr_threshold": fdr_threshold,
                    "variance_weight": variance_weight,
                },
            ),
        )

    # --- Pure Python path ---
    pair_scope = _format_pair_scope(restrict)
    transformed = {g: _transform_bl(branch_vectors[g], transform) for g in genes}
    master = _compute_master_bl(transformed, trim_prop=0.05)

    if method == "residual":
        weights = _compute_variance_weights(transformed, master) if variance_weight else None
        vectors = {g: _compute_residuals(transformed[g], master, weights) for g in genes}
    else:
        vectors = transformed

    pairs: list[dict] = []
    for ga, gb in combinations(genes, 2):
        if not _pair_allowed(ga, gb, compartment_map, restrict):
            continue
        r, rho, tau, n = _correlate(vectors[ga], vectors[gb], min_overlap)
        if n >= min_overlap:
            pairs.append(
                {
                    "gene_a": ga,
                    "gene_b": gb,
                    "r": round(r, 4),
                    "rho": round(rho, 4),
                    "tau": round(tau, 4),
                    "n_branches": n,
                    "pval": _pearson_pvalue(r, n),
                }
            )

    _apply_fdr(pairs, fdr_threshold)
    pairs.sort(key=lambda p: p["pval"])
    significant = [p for p in pairs if p.get("significant", False)]
    mean_r = sum(abs(p["r"]) for p in pairs) / len(pairs) if pairs else 0.0

    return _ok(
        "rate_covariation",
        artifacts=(),
        metrics={
            "pairs_tested": len(pairs),
            "significant_pairs": len(significant),
            "mean_abs_r": round(mean_r, 4),
            "method": method,
            "transform": transform,
            "pair_scope": pair_scope,
            "pairs": pairs,
        },
        findings=tuple(
            _finding(
                "erc_pair",
                metric=f"{p['gene_a']}-{p['gene_b']}",
                value=p["r"],
            )
            for p in significant[:5]
        )
        or (_finding("mean_abs_r", value=round(mean_r, 4)),),
        flags=("covariation_detected",) if significant else (),
        summary_text=f"{len(pairs)} pairs tested; {len(significant)} significant "
        f"(FDR<{fdr_threshold}); mean |r|={mean_r:.3f}.",
        provenance=_provenance(
            "rate_covariation",
            method=method,
            parameters={
                "method": method,
                "transform": transform,
                "min_overlap": min_overlap,
                "fdr_threshold": fdr_threshold,
                "variance_weight": variance_weight,
                "pair_scope": pair_scope,
            },
        ),
    )


# ===========================================================================
# 4. Coevolution network
# ===========================================================================


def coevolution_network(
    correlations: list[Mapping[str, str | int | float | bool | None]],
    *,
    r_threshold: float = 0.5,
    use_fdr: bool = True,
) -> OrganelleResult:
    """Build a coevolution network from correlation results."""
    nodes: set[str] = set()
    edges: list[tuple[str, str, float]] = []
    for p in correlations:
        include = (use_fdr and p.get("significant")) or (abs(p.get("r", 0)) >= r_threshold)
        if include:
            a, b = p["gene_a"], p["gene_b"]
            nodes.add(a)
            nodes.add(b)
            edges.append((a, b, p["r"]))

    communities = _connected_components(list(nodes), [(a, b) for a, b, _ in edges])
    return _ok(
        "coevolution_network",
        artifacts=(),
        metrics={
            "nodes": len(nodes),
            "edges": len(edges),
            "communities": len(communities),
        },
        findings=(
            _finding("edges", value=len(edges)),
            _finding("communities", value=len(communities)),
        ),
        flags=("network_built",),
        summary_text=f"Network: {len(nodes)} nodes, {len(edges)} edges, "
        f"{len(communities)} communities.",
        provenance=_provenance(
            "coevolution_network",
            method="organelleverse",
            parameters={"r_threshold": r_threshold, "use_fdr": use_fdr},
        ),
    )


# ===========================================================================
# Back-compat: phylogenomics planner (old API — delegates to executor)
# ===========================================================================


def phylogenomics(
    orthofinder_dir: str | Path,
    *,
    method: str = "mafft_iqtree",
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Plan/run phylogenomics stage: HOG filter -> MAFFT -> IQ-TREE (executor hook)."""
    argv = ["python3", "Phylogenomics.py", "-i", str(orthofinder_dir)]
    ran = False
    if executor is not None:
        try:
            executor(list(argv))
            ran = True
        except Exception:
            pass
    return _ok(
        "phylogenomics",
        artifacts=(),
        metrics={"method": method, "argv": argv},
        findings=(_finding("argv", value=argv),),
        flags=("executed",) if ran else ("planned",),
        summary_text=f"Phylogenomics ({method}); {'ran' if ran else 'planned'}.",
        provenance=_provenance(
            "phylogenomics",
            method=method,
            argv=argv,
            parameters={"method": method, "orthofinder_dir": str(orthofinder_dir)},
        ),
    )


# ===========================================================================
# ERC2 algorithm helpers (faithful to ERCnet2/ERC_functions.py)
# ===========================================================================


def _is_missing(v: float) -> bool:
    """True if v is NaN (missing data). A true 0.0 is NOT missing."""
    return v != v  # NaN is the only float where v != v


def _species_tree_leaf_count(species_tree_newick: str) -> int:
    try:
        return len(parse_newick(species_tree_newick).leaves())
    except Exception:
        return 0


def _coerce_branch_length_vectors(
    branch_lengths: dict[str, list[float] | dict[str, float]],
) -> dict[str, list[float]]:
    """Coerce list- or branch-keyed matrices into aligned numeric vectors."""
    if not branch_lengths:
        return {}
    first = next(iter(branch_lengths.values()))
    if isinstance(first, dict):
        keys = sorted({k for row in branch_lengths.values() if isinstance(row, dict) for k in row})
        return {
            gene: [float(row.get(k, float("nan"))) for k in keys]
            for gene, row in branch_lengths.items()
            if isinstance(row, dict)
        }
    return {gene: [float(v) for v in row] for gene, row in branch_lengths.items()}


def _canonical_compartment(raw: Any) -> str | None:
    """Normalize compartment labels used for cytonuclear/organelle ERC."""
    if isinstance(raw, dict):
        raw = raw.get("compartment") or raw.get("source") or raw.get("origin") or raw.get("target")
    if raw is None:
        return None
    value = str(raw).strip().lower()
    aliases = {
        "mt": "mito",
        "mitochondrial": "mito",
        "mitochondrion": "mito",
        "mitochondria": "mito",
        "nuc": "nuclear",
        "nucl": "nuclear",
        "nucleus": "nuclear",
        "cp": "plastid",
        "chloro": "plastid",
        "chloroplast": "plastid",
        "chloroplastid": "plastid",
        "plastome": "plastid",
    }
    return aliases.get(value, value)


def _format_pair_scope(restrict: tuple[str, str] | None) -> str:
    if restrict is None:
        return "all"
    a = _canonical_compartment(restrict[0]) or str(restrict[0])
    b = _canonical_compartment(restrict[1]) or str(restrict[1])
    return f"{a}:{b}"


def _pair_allowed(
    gene_a: str,
    gene_b: str,
    compartment_map: dict[str, Any] | None,
    restrict: tuple[str, str] | None,
) -> bool:
    """Return True when a gene pair matches the optional compartment scope."""
    if restrict is None:
        return True
    if compartment_map is None:
        return False
    want_a = _canonical_compartment(restrict[0])
    want_b = _canonical_compartment(restrict[1])
    got_a = _canonical_compartment(compartment_map.get(gene_a))
    got_b = _canonical_compartment(compartment_map.get(gene_b))
    if got_a is None or got_b is None:
        return False
    if want_a == want_b:
        return got_a == want_a and got_b == want_b
    return {got_a, got_b} == {want_a, want_b}


def compartment_map_from_prefix(
    genes: list[str],
    prefix_rules: dict[str, str],
    default: str = "nuclear",
) -> dict[str, str]:
    """Build a ``{gene: compartment}`` map from name-prefix rules.

    A convenience helper for the common ERCnet cytonuclear pattern, where
    compartment origin is encoded as a gene-name prefix (e.g. ``"MT_"`` for
    mitochondrial-encoded genes). The reference
    ``run_targeted_mt_vs_nuclear_bxb_r2t.py`` splits by ``hog.startswith("MT_")``;
    this helper generalizes that to arbitrary prefixes and compartments.

    Parameters
    ----------
    genes : list[str]
        Gene identifiers (HOG ids, gene names, …).
    prefix_rules : dict[str, str]
        ``{prefix: compartment}``. A gene whose name *starts with* a prefix is
        assigned that compartment. Example::

            {"MT_": "mito", "CP_": "plastid"}
    default : str
        Compartment assigned to genes matching no prefix. Default ``"nuclear"``
        (the cytonuclear convention: anything not organelle-encoded is nuclear).

    Returns
    -------
    dict[str, str]
        ``{gene: canonical_compartment}`` with values normalized via
        :func:`_canonical_compartment` (e.g. ``"mito"`` / ``"nuclear"`` /
        ``"plastid"``). Genes are checked against prefixes in dict order; the
        first match wins, so order longer/more-specific prefixes first.

    Examples
    --------
    >>> compartment_map_from_prefix(
    ...     ["MT_rps12", "MT_nad1", "HOG0001", "CP_rbcL"],
    ...     {"MT_": "mito", "CP_": "plastid"},
    ... )
    {'MT_rps12': 'mito', 'MT_nad1': 'mito', 'HOG0001': 'nuclear', 'CP_rbcL': 'plastid'}
    """
    out: dict[str, str] = {}
    for gene in genes:
        comp = default
        for prefix, compartment in prefix_rules.items():
            if gene.startswith(prefix):
                comp = compartment
                break
        canon = _canonical_compartment(comp)
        out[gene] = canon if canon else str(default)
    return out


def _transform_bl(vec: list[float], method: str) -> list[float]:
    """Transform branch lengths (transformPaths equivalent).

    Faithful to ERCnet2: sqrt = np.sqrt, log = np.log(v + min_pos).
    Branch lengths are non-negative; no sign handling needed.

    Missing data (NaN) is preserved as NaN (matching ``np.sqrt(NaN) = NaN``).
    A true 0.0 is transformed to 0.0 (sqrt) / log(min_pos) (log).
    """
    if method == "sqrt":
        return [math.sqrt(v) if v > 0 else (0.0 if v == 0.0 else float("nan")) for v in vec]
    if method == "log":
        min_pos = min((v for v in vec if v > 0), default=1e-6)
        return [math.log(v + min_pos) if v >= 0 else float("nan") for v in vec]
    return list(vec)


def _compute_master_bl(genes: dict[str, list[float]], trim_prop: float = 0.05) -> list[float]:
    """Trimmed-mean master branch lengths (compute_master_branch_lengths).

    Missing data is represented as NaN (matching ERCnet2); a true 0.0 branch
    length is a valid value and is included in the trimmed mean.
    """
    if not genes:
        return []
    L = min(len(v) for v in genes.values())
    master = []
    for i in range(L):
        col = sorted(v[i] for v in genes.values() if i < len(v) and not _is_missing(v[i]))
        if not col:
            master.append(float("nan"))
            continue
        n = len(col)
        trim = int(n * trim_prop)
        trimmed = col[trim : n - trim] if n > 2 * trim else col
        master.append(sum(trimmed) / len(trimmed) if trimmed else float("nan"))
    return master


def _compute_variance_weights(genes: dict[str, list[float]], master: list[float]) -> list[float]:
    """Variance-based WLS weights (compute_variance_weights).

    Fits log(var) ~ log(mean) and returns 1/predicted_var.
    """
    L = min(len(v) for v in genes.values()) if genes else 0
    means: list[float] = []
    variances: list[float] = []
    for i in range(L):
        col = [v[i] for v in genes.values() if i < len(v) and not _is_missing(v[i])]
        if len(col) > 2:
            m = sum(col) / len(col)
            var = sum((x - m) ** 2 for x in col) / (len(col) - 1)
            if m > 0 and var > 0:
                means.append(m)
                variances.append(var)
    if len(means) < 5:
        return [1.0] * L
    log_m = [math.log(m) for m in means]
    log_v = [math.log(v) for v in variances]
    n = len(log_m)
    denom = n * sum(lm**2 for lm in log_m) - sum(log_m) ** 2 + 1e-10
    b = (n * sum(lm * lv for lm, lv in zip(log_m, log_v, strict=False)) - sum(log_m) * sum(log_v)) / denom
    a = (sum(log_v) - b * sum(log_m)) / n
    return [
        1.0 / math.exp(a + b * math.log(v))
        if (v > 0 and math.exp(a + b * math.log(v)) > 0)
        else 1.0
        for v in master
    ]


def _compute_residuals(
    vec: list[float], master: list[float], weights: list[float] | None = None
) -> list[float]:
    """OLS or WLS residuals (compute_residuals).

    Missing data (NaN) is excluded from the fit; the residual at missing
    positions is NaN (matching ERCnet2 semantics). A true 0.0 is a valid value.
    """
    n = min(len(vec), len(master))
    valid_mask = [i for i in range(n) if not _is_missing(master[i]) and not _is_missing(vec[i])]
    pairs = [(master[i], vec[i]) for i in valid_mask]
    if len(pairs) < 2:
        return [float("nan")] * n
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    if weights is not None:
        w = [weights[i] for i in valid_mask]
        sw = sum(w) or 1.0
        w = [wi * len(w) / sw for wi in w]
        swmx = sum(wi * (p[0] - mx) ** 2 for wi, p in zip(w, pairs, strict=False))
        swmy = sum(wi * (p[0] - mx) * (p[1] - my) for wi, p in zip(w, pairs, strict=False))
        slope = swmy / swmx if swmx else 0.0
    else:
        num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
        den = sum((p[0] - mx) ** 2 for p in pairs)
        slope = num / den if den else 0.0
    intercept = my - slope * mx
    result = [float("nan")] * n
    for i in valid_mask:
        result[i] = vec[i] - (slope * master[i] + intercept)
    return result


def _correlate(a: list[float], b: list[float], min_overlap: int) -> tuple[float, float, float, int]:
    """Pearson + Spearman + Kendall correlation over overlapping entries.

    Missing data (NaN) is excluded from the overlap. Returns (r, rho, tau, n).
    Uses scipy.stats for all three correlations.
    """
    n = min(len(a), len(b))
    pairs = [(a[i], b[i]) for i in range(n) if not _is_missing(a[i]) and not _is_missing(b[i])]
    if len(pairs) < min_overlap:
        return 0.0, 0.0, 0.0, len(pairs)
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0, 0.0, 0.0, len(pairs)
    try:
        r = float(_scipy_stats.pearsonr(x, y)[0])
    except Exception:
        r = 0.0
    if r != r:
        r = 0.0
    try:
        rho = float(_scipy_stats.spearmanr(x, y)[0])
    except Exception:
        rho = 0.0
    if rho != rho:
        rho = 0.0
    try:
        tau = float(_scipy_stats.kendalltau(x, y)[0])
    except Exception:
        tau = 0.0
    if tau != tau:
        tau = 0.0
    return r, rho, tau, len(pairs)


def _kendall_tau(x: list[float], y: list[float]) -> float:
    """Kendall's tau-b via scipy.stats (kept for backward compat)."""
    if len(x) < 2:
        return 0.0
    t = float(_scipy_stats.kendalltau(x, y)[0])
    return 0.0 if t != t else t


def _rank(values: list[float]) -> list[float]:
    """Compute ranks (average for ties) via scipy.stats.rankdata."""
    return list(_scipy_stats.rankdata(values, method="average"))


def _pearson_pvalue(r: float, n: int) -> float:
    """Two-tailed p-value for Pearson r via scipy.stats.t."""
    if n <= 2:
        return 1.0
    if abs(r) >= 1.0:
        return 0.0
    df = n - 2
    t_stat = r * math.sqrt(df / (1.0 - r * r))
    return float(_scipy_stats.t.sf(abs(t_stat), df) * 2)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Deprecated shim — p-values now come from scipy.stats.t."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return float(_scipy_stats.betainc(a, b, x))


def _apply_fdr(pairs: list[dict], threshold: float) -> None:
    """Benjamini-Hochberg FDR via scipy.stats.false_discovery_control."""
    if not pairs:
        return
    if hasattr(_scipy_stats, "false_discovery_control"):
        pvals = [p["pval"] for p in pairs]
        adjusted = _scipy_stats.false_discovery_control(pvals, method="bh")
        for p, adj in zip(pairs, adjusted, strict=False):
            adj_f = float(adj)
            p["fdr_pval"] = adj_f
            p["significant"] = adj_f < threshold
        return
    # Fallback: hand-rolled BH step-up
    sorted_pairs = sorted(pairs, key=lambda p: p["pval"])
    m = len(sorted_pairs)
    for rank, p in enumerate(sorted_pairs, 1):
        p["fdr_pval"] = min(1.0, p["pval"] * m / rank)
        p["significant"] = p["fdr_pval"] < threshold
    prev = 1.0
    for p in reversed(sorted_pairs):
        p["fdr_pval"] = min(p["fdr_pval"], prev)
        p["significant"] = p["fdr_pval"] < threshold
        prev = p["fdr_pval"]


def _connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen: set[str] = set()
    comps: list[list[str]] = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        comp: list[str] = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj.get(x, set()) - seen)
        comps.append(comp)
    return comps


# ===========================================================================
# P2: Composable cores + OrthoFinder orchestrator
# ===========================================================================


def _default_species_of(name: str) -> str:
    """Extract species code from a leaf name (default: prefix before _/|/@)."""
    return name.split("_")[0].split("|")[0].split("@")[0]


def _branch_length_matrix(
    gene_trees: dict[str, str],
    species_of: Callable[[str], str] = _default_species_of,
) -> tuple[list[str], dict[str, list[float]]]:
    """Build (species_order, {gene: [R2T per species]}) from gene-tree newicks.

    Each gene tree's leaf root-to-tip distances are collected per species;
    missing species get 0.0. This is the shared core for
    ``extract_branch_lengths`` and the ``run_coevolution`` orchestrator.
    """
    rows: dict[str, dict[str, float]] = {}
    all_species: set[str] = set()
    for gene, newick in gene_trees.items():
        try:
            tree = parse_newick(newick)
        except Exception:
            continue
        row: dict[str, float] = {}
        for leaf in tree.leaves():
            if leaf.name:
                row[species_of(leaf.name)] = leaf.get_distance_to_root()
        rows[gene] = row
        all_species.update(row)
    species_order = sorted(all_species)
    matrix = {g: [row.get(sp, 0.0) for sp in species_order] for g, row in rows.items()}
    return species_order, matrix


def _erc_pairs(
    branch_lengths: dict[str, list[float] | dict[str, float]],
    *,
    method: str = "residual",
    transform: str = "sqrt",
    variance_weight: bool = False,
    min_overlap: int = 5,
    fdr_threshold: float = 0.05,
    compartment_map: dict[str, Any] | None = None,
    restrict: tuple[str, str] | None = None,
) -> list[dict]:
    """All-by-all ERC pairs (with FDR). Rust fast path when available.

    Returns a list of dicts: gene_a, gene_b, r, rho, n_branches, pval,
    fdr_pval, significant.
    """
    branch_vectors = _coerce_branch_length_vectors(branch_lengths)
    genes = list(branch_vectors)
    if len(genes) < 2:
        return []
    from .. import accel

    rust_kernel = _rust_erc_kernel(accel)
    if rust_kernel is not None and not variance_weight and restrict is None:
        bl_list = [branch_vectors[g] for g in genes]
        return list(rust_kernel(genes, bl_list, method, transform, min_overlap, fdr_threshold))
    transformed = {g: _transform_bl(branch_vectors[g], transform) for g in genes}
    master = _compute_master_bl(transformed, trim_prop=0.05)
    if method == "residual":
        weights = _compute_variance_weights(transformed, master) if variance_weight else None
        vectors = {g: _compute_residuals(transformed[g], master, weights) for g in genes}
    else:
        vectors = transformed
    pairs: list[dict] = []
    for ga, gb in combinations(genes, 2):
        if not _pair_allowed(ga, gb, compartment_map, restrict):
            continue
        r, rho, tau, n = _correlate(vectors[ga], vectors[gb], min_overlap)
        if n >= min_overlap:
            pairs.append(
                {
                    "gene_a": ga,
                    "gene_b": gb,
                    "r": round(r, 4),
                    "rho": round(rho, 4),
                    "tau": round(tau, 4),
                    "n_branches": n,
                    "pval": _pearson_pvalue(r, n),
                }
            )
    _apply_fdr(pairs, fdr_threshold)
    pairs.sort(key=lambda p: p["pval"])
    return pairs


def _network_from_pairs(
    pairs: list[dict],
    *,
    r_threshold: float = 0.5,
    use_fdr: bool = True,
) -> tuple[list[str], list[tuple[str, str, float]], list[list[str]]]:
    """Return (nodes, edges, communities) from ERC pairs (connected-components)."""
    nodes: set[str] = set()
    edges: list[tuple[str, str, float]] = []
    for p in pairs:
        include = (use_fdr and p.get("significant")) or (abs(p.get("r", 0)) >= r_threshold)
        if include:
            a, b = p["gene_a"], p["gene_b"]
            nodes.add(a)
            nodes.add(b)
            edges.append((a, b, p["r"]))
    communities = _connected_components(list(nodes), [(a, b) for a, b, _ in edges])
    return sorted(nodes), edges, communities


def _locate_results(output_dir: str | Path) -> Path | None:
    """Find the OrthoFinder Results directory under output_dir."""
    out = Path(output_dir)
    if not out.exists():
        return None
    candidates = sorted(out.glob("Results_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    if (out / "Species_Tree").is_dir():
        return out
    return None


def _parse_orthofinder_outputs(results_dir: str | Path) -> tuple[str, dict[str, str]]:
    """Parse species tree + per-OG gene trees from an OrthoFinder Results dir.

    OrthoFinder 3.x writes all resolved gene trees into one merged file
    (``Resolved_Gene_Trees/Resolved_Gene_Trees.txt``, one ``OG id: newick``
    per line); 2.x wrote one ``*_tree.txt`` per orthogroup under
    ``Gene_Trees/``. Both layouts are accepted.
    """
    rd = Path(results_dir)
    sp_dir = rd / "Species_Tree"
    labelled_tree = sp_dir / "SpeciesTree_rooted_node_labels.txt"
    sp_file = labelled_tree if labelled_tree.exists() else sp_dir / "SpeciesTree_rooted.txt"
    species_tree = sp_file.read_text().strip() if sp_file.exists() else ""
    gene_trees: dict[str, str] = {}
    merged = rd / "Resolved_Gene_Trees" / "Resolved_Gene_Trees.txt"
    if merged.is_file():
        for line in merged.read_text().splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            og, newick = line.split(":", 1)
            og = og.strip()
            if og and newick.strip():
                gene_trees[og] = newick.strip()
        return species_tree, gene_trees
    gt_dir = rd / "Resolved_Gene_Trees"
    if not gt_dir.is_dir():
        gt_dir = rd / "Gene_Trees"
    if gt_dir.is_dir():
        for f in sorted(gt_dir.glob("*.txt")):
            og = f.stem.replace("_tree", "")
            gene_trees[og] = f.read_text().strip()
    return species_tree, gene_trees


def run_orthofinder(
    proteome_dir: str | Path,
    *,
    output_dir: str | Path,
    threads: int = 8,
    msa: bool = True,
    search: str = "diamond",
    executor: Callable[[list[str]], Any] | None = None,
) -> OrganelleResult:
    """Run (or plan) OrthoFinder3 on a directory of per-species proteome FASTAs."""
    argv = [
        "orthofinder",
        "-f",
        str(proteome_dir),
        "-t",
        str(threads),
        "-S",
        search,
        "-o",
        str(output_dir),
    ]
    if msa:
        argv += ["-M", "msa", "-A", "mafft", "-T", "fasttree"]
    prov = _provenance(
        "run_orthofinder",
        method="orthofinder3",
        argv=argv,
        parameters={
            "proteome_dir": str(proteome_dir),
            "threads": threads,
            "msa": msa,
            "search": search,
        },
    )
    if executor is None:
        return _ok(
            "run_orthofinder",
            metrics={"argv": argv},
            findings=(_finding("argv", value=argv),),
            flags=("planned",),
            summary_text="Planned OrthoFinder3 run (executor not provided).",
            provenance=prov,
        )
    try:
        completed = executor(list(argv))
    except Exception as exc:
        return _failed(
            "run_orthofinder",
            summary_text=f"OrthoFinder execution failed: {exc}",
            anomalies=[f"executor_error:{type(exc).__name__}"],
            provenance=prov,
        )
    # A zero-exit check matters: executors capture stderr, so a failed run
    # would otherwise surface later as a misleading "no Results_* directory".
    returncode = getattr(completed, "returncode", None)
    if returncode not in (None, 0):
        stderr_tail = str(getattr(completed, "stderr", "") or "")[-300:]
        return _failed(
            "run_orthofinder",
            summary_text=(
                f"OrthoFinder exited with status {returncode}."
                + (f" {stderr_tail}" if stderr_tail else "")
            ),
            anomalies=["orthofinder_failed"],
            provenance=prov,
        )
    results = _locate_results(output_dir)
    if results is None:
        return _failed(
            "run_orthofinder",
            summary_text="OrthoFinder ran but no Results_* directory was found.",
            anomalies=["no_results_dir"],
            provenance=prov,
        )
    sp_tree, gene_trees = _parse_orthofinder_outputs(results)
    # ``results`` is a directory tree; ArtifactRef addresses single files, so the
    # OrthoFinder results root is reported as a metric instead of an artifact.
    return _ok(
        "run_orthofinder",
        artifacts=(),
        metrics={
            "n_gene_trees": len(gene_trees),
            "has_species_tree": bool(sp_tree),
            "results_dir": str(results),
        },
        findings=(_finding("n_gene_trees", value=len(gene_trees)),),
        flags=("executed",),
        summary_text=f"OrthoFinder3 produced {len(gene_trees)} gene trees.",
        provenance=prov,
    )


def run_coevolution(
    proteome_dir: str | Path,
    *,
    output_dir: str | Path,
    method: str = "ercnet",
    threads: int = 8,
    executor: Callable[[list[str]], Any] | None = None,
    results_dir: str | Path | None = None,
    transform: str = "sqrt",
    fdr_threshold: float = 0.05,
    r_threshold: float = 0.5,
) -> OrganelleResult:
    """End-to-end coevolution: OrthoFinder3 → ERC → network.

    Two methods:
      method="ercnet" — ERCnet pipeline (BXB species-edge branch space + FDR)
      method="erc"    — ERC 2.0 (master-tree edge-space + Fisher + permutation)

    With ``executor=None`` (plan-only), stops after emitting the planned
    OrthoFinder argv.
    """
    if method not in ("ercnet", "erc"):
        return _failed(
            "run_coevolution",
            summary_text=f"method={method!r} not supported; use 'ercnet' or 'erc'.",
            anomalies=["method_not_implemented"],
        )
    out = Path(output_dir)
    if executor is not None or results_dir is not None:
        out.mkdir(parents=True, exist_ok=True)
    # 1. OrthoFinder upstream (or reuse an existing Results dir)
    if results_dir is None:
        of = run_orthofinder(proteome_dir, output_dir=out, threads=threads, executor=executor)
        if executor is None:
            return of  # plan-only: stop after planning OrthoFinder
        if of.status != "ok":
            return of
        results = _locate_results(out)
    else:
        results = Path(results_dir)
    if results is None or not results.exists():
        return _failed(
            "run_coevolution",
            summary_text="No OrthoFinder results to analyze.",
            anomalies=["no_results_dir"],
        )
    # 2. parse
    sp_tree, gene_trees = _parse_orthofinder_outputs(results)
    if len(gene_trees) < 2:
        return _failed(
            "run_coevolution",
            summary_text=f"Need >=2 gene trees, got {len(gene_trees)}.",
            anomalies=["too_few_gene_trees"],
        )

    # --- ERC 2.0 branch (method="erc") ---
    if method == "erc":
        from .erc_engine import run_erc

        erc_result = run_erc(
            gene_trees,
            sp_tree,
            transform=transform,
            min_overlap=5,
        )
        metrics = dict(erc_result.metrics)
        metrics["method"] = "erc"
        metrics["n_gene_trees"] = len(gene_trees)
        return erc_result.evolve(
            operation_id=f"{SUITE}.run_coevolution",
            metrics=metrics,
            provenance=_provenance(
                "run_coevolution",
                method="erc",
                parameters={"method": "erc", "transform": transform},
            ),
        )

    # --- ERCnet branch (method="ercnet") ---
    # 3. reconcile (collect dup/spec totals, ERCnet convention Species__GENE)
    def species_of(leaf: str) -> str:
        return leaf.split("__")[0]

    dup_total = spec_total = 0
    if sp_tree:
        for gt in gene_trees.values():
            rec = reconcile_trees(gt, sp_tree)
            if rec.status == "ok":
                dup_total += rec.metrics.get("duplications", 0)
                spec_total += rec.metrics.get("speciations", 0)
    # 4. branch matrix: ERCnet branch-by-branch species-edge space
    if not sp_tree:
        return _failed(
            "run_coevolution",
            summary_text="ERCnet BXB requires a species tree.",
            anomalies=["species_tree_required"],
        )
    bxb = extract_branch_lengths(
        gene_trees,
        method="BXB",
        species_tree=sp_tree,
        bxb_normalization="gene_sum",
    )
    if bxb.status != "ok":
        return bxb.evolve(
            operation_id=f"{SUITE}.run_coevolution",
            provenance=_provenance("run_coevolution", method="ercnet_bxb"),
        )
    branch_order = [str(edge) for edge in bxb.metrics.get("branch_order", ())]
    bl = _thaw_branch_lengths(
        bxb.metrics.get("branch_lengths", {}), bxb.metrics.get("gene_order", ())
    )
    n_species = int(bxb.metrics.get("n_species", 0))
    if n_species < 2:
        return _failed(
            "run_coevolution",
            summary_text=f"Need >=2 species, got {n_species}.",
            anomalies=["too_few_species"],
        )
    # 5. ERC + network
    pairs = _erc_pairs(bl, transform=transform, fdr_threshold=fdr_threshold)
    nodes, edges, communities = _network_from_pairs(pairs, r_threshold=r_threshold, use_fdr=True)
    significant = [p for p in pairs if p.get("significant")]
    # 6. artifacts
    (out / "branch_lengths.tsv").write_text(
        "gene\t"
        + "\t".join(branch_order)
        + "\n"
        + "\n".join(
            g
            + "\t"
            + "\t".join(
                "nan"
                if row.get(edge, float("nan")) != row.get(edge, float("nan"))
                else f"{row.get(edge, 0.0):.6f}"
                for edge in branch_order
            )
            for g, row in bl.items()
        )
        + "\n"
    )
    (out / "erc_pairs.tsv").write_text(
        "gene_a\tgene_b\tr\trho\tpval\tfdr_pval\tsignificant\n"
        + "\n".join(
            f"{p['gene_a']}\t{p['gene_b']}\t{p['r']}\t{p.get('rho', '')}\t"
            f"{p['pval']:.6g}\t{p.get('fdr_pval', '')}\t{p.get('significant', False)}"
            for p in pairs
        )
        + "\n"
    )
    (out / "coevolution_network.tsv").write_text(
        "gene_a\tgene_b\tr\n" + "\n".join(f"{a}\t{b}\t{r}" for a, b, r in edges) + "\n"
    )
    artifacts = tuple(
        ArtifactRef.from_path(
            out / name,
            kind=kind,
            format="tsv",
            media_type="text/tab-separated-values",
        )
        for name, kind in (
            ("branch_lengths.tsv", "coevolution_branch_lengths"),
            ("erc_pairs.tsv", "coevolution_erc_pairs"),
            ("coevolution_network.tsv", "coevolution_network"),
        )
    )
    return _ok(
        "run_coevolution",
        artifacts=artifacts,
        metrics={
            "n_gene_trees": len(gene_trees),
            "n_species": n_species,
            "n_branches": len(branch_order),
            "branch_method": "BXB",
            "branch_normalization": bxb.metrics.get("normalization", "gene_sum"),
            "duplications": dup_total,
            "speciations": spec_total,
            "pairs_tested": len(pairs),
            "significant_pairs": len(significant),
            "network_nodes": len(nodes),
            "network_edges": len(edges),
            "communities": len(communities),
        },
        findings=(
            _finding("significant_pairs", value=len(significant)),
            _finding("communities", value=len(communities)),
        ),
        flags=("ercnet_chain_complete",) + (("covariation_detected",) if significant else ()),
        summary_text=(
            f"ERCnet BXB chain: {len(gene_trees)} OGs x {n_species} species; "
            f"{len(significant)}/{len(pairs)} significant pairs; "
            f"{len(communities)} communities."
        ),
        provenance=_provenance(
            "run_coevolution",
            method="ercnet_bxb",
            parameters={
                "method": "ercnet",
                "transform": transform,
                "fdr_threshold": fdr_threshold,
                "r_threshold": r_threshold,
                "threads": threads,
                "proteome_dir": str(proteome_dir),
                "results_dir": str(results_dir) if results_dir else None,
            },
        ),
    )
