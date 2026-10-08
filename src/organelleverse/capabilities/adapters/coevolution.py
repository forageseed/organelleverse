"""Adapter decisions for the ``coevolution`` domain's restored capabilities.

**A caveat carried across most of this domain, in the same spirit as the
``diversity`` adapter's NaN caveat.** Every ERC2/BXB numeric core in this
domain (``project_paths``, ``residual_matrix``, ``correlation_matrix``,
``perm_test``, ``bxb_lengths_lca``/``.bxb_lengths_dlcpar``,
``normalize_edge_mean``/``.normalize_gene_sum``, ``filter_outlier_tree``)
can genuinely produce ``float("nan")`` in its output for missing/discordant
data (documented in each function's own docstring: "Missing edges ...
NaN"). ``json_metric``'s runtime check (``codecs._is_finite_json``)
requires every float to satisfy ``math.isfinite`` and correctly rejects a
NaN-bearing result with ``capability.result_codec_invalid`` rather than
silently coercing it. This is not a wrong codec choice - ``json_metric`` is
still the honest binding for each function's general (non-degenerate)
case - it is a real, data-dependent gap already present in the restored
implementations, left exactly as found. Every real-invocation test below
therefore uses concordant, complete inputs designed to avoid NaN, and does
not claim the NaN case is handled.

Ledger records for this domain (``docs/operations/restored-capabilities.toml``):

* ``coevolution.species_tree_edges`` - resolved below: ``species_tree_newick:
  str`` (a Newick string, not a file path) gets ``json``; the ``list[str]``
  return is finite JSON, so ``json_metric`` under ``"edges"``.
* ``coevolution.bxb_lengths_lca`` - resolved below: ``gene_tree_newick``/
  ``species_tree_newick`` (``str``) and ``edges: list[str]`` all get
  ``json``; the ``dict[str, float]`` return is finite JSON (modulo the NaN
  caveat above), so ``json_metric`` under ``"edge_lengths"``.
  ``species_map`` keeps its own default.
* ``coevolution.bxb_lengths_dlcpar`` - resolved below: ``gene_tree_newick``
  (a Newick string) and ``edges: list[str]`` get ``json``;
  ``locus_recon_path``/``coal_recon_path`` get ``path`` (real DLCpar
  ``.locus.recon``/``.coal.recon`` files the implementation opens); the
  ``dict[str, float]`` return is finite JSON (modulo the NaN caveat), so
  ``json_metric`` under ``"edge_lengths"``. ``side_effects =
  ["read_files"]``.
* ``coevolution.filter_outlier_tree`` - resolved below: ``edge_lengths:
  dict[str, float]`` (fully parameterized, unlike the bare-``dict`` gap
  documented in ``phylogeny``) gets ``json``; the ``dict[str, float] |
  None`` return is finite JSON either way (``None`` is a valid finite-JSON
  value), so ``json_metric`` under ``"filtered_edges"``. ``max_ratio``
  keeps its own default.
* ``coevolution.normalize_gene_sum`` / ``.normalize_edge_mean`` - resolved
  below: ``matrix: dict[str, dict[str, float]]`` (nested-but-fully-
  parameterized, so JSON-safe even though a *bare* ``dict`` inside a
  container is not) gets ``json``; both ``dict[str, dict[str, float]]``
  returns are finite JSON (modulo the NaN caveat - both explicitly emit a
  NaN sentinel for missing/zero-mean edges), so ``json_metric`` under
  ``"normalized_matrix"`` for both.
* ``coevolution.compartment_map_from_prefix`` - resolved below: ``genes:
  list[str]`` and ``prefix_rules: dict[str, str]`` both get ``json``; the
  ``dict[str, str]`` return is finite JSON, so ``json_metric`` under
  ``"compartment_map"``. ``default`` keeps its own default.
* ``coevolution.correlation_matrix`` - resolved below: ``residuals: dict[str,
  list[float]]`` gets ``json``; the ``(cor_matrix, count_matrix)`` 2-tuple
  return is finite JSON, so ``json_metric`` under ``"correlation_matrix"``.
  ``method``/``min_overlap`` keep their own defaults.
* ``coevolution.fisher`` - resolved below: ``cor: dict[str, dict[str,
  float]]`` and ``count: dict[str, dict[str, int]]`` both get ``json``; the
  ``dict[str, dict[str, float]]`` return is finite JSON (the implementation
  explicitly falls back to ``0.0``, never NaN, for insufficient overlap or
  out-of-range r), so ``json_metric`` under ``"fisher_z"``.
* ``coevolution.perm_test`` - resolved below: ``gene_set: list[str]`` and
  ``cor: dict[str, dict[str, float]]`` both get ``json``; the ``dict[str,
  Any]`` return (really ``{"observed", "p_value", "null_mean"}``, all
  scalars) is finite JSON at every real call with >=2 overlapping genes
  (fewer degrades to an explicit NaN "observed" - the caveat above), so
  ``json_metric`` under ``"perm_test"``. ``n_perms``/``seed`` keep their
  own defaults.
* ``coevolution.project_paths`` - resolved below: ``gene_trees: dict[str,
  str]`` and ``species_tree_newick: str`` both get ``json``; the
  ``dict[str, list[float]]`` return is finite JSON (modulo the NaN
  caveat), so ``json_metric`` under ``"path_matrix"``.
* ``coevolution.residual_matrix`` - resolved below: ``path_matrix: dict[str,
  list[float]]`` gets ``json``; the ``dict[str, list[float]]`` return is
  finite JSON (modulo the NaN caveat), so ``json_metric`` under
  ``"residual_matrix"``. ``transform``/``trim_prop`` keep their own
  defaults.
* ``coevolution.extract_branch_lengths`` - ``canonical`` result_shape,
  first parameter ``gene_trees: dict[str, str]`` (not a bare
  ``OrganelleGenome``), so it does not match ``canonical_core``. Resolved
  below: ``gene_trees`` gets ``json``; the implementation already returns a
  real ``OrganelleResult`` it built itself, so ``canonical``. Every other
  parameter (including ``dlcpar_dir: str | Path | None``, a directory, but
  optional with a ``None`` default and only consulted when
  ``reconciliation="dlcpar"``) keeps its own default.
* ``coevolution.rate_covariation`` - resolved below: ``branch_lengths:
  dict[str, list[float] | dict[str, float]]`` is a mapping whose value type
  is itself a ``Union`` of two JSON-safe shapes - ``_is_supported_json_annotation``'s
  ``Union`` branch requires every arm to be JSON-safe, and both
  (``list[float]``, ``dict[str, float]``) are - so it gets ``json``;
  ``canonical`` result (a real ``OrganelleResult``). ``compartment_map:
  dict[str, Any] | None`` and ``restrict: tuple[str, str] | None`` keep
  their own defaults.
* ``coevolution.reconcile_trees`` - resolved below: ``gene_tree_newick``/
  ``species_tree_newick`` (both ``str`` Newick text, not files) get
  ``json``; ``canonical`` result. ``species_map`` keeps its own default.
* ``coevolution.run_erc`` - resolved below: ``gene_trees: dict[str, str]``
  and ``species_tree_newick: str`` get ``json``; ``canonical`` result -
  composes the typed cores above with no I/O of its own. Every other
  parameter (including ``gene_sets: dict[str, list[str]] | None``) keeps
  its own default.
* ``coevolution.coevolution_network`` - **excluded**: required
  ``correlations: list[dict]`` - the bare-``dict``-in-``list`` gap first
  documented in the ``phylogeny`` adapter, since Capability Plan 03 Task 4
  rejected by the generator's own ``_looks_json_safe`` at generation time
  too (it always failed at the real runtime binder). No entry here.
* ``coevolution.parse_newick`` - **excluded**: returns a custom
  ``TreeNode`` instance - not JSON-safe, not an ``OrganelleResult``, not an
  artifact shape. No entry here.
* ``coevolution.phylogenomics`` - resolved below: required
  ``orthofinder_dir: str | Path`` gets ``directory`` with ``path_role =
  "input"`` - a pre-existing OrthoFinder results tree the (planned)
  ``Phylogenomics.py -i <orthofinder_dir>`` step would read;
  ``str(orthofinder_dir)`` is only ever embedded into the planned ``argv``
  here, never opened directly, the same "input, never itself read by this
  function" shape ``population.call_variants``'s ``bam_dir`` already
  established. ``method`` keeps its own default; ``executor`` is never
  Agent-exposed, so every real call takes the planned branch, which is
  still a real, self-constructed ``OrganelleResult``, so ``canonical``.
  ``side_effects = ()`` - no genuine filesystem or subprocess activity on
  the only reachable real path.
* ``coevolution.run_orthofinder`` - resolved below: required
  ``proteome_dir: str | Path`` gets ``directory`` with ``path_role =
  "input"`` (a pre-existing tree of per-species proteome FASTAs OrthoFinder
  would read). The public service allocates an output location inside the
  managed run root; it is not an Agent parameter. ``threads``/``msa``/
  ``search`` keep their defaults. Without a Python ``executor``, the call
  returns a plan without creating a directory. ``canonical`` result.
* ``coevolution.run_coevolution`` - the same managed-output contract, with
  ``proteome_dir`` as the sole required Agent path. ``method``/``threads``/
  ``results_dir``/``transform``/``fdr_threshold``/``r_threshold`` retain
  their defaults. Agent calls return the nested OrthoFinder plan without
  creating an empty run directory; a Python executor may write managed
  intermediates. ``canonical`` result.
* ``coevolution.erc_master_paths``, ``.erc_residual_matrix``,
  ``.erc_correlation_matrix``, ``.fisher_transform``, ``.perm_test_matrix``
  - **excluded**: all five are "back-compat alias" wrappers
  (``erc_engine.py:496-519``) with *no parameter annotations at all*
  (``def erc_master_paths(gene_trees, species_tree_newick, **kw):``, etc.) -
  the same shape already documented for
  ``pangenome.compute_gene_pav``/``trans_splicing.compute_trans_splicing``.
  No entry here for any of the five.
 **Zero-out rulings (owner mandate 2026-08-14, final):** ``erc_correlation_matrix``/``erc_master_paths``/``erc_residual_matrix``/
 ``fisher_transform``/``perm_test_matrix`` are INTERNAL (back-compat alias
 wrappers of published primitives - duplicate agent surface adds nothing);
 ``parse_newick`` stays REJECT (untyped parser helper).
"""

from __future__ import annotations

from dataclasses import dataclass

from organelleverse.operations.spec import ParameterCodec, ResultCodec, SideEffect


@dataclass(frozen=True)
class ParameterOverride:
    """One named, Agent-facing parameter the generator could not derive alone."""

    name: str
    codec: ParameterCodec
    path_role: str | None = None


@dataclass(frozen=True)
class NamedParameterOverride:
    """The full ``named_parameters`` binding plan for one restored capability."""

    parameters: tuple[ParameterOverride, ...]
    result_codec: ResultCodec
    result_key: str | None
    side_effects: tuple[SideEffect, ...]


OVERRIDES: dict[str, NamedParameterOverride] = {
    # Zero-out batch: the row shapes are all JSON primitives (gene names,
    # r floats, significant flags), verified against the consumption loop.
    "coevolution.coevolution_network": NamedParameterOverride(
        parameters=(ParameterOverride(name="correlations", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "coevolution.species_tree_edges": NamedParameterOverride(
        parameters=(ParameterOverride(name="species_tree_newick", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="edges",
        side_effects=(),
    ),
    "coevolution.bxb_lengths_lca": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gene_tree_newick", codec=ParameterCodec.JSON),
            ParameterOverride(name="species_tree_newick", codec=ParameterCodec.JSON),
            ParameterOverride(name="edges", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="edge_lengths",
        side_effects=(),
    ),
    "coevolution.bxb_lengths_dlcpar": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gene_tree_newick", codec=ParameterCodec.JSON),
            ParameterOverride(name="locus_recon_path", codec=ParameterCodec.PATH),
            ParameterOverride(name="coal_recon_path", codec=ParameterCodec.PATH),
            ParameterOverride(name="edges", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="edge_lengths",
        side_effects=(SideEffect.READ_FILES,),
    ),
    "coevolution.filter_outlier_tree": NamedParameterOverride(
        parameters=(ParameterOverride(name="edge_lengths", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="filtered_edges",
        side_effects=(),
    ),
    "coevolution.normalize_gene_sum": NamedParameterOverride(
        parameters=(ParameterOverride(name="matrix", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="normalized_matrix",
        side_effects=(),
    ),
    "coevolution.normalize_edge_mean": NamedParameterOverride(
        parameters=(ParameterOverride(name="matrix", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="normalized_matrix",
        side_effects=(),
    ),
    "coevolution.compartment_map_from_prefix": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="genes", codec=ParameterCodec.JSON),
            ParameterOverride(name="prefix_rules", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="compartment_map",
        side_effects=(),
    ),
    "coevolution.correlation_matrix": NamedParameterOverride(
        parameters=(ParameterOverride(name="residuals", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="correlation_matrix",
        side_effects=(),
    ),
    "coevolution.fisher": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="cor", codec=ParameterCodec.JSON),
            ParameterOverride(name="count", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="fisher_z",
        side_effects=(),
    ),
    "coevolution.perm_test": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gene_set", codec=ParameterCodec.JSON),
            ParameterOverride(name="cor", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="perm_test",
        side_effects=(),
    ),
    "coevolution.project_paths": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gene_trees", codec=ParameterCodec.JSON),
            ParameterOverride(name="species_tree_newick", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="path_matrix",
        side_effects=(),
    ),
    "coevolution.residual_matrix": NamedParameterOverride(
        parameters=(ParameterOverride(name="path_matrix", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="residual_matrix",
        side_effects=(),
    ),
    "coevolution.extract_branch_lengths": NamedParameterOverride(
        parameters=(ParameterOverride(name="gene_trees", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "coevolution.rate_covariation": NamedParameterOverride(
        parameters=(ParameterOverride(name="branch_lengths", codec=ParameterCodec.JSON),),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "coevolution.reconcile_trees": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gene_tree_newick", codec=ParameterCodec.JSON),
            ParameterOverride(name="species_tree_newick", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "coevolution.run_erc": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gene_trees", codec=ParameterCodec.JSON),
            ParameterOverride(name="species_tree_newick", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "coevolution.phylogenomics": NamedParameterOverride(
        parameters=(
            ParameterOverride(
                name="orthofinder_dir", codec=ParameterCodec.DIRECTORY, path_role="input"
            ),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "coevolution.run_orthofinder": NamedParameterOverride(
        parameters=(
            ParameterOverride(
                name="proteome_dir", codec=ParameterCodec.DIRECTORY, path_role="input"
            ),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "coevolution.run_coevolution": NamedParameterOverride(
        parameters=(
            ParameterOverride(
                name="proteome_dir", codec=ParameterCodec.DIRECTORY, path_role="input"
            ),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(SideEffect.WRITE_FILES,),
    ),
}
"""``coevolution.coevolution_network``, ``.parse_newick``,
``.erc_master_paths``, ``.erc_residual_matrix``, ``.erc_correlation_matrix``,
``.fisher_transform``, and ``.perm_test_matrix`` are deliberately absent -
see the module docstring for why each fails closed honestly.
"""

__all__ = ["OVERRIDES", "NamedParameterOverride", "ParameterOverride"]
