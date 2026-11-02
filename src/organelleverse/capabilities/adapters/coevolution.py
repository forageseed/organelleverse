"""Bindings for the coevolution scientific stages.

Public numeric stages use None/JSON null for missing paths, lengths and
untestable correlations. Fisher estimates include an explicit status and a
nullable z, so perfect correlations never become zero or non-finite JSON.
Tree context is carried explicitly by the residual/correlation bindings;
full run_erc supplies that context when fitting and selecting independent
common-species edges. LCA and DLCpar remain distinct explicit reconciliation
models. Frozen result mappings and sequences are accepted by native chaining.
The operation's scientific contract still determines whether input data are
sufficient; successful JSON transport alone is not scientific acceptance.
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
        parameters=(
            ParameterOverride(name="correlations", codec=ParameterCodec.JSON),
            ParameterOverride(name='r_threshold', codec=ParameterCodec.JSON),
            ParameterOverride(name='use_fdr', codec=ParameterCodec.JSON),
        ),
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
            ParameterOverride(name='default', codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="compartment_map",
        side_effects=(),
    ),
    "coevolution.correlation_matrix": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="residuals", codec=ParameterCodec.JSON),
            ParameterOverride(name='method', codec=ParameterCodec.JSON),
            ParameterOverride(name='min_overlap', codec=ParameterCodec.JSON),
            ParameterOverride(name="gene_trees", codec=ParameterCodec.JSON),
            ParameterOverride(name="species_tree_newick", codec=ParameterCodec.JSON),
            ParameterOverride(name="winsorize", codec=ParameterCodec.JSON),
        ),
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
            ParameterOverride(name='n_perms', codec=ParameterCodec.JSON),
            ParameterOverride(name='seed', codec=ParameterCodec.JSON),
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
        parameters=(
            ParameterOverride(name="path_matrix", codec=ParameterCodec.JSON),
            ParameterOverride(name='transform', codec=ParameterCodec.JSON),
            ParameterOverride(name='trim_prop', codec=ParameterCodec.JSON),
            ParameterOverride(name="gene_trees", codec=ParameterCodec.JSON),
            ParameterOverride(name="species_tree_newick", codec=ParameterCodec.JSON),
            ParameterOverride(name="variance_weight", codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.JSON_METRIC,
        result_key="residual_matrix",
        side_effects=(),
    ),
    "coevolution.extract_branch_lengths": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gene_trees", codec=ParameterCodec.JSON),
            ParameterOverride(name='method', codec=ParameterCodec.JSON),
            ParameterOverride(name='species_tree', codec=ParameterCodec.JSON),
            ParameterOverride(name='species_map', codec=ParameterCodec.JSON),
            ParameterOverride(name='reconciliation', codec=ParameterCodec.JSON),
            ParameterOverride(name='dlcpar_dir', codec=ParameterCodec.DIRECTORY, path_role="input"),
            ParameterOverride(name='bxb_normalization', codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "coevolution.rate_covariation": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="branch_lengths", codec=ParameterCodec.JSON),
            ParameterOverride(name='method', codec=ParameterCodec.JSON),
            ParameterOverride(name='transform', codec=ParameterCodec.JSON),
            ParameterOverride(name='variance_weight', codec=ParameterCodec.JSON),
            ParameterOverride(name='min_overlap', codec=ParameterCodec.JSON),
            ParameterOverride(name='fdr_threshold', codec=ParameterCodec.JSON),
            ParameterOverride(name='restrict', codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "coevolution.reconcile_trees": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gene_tree_newick", codec=ParameterCodec.JSON),
            ParameterOverride(name="species_tree_newick", codec=ParameterCodec.JSON),
            ParameterOverride(name='species_map', codec=ParameterCodec.JSON),
        ),
        result_codec=ResultCodec.CANONICAL,
        result_key=None,
        side_effects=(),
    ),
    "coevolution.run_erc": NamedParameterOverride(
        parameters=(
            ParameterOverride(name="gene_trees", codec=ParameterCodec.JSON),
            ParameterOverride(name="species_tree_newick", codec=ParameterCodec.JSON),
            ParameterOverride(name='transform', codec=ParameterCodec.JSON),
            ParameterOverride(name='method', codec=ParameterCodec.JSON),
            ParameterOverride(name='min_overlap', codec=ParameterCodec.JSON),
            ParameterOverride(name='gene_sets', codec=ParameterCodec.JSON),
            ParameterOverride(name='n_perms', codec=ParameterCodec.JSON),
            ParameterOverride(name="fdr_threshold", codec=ParameterCodec.JSON),
            ParameterOverride(name="seed", codec=ParameterCodec.JSON),
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
            ParameterOverride(name='method', codec=ParameterCodec.JSON),
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
            ParameterOverride(name='threads', codec=ParameterCodec.JSON),
            ParameterOverride(name='msa', codec=ParameterCodec.JSON),
            ParameterOverride(name='search', codec=ParameterCodec.JSON),
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
            ParameterOverride(name='method', codec=ParameterCodec.JSON),
            ParameterOverride(name='threads', codec=ParameterCodec.JSON),
            ParameterOverride(name='results_dir', codec=ParameterCodec.DIRECTORY, path_role="input"),
            ParameterOverride(name='transform', codec=ParameterCodec.JSON),
            ParameterOverride(name='fdr_threshold', codec=ParameterCodec.JSON),
            ParameterOverride(name='r_threshold', codec=ParameterCodec.JSON),
            ParameterOverride(name="reconciliation", codec=ParameterCodec.JSON),
            ParameterOverride(name="dlcpar_dir", codec=ParameterCodec.DIRECTORY, path_role="input"),
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
