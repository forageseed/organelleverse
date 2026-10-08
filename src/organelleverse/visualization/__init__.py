"""Visualization suite: genome maps, heatmaps, trees, scatter plots.

Genome maps are drawn by the self-contained OGDraw matplotlib port
(:func:`plot_ogdraw_map`) or, when installed, by the higher-quality external
``gbdraw`` backend (:func:`plot_genome_map`). Other figures use seaborn,
matplotlib, toytree, and Bio.Phylo.
"""

from __future__ import annotations

from .gbdraw import plot_collinearity, plot_gbdraw, plot_gene_structure
from .gfa_graph import plot_gfa_graph, read_gfa_graph
from .ogdraw import (
    GENE_COLORS,
    classify_gene,
    draw_mito_map,
    parse_genbank,
    plot_ogdraw_map,
    spread_labels,
)
from .pan_circular import plot_pan_circular
from .plots import plot_genome_map, plot_heatmap, plot_scatter, plot_tree
from .structure_maps import plot_structure_map
from .suite_plots import (
    ideogram,
    nuclear_transfer_ideogram,
    plot_erc_distribution,
    plot_erc_group_ridges,
    plot_erc_network,
    plot_erc_pair_scatter,
    plot_erc_significance,
    plot_genome_identity,
    plot_localization_bars,
    plot_matrix_heatmap,
    plot_network,
    plot_qc_dashboard,
    plot_rna_editing_summary,
    plot_rscu_usage,
    plot_selection_summary,
    plot_splicing_schematic,
    plot_synteny_matrix,
    plot_track_density,
    plot_transfer_schematic,
    plot_transfer_tracks,
    save_plot,
    summarize_synteny_matrix,
    write_erc_visualization_report,
    write_ideogram,
    write_nuclear_transfer_ideogram,
    write_visualization_preview_report,
)

__all__ = [
    "GENE_COLORS",
    "classify_gene",
    "collinearity",
    "draw_mito_map",
    "erc_distribution",
    "erc_group_ridges",
    "erc_network",
    "erc_pair_scatter",
    "erc_significance",
    "gene_structure",
    "genome_map",
    "heatmap",
    "ideogram",
    "localization_bars",
    "matrix_heatmap",
    "network",
    "nuclear_transfer_ideogram",
    "object",
    "ogdraw_map",
    "parse_genbank",
    "plot_collinearity",
    "plot_erc_distribution",
    "plot_erc_group_ridges",
    "plot_erc_network",
    "plot_erc_pair_scatter",
    "plot_erc_significance",
    "plot_gbdraw",
    "plot_gene_structure",
    "plot_genome_identity",
    "plot_genome_map",
    "plot_gfa_graph",
    "plot_heatmap",
    "plot_localization_bars",
    "plot_matrix_heatmap",
    "plot_network",
    "plot_ogdraw_map",
    "plot_pan_circular",
    "plot_qc_dashboard",
    "plot_rna_editing_summary",
    "plot_rscu_usage",
    "plot_scatter",
    "plot_selection_summary",
    "plot_splicing_schematic",
    "plot_structure_map",
    "plot_synteny_matrix",
    "plot_track_density",
    "plot_transfer_schematic",
    "plot_transfer_tracks",
    "plot_tree",
    "qc_dashboard",
    "read_gfa_graph",
    "rna_editing_summary",
    "rscu_usage",
    "save_plot",
    "scatter",
    "selection_summary",
    "splicing_schematic",
    "spread_labels",
    "summarize_synteny_matrix",
    "synteny_matrix",
    "track_density",
    "transfer_schematic",
    "transfer_tracks",
    "tree",
    "write_erc_visualization_report",
    "write_ideogram",
    "write_nuclear_transfer_ideogram",
    "write_visualization_preview_report",
]

# Short human-facing spellings. ``plot_`` is redundant inside a module already
# named visualization, so the unprefixed name is the one people type and the one
# operation ids use; the prefixed spelling stays bound to the same object so no
# existing call breaks. Three names keep their prefix because dropping it would
# collide with a submodule of the same name.
collinearity = plot_collinearity
erc_distribution = plot_erc_distribution
erc_group_ridges = plot_erc_group_ridges
erc_network = plot_erc_network
erc_pair_scatter = plot_erc_pair_scatter
erc_significance = plot_erc_significance
gene_structure = plot_gene_structure
genome_identity = plot_genome_identity
genome_map = plot_genome_map
heatmap = plot_heatmap
localization_bars = plot_localization_bars
matrix_heatmap = plot_matrix_heatmap
network = plot_network
object = plot_object
ogdraw_map = plot_ogdraw_map
qc_dashboard = plot_qc_dashboard
rna_editing_summary = plot_rna_editing_summary
rscu_usage = plot_rscu_usage
scatter = plot_scatter
selection_summary = plot_selection_summary
splicing_schematic = plot_splicing_schematic
synteny_matrix = plot_synteny_matrix
track_density = plot_track_density
transfer_schematic = plot_transfer_schematic
transfer_tracks = plot_transfer_tracks
tree = plot_tree
