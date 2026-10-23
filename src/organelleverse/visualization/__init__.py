"""Visualization suite: genome maps, heatmaps, trees, scatter plots.

Genome maps are drawn by the self-contained OGDraw matplotlib port
(:func:`plot_ogdraw_map`) or, when installed, by the higher-quality external
``gbdraw`` backend (:func:`plot_genome_map`). Other figures use seaborn,
matplotlib, toytree, and Bio.Phylo.
"""

from __future__ import annotations

from . import plot_object as plot_object

from .assembly_graph import plot_assembly_graph
from .circos import plot_transfer_circos, transfer_links
from .compose import compose_figure
from ..inputs import make_reader

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
from .paper_figures import (
    plot_gene_presence,
    plot_ir_boundaries,
    plot_nucleotide_diversity,
    plot_repeat_summary,
)
from .plots import plot_genome_map, plot_heatmap, plot_scatter, plot_tree
from .tree_plots import plot_phylo_tree, plot_tree_conflict
from .popgen_plots import (
    plot_fst_pi_stack,
    plot_fst_scan,
    plot_pi_ratio,
    plot_presence_tracks,
    plot_sv_frequency,
    plot_sv_schematic,
)
from .stability_plots import (
    plot_depth_profile,
    plot_variant_frequencies,
    plot_variant_site_heatmap,
)
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
    plot_nuclear_transfer_ideogram,
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
from .transfer_plots import (
    plot_transfer_length_profile,
    plot_transfer_region_te,
    summarize_transfer_regions,
    transfer_length_bins,
)


read = make_reader("visualization", "tree", "gfa")

__all__ = [
    "read",
    "GENE_COLORS",
    "classify_gene",
    "collinearity",
    "compose_figure",
    "depth_profile",
    "draw_mito_map",
    "erc_distribution",
    "erc_group_ridges",
    "erc_network",
    "erc_pair_scatter",
    "erc_significance",
    "fst_pi_stack",
    "fst_scan",
    "gene_presence",
    "gene_structure",
    "genome_map",
    "heatmap",
    "ideogram",
    "ir_boundaries",
    "localization_bars",
    "matrix_heatmap",
    "network",
    "nuclear_transfer_ideogram",
    "nucleotide_diversity",
    "object",
    "ogdraw_map",
    "parse_genbank",
    "pi_ratio",
    "plot_assembly_graph",
    "plot_collinearity",
    "plot_depth_profile",
    "plot_erc_distribution",
    "plot_erc_group_ridges",
    "plot_erc_network",
    "plot_erc_pair_scatter",
    "plot_erc_significance",
    "plot_fst_pi_stack",
    "plot_fst_scan",
    "plot_gbdraw",
    "plot_gene_presence",
    "plot_gene_structure",
    "plot_genome_identity",
    "plot_genome_map",
    "plot_gfa_graph",
    "plot_heatmap",
    "plot_ir_boundaries",
    "plot_localization_bars",
    "plot_matrix_heatmap",
    "plot_network",
    "plot_nuclear_transfer_ideogram",
    "plot_nucleotide_diversity",
    "plot_ogdraw_map",
    "plot_pan_circular",
    "plot_phylo_tree",
    "plot_pi_ratio",
    "plot_presence_tracks",
    "plot_qc_dashboard",
    "plot_repeat_summary",
    "plot_rna_editing_summary",
    "plot_rscu_usage",
    "plot_scatter",
    "plot_selection_summary",
    "plot_splicing_schematic",
    "plot_structure_map",
    "plot_sv_frequency",
    "plot_sv_schematic",
    "plot_synteny_matrix",
    "plot_track_density",
    "plot_transfer_circos",
    "plot_transfer_length_profile",
    "plot_transfer_region_te",
    "plot_transfer_schematic",
    "plot_transfer_tracks",
    "plot_tree",
    "plot_tree_conflict",
    "plot_variant_frequencies",
    "plot_variant_site_heatmap",
    "presence_tracks",
    "qc_dashboard",
    "read_gfa_graph",
    "repeat_summary",
    "rna_editing_summary",
    "rscu_usage",
    "save_plot",
    "scatter",
    "selection_summary",
    "splicing_schematic",
    "spread_labels",
    "summarize_synteny_matrix",
    "summarize_transfer_regions",
    "sv_frequency",
    "sv_schematic",
    "synteny_matrix",
    "track_density",
    "transfer_circos",
    "transfer_length_bins",
    "transfer_links",
    "transfer_schematic",
    "transfer_tracks",
    "tree",
    "variant_frequencies",
    "variant_site_heatmap",
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
fst_pi_stack = plot_fst_pi_stack
fst_scan = plot_fst_scan
gene_structure = plot_gene_structure
genome_identity = plot_genome_identity
gene_presence = plot_gene_presence
genome_map = plot_genome_map
heatmap = plot_heatmap
ir_boundaries = plot_ir_boundaries
localization_bars = plot_localization_bars
matrix_heatmap = plot_matrix_heatmap
network = plot_network
nucleotide_diversity = plot_nucleotide_diversity
object = plot_object
ogdraw_map = plot_ogdraw_map
pi_ratio = plot_pi_ratio
presence_tracks = plot_presence_tracks
qc_dashboard = plot_qc_dashboard
repeat_summary = plot_repeat_summary
rna_editing_summary = plot_rna_editing_summary
rscu_usage = plot_rscu_usage
scatter = plot_scatter
selection_summary = plot_selection_summary
splicing_schematic = plot_splicing_schematic
sv_frequency = plot_sv_frequency
sv_schematic = plot_sv_schematic
synteny_matrix = plot_synteny_matrix
track_density = plot_track_density
transfer_circos = plot_transfer_circos
transfer_schematic = plot_transfer_schematic
transfer_tracks = plot_transfer_tracks
tree = plot_tree
phylo_tree = plot_phylo_tree
tree_conflict = plot_tree_conflict
depth_profile = plot_depth_profile
variant_frequencies = plot_variant_frequencies
variant_site_heatmap = plot_variant_site_heatmap
