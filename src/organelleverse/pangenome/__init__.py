"""Pangenome suite with a managed, destination-free graph builder."""

from __future__ import annotations

from .analysis_service import analyze_graph
from .graph_formats import convert_graph
from .graph_operations import convert_graph_result, extract_subgraph
from .install import PAN_BACKEND_INFO, check_all_backends, check_backend, install_hint
from .pangenome import (
    classify_sequences,
    gene_pav,
    pan_repeats,
    write_pav,
)
from .pangenome_core import compute_gene_pav
from .project import PangenomeProject, PangenomeSample, PangenomeStagingManifest
from .service import build_graph
from .tuning_service import recommend_parameters

managed_build_graph = build_graph

__all__ = [
    "PAN_BACKEND_INFO",
    "PangenomeProject",
    "PangenomeSample",
    "PangenomeStagingManifest",
    "analyze_graph",
    "build_graph",
    "check_all_backends",
    "check_backend",
    "classify_sequences",
    "compute_gene_pav",
    "convert_graph",
    "convert_graph_result",
    "extract_subgraph",
    "gene_pav",
    "install_hint",
    "managed_build_graph",
    "pan_repeats",
    "recommend_parameters",
    "write_pav",
]
