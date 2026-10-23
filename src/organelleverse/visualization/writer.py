"""Materialization of canonical ``visualization.*`` Results.

This is the suite's single publication boundary. ``ov.write(plot, output)``
dispatches here; the plot's captured renderer runs, every file it writes is
declared as an :class:`~organelleverse.core.artifacts.ArtifactRef` (role,
path, SHA-256, media type), and a released Result carrying those artifacts is
returned. No public plot call writes a file before this point.
"""

from __future__ import annotations

from pathlib import Path

from ..core.errors import OrganelleInputError
from ..core.frozen import thaw_json
from ..core.result import OrganelleResult
from .plot_object import OrganellePlot, figure_artifacts
from .popgen_plots import POPGEN_OPERATIONS

_DATA_PLOT_KINDS = frozenset({"ideogram", "nuclear_transfer_ideogram"})
_STABILITY_PLOTS = frozenset(
    {
        "visualization.plot_depth_profile",
        "visualization.plot_variant_frequencies",
        "visualization.plot_variant_site_heatmap",
    }
)


def materialize_result(result: OrganelleResult, output: str | Path) -> OrganelleResult:
    """Render one canonical visualization Result and declare its artifacts."""
    if not result.operation_id.startswith("visualization."):
        raise OrganelleInputError(
            code="visualization.unsupported_operation",
            message="the visualization writer accepts visualization.* Results",
            details={"operation_id": result.operation_id},
        )
    result.raise_for_failure()
    destination = Path(output)

    if isinstance(result, OrganellePlot) and result.renderer is not None:
        return result.materialize(destination)

    plot_kind = str(result.metrics.get("plot_kind", ""))
    if result.operation_id == "visualization.plot_assembly_graph":
        from .assembly_graph import _restore_assembly_graph_plot

        return _restore_assembly_graph_plot(result).materialize(destination)

    if result.operation_id == "visualization.plot_pan_circular":
        from .pan_circular import _restore_pan_circular_plot

        return _restore_pan_circular_plot(result).materialize(destination)

    if result.operation_id in POPGEN_OPERATIONS:
        from .popgen_plots import _restore_popgen_plot

        return _restore_popgen_plot(result).materialize(destination)

    if result.operation_id == "visualization.plot_structure_map":
        from .structure_maps import _restore_structure_plot

        return _restore_structure_plot(result).materialize(destination)
    if result.operation_id == "visualization.plot_transfer_circos":
        from .circos import _restore_transfer_circos

        return _restore_transfer_circos(result).materialize(destination)

    if result.operation_id in _STABILITY_PLOTS:
        from .stability_plots import _restore_stability_plot

        return _restore_stability_plot(result).materialize(destination)

    from .tree_plots import TREE_PLOT_OPERATIONS, restore_tree_plot

    if result.operation_id in TREE_PLOT_OPERATIONS:
        return restore_tree_plot(result).materialize(destination)

    from .paper_figures import PAPER_FIGURE_OPERATIONS, restore_paper_figure

    if result.operation_id in PAPER_FIGURE_OPERATIONS:
        return restore_paper_figure(result).materialize(destination)

    if plot_kind in _DATA_PLOT_KINDS:
        from .suite_plots import save_plot

        written = save_plot(result, destination)
        return _released(result, (written,))

    raise OrganelleInputError(
        code="visualization.unrenderable_result",
        message=(
            "ov.write accepts a plot with a captured renderer or a prepared "
            "ideogram/nuclear_transfer_ideogram Result"
        ),
        details={"operation_id": result.operation_id, "plot_kind": plot_kind},
    )


def _released(source: OrganelleResult, paths: tuple[Path, ...]) -> OrganelleResult:
    """Return ``source`` with every written file declared as an artifact."""
    return OrganelleResult(
        operation_id=source.operation_id,
        operation_version=source.operation_version,
        scope=source.scope,
        status=source.status,
        summary_text=source.summary_text,
        metrics=thaw_json(source.metrics),
        findings=source.findings,
        flags=source.flags,
        artifacts=figure_artifacts(paths),
        provenance=source.provenance,
        suggested_operations=source.suggested_operations,
    )


__all__ = ["materialize_result"]
