"""Compose several OrganelleVerse plots into one lettered, journal-sized figure."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .plot_object import OrganellePlot, plot_result
from .style import FONT_PT, add_panel_label, publication_style, width_inches

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_GUTTER_MM = 4.0
_HEADER_MM = 4.0
_LEGEND_ROW_MM = 3.6


def compose_figure(
    panels: Sequence[OrganellePlot],
    *,
    titles: Sequence[str] | None = None,
    columns: int | None = None,
    width: str | float = "double",
    legend_columns: int = 4,
    dpi: int = 600,
) -> OrganellePlot:
    """Lay out plots as panels a, b, c … of one figure at printed size (compute-only).

    Panels share one width and one height per row, sit on a regular grid with
    equal gutters, and carry a bold lower-case letter with an optional header on
    the same baseline. Legends are merged: each distinct entry is shown once,
    under the grid. ``width`` is ``"single"`` (89 mm), ``"onehalf"`` (120 mm),
    ``"double"`` (183 mm) or millimetres; ``columns`` defaults to 2 (1 for a
    single-column figure). Only plots that can draw into a panel are accepted
    (``plot.panel`` is set: OGDraw and genome maps of one record so far).
    Render with ``ov.write(figure, output)``.
    """
    if not panels:
        raise ValueError("compose_figure needs at least one panel")
    missing = [i for i, p in enumerate(panels) if getattr(p, "panel", None) is None]
    if missing:
        names = ", ".join(str(getattr(panels[i], "operation_id", i)) for i in missing)
        raise ValueError(f"these plots cannot be drawn as a panel yet: {names}")
    if titles is not None and len(titles) != len(panels):
        raise ValueError("titles must have one entry per panel")
    total_w = width_inches(width) * 25.4
    ncols = columns or (1 if total_w < 100 else 2)
    ncols = max(1, min(ncols, len(panels)))
    nrows = math.ceil(len(panels) / ncols)
    panel_w = (total_w - _GUTTER_MM * (ncols - 1)) / ncols
    row_h = [
        panel_w * max(p.panel.aspect for p in panels[r * ncols:(r + 1) * ncols])
        for r in range(nrows)
    ]
    layout = {
        "width_mm": round(total_w, 2),
        "columns": ncols,
        "rows": nrows,
        "panel_width_mm": round(panel_w, 2),
        "row_heights_mm": [round(h, 2) for h in row_h],
        "letters": list(_LETTERS[: len(panels)]),
    }

    def _render(path: str | Path) -> Path:
        return _render_composition(
            panels, Path(path), titles=titles, ncols=ncols, panel_w=panel_w, row_h=row_h,
            total_w=total_w, legend_columns=legend_columns, dpi=dpi,
        )

    return plot_result(
        "compose_figure",
        _render,
        organelle="mixed",
        metrics={"layout": layout, "panels": [p.operation_id for p in panels]},
        summary=f"{len(panels)}-panel figure prepared.",
    )


def _render_composition(
    panels, path, *, titles, ncols, panel_w, row_h, total_w, legend_columns, dpi
) -> Path:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    nrows = len(row_h)
    legend_reserve = _LEGEND_ROW_MM * 6
    total_h = sum(row_h) + _HEADER_MM * nrows + _GUTTER_MM * (nrows - 1) + legend_reserve
    with publication_style():
        fig = plt.figure(figsize=(total_w / 25.4, total_h / 25.4))
        try:
            handles: dict[str, Any] = {}
            axes = []
            top = 0.0
            for r in range(nrows):
                top += _HEADER_MM
                for c in range(ncols):
                    index = r * ncols + c
                    if index >= len(panels):
                        break
                    x = c * (panel_w + _GUTTER_MM)
                    ax = fig.add_axes((
                        x / total_w,
                        (total_h - top - row_h[r]) / total_h,
                        panel_w / total_w,
                        row_h[r] / total_h,
                    ))
                    ax.set_anchor("N")
                    axes.append(ax)
                top += row_h[r] + _GUTTER_MM
            # one gene-name size for the figure: the most crowded panel sets it
            sizes = [
                p.panel.label_pt(ax)
                for p, ax in zip(panels, axes, strict=True)
                if getattr(p.panel, "label_pt", None) is not None
            ]
            shared_pt = min(sizes) if sizes else None
            for index, (plot, ax) in enumerate(zip(panels, axes, strict=True)):
                if shared_pt is not None and getattr(plot.panel, "label_pt", None) is not None:
                    drawn = plot.panel.draw(ax, False, label_pt=shared_pt)
                else:
                    drawn = plot.panel.draw(ax, False)
                for handle in drawn or ():
                    handles.setdefault(handle.get_label(), handle)
                add_panel_label(ax, _LETTERS[index], x_pt=-3.0, y_pt=3.0)
                header = titles[index] if titles else plot.panel.header
                if header:
                    ax.set_title(header, loc="left", fontsize=FONT_PT["body"], pad=3)
            if handles:
                fig.legend(
                    handles=list(handles.values()),
                    loc="upper left",
                    bbox_to_anchor=(0.0, 1 - (top - _GUTTER_MM + 1.0) / total_h),
                    ncol=legend_columns,
                    fontsize=FONT_PT["body"],
                    handlelength=1.0,
                    handleheight=0.8,
                    columnspacing=1.2,
                    borderaxespad=0,
                    borderpad=0,
                    frameon=False,
                )
            fig.savefig(str(path), dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        finally:
            plt.close(fig)
    return path
