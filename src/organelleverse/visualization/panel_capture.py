"""Journal sizing and panel capture for renderers that build their own figure.

The older suite renderers create one figure with ``plt.subplots`` and save it.
Routing those two calls through :func:`new_axes` and :func:`finish` keeps their
drawing code unchanged while letting :func:`renderer_panel` draw the same plot
into one panel of a :func:`~organelleverse.visualization.compose_figure`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .style import JOURNAL_WIDTHS_MM, MAX_HEIGHT_MM, MM_PER_INCH

#: ``("draw", ax)`` while a renderer draws into a panel, ``("measure", None)``
#: while its figure size is being read, ``None`` for a normal render.
_TARGET: ContextVar[tuple[str, Any] | None] = ContextVar("ov_panel_target", default=None)


class _Measured(Exception):  # noqa: N818 - control flow, not an error
    def __init__(self, figsize: tuple[float, float]) -> None:
        super().__init__("figure size measured")
        self.figsize = figsize


def journal_figsize(width_in: float, height_in: float) -> tuple[float, float]:
    """A legacy canvas mapped to the nearest journal width, height scaled to match.

    Text is set in printed points, so the canvas must be its printed size: wide
    layouts (>= 6.5 in) become the 183 mm double column, mid-width ones 120 mm,
    narrow ones the 89 mm single column. The height keeps the original aspect
    and never exceeds the 170 mm page limit. Journal sizes map to themselves.
    """
    if width_in >= 6.5:
        target_mm = JOURNAL_WIDTHS_MM["double"]
    elif width_in >= 4.5:
        target_mm = JOURNAL_WIDTHS_MM["onehalf"]
    else:
        target_mm = JOURNAL_WIDTHS_MM["single"]
    width = target_mm / MM_PER_INCH
    height = min(height_in * width / width_in, MAX_HEIGHT_MM / MM_PER_INCH)
    return width, height


def new_axes(figsize: tuple[float, float]) -> tuple[Any, Any]:
    """``plt.subplots(figsize=...)``, or the panel's axes when drawing into a composed figure."""
    target = _TARGET.get()
    if target is None:
        import matplotlib.pyplot as plt  # type: ignore

        return plt.subplots(figsize=figsize)
    mode, ax = target
    if mode == "measure":
        raise _Measured(figsize)
    return ax.figure, ax


def finish(fig: Any, output: str | Path, *, dpi: int = 600) -> Path:
    """Lay out, save and close a figure; a no-op while drawing into a panel."""
    if _TARGET.get() is not None:
        return Path(output)
    import matplotlib.pyplot as plt  # type: ignore

    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(output)


def renderer_panel(render: Callable[[Path], Any], *, header: str = "") -> Any:
    """A :class:`~.plot_object.PanelDrawer` for a single-axes renderer.

    The renderer runs once in "measure" mode, stopping at :func:`new_axes`, to
    read its canvas aspect; drawing runs it again with the panel's axes. Returns
    ``None`` when the renderer does not create its figure through ``new_axes``
    (multi-axes layouts), so such plots stay out of composed figures.
    """
    from .plot_object import PanelDrawer

    token = _TARGET.set(("measure", None))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            render(Path(tmp) / "measure.png")
        return None  # finished without asking for axes: not a single-axes renderer
    except _Measured as measured:
        width, height = measured.figsize
        aspect = height / width
    except Exception:  # noqa: BLE001 - a renderer that cannot be measured is not composable
        return None
    finally:
        _TARGET.reset(token)

    def draw(ax: Any, legend: bool) -> list[Any]:
        token = _TARGET.set(("draw", ax))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                render(Path(tmp) / "panel.png")
        finally:
            _TARGET.reset(token)
        return []  # legacy renderers keep their legend inside the panel

    return PanelDrawer(draw, aspect=aspect, header=header)
