"""Journal figure geometry and typography shared by every OrganelleVerse plot.

Figures are drawn at their printed size so that the point sizes below are what
the reader sees: Nature-family widths (89 mm single column, 120 mm one and a
half, 183 mm double), 7 pt body text, 6 pt for dense gene labels and never
below 5 pt, bold lower-case 8 pt panel letters, an Arial-compatible sans serif
and editable text in PDF/SVG. A plot that is drawn larger than its printed
width only shrinks its text when placed in a manuscript, which is why text
looked small in the older 12-23 in canvases.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

MM_PER_INCH = 25.4

#: Printed figure widths (mm) for Nature-family journals.
JOURNAL_WIDTHS_MM: Mapping[str, float] = {"single": 89.0, "onehalf": 120.0, "double": 183.0}
#: Tallest figure a journal page takes with its legend (mm).
MAX_HEIGHT_MM = 170.0

#: Point sizes at printed size.
FONT_PT: Mapping[str, float] = {
    "body": 7.0,  # axis labels, legend text, centre text
    "tick": 6.0,
    "label": 6.0,  # gene and feature labels
    "title": 8.0,  # panel titles
    "panel": 8.0,  # panel letters (bold, lower case)
    "min": 5.0,  # nothing rendered is smaller than this
}

FONT_FAMILY: Sequence[str] = ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans")


def width_inches(width: str | float) -> float:
    """``"single"``/``"onehalf"``/``"double"`` or a width in millimetres, as inches."""
    if isinstance(width, str):
        try:
            return JOURNAL_WIDTHS_MM[width] / MM_PER_INCH
        except KeyError as exc:
            raise ValueError(
                f"unknown figure width {width!r}; use one of {sorted(JOURNAL_WIDTHS_MM)} or millimetres"
            ) from exc
    if width <= 0:
        raise ValueError("figure width must be positive")
    return float(width) / MM_PER_INCH


def publication_rc() -> dict[str, Any]:
    """Matplotlib settings for printed-size figures with editable text."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": list(FONT_FAMILY),
        "font.size": FONT_PT["body"],
        "axes.titlesize": FONT_PT["title"],
        "axes.labelsize": FONT_PT["body"],
        "xtick.labelsize": FONT_PT["tick"],
        "ytick.labelsize": FONT_PT["tick"],
        "legend.fontsize": FONT_PT["body"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",  # text stays text in SVG
        "pdf.fonttype": 42,  # TrueType, editable in PDF
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }


@contextmanager
def publication_style() -> Iterator[None]:
    """Apply :func:`publication_rc` for the duration of a render."""
    import matplotlib as mpl  # type: ignore

    with mpl.rc_context(publication_rc()):
        yield


@lru_cache(maxsize=1)
def journal_family() -> str:
    """The first installed family of :data:`FONT_FAMILY` (resolved once, no fallback warnings)."""
    from matplotlib.font_manager import fontManager  # type: ignore

    installed = {font.name for font in fontManager.ttflist}
    return next((family for family in FONT_FAMILY if family in installed), "DejaVu Sans")


@lru_cache(maxsize=8192)
def text_width_pt(text: str, size: float, bold: bool = False, italic: bool = False) -> float:
    """Measured advance width (points) of ``text`` set in the journal font at ``size`` pt."""
    from matplotlib.font_manager import FontProperties  # type: ignore
    from matplotlib.textpath import TextToPath  # type: ignore

    props = FontProperties(
        family=journal_family(), size=size, weight="bold" if bold else "normal",
        style="italic" if italic else "normal",
    )
    width, _height, _descent = TextToPath().get_text_width_height_descent(text, props, ismath=False)
    return float(width)


def floor_font(size: float) -> float:
    """``size`` raised to the 5 pt floor."""
    return max(FONT_PT["min"], float(size))


def add_panel_label(ax, letter: str, *, x_pt: float = -2.0, y_pt: float = 3.0) -> Any:
    """Bold lower-case panel letter whose right edge and baseline sit at the axes' top-left.

    The offset is in points from the axes box, so letters of panels in one
    column share an edge and letters in one row share a baseline, which is also
    the baseline of a left-aligned title set with ``pad=y_pt``.
    """
    from matplotlib.transforms import ScaledTranslation  # type: ignore

    offset = ScaledTranslation(x_pt / 72, y_pt / 72, ax.figure.dpi_scale_trans)
    return ax.text(
        0,
        1,
        letter.lower(),
        transform=ax.transAxes + offset,
        ha="right",
        va="baseline",
        fontsize=FONT_PT["panel"],
        fontweight="bold",
    )
