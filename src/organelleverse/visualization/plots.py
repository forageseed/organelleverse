"""Publication-quality visualizations using standard Python viz libraries.

Genome maps have two backends:
  - ``ogdraw``   — **default**. Self-contained matplotlib port of OGDrawR
                   (always available, no external tool required).
  - ``gbdraw``   — external gbdraw CLI (best quality, SVG/PNG/PDF),
                   opt-in via ``method="gbdraw"``.
Heatmaps use seaborn, scatter matplotlib, phylogenetic trees toytree or
Bio.Phylo. No hand-rolled SVG/raster code.
"""

from __future__ import annotations

from organelleverse._public_validation import validated_public

from collections.abc import Sequence
from pathlib import Path

from .ogdraw import _genbank_display_name
from .panel_capture import finish, journal_figsize, new_axes, renderer_panel
from .plot_object import OrganellePlot, plot_result
from .style import FONT_PT, publication_rc
from ..sequences import as_path
from ..inputs import Tree


@validated_public('visualization.plot_genome_map')
def plot_genome_map(
    genbank_path: Sequence[str | Path],
    *,
    title: str = "",
    dpi: int = 300,
    width: float | None = None,
    height: float | None = None,
    method: str = "ogdraw",
    show_ir: bool | None = None,
) -> OrganellePlot:
    """Draw a circular organelle genome map (compute-only).

    ``genbank_path`` is a list of GenBank files; one file with several records, or several
    files, are drawn as one circle per record on a single figure. A bare path string is
    still accepted from Python and means one file. (The annotation is a list so the
    capability binding exposes a list-of-paths parameter.)

    Backend selection (``method=``):
    - ``"ogdraw"`` — **default**. Self-contained matplotlib port of OGDrawR
      (17-colour functional categories, exon/intron blocks, dual strand, GC
      ring, legend). Always available, no external tool required.
    - ``"gbdraw"`` — external gbdraw CLI (best quality, SVG + PNG + PDF,
      gene shapes + legend). Must be requested explicitly.
    - ``"auto"`` — alias for ``"ogdraw"`` (kept for backward compatibility).

    Returns an :class:`OrganellePlot`; call ``ov.write(plot, output)`` (or
    ``plot.save(output)``) to render a ``.png``/``.svg``/``.pdf``/``.tiff``.

    Parameters
    ----------
    genbank_path : path to annotated GenBank file
    title : figure title
    dpi : resolution for raster formats
    width, height : figure size in inches
    method : "ogdraw" (default) | "gbdraw" | "auto"
    show_ir : plastid LSC/IRb/SSC/IRa band on the ogdraw map — ``None`` (default)
        draws it when a single plastid record has one, ``True`` requires it,
        ``False`` never draws it. gbdraw draws its own skew/GC rings instead.
    """
    # "auto" is kept as an alias for backward compatibility but no longer
    # prefers gbdraw.
    resolved = "ogdraw" if method == "auto" else method
    if resolved not in {"ogdraw", "gbdraw"}:
        raise ValueError(f"unknown method={method!r}; expected 'ogdraw' | 'gbdraw' | 'auto'")
    regions: dict = {}
    if resolved == "ogdraw":
        from .ogdraw import _ogdraw_regions

        regions = _ogdraw_regions(genbank_path, show_ir)

    def _render(path: str | Path) -> Path:
        return _render_genome_map(
            genbank_path,
            path,
            title=title,
            dpi=dpi,
            width=width,
            height=height,
            method=resolved,
            regions=regions,
        )

    plot = plot_result(
        "plot_genome_map",
        _render,
        organelle="mito",
        metrics={"method": resolved, "render_method": resolved, "regions": regions},
        flags=(f"{resolved}_map",),
        summary=f"{resolved} organelle map prepared.",
    )
    if resolved == "ogdraw":
        from .ogdraw import ogdraw_panel

        plot._panel = ogdraw_panel(
            genbank_path, genome_name=_genbank_display_name(genbank_path), regions=regions
        )
    return plot


def _render_genome_map(
    genbank_path: str | Path,
    output: str | Path,
    *,
    title: str = "",
    dpi: int = 300,
    width: float | None = None,
    height: float | None = None,
    method: str = "ogdraw",
    regions: dict | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_genome_map`."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if method == "ogdraw":
        from .ogdraw import _render_ogdraw_map

        return _render_ogdraw_map(
            genbank_path,
            output,
            genome_name=_genbank_display_name(genbank_path),
            title=title,
            dpi=dpi,
            width=width,
            height=height,
            regions=regions,
        )
    if not isinstance(genbank_path, (str, Path)):
        raise ValueError("method='gbdraw' draws one GenBank file; use method='ogdraw' for several")
    return _render_genome_map_gbdraw(genbank_path, output, title=title, dpi=dpi)


def _apply_white_background(drawing) -> None:
    """Insert an opaque white ``<rect>`` as the bottom-most rendered layer.

    gbdraw emits a transparent SVG background; inserting a full-canvas white
    rect (after the ``<defs>`` block so it stays behind every drawn element)
    ensures the figure sits on white for display and raster export.
    """
    bg = drawing.rect(insert=(0, 0), size=("100%", "100%"), fill="white")
    elements = getattr(drawing, "elements", None)
    if elements is None:
        drawing.add(bg)
        return
    # svgwrite places a <defs> container at index 0; insert right after it so
    # the rect is the first rendered element. Fall back to index 0 otherwise.
    insert_at = 1 if (len(elements) > 0 and elements[0].__class__.__name__ == "Defs") else 0
    elements.insert(insert_at, bg)


def _render_genome_map_gbdraw(
    genbank_path: str | Path,
    output: Path,
    *,
    title: str = "",
    dpi: int = 300,
) -> Path:
    """Genome map via the gbdraw CLI (highest quality, SVG/PNG/PDF).

    Private path-taking helper invoked at write time by ``plot_genome_map``;
    drives the gbdraw renderer directly so the compute step stays path-free.
    """
    from .gbdraw import _find_gbdraw, _render_gbdraw, _resolve_inputs

    exe = _find_gbdraw(None, runner=None)
    if exe is None:
        raise RuntimeError("gbdraw executable not found for plot_genome_map(method='gbdraw').")
    input_paths, fasta_paths, _organelle, resolved_format = _resolve_inputs(
        genbank_path, fasta=None, organelle="mito", input_format=None
    )
    return _render_gbdraw(
        exe,
        input_paths,
        fasta_paths,
        resolved_format,
        output,
        mode="circular",
        formats=None,
        runner=None,
        white_background=True,
        title=title,
        extra_args=(),
        options={
            "plot_title_position": "top" if title else "none",
            "separate_strands": True,
        },
    )


@validated_public('visualization.plot_heatmap')
def plot_heatmap(
    data: dict[str, dict[str, float]],
    *,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 300,
    width: float | None = None,
    height: float | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_label: str = "",
) -> OrganellePlot:
    """Draw a heatmap using seaborn (for ERC correlation matrices, RSCU, etc.).

    Compute-only; render with ``ov.write(plot, output)`` (``.png``/``.svg``/``.pdf``).

    Parameters
    ----------
    data : {row_label: {col_label: value}}
    cmap : colormap name
    width, height : figure size in inches (mapped to the nearest journal width);
        ``None`` sizes the figure from the matrix (square cells of about 6.5 mm)
    colorbar_label : name of the colour scale, e.g. ``"Pearson r"``

    Without ``vmin``/``vmax`` the scale is centred on 0; values of both signs
    get limits symmetric about 0 so equal magnitudes read as equal colours.
    """

    def _render(path: str | Path) -> Path:
        return _render_heatmap(
            data,
            path,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            dpi=dpi,
            width=width,
            height=height,
            figsize=figsize,
            colorbar_label=colorbar_label,
        )

    plot = plot_result(
        "plot_heatmap",
        _render,
        metrics={"render_method": "seaborn_heatmap"},
        summary="Seaborn heatmap prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


_MM = 25.4
_JOURNAL_MM = (89.0, 120.0, 183.0)
_HEATMAP_CELL_MM = 6.5
_INK = "#1F2933"
_NEUTRAL = "#52606D"
_SCATTER_BASE = "#0072B2"
_SCATTER_ACCENT = "#D55E00"


def _heatmap_auto_size(rows: Sequence[str], cols: Sequence[str]) -> tuple[float, float]:
    """Smallest journal width that holds square ~6.5 mm cells, height to match (inches)."""
    from .style import text_width_pt

    tick = FONT_PT["tick"]
    row_mm = max(text_width_pt(str(label), tick) for label in rows) / 72 * _MM + 3.0
    col_pt = max(text_width_pt(str(label), tick) for label in cols)
    cbar_mm = 15.0
    n_rows, n_cols = len(rows), len(cols)
    width_mm = next(
        (w for w in _JOURNAL_MM if row_mm + n_cols * _HEATMAP_CELL_MM + cbar_mm <= w),
        _JOURNAL_MM[-1],
    )
    cell_mm = min(_HEATMAP_CELL_MM, max(1.5, (width_mm - row_mm - cbar_mm) / n_cols))
    rotate = col_pt > 0.9 * cell_mm / _MM * 72
    col_mm = (col_pt * 0.72 + tick if rotate else tick * 1.2) / 72 * _MM + 3.0
    return width_mm / _MM, (n_rows * cell_mm + col_mm + 6.0) / _MM


def _render_heatmap(
    data: dict[str, dict[str, float]],
    output: str | Path,
    *,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 300,
    width: float | None = None,
    height: float | None = None,
    figsize: tuple[float, float] | None = None,
    colorbar_label: str = "",
) -> Path:
    """Private path-taking renderer backing :func:`plot_heatmap`."""
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore
    import seaborn as sns  # type: ignore

    from .style import text_width_pt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data).T.fillna(0)
    if figsize:
        w, h = figsize
    else:
        auto_w, auto_h = _heatmap_auto_size(list(map(str, df.index)), list(map(str, df.columns)))
        w = auto_w if width is None else width
        h = auto_h if height is None else height
    center = None
    if vmin is None and vmax is None:
        center = 0
        lo, hi = float(df.min().min()), float(df.max().max())
        if lo < 0 < hi:  # both signs: symmetric limits about 0
            vmax = max(-lo, hi)
            vmin = -vmax
    plt.rcParams.update(publication_rc())
    fig, ax = new_axes(journal_figsize(w, h))
    n_rows, n_cols = df.shape
    # colour bar 2.2 mm wide, as tall as the (square-celled) matrix
    box = ax.get_position()
    drawn_w_pt = min(
        box.width * fig.get_figwidth() * 72,
        box.height * fig.get_figheight() * 72 * n_cols / max(1, n_rows),
    )
    bar_fraction = (2.2 / _MM * 72) / max(1e-6, drawn_w_pt)
    cax = ax.inset_axes((1.0 + 0.5 * bar_fraction, 0.0, bar_fraction, 1.0))
    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_ax=cax,
    )
    box = ax.get_position()
    cell_pt = min(
        box.width * fig.get_figwidth() * 72 / max(1, n_cols),
        box.height * fig.get_figheight() * 72 / max(1, n_rows),
    )
    col_labels = [str(c) for c in df.columns]
    widest = max((text_width_pt(c, FONT_PT["tick"]) for c in col_labels), default=0.0)
    ax.set_xticks([i + 0.5 for i in range(n_cols)], col_labels)
    ax.set_yticks([i + 0.5 for i in range(n_rows)], [str(r) for r in df.index], rotation=0)
    if widest > 0.9 * cell_pt:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
    ax.tick_params(length=0, pad=2, labelsize=FONT_PT["tick"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    colorbar = ax.collections[0].colorbar if ax.collections else None
    if colorbar is not None:
        from matplotlib.ticker import MaxNLocator  # type: ignore

        colorbar.locator = MaxNLocator(nbins=5)
        colorbar.update_ticks()
    cax.tick_params(labelsize=FONT_PT["tick"], length=2, width=0.6, pad=1.5)
    for spine in cax.spines.values():
        spine.set_visible(False)
    if colorbar_label:
        cax.set_ylabel(colorbar_label, fontsize=FONT_PT["body"], labelpad=3)
    if title:
        ax.set_title(title, loc="left", fontsize=FONT_PT["title"], pad=3)
    return finish(fig, output, dpi=max(dpi, 600))


@validated_public('visualization.plot_tree')
def plot_tree(
    newick_path: str | Path | Tree,
    *,
    title: str = "",
    layout: str = "rectangular",
    dpi: int = 300,
    width: float | None = None,
    height: float | None = None,
    method: str = "auto",
    show_support: bool = True,
    italic_labels: bool = True,
) -> OrganellePlot:
    """Draw a phylogenetic tree (compute-only).

    Backend selection (``method=``):
    - ``"biophylo"`` — Bio.Phylo parsing + OrganelleVerse matplotlib drawing
      (journal widths, composable with ``compose_figure``)
    - ``"toytree"`` — toytree/toyplot, when installed (not composable)
    - ``"auto"`` — ``"biophylo"``, so one call draws the same figure on every
      machine whether or not toytree happens to be installed

    Render with ``ov.write(plot, output)`` (``.png``/``.svg``/``.pdf``).

    Parameters
    ----------
    newick_path : path to Newick tree file
    title : figure title
    layout : "rectangular" | "circular" (circular is drawn by OrganelleVerse;
        for groups, clade brackets, data rings or tree conflict see
        :func:`~organelleverse.visualization.tree_plots.plot_phylo_tree` and
        :func:`~organelleverse.visualization.tree_plots.plot_tree_conflict`)
    dpi : resolution for raster formats
    width, height : figure size in inches; ``None`` sizes the figure from the
        number of tips (89 or 120 mm wide, ~3.4 mm per tip; circular is square)
    method : "auto" | "toytree" | "biophylo"
    show_support : write internal-node support values
    italic_labels : set tip names in italic, as species names; underscores in
        tip names are shown as spaces
    """
    newick_path = as_path(newick_path)
    if method not in {"auto", "biophylo", "toytree"}:
        raise ValueError("method must be 'auto', 'biophylo' or 'toytree'")
    resolved = "biophylo" if method == "auto" else method
    if resolved == "toytree":
        try:
            import toytree  # noqa: F401
        except ImportError as exc:
            raise ImportError("method='toytree' needs toytree; use method='biophylo'") from exc

    def _render(path: str | Path) -> Path:
        return _render_tree(
            newick_path,
            path,
            title=title,
            layout=layout,
            dpi=dpi,
            width=width,
            height=height,
            method=resolved,
            show_support=show_support,
            italic_labels=italic_labels,
        )

    if layout not in {"rectangular", "circular"}:
        raise ValueError("layout must be 'rectangular' or 'circular'")

    plot = plot_result(
        "plot_tree",
        _render,
        metrics={"method": resolved, "layout": layout},
        summary=f"Phylogenetic tree prepared ({resolved}).",
    )
    if resolved == "biophylo" and layout == "rectangular":
        plot._panel = renderer_panel(_render)
    return plot


def _render_tree(
    newick_path: str | Path,
    output: str | Path,
    *,
    title: str = "",
    layout: str = "rectangular",
    dpi: int = 300,
    width: float | None = None,
    height: float | None = None,
    method: str = "biophylo",
    show_support: bool = True,
    italic_labels: bool = True,
) -> Path:
    """Private path-taking renderer backing :func:`plot_tree`."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if layout == "circular" and method != "toytree":
        from .tree_plots import render_circular_tree

        return render_circular_tree(
            newick_path, output, italic=italic_labels, show_support=show_support
        )
    if method == "toytree":
        return _render_tree_toytree(
            newick_path,
            output,
            title=title,
            layout=layout,
            dpi=dpi,
            width=width,
            height=height,
            show_support=show_support,
            italic_labels=italic_labels,
        )
    return _render_tree_biophylo(
        newick_path,
        output,
        title=title,
        dpi=dpi,
        width=width,
        height=height,
        show_support=show_support,
        italic_labels=italic_labels,
    )


def _render_tree_toytree(
    newick_path: str | Path,
    output: Path,
    *,
    title: str = "",
    layout: str = "rectangular",
    dpi: int = 300,
    width: float | None = None,
    height: float | None = None,
    show_support: bool = True,
    italic_labels: bool = True,
) -> Path:
    """Tree via toytree/toyplot at journal size; the file type follows the suffix.

    toyplot measures in CSS pixels (96 per inch, so 1 pt = 4/3 px). toytree
    draws the branches; tip names are written here with toyplot text so they can
    be italic (toytree strips markup from its own labels), and support is shown
    as dots like the Bio.Phylo backend (black >= 95 or 0.95, grey >= 70 or 0.90).
    PNG, SVG and PDF use toyplot's own renderers; any other suffix is an error
    rather than a file whose content does not match its name.
    """
    import math

    import toyplot  # type: ignore
    import toyplot.pdf  # type: ignore
    import toyplot.png  # type: ignore
    import toyplot.svg  # type: ignore
    import toytree  # type: ignore

    from ..phylogeny.treecompare import read_tree
    from .style import text_width_pt

    writers = {".png": toyplot.png.render, ".svg": toyplot.svg.render, ".pdf": toyplot.pdf.render}
    suffix = output.suffix.lower()
    if suffix not in writers:
        raise ValueError(f"toytree writes .png, .svg or .pdf, not {suffix or 'no suffix'!r}")

    tre = toytree.tree(read_tree(newick_path))
    labels = [_tree_tip_text(name) for name in tre.get_tip_labels()]
    n_tips = len(labels)
    n_nodes = tre.nnodes
    circular = layout == "circular"
    px = 4.0 / 3.0
    size = FONT_PT["label"]
    label_px = max(
        (text_width_pt(t, size, italic=italic_labels) for t in labels), default=0.0
    ) * px + 4.0 * px
    if width is None:
        if circular:
            # the tree keeps at least ~45% of the diameter once labels sit around it
            need_mm = 2 * label_px / px / 72 * _MM / 0.55
            width = next((mm for mm in _JOURNAL_MM if mm >= need_mm), _JOURNAL_MM[-1]) / _MM
        else:
            width = (89 if label_px / px <= 0.42 * 89 / _MM * 72 else 120) / _MM
    if height is None:
        height = width if circular else 0.135 * n_tips + 0.42
    width, height = journal_figsize(width, height)

    dists = [float(d or 0) for d in tre.get_node_data("dist")]
    has_lengths = any(d > 0 for d in dists[:-1])
    sizes, colours = [0.0] * n_nodes, ["white"] * n_nodes
    if show_support:
        values = []
        for value in tre.get_node_data("support"):
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(math.nan)
        finite = [v for v in values if not math.isnan(v)]
        # same thresholds as plot_phylo_tree: bootstrap 70/95, posterior 0.90/0.95
        moderate, strong = (0.90, 0.95) if finite and max(finite) <= 1.0 else (70.0, 95.0)
        # toytree numbers tips first and the root last; neither carries support
        for i in range(n_tips, n_nodes - 1):
            if math.isnan(values[i]):
                continue
            if values[i] >= strong:
                sizes[i], colours[i] = 3.2 * px, _INK
            elif values[i] >= moderate:
                sizes[i], colours[i] = 3.2 * px, "#8C939D"

    canvas = toyplot.Canvas(width=width * 96, height=height * 96, style={"background-color": "white"})
    pad = 3.0 * px
    if circular:
        margin = (label_px + pad,) * 4
    else:
        margin = (pad + (8 * px if title else 0), label_px + pad,
                  (18 * px if has_lengths else pad), pad)
    axes = canvas.cartesian(margin=margin)
    _canvas, axes, mark = tre.draw(
        axes=axes,
        layout="c" if circular else "r",
        # toytree 3 draws circular trees with straight edges only
        edge_type="c" if circular else "p",
        tip_labels=False,
        edge_style={"stroke": _INK, "stroke-width": 0.7 * px, "stroke-linecap": "square"},
        node_mask=False,
        node_sizes=sizes,
        node_colors=colours,
        node_style={"stroke": "white", "stroke-width": 0.4 * px},
        scale_bar=has_lengths and not circular,
        use_edge_lengths=has_lengths,
        tip_labels_align=False,
        edge_align_style={"stroke-opacity": 0.0},
    )
    text = [f"<i>{t}</i>" if italic_labels else t for t in labels]
    style = {"font-size": f"{size * px:.2f}px", "fill": _INK}
    coords = mark.ntable[:n_tips]
    if circular:
        # toytree's own extents leave the circle small; the margins already hold the labels
        radius = float(max(math.hypot(x, y) for x, y in mark.ntable)) or 1.0
        axes.x.domain.min, axes.x.domain.max = -radius, radius
        axes.y.domain.min, axes.y.domain.max = -radius, radius
        # labels go on the canvas (axes clip text), at the tips' projected positions
        right, left = [], []
        for (x, y), t in zip(coords, text, strict=True):
            angle = math.degrees(math.atan2(y, x)) % 360.0
            # data -> canvas px: square axes inset by the same margin on every side
            cx = margin[0] + (x + radius) / (2 * radius) * (width * 96 - 2 * margin[0])
            cy = margin[0] + (radius - y) / (2 * radius) * (height * 96 - 2 * margin[0])
            (left if 90.0 < angle < 270.0 else right).append((cx, cy, angle, t))
        for group, anchor, shift, turn in ((right, "start", 2.5, 0.0), (left, "end", -2.5, 180.0)):
            if group:
                xs, ys, angles, texts = zip(*group, strict=True)
                canvas.text(list(xs), list(ys), list(texts), angle=[a - turn for a in angles], style={
                    **style, "text-anchor": anchor, "-toyplot-anchor-shift": f"{shift * px:.1f}px",
                })
        axes.show = False
    else:
        axes.text(list(coords[:, 0]), list(coords[:, 1]), text, style={
            **style, "text-anchor": "start", "-toyplot-anchor-shift": f"{2.5 * px:.1f}px",
        })
        axes.y.show = False
        if has_lengths:
            axes.x.ticks.labels.style = {"font-size": f"{FONT_PT['tick'] * px:.2f}px"}
            axes.x.spine.style = {"stroke-width": 0.6 * px}
        else:
            axes.x.show = False
    if title:
        canvas.text(pad, pad + 4 * px, title, style={
            "font-size": f"{FONT_PT['title'] * px:.2f}px", "text-anchor": "start", "fill": _INK,
        })
    if suffix == ".png":
        writers[suffix](canvas, str(output), scale=max(dpi, 600) / 96)
    else:
        writers[suffix](canvas, str(output))
    return output


def _tree_tip_text(name: str | None) -> str:
    return (name or "").strip().strip("'\"").replace("_", " ")


def _tree_scale_length(span: float) -> float:
    """A round scale-bar length (1, 2 or 5 x 10^k) of at most a quarter of the tree depth."""
    from math import floor, log10

    limit = span * 0.25
    base = 10 ** floor(log10(limit))
    return max(m * base for m in (1, 2, 5) if m * base <= limit)


def _render_tree_biophylo(
    newick_path: str | Path,
    output: Path,
    *,
    title: str = "",
    dpi: int = 300,
    width: float | None = None,
    height: float | None = None,
    show_support: bool = True,
    italic_labels: bool = True,
) -> Path:
    """Tree via Bio.Phylo + matplotlib (clean academic style).

    Rectangular phylogram: tips in italic without underscores, a scale bar
    instead of an x axis and optional support values at internal nodes.
    """
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from Bio import Phylo  # type: ignore
    from matplotlib.collections import LineCollection  # type: ignore

    from .style import text_width_pt

    tree = Phylo.read(str(newick_path), "newick")
    terminals = tree.get_terminals()
    depths = tree.depths()
    has_lengths = bool(max(depths.values()))
    if not has_lengths:
        depths = tree.depths(unit_branch_lengths=True)
    span = max(depths.values()) or 1.0

    y_pos: dict = {tip: float(i) for i, tip in enumerate(terminals)}
    parents: dict = {}
    for clade in tree.find_clades(order="postorder"):
        for child in clade.clades:
            parents[child] = clade
        if clade not in y_pos:
            y_pos[clade] = (y_pos[clade.clades[0]] + y_pos[clade.clades[-1]]) / 2

    labels = [_tree_tip_text(tip.name) for tip in terminals]
    size = FONT_PT["label"]
    label_pt = max(
        (text_width_pt(label, size, italic=italic_labels) for label in labels), default=0.0
    )
    n_tips = len(terminals)
    if width is None:
        width = (89 if label_pt <= 0.42 * 89 / _MM * 72 else 120) / _MM
    if height is None:
        height = 0.135 * n_tips + (0.42 if has_lengths else 0.22)

    plt.rcParams.update(publication_rc())
    fig, ax = new_axes(journal_figsize(width, height))

    segments = []
    for clade in tree.find_clades():
        x, y = depths[clade], y_pos[clade]
        start = depths[parents[clade]] if clade in parents else -0.015 * span
        segments.append([(start, y), (x, y)])
        if clade.clades:
            segments.append([(x, y_pos[clade.clades[0]]), (x, y_pos[clade.clades[-1]])])
    ax.add_collection(
        LineCollection(segments, colors=_INK, linewidths=0.7, capstyle="projecting")
    )
    for tip, label in zip(terminals, labels, strict=True):
        ax.annotate(
            label,
            (depths[tip], y_pos[tip]),
            xytext=(2.5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=size,
            fontstyle="italic" if italic_labels else "normal",
            color=_INK,
        )
    if show_support:
        for clade in tree.get_nonterminals():
            confidence = getattr(clade, "confidence", None)
            if clade is tree.root or confidence is None:
                continue
            text = f"{confidence:.0f}" if confidence > 1 else f"{confidence:.2f}"
            # right of the node, in the gap between its two child branches
            ax.annotate(
                text,
                (depths[clade], y_pos[clade]),
                xytext=(1.5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=FONT_PT["label"],
                color=_NEUTRAL,
                # a short child branch can pass under the value; keep it legible
                bbox={"boxstyle": "square,pad=0.05", "fc": "white", "ec": "none"},
                zorder=3,
            )
    bottom = n_tips - 1 + 0.6
    if has_lengths:
        length = _tree_scale_length(span)
        bar_y = n_tips - 1 + 1.2
        ax.plot([0, length], [bar_y, bar_y], color=_INK, lw=0.7, solid_capstyle="butt")
        ax.annotate(
            f"{length:g}",
            (length / 2, bar_y),
            xytext=(0, -1.5),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=FONT_PT["tick"],
            color=_INK,
        )
        bottom = bar_y + 0.9
    # leave room right of the deepest tip for its label so the plot fits its axes
    axes_pt = ax.get_position().width * fig.get_figwidth() * 72
    usable = max(axes_pt * 0.3, axes_pt - label_pt - 5)
    left = -0.03 * span
    ax.set_xlim(left, left + (span - left) * axes_pt / usable)
    ax.set_ylim(bottom, -0.7)
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", fontsize=FONT_PT["title"], pad=3)
    return finish(fig, output, dpi=max(dpi, 600))


@validated_public('visualization.plot_scatter')
def plot_scatter(
    x: list[float],
    y: list[float],
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    labels: list[str] | None = None,
    highlight_threshold: float | None = None,
    dpi: int = 300,
    width: float | None = None,
    height: float | None = None,
) -> OrganellePlot:
    """Draw a scatter plot (for Ka/Ks, GC content, etc.) (compute-only).

    Render with ``ov.write(plot, output)`` (``.png``/``.svg``/``.pdf``).

    Parameters
    ----------
    x, y : data arrays
    labels : point labels (placed beside their points, avoiding overlaps)
    highlight_threshold : draw a horizontal line at this y value; points above it
        are drawn in the accent colour
    width, height : figure size in inches (mapped to the nearest journal width);
        ``None`` gives 89 mm (120 mm for more than 20 labelled points)
    """

    def _render(path: str | Path) -> Path:
        return _render_scatter(
            x,
            y,
            path,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            labels=labels,
            highlight_threshold=highlight_threshold,
            dpi=dpi,
            width=width,
            height=height,
        )

    plot = plot_result(
        "plot_scatter",
        _render,
        metrics={"points": len(x)},
        summary="Scatter plot prepared.",
    )
    plot._panel = renderer_panel(_render)
    return plot


#: Label positions tried around a point, in points: (dx, dy, ha, va).
_SCATTER_LABEL_SLOTS = (
    (3.0, 0.0, "left", "center"),
    (2.5, 2.0, "left", "bottom"),
    (2.5, -2.0, "left", "top"),
    (-3.0, 0.0, "right", "center"),
    (-2.5, 2.0, "right", "bottom"),
    (-2.5, -2.0, "right", "top"),
    (0.0, 3.0, "center", "bottom"),
    (0.0, -3.0, "center", "top"),
)


def _scatter_place_labels(
    ax, xs: Sequence[float], ys: Sequence[float], labels: Sequence[str], *,
    size: float, marker_pt: float, first: Sequence[int] = (),
) -> None:
    """Greedy label placement: first free slot around each point, clear of points and labels."""
    from .style import text_width_pt

    px = ax.figure.dpi / 72
    centres = ax.transData.transform(list(zip(xs, ys, strict=False)))
    radius = marker_pt / 2 * px
    taken = [(cx - radius, cy - radius, cx + radius, cy + radius) for cx, cy in centres]
    frame = ax.get_window_extent()
    order = list(first) + [i for i in range(len(labels)) if i not in set(first)]

    def free(rect) -> bool:
        x0, y0, x1, y1 = rect
        if x0 < frame.x0 or x1 > frame.x1 or y0 < frame.y0 or y1 > frame.y1:
            return False
        return all(x1 <= a0 or x0 >= a1 or y1 <= b0 or y0 >= b1 for a0, b0, a1, b1 in taken)

    for i in order:
        label = str(labels[i])
        if not label:
            continue
        cx, cy = centres[i]
        w = text_width_pt(label, size) * px
        h = size * px
        placed = None
        for dx, dy, ha, va in _SCATTER_LABEL_SLOTS:
            ox, oy = cx + dx * px, cy + dy * px
            x0 = ox if ha == "left" else ox - w if ha == "right" else ox - w / 2
            y0 = oy if va == "bottom" else oy - h if va == "top" else oy - h / 2
            rect = (x0, y0, x0 + w, y0 + h)
            if free(rect):
                placed = (dx, dy, ha, va, rect)
                break
        if placed is None:  # crowded: keep the default slot rather than drop the label
            dx, dy, ha, va = _SCATTER_LABEL_SLOTS[0]
            rect = (cx + dx * px, cy - h / 2, cx + dx * px + w, cy + h / 2)
            placed = (dx, dy, ha, va, rect)
        dx, dy, ha, va, rect = placed
        taken.append(rect)
        ax.annotate(
            label, (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, fontsize=size, color=_INK,
        )


def _render_scatter(
    x: list[float],
    y: list[float],
    output: str | Path,
    *,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    labels: list[str] | None = None,
    highlight_threshold: float | None = None,
    dpi: int = 300,
    width: float | None = None,
    height: float | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_scatter`."""
    import matplotlib.pyplot as plt  # type: ignore

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    many_labels = bool(labels) and len(x) > 20
    default_w = (120 if many_labels else 89) / _MM
    w = default_w if width is None else width
    h = default_w * 0.72 if height is None else height
    plt.rcParams.update(publication_rc())
    fig, ax = new_axes(journal_figsize(w, h))
    marker_area = 14.0
    above = [
        highlight_threshold is not None and yi > highlight_threshold for yi in y
    ]
    colors = [_SCATTER_ACCENT if flag else _SCATTER_BASE for flag in above]
    ax.scatter(
        x, y, s=marker_area, c=colors, alpha=0.8, edgecolors="white", linewidths=0.3, zorder=2
    )
    if highlight_threshold is not None:
        ax.axhline(
            y=highlight_threshold, color=_NEUTRAL, linestyle=(0, (3, 2)), lw=0.6, zorder=1
        )
    ax.margins(0.06)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    if labels:
        n_labelled = min(len(x), len(y), len(labels))
        _scatter_place_labels(
            ax,
            x[:n_labelled],
            y[:n_labelled],
            labels[:n_labelled],
            size=FONT_PT["label"],
            marker_pt=marker_area**0.5,
            first=[i for i, flag in enumerate(above[:n_labelled]) if flag],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=FONT_PT["title"], pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    return finish(fig, output, dpi=max(dpi, 600))
