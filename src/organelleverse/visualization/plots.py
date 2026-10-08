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

from collections.abc import Sequence
from pathlib import Path

from .ogdraw import _genbank_display_name
from .plot_object import OrganellePlot, plot_result


def plot_genome_map(
    genbank_path: Sequence[str | Path],
    *,
    title: str = "",
    dpi: int = 300,
    width: float = 8,
    height: float = 8,
    method: str = "ogdraw",
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
    """
    # "auto" is kept as an alias for backward compatibility but no longer
    # prefers gbdraw.
    resolved = "ogdraw" if method == "auto" else method
    if resolved not in {"ogdraw", "gbdraw"}:
        raise ValueError(f"unknown method={method!r}; expected 'ogdraw' | 'gbdraw' | 'auto'")

    def _render(path: str | Path) -> Path:
        return _render_genome_map(
            genbank_path,
            path,
            title=title,
            dpi=dpi,
            width=width,
            height=height,
            method=resolved,
        )

    return plot_result(
        "plot_genome_map",
        _render,
        organelle="mito",
        metrics={"method": resolved, "render_method": resolved},
        flags=(f"{resolved}_map",),
        summary=f"{resolved} organelle map prepared.",
    )


def _render_genome_map(
    genbank_path: str | Path,
    output: str | Path,
    *,
    title: str = "",
    dpi: int = 300,
    width: float = 8,
    height: float = 8,
    method: str = "ogdraw",
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


def plot_heatmap(
    data: dict[str, dict[str, float]],
    *,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 300,
    width: float = 8,
    height: float = 7,
    figsize: tuple[float, float] | None = None,
) -> OrganellePlot:
    """Draw a heatmap using seaborn (for ERC correlation matrices, RSCU, etc.).

    Compute-only; render with ``ov.write(plot, output)`` (``.png``/``.svg``/``.pdf``).

    Parameters
    ----------
    data : {row_label: {col_label: value}}
    cmap : colormap name
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
        )

    return plot_result(
        "plot_heatmap",
        _render,
        metrics={"render_method": "seaborn_heatmap"},
        summary="Seaborn heatmap prepared.",
    )


def _render_heatmap(
    data: dict[str, dict[str, float]],
    output: str | Path,
    *,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 300,
    width: float = 8,
    height: float = 7,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Private path-taking renderer backing :func:`plot_heatmap`."""
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore
    import seaborn as sns  # type: ignore

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data).T.fillna(0)
    if figsize:
        w, h = figsize
    else:
        w, h = width, height
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0 if vmin is None and vmax is None else None,
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_tree(
    newick_path: str | Path,
    *,
    title: str = "",
    layout: str = "rectangular",
    dpi: int = 300,
    width: float = 10,
    height: float = 8,
    method: str = "auto",
) -> OrganellePlot:
    """Draw a phylogenetic tree (compute-only).

    Backend selection (``method=``):
    - ``"toytree"`` — toytree (best quality, publication-ready, clean aesthetics)
    - ``"biophylo"`` — Bio.Phylo + matplotlib (clean academic style)
    - ``"auto"`` — toytree if installed, else Bio.Phylo

    Render with ``ov.write(plot, output)`` (``.png``/``.svg``/``.pdf``).

    Parameters
    ----------
    newick_path : path to Newick tree file
    title : figure title
    layout : "rectangular" | "circular"
    dpi : resolution for raster formats
    width, height : figure size in inches
    method : "auto" | "toytree" | "biophylo"
    """
    resolved = method
    if resolved == "auto":
        try:
            import toytree  # noqa: F401

            resolved = "toytree"
        except ImportError:
            resolved = "biophylo"

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
        )

    return plot_result(
        "plot_tree",
        _render,
        metrics={"method": resolved, "layout": layout},
        summary=f"Phylogenetic tree prepared ({resolved}).",
    )


def _render_tree(
    newick_path: str | Path,
    output: str | Path,
    *,
    title: str = "",
    layout: str = "rectangular",
    dpi: int = 300,
    width: float = 10,
    height: float = 8,
    method: str = "biophylo",
) -> Path:
    """Private path-taking renderer backing :func:`plot_tree`."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if method == "toytree":
        return _render_tree_toytree(newick_path, output, title=title, width=width, height=height)
    return _render_tree_biophylo(
        newick_path, output, title=title, dpi=dpi, width=width, height=height
    )


def _render_tree_toytree(
    newick_path: str | Path,
    output: Path,
    *,
    title: str = "",
    width: float = 10,
    height: float = 8,
) -> Path:
    """Tree via toytree (best quality)."""
    import toyplot.png  # type: ignore
    import toyplot.svg  # type: ignore
    import toytree  # type: ignore

    newick_text = Path(newick_path).read_text().strip()
    tre = toytree.tree(newick_text)

    w_px = int(width * 80)
    h_px = int(height * 80)
    canvas, _axes, _mark = tre.draw(
        width=w_px,
        height=h_px,
        tip_labels_style={"font-size": "13px", "font-family": "Helvetica", "font-weight": "normal"},
        edge_style={"stroke-width": 2.5, "stroke": "#2c2c2c"},
        node_labels=False,
        scale_bar=True,
    )

    suffix = output.suffix.lower()
    if suffix == ".png":
        toyplot.png.render(canvas, str(output))
    elif suffix == ".svg":
        toyplot.svg.render(canvas, str(output))
    else:
        toyplot.png.render(canvas, str(output))
    return output


def _render_tree_biophylo(
    newick_path: str | Path,
    output: Path,
    *,
    title: str = "",
    dpi: int = 300,
    width: float = 10,
    height: float = 8,
) -> Path:
    """Tree via Bio.Phylo + matplotlib (clean academic style)."""
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from Bio import Phylo  # type: ignore

    tree = Phylo.read(str(newick_path), "newick")
    fig, ax = plt.subplots(figsize=(width, height))
    Phylo.draw(
        tree,
        axes=ax,
        do_show=False,
        label_func=lambda clade: clade.name or "",
        show_confidence=False,
    )
    ax.set_xlabel("Branch Length", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    for txt in ax.texts:
        txt.set_fontsize(10)
        txt.set_fontstyle("italic")
    if title:
        ax.set_title(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


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
    width: float = 8,
    height: float = 6,
) -> OrganellePlot:
    """Draw a scatter plot (for Ka/Ks, GC content, etc.) (compute-only).

    Render with ``ov.write(plot, output)`` (``.png``/``.svg``/``.pdf``).

    Parameters
    ----------
    x, y : data arrays
    labels : point labels
    highlight_threshold : draw a horizontal line at this y value
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

    return plot_result(
        "plot_scatter",
        _render,
        metrics={"points": len(x)},
        summary="Scatter plot prepared.",
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
    width: float = 8,
    height: float = 6,
) -> Path:
    """Private path-taking renderer backing :func:`plot_scatter`."""
    import matplotlib.pyplot as plt  # type: ignore

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.scatter(x, y, alpha=0.6, s=30, edgecolors="white", linewidth=0.3)
    if highlight_threshold is not None:
        ax.axhline(y=highlight_threshold, color="red", linestyle="--", alpha=0.5)
    if labels:
        for xi, yi, label in zip(x, y, labels, strict=False):
            ax.annotate(label, (xi, yi), fontsize=5, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output
