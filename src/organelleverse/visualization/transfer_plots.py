"""NUMT/NUPT figures on nuclear chromosomes.

Backs :func:`~organelleverse.visualization.suite_plots.plot_transfer_tracks`
(horizontal chromosome tracks) and
:func:`~organelleverse.visualization.suite_plots.nuclear_transfer_ideogram`
(vertical ideogram). Both share one drawing core:

* chromosomes in natural order (1, 2, ..., 10; ``chr`` prefixes and
  alphanumeric names handled), coordinates in Mb, optional centromeres;
* each insertion is a thin semi-transparent tick outside the chromosome,
  tick length and width encode the fragment length (binned, with a legend);
* a density heat fill inside each chromosome (candidates per window) so
  thousands of overlapping insertions stay readable;
* NUMT on the upper/left side, NUPT on the lower/right side, so both
  organelles can share one figure;
* colour by organelle, evidence category, identity or organelle gene.

Per-sample summaries live here too: :func:`plot_transfer_length_profile`
(counts per length bin, one line per sample/haplotype) and
:func:`plot_transfer_region_te` (share per functional region, split by TE
class) with the interval statistics :func:`summarize_transfer_regions`.

Everything is drawn in batches (one ``LineCollection``/``PolyCollection`` per
layer), so thousands of candidates render in well under a second.
"""

from __future__ import annotations

import bisect
import itertools
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection  # type: ignore
from matplotlib.colors import (  # type: ignore
    LinearSegmentedColormap,
    Normalize,
    PowerNorm,
    to_hex,
    to_rgba,
)
from matplotlib.lines import Line2D  # type: ignore
from matplotlib.patches import FancyBboxPatch, Patch, Polygon  # type: ignore
from matplotlib.ticker import FuncFormatter, MaxNLocator  # type: ignore
from matplotlib.transforms import Affine2D  # type: ignore

from .plot_object import PanelDrawer
from .style import (
    FONT_PT,
    MAX_HEIGHT_MM,
    MM_PER_INCH,
    add_panel_label,
    publication_style,
    width_inches,
)

TRANSFER_COLORS = {"NUMT": "#0072B2", "NUPT": "#009E73"}
TRANSFER_ORGANELLES = {"NUMT": "mitochondrion", "NUPT": "plastid"}
# NUMT ticks sit above (tracks) / left of (ideogram) the chromosome, NUPT below / right.
TRANSFER_SIDES = {"NUMT": -1, "NUPT": 1}

# Evidence categories, from ``detect_transfers_evidence``: ``evidence_level``
# counts the validation layers that support a BLAST candidate (0 = BLAST only,
# +1 read depth, +1 long-read linkage), so level 1 may be either layer.
EVIDENCE_ORDER = ("candidate", "not_validated", "depth", "longread", "one_layer", "both")
EVIDENCE_LABELS = {
    "candidate": "Candidate (alignment only)",
    "not_validated": "Alignment only (not validated)",
    "depth": "Read depth",
    "longread": "Long-read linked",
    "one_layer": "One validation layer",
    "both": "Read depth + long read",
}
EVIDENCE_COLORS = {
    "not_validated": "#AEB6BF",
    "depth": "#E69F00",
    "longread": "#7B5EA7",
    "one_layer": "#7B5EA7",
    "both": "#C0392B",
}
GENE_CLASS_LABELS = {"within_gene": "Within nuclear gene", "intergenic": "Intergenic"}
COLOR_MODES = ("evidence_level", "organelle", "identity", "organelle_gene")

# (minimum bp, legend label, relative tick length, line width)
LENGTH_BINS = (
    (0, "<0.5 kb", 0.36, 0.45),
    (500, "0.5–2 kb", 0.56, 0.6),  # noqa: RUF001 (en dash in a range)
    (2_000, "2–10 kb", 0.78, 0.85),  # noqa: RUF001
    (10_000, "≥10 kb", 1.0, 1.25),
)
_GENE_PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#8C6D31",
    "#7B5EA7",
)
_INK = "#1F2933"
_MUTED = "#52606D"
_FAINT = "#9AA5B1"
_BAR_FILL = "#F3F4F6"
_BAR_EDGE = "#8A96A3"
_GRID = "#E9ECEF"
_TICK_ALPHA = 0.6
_CHROM_PREFIX = re.compile(r"^(?:chromosome|chrom|chr)[\s_.-]*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data normalisation
# ---------------------------------------------------------------------------


def chromosome_sort_key(name: str) -> tuple[Any, ...]:
    """Natural chromosome order: 1, 2, ..., 10, then X/Y/Mt-style names, then the rest.

    ``chr``/``Chr``/``chromosome`` prefixes are ignored, numeric parts compare
    as numbers (``1A`` < ``1B`` < ``2A``; ``scaffold_2`` < ``scaffold_10``), and
    short names (``X``, ``Y``, ``W``, ``Z``, ``Mt``) come before scaffolds.
    """
    text = str(name)
    stripped = _CHROM_PREFIX.sub("", text) or text
    match = re.match(r"^(\d+)(.*)$", stripped)
    if match:
        return (0, int(match.group(1)), _natural_parts(match.group(2)), text)
    return (1 if len(stripped) <= 2 else 2, 0, _natural_parts(stripped), text)


def _natural_parts(text: str) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part.lower())
        for part in re.split(r"(\d+)", text)
        if part
    )


def chromosome_display(name: str) -> str:
    """``"1"`` -> ``"Chr1"``; any other name is shown as given."""
    text = str(name)
    return f"Chr{text}" if text.isdigit() else text


def evidence_category(row: Mapping[str, Any]) -> str:
    """Name the evidence behind one candidate (see :data:`EVIDENCE_LABELS`).

    The depth/long-read flags are ``None`` when that layer was not run. With no
    flags at all (only BLAST, or rows that kept just ``evidence_level``), level
    0 is a plain candidate, 1 is one unnamed validation layer, 2 is both.
    """
    depth = row.get("depth_supported")
    longread = row.get("longread_linked")
    if depth is None and longread is None:
        if row.get("evidence_level") in (None, ""):
            return "unspecified"  # plain fragment rows carry no evidence at all
        level = _int(row.get("evidence_level"))
        if level >= 2:
            return "both"
        if level == 1:
            return "one_layer"
        return "candidate"
    if depth and longread:
        return "both"
    if depth:
        return "depth"
    if longread:
        return "longread"
    return "not_validated"


def transfer_label_from_rows(rows: Sequence[Mapping[str, Any]], default: str = "NUMT") -> str:
    """Guess NUMT/NUPT from ``organelle``/``organelle_seqid`` fields."""
    for row in rows[:50]:
        text = f"{row.get('organelle', '')} {row.get('organelle_seqid', '')}".lower()
        if any(token in text for token in ("plastid", "chloroplast", "plastome", "_cp", "ptdna")):
            return "NUPT"
        if any(token in text for token in ("mito", "_mt", "mtdna")):
            return "NUMT"
    return default


def normalize_transfer_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reduce one candidate/fragment mapping to the fields the figures use.

    Coordinates are returned in bp. ``nuclear_start``/``nuclear_end`` are
    always bp; ``*_mb`` keys are Mb; generic ``start``/``end`` keep the legacy
    ideogram rule (values up to 1000 are Mb).
    """
    lowered = {str(key).lower(): value for key, value in row.items()}
    seqid = _first(lowered, "nuclear_seqid", "chromosome", "chrom", "chr", "seqid", "sequence")
    if seqid in (None, ""):
        return None
    start_mb = _first(lowered, "start_mb", "nuclear_start_mb", "position_mb")
    if start_mb not in (None, ""):
        end_mb = _first(lowered, "end_mb", "nuclear_end_mb")
        start = _float(start_mb) * 1_000_000
        end = _float(end_mb) * 1_000_000 if end_mb not in (None, "") else start
    elif _first(lowered, "nuclear_start") not in (None, ""):
        start = _float(lowered["nuclear_start"])
        end_value = _first(lowered, "nuclear_end")
        end = _float(end_value) if end_value not in (None, "") else start
    else:
        start = _float(_first(lowered, "start", "sstart", "position", "pos"))
        end_value = _first(lowered, "end", "send")
        end = _float(end_value) if end_value not in (None, "") else start
        if max(abs(start), abs(end)) <= 1000:
            start, end = start * 1_000_000, end * 1_000_000
    if end < start:
        start, end = end, start
    length = _first(lowered, "length_bp", "length")
    length_bp = _float(length) if length not in (None, "") else 0.0
    if length_bp <= 0:
        length_mb = _first(lowered, "length_mb")
        length_bp = _float(length_mb) * 1_000_000 if length_mb not in (None, "") else 0.0
    if length_bp <= 0:
        length_bp = end - start + 1
    identity = _first(lowered, "identity", "pident")
    evidence = _first(lowered, "evidence")
    return {
        "seqid": str(seqid),
        "start": round(start),
        "end": round(end),
        "length": round(length_bp),
        "identity": None if identity in (None, "") else round(_float(identity), 3),
        "evidence": str(evidence) if evidence else evidence_category(row),
        "organelle_gene": str(_first(lowered, "organelle_gene") or ""),
        "nuclear_gene": str(_first(lowered, "nuclear_gene") or ""),
    }


def normalize_chromosomes(
    chromosomes: Iterable[Mapping[str, Any]],
    *,
    order: Sequence[str] | None = None,
    centromeres: Mapping[str, Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    """Order chromosome rows naturally (or by ``order``) and attach centromeres.

    Rows carry ``name``, ``length_bp`` and optional ``centromere`` ``[start,
    end]`` in bp. ``order`` both orders and selects the chromosomes.
    """
    rows = {str(row["name"]): dict(row) for row in chromosomes}
    for name, span in (centromeres or {}).items():
        if str(name) in rows and span is not None and len(span) >= 2:
            low, high = sorted((_float(span[0]), _float(span[1])))
            rows[str(name)]["centromere"] = [int(low), int(high)]
    if order is not None:
        names = [str(name) for name in order if str(name) in rows]
    else:
        names = sorted(rows, key=chromosome_sort_key)
    return [rows[name] for name in names]


def split_placed(
    rows: Iterable[dict[str, Any]], chromosome_names: Iterable[str]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Return (rows on drawn chromosomes, Counter of seqids that are not drawn)."""
    names = set(chromosome_names)
    placed: list[dict[str, Any]] = []
    unplaced: Counter[str] = Counter()
    for row in rows:
        if row["seqid"] in names:
            placed.append(row)
        else:
            unplaced[row["seqid"]] += 1
    return placed, unplaced


def read_sequence_lengths(fasta: str | Path) -> dict[str, int]:
    """Sequence lengths from ``<fasta>.fai`` when present, else by parsing."""
    path = Path(fasta)
    fai = path.with_name(path.name + ".fai")
    lengths: dict[str, int] = {}
    if fai.is_file():
        for line in fai.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) >= 2 and fields[1].isdigit():
                lengths[fields[0]] = int(fields[1])
        if lengths:
            return lengths
    from .._bio import read_fasta

    for name, seq in read_fasta(path):
        lengths[name] = len(seq)
    return lengths


def evidence_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("evidence", "candidate")) for row in rows)
    return {key: counts[key] for key in EVIDENCE_ORDER if counts.get(key)}


def resolve_gene_annotation(
    sets: Sequence[Mapping[str, Any]], nuclear_annotated: bool | None
) -> bool:
    """Whether the candidates carry nuclear-gene annotation.

    ``None`` infers it: an empty ``nuclear_gene`` is ambiguous (no annotation
    or intergenic), so annotation is assumed only when some candidate names a
    nuclear gene. Pass ``True`` when an annotation was used but every insertion
    is intergenic.
    """
    if nuclear_annotated is not None:
        return bool(nuclear_annotated)
    return any(row.get("nuclear_gene") for item in sets for row in item["rows"])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_transfer_figure(spec: Mapping[str, Any], output: str | Path) -> Path:
    """Render a prepared transfer figure ``spec`` at printed size (see module docstring)."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with publication_style():
        fig = _Painter(spec).draw()
        try:
            # No bbox_inches="tight": the figure is already sized to its content,
            # and tight cropping would remap the inch-anchored legends.
            fig.savefig(out)
        finally:
            plt.close(fig)
    return out


def density_peaks(spec: Mapping[str, Any]) -> dict[str, int]:
    """Largest insertion count per density window for each transfer set."""
    window = max(1_000, int(spec.get("density_window") or 500_000))
    peaks = {}
    for item in spec["sets"]:
        by_chrom: dict[str, list[float]] = {}
        for row in item["rows"]:
            by_chrom.setdefault(row["seqid"], []).append((row["start"] + row["end"]) / 2)
        peak = 0
        for chrom in spec["chromosomes"]:
            mids = np.asarray(by_chrom.get(chrom["name"], ()), dtype=float)
            if mids.size:
                n_bins = max(1, int(np.ceil(chrom["length_bp"] / window)))
                bins = np.clip((mids // window).astype(int), 0, n_bins - 1)
                peak = max(peak, int(np.bincount(bins).max()))
        peaks[item["transfer"]] = peak
    return peaks


def transfer_caption(spec: Mapping[str, Any]) -> str:
    """Figure-legend text for a tracks/ideogram ``spec`` (kept out of the plot)."""
    horizontal = spec.get("orientation", "horizontal") == "horizontal"
    transfers = [item["transfer"] for item in spec["sets"]]
    sides = {
        "NUMT": "above" if horizontal else "left of",
        "NUPT": "below" if horizontal else "right of",
    }
    where = "; ".join(f"{t} {sides[t]} the chromosome" for t in transfers)
    window = _format_window(max(1_000, int(spec.get("density_window") or 500_000)))
    parts = [
        f"Each tick is one insertion ({where}); tick length and width give the fragment "
        "length (<0.5, 0.5-2, 2-10, >=10 kb)."
    ]
    if spec.get("show_density", True):
        peaks = density_peaks(spec)
        darkest = " / ".join(f"{peaks[t]:,} {t}" for t in transfers)
        parts.append(
            f"Chromosome shading: insertions per {window} (square-root scale; darkest = {darkest})."
        )
    rows = [row for item in spec["sets"] for row in item["rows"]]
    if rows and all(row.get("evidence") == "candidate" for row in rows):
        parts.append(
            "Insertions are BLAST candidates (alignment only; not validated by read depth or "
            "long reads)."
        )
    if spec.get("label_top"):
        parts.append(f"Numbers/labels mark the {spec['label_top']} longest insertions.")
    if spec.get("note"):
        parts.append(str(spec["note"]).replace("n = ", "Drawn: ", 1) + ".")
    return " ".join(parts)


def transfer_panel(spec: Mapping[str, Any]) -> PanelDrawer:
    """Panel form of a tracks/ideogram ``spec`` for :func:`compose_figure`."""
    n = len(spec["chromosomes"])
    combined = len({item["transfer"] for item in spec["sets"]}) > 1
    if spec.get("orientation", "horizontal") == "horizontal":
        pitch = (0.3 if combined else 0.22) + (0.05 if spec.get("label_top") else 0.0)
        aspect = min(1.4, (n * pitch + 0.35) / 3.5)
    else:
        aspect = 0.62

    def draw(ax: Any, legend: bool) -> list[Any]:
        return _Painter(spec, ax=ax).draw_panel(legend)

    return PanelDrawer(draw, aspect=aspect, header=str(spec.get("title") or ""))


class _Painter:
    """Lay out and draw one figure; ``along`` = chromosome axis, ``across`` = rows.

    Everything outside the data axes (legends, colour bars, note, title) is
    positioned in inches from the figure's provisional bottom-left corner via
    ``self.inch``; :meth:`_finalize` then grows the figure to fit and shifts
    that origin, so nothing is clipped and ``figure_width`` stays exact.
    """

    def __init__(self, spec: Mapping[str, Any], ax: Any = None) -> None:
        self.spec = spec
        self.panel_ax = ax
        self.horizontal = spec.get("orientation", "horizontal") == "horizontal"
        self.chroms = list(spec["chromosomes"])
        names = {row["name"] for row in self.chroms}
        # only insertions on drawn chromosomes take part (counts, scales, legends)
        self.sets = [
            {**item, "rows": [row for row in item["rows"] if row["seqid"] in names]}
            for item in spec["sets"]
            if item.get("rows") is not None
        ]
        self.combined = len({item["transfer"] for item in self.sets}) > 1
        self.window = max(1_000, int(spec.get("density_window") or 500_000))
        self.show_density = bool(spec.get("show_density", True))
        self.label_top = max(0, int(spec.get("label_top") or 0))
        self.gene_classes = bool(spec.get("gene_classes", False))
        max_bp = spec.get("max_bp")
        longest = max(row["length_bp"] for row in self.chroms)
        self.max_mb = (float(max_bp) if max_bp else longest) / 1_000_000
        self.color_mode = self._resolve_color_mode(str(spec.get("color_by") or "evidence_level"))
        self.header_top = 0.0
        self.footer_bottom = 0.0
        self.bar_half = 0.13
        self.tick_gap = 0.035
        self.tick_max = 0.34 if self.horizontal else 0.26
        self.callouts: list[tuple[int, str]] = []
        # printed point sizes (style.FONT_PT): never scaled with the canvas
        self.font = FONT_PT["body"]
        self.tick_font = FONT_PT["tick"]
        self.label_font = _LABEL_PT

    # -- colours -----------------------------------------------------------
    def _resolve_color_mode(self, requested: str) -> str:
        mode = requested if requested in COLOR_MODES else "evidence_level"
        if mode == "evidence_level":
            present = {row["evidence"] for item in self.sets for row in item["rows"]}
            if present <= {"candidate", "unspecified"}:
                return "organelle"
        if mode == "identity" and not any(
            row.get("identity") is not None for item in self.sets for row in item["rows"]
        ):
            return "organelle"
        if mode == "organelle_gene" and not any(
            row.get("organelle_gene") for item in self.sets for row in item["rows"]
        ):
            return "organelle"
        return mode

    def _prepare_colors(self) -> None:
        rows = [row for item in self.sets for row in item["rows"]]
        self.identity_norm = None
        self.gene_colors: dict[str, str] = {}
        if self.color_mode == "identity":
            values = [row["identity"] for row in rows if row.get("identity") is not None]
            low = min(values) if values else 80.0
            low = float(np.floor(min(low, 99.0)))
            self.identity_norm = Normalize(vmin=low, vmax=100.0)
            self.identity_cmap = plt.get_cmap("viridis")
        elif self.color_mode == "organelle_gene":
            counts = Counter(row["organelle_gene"] for row in rows if row.get("organelle_gene"))
            for index, (gene, _count) in enumerate(counts.most_common(len(_GENE_PALETTE) - 1)):
                self.gene_colors[gene] = _GENE_PALETTE[index]

    def _row_color(self, row: Mapping[str, Any], transfer: str) -> tuple[float, ...]:
        if self.color_mode == "evidence_level":
            color = EVIDENCE_COLORS.get(row["evidence"], TRANSFER_COLORS[transfer])
            return to_rgba(color, _TICK_ALPHA)
        if self.color_mode == "identity" and self.identity_norm is not None:
            value = row.get("identity")
            if value is None:
                return to_rgba(_FAINT, _TICK_ALPHA)
            return tuple(self.identity_cmap(self.identity_norm(value), alpha=0.75))
        if self.color_mode == "organelle_gene":
            gene = row.get("organelle_gene") or ""
            if not gene:
                return to_rgba("#C9CED6", 0.45)
            return to_rgba(self.gene_colors.get(gene, "#6B7280"), _TICK_ALPHA + 0.1)
        return to_rgba(TRANSFER_COLORS[transfer], _TICK_ALPHA)

    # -- geometry ----------------------------------------------------------
    def _xy(self, along: Any, across: Any) -> tuple[Any, Any]:
        return (along, across) if self.horizontal else (across, along)

    def _layout(self) -> tuple[Any, Any]:
        n = len(self.chroms)
        width = self.spec.get("figure_width")
        height = self.spec.get("figure_height")
        # printed size: width is a journal width (default double column, 183 mm)
        # in inches or a name; ``height`` sets the data area and the whole
        # figure stays within the 170 mm page height
        fig_w = width_inches(width) if isinstance(width, str) else float(width or 0)
        fig_w = fig_w or width_inches("double")
        max_ax_h = MAX_HEIGHT_MM / MM_PER_INCH - 1.1  # leave room for legends and note
        if self.horizontal:
            pitch = 0.36 if self.combined else 0.27
            if self.label_top:
                pitch += 0.06
            left, right, top, bottom = 0.5, 0.42, 0.05, 0.42
            ax_h = min(pitch * n, max_ax_h) if not height else max(1.0, float(height))
        else:
            left, right, top, bottom = 0.5, 0.08, 0.3, 0.1
            ax_h = min(3.4, max_ax_h) if not height else max(1.5, float(height))
        ax_w = fig_w - left - right
        fig_h = ax_h + top + bottom
        fig = plt.figure(figsize=(fig_w, fig_h))
        self.fig = fig
        self.fig_w, self.fig_h = fig_w, fig_h
        self.inch_offset = Affine2D()
        self.inch = self.inch_offset + fig.dpi_scale_trans
        self.extent = [0.0, fig_h]  # lowest / highest content, inches
        self.axes_boxes: list[tuple[Any, tuple[float, float, float, float]]] = []
        ax = self._add_axes((left, bottom, ax_w, ax_h))
        self.ax_box = (left, bottom, ax_w, ax_h)
        self._set_limits(ax, ax_w, ax_h)
        return fig, ax

    def _layout_panel(self) -> Any:
        """Draw into the composer's axes; its size in inches sets the scales.

        The composer's box is the whole panel, so the data area is inset to
        keep tick labels and axis titles inside it.
        """
        ax = self.panel_ax
        self.fig = ax.figure
        if self.horizontal:
            _inset_panel_axes(ax, bottom=0.34, right=0.32)
        else:
            _inset_panel_axes(ax, bottom=0.16)
        fig_w, fig_h = self.fig.get_size_inches()
        box = ax.get_position()
        self.ax_box = (box.x0 * fig_w, box.y0 * fig_h, box.width * fig_w, box.height * fig_h)
        self._set_limits(ax, box.width * fig_w, box.height * fig_h)
        return ax

    def _set_limits(self, ax: Any, ax_w: float, ax_h: float) -> None:
        n = len(self.chroms)
        pad = 0.62 if self.combined or not self.horizontal else 0.58
        across_lim = (-pad, n - 1 + pad)
        along_lim = (-self.max_mb * (0.004 if self.horizontal else 0.035), self.max_mb * 1.01)
        if self.horizontal:
            ax.set_xlim(*along_lim)
            ax.set_ylim(across_lim[1], across_lim[0])
            self.along_per_in = (along_lim[1] - along_lim[0]) / ax_w
            self.across_per_in = (across_lim[1] - across_lim[0]) / ax_h
        else:
            ax.set_xlim(*across_lim)
            ax.set_ylim(along_lim[1], along_lim[0])
            self.along_per_in = (along_lim[1] - along_lim[0]) / ax_h
            self.across_per_in = (across_lim[1] - across_lim[0]) / ax_w

    def _add_axes(self, box: tuple[float, float, float, float]) -> Any:
        x, y, w, h = box
        ax = self.fig.add_axes((x / self.fig_w, y / self.fig_h, w / self.fig_w, h / self.fig_h))
        self.axes_boxes.append((ax, box))
        self._reach(y, y + h)
        return ax

    def _reach(self, low: float, high: float) -> None:
        self.extent[0] = min(self.extent[0], low)
        self.extent[1] = max(self.extent[1], high)

    def _text(self, x: float, y: float, text: str, **kw: Any) -> None:
        """Figure text at (x, y) inches; extent assumes one line of ``fontsize``."""
        artist = self.fig.text(x, y, text, transform=self.inch, **kw)
        size = float(kw.get("fontsize", self.font)) * 1.3 / 72
        va = kw.get("va", "baseline")
        low = y - size if va == "top" else (y - size / 2 if va == "center" else y)
        self._reach(low, low + size)
        return artist

    def _finalize(self) -> None:
        """Grow the figure to the content extent and move everything with it."""
        margin = 0.05
        below = max(0.0, margin - self.extent[0])
        above = max(0.0, self.extent[1] + margin - self.fig_h)
        height = self.fig_h + below + above
        self.fig.set_size_inches(self.fig_w, height)
        self.inch_offset.clear().translate(0, below)  # applied in inches, before dpi scaling
        for ax, (x, y, w, h) in self.axes_boxes:
            ax.set_position((x / self.fig_w, (y + below) / height, w / self.fig_w, h / height))

    # -- drawing -----------------------------------------------------------
    def _draw_data(self) -> list[dict[str, Any]]:
        self._draw_axes()
        bars = self._draw_chromosomes()
        density_scales = self._draw_density(bars) if self.show_density else []
        self._draw_ticks()
        if self.gene_classes:
            self._draw_gene_classes()
        if self.label_top:
            self._draw_top_labels()
        return density_scales

    def draw(self) -> Any:
        """Stand-alone figure (call inside :func:`publication_style`)."""
        self._prepare_colors()
        fig, ax = self._layout()
        self.ax = ax
        density_scales = self._draw_data()
        self._draw_legends(density_scales)
        # A title is a left-aligned panel title on the same baseline as the
        # panel letter (style guide: no large in-figure titles).
        _left, bottom, _w, ax_h = self.ax_box
        top_in = self.header_top + 0.04 if self.horizontal else bottom + ax_h + 0.2
        pad_pt = (top_in - bottom - ax_h) * 72
        if self.spec.get("title"):
            ax.set_title(str(self.spec["title"]), loc="left", fontsize=FONT_PT["body"], pad=pad_pt)
        if self.spec.get("panel_label"):
            add_panel_label(ax, str(self.spec["panel_label"]), y_pt=pad_pt)
        if self.spec.get("title") or self.spec.get("panel_label"):
            self._reach(top_in, top_in + FONT_PT["panel"] * 1.3 / 72)
        self._finalize()
        return fig

    def draw_panel(self, legend: bool) -> list[Any]:
        """Draw into the given axes; return legend handles unless ``legend``."""
        self._prepare_colors()
        self.ax = self._layout_panel()
        density_scales = self._draw_data()
        handles = self._legend_handles(density_scales, panel=True)
        if legend:
            self.ax.legend(handles=handles, loc="lower right", fontsize=FONT_PT["body"])
            return []
        return handles

    def _draw_axes(self) -> None:
        ax = self.ax
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        locator = MaxNLocator(nbins=8 if self.horizontal else 7, steps=[1, 2, 2.5, 5, 10])
        formatter = FuncFormatter(lambda value, _pos: f"{value:g}")
        names = [chromosome_display(row["name"]) for row in self.chroms]
        positions = list(range(len(self.chroms)))
        if self.horizontal:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.set_xlabel("Position (Mb)", fontsize=self.font, color=_INK, labelpad=2)
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_bounds(0, self.max_mb)
            ax.spines["bottom"].set_color(_MUTED)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.tick_params(
                axis="x", length=2.5, width=0.6, colors=_MUTED, labelsize=self.tick_font, pad=1.5
            )
            ax.set_yticks(positions, names)
            ax.tick_params(axis="y", length=0, pad=3, labelsize=self.tick_font, labelcolor=_INK)
        else:
            ax.yaxis.set_major_locator(locator)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_ylabel("Position (Mb)", fontsize=self.font, color=_INK, labelpad=3)
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_bounds(0, self.max_mb)
            ax.spines["left"].set_color(_MUTED)
            ax.spines["left"].set_linewidth(0.6)
            ax.tick_params(
                axis="y", length=2.5, width=0.6, colors=_MUTED, labelsize=self.tick_font, pad=1.5
            )
            if self.panel_ax is None:
                ax.xaxis.tick_top()  # panel titles take the top in a composite
            ax.set_xticks(positions, names)
            ax.tick_params(axis="x", length=0, pad=2, labelsize=self.tick_font, labelcolor=_INK)

    def _bar_patch(self, index: int, length_mb: float, *, fill: str, edge: str, z: float) -> Any:
        half = self.bar_half
        x_per_in = self.along_per_in if self.horizontal else self.across_per_in
        y_per_in = self.across_per_in if self.horizontal else self.along_per_in
        across_in = half / self.across_per_in
        radius_x = across_in * x_per_in
        (x0, y0) = self._xy(0.0, index - half)
        (w, h) = self._xy(length_mb, 2 * half)
        patch = FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius_x}",
            mutation_aspect=y_per_in / x_per_in,
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.6,
            zorder=z,
        )
        self.ax.add_patch(patch)
        return patch

    def _draw_chromosomes(self) -> list[Any]:
        bars = []
        for index, row in enumerate(self.chroms):
            length_mb = min(row["length_bp"] / 1_000_000, self.max_mb)
            bars.append(self._bar_patch(index, length_mb, fill=_BAR_FILL, edge="none", z=1))
            self._bar_patch(index, length_mb, fill="none", edge=_BAR_EDGE, z=4)
            centromere = row.get("centromere")
            if centromere:
                self._draw_centromere(index, centromere)
            if self.horizontal:
                self.ax.text(
                    length_mb + self.max_mb * 0.008,
                    index,
                    f"{row['length_bp'] / 1_000_000:.1f} Mb",
                    ha="left",
                    va="center",
                    fontsize=self.tick_font,
                    color=_FAINT,
                )
        return bars

    def _draw_centromere(self, index: int, span: Sequence[float]) -> None:
        low, high = (_float(span[0]) / 1_000_000, _float(span[1]) / 1_000_000)
        pinch = max(high - low, self.max_mb * 0.012)
        mid = (low + high) / 2
        half = self.bar_half
        for sign in (-1, 1):
            edge = index + sign * half
            points = [
                self._xy(mid - pinch / 2, edge + sign * 0.01),
                self._xy(mid, index + sign * half * 0.35),
                self._xy(mid + pinch / 2, edge + sign * 0.01),
            ]
            self.ax.add_patch(
                Polygon(points, closed=True, facecolor="white", edgecolor="none", zorder=4.5)
            )
            xs, ys = zip(*points, strict=True)
            self.ax.plot(xs, ys, color=_BAR_EDGE, lw=0.6, zorder=4.6, solid_joinstyle="miter")

    def _density_cmap(self, transfer: str) -> Any:
        # organelle hues only when the ticks use them too; otherwise a neutral
        # grey keeps the fill from competing with the evidence/identity colours
        color = TRANSFER_COLORS[transfer] if self.color_mode == "organelle" else "#4B5563"
        return LinearSegmentedColormap.from_list(f"density_{transfer}", ["#FFFFFF", color])

    def _draw_density(self, bars: Sequence[Any]) -> list[dict[str, Any]]:
        scales: list[dict[str, Any]] = []
        transfers = [item["transfer"] for item in self.sets]
        for item in self.sets:
            transfer = item["transfer"]
            by_chrom: dict[str, list[float]] = {}
            for row in item["rows"]:
                by_chrom.setdefault(row["seqid"], []).append((row["start"] + row["end"]) / 2)
            counts_by_index: list[np.ndarray] = []
            for chrom in self.chroms:
                n_bins = max(1, int(np.ceil(chrom["length_bp"] / self.window)))
                mids = np.asarray(by_chrom.get(chrom["name"], ()), dtype=float)
                bins = np.clip((mids // self.window).astype(int), 0, n_bins - 1)
                counts_by_index.append(np.bincount(bins, minlength=n_bins))
            peak = max((int(c.max()) for c in counts_by_index if c.size), default=0)
            if peak <= 0:
                continue
            cmap = self._density_cmap(transfer)
            norm = PowerNorm(gamma=0.5, vmin=0, vmax=peak)  # one hotspot must not wash out the rest
            if len(transfers) > 1:
                lo, hi = (
                    (-self.bar_half, 0.0) if TRANSFER_SIDES[transfer] < 0 else (0.0, self.bar_half)
                )
            else:
                lo, hi = -self.bar_half, self.bar_half
            for index, counts in enumerate(counts_by_index):
                nonzero = np.flatnonzero(counts)
                if not nonzero.size:
                    continue
                length_mb = self.chroms[index]["length_bp"] / 1_000_000
                polys = []
                for b in nonzero:
                    a0 = b * self.window / 1_000_000
                    a1 = min((b + 1) * self.window / 1_000_000, length_mb)
                    polys.append(
                        [
                            self._xy(a0, index + lo),
                            self._xy(a1, index + lo),
                            self._xy(a1, index + hi),
                            self._xy(a0, index + hi),
                        ]
                    )
                colors = cmap(norm(counts[nonzero]) * 0.92 + 0.08)
                collection = PolyCollection(
                    polys, facecolors=colors, edgecolors="none", linewidths=0, zorder=2
                )
                collection.set_clip_path(bars[index])
                self.ax.add_collection(collection)
            scales.append({"transfer": transfer, "cmap": cmap, "peak": peak})
        return scales

    def _tick_geometry(self, length_bp: float) -> tuple[float, float]:
        rel, width = LENGTH_BINS[0][2], LENGTH_BINS[0][3]
        for threshold, _label, bin_rel, bin_width in LENGTH_BINS:
            if length_bp >= threshold:
                rel, width = bin_rel, bin_width
        return rel, width

    def _draw_ticks(self) -> None:
        index_of = {row["name"]: index for index, row in enumerate(self.chroms)}
        for item in self.sets:
            transfer = item["transfer"]
            side = TRANSFER_SIDES[transfer]
            base = self.bar_half + self.tick_gap
            rows = sorted(item["rows"], key=lambda row: row["length"])
            segments, colors, widths = [], [], []
            for row in rows:
                index = index_of.get(row["seqid"])
                if index is None:
                    continue
                along = min((row["start"] + row["end"]) / 2 / 1_000_000, self.max_mb)
                rel, width = self._tick_geometry(row["length"])
                inner = index + side * base
                outer = index + side * (base + self.tick_max * rel)
                segments.append([self._xy(along, inner), self._xy(along, outer)])
                colors.append(self._row_color(row, transfer))
                widths.append(width)
            if segments:
                self.ax.add_collection(
                    LineCollection(
                        segments,
                        colors=colors,
                        linewidths=widths,
                        capstyle="butt",
                        zorder=3,
                    )
                )

    def _gene_class_points(self) -> list[tuple[int, int, float, str]]:
        """(chromosome index, side, position bp, class) for every gene marker.

        Explicit ``gene_hits`` rows (legacy ideogram input) win over classes
        derived from the candidates' ``nuclear_gene``.
        """
        index_of = {row["name"]: index for index, row in enumerate(self.chroms)}
        hits = self.spec.get("gene_hits") or ()
        if hits:
            side = TRANSFER_SIDES[self.sets[0]["transfer"]] if self.sets else -1
            return [
                (index_of[hit["seqid"]], side, float(hit["position"]), hit["class"])
                for hit in hits
                if hit["seqid"] in index_of
            ]
        return [
            (
                index_of[row["seqid"]],
                TRANSFER_SIDES[item["transfer"]],
                (row["start"] + row["end"]) / 2,
                _gene_class(row),
            )
            for item in self.sets
            for row in item["rows"]
            if row["seqid"] in index_of
        ]

    def _draw_gene_classes(self) -> None:
        offset = self.bar_half + self.tick_gap + self.tick_max + 0.05
        points = self._gene_class_points()
        for gene_class, style in (
            ("within_gene", {"facecolors": _INK, "edgecolors": "none"}),
            ("intergenic", {"facecolors": "none", "edgecolors": _FAINT}),
        ):
            coords = [
                self._xy(position / 1_000_000, index + side * offset)
                for index, side, position, kind in points
                if kind == gene_class
            ]
            if coords:
                xs, ys = zip(*coords, strict=True)
                self.ax.scatter(xs, ys, s=3.5, linewidths=0.4, zorder=3.5, **style)

    def _draw_top_labels(self) -> None:
        """Label the ``label_top`` longest insertions.

        Tracks get text (length and organelle gene). A vertical ideogram has no
        room between chromosomes, so it gets numbers listed in a legend,
        stacked in a column at a cluster. On tracks, insertions closer than one
        label merge into one (``22.9, 19.0 kb · nad1``) instead of fanning out
        leader lines that would cross the neighbouring labels.
        """
        index_of = {row["name"]: index for index, row in enumerate(self.chroms)}
        pool = [
            (row, item["transfer"])
            for item in self.sets
            for row in item["rows"]
            if row["seqid"] in index_of
        ]
        pool.sort(key=lambda pair: pair[0]["length"], reverse=True)
        fontsize = self.label_font
        line_in = fontsize * 1.25 / 72
        char_in = fontsize * 0.6 / 72
        groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for rank, (row, transfer) in enumerate(pool[: self.label_top], start=1):
            gene = row.get("organelle_gene") or ""
            if not self.horizontal:
                where = chromosome_display(row["seqid"])
                text = _format_length(row["length"]) + (f" · {gene}" if gene else "")
                self.callouts.append((rank, f"{text} · {transfer} · {where}"))
            groups.setdefault((index_of[row["seqid"]], TRANSFER_SIDES[transfer]), []).append(
                {
                    "along": (row["start"] + row["end"]) / 2 / 1_000_000,
                    "rank": rank,
                    "length": row["length"],
                    "gene": gene,
                }
            )
        near = (0.3 if self.horizontal else line_in) * self.along_per_in
        for (index, side), entries in groups.items():
            entries.sort(key=lambda entry: entry["along"])
            clusters: list[list[dict[str, Any]]] = []
            for entry in entries:
                # vertical numbers stay separate: a merged "1, 2, 3" is wider than
                # the gap between chromosomes; they stack in a column instead
                if (
                    self.horizontal
                    and clusters
                    and entry["along"] - clusters[-1][-1]["along"] < near
                ):
                    clusters[-1].append(entry)
                else:
                    clusters.append([entry])
            texts = [self._cluster_text(cluster) for cluster in clusters]
            centres = [sum(e["along"] for e in c) / len(c) for c in clusters]
            anchor = index + side * (self.bar_half + self.tick_gap + self.tick_max + 0.02)
            if self.horizontal:
                widths = [(len(text) * char_in + 0.08) * self.along_per_in for text in texts]
            else:
                widths = [line_in * self.along_per_in] * len(texts)
            placed = _spread(centres, widths)
            for text, centre, pos, width in zip(texts, centres, placed, widths, strict=True):
                moved = abs(pos - centre) > width * 0.05
                if self.horizontal:
                    self._label(
                        (pos, anchor + side * (0.05 if moved else 0.01)),
                        (centre, anchor),
                        text,
                        leader=moved,
                        ha="center",
                        va="bottom" if side < 0 else "top",
                    )
                else:
                    self._label(
                        (anchor, pos),
                        (anchor, centre),
                        text,
                        leader=False,  # leaders of a stacked column would cross numbers
                        boxed=False,
                        fontweight="bold",
                        ha="right" if side < 0 else "left",
                        va="center",
                    )

    def _cluster_text(self, cluster: Sequence[Mapping[str, Any]]) -> str:
        if not self.horizontal:
            return ", ".join(
                str(entry["rank"]) for entry in sorted(cluster, key=lambda e: e["rank"])
            )
        by_gene: dict[str, list[str]] = {}
        for entry in sorted(cluster, key=lambda e: -e["length"]):
            by_gene.setdefault(entry["gene"], []).append(f"{entry['length'] / 1000:.1f}")
        parts = []
        for gene, sizes in by_gene.items():
            part = ", ".join(sizes) + " kb"
            parts.append(f"{part} · {gene}" if gene else part)
        return "; ".join(parts)

    def _label(
        self,
        text_at: tuple[float, float],
        point: tuple[float, float],
        text: str,
        *,
        leader: bool,
        boxed: bool = True,
        **kw: Any,
    ) -> None:
        """``text_at`` and ``point`` are (x, y) data coordinates.

        A leader line is drawn only for labels moved off their insertion; a
        stub next to an unmoved label would read as a minus sign.
        """
        # an unboxed callout keeps its leader off the glyph
        shrink = 0 if boxed else 3.5
        arrow = {"arrowstyle": "-", "color": _MUTED, "lw": 0.4, "shrinkA": shrink, "shrinkB": 0}
        self.ax.annotate(
            text,
            xy=point,
            xytext=text_at,
            fontsize=self.label_font,
            color=_INK,
            zorder=6,
            arrowprops=arrow if leader else None,
            bbox=None
            if not boxed
            else {"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85},
            **kw,
        )

    # -- legends -----------------------------------------------------------
    def _color_handles(self, *, counts: bool = True) -> tuple[str | None, list[Any]]:
        handles: list[Any] = []
        n_by_transfer = {item["transfer"]: len(item["rows"]) for item in self.sets}
        if self.color_mode == "organelle":
            for item in self.sets:
                transfer = item["transfer"]
                every_candidate = all(row["evidence"] == "candidate" for row in item["rows"])
                suffix = " candidate (alignment only)" if every_candidate else ""
                if counts:
                    label = f"{transfer}{suffix} · {TRANSFER_ORGANELLES[transfer]}"
                    label += f" (n = {n_by_transfer[transfer]:,})"
                else:  # a composite's shared legend stays short; the caption explains
                    label = transfer
                handles.append(_tick_handle(TRANSFER_COLORS[transfer], label, 1.2))
            return None, handles
        if self.color_mode == "evidence_level":
            rows = [row for item in self.sets for row in item["rows"]]
            for key, count in evidence_counts(rows).items():
                color = EVIDENCE_COLORS.get(key, TRANSFER_COLORS[self.sets[0]["transfer"]])
                handles.append(_tick_handle(color, f"{EVIDENCE_LABELS[key]} ({count:,})", 1.2))
            return "Evidence", handles
        if self.color_mode == "organelle_gene":
            rows = [row for item in self.sets for row in item["rows"]]
            counts = Counter(row["organelle_gene"] for row in rows if row.get("organelle_gene"))
            for gene, color in self.gene_colors.items():
                handles.append(_tick_handle(color, f"{gene} ({counts[gene]:,})", 1.2))
            other = sum(v for g, v in counts.items() if g not in self.gene_colors)
            if other:
                handles.append(_tick_handle("#6B7280", f"Other genes ({other:,})", 1.2))
            missing = sum(1 for row in rows if not row.get("organelle_gene"))
            if missing:
                handles.append(_tick_handle("#C9CED6", f"No organelle gene ({missing:,})", 1.2))
            return "Organelle gene", handles
        return None, handles

    def _callout_handles(self) -> list[Any]:
        return [
            Line2D(
                [0],
                [0],
                marker=f"${rank}$",
                linestyle="none",
                markersize=6.5,
                markeredgewidth=0,
                color=_INK,
                label=text,
            )
            for rank, text in self.callouts
        ]

    def _legend_handles(
        self, density_scales: Sequence[Mapping[str, Any]], *, panel: bool
    ) -> list[Any]:
        """Flat legend handles for a composed figure (one merged legend).

        Colour bars cannot be merged, so density and identity scales become
        three swatches each (low / middle / high).
        """
        _title, handles = self._color_handles(counts=not panel)
        present = {self._tick_bin(row["length"]) for item in self.sets for row in item["rows"]}
        handles += [
            _tick_handle(_MUTED, label, width, rel)
            for index, (_t, label, rel, width) in enumerate(LENGTH_BINS)
            if index in present
        ]
        if self.gene_classes:
            classes = {kind for _i, _s, _p, kind in self._gene_class_points()}
            if "within_gene" in classes:
                handles.append(_dot_handle(_INK, "none", GENE_CLASS_LABELS["within_gene"]))
            if "intergenic" in classes:
                handles.append(_dot_handle("none", _FAINT, GENE_CLASS_LABELS["intergenic"]))
        # density shading is described in the caption (metrics["caption"]), not
        # in a composite's shared legend
        for scale in () if panel else density_scales:
            peak = scale["peak"]
            norm = PowerNorm(gamma=0.5, vmin=0, vmax=peak)
            for value in sorted({max(1, round(peak / 10)), max(1, round(peak / 2)), peak}):
                handles.append(
                    Patch(
                        facecolor=scale["cmap"](norm(value) * 0.92 + 0.08),
                        edgecolor=_FAINT,
                        lw=0.3,
                        label=f"{value:,} {scale['transfer']} per {_format_window(self.window)}",
                    )
                )
        handles += self._callout_handles()
        if self.color_mode == "identity" and self.identity_norm is not None:
            low = self.identity_norm.vmin
            for value in (low, (low + 100) / 2, 100.0):
                handles.append(
                    _tick_handle(
                        to_hex(self.identity_cmap(self.identity_norm(value))),
                        f"Identity {value:g}%",
                        1.2,
                    )
                )
        return handles

    def _draw_legends(self, density_scales: Sequence[Mapping[str, Any]]) -> None:
        """Measure every legend block, then flow them into rows.

        Tracks stack the rows upwards from just above the axes; the ideogram
        stacks them downwards below the chromosomes (its top holds the names).
        """
        fig = self.fig
        left, bottom, ax_w, ax_h = self.ax_box
        renderer = fig.canvas.get_renderer()
        blocks: list[dict[str, Any]] = []

        def add_legend(title: str | None, handles: list[Any], ncol: int = 1) -> None:
            if not handles:
                return
            legend = fig.legend(
                handles=handles,
                title=title,
                loc="upper left",
                bbox_to_anchor=(0, 0),
                frameon=False,
                fontsize=FONT_PT["body"],
                title_fontsize=FONT_PT["body"],
                ncol=ncol,
                handlelength=1.1,
                handleheight=1.0,
                handletextpad=0.45,
                columnspacing=1.0,
                labelspacing=0.25,
                borderaxespad=0.0,
                borderpad=0.0,
                alignment="left",
            )
            box = legend.get_window_extent(renderer)
            blocks.append({"legend": legend, "w": box.width / fig.dpi, "h": box.height / fig.dpi})

        title, handles = self._color_handles()
        add_legend(title, handles, ncol=1 if len(handles) <= 3 else 2)
        present = {self._tick_bin(row["length"]) for item in self.sets for row in item["rows"]}
        length_handles = [
            _tick_handle(_MUTED, label, width, rel)
            for index, (_t, label, rel, width) in enumerate(LENGTH_BINS)
            if index in present
        ]
        add_legend("Fragment length", length_handles, ncol=2 if len(length_handles) > 2 else 1)
        if self.gene_classes:
            classes = {kind for _i, _s, _p, kind in self._gene_class_points()}
            gene_handles = []
            if "within_gene" in classes:
                gene_handles.append(_dot_handle(_INK, "none", GENE_CLASS_LABELS["within_gene"]))
            if "intergenic" in classes:
                gene_handles.append(_dot_handle("none", _FAINT, GENE_CLASS_LABELS["intergenic"]))
            add_legend("Nuclear locus", gene_handles)
        add_legend("Longest insertions", self._callout_handles())
        density_bars = []
        for scale in density_scales:
            peak = scale["peak"]
            quarter = round(peak / 4)
            ticks = [0, peak] if peak < 8 else [0, quarter, peak]
            label = f"{scale['transfer']} per {_format_window(self.window)}"
            norm = PowerNorm(gamma=0.5, vmin=0, vmax=peak)
            density_bars.append((label, scale["cmap"], norm, ticks))
        if density_bars:
            # NUMT and NUPT scales stack in one block so they share a column
            blocks.append({"bars": density_bars, "w": 1.0, "h": 0.36 * len(density_bars)})
        if self.color_mode == "identity" and self.identity_norm is not None:
            norm = self.identity_norm
            identity_bar = ("Identity to organelle (%)", self.identity_cmap, norm, [norm.vmin, 100])
            blocks.append({"bar": identity_bar, "w": 1.1, "h": 0.36})

        rows: list[list[dict[str, Any]]] = [[]]
        used = 0.0
        for block in blocks:
            if rows[-1] and used + block["w"] > ax_w + 0.4:
                rows.append([])
                used = 0.0
            rows[-1].append(block)
            used += block["w"] + 0.22
        row_gap = 0.08
        row_tops: list[tuple[list[dict[str, Any]], float]] = []
        if self.horizontal:
            cursor = bottom + ax_h + 0.1  # the first row sits nearest the tracks
            for row in rows:
                top = cursor + max((b["h"] for b in row), default=0.0)
                row_tops.append((row, top))
                cursor = top + row_gap
            self.header_top = cursor
        else:
            cursor = bottom - 0.12
            for row in rows:
                row_tops.append((row, cursor))
                cursor -= max((b["h"] for b in row), default=0.0) + row_gap
            self.footer_bottom = cursor
        for row, top in row_tops:
            x = left
            for block in row:
                if "legend" in block:
                    block["legend"].set_bbox_to_anchor((x, top), transform=self.inch)
                    self._reach(top - block["h"], top)
                else:
                    for i, bar in enumerate(block.get("bars") or [block["bar"]]):
                        self._colorbar(x, top - 0.36 * i, block["w"] - 0.1, *bar)
                    self._reach(top - block["h"], top)
                x += block["w"] + 0.22
        note = self.spec.get("note")
        if note:
            note_y = bottom - 0.34 if self.horizontal else self.footer_bottom + 0.02
            self._text(left, note_y, str(note), fontsize=self.tick_font, color=_MUTED, va="top")

    def _colorbar(
        self, x: float, top: float, width: float, label: str, cmap: Any, norm: Any, ticks: Any
    ) -> None:
        fig = self.fig
        cax = self._add_axes((x, top - 0.2, width, 0.065))
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        colorbar.outline.set_linewidth(0.4)
        colorbar.outline.set_edgecolor(_FAINT)
        colorbar.set_ticks(ticks)
        colorbar.ax.set_xticklabels([f"{t:g}" for t in ticks])
        cax.tick_params(labelsize=self.tick_font, length=1.8, width=0.4, pad=1, colors=_MUTED)
        cax.set_title(label, fontsize=FONT_PT["body"], pad=2, loc="left", color=_INK)

    def _tick_bin(self, length_bp: float) -> int:
        found = 0
        for index, (threshold, _label, _rel, _width) in enumerate(LENGTH_BINS):
            if length_bp >= threshold:
                found = index
        return found


# ---------------------------------------------------------------------------
# Preparation (compute-only): inputs -> JSON-able spec + metrics
# ---------------------------------------------------------------------------


def canonical_transfer(label: str | None) -> str:
    """Map a free transfer label onto ``NUMT``/``NUPT`` (side and colour)."""
    text = str(label or "").lower()
    if "nupt" in text or "plast" in text or "chloro" in text:
        return "NUPT"
    return "NUMT"


def transfer_input_rows(value: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Candidate rows plus the organelle label a transfer Result carries.

    Accepts a ``detect_transfers*`` Result (or its mapping form) or a plain
    sequence of candidate/fragment mappings.
    """
    from .suite_plots import _is_transfer_result, _transfer_candidate_rows, _transfer_result_label

    if value is None:
        return [], None
    if _is_transfer_result(value):
        return _transfer_candidate_rows(value), _transfer_result_label(value)
    return [dict(row) for row in value if isinstance(row, Mapping)], None


def build_transfer_sets(
    primary: Any,
    *,
    primary_label: str | None,
    plastid: Any = None,
) -> list[dict[str, Any]]:
    """Normalise one or two candidate inputs into labelled sets.

    With ``plastid`` given, ``primary`` is the mitochondrial (NUMT) set.
    """
    rows, detected = transfer_input_rows(primary)
    sets: list[dict[str, Any]] = []
    normalized = [r for r in (normalize_transfer_row(row) for row in rows) if r is not None]
    if plastid is not None:
        label = "NUMT"
    else:
        label = canonical_transfer(primary_label or detected or transfer_label_from_rows(rows))
    sets.append({"transfer": label, "rows": normalized})
    if plastid is not None:
        plastid_rows, _ = transfer_input_rows(plastid)
        sets.append(
            {
                "transfer": "NUPT",
                "rows": [
                    r
                    for r in (normalize_transfer_row(row) for row in plastid_rows)
                    if r is not None
                ],
            }
        )
    return sets


def placement_summary(
    sets: Sequence[Mapping[str, Any]], chromosomes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Counts of drawn vs unplaced candidates, plus a one-line figure note."""
    names = [row["name"] for row in chromosomes]
    per_set: dict[str, dict[str, Any]] = {}
    unplaced_all: Counter[str] = Counter()
    parts = []
    for item in sets:
        placed, unplaced = split_placed(item["rows"], names)
        unplaced_all.update(unplaced)
        per_set[item["transfer"]] = {
            "candidates": len(item["rows"]),
            "drawn": len(placed),
            "unplaced": sum(unplaced.values()),
            "evidence_counts": evidence_counts(item["rows"]),
        }
        parts.append(f"{len(placed):,} {item['transfer']}")
    note = f"n = {' + '.join(parts)} on {len(names)} chromosome{'s' if len(names) != 1 else ''}"
    if unplaced_all:
        total = sum(unplaced_all.values())
        note += (
            f"; {total:,} on {len(unplaced_all)} unplaced sequence"
            f"{'s' if len(unplaced_all) != 1 else ''} not drawn"
        )
    return {
        "by_transfer": per_set,
        "drawn_candidates": sum(v["drawn"] for v in per_set.values()),
        "unplaced_candidates": sum(unplaced_all.values()),
        "unplaced_seqids": dict(sorted(unplaced_all.items())),
        "note": note,
    }


def resolve_track_chromosomes(
    sets: Sequence[Mapping[str, Any]],
    *,
    nuclear_fasta: str | Path | None,
    chromosome_lengths: Mapping[str, int] | None,
    min_chromosome_fraction: float,
) -> list[dict[str, Any]]:
    """Chromosomes to draw on the tracks.

    Explicit ``chromosome_lengths`` are drawn as given (and nothing else); a
    FASTA contributes sequences at least ``min_chromosome_fraction`` of the
    longest one, so unplaced contigs and scaffolds are left out. Without either,
    every candidate seqid is drawn up to its furthest insertion.
    """
    if chromosome_lengths:
        lengths = {str(k): int(v) for k, v in chromosome_lengths.items() if int(v) > 0}
    elif nuclear_fasta is not None:
        all_lengths = read_sequence_lengths(nuclear_fasta)
        longest = max(all_lengths.values(), default=0)
        cutoff = longest * max(0.0, float(min_chromosome_fraction))
        lengths = {name: size for name, size in all_lengths.items() if size >= cutoff}
    else:
        lengths = {}
        for item in sets:
            for row in item["rows"]:
                lengths[row["seqid"]] = max(lengths.get(row["seqid"], 0), row["end"])
    if not lengths:
        raise ValueError(
            "no chromosome lengths available; pass nuclear_fasta or chromosome_lengths"
        )
    return [{"name": name, "length_bp": size} for name, size in lengths.items()]


# ---------------------------------------------------------------------------
# Per-sample summaries: length profile and functional region x TE class
# ---------------------------------------------------------------------------

# Lower bounds of the default length bins (bp); the last bin is open-ended.
LENGTH_PROFILE_BINS = (200, 400, 600, 800, 1_000, 3_000)
REGION_ORDER = ("Promoter", "Exon", "Intron", "Terminator", "Intergenic")
TE_CLASSES = (
    "LTR",
    "DNA",
    "MITE",
    "Helitron",
    "LINE",
    "SINE",
    "rRNA",
    "Simple_repeat",
    "other",
    "nonTE",
)
TE_COLORS = {
    "LTR": "#C0392B",
    "DNA": "#0072B2",
    "MITE": "#56B4E9",
    "Helitron": "#CC79A7",
    "LINE": "#E69F00",
    "SINE": "#F0C808",
    "rRNA": "#009E73",
    "Simple_repeat": "#8C6D31",
    "other": "#7B8794",
    "nonTE": "#E4E7EB",
    "not_annotated": "#BCC3CB",
}
_GROUP_PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7B5EA7")


def length_bin_labels(bins: Sequence[int] = LENGTH_PROFILE_BINS) -> list[str]:
    """Bin labels with en-dash ranges and an open last bin (200-399 ... >=3,000)."""
    edges = [int(b) for b in bins]
    labels = [f"{low:,}–{high - 1:,}" for low, high in itertools.pairwise(edges)]  # noqa: RUF001 (en dash in a range)
    return [*labels, f"≥{edges[-1]:,}"]


def transfer_length_bins(
    candidates: Any, bins: Sequence[int] = LENGTH_PROFILE_BINS
) -> dict[str, Any]:
    """Count candidates per fragment-length bin (compute-only).

    ``candidates`` are rows or a ``detect_transfers*`` Result. Fragments
    shorter than the first bin edge are reported as ``below_min``.
    """
    rows, _label = transfer_input_rows(candidates)
    edges = sorted(int(b) for b in bins)
    labels = length_bin_labels(edges)
    counts = dict.fromkeys(labels, 0)
    below = 0
    for row in rows:
        normalized = normalize_transfer_row(row)
        if normalized is None:
            continue
        length = normalized["length"]
        if length < edges[0]:
            below += 1
            continue
        counts[labels[bisect.bisect_right(edges, length) - 1]] += 1
    return {"bins": labels, "counts": counts, "below_min": below, "total": len(rows)}


def te_category(text: str) -> str:
    """Map a RepeatMasker/EDTA class string onto :data:`TE_CLASSES`."""
    t = str(text or "").strip().lower()
    if not t or t in {"nonte", "none"}:
        return "nonTE"
    if "mite" in t:
        return "MITE"
    if "helitron" in t or t.startswith("rc/") or t.startswith("dhh"):
        return "Helitron"
    if "ltr" in t or "gypsy" in t or "copia" in t or t.startswith(("rlc", "rlg", "rlx")):
        return "LTR"
    if "sine" in t:
        return "SINE"
    if t.startswith("line") or "/line" in t or "_line" in t or t.startswith(("ril", "rit", "l1")):
        return "LINE"
    dna_tokens = ("tir", "hat", "mutator", "cacta", "harbinger", "mariner", "en-spm", "pif")
    if t.startswith(("dna", "dta", "dtc", "dth", "dtm", "dtt")) or any(
        token in t for token in dna_tokens
    ):
        return "DNA"
    if "rrna" in t:
        return "rRNA"
    if "simple" in t or "low_complexity" in t:
        return "Simple_repeat"
    return "other"


def summarize_transfer_regions(
    candidates: Any,
    gene_annotation: str | Path,
    te_annotation: str | Path | None = None,
    *,
    promoter: int = 2_000,
    terminator: int = 2_000,
    te_min_fraction: float = 0.5,
    seqids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Place NUMT/NUPT candidates in nuclear functional regions and TE classes.

    Compute-only interval statistics behind :func:`plot_transfer_region_te`.

    * Region: every base of an insertion is labelled Exon > Intron > Promoter
      > Terminator > Intergenic (first match wins, so a base inside one
      gene's exon and another's promoter counts as Exon); the insertion takes
      the label covering most of its bases. Promoter/terminator are the
      ``promoter``/``terminator`` bp upstream/downstream of each gene,
      strand-aware. Exons come from ``exon`` features (CDS/UTR when there are
      none); introns are the rest of the gene body (``gene`` features, or
      mRNA/transcript spans when the GFF has no gene lines).
    * TE class: the class (:func:`te_category`) covering most bases, provided
      TEs cover at least ``te_min_fraction`` of the insertion; otherwise
      ``nonTE``. Without ``te_annotation`` every insertion is
      ``not_annotated``.

    ``gene_annotation`` is GFF3 (``.gz`` allowed). ``te_annotation`` is a
    RepeatMasker ``.out`` or a TE GFF3 (EDTA ``Classification=``, or the
    feature type). ``seqids`` restricts the statistics to those sequences.
    """
    rows, _label = transfer_input_rows(candidates)
    keep = set(seqids) if seqids is not None else None
    inserts = [
        r
        for r in (normalize_transfer_row(row) for row in rows)
        if r is not None and (keep is None or r["seqid"] in keep)
    ]
    wanted = {r["seqid"] for r in inserts}
    layers = _region_layers(Path(gene_annotation), wanted, promoter, terminator)
    te_index = _te_intervals(Path(te_annotation), wanted) if te_annotation else None
    labels = ("Exon", "Intron", "Promoter", "Terminator", "Intergenic")
    per_insert = []
    counts: dict[str, Counter[str]] = {region: Counter() for region in REGION_ORDER}
    for ins in inserts:
        start, end = ins["start"], max(ins["start"], ins["end"])
        size = end - start + 1
        label = np.full(size, len(labels) - 1, dtype=np.int8)
        for code in range(len(labels) - 2, -1, -1):  # lowest priority first
            for low, high in _hits(layers[code].get(ins["seqid"]), start, end):
                a, b = max(low, start) - start, min(high, end) - start + 1
                label[a:b] = code
        region = labels[int(np.bincount(label, minlength=len(labels)).argmax())]
        if te_index is None:
            te_class = "not_annotated"
        else:
            cover = np.zeros(size, dtype=bool)
            by_class: Counter[str] = Counter()
            for low, high, cls in _hits(te_index.get(ins["seqid"]), start, end, payload=True):
                a, b = max(low, start) - start, min(high, end) - start + 1
                by_class[cls] += b - a
                cover[a:b] = True
            fraction = cover.sum() / size
            te_class = (
                by_class.most_common(1)[0][0]
                if by_class and fraction >= te_min_fraction
                else "nonTE"
            )
        counts[region][te_class] += 1
        per_insert.append({**ins, "region": region, "te_class": te_class})
    total = len(per_insert)
    return {
        "total": total,
        "regions": list(REGION_ORDER),
        "counts": {region: dict(counts[region]) for region in REGION_ORDER},
        "region_fraction": {
            region: (sum(counts[region].values()) / total if total else 0.0)
            for region in REGION_ORDER
        },
        "te_annotated": te_index is not None,
        "parameters": {
            "promoter": promoter,
            "terminator": terminator,
            "te_min_fraction": te_min_fraction,
        },
        "candidates": per_insert,
    }


def _open_text(path: Path) -> Iterable[str]:
    import gzip

    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def _gff_attrs(text: str) -> dict[str, str]:
    out = {}
    for part in text.strip().split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _region_layers(
    path: Path, seqids: set[str], promoter: int, terminator: int
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Exon, gene-body, promoter and terminator interval indexes per seqid."""
    genes: dict[str, list[tuple[int, int, str]]] = {}
    transcripts: dict[str, list[tuple[int, int, str]]] = {}
    exons: dict[str, list[tuple[int, int]]] = {}
    coding: dict[str, list[tuple[int, int]]] = {}
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] not in seqids:
                continue
            kind = fields[2].lower()
            start, end = int(fields[3]), int(fields[4])
            if kind == "gene":
                genes.setdefault(fields[0], []).append((start, end, fields[6]))
            elif kind in {"mrna", "transcript"}:
                transcripts.setdefault(fields[0], []).append((start, end, fields[6]))
            elif kind == "exon":
                exons.setdefault(fields[0], []).append((start, end))
            elif kind in {"cds", "five_prime_utr", "three_prime_utr"}:
                coding.setdefault(fields[0], []).append((start, end))
    layers: list[dict[str, tuple[np.ndarray, np.ndarray]]] = [{}, {}, {}, {}]
    for seqid in seqids:
        # annotations without ``gene`` lines (mRNA/exon/CDS only) use transcripts
        gene_rows = genes.get(seqid) or transcripts.get(seqid, [])
        exon_rows = exons.get(seqid) or coding.get(seqid, [])
        up, down = [], []
        for start, end, strand in gene_rows:
            five = (max(1, start - promoter), start - 1)
            three = (end + 1, end + terminator)
            if strand == "-":
                five, three = (end + 1, end + promoter), (max(1, start - terminator), start - 1)
            if five[1] >= five[0]:
                up.append(five)
            if three[1] >= three[0]:
                down.append(three)
        spans = [(s, e) for s, e, _strand in gene_rows]
        for code, intervals in enumerate((exon_rows, spans, up, down)):
            layers[code][seqid] = _interval_index(intervals)
    return layers


def _te_intervals(path: Path, seqids: set[str]) -> dict[str, tuple[np.ndarray, ...]]:
    """TE intervals with their class from a RepeatMasker ``.out`` or a GFF3."""
    rows: dict[str, list[tuple[int, int, str]]] = {}
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 9:  # GFF3
                if fields[0] not in seqids:
                    continue
                attrs = _gff_attrs(fields[8])
                cls = (
                    attrs.get("Classification")
                    or attrs.get("classification")
                    or attrs.get("Class")
                    or fields[2]
                )
                rows.setdefault(fields[0], []).append(
                    (int(fields[3]), int(fields[4]), te_category(cls))
                )
                continue
            parts = line.split()  # RepeatMasker .out (whitespace columns)
            if len(parts) < 11 or not parts[0].isdigit() or parts[4] not in seqids:
                continue
            rows.setdefault(parts[4], []).append(
                (int(parts[5]), int(parts[6]), te_category(parts[10]))
            )
    out: dict[str, tuple[np.ndarray, ...]] = {}
    for seqid, intervals in rows.items():
        intervals.sort()
        starts = np.array([s for s, _e, _c in intervals], dtype=np.int64)
        ends = np.array([e for _s, e, _c in intervals], dtype=np.int64)
        classes = np.array([c for _s, _e, c in intervals], dtype=object)
        out[seqid] = (starts, ends, np.maximum.accumulate(ends), classes)
    return out


def _interval_index(intervals: Iterable[tuple[int, int]]) -> tuple[np.ndarray, ...]:
    ordered = sorted(intervals)
    starts = np.array([s for s, _e in ordered], dtype=np.int64)
    ends = np.array([e for _s, e in ordered], dtype=np.int64)
    running = np.maximum.accumulate(ends) if ends.size else ends
    return starts, ends, running


def _hits(index: Any, start: int, end: int, *, payload: bool = False) -> list[Any]:
    """Intervals of a sorted index overlapping ``[start, end]``.

    ``running`` (prefix maximum of ends) bounds the scan: everything before the
    first position whose running end reaches ``start`` cannot overlap.
    """
    if index is None or not index[0].size:
        return []
    starts, ends, running = index[0], index[1], index[2]
    first = int(np.searchsorted(running, start, side="left"))
    last = int(np.searchsorted(starts, end, side="right"))
    out = []
    for i in range(first, last):
        if ends[i] >= start:
            out.append(
                (int(starts[i]), int(ends[i]), index[3][i])
                if payload
                else (int(starts[i]), int(ends[i]))
            )
    return out


def plot_transfer_length_profile(
    samples: Mapping[str, Any],
    *,
    groups: Mapping[str, str] | None = None,
    transfer_label: str = "NUMT",
    bins: Sequence[int] = LENGTH_PROFILE_BINS,
    title: str | None = None,
    panel_label: str | None = None,
    width: str | float | None = None,
    height: float | None = None,
) -> Any:
    """Number of NUMTs/NUPTs per fragment-length bin, one line per sample.

    ``samples`` maps a sample/haplotype name to its candidates (rows or a
    transfer Result) or to a precomputed :func:`transfer_length_bins` result.
    ``groups`` maps sample -> population/group: lines take the group colour
    (one legend entry per group) and each sample keeps its own marker and its
    name at the line end. Compute-only; render with ``ov.write(plot, output)``.
    """
    from .plot_object import plot_result

    if not samples:
        raise ValueError("samples must contain at least one sample")
    profiles = {}
    for name, value in samples.items():
        if isinstance(value, Mapping) and "counts" in value and "bins" in value:
            profiles[str(name)] = dict(value)
        else:
            profiles[str(name)] = transfer_length_bins(value, bins)
    labels = next(iter(profiles.values()))["bins"]
    group_of = {name: str((groups or {}).get(name, name)) for name in profiles}
    group_names = list(dict.fromkeys(group_of.values()))
    first_edge = sorted(int(b) for b in bins)[0]
    caption = (
        f"Number of {transfer_label}s per fragment-length bin, one line per sample "
        f"({'colour = group, ' if groups is not None else ''}marker and end label = sample). "
        f"Fragments shorter than {first_edge} bp are not counted."
    )
    spec = {
        "profiles": profiles,
        "group_of": group_of,
        "group_names": group_names,
        "grouped": groups is not None,
        "transfer": canonical_transfer(transfer_label),
        "transfer_label": transfer_label,
        "bins": labels,
        "title": title,
        "panel_label": panel_label,
        "width": width,
        "height": height,
    }

    def _render(path: str | Path) -> Path:
        return _render_length_profile(spec, path)

    plot = plot_result(
        "plot_transfer_length_profile",
        _render,
        organelle="nuclear",
        metrics={
            "transfer_label": transfer_label,
            "bins": labels,
            "samples": {name: p["counts"] for name, p in profiles.items()},
            "below_min": {name: p["below_min"] for name, p in profiles.items()},
            "groups": group_of,
            "caption": caption,
        },
        summary=f"{transfer_label} length profile prepared for {len(profiles)} sample(s).",
    )
    plot._panel = PanelDrawer(
        lambda ax, legend: _draw_length_profile(spec, ax, legend),
        aspect=0.5,
        header=title or "",
    )
    return plot


def plot_transfer_region_te(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    transfer_label: str = "NUMT",
    regions: Sequence[str] = REGION_ORDER,
    groups: Mapping[str, str] | None = None,
    title: str | None = None,
    panel_label: str | None = None,
    width: str | float | None = None,
    height: float | None = None,
) -> Any:
    """Share of NUMTs/NUPTs per nuclear functional region, split by TE class.

    ``summaries`` maps sample/haplotype -> :func:`summarize_transfer_regions`
    output. For every region there is one bar per sample (side by side); bar
    height is the percentage of that sample's insertions in the region.

    The stack inside a bar is the TE class only when TEs were evaluated and
    some insertion is TE-covered. Without a TE annotation the bars are plain
    (coloured by ``groups``) and marked "TE classes not evaluated" - never
    drawn as "no TE"; when TEs were evaluated but none qualifies, the plot is
    the region shares alone. Compute-only; render with ``ov.write``.
    """
    from .plot_object import plot_result

    if not summaries:
        raise ValueError("summaries must contain at least one sample")
    table: dict[str, dict[str, dict[str, float]]] = {}
    totals: dict[str, int] = {}
    present: set[str] = set()
    annotated = True
    params: Mapping[str, Any] = {}
    for name, summary in summaries.items():
        total = int(summary.get("total", 0)) or 1
        totals[str(name)] = int(summary.get("total", 0))
        annotated = annotated and bool(summary.get("te_annotated"))
        params = summary.get("parameters", params) or params
        counts = summary.get("counts", {})
        table[str(name)] = {
            region: {te: 100.0 * n / total for te, n in dict(counts.get(region, {})).items() if n}
            for region in regions
        }
        for region in regions:
            present.update(te for te, n in dict(counts.get(region, {})).items() if n)
    if not annotated:
        te_mode = "not_evaluated"
    elif present <= {"nonTE"}:
        te_mode = "no_te_overlap"
    else:
        te_mode = "te_classes"
    fraction = float(params.get("te_min_fraction", 0.5))
    order = [te for te in TE_CLASSES if te in present] if te_mode == "te_classes" else []
    group_of = {name: str((groups or {}).get(name, "")) for name in table}
    region_text = (
        f"Share of each sample's {transfer_label}s in nuclear functional regions (label "
        f"covering most bases; promoter/terminator = {params.get('promoter', 2000):,}/"
        f"{params.get('terminator', 2000):,} bp up/downstream of genes)."
    )
    te_text = {
        "not_evaluated": "TE classes not evaluated: no transposable-element annotation was supplied.",
        "no_te_overlap": f"No insertion is at least {fraction:.0%} covered by annotated TEs.",
        "te_classes": f"Stacks: TE class covering most bases (at least {fraction:.0%} of the insertion).",
    }[te_mode]
    spec = {
        "table": table,
        "totals": totals,
        "regions": list(regions),
        "te_order": order,
        "te_mode": te_mode,
        "te_min_fraction": fraction,
        "group_of": group_of,
        "group_names": [g for g in dict.fromkeys(group_of.values()) if g],
        "transfer": canonical_transfer(transfer_label),
        "transfer_label": transfer_label,
        "title": title,
        "panel_label": panel_label,
        "width": width,
        "height": height,
    }

    def _render(path: str | Path) -> Path:
        return _render_region_te(spec, path)

    plot = plot_result(
        "plot_transfer_region_te",
        _render,
        organelle="nuclear",
        metrics={
            "transfer_label": transfer_label,
            "regions": list(regions),
            "percent": table,
            "totals": totals,
            "te_annotated": annotated,
            "te_mode": te_mode,
            "caption": f"{region_text} {te_text}",
        },
        summary=f"{transfer_label} region/TE composition prepared for {len(table)} sample(s).",
    )
    plot._panel = PanelDrawer(
        lambda ax, legend: _draw_region_te(spec, ax, legend),
        aspect=0.55,
        header=title or "",
    )
    return plot


_PANEL_LEFT_IN = 0.45  # same in every transfer panel so letters line up in a column
_LABEL_PT = max(FONT_PT["label"], 6.5)  # sample names, callouts, insertion labels
_SAMPLE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<")


def _group_label(group: str, size: int) -> str:
    return f"{group} (n = {size})"


def _inset_panel_axes(
    ax: Any, *, bottom: float, right: float = 0.05, left: float = _PANEL_LEFT_IN
) -> None:
    """Shrink ``ax`` inside its composer box by margins given in inches."""
    fig_w, fig_h = ax.figure.get_size_inches()
    box = ax.get_position()
    x0 = box.x0 + left / fig_w
    y0 = box.y0 + bottom / fig_h
    width = max(box.width - (left + right) / fig_w, 0.1 * box.width)
    height = max(box.height - bottom / fig_h, 0.1 * box.height)
    ax.set_position((x0, y0, width, height))


def _style_axes(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=_MUTED, labelcolor=_INK, width=0.6, length=2.5)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def _render_axes_figure(
    spec: Mapping[str, Any],
    output: str | Path,
    draw: Any,
    *,
    default_width: str,
    default_height: float,
) -> Path:
    """Stand-alone printed-size figure around one ``draw(ax, legend)`` panel."""
    width = spec.get("width")
    fig_w = width_inches(width) if isinstance(width, str) else float(width or 0)
    fig_w = fig_w or width_inches(default_width)
    fig_h = min(float(spec.get("height") or default_height), MAX_HEIGHT_MM / MM_PER_INCH)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with publication_style():
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        try:
            fig.subplots_adjust(left=0.14, right=0.97, bottom=0.3, top=0.92)
            draw(ax, True)
            pad = 3.0
            if spec.get("title"):
                ax.set_title(str(spec["title"]), loc="left", fontsize=FONT_PT["body"], pad=pad)
            if spec.get("panel_label"):
                add_panel_label(ax, str(spec["panel_label"]), y_pt=pad)
            fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        finally:
            plt.close(fig)
    return out


def _draw_length_profile(spec: Mapping[str, Any], ax: Any, legend: bool) -> list[Any]:
    labels = spec["bins"]
    if not legend:  # composite panel: keep rotated bin labels and titles inside
        _inset_panel_axes(ax, bottom=0.66)
    xs = np.arange(len(labels))
    keys = spec["group_names"] if spec["grouped"] else list(spec["profiles"])
    colors = {key: _GROUP_PALETTE[i % len(_GROUP_PALETTE)] for i, key in enumerate(keys)}
    many = len(spec["profiles"]) > 12
    rank_in_group: Counter[str] = Counter()
    ends: list[tuple[float, str, str, str]] = []
    peak = 1
    for name, profile in spec["profiles"].items():
        key = spec["group_of"][name] if spec["grouped"] else name
        marker = _SAMPLE_MARKERS[rank_in_group[key] % len(_SAMPLE_MARKERS)]
        rank_in_group[key] += 1
        ys = [profile["counts"][label] for label in labels]
        peak = max(peak, *ys)
        ax.plot(
            xs,
            ys,
            color=colors[key],
            lw=0.6 if many else 0.8,
            alpha=0.55 if many else 0.95,
            marker=None if many else marker,
            markersize=3.0,
            markerfacecolor="white",
            markeredgecolor=colors[key],
            markeredgewidth=0.7,
            zorder=3,
        )
        ends.append((ys[-1], name, colors[key], marker))
    ax.set_ylim(0, peak * 1.08)
    if not many:
        _label_line_ends(ax, xs[-1], ends, peak * 1.08)
    else:
        ax.set_xlim(-0.3, len(labels) - 0.7)
    sizes = Counter(spec["group_of"].values())
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[key],
            lw=1.0,
            label=_group_label(key, sizes[key]) if spec["grouped"] else key,
        )
        for key in keys
    ]
    ax.set_xticks(xs, labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel(f"Number of {spec['transfer_label']}s")
    _style_axes(ax)
    if not spec["grouped"]:
        return []  # end labels already name every sample
    if legend:
        ax.legend(handles=handles, loc="upper right", handlelength=1.4, borderaxespad=0.2)
        return []
    return handles


def _label_line_ends(
    ax: Any, x_last: float, ends: Sequence[tuple[float, str, str, str]], y_top: float
) -> None:
    """Name each line at its right end, spread vertically so names never touch.

    Each name starts with the sample's marker, which ties it to its line
    without leader lines (leaders to a stacked column cross the other names).
    """
    fig_w, fig_h = ax.figure.get_size_inches()
    box = ax.get_position()
    ax_w, ax_h = box.width * fig_w, box.height * fig_h
    marker_in = 0.07
    widest = max(len(name) for _y, name, _c, _m in ends) * _LABEL_PT * 0.56 / 72 + marker_in
    span = x_last + 0.5  # data units before the labels
    per_in = span / max(ax_w - widest, 0.5)
    extra = 0.3 + widest * per_in  # marker at +0.2, then the name
    ax.set_xlim(-0.3, x_last + extra)
    gap = _LABEL_PT * 1.25 / 72 * y_top / ax_h
    ordered = sorted(ends)
    placed = _spread([y for y, _n, _c, _m in ordered], [gap] * len(ordered))
    shift = max(0.0, gap / 2 - min(placed))
    x_marker = x_last + 0.2
    for (_y, name, color, marker), pos in zip(ordered, placed, strict=True):
        y = pos + shift
        ax.plot(
            [x_marker],
            [y],
            marker=marker,
            markersize=3.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.7,
            linestyle="none",
            clip_on=False,
            zorder=4,
        )
        ax.text(
            x_marker + marker_in * per_in,
            y,
            name,
            fontsize=_LABEL_PT,
            color=color,
            va="center",
            ha="left",
            clip_on=False,
        )


def _draw_region_te(spec: Mapping[str, Any], ax: Any, legend: bool) -> list[Any]:
    samples = list(spec["table"])
    longest = max((len(s) for s in samples), default=0) if len(samples) > 1 else 0
    if not legend:  # composite panel: sample and region names stay inside
        _inset_panel_axes(ax, bottom=0.22 + longest * _LABEL_PT * 0.58 / 72)
    regions = spec["regions"]
    n = len(samples)
    bar_w = min(0.82 / max(n, 1), 0.32)
    group_w = bar_w * n
    split = spec["te_mode"] == "te_classes"
    te_colors = {te: TE_COLORS.get(te, "#7B8794") for te in spec["te_order"]}
    group_colors = {
        g: _GROUP_PALETTE[i % len(_GROUP_PALETTE)] for i, g in enumerate(spec["group_names"])
    }
    top = 0.0
    for r_index, region in enumerate(regions):
        for s_index, sample in enumerate(samples):
            x = r_index - group_w / 2 + bar_w * (s_index + 0.5)
            shares = spec["table"][sample][region]
            if split:
                bottom = 0.0
                for te in spec["te_order"]:
                    value = shares.get(te, 0.0)
                    if value <= 0:
                        continue
                    ax.bar(
                        x,
                        value,
                        width=bar_w * 0.88,
                        bottom=bottom,
                        color=te_colors[te],
                        edgecolor=_FAINT if te == "nonTE" else "white",
                        linewidth=0.25,
                        zorder=3,
                    )
                    bottom += value
                top = max(top, bottom)
            else:
                value = sum(shares.values())
                color = group_colors.get(spec["group_of"].get(sample, ""), "#8A96A3")
                ax.bar(x, value, width=bar_w * 0.88, color=color, linewidth=0, zorder=3)
                top = max(top, value)
            if n > 1:
                # sample names under each bar; region names below them
                ax.annotate(
                    sample,
                    xy=(x, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -3.5),
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=_LABEL_PT,
                    color=_MUTED,
                )
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, fontsize=FONT_PT["body"])
    if n > 1:
        ax.tick_params(axis="x", pad=5.5 + longest * _LABEL_PT * 0.58, length=0, bottom=False)
    ax.set_xlim(-0.6, len(regions) - 0.4)
    ax.set_ylim(0, top * 1.12 if top else 1)
    ax.set_ylabel(f"{spec['transfer_label']}s (% of sample)")
    _style_axes(ax)
    note = {
        "not_evaluated": "TE classes not evaluated",
        "no_te_overlap": f"No insertion ≥{spec['te_min_fraction']:.0%} TE-covered",
    }.get(spec["te_mode"])
    if note:
        ax.text(
            0.01,
            0.98,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_LABEL_PT,
            color=_MUTED,
            style="italic",
        )
    if split:
        handles = [
            Patch(
                facecolor=te_colors[te],
                edgecolor=_FAINT if te == "nonTE" else "none",
                lw=0.4,
                label=te,
            )
            for te in spec["te_order"]
        ]
        title = "TE class"
    else:
        sizes = Counter(g for g in spec["group_of"].values() if g)
        handles = [
            Patch(facecolor=group_colors[g], edgecolor="none", label=_group_label(g, sizes[g]))
            for g in spec["group_names"]
        ]
        title = None
    if legend and handles:
        ax.legend(
            handles=handles,
            title=title,
            title_fontsize=FONT_PT["body"],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            handlelength=1.0,
            alignment="left",
        )
        return []
    return handles


def _render_length_profile(spec: Mapping[str, Any], output: str | Path) -> Path:
    return _render_axes_figure(
        spec,
        output,
        lambda ax, legend: _draw_length_profile(spec, ax, legend),
        default_width="single",
        default_height=2.4,
    )


def _render_region_te(spec: Mapping[str, Any], output: str | Path) -> Path:
    return _render_axes_figure(
        spec,
        output,
        lambda ax, legend: _draw_region_te(spec, ax, legend),
        default_width="double" if len(spec["table"]) > 2 else "onehalf",
        default_height=2.6,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _tick_handle(color: str, label: str, width: float, rel: float = 1.0) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker="|",
        linestyle="none",
        markersize=3 + 5 * rel,
        markeredgewidth=width * 1.4,
        color=color,
        label=label,
    )


def _dot_handle(face: str, edge: str, label: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=3,
        markerfacecolor=face,
        markeredgecolor=edge,
        markeredgewidth=0.6,
        label=label,
    )


def _gene_class(row: Mapping[str, Any]) -> str:
    return "within_gene" if row.get("nuclear_gene") else "intergenic"


def _spread(positions: Sequence[float], widths: Sequence[float]) -> list[float]:
    """Place labels at sorted ``positions`` without overlap, as close as possible.

    Overlapping neighbours merge into a block that is packed edge to edge and
    centred on the mean of its members' wanted positions; blocks keep merging
    until none overlap (a 1-D label repel).
    """
    # each block: [first index, last index, start coordinate]
    blocks: list[list[float]] = []
    for i, (pos, width) in enumerate(zip(positions, widths, strict=True)):
        blocks.append([i, i, pos - width / 2])
        while len(blocks) > 1:
            prev, cur = blocks[-2], blocks[-1]
            prev_end = prev[2] + sum(widths[int(prev[0]) : int(prev[1]) + 1])
            if prev_end <= cur[2]:
                break
            first, last = int(prev[0]), int(cur[1])
            total = sum(widths[first : last + 1])
            centre = sum(positions[first : last + 1]) / (last - first + 1)
            blocks[-2:] = [[first, last, centre - total / 2]]
    placed: list[float] = []
    for first, last, start in blocks:
        cursor = start
        for j in range(int(first), int(last) + 1):
            placed.append(cursor + widths[j] / 2)
            cursor += widths[j]
    return placed


def _format_length(length_bp: float) -> str:
    if length_bp >= 1000:
        return f"{length_bp / 1000:.1f} kb"
    return f"{int(length_bp)} bp"


def _format_window(window: int) -> str:
    if window >= 1_000_000 and window % 1_000_000 == 0:
        return f"{window // 1_000_000} Mb"
    if window >= 1000:
        return f"{window / 1000:g} kb"
    return f"{window} bp"


def _first(lowered: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
