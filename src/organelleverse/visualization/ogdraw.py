"""OGDraw-style circular organelle genome maps — Python port of OGDrawR.

Faithful reproduction of the OGDrawR R package (circlize-based) in matplotlib:

- GenBank parsing → gene coordinates, exon/intron boundaries, strand, GC content
- 17-colour OGDraw functional-category palette
- exon/intron-aware gene blocks (exons full-height, introns as narrow bars)
- dual-strand layout (plus outside, minus inside the backbone)
- sliding-window GC content ring with mean line
- iterative label de-overlap on circular coordinates (``spread_labels``)
- italic gene names + centre genome name, publication-ready legend
- PDF / PNG / TIFF / SVG output

Ported from https://github.com/xibeixingchen/OGDrawR (R/circlize) →
matplotlib. The pure-logic functions (``classify_gene``, ``spread_labels``,
``GENE_COLORS``) mirror the R implementation 1:1 and are unit-tested against the
R package's own testthat cases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from organelleverse._public_validation import validated_public

from .plot_object import OrganellePlot
from .style import FONT_PT, publication_style, width_inches

# ── OGDraw 17-colour palette (ported from OGDrawR R/colors.R) ────────────────
GENE_COLORS: dict[str, str] = {
    "complex_I": "#FFD700",
    "complex_II": "#32CD32",
    "complex_III": "#FFDEAD",
    "complex_IV": "#FFB6C1",
    "atp_synthase": "#9ACD32",
    "cytochrome_c": "#228B22",
    "rna_pol": "#B22222",
    "ribo_SSU": "#F5DEB3",
    "ribo_LSU": "#D2B48C",
    "maturase": "#FF8C00",
    "other": "#DA70D6",
    "orf": "#AFEEEE",
    "tRNA": "#00008B",
    "rRNA": "#FF0000",
    "ori_rep": "#FFB6C1",
    "poly_trans": "#BA55D3",
    "intron": "#FFFFFF",
    # Plastid functional classes; OGDrawR's palette above covers only
    # mitochondrial complexes, so these colours are OrganelleVerse's own.
    "photosystem_I": "#1B5E20",
    "photosystem_II": "#7CB342",
    "cytochrome_b6f": "#4DB6AC",
    "rubisco": "#C0CA33",
    "clp_protease": "#6D4C41",
    "ycf": "#90CAF9",
}

# Ordered prefix → category rules (mirror OGDrawR .classify_gene order).
_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("trn", "tRNA"),
    ("rrn", "rRNA"),
    ("nad", "complex_I"),
    ("sdh", "complex_II"),
    ("cob", "complex_III"),
    ("cox", "complex_IV"),
    ("atp", "atp_synthase"),
    ("ccm", "cytochrome_c"),
    ("rps", "ribo_SSU"),
    ("rpl", "ribo_LSU"),
    ("mat", "maturase"),
    ("orf", "orf"),
    ("rpo", "rna_pol"),
)

# Plastid-only prefixes, tried before ``_CATEGORY_RULES``. ``ndh`` is the
# plastid NADH dehydrogenase, which the mitochondrial ``nad`` rule never matches.
_PLASTID_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("psa", "photosystem_I"),
    ("psb", "photosystem_II"),
    ("pet", "cytochrome_b6f"),
    ("ndh", "complex_I"),
    ("rbc", "rubisco"),
    ("clp", "clp_protease"),
    ("ycf", "ycf"),
)

# Legend rows: (category key, human-readable label) in OGDraw order.
_LEGEND: tuple[tuple[str, str], ...] = (
    ("complex_I", "complex I (NADH dehydrogenase)"),
    ("complex_II", "complex II (succinate dehydrogenase)"),
    ("complex_III", "complex III (ubichinol cytochrome c reductase)"),
    ("complex_IV", "complex IV (cytochrome c oxidase)"),
    ("atp_synthase", "ATP synthase"),
    ("cytochrome_c", "cytochrome c biogenesis"),
    ("rna_pol", "RNA polymerase"),
    ("ribo_SSU", "ribosomal proteins (SSU)"),
    ("ribo_LSU", "ribosomal proteins (LSU)"),
    ("maturase", "maturases"),
    ("other", "other genes"),
    ("orf", "ORFs"),
    ("tRNA", "transfer RNAs"),
    ("rRNA", "ribosomal RNAs"),
    ("ori_rep", "origin of replication"),
    ("poly_trans", "polycistronic transcripts"),
    ("intron", "introns"),
)

_PLASTID_LEGEND: tuple[tuple[str, str], ...] = (
    ("photosystem_I", "photosystem I"),
    ("photosystem_II", "photosystem II"),
    ("cytochrome_b6f", "cytochrome b6/f complex"),
    ("atp_synthase", "ATP synthase"),
    ("complex_I", "NADH dehydrogenase (ndh)"),
    ("rubisco", "RubisCO large subunit"),
    ("rna_pol", "RNA polymerase"),
    ("ribo_SSU", "ribosomal proteins (SSU)"),
    ("ribo_LSU", "ribosomal proteins (LSU)"),
    ("clp_protease", "clpP protease"),
    ("maturase", "maturases"),
    ("ycf", "hypothetical reading frames (ycf)"),
    ("other", "other genes"),
    ("orf", "ORFs"),
    ("tRNA", "transfer RNAs"),
    ("rRNA", "ribosomal RNAs"),
    ("intron", "introns"),
)


def legend_rows(organelle: str) -> tuple[tuple[str, str], ...]:
    """Legend rows for an organelle: plastid classes for plastids, else OGDraw's."""
    return _PLASTID_LEGEND if organelle == "plastid" else _LEGEND


_ORGANELLE_SUBTITLE = {"mito": "mitochondrial genome", "plastid": "plastid genome"}


@validated_public('visualization.classify_gene')
def classify_gene(name: str, organelle: str = "mito") -> str:
    """Classify a gene name into an OGDraw functional category.

    Mirrors OGDrawR ``.classify_gene``: case-insensitive prefix match, first
    rule wins, unmatched → ``"other"``. With ``organelle="plastid"`` the plastid
    classes (photosystems, cytochrome b6/f, ndh, RubisCO, clpP, ycf) are tried
    first; the default keeps the OGDrawR mitochondrial behaviour unchanged.
    """
    nl = str(name).lower()
    rules = _CATEGORY_RULES
    if organelle == "plastid":
        rules = _PLASTID_CATEGORY_RULES + _CATEGORY_RULES
    for prefix, category in rules:
        if nl.startswith(prefix):
            return category
    return "other"


@validated_public('visualization.spread_labels')
def spread_labels(
    gene_mids: list[float],
    genome_length: float,
    min_gap_bp: float = 4500,
    max_iter: int = 20,
) -> tuple[list[float], list[bool]]:
    """Iteratively push apart labels closer than ``min_gap_bp`` on the circle.

    Port of OGDrawR ``spread_labels``. Returns ``(pos, moved)`` in the *input
    order*; ``moved[i]`` marks labels displaced by more than 800 bp. OGDrawR
    stops after 20 passes; the map drawer passes more so a dense cluster (the
    plastid ribosomal-protein operon) finishes spreading instead of overlapping.
    """
    n = len(gene_mids)
    if n <= 1:
        return list(gene_mids), [False] * n

    order = sorted(range(n), key=lambda i: gene_mids[i])
    # Spread on an unrolled line cut at the widest gap between neighbours, so a
    # cluster that straddles the origin moves as one block instead of being
    # pushed across 0 and folded back onto the labels at the far end. When the
    # widest gap is already the one across the origin this is OGDrawR's layout.
    values = [gene_mids[i] for i in order]
    gaps = [values[k + 1] - values[k] for k in range(n - 1)]
    wrap_gap = values[0] + genome_length - values[-1]
    if gaps and max(gaps) > wrap_gap:
        cut = gaps.index(max(gaps)) + 1
        order = order[cut:] + order[:cut]
        values = values[cut:] + [v + genome_length for v in values[:cut]]
    adj = list(values)
    orig = list(adj)

    for _ in range(max_iter):
        moved = False
        for i in range(1, n):
            gap = adj[i] - adj[i - 1]
            if gap < min_gap_bp:
                shift = (min_gap_bp - gap) / 2
                adj[i - 1] -= shift
                adj[i] += shift
                moved = True
        gap_wrap = (adj[0] + genome_length) - adj[-1]
        if gap_wrap < min_gap_bp:
            shift = (min_gap_bp - gap_wrap) / 2
            adj[-1] -= shift
            adj[0] += shift
        if not moved:
            break

    adj = [a % genome_length for a in adj]
    orig = [o % genome_length for o in orig]  # unchanged unless the line was unrolled
    was_moved = [abs(adj[k] - orig[k]) > 800 for k in range(n)]

    result_pos = [0.0] * n
    result_moved = [False] * n
    for k, idx in enumerate(order):
        result_pos[idx] = adj[k]
        result_moved[idx] = was_moved[k]
    return result_pos, result_moved


@dataclass
class Gene:
    """A single gene with exon/intron structure and functional category."""

    name: str
    start: int
    end: int
    strand: int
    has_intron: bool
    exons: list[tuple[int, int]]
    category: str
    trans_spliced: bool | None = None
    exon_strands: list[int] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2

    @property
    def length(self) -> int:
        return abs(self.end - self.start)

    @property
    def anchor_bp(self) -> float:
        """bp used to anchor the label + leader line.

        Midpoint of the *largest exon*. For trans-spliced / multi-exon genes
        (e.g. nad1/nad2/nad5, whose exons are tens of kb apart) the whole-gene
        midpoint can fall inside an intron; the largest exon's midpoint always
        lands on a real block, so the leader points at an exon, not empty space.
        """
        if self.has_intron and self.exons:
            a, b = max(self.exons, key=lambda e: e[1] - e[0])
            return (a + b) / 2
        return self.mid


@dataclass
class MitoGenome:
    """Parsed organelle genome ready for drawing (OGDrawR ``mito_genome``)."""

    genes: list[Gene]
    gc: list[dict[str, float]]
    genome_length: int
    organelle: str = "mito"
    _seen: set[str] = field(default_factory=set, repr=False)

    def summary(self) -> dict[str, Any]:
        cats: dict[str, int] = {}
        for g in self.genes:
            cats[g.category] = cats.get(g.category, 0) + 1
        n_intron = sum(1 for g in self.genes if g.has_intron)
        mean_gc = round(sum(w["gc"] for w in self.gc) / len(self.gc), 4) if self.gc else 0.0
        return {
            "genome_length": self.genome_length,
            "n_genes": len(self.genes),
            "n_intron_genes": n_intron,
            "mean_gc": mean_gc,
            "categories": cats,
        }


def record_organelle(record) -> str:
    """``"plastid"`` when the source feature's /organelle says so, else ``"mito"``."""
    src = next((f for f in record.features if f.type == "source"), None)
    if src is not None:
        organ = " ".join(src.qualifiers.get("organelle", [""])).lower()
        if "plastid" in organ or "chloroplast" in organ:
            return "plastid"
    return "mito"


def _parse_genbank_record(record, gc_window: int = 200) -> MitoGenome:
    """Parse one already-loaded SeqRecord into a :class:`MitoGenome`."""
    sequence = str(record.seq).upper()
    genome_length = len(sequence)

    organelle = record_organelle(record)
    # Same loci as the structure map: products supersede their gene span, every
    # copy is kept (both IR copies), exons keep their own strand, and
    # trans-splicing is read from /trans_splicing or inferred from the layout.
    from .features import annotation_rows, genes_from_rows

    genes = genes_from_rows(annotation_rows(record, organelle, strict=False))

    gc: list[dict[str, float]] = []
    n_win = genome_length // gc_window if gc_window else 0
    for i in range(n_win):
        s = i * gc_window
        e = min((i + 1) * gc_window, genome_length)
        window = sequence[s:e]
        wl = len(window) or 1
        gc_val = (window.count("G") + window.count("C")) / wl
        gc.append({"start": s + 1, "end": e, "mid": (s + 1 + e) / 2, "gc": gc_val})

    return MitoGenome(genes=genes, gc=gc, genome_length=genome_length, organelle=organelle)


def _genbank_paths(genbank_path) -> list[Path]:
    if isinstance(genbank_path, (str, Path)):
        return [Path(genbank_path)]
    paths = [Path(p) for p in genbank_path]
    if not paths:
        raise ValueError("no GenBank files given")
    return paths


def _load_genbank_records(genbank_path) -> list[tuple[str, Any]]:
    """``(display_name, SeqRecord)`` for every record of one or more GenBank files.

    A name that occurs twice (two files that both hold ``NC_000932``, or two
    files that each call their record ``ctg1``) is prefixed with its file stem
    so every panel and exon table row stays attributable.
    """
    from Bio import SeqIO  # type: ignore

    found: list[tuple[Path, Any]] = []
    for path in _genbank_paths(genbank_path):
        if not path.exists():
            raise FileNotFoundError(f"GenBank file not found: {path}")
        found.extend((path, record) for record in SeqIO.parse(str(path), "genbank"))
    names = [record.name or record.id for _, record in found]
    loaded = []
    for (path, record), name in zip(found, names, strict=True):
        unique = name if names.count(name) == 1 else f"{path.stem}:{name}"
        record.name = unique
        loaded.append((unique, record))
    return loaded


def _genbank_display_name(genbank_path) -> str:
    """The record's species when the file names one, else the file stem (or a file count)."""
    from .features import genbank_display_name

    return genbank_display_name(genbank_path)


def parse_genbank(gb_file: str | Path, gc_window: int = 200) -> MitoGenome:
    """Parse a GenBank file into a :class:`MitoGenome` (OGDrawR ``parse_genbank``).

    Uses Biopython for robust location parsing (compound ``join`` locations →
    exons, ``complement`` → minus strand). Unlike OGDrawR, which keeps only the
    first gene of each name, every copy is kept (a plastome draws both IR
    copies) and CDS/tRNA/rRNA joins supply the exons, so a gene written as one
    span over a trans-spliced locus is not drawn as one long block. Original
    identifier casing is preserved (gene classification is case-insensitive at
    match time only). Multi-record files are rejected here; structure maps fan
    them out per contig instead of concatenating.
    """
    from Bio import SeqIO  # type: ignore

    gb_file = Path(gb_file)
    if not gb_file.exists():
        raise FileNotFoundError(f"GenBank file not found: {gb_file}")

    record = SeqIO.read(str(gb_file), "genbank")
    return _parse_genbank_record(record, gc_window)


# ── drawing ─────────────────────────────────────────────────────────────────
# Radial band layout (outer → inner). Gene blocks form two thin adjacent rings
# (plus outside, minus inside the backbone); ALL gene labels radiate outward
# from a common ring with leader lines, which de-crowds dense clusters far
# better than tangential ("clockwise-facing") labels.
_R_LABEL_OUT = 0.975  # plus-strand label anchor; text reads outward
_R_LABEL_MINUS_HUG = 0.975  # single-ring layout: minus-strand (grey) names share the outer anchor
_R_PLUS_BLOCK_OUT = 0.90
_R_PLUS_BLOCK_IN = 0.862
_R_BACKBONE = 0.859
_R_MINUS_BLOCK_OUT = 0.857
_R_MINUS_BLOCK_IN = 0.820
_R_LABEL_IN = 0.765  # minus-strand label anchor; text reads inward
# The elbow of a leader line: radial out of the block, a slanted run to the
# label's (spread) angle, radial again into the label. Labels keep genome order,
# so the slanted runs never cross.
_LEADER_ELBOW = 0.3  # share of the block-to-label gap spent on each radial leg
_R_GC_OUT = 0.58
_R_GC_IN = 0.47
# With an LSC/IR/SSC band the GC ring steps inward so the band sits between it
# and the inward-reading minus-strand labels.
_R_REGION_OUT = 0.555
_R_REGION_IN = 0.505
_R_GC_OUT_WITH_REGIONS = 0.49
_R_GC_IN_WITH_REGIONS = 0.38
_R_TICK = 0.035  # kb tick length, drawn inward from the GC ring's inner edge
REGION_COLORS = {"LSC": "#cde8bc", "SSC": "#ffd6a0", "IRa": "#a8d4ec", "IRb": "#a8d4ec"}

# Axis window in data units (circle radius 1). The standalone map puts its
# legend in one column right of the circle, vertically centred.
_XLIM = (-1.4, 2.62)
_YLIM_TITLE = (-1.34, 1.48)
_YLIM = (-1.34, 1.34)
_LEGEND_X = 1.42
_XLIM_LEGEND_BELOW = (-1.5, 1.5)
_YLIM_LEGEND_BELOW_BOTTOM = -2.1
# A legend-less circle under the narrow window: labels reach about the same
# radius at the bottom as at the top, so the panel is trimmed symmetrically.
_YLIM_PANEL_BOTTOM = -1.34

# Text is set in printed points (``style.FONT_PT``), not scaled with the canvas:
# the default canvas is the 183 mm double-column width, so these are the sizes
# on the journal page. A larger canvas only gives labels more room; a crowded
# ring shrinks its labels, never below the 5 pt floor.
_FONT_LABEL = 6.5
_FONT_NAME = 8.5
_FONT_CENTRE = FONT_PT["body"]
_FONT_LEGEND = FONT_PT["body"]
_FONT_REGION = 6.5
_FONT_TITLE = FONT_PT["title"]
_FONT_TICK = FONT_PT["tick"]
_GC_BAND = "#F1F1F1"
_GC_HIGH = "#8C8C8C"
_GC_LOW = "#C8C8C8"
_LABEL_LINE = 1.15  # label pitch along the ring, in font heights
_CROWDED_PT = 6.0  # below this a crowded panel drops tRNA labels instead
_MAX_LABEL_FILL = 0.85  # labels may occupy at most this share of the ring
_SPREAD_PASSES = 300
#: The GC ring's inner edge never recedes past this radius under inward names.
_MIN_INNER_GC_RADIUS = 0.14


def default_map_size(*, title: bool = False, legend_below: bool = False) -> tuple[float, float]:
    """(width, height) in inches: double-column width, height from the axis window."""
    xlim = _XLIM_LEGEND_BELOW if legend_below else _XLIM
    ylim = (
        (_YLIM_LEGEND_BELOW_BOTTOM, _YLIM[1]) if legend_below else (_YLIM_TITLE if title else _YLIM)
    )
    width = width_inches("double")
    return width, width * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])


def _inches_per_unit(ax) -> float:
    """Inches one data unit spans on this axes (equal aspect, current limits)."""
    fig = ax.figure
    box = ax.get_position()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return min(
        box.width * fig.get_figwidth() / (x1 - x0),
        box.height * fig.get_figheight() / (y1 - y0),
    )


def _label_gap_and_fit(
    count: int,
    genome_length: int,
    radius: float,
    unit_in: float,
    font_pt: float,
    floor_bp: float,
) -> tuple[float, float]:
    """bp gap between neighbouring labels on a ring, and a font shrink factor (≤ 1).

    The gap is one label pitch (``_LABEL_LINE`` font heights) along the ring at
    ``radius``, never below ``floor_bp`` (OGDrawR's fixed gap). When ``count``
    labels at that gap would fill more than ``_MAX_LABEL_FILL`` of the ring, the
    gap is capped and the font shrinks to match, never below the 5 pt floor
    (past that the canvas is too small for the genome and labels may touch).
    """
    if count <= 1:
        return max(floor_bp, 1.0), 1.0
    circumference_in = 2 * math.pi * radius * unit_in
    font_gap = genome_length * (font_pt / 72 * _LABEL_LINE) / circumference_in
    gap = max(font_gap, floor_bp)
    cap = _MAX_LABEL_FILL * genome_length / count
    if gap <= cap:
        return gap, 1.0
    return cap, max(FONT_PT["min"] / font_pt, min(1.0, cap / font_gap))


def _draw_regions(ax, regions, genome_length: int, shift: float = 0.0) -> None:
    """LSC/IRb/SSC/IRa band with tangential names (OGDraw's inner quadripartite ring).

    ``shift`` (<= 0) moves the band inward to clear inward minus-strand names.
    """
    for region, span in regions.items():
        a = span["start"]
        b = span["start"] + span["length"]
        _draw_arc_block(
            ax, a, b, _R_REGION_IN + shift, _R_REGION_OUT + shift, genome_length,
            REGION_COLORS.get(region, "#dddddd"), lw=0.3, edge="0.35",
        )
        mid = math.radians(_bp_to_deg((a + b) / 2, genome_length))
        deg = math.degrees(mid) % 360
        rot = deg - 90 if deg <= 180 else deg + 90  # tangential, never upside down
        r = (_R_REGION_IN + _R_REGION_OUT) / 2 + shift
        ax.text(
            r * math.cos(mid),
            r * math.sin(mid),
            region,
            rotation=rot,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=_FONT_REGION,
            zorder=3,
        )


def _bp_to_deg(bp: float, genome_length: int) -> float:
    """bp → matplotlib angle in degrees (start at top=90°, clockwise)."""
    return 90.0 - 360.0 * (bp / genome_length)


def _draw_arc_block(
    ax,
    a: float,
    b: float,
    r_in: float,
    r_out: float,
    genome_length: int,
    color: str,
    lw: float = 0.3,
    edge: str = "black",
) -> None:
    from matplotlib.patches import Wedge  # type: ignore

    if b < a:
        b = b + genome_length  # tolerate tiny wrap
    theta1 = _bp_to_deg(b, genome_length)
    theta2 = _bp_to_deg(a, genome_length)
    ax.add_patch(
        Wedge(
            (0, 0),
            r_out,
            theta1,
            theta2,
            width=r_out - r_in,
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
        )
    )


_TRANS_SPLICE_SPAN = 25000  # exons farther apart than this ⇒ treat as trans-spliced


def _is_trans_spliced(g: Gene) -> bool:
    """True for multi-exon genes whose exons are far apart (e.g. nad1/nad2/nad5)."""
    if g.trans_spliced is not None:
        return g.trans_spliced
    return g.has_intron and len(g.exons) >= 2 and (g.end - g.start) > _TRANS_SPLICE_SPAN


def _gene_label_points(g: Gene) -> list[tuple[float, str]]:
    """(anchor_bp, display_name) pairs for a gene's label(s).

    Compact / single-exon genes get one label. Trans-spliced genes get one
    label *per exon fragment*, numbered (``nad2·1``, ``nad2·2``) so each scattered
    block is identified without looking like separate gene copies.
    """
    if _is_trans_spliced(g):
        return [((a + b) / 2, f"{g.name}·{i}") for i, (a, b) in enumerate(g.exons, 1)]
    return [(g.anchor_bp, "tRNA?" if g.name == "tRNA" else g.name)]


def _draw_gene_blocks(
    ax, g: Gene, colors: dict[str, str], genome_length: int, r_in: float, r_out: float
) -> None:
    color = colors.get(g.category, colors["other"])
    # A narrow intron connector bar only makes sense for a *compact* cis-spliced
    # gene whose exons sit close together. Trans-spliced genes (nad1/nad2/nad5,
    # exons tens of kb apart) would otherwise get a huge bar spanning an intron
    # the leader then points into — so render their exons as separate blocks.
    compact = g.has_intron and len(g.exons) >= 2 and not _is_trans_spliced(g)
    if compact:
        mid_lo = r_in + (r_out - r_in) * 0.3
        mid_hi = r_in + (r_out - r_in) * 0.7
        if g.exon_strands:
            for left, right in pairwise(g.exons):
                a, b = (left[1], right[0] - 1) if g.strand == 1 else (right[1], left[0] - 1)
                if (b - a) % genome_length:
                    _draw_arc_block(ax, a, b, mid_lo, mid_hi, genome_length, colors["intron"], lw=0.2)
        else:
            _draw_arc_block(ax, g.start, g.end, mid_lo, mid_hi, genome_length, colors["intron"], lw=0.2)
    for i, (a, b) in enumerate(g.exons):
        if g.exon_strands:
            r_in, r_out = (
                (_R_PLUS_BLOCK_IN, _R_PLUS_BLOCK_OUT)
                if g.exon_strands[i] == 1
                else (_R_MINUS_BLOCK_IN, _R_MINUS_BLOCK_OUT)
            )
        _draw_arc_block(ax, a - 1 if g.exon_strands else a, b, r_in, r_out, genome_length, color)


def _draw_label(
    ax,
    label_bp: float,
    gene_bp: float,
    r_block: float,
    r_anchor: float,
    name: str,
    cex: float,
    genome_length: int,
    outward: bool = True,
    color: str = "#111111",
    draw_leader: bool = True,
    weight: str = "normal",
) -> None:
    """Draw one gene label as a radial spoke, with a leader line to its block.

    The label anchor sits at the (de-overlapped) ``label_bp`` angle while the
    leader line traces back to the gene's true ``gene_bp`` angle on its block
    ring, so dense clusters fan out cleanly instead of stacking. Plus-strand
    (outer ring) labels read outward; minus-strand (inner ring) labels read
    inward toward the centre, matching the OGDraw dual-strand convention.
    """
    ga = math.radians(_bp_to_deg(gene_bp, genome_length))
    la = math.radians(_bp_to_deg(label_bp, genome_length))
    if draw_leader:
        # Elbow leader: radial leg out of the block, slanted run to the label's
        # angle, radial leg into the label (straight when the label did not move).
        leg = (r_anchor - r_block) * _LEADER_ELBOW
        radii = (r_block, r_block + leg, r_anchor - leg, r_anchor)
        angles = (ga, ga, la, la)
        ax.plot(
            [r * math.cos(a) for r, a in zip(radii, angles, strict=True)],
            [r * math.sin(a) for r, a in zip(radii, angles, strict=True)],
            color="0.62",
            lw=0.35,
            solid_joinstyle="round",
            zorder=1,
        )
    deg = math.degrees(la) % 360
    left = 90 < deg < 270
    rot = deg - 180 if left else deg  # keep text upright (niceFacing)
    # Inward labels pin the opposite end so text extends toward the centre.
    ha = ("right" if left else "left") if outward else ("left" if left else "right")
    ax.text(
        r_anchor * math.cos(la),
        r_anchor * math.sin(la),
        " " + name + " ",
        rotation=rot,
        rotation_mode="anchor",
        ha=ha,
        va="center",
        fontsize=cex * 11,
        fontstyle="italic",
        fontweight=weight,
        color=color,
        zorder=3,
    )


def draw_mito_map(
    parsed: MitoGenome,
    genome_name: str = "Genome",
    output_file: str | Path | None = None,
    *,
    width: float | None = None,
    height: float | None = None,
    dpi: int = 600,
    colors: dict[str, str] | None = None,
    min_label_gap: float | None = 4500,
    label_cex: tuple[float, float, float] = (_FONT_LABEL / 11,) * 3,
    title: str = "",
    draw_gc: bool = True,
    draw_center_text: bool = True,
    draw_legend: bool = True,
    inner_tracks: Any | None = None,
    fold_minus_labels_out: bool = False,
    minus_label_color: str = "#9AA0A6",
    minus_label_mode: str = "outer_gray",
    regions: dict[str, dict[str, int]] | None = None,
    legend_below: bool = False,
    omit_trna_labels_when_crowded: bool = False,
    ax: Any | None = None,
) -> Path | None:
    """Draw an OGDraw-style circular map from a :class:`MitoGenome`.

    Reproduces OGDrawR ``draw_mito_map``. If ``output_file`` is given the figure
    is written (format from extension: pdf/png/tiff/svg) and its path returned;
    otherwise the live matplotlib figure is left open and ``None`` returned.

    Text is in printed points (journal style: 6.5 pt gene labels, 7 pt legend
    and centre text, Arial-compatible sans serif, editable in PDF/SVG). The
    default canvas is the 183 mm double-column width with the height of the
    axis window, so ``width``/``height`` (inches) are only needed for another
    size; a larger canvas gives labels more room rather than larger type.
    Label spacing follows the font height; ``min_label_gap`` is a lower bound in
    bp (``None``: none beyond the font). ``label_cex`` is point size / 11 for
    tRNA, rRNA and other labels. ``regions`` (plastid LSC/IRb/SSC/IRa, 0-based
    start and length) adds the quadripartite band inside the gene rings.
    ``legend_below`` puts the legend in two columns under the circle instead of
    beside it, so a narrow panel spends its width on the circle.
    ``omit_trna_labels_when_crowded`` (single-ring labels only) drops tRNA
    labels when the full set would print below 6 pt; the tRNA blocks stay and
    ``ax.ov_trna_labels_omitted`` records it so a caller can say so.
    """
    with publication_style():
        return _draw_mito_map(
            parsed, genome_name, output_file, width=width, height=height, dpi=dpi,
            colors=colors, min_label_gap=min_label_gap, label_cex=label_cex, title=title,
            draw_gc=draw_gc, draw_center_text=draw_center_text, draw_legend=draw_legend,
            inner_tracks=inner_tracks, fold_minus_labels_out=fold_minus_labels_out,
            minus_label_color=minus_label_color, minus_label_mode=minus_label_mode,
            regions=regions, legend_below=legend_below,
            omit_trna_labels_when_crowded=omit_trna_labels_when_crowded, ax=ax,
        )


def _draw_mito_map(
    parsed: MitoGenome,
    genome_name: str,
    output_file: str | Path | None,
    *,
    width: float | None,
    height: float | None,
    dpi: int,
    colors: dict[str, str] | None,
    min_label_gap: float | None,
    label_cex: tuple[float, float, float],
    title: str,
    draw_gc: bool,
    draw_center_text: bool,
    draw_legend: bool,
    inner_tracks: Any | None,
    fold_minus_labels_out: bool,
    minus_label_color: str,
    minus_label_mode: str,
    regions: dict[str, dict[str, int]] | None,
    legend_below: bool,
    omit_trna_labels_when_crowded: bool,
    ax: Any | None,
) -> Path | None:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Patch, Wedge  # type: ignore

    colors = dict(colors or GENE_COLORS)
    L = parsed.genome_length
    plus = [g for g in parsed.genes if g.strand == 1]
    minus = [g for g in parsed.genes if g.strand == -1]

    if ax is None:
        default_w, default_h = default_map_size(title=bool(title), legend_below=legend_below)
        if width is not None and height is None:
            height = width * default_h / default_w
        fig = plt.figure(figsize=(width or default_w, height or default_h))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    else:
        fig = ax.figure
    ax.set_aspect("equal")
    top = (_YLIM_TITLE if title else _YLIM)[1]
    if legend_below:
        ax.set_xlim(*_XLIM_LEGEND_BELOW)
        ax.set_ylim(_YLIM_LEGEND_BELOW_BOTTOM if draw_legend else _YLIM_PANEL_BOTTOM, top)
    else:
        ax.set_xlim(*_XLIM)  # extra left margin so the legend clears gene labels
        ax.set_ylim(_YLIM[0], top)
    ax.axis("off")
    unit_in = _inches_per_unit(ax)

    inner_shift = 0.0  # set by the two-ring layout when inward names need room
    cex_by_cat = {"tRNA": label_cex[0], "rRNA": label_cex[1]}
    other_cex = label_cex[2]
    label_font_pt = max(label_cex) * 11
    floor_gap = max(min_label_gap, L / 90.0) if min_label_gap is not None else 0.0

    # backbone circle between the two strand rings (thin but clearly visible)
    ax.add_patch(
        Wedge(
            (0, 0),
            _R_PLUS_BLOCK_IN + 0.002,
            0,
            360,
            width=0.008,
            facecolor="0.2",
            edgecolor="none",
            zorder=0,
        )
    )

    # gene blocks: plus outside, minus inside the backbone
    for g in plus:
        _draw_gene_blocks(ax, g, colors, L, _R_PLUS_BLOCK_IN, _R_PLUS_BLOCK_OUT)
    for g in minus:
        _draw_gene_blocks(ax, g, colors, L, _R_MINUS_BLOCK_IN, _R_MINUS_BLOCK_OUT)

    # Labels are strand-aware: plus-strand (outer ring) radiate outward,
    # minus-strand (inner ring) radiate inward. Each strand is de-overlapped
    # independently; the gap follows the font height on that ring, and a ring
    # with more labels than fit shrinks its font instead of overprinting.
    if fold_minus_labels_out:
        # Both strands labelled on the outer ring; inner-strand (minus) genes
        # are tinted grey so they remain identifiable without a second ring.
        entries_c: list[tuple[float, str, float, float, str, bool]] = []
        for g in parsed.genes:
            cex = cex_by_cat.get(g.category, other_cex)
            for exon_index, (bp, name) in enumerate(_gene_label_points(g)):
                strand = (
                    g.exon_strands[exon_index]
                    if g.exon_strands and _is_trans_spliced(g)
                    else g.strand
                )
                is_plus = strand == 1
                r_block = _R_PLUS_BLOCK_OUT if is_plus else _R_MINUS_BLOCK_OUT
                color = "#111111" if is_plus else minus_label_color
                entries_c.append((bp, name, cex, r_block, color, is_plus, g.category))
        entries_c.sort(key=lambda e: e[0])
        gap, fit = _label_gap_and_fit(
            len(entries_c), L, _R_LABEL_MINUS_HUG, unit_in, label_font_pt, floor_gap
        )
        ax.ov_trna_labels_omitted = False
        if omit_trna_labels_when_crowded and label_font_pt * fit < _CROWDED_PT:
            entries_c = [e for e in entries_c if e[6] != "tRNA"]
            ax.ov_trna_labels_omitted = True
            gap, fit = _label_gap_and_fit(
                len(entries_c), L, _R_LABEL_MINUS_HUG, unit_in, label_font_pt, floor_gap
            )
        positions, _ = spread_labels([e[0] for e in entries_c], L, gap, _SPREAD_PASSES)
        for i, (bp, name, cex, r_block, color, is_plus, _category) in enumerate(entries_c):
            cex *= fit
            if is_plus:
                _draw_label(
                    ax,
                    positions[i],
                    bp,
                    r_block,
                    _R_LABEL_OUT,
                    name,
                    cex,
                    L,
                    outward=True,
                    color=color,
                    draw_leader=True,
                )
            elif minus_label_mode == "on_ring":
                # tangential label sitting on the minus block, no leader
                _draw_label(
                    ax,
                    bp,
                    bp,
                    _R_MINUS_BLOCK_IN,
                    _R_MINUS_BLOCK_IN,
                    name,
                    cex,
                    L,
                    outward=False,
                    color=color,
                    draw_leader=False,
                )
            else:  # outer_gray — grey names on the outer anchor, same elbow leader
                _draw_label(
                    ax,
                    positions[i],
                    bp,
                    r_block,
                    _R_LABEL_MINUS_HUG,
                    name,
                    cex,
                    L,
                    outward=True,
                    color=color,
                    draw_leader=True,
                )
    else:
        # plus strand outward, minus strand inward; both rings use the tighter
        # ring's size so the map has one gene-name size
        from .style import text_width_pt

        rings = []
        for strand, r_block, r_anchor, outward in (
            (1, _R_PLUS_BLOCK_OUT, _R_LABEL_OUT, True),
            (-1, _R_MINUS_BLOCK_IN, _R_LABEL_IN, False),
        ):
            entries = _strand_label_entries(parsed, strand, cex_by_cat, other_cex)
            # Inward names converge: neighbours are closest at their inner ends,
            # so that ring is spaced at the radius a typical name reaches.
            r_pitch = r_anchor
            if not outward and entries:
                widths = sorted(
                    text_width_pt(f" {name} ", round(cex * 11, 2), italic=True)
                    for _bp, name, cex in entries
                )
                typical = widths[int(0.75 * (len(widths) - 1))] / 72 / unit_in
                r_pitch = max(0.3, r_anchor - typical)
            _gap, fit = _label_gap_and_fit(
                len(entries), L, r_pitch, unit_in, label_font_pt, floor_gap
            )
            rings.append((entries, r_block, r_anchor, outward, fit, r_pitch))
        fit = min(ring[4] for ring in rings)
        # inward names reach toward the centre: the IR band and GC ring recede to clear them

        inner_entries = rings[1][0]
        if inner_entries:
            reach = max(
                text_width_pt(f" {name} ", round(cex * fit * 11, 2), italic=True)
                for _bp, name, cex in inner_entries
            ) / 72 / unit_in
            top = _R_REGION_OUT if regions else _R_GC_OUT
            gc_floor = (_R_GC_IN_WITH_REGIONS if regions else _R_GC_IN) - _MIN_INNER_GC_RADIUS
            inner_shift = max(-gc_floor, min(0.0, _R_LABEL_IN - reach - 0.02 - top))
        for entries, r_block, r_anchor, outward, _fit, r_pitch in rings:
            pitch_pt = label_font_pt * fit / 72 * _LABEL_LINE
            # OGDrawR's fixed floor only while the ring can hold it: a floor the
            # labels cannot all keep leaves the spreader oscillating into overlaps.
            cap = _MAX_LABEL_FILL * L / max(1, len(entries))
            gap = max(min(floor_gap, cap), L * pitch_pt / (2 * math.pi * r_pitch * unit_in))
            positions, _ = spread_labels([e[0] for e in entries], L, gap, _SPREAD_PASSES)
            for i, (bp, name, cex) in enumerate(entries):
                _draw_label(
                    ax, positions[i], bp, r_block, r_anchor, name, cex * fit, L, outward=outward
                )

    # Optional inner tracks (e.g. pangenome variant rings) drawn on the same
    # Cartesian axes, using the OGDraw radial coordinate system.
    if inner_tracks is not None:
        inner_tracks(ax, L)

    if regions:
        _draw_regions(ax, regions, L, inner_shift)
    gc_out, gc_in = (
        (_R_GC_OUT_WITH_REGIONS, _R_GC_IN_WITH_REGIONS) if regions else (_R_GC_OUT, _R_GC_IN)
    )
    gc_out, gc_in = gc_out + inner_shift, gc_in + inner_shift

    # GC content ring: each window's deviation from the genome mean, outward
    # (darker) when richer, inward (lighter) when poorer, on a faint band.
    if draw_gc and parsed.gc:
        gc_vals = [w["gc"] for w in parsed.gc]
        gc_mean = sum(gc_vals) / len(gc_vals)
        spread = max(abs(v - gc_mean) for v in gc_vals) or 1.0
        band = gc_out - gc_in
        r_mean = gc_in + band / 2
        ax.add_patch(
            Wedge((0, 0), gc_out, 0, 360, width=band, facecolor=_GC_BAND, edgecolor="none")
        )
        for w in parsed.gc:
            r_val = r_mean + (band / 2) * (w["gc"] - gc_mean) / spread
            if abs(r_val - r_mean) < 1e-4:
                continue
            lo, hi, color = (r_mean, r_val, _GC_HIGH) if r_val > r_mean else (r_val, r_mean, _GC_LOW)
            _draw_arc_block(ax, w["start"], w["end"], lo, hi, L, color, lw=0.0, edge="none")
        ax.add_patch(
            Wedge((0, 0), r_mean + 0.0012, 0, 360, width=0.0024, facecolor="#555555",
                  edgecolor="none")
        )
        _draw_kb_ticks(ax, L, gc_in)

    # centre text: three centred lines, spaced by their own point size
    if draw_center_text:
        subtitle = _ORGANELLE_SUBTITLE.get(parsed.organelle, "organelle genome")
        line = 1.3 / 72 / unit_in  # data units per point of type, with leading
        name_pt = _FONT_NAME
        if draw_gc and parsed.gc:
            from .style import text_width_pt

            # inside the kb labels (inner GC edge less one tick-label height)
            free = 2 * (gc_in - 0.008 - _FONT_TICK * 1.2 / 72 / unit_in - 0.02) * unit_in * 72
            wide = text_width_pt(genome_name, _FONT_NAME, italic=True)
            if wide > free:
                name_pt = max(6.0, _FONT_NAME * free / wide)
        ax.text(0, line * (name_pt + _FONT_CENTRE) / 2, genome_name, ha="center",
                va="center", fontsize=name_pt, fontstyle="italic")
        ax.text(0, 0, subtitle, ha="center", va="center", fontsize=_FONT_CENTRE)
        ax.text(0, -line * _FONT_CENTRE, f"{L:,} bp", ha="center", va="center",
                fontsize=_FONT_CENTRE)
    if title:
        ax.text(0, 1.28, title, ha="center", va="center", fontsize=_FONT_TITLE)

    # legend: only the categories drawn on this map, plus the GC key
    if draw_legend:
        handles = legend_handles(parsed.organelle, colors, present=present_categories(parsed))
        if draw_gc and parsed.gc:
            handles.append(
                Patch(facecolor=_GC_HIGH, edgecolor="none", label="GC above / below mean")
            )
        ax.legend(
            handles=handles,
            loc="upper center" if legend_below else "center left",
            bbox_to_anchor=(0, -1.38) if legend_below else (_LEGEND_X, 0.0),
            bbox_transform=ax.transData,
            ncol=2 if legend_below else 1,
            frameon=False,
            fontsize=_FONT_LEGEND,
            handlelength=1.0,
            handleheight=0.8,
            labelspacing=0.3,
            columnspacing=1.5,
            borderaxespad=0,
            borderpad=0,
        )

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_file), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_file
    return None


def _strand_label_entries(parsed, strand: int, cex_by_cat, other_cex) -> list[tuple[float, str, float]]:
    """``(bp, name, cex)`` for every label of one strand, in genome order."""
    entries: list[tuple[float, str, float]] = []
    for g in parsed.genes:
        cex = cex_by_cat.get(g.category, other_cex)
        for exon_index, (bp, name) in enumerate(_gene_label_points(g)):
            exon_strand = (
                g.exon_strands[exon_index] if g.exon_strands and _is_trans_spliced(g) else g.strand
            )
            if exon_strand == strand:
                entries.append((bp, name, cex))
    entries.sort(key=lambda e: e[0])
    return entries


def two_ring_label_pt(parsed, unit_in: float, font_pt: float = _FONT_LABEL) -> float:
    """Gene-name size the two-ring map uses at ``unit_in`` inches per unit (>= 5 pt)."""
    fits = []
    for strand, radius in ((1, _R_LABEL_OUT), (-1, _R_LABEL_IN)):
        entries = _strand_label_entries(parsed, strand, {}, font_pt / 11)
        fits.append(_label_gap_and_fit(len(entries), parsed.genome_length, radius, unit_in, font_pt, 0.0)[1])
    return max(FONT_PT["min"], font_pt * min(fits))


@validated_public('visualization.plot_ogdraw_map')
def plot_ogdraw_map(
    genbank_path: Sequence[str | Path],
    *,
    genome_name: str | None = None,
    title: str = "",
    dpi: int = 600,
    width: float | None = None,
    height: float | None = None,
    gc_window: int = 200,
    show_ir: bool | None = None,
    **kwargs: Any,
) -> OrganellePlot:
    """One-step OGDraw-style map: parse a GenBank file then draw (OGDrawR port).

    ``genbank_path`` is a list of GenBank files; one file with several records, or several
    files, are drawn as one circle per record on a single figure. A bare path string is
    still accepted from Python and means one file. (The annotation is a list so the
    capability binding exposes a list-of-paths parameter.)

    ``show_ir`` draws the plastid LSC/IRb/SSC/IRa band: ``None`` (default) draws it
    for a single plastid record whose inverted repeats are found and skips it
    otherwise, ``True`` fails when it cannot be resolved, ``False`` never draws it.
    The resolved spans are in ``metrics["regions"]`` (0-based start, length).

    Compute-only; render with ``ov.write(plot, output)``. This is the
    ``method="ogdraw"`` backend of :func:`plot_genome_map`.
    """
    from .plot_object import plot_result

    resolved_name = (
        genome_name if genome_name is not None else _genbank_display_name(genbank_path)
    )
    regions = _ogdraw_regions(genbank_path, show_ir)

    def _render(path: str | Path) -> Path:
        return _render_ogdraw_map(
            genbank_path,
            path,
            genome_name=resolved_name,
            title=title,
            dpi=dpi,
            width=width,
            height=height,
            gc_window=gc_window,
            regions=regions,
            **kwargs,
        )

    plot = plot_result(
        "plot_ogdraw_map",
        _render,
        metrics={"render_method": "ogdraw", "genome_name": resolved_name, "regions": regions},
        flags=("ogdraw_map",),
        summary="OGDraw organelle map prepared.",
    )
    plot._panel = ogdraw_panel(
        genbank_path, genome_name=resolved_name, gc_window=gc_window, regions=regions,
        colors=kwargs.get("colors"),
    )
    return plot


def present_categories(parsed: MitoGenome) -> set[str]:
    """Categories a map actually draws: its genes' classes, plus introns when a cis gene has one."""
    present = {g.category for g in parsed.genes}
    if any(g.has_intron and len(g.exons) >= 2 and not _is_trans_spliced(g) for g in parsed.genes):
        present.add("intron")
    return present


def legend_handles(
    organelle: str, colors: dict[str, str] | None = None, *, present: set[str] | None = None
) -> list[Any]:
    """Category legend entries (matplotlib patches) for one organelle's map.

    ``present`` keeps only those categories (a map lists what it draws, not the
    whole palette); ``None`` lists every category of the organelle.
    """
    from matplotlib.patches import Patch  # type: ignore

    palette = dict(colors or GENE_COLORS)
    return [
        Patch(facecolor=palette[key], edgecolor="black", linewidth=0.4, label=label)
        for key, label in legend_rows(organelle)
        if present is None or key in present
    ]


def _nice_step(length: int, target: int = 10) -> int:
    """A 1/2/5 x 10^k bp step giving about ``target`` ticks round the genome."""
    raw = max(1.0, length / target)
    power = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        if factor * power >= raw:
            return int(factor * power)
    return int(10 * power)


def _draw_kb_ticks(ax, genome_length: int, radius: float) -> None:
    """kb ticks on the GC ring's inner edge, labels just inside (0 at the top, clockwise).

    A tick closer than half a step to the end would crowd the 0 label, so it is skipped.
    """
    step = _nice_step(genome_length)
    for bp in range(0, genome_length, step):
        if bp and genome_length - bp < step / 2:
            continue
        a = math.radians(_bp_to_deg(bp, genome_length))
        ax.plot(
            [radius * math.cos(a), (radius + _R_TICK) * math.cos(a)],
            [radius * math.sin(a), (radius + _R_TICK) * math.sin(a)],
            color="0.35",
            lw=0.45,
            zorder=2,
        )
        label = "0" if bp == 0 else f"{bp / 1000:g} kb"
        r = radius - 0.008
        deg = math.degrees(a) % 360
        rot = deg - 90 if deg <= 180 else deg + 90  # tangential, never upside down
        ax.text(
            r * math.cos(a), r * math.sin(a), label, rotation=rot, rotation_mode="anchor",
            ha="center", va="top" if deg <= 180 else "bottom", fontsize=_FONT_TICK, color="0.35",
        )


def ogdraw_panel(genbank_path, *, genome_name, gc_window=200, regions=None, colors=None):
    """A :class:`~.plot_object.PanelDrawer` for a single-record map, else ``None``."""
    from .plot_object import PanelDrawer

    loaded = _load_genbank_records(genbank_path)
    if len(loaded) != 1:
        return None
    record = loaded[0][1]
    organelle = record_organelle(record)
    header = f"{genome_name} · {len(record):,} bp · {organelle}"

    def draw(ax, legend: bool, label_pt: float | None = None):
        # OGDraw layout as in the standalone map: plus-strand names outside, minus-strand
        # names inward inside the ring; the name moves from the centre to the header.
        # ``label_pt`` (from the composer) sets one gene-name size across panels.
        parsed = parse_genbank(_genbank_paths(genbank_path)[0], gc_window=gc_window)
        draw_mito_map(
            parsed, genome_name=genome_name, ax=ax, colors=colors, regions=regions,
            draw_legend=legend, legend_below=True, draw_center_text=False,
            fold_minus_labels_out=False, min_label_gap=None, omit_trna_labels_when_crowded=True,
            label_cex=((label_pt or _FONT_LABEL) / 11,) * 3,
        )
        if getattr(ax, "ov_trna_labels_omitted", False):
            ax.text(0.0, 0.0, "tRNA genes drawn, not labelled (space)", transform=ax.transAxes,
                    fontsize=FONT_PT["min"] + 0.5, color="0.4", ha="left", va="bottom")
        return [] if legend else legend_handles(
            parsed.organelle, colors, present=present_categories(parsed)
        )

    def label_pt(ax) -> float:
        """Gene-name size this panel would use on ``ax`` (same window as ``draw``)."""
        ax.set_aspect("equal")
        ax.set_xlim(*_XLIM_LEGEND_BELOW)
        ax.set_ylim(_YLIM_PANEL_BOTTOM, _YLIM[1])
        parsed = parse_genbank(_genbank_paths(genbank_path)[0], gc_window=0)
        return two_ring_label_pt(parsed, _inches_per_unit(ax))

    span_x = _XLIM_LEGEND_BELOW[1] - _XLIM_LEGEND_BELOW[0]
    return PanelDrawer(
        draw, aspect=(_YLIM[1] - _YLIM_PANEL_BOTTOM) / span_x, header=header, label_pt=label_pt
    )


def _ogdraw_regions(genbank_path, show_ir: bool | None) -> dict[str, dict[str, int]]:
    """Quadripartite spans for a single-record map; several records never get a band."""
    if show_ir is False:
        return {}
    from .features import resolve_ir_regions

    loaded = _load_genbank_records(genbank_path)
    if len(loaded) != 1:
        if show_ir:
            from ..core.errors import OrganelleParameterError

            raise OrganelleParameterError(
                code="visualization.structure.ir_single_record",
                message="IR partitioning requires a single-record GenBank.",
            )
        return {}
    record = loaded[0][1]
    return resolve_ir_regions(record, record_organelle(record), show_ir)


def _render_ogdraw_map(
    genbank_path: str | Path,
    output: str | Path,
    *,
    genome_name: str,
    title: str = "",
    dpi: int = 600,
    width: float | None = None,
    height: float | None = None,
    gc_window: int = 200,
    regions: dict[str, dict[str, int]] | None = None,
    **kwargs: Any,
) -> Path:
    """Private path-taking renderer backing :func:`plot_ogdraw_map`.

    Several GenBank files, or one file with several records, are drawn as a grid
    of circles on a single figure (see :func:`_render_ogdraw_grid`).
    """
    loaded = _load_genbank_records(genbank_path)
    if len(loaded) > 1:
        return _render_ogdraw_grid(
            loaded,
            output,
            genome_name=genome_name,
            title=title,
            dpi=dpi,
            colors=kwargs.get("colors"),
        )
    parsed = parse_genbank(_genbank_paths(genbank_path)[0], gc_window=gc_window)
    out = draw_mito_map(
        parsed,
        genome_name=genome_name,
        output_file=output,
        width=width,
        height=height,
        dpi=dpi,
        title=title,
        regions=regions,
        **kwargs,
    )
    return Path(out) if out is not None else Path(output)


def _render_ogdraw_grid(
    loaded: list[tuple[str, Any]],
    output: str | Path,
    *,
    genome_name: str,
    title: str = "",
    dpi: int = 300,
    colors: dict[str, str] | None = None,
    max_columns: int = 2,
) -> Path:
    """One circular map per record on a double-column figure with one shared legend.

    Panels are lettered a, b, c in reading order; each carries a left-aligned
    header (record, length, organelle) on the panel letter's baseline.
    """
    import math

    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    from .style import add_panel_label

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = min(len(loaded), max_columns)
    rows = math.ceil(len(loaded) / columns)
    panel_w = width_inches("double") / columns
    xlim_span = _XLIM_LEGEND_BELOW[1] - _XLIM_LEGEND_BELOW[0]
    panel_h = panel_w * (_YLIM[1] - _YLIM[0]) / xlim_span
    with publication_style():
        fig, axes = plt.subplots(
            rows, columns, figsize=(panel_w * columns, panel_h * rows), squeeze=False,
            gridspec_kw={"left": 0.03, "right": 0.99, "top": 0.97, "bottom": 0.02,
                         "wspace": 0.04, "hspace": 0.08},
        )
        organelles: list[str] = []
        try:
            for index, (name, record) in enumerate(loaded):
                ax = axes[index // columns][index % columns]
                parsed = _parse_genbank_record(record, gc_window=0)
                organelles.append(parsed.organelle)
                draw_mito_map(
                    parsed,
                    genome_name=name,
                    ax=ax,
                    colors=colors,
                    draw_gc=False,
                    draw_center_text=False,
                    draw_legend=False,
                    legend_below=True,
                    min_label_gap=None,
                    fold_minus_labels_out=False,
                )
                ax.set_title(
                    f"{name} · {parsed.genome_length:,} bp · {parsed.organelle}",
                    loc="left", fontsize=FONT_PT["body"], pad=2,
                )
                add_panel_label(ax, "abcdefghijklmnopqrstuvwxyz"[index % 26])
            for index in range(len(loaded), rows * columns):
                axes[index // columns][index % columns].axis("off")
            # one legend below the grid serves every panel (records may mix organelles)
            palette = dict(colors or GENE_COLORS)
            labels: dict[str, str] = {}
            for organelle in dict.fromkeys(organelles):
                for key, label in legend_rows(organelle):
                    labels.setdefault(label, palette[key])
            from matplotlib.patches import Patch  # type: ignore

            fig.legend(
                handles=[
                    Patch(facecolor=c, edgecolor="black", label=text) for text, c in labels.items()
                ],
                loc="upper left",
                bbox_to_anchor=(0.03, 0.0),
                ncol=4,
                fontsize=_FONT_LEGEND,
                handlelength=1.0,
                handleheight=0.8,
                columnspacing=1.2,
                frameon=False,
            )
            if title:
                fig.suptitle(title, x=0.03, ha="left", fontsize=_FONT_TITLE)
            fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
        finally:
            plt.close(fig)
    return output
