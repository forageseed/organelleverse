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

from .plot_object import OrganellePlot

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


def spread_labels(
    gene_mids: list[float], genome_length: float, min_gap_bp: float = 4500
) -> tuple[list[float], list[bool]]:
    """Iteratively push apart labels closer than ``min_gap_bp`` on the circle.

    Port of OGDrawR ``spread_labels``. Returns ``(pos, moved)`` in the *input
    order*; ``moved[i]`` marks labels displaced by more than 800 bp.
    """
    n = len(gene_mids)
    if n <= 1:
        return list(gene_mids), [False] * n

    order = sorted(range(n), key=lambda i: gene_mids[i])
    adj = [gene_mids[i] for i in order]
    orig = list(adj)

    for _ in range(20):
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


def _parse_genbank_record(record, gc_window: int = 200) -> MitoGenome:
    """Parse one already-loaded SeqRecord into a :class:`MitoGenome`."""
    sequence = str(record.seq).upper()
    genome_length = len(sequence)

    organelle = "mito"
    src = next((f for f in record.features if f.type == "source"), None)
    if src is not None:
        organ = " ".join(src.qualifiers.get("organelle", [""])).lower()
        if "plastid" in organ or "chloroplast" in organ:
            organelle = "plastid"

    genes: list[Gene] = []
    seen: set[str] = set()
    for feat in record.features:
        if feat.type != "gene":
            continue
        name = (feat.qualifiers.get("gene", [None]) or [None])[0]
        if not name or name in seen:
            continue
        parts = list(feat.location.parts)
        # 1-based inclusive coords to match GenBank/OGDrawR display.
        exons = [(int(p.start) + 1, int(p.end)) for p in parts]
        has_intron = len(exons) > 1
        # min/max span avoids the trans-spliced end<start artefact in the R port.
        start = min(e[0] for e in exons)
        end = max(e[1] for e in exons)
        strand = -1 if feat.location.strand and feat.location.strand < 0 else 1
        seen.add(name)
        genes.append(
            Gene(
                name=name,
                start=start,
                end=end,
                strand=strand,
                has_intron=has_intron,
                exons=exons,
                category=classify_gene(name, organelle),
            )
        )

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
    paths = _genbank_paths(genbank_path)
    return paths[0].stem if len(paths) == 1 else f"{len(paths)} GenBank files"


def parse_genbank(gb_file: str | Path, gc_window: int = 200) -> MitoGenome:
    """Parse a GenBank file into a :class:`MitoGenome` (OGDrawR ``parse_genbank``).

    Uses Biopython for robust location parsing (compound ``join`` locations →
    exons, ``complement`` → minus strand). Duplicate gene names keep the first
    occurrence, matching the R implementation. Original identifier casing is
    preserved (gene classification is case-insensitive at match time only).
    Multi-record files are rejected here; structure maps fan them out per
    contig instead of concatenating.
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
_R_LABEL_OUT = 0.925  # plus-strand label anchor; text reads outward
_R_LABEL_MINUS_HUG = 0.902  # minus-strand (grey) labels hug the boundary ring
_R_PLUS_BLOCK_OUT = 0.90
_R_PLUS_BLOCK_IN = 0.862
_R_BACKBONE = 0.859
_R_MINUS_BLOCK_OUT = 0.857
_R_MINUS_BLOCK_IN = 0.820
_R_LABEL_IN = 0.80  # minus-strand label anchor; text reads inward
_R_GC_OUT = 0.52
_R_GC_IN = 0.38


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
        ax.plot(
            [r_block * math.cos(ga), r_anchor * math.cos(la)],
            [r_block * math.sin(ga), r_anchor * math.sin(la)],
            color="0.6",
            lw=0.3,
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
        color=color,
        zorder=3,
    )


def draw_mito_map(
    parsed: MitoGenome,
    genome_name: str = "Genome",
    output_file: str | Path | None = None,
    *,
    width: float = 12,
    height: float = 12,
    dpi: int = 600,
    colors: dict[str, str] | None = None,
    min_label_gap: float = 4500,
    label_cex: tuple[float, float, float] = (0.65, 0.70, 0.73),
    title: str = "",
    draw_gc: bool = True,
    draw_center_text: bool = True,
    draw_legend: bool = True,
    inner_tracks: Any | None = None,
    fold_minus_labels_out: bool = False,
    minus_label_color: str = "#9AA0A6",
    minus_label_mode: str = "outer_gray",
    ax: Any | None = None,
) -> Path | None:
    """Draw an OGDraw-style circular map from a :class:`MitoGenome`.

    Reproduces OGDrawR ``draw_mito_map``. If ``output_file`` is given the figure
    is written (format from extension: pdf/png/tiff/svg) and its path returned;
    otherwise the live matplotlib figure is left open and ``None`` returned.
    """
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Patch, Wedge  # type: ignore

    colors = dict(colors or GENE_COLORS)
    L = parsed.genome_length
    plus = [g for g in parsed.genes if g.strand == 1]
    minus = [g for g in parsed.genes if g.strand == -1]

    cex_by_cat = {"tRNA": label_cex[0], "rRNA": label_cex[1]}
    other_cex = label_cex[2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 1.5)  # extra left margin so the legend clears gene labels
    ax.set_ylim(-1.62, 1.42)
    ax.axis("off")

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
    # independently; the gap auto-scales to the genome so dense clusters that a
    # fixed bp gap can't separate still fan out legibly.
    gap = max(min_label_gap, L / 90.0)
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
                entries_c.append((bp, name, cex, r_block, color, is_plus))
        entries_c.sort(key=lambda e: e[0])
        positions, _ = spread_labels([e[0] for e in entries_c], L, gap)
        for i, (bp, name, cex, r_block, color, is_plus) in enumerate(entries_c):
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
            else:  # outer_gray — grey labels hugging the boundary ring, no gap
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
                    draw_leader=False,
                )
    else:
        for strand, r_block, r_anchor, outward, cex_scale in (
            (1, _R_PLUS_BLOCK_OUT, _R_LABEL_OUT, True, 1.0),
            (-1, _R_MINUS_BLOCK_IN, _R_LABEL_IN, False, 1.0),  # inner = outer font size
        ):
            entries: list[tuple[float, str, float]] = []
            for g in parsed.genes:
                cex = cex_by_cat.get(g.category, other_cex) * cex_scale
                for exon_index, (bp, name) in enumerate(_gene_label_points(g)):
                    exon_strand = (
                        g.exon_strands[exon_index]
                        if g.exon_strands and _is_trans_spliced(g)
                        else g.strand
                    )
                    if exon_strand == strand:
                        entries.append((bp, name, cex))
            entries.sort(key=lambda e: e[0])
            positions, _ = spread_labels([e[0] for e in entries], L, gap)
            for i, (bp, name, cex) in enumerate(entries):
                _draw_label(ax, positions[i], bp, r_block, r_anchor, name, cex, L, outward=outward)

    # Optional inner tracks (e.g. pangenome variant rings) drawn on the same
    # Cartesian axes, using the OGDraw radial coordinate system.
    if inner_tracks is not None:
        inner_tracks(ax, L)

    # GC content ring (inverted bars + mean line, à la OGDrawR)
    if draw_gc and parsed.gc:
        gc_vals = [w["gc"] for w in parsed.gc]
        gc_mean = sum(gc_vals) / len(gc_vals)
        gc_min = min(gc_vals) - 0.01
        gc_max = max(gc_vals) + 0.01
        rng = (gc_max - gc_min) or 1.0
        band = _R_GC_OUT - _R_GC_IN
        ax.add_patch(
            Wedge((0, 0), _R_GC_OUT, 0, 360, width=band, facecolor="#E0E0E0", edgecolor="none")
        )
        for w in parsed.gc:
            frac = (w["gc"] - gc_min) / rng
            r_bar_in = _R_GC_OUT - band * frac
            _draw_arc_block(
                ax, w["start"], w["end"], r_bar_in, _R_GC_OUT, L, "#9A9A9A", lw=0.0, edge="none"
            )
        r_mean = _R_GC_OUT - band * ((gc_mean - gc_min) / rng)
        ax.add_patch(
            Wedge(
                (0, 0), r_mean + 0.0015, 0, 360, width=0.003, facecolor="#444444", edgecolor="none"
            )
        )

    # centre text
    if draw_center_text:
        subtitle = _ORGANELLE_SUBTITLE.get(parsed.organelle, "organelle genome")
        ax.text(0, 0.045, genome_name, ha="center", va="center", fontsize=12, fontstyle="italic")
        ax.text(0, -0.02, subtitle, ha="center", va="center", fontsize=9.5)
        ax.text(0, -0.075, f"{L:,} bp", ha="center", va="center", fontsize=9.5)
    if title:
        ax.text(0, 1.28, title, ha="center", va="center", fontsize=14)

    # legend
    if draw_legend:
        handles = [
            Patch(facecolor=colors[key], edgecolor="black", label=lab)
            for key, lab in legend_rows(parsed.organelle)
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(-2.18, -0.55),
            bbox_transform=ax.transData,
            frameon=False,
            fontsize=8.5,
            handlelength=1.0,
            labelspacing=0.25,
            borderaxespad=0,
        )

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_file), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_file
    return None


def plot_ogdraw_map(
    genbank_path: Sequence[str | Path],
    *,
    genome_name: str | None = None,
    title: str = "",
    dpi: int = 600,
    width: float = 12,
    height: float = 12,
    gc_window: int = 200,
    **kwargs: Any,
) -> OrganellePlot:
    """One-step OGDraw-style map: parse a GenBank file then draw (OGDrawR port).

    ``genbank_path`` is a list of GenBank files; one file with several records, or several
    files, are drawn as one circle per record on a single figure. A bare path string is
    still accepted from Python and means one file. (The annotation is a list so the
    capability binding exposes a list-of-paths parameter.)

    Compute-only; render with ``ov.write(plot, output)``. This is the
    ``method="ogdraw"`` backend of :func:`plot_genome_map`.
    """
    from .plot_object import plot_result

    resolved_name = (
        genome_name if genome_name is not None else _genbank_display_name(genbank_path)
    )

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
            **kwargs,
        )

    return plot_result(
        "plot_ogdraw_map",
        _render,
        metrics={"render_method": "ogdraw", "genome_name": resolved_name},
        flags=("ogdraw_map",),
        summary="OGDraw organelle map prepared.",
    )


def _render_ogdraw_map(
    genbank_path: str | Path,
    output: str | Path,
    *,
    genome_name: str,
    title: str = "",
    dpi: int = 600,
    width: float = 12,
    height: float = 12,
    gc_window: int = 200,
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
    max_columns: int = 3,
) -> Path:
    """One circular map per record, side by side on one figure with one shared legend."""
    import math

    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = min(len(loaded), max_columns)
    rows = math.ceil(len(loaded) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(8 * columns, 7.5 * rows), squeeze=False)
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
                min_label_gap=parsed.genome_length / 150,
                fold_minus_labels_out=True,
            )
            ax.set_title(f"{name}\n{parsed.genome_length:,} bp · {parsed.organelle}", fontsize=13)
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
            handles=[Patch(facecolor=c, edgecolor="black", label=text) for text, c in labels.items()],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(6, len(labels)),
            fontsize=9,
            frameon=False,
        )
        fig.suptitle(title or f"{genome_name} · {len(loaded)} records", fontsize=14)
        fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output
