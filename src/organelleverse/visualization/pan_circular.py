"""Circular pangenome overview plots for organelle genomes.

``plot_pan_circular`` draws the five-layer pangenome circle used in organelle
pangenome papers:

I/II  genes of the reference on the forward / reverse strand (the OGDraw
      ring, with every copy kept, exons on their own strand and trans-splicing
      read or inferred exactly as in the OGDraw and structure maps), plus the
      LSC/IRb/SSC/IRa band for a plastid reference;
III   one heatmap row per genome: variants per window against the reference,
      log10(n + 1), with windows where the genome has no call shown apart;
IV    nucleotide diversity (pi) of SNPs and of indels per window, computed from
      the genotypes, optionally with the reference GC content;
V     the pangenome graph itself, collapsed to its large-scale blocks and drawn
      by OrganelleVerse (``pan_graph``), coloured core / shell / private.

The call is compute-only: every number the figure shows is computed now and
kept in ``metrics``; ``ov.write(plot, "figure.png"|".svg"|".pdf")`` draws it,
and a serialized Result redraws through the visualization writer.
"""

from __future__ import annotations

import csv
import gzip
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import OrganelleDependencyError, OrganelleInputError, OrganelleParameterError
from .plot_object import OrganellePlot, plot_result
from .style import FONT_PT, JOURNAL_WIDTHS_MM, MAX_HEIGHT_MM, MM_PER_INCH, publication_style

#: Variants differing in length by at least this many bp count as SVs.
SV_MIN_LENGTH = 50

_ORGANELLE_ALIASES = {
    "mito": "mito",
    "mitochondrion": "mito",
    "mitochondrial": "mito",
    "mitogenome": "mito",
    "plastid": "plastid",
    "chloroplast": "plastid",
    "plastome": "plastid",
}
_ORGANELLE_WORD = {"mito": "mitochondrial", "plastid": "plastid"}

# Ring geometry in OGDraw data units (gene blocks span 0.82-0.90).
_R_TICK_OUT = 0.817
_R_TICK_IN = 0.808
_R_TICK_LABEL = 0.773  # labelled positions carry no tick stroke through the text
_R_IR = (0.706, 0.738)
_R_TRACKS_OUT = 0.746
_R_TRACKS_OUT_IR = 0.694
_R_TRACKS_IN = 0.462
_R_CENTER = 0.44
_RING_GAP = 0.016
_NUMERAL_DEG = 92.2  # just before the origin, so numerals sit in one column
_VIEW = 1.3

# Printed sizes (visualization/style.py): the figure is drawn at 183 mm, so
# these are the point sizes on the journal page.
_PT_HEADING = FONT_PT["title"]  # legend section headings, bold
_PT_BODY = FONT_PT["body"]
_PT_TICK = 6.5  # kb scale, colour-bar ticks
_PT_LABEL = 6.5  # graph block labels, sample rows
_PT_REGION = 6.5
# Layout in millimetres: circle panel, gene-category strip under it, key column.
_WIDTH_MM = JOURNAL_WIDTHS_MM["double"]
_CIRCLE_MM = 120.0
_GUTTER_MM = 2.0
_STRIP_ROW_MM = 3.3
_STRIP_COLUMNS = 3

_SNP_COLOR = "#1B7837"
_INDEL_COLOR = "#762A83"
_GC_COLOR = "#5F6B73"
_SV_COLOR = "#A63603"
_INK = "#1F2933"
_MUTED = "#5B6573"


def plot_pan_circular(
    reference_gbk: str | Path,
    *,
    variants: str | Path | None = None,
    variant_matrix: str | Path | None = None,
    diversity: str | Path | None = None,
    sv_variants: str | Path | None = None,
    pan_segments: str | Path | None = None,
    pangenome_gfa: str | Path | None = None,
    metadata: Mapping[str, str | int | float | bool | None] | None = None,
    window_size: int | None = None,
    bins: int = 120,
    max_heatmap_rows: int = 220,
    label_limit: int = 70,
    outer_method: str = "ogdraw",
    graph_renderer: str = "ov",
    graph_max_blocks: int = 30,
    graph_overview_blocks: int = 12,
    gc: bool = False,
    show_ir: bool | None = None,
    sample_labels: Mapping[str, str] | None = None,
    bandage_executable: str | Path | None = None,
    bandage_runner: Any | None = None,
    bandage_labels: tuple[str, ...] = ("name", "length", "depth"),
    bandage_color_by: str = "depth",
    title: str | None = None,
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
    write_intermediates: bool = True,
) -> OrganellePlot:
    """Draw a circular mitochondrial/plastid pangenome overview (compute-only).

    ``variants`` is a multi-sample VCF against the reference (``vg
    deconstruct`` / ``pggb -V`` output; plain or gzipped) or a table with
    accession/position/type columns. From a VCF, ring III counts each
    genome's non-reference calls per window and ring IV is the per-site
    nucleotide diversity pi of SNP and of indel sites, computed from the
    genotypes (the reference haplotype counts as one more genome when the VCF
    leaves it out, as ``vg deconstruct`` does). Records nested in a parent
    snarl that is also present (``PS``) are skipped so nothing is counted
    twice. From a table, ring IV is calls per genome per window, labelled as
    such. ``variant_matrix`` / ``diversity`` supply rings III / IV directly.

    ``pangenome_gfa`` fills the centre: ``graph_renderer="ov"`` (default)
    collapses the graph to at most ``graph_max_blocks`` blocks and draws it
    natively; ``"bandage"`` renders it with Bandage instead (an error when
    Bandage is not installed); ``"none"`` leaves the centre to
    ``pan_segments`` or the genome name. ``gc`` overlays the reference GC
    content on ring IV; ``show_ir`` controls the plastid IR band (``None``:
    drawn when a plastid reference has one).

    Returns an :class:`OrganellePlot`; ``ov.write(plot, output)`` writes the
    figure at ``output`` and, with ``write_intermediates``, the matrix,
    diversity and graph-block tables into ``output`` with its suffix removed.
    ``label_limit`` is accepted for compatibility and unused.
    """
    del label_limit, bandage_labels, bandage_color_by
    if outer_method not in {"ogdraw", "none", ""}:
        raise OrganelleParameterError(
            code="visualization.pan_circular.outer_method",
            message="outer_method must be 'ogdraw' or 'none'.",
            details={"outer_method": outer_method},
        )
    if graph_renderer not in {"ov", "bandage", "none"}:
        raise OrganelleParameterError(
            code="visualization.pan_circular.graph_renderer",
            message="graph_renderer must be 'ov', 'bandage' or 'none'.",
            details={"graph_renderer": graph_renderer},
        )
    from Bio import SeqIO  # type: ignore

    from .features import annotation_rows, record_display_name, resolve_ir_regions
    from .ogdraw import record_organelle

    ref = Path(reference_gbk)
    record = SeqIO.read(str(ref), "genbank")
    meta = dict(metadata or {})
    organelle = _ORGANELLE_ALIASES.get(str(meta.get("organelle", "")).lower()) or record_organelle(
        record
    )
    genome_length = len(record.seq)
    loci = annotation_rows(record, organelle, strict=False) if outer_method == "ogdraw" else []
    regions = resolve_ir_regions(record, organelle, show_ir)
    species = str(meta.get("species") or record_display_name(record, ref.stem))
    reference = str(meta.get("reference") or record.id)

    flags = ["pan_circular_plot"]
    if variant_matrix:
        tracks = _matrix_tracks(Path(variant_matrix), genome_length)
    else:
        starts, ends = _make_bins(genome_length, bins=bins, window_size=window_size)
        if variants and _is_vcf(Path(variants)):
            tracks = _vcf_tracks(Path(variants), str(record.id), starts, ends)
            flags.append("vcf_input")
        elif variants:
            tracks = _table_tracks(Path(variants), starts, ends)
        else:
            tracks = _empty_tracks(starts, ends)
    if diversity:
        tracks["snp_pi"], tracks["indel_pi"] = _read_diversity(
            Path(diversity), len(tracks["starts"])
        )
        tracks["diversity_kind"] = "supplied"
    sv_bins = None
    sv_count = tracks.get("sv_sites", 0)
    if sv_variants:
        sv_bins, sv_count = _load_sv_density(Path(sv_variants), tracks["starts"], tracks["ends"])
    gc_bins = _gc_bins(str(record.seq), tracks["starts"], tracks["ends"]) if gc else None

    labels = dict(sample_labels or {})
    order = _row_order(tracks["counts"])
    rows_total = len(order)
    order = order[:max_heatmap_rows]
    heatmap = {
        "samples": [tracks["samples"][i] for i in order],
        "labels": [labels.get(tracks["samples"][i], tracks["samples"][i]) for i in order],
        "counts": [list(tracks["counts"][i]) for i in order],
        "no_call": [list(tracks["no_call"][i]) for i in order],
    }
    no_call_records = tracks.get("no_call_records")
    coverage = None
    if pangenome_gfa and heatmap["samples"] and tracks["unit"] == "sites":
        from .pan_graph import path_window_coverage

        shares = path_window_coverage(
            Path(pangenome_gfa), str(record.id), tracks["starts"], tracks["ends"]
        )
        if all(sample in shares for sample in heatmap["samples"]):
            coverage = [shares[sample] for sample in heatmap["samples"]]
            flags.append("coverage_from_graph")
    heatmap["state"], heatmap["summary"] = _cell_states(
        heatmap,
        [list(no_call_records[i]) for i in order] if no_call_records is not None else None,
        coverage,
    )

    graph = None
    graph_layout_points = None
    overview = None
    overview_layout = None
    center = "none"
    if pangenome_gfa and graph_renderer == "ov":
        from .pan_graph import pangenome_skeleton, skeleton_layout

        def rounded_layout(skeleton):
            return {
                name: [[round(float(x), 3), round(float(y), 3)] for x, y in pts]
                for name, pts in skeleton_layout(skeleton).items()
            }

        graph = pangenome_skeleton(
            Path(pangenome_gfa), max_blocks=graph_max_blocks, reference=str(record.id)
        )
        graph_layout_points = rounded_layout(graph)
        # The centre is a simplified overview; the full skeleton is its own file.
        overview = pangenome_skeleton(
            Path(pangenome_gfa),
            max_blocks=min(graph_overview_blocks, graph_max_blocks),
            growth=1.15,  # finer steps, so the overview lands near its block budget
            reference=str(record.id),
        )
        overview_layout = rounded_layout(overview)
        center = "ov_graph"
    elif pangenome_gfa and graph_renderer == "bandage":
        from .gfa_graph import _find_bandage

        if _find_bandage(bandage_executable, runner=bandage_runner) is None:
            raise OrganelleDependencyError(
                code="visualization.missing_bandage",
                message=(
                    "graph_renderer='bandage' needs Bandage/Bandage-NG; install it "
                    "(`mamba install -c bioconda bandage`) or use graph_renderer='ov'."
                ),
                details={"executable": str(bandage_executable or "")},
            )
        center = "bandage"
    segments = _read_pan_segments(Path(pan_segments), genome_length) if pan_segments else []
    if center == "none" and segments:
        center = "segments"

    accessions = int(meta.get("accessions") or (graph or {}).get("genomes") or 0) or (
        len(tracks["samples"]) + (1 if tracks.get("reference_haplotype") else 0)
    )
    word = _ORGANELLE_WORD.get(organelle, "organelle")
    metrics: dict[str, Any] = {
        "title": title if title is not None else f"{species} {word} pan-genome",
        "species": species,
        "reference": reference,
        "genome_length": genome_length,
        "organelle": organelle,
        "accessions": accessions,
        "loci": loci,
        "regions": regions,
        "bins": {"start": list(tracks["starts"]), "end": list(tracks["ends"])},
        "heatmap": heatmap,
        "heatmap_rows_total": rows_total,
        "snp_pi": _rounded(tracks["snp_pi"]),
        "indel_pi": _rounded(tracks["indel_pi"]),
        "diversity_kind": tracks["diversity_kind"],
        "gc": _rounded(gc_bins) if gc_bins is not None else None,
        "sv": [float(v) for v in sv_bins] if sv_bins is not None else None,
        "graph": graph,
        "graph_layout": graph_layout_points,
        "graph_overview": overview,
        "graph_overview_layout": overview_layout,
        "graph_renderer": graph_renderer if pangenome_gfa else "none",
        "gfa": str(pangenome_gfa) if center == "bandage" else None,
        "segments": segments,
        "variant_unit": tracks["unit"],
        "reference_haplotype": tracks.get("reference_haplotype"),
        "dpi": dpi,
        "figsize": list(figsize) if figsize else _journal_size(loci, organelle),
        "write_intermediates": write_intermediates,
        # Summary counts (sites from a VCF, calls from a table).
        "features": len(loci),
        "forward_features": sum(1 for row in loci if row["exons"][0]["strand"] >= 0),
        "reverse_features": sum(1 for row in loci if row["exons"][0]["strand"] < 0),
        "variant_bins": len(tracks["starts"]),
        "variant_samples": len(tracks["samples"]),
        "snp_count": tracks["snp_count"],
        "indel_count": tracks["indel_count"],
        "sv_count": int(sv_count),
        "pan_segments": len(segments),
        "center_image": center,
        "outer_renderer": "ogdraw" if loci else "none",
        "inner_renderer": {"ov_graph": "ov_graph", "bandage": "bandage"}.get(center, "schematic"),
    }
    metrics["caption"] = _caption(metrics)
    if loci:
        flags += ["gene_structure_tracks", "ogdraw_outer"]
    if regions:
        flags.append("ir_band")
    if heatmap["samples"]:
        flags.append("variant_heatmap")
    if rows_total > len(order):
        flags.append("heatmap_rows_truncated")
    if np.any(tracks["snp_pi"]) or np.any(tracks["indel_pi"]):
        flags.append("diversity_curve")
    if tracks["diversity_kind"] == "pi":
        flags.append("pi_from_genotypes")
    if tracks.get("reference_haplotype"):
        flags.append("reference_haplotype_included")
    if gc_bins is not None:
        flags.append("gc_overlay")
    if sv_bins is not None and np.any(sv_bins):
        flags.append("sv_density_track")
    flags.append(
        {
            "ov_graph": "ov_graph_center",
            "bandage": "bandage_inner",
            "segments": "pan_schematic",
        }.get(center, "center_summary")
    )
    unit = "site(s)" if tracks["unit"] == "sites" else "call(s)"
    plot = plot_result(
        "plot_pan_circular",
        _renderer(metrics, bandage_executable, bandage_runner),
        organelle=organelle,
        method="ov_pan_circular",
        metrics=metrics,
        key_findings=(
            {"metric": "accessions", "value": accessions},
            {"metric": "genome_length", "value": genome_length},
            {"metric": "snp_count", "value": tracks["snp_count"]},
            {"metric": "indel_count", "value": tracks["indel_count"]},
        ),
        flags=tuple(flags),
        summary=(
            f"Pan-genome circular plot for {accessions} genome(s): "
            f"{tracks['snp_count']} SNP and {tracks['indel_count']} indel {unit}"
            + (
                f"; graph of {graph['raw_nodes']} nodes drawn as {len(graph['blocks'])} blocks."
                if graph
                else "."
            )
        ),
    )
    plot._panel = _panel_drawer(metrics, bandage_executable, bandage_runner)
    return plot


def _strip_mm(loci: Sequence[Mapping[str, Any]], organelle: str) -> float:
    """Height of the gene-category strip under the circle (0 without genes)."""
    from .ogdraw import legend_rows

    present = {row["category"] for row in loci}
    count = sum(1 for key, _ in legend_rows(organelle) if key in present)
    if not count:
        return 0.0
    return 9.5 + _STRIP_ROW_MM * math.ceil(count / _STRIP_COLUMNS)  # 2 note lines + rows


def _journal_size(loci: Sequence[Mapping[str, Any]], organelle: str) -> list[float]:
    """Double-column width; height = circle panel + gene strip (<= 170 mm)."""
    height = min(MAX_HEIGHT_MM, _CIRCLE_MM + _strip_mm(loci, organelle))
    return [round(_WIDTH_MM / MM_PER_INCH, 4), round(height / MM_PER_INCH, 4)]


def _renderer(metrics: Mapping[str, Any], executable: Any = None, runner: Any = None):
    def render(path: str | Path) -> Path:
        return _render_pan_circular(Path(path), metrics, executable, runner)

    return render


def _panel_drawer(metrics: Mapping[str, Any], executable: Any = None, runner: Any = None):
    """Draws the circle into one panel of ``compose_figure``; the keys go to the merged legend."""
    from .plot_object import PanelDrawer

    def draw(ax, legend: bool):
        from ..core.frozen import thaw_json

        data = thaw_json(metrics)
        info = _draw_circle(ax, data, None, executable, runner)
        handles = _legend_handles(data, info)
        if legend and handles:
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=3,
                fontsize=_PT_BODY,
                frameon=False,
                handlelength=1.0,
                borderaxespad=0,
            )
            return []
        return handles

    return PanelDrawer(draw, aspect=1.0, header=str(metrics.get("title", "")))


def _restore_pan_circular_plot(result) -> OrganellePlot:
    """Reattach rendering to a serialized, fully computed pan-circle Result."""
    plot = OrganellePlot.model_validate(result.model_dump(mode="python"))
    plot._renderer = _renderer(plot.metrics)
    plot._panel = _panel_drawer(plot.metrics)
    return plot


# ------------------------------------------------------------------ variants


def _is_vcf(path: Path) -> bool:
    if path.suffix in {".vcf"} or path.name.endswith(".vcf.gz"):
        return True
    try:
        with _open_text(path) as handle:
            return handle.readline().startswith("##fileformat=VCF")
    except (OSError, UnicodeDecodeError):
        return False


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def _chrom_matches(chrom: str, record_id: str) -> bool:
    base = record_id.split(".")[0]
    parts = chrom.split("#")
    return chrom == record_id or record_id in parts or base in (p.split(".")[0] for p in parts)


def _variant_class(ref: str, alts: Sequence[str]) -> str:
    kinds = set()
    for alt in alts:
        if alt.startswith("<") or "[" in alt or "]" in alt:
            kinds.add("sv")
        elif len(alt) == len(ref):
            kinds.add("snp")
        elif abs(len(alt) - len(ref)) >= SV_MIN_LENGTH:
            kinds.add("sv")
        else:
            kinds.add("indel")
    if "sv" in kinds:
        return "sv"
    return "indel" if "indel" in kinds else "snp"


def _vcf_tracks(path: Path, record_id: str, starts: Sequence[int], ends: Sequence[int]) -> dict:
    """Per-genome window counts and per-site pi of SNPs/indels from a VCF."""
    samples: list[str] = []
    records: list[list[str]] = []
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            if line.strip():
                records.append(line.rstrip("\n").split("\t"))
    if not records:
        return _empty_tracks(starts, ends, samples=samples, unit="sites", kind="pi")
    chroms = list(dict.fromkeys(rec[0] for rec in records))
    chosen = [c for c in chroms if _chrom_matches(c, record_id)]
    if not chosen:
        if len(chroms) != 1:
            raise OrganelleInputError(
                code="visualization.pan_circular.vcf_chrom",
                message="No VCF CHROM matches the reference record.",
                details={"record_id": record_id, "chroms": chroms[:10]},
            )
        chosen = chroms
    chrom = chosen[0]
    records = [rec for rec in records if rec[0] == chrom]
    ref_sample = chrom.split("#", 1)[0] if "#" in chrom else None
    add_reference = bool(ref_sample) and ref_sample not in samples
    ids = {rec[2] for rec in records if rec[2] != "."}

    n_bins = len(starts)
    counts = np.zeros((len(samples), n_bins), dtype=int)
    no_call = np.zeros((len(samples), n_bins), dtype=int)
    called = np.zeros((len(samples), n_bins), dtype=int)
    snp_pi = np.zeros(n_bins)
    indel_pi = np.zeros(n_bins)
    sites = {"snp": 0, "indel": 0, "sv": 0}
    bin_starts = np.asarray(starts)
    for rec in records:
        info = rec[7]
        parent = next((f[3:] for f in info.split(";") if f.startswith("PS=")), None)
        if parent is not None and parent in ids:
            continue  # nested in a snarl that is reported itself
        idx = int(np.searchsorted(bin_starts, int(rec[1]) - 1, side="right")) - 1
        idx = min(n_bins - 1, max(0, idx))
        kind = _variant_class(rec[3], rec[4].split(","))
        sites[kind] += 1
        alleles: list[int] = [0] if add_reference else []
        for s, field in enumerate(rec[9:]):
            gt = field.split(":", 1)[0]
            calls = [a for a in gt.replace("|", "/").split("/") if a not in {".", ""}]
            if not calls:
                no_call[s, idx] += 1
                continue
            called[s, idx] += 1
            values = [int(a) for a in calls]
            alleles.extend(values)
            if any(values):
                counts[s, idx] += 1
        n = len(alleles)
        if n >= 2 and kind != "sv":
            freqs = np.bincount(alleles) / n
            pi = n / (n - 1) * (1.0 - float((freqs**2).sum()))
            if kind == "snp":
                snp_pi[idx] += pi
            else:
                indel_pi[idx] += pi
    widths = np.maximum(1, np.asarray(ends) - bin_starts)
    return {
        "samples": samples,
        "starts": list(starts),
        "ends": list(ends),
        "counts": counts,
        # A window counts as "no call" for a genome only when every record there misses it.
        "no_call": ((no_call > 0) & (called == 0)).astype(int),
        "no_call_records": no_call,
        "snp_pi": snp_pi / widths,
        "indel_pi": indel_pi / widths,
        "snp_count": sites["snp"],
        "indel_count": sites["indel"],
        "sv_sites": sites["sv"],
        "unit": "sites",
        "diversity_kind": "pi",
        "reference_haplotype": ref_sample if add_reference else None,
    }


def _table_tracks(path: Path, starts: Sequence[int], ends: Sequence[int]) -> dict:
    rows = _read_table(path)
    sample_col = _column(rows, "accession", "sample", "individual", "id")
    pos_col = _column(rows, "position", "pos", "start")
    type_col = _column(rows, "type", "variant_type", "kind", "class", required=False)
    samples = list(dict.fromkeys(str(row.get(sample_col, "") or "sample") for row in rows))
    index = {sample: i for i, sample in enumerate(samples)}
    counts = np.zeros((len(samples), len(starts)), dtype=int)
    snp = np.zeros(len(starts))
    indel = np.zeros(len(starts))
    bin_starts = np.asarray(starts)
    for row in rows:
        pos = _safe_int(row.get(pos_col), default=1) - 1
        idx = min(len(starts) - 1, max(0, int(np.searchsorted(bin_starts, pos, side="right")) - 1))
        counts[index[str(row.get(sample_col, "") or "sample")], idx] += 1
        kind = str(row.get(type_col, "SNP") if type_col else "SNP").lower()
        if "indel" in kind or "ins" in kind or "del" in kind:
            indel[idx] += 1
        else:
            snp[idx] += 1
    denom = max(1, len(samples))
    return {
        "samples": samples,
        "starts": list(starts),
        "ends": list(ends),
        "counts": counts,
        "no_call": np.zeros_like(counts),
        "snp_pi": snp / denom,
        "indel_pi": indel / denom,
        "snp_count": int(snp.sum()),
        "indel_count": int(indel.sum()),
        "unit": "calls",
        "diversity_kind": "calls_per_genome",
    }


def _matrix_tracks(path: Path, genome_length: int) -> dict:
    rows = _read_table(path)
    if not rows:
        return _empty_tracks(*_make_bins(genome_length, bins=1, window_size=None))
    first = next(iter(rows[0]))
    labels = list(rows[0])[1:]
    starts, ends = _bins_from_labels(labels, genome_length)
    counts = np.array(
        [[_safe_float(row.get(label), default=0.0) for label in labels] for row in rows]
    )
    totals = counts.sum(axis=0)
    return {
        "samples": [str(row.get(first, "")) for row in rows],
        "starts": starts,
        "ends": ends,
        "counts": counts,
        "no_call": np.zeros_like(counts, dtype=int),
        "snp_pi": totals / max(1, len(rows)),
        "indel_pi": np.zeros(len(starts)),
        "snp_count": int(totals.sum()),
        "indel_count": 0,
        "unit": "calls",
        "diversity_kind": "calls_per_genome",
    }


def _empty_tracks(starts, ends, *, samples=(), unit="calls", kind="none") -> dict:
    zeros = np.zeros((len(samples), len(starts)), dtype=int)
    return {
        "samples": list(samples),
        "starts": list(starts),
        "ends": list(ends),
        "counts": zeros,
        "no_call": zeros,
        "snp_pi": np.zeros(len(starts)),
        "indel_pi": np.zeros(len(starts)),
        "snp_count": 0,
        "indel_count": 0,
        "unit": unit,
        "diversity_kind": kind,
    }


#: Heatmap cell states (metrics["heatmap"]["state"]).
STATE_SAME, STATE_VARIANT, STATE_MISSING, STATE_UNALIGNED, STATE_NO_CALL = range(5)
#: A genome whose path shares less than this of a window's reference bp did not align there.
MIN_ALIGNED_SHARE = 0.1


def _cell_states(heatmap, no_call_records, coverage):
    """State per genome and window, and per-genome counts for the report.

    With graph coverage: *variant* (>= 1 non-reference call), *unaligned* (the
    genome's path shares < 10 % of the window's reference bp, so nothing of it
    aligned there), *missing* (it aligned there, but at >= 1 site it carries no
    allele: VCF '.', the snarl lies off its path), else *same as reference*.
    Without a graph the two kinds of '.' cannot be told apart: *no call*.
    """
    states, summary = [], []
    for row, sample in enumerate(heatmap["samples"]):
        counts = heatmap["counts"][row]
        records = no_call_records[row] if no_call_records is not None else [0] * len(counts)
        row_states = []
        tally = {"variant": 0, "missing": 0, "unaligned": 0, "no_call": 0, "same": 0}
        miss_rec = {"aligned": 0, "unaligned": 0}
        for k, n in enumerate(counts):
            share = coverage[row][k] if coverage is not None else None
            if share is not None and share < MIN_ALIGNED_SHARE:
                miss_rec["unaligned"] += int(records[k])
            else:
                miss_rec["aligned"] += int(records[k])
            if n > 0:
                state, key = STATE_VARIANT, "variant"
            elif share is not None and share < MIN_ALIGNED_SHARE:
                state, key = STATE_UNALIGNED, "unaligned"
            elif coverage is not None and records[k] > 0:
                state, key = STATE_MISSING, "missing"
            elif coverage is None and heatmap["no_call"][row][k]:
                state, key = STATE_NO_CALL, "no_call"
            else:
                state, key = STATE_SAME, "same"
            row_states.append(state)
            tally[key] += 1
        states.append(row_states)
        summary.append(
            {
                "sample": sample,
                "label": heatmap["labels"][row],
                "windows_variant": tally["variant"],
                "windows_missing": tally["missing"],
                "windows_unaligned": tally["unaligned"],
                "windows_no_call": tally["no_call"],
                "windows_same": tally["same"],
                "no_call_records": int(sum(records)),
                "no_call_records_aligned": miss_rec["aligned"],
                "no_call_records_unaligned": miss_rec["unaligned"],
                "aligned_share": (
                    round(float(np.mean(coverage[row])), 4) if coverage is not None else None
                ),
            }
        )
    return states, summary


def _row_order(counts: np.ndarray) -> list[int]:
    """Heatmap rows ordered so similar genomes sit together (optimal leaf order)."""
    n = len(counts)
    if n < 3:
        return list(range(n))
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering

    data = np.log10(np.asarray(counts, dtype=float) + 1.0)
    if not np.any(np.ptp(data, axis=0)):
        return list(range(n))
    tree = linkage(data, method="average", metric="euclidean")
    return [int(i) for i in leaves_list(optimal_leaf_ordering(tree, data))]


def _gc_bins(sequence: str, starts: Sequence[int], ends: Sequence[int]) -> list[float]:
    seq = sequence.upper()
    out = []
    for a, b in zip(starts, ends, strict=False):
        window = seq[a:b]
        acgt = sum(window.count(c) for c in "ACGT") or 1
        out.append((window.count("G") + window.count("C")) / acgt)
    return out


def _rounded(values) -> list[float]:
    return [float(f"{float(v):.6g}") for v in values]


def _read_diversity(path: Path, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_table(path)
    snp_col = _column(rows, "snp", "snps", "snp_diversity", "pi_snp", "pi")
    indel_col = _column(rows, "indel", "indels", "indel_diversity", "pi_indel", required=False)
    snp = np.zeros(n_bins)
    indel = np.zeros(n_bins)
    for i, row in enumerate(rows[:n_bins]):
        snp[i] = _safe_float(row.get(snp_col), default=0.0)
        if indel_col:
            indel[i] = _safe_float(row.get(indel_col), default=0.0)
    return snp, indel


def _load_sv_density(
    path: Path, bin_starts: Sequence[int], bin_ends: Sequence[int]
) -> tuple[np.ndarray, int]:
    """Bin SV calls (a density table with an sv column, or a position list) onto the windows."""
    density = np.zeros(len(bin_starts), dtype=float)
    rows = _read_table(path)
    if not rows or not len(bin_starts):
        return density, 0
    sv_col = _column(rows, "sv", "svs", "sv_density", "count", "n_sv", required=False)
    if sv_col:
        for i, row in enumerate(rows[: len(bin_starts)]):
            density[i] = _safe_float(row.get(sv_col), default=0.0)
        return density, int(np.nansum(density))
    pos_col = _column(rows, "position", "pos", "start", "breakpoint")
    edges = np.asarray(bin_starts)
    total = 0
    for row in rows:
        pos = _safe_int(row.get(pos_col), default=-1)
        if pos < 0:
            continue
        idx = min(
            len(bin_starts) - 1, max(0, int(np.searchsorted(edges, pos - 1, side="right")) - 1)
        )
        density[idx] += 1.0
        total += 1
    return density, total


def _read_pan_segments(path: Path, genome_length: int) -> list[dict[str, Any]]:
    rows = _read_table(path)
    start_col = _column(rows, "start", "pos", "from")
    end_col = _column(rows, "end", "stop", "to")
    label_col = _column(rows, "label", "name", "segment", "id", required=False)
    class_col = _column(rows, "class", "type", "segment_class", "category", required=False)
    segments = []
    for i, row in enumerate(rows):
        start = max(0, min(genome_length, _safe_int(row.get(start_col), default=0)))
        end = max(start + 1, min(genome_length, _safe_int(row.get(end_col), default=genome_length)))
        segments.append(
            {
                "start": start,
                "end": end,
                "label": str(row.get(label_col) if label_col else f"s{i + 1}"),
                "class": str(row.get(class_col) if class_col else "core").lower(),
            }
        )
    return segments


# ------------------------------------------------------------------ rendering


def _render_pan_circular(
    out: Path, metrics: Mapping[str, Any], executable: Any = None, runner: Any = None
) -> Path:
    """Draw the figure at its printed size: circle, gene strip under it, key column."""
    from ..core.frozen import thaw_json

    data = thaw_json(metrics)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.with_suffix("")
    if data.get("write_intermediates", True):
        _write_intermediates(work, data)
    # The full graph is part of this figure's output: the centre only shows an overview.
    full = _write_full_graph(work, data) if data["center_image"] == "ov_graph" else None
    if full is not None:
        data["_full_graph_file"] = f"{work.name}/{full.name}"

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

    default = _journal_size(data["loci"], data["organelle"])
    width, height = (float(v) for v in data.get("figsize") or default)
    w_mm, h_mm = width * MM_PER_INCH, height * MM_PER_INCH
    strip = _strip_mm(data["loci"], data["organelle"])
    circle = max(40.0, min(h_mm - strip, w_mm * _CIRCLE_MM / _WIDTH_MM))
    with publication_style():
        fig = plt.figure(figsize=(width, height))
        try:
            ax = fig.add_axes((0.0, (h_mm - circle) / h_mm, circle / w_mm, circle / h_mm))
            info = _draw_circle(ax, data, out.with_suffix(""), executable, runner)
            if strip:
                strip_ax = fig.add_axes((0.0, 0.0, circle / w_mm, (h_mm - circle) / h_mm))
                strip_ax.axis("off")
                _draw_gene_strip(fig, strip_ax, data, info)
            x0 = circle + _GUTTER_MM
            key = fig.add_axes((x0 / w_mm, 0.02, (w_mm - x0) / w_mm, 0.96))
            key.axis("off")
            _draw_legend_panel(fig, key, data, info["heat_scale"], info["div_scale"])
            fig.savefig(str(out), dpi=int(data.get("dpi", 300)), facecolor="white")
        finally:
            plt.close(fig)
    return out


def _draw_circle(ax, data: Mapping[str, Any], work: Path | None, executable=None, runner=None):
    """Rings I-V on ``ax`` (equal aspect, +-_VIEW). Returns the scales the keys need."""
    from .features import genes_from_rows
    from .ogdraw import MitoGenome, _inches_per_unit, draw_mito_map

    L = int(data["genome_length"])
    gap = _numeral_angle(data) if data["loci"] else None
    data = {**data, "_numeral_deg": _NUMERAL_DEG if gap is None else gap}
    parsed = MitoGenome(
        genes=genes_from_rows(data["loci"]), gc=[], genome_length=L, organelle=data["organelle"]
    )
    trna_omitted = False
    if data["loci"]:
        # Every label outside (minus strand in grey); a crowded ring shrinks its
        # labels to 5 pt and then stops labelling tRNAs, which the key states.
        draw_mito_map(
            parsed,
            genome_name=data["species"],
            ax=ax,
            title="",
            draw_gc=False,
            draw_center_text=False,
            draw_legend=False,
            fold_minus_labels_out=True,
            minus_label_color="#8A9199",
            legend_below=True,
            min_label_gap=None,
            omit_trna_labels_when_crowded=True,
        )
        trna_omitted = bool(getattr(ax, "ov_trna_labels_omitted", False))
    ax.set_xlim(-_VIEW, _VIEW)
    ax.set_ylim(-_VIEW, _VIEW)
    ax.set_aspect("equal")
    ax.axis("off")
    unit_in = _inches_per_unit(ax)

    _draw_ticks(ax, L, _PT_TICK)
    numerals: list[tuple[str, float]] = []
    if data["loci"] and gap is not None:
        # One badge across both strand rings, in a gap between genes; on a ring
        # with no such gap the strip under the circle names the rings instead.
        numerals.append(("I·II", 0.86))
    outer = _R_TRACKS_OUT
    if data["regions"]:
        _draw_ir_band(ax, data["regions"], L, _PT_REGION)
        outer = _R_TRACKS_OUT_IR

    from .circular.tracks import allocate_rings

    heat = data["heatmap"]
    sv = data.get("sv")
    names = _ring_numerals(data)
    weights = {"variants": 1.8, "sv": 0.35, "pi": 1.0, "gc": 0.55}
    tracks = [kind for kind in ("variants", "sv", "pi", "gc") if kind in names]
    slots = dict(
        zip(
            tracks,
            allocate_rings(
                [weights[k] for k in tracks], outer=outer, inner=_R_TRACKS_IN, gap=_RING_GAP
            ),
            strict=True,
        )
    )

    def mid(kind: str) -> float:
        return (slots[kind].r_in + slots[kind].r_out) / 2

    starts, ends = data["bins"]["start"], data["bins"]["end"]
    heat_scale = None
    if "variants" in slots:
        heat_scale = _draw_heatmap(ax, heat, starts, ends, slots["variants"], L)
        numerals.append((names["variants"], mid("variants")))
    if "sv" in slots:
        _draw_sv(ax, sv, starts, ends, slots["sv"], L)
        numerals.append((names["sv"], mid("sv")))
    div_scale = _draw_diversity(ax, data, starts, ends, slots["pi"], L)
    numerals.append((names["pi"], mid("pi")))
    if "gc" in slots:
        div_scale["gc"] = _draw_gc(ax, data, starts, ends, slots["gc"], L)
        numerals.append((names["gc"], mid("gc")))

    center = data["center_image"]
    if center == "ov_graph":
        from .pan_graph import draw_pangenome_graph

        draw_pangenome_graph(
            ax,
            data["graph_overview"],
            radius=_R_CENTER,
            unit_in=unit_in,
            layout={k: np.asarray(v) for k, v in data["graph_overview_layout"].items()},
            label_blocks=4,
            font_pt=_PT_LABEL,
        )
        numerals.append((names["center"], _R_CENTER - 0.03))
    elif center == "bandage":
        _draw_bandage_center(ax, data, work or Path("."), executable, runner)
        numerals.append((names["center"], _R_CENTER - 0.03))
    elif center == "segments":
        _draw_segments(ax, data["segments"], L)
        numerals.append((names["center"], 0.36))
    else:
        _draw_center_text(ax, data, _PT_BODY)
    _draw_numerals(ax, numerals, _PT_BODY, data["_numeral_deg"])
    return {"heat_scale": heat_scale, "div_scale": div_scale, "trna_omitted": trna_omitted}


_ROMAN = ("III", "IV", "V", "VI", "VII", "VIII")


def _caption(m: Mapping[str, Any]) -> str:
    """Figure legend text: everything the drawn key leaves out."""
    names = _ring_numerals(m)
    unit = "sites" if m["variant_unit"] == "sites" else "calls"
    parts = [
        f"{m['title']}. {m['accessions']} genomes against the reference {m['reference']} "
        f"({m['genome_length']:,} bp); {m['snp_count']:,} SNP and {m['indel_count']:,} indel "
        f"{unit}" + (f", {m['sv_count']:,} SV" if m.get("sv_count") else "") + "."
    ]
    if m["loci"]:
        trans = next((r["gene"] for r in m["loci"] if r["splicing"] == "trans"), None)
        parts.append(
            "I·II, reference genes on the + (outer) and \N{MINUS SIGN} (inner) strand, coloured "
            "by function; \N{MINUS SIGN}-strand names in grey"
            + (
                f"; exons of trans-spliced genes are numbered ({trans}·1, {trans}·2, ...)"
                if trans
                else ""
            )
            + "."
        )
    heat = m["heatmap"]
    if heat["samples"]:
        text = (
            f"{names['variants']}, one row per genome (outer to inner as listed): non-reference "
            "calls per window, log10(n + 1)."
        )
        if any(r.get("aligned_share") is not None for r in heat.get("summary", [])):
            text += (
                " Grey: the genome's graph path shares < 10 % of the window's reference sequence "
                "(unaligned / no coverage). Lavender: the genome is aligned there but has no "
                "allele at >= 1 variant site (VCF '.'; the segment or allele is absent)."
            )
        elif any(STATE_NO_CALL in row for row in heat.get("state", [])):
            text += " Grey: no call at every site in the window (VCF '.'); no graph to separate "
            text += "absent from unaligned sequence."
        parts.append(text)
    if m["diversity_kind"] == "pi":
        text = (
            f"{names['pi']}, nucleotide diversity \N{GREEK SMALL LETTER PI} per site of SNP (filled) "
            "and indel sites per window, computed from the genotypes on one radial scale"
        )
        if m.get("reference_haplotype"):
            text += "; the reference counts as one more genome"
        parts.append(text + ".")
    if m.get("gc"):
        parts.append(f"{names['gc']}, reference GC content per window (lowest to highest).")
    graph = m.get("graph_overview")
    if m["center_image"] == "ov_graph" and graph:
        full = m.get("graph") or graph
        parts.append(
            f"{names['center']}, simplified overview of the pangenome graph: "
            f"{full['raw_nodes']:,} graph nodes collapsed (short nodes dropped, each genome's path "
            "re-linked, unbranched runs carried by the same genomes merged) to "
            f"{len(graph['blocks'])} blocks >= {_window_label(graph['min_block'])}; core = all "
            "genomes, shell = some, private = one; link width grows with the genomes using it; "
            "the four longest blocks are labelled with their length. The full skeleton "
            f"({len(full['blocks'])} blocks >= {_window_label(full['min_block'])}, "
            f"{full['represented_bp'] / max(1, full['raw_bp']):.0%} of the graph sequence) is "
            "written beside the figure as pan_graph_full.pdf."
        )
    parts.append("Ring numerals sit in one column; the curves of the line rings open under them.")
    return " ".join(parts)


def _ring_numerals(data: Mapping[str, Any]) -> dict[str, str]:
    """Numeral of each drawn layer, outside in (genes are always I·II)."""
    names: dict[str, str] = {}
    if data["loci"]:
        names["genes"] = "I·II"
    kinds = []
    if data["heatmap"]["samples"]:
        kinds.append("variants")
    if data.get("sv") and any(data["sv"]):
        kinds.append("sv")
    kinds.append("pi")
    if data.get("gc"):
        kinds.append("gc")
    if data["center_image"] in {"ov_graph", "bandage", "segments"}:
        kinds.append("center")
    names.update(zip(kinds, _ROMAN, strict=False))
    return names


def _legend_handles(data: Mapping[str, Any], info: Mapping[str, Any]) -> list[Any]:
    """Legend entries for a composed figure (each distinct label is shown once there)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from .ogdraw import GENE_COLORS, legend_rows
    from .pan_graph import CLASS_COLORS, CLASS_LABELS

    present = {row["category"] for row in data["loci"]}
    handles: list[Any] = [
        Patch(
            facecolor=GENE_COLORS.get(key, "#cccccc"), edgecolor="black", linewidth=0.4, label=text
        )
        for key, text in legend_rows(data["organelle"])
        if key in present
    ]
    names = _ring_numerals(data)
    if data["heatmap"]["samples"]:
        n = names["variants"]
        handles.append(Patch(facecolor="#4A8FC4", label=f"{n} variants per genome"))
        handles += [
            Patch(facecolor=_STATE_COLORS[s], label=f"{n} {_STATE_LABELS[s]}")
            for s in _present_states(data)
        ]
    if info["div_scale"]["top"] > 0:
        handles.append(Line2D([], [], color=_SNP_COLOR, label=f"{names['pi']} SNP π"))
        if any(data["indel_pi"]):
            handles.append(Line2D([], [], color=_INDEL_COLOR, label=f"{names['pi']} indel π"))
    if info["div_scale"]["gc"]:
        handles.append(Line2D([], [], color=_GC_COLOR, label=f"{names['gc']} GC content"))
    graph = data.get("graph_overview") or data.get("graph")
    if data["center_image"] == "ov_graph" and graph:
        classes = {b["class"] for b in graph["blocks"]}
        handles += [
            Patch(facecolor=CLASS_COLORS[c], label=f"{names['center']} {CLASS_LABELS[c]}")
            for c in CLASS_COLORS
            if c in classes
        ]
    return handles


_STATE_COLORS = {STATE_MISSING: "#B9A5C9", STATE_UNALIGNED: "#D5D9DE", STATE_NO_CALL: "#C8CCD2"}
_STATE_LABELS = {
    STATE_MISSING: "missing: aligned, no allele/segment",
    STATE_UNALIGNED: "unaligned / no coverage",
    STATE_NO_CALL: "no call (VCF '.')",
}


def _present_states(data: Mapping[str, Any]) -> list[int]:
    seen = {s for row in data["heatmap"].get("state", []) for s in row}
    return [s for s in (STATE_MISSING, STATE_UNALIGNED, STATE_NO_CALL) if s in seen]


def _draw_gene_strip(fig, ax, data: Mapping[str, Any], info: Mapping[str, Any]) -> None:
    """Gene functional classes in rows of three under the circle."""
    from matplotlib.patches import Rectangle

    from .ogdraw import GENE_COLORS, legend_rows

    present = {row["category"] for row in data["loci"]}
    rows = [(k, t) for k, t in legend_rows(data["organelle"]) if k in present]
    box = ax.get_position()
    h_in = box.height * fig.get_figheight()
    w_in = box.width * fig.get_figwidth()

    def y_of(points: float) -> float:
        return 1.0 - points / 72 / h_in

    ax.text(
        0.02,
        y_of(2),
        "I·II  Genes (I: + strand, outer; II: \N{MINUS SIGN} strand, inner)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=_PT_HEADING,
        fontweight="bold",
        color=_INK,
    )
    if info.get("trna_omitted"):
        ax.text(
            0.02,
            y_of(13),
            "tRNA genes drawn, not labelled (no space)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_PT_BODY,
            color=_MUTED,
        )
    per_col = math.ceil(len(rows) / _STRIP_COLUMNS)
    sw_w, sw_h = (2.6 / MM_PER_INCH) / w_in, (_PT_BODY * 0.95) / 72 / h_in
    for i, (key, text) in enumerate(rows):
        col, row = divmod(i, per_col)
        x = 0.02 + col * (0.98 / _STRIP_COLUMNS)
        y = y_of(22 + row * _STRIP_ROW_MM / 25.4 * 72)
        ax.add_patch(
            Rectangle(
                (x, y - sw_h),
                sw_w,
                sw_h,
                transform=ax.transAxes,
                fc=GENE_COLORS.get(key, "#cccccc"),
                ec="#3A3A3A",
                lw=0.4,
                clip_on=False,
            )
        )
        ax.text(
            x + sw_w + 0.012,
            y - sw_h / 2,
            _short(text),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=_PT_BODY,
            color=_INK,
        )


def _deg(bp: float, L: int) -> float:
    return 90.0 - 360.0 * (bp / L)


def _arc(a: float, b: float, r: float, L: int, step_deg: float = 1.0) -> np.ndarray:
    t0, t1 = _deg(a, L), _deg(b, L)
    n = max(2, math.ceil(abs(t1 - t0) / step_deg) + 1)
    t = np.radians(np.linspace(t0, t1, n))
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _cell(a: float, b: float, r_in: float, r_out: float, L: int) -> np.ndarray:
    return np.vstack([_arc(a, b, r_out, L), _arc(b, a, r_in, L)])


def _ring(
    ax, r_in: float, r_out: float, color: str, zorder: float = 0.4, edge: str = "none"
) -> None:
    from matplotlib.patches import Wedge

    ax.add_patch(
        Wedge(
            (0, 0),
            r_out,
            0,
            360,
            width=r_out - r_in,
            facecolor=color,
            edgecolor=edge,
            linewidth=0.4,
            zorder=zorder,
        )
    )


def _draw_ticks(ax, L: int, font: float) -> None:
    """kb scale just inside the gene ring: ticks every step, every 2nd one labelled.

    A labelled position is marked by its label alone (no tick stroke runs
    through the text); unlabelled positions get a short tick.
    """
    step = _nice_step(L / 12)
    pos = 0
    i = 0
    while pos < L:
        ang = math.radians(_deg(pos, L))
        major = i % 2 == 0
        if not major or pos == 0:
            ax.plot(
                [_R_TICK_IN * math.cos(ang), _R_TICK_OUT * math.cos(ang)],
                [_R_TICK_IN * math.sin(ang), _R_TICK_OUT * math.sin(ang)],
                color=_MUTED,
                lw=0.5,
                zorder=3,
                solid_capstyle="butt",
            )
        if major and pos > 0:
            deg = _deg(pos, L) % 360
            rot = deg - 90 if deg <= 180 else deg + 90
            ax.text(
                _R_TICK_LABEL * math.cos(ang),
                _R_TICK_LABEL * math.sin(ang),
                _bp_label(pos, step),
                rotation=rot,
                rotation_mode="anchor",
                ha="center",
                va="center",
                fontsize=font,
                color=_MUTED,
                zorder=3,
            )
        pos += step
        i += 1
    from matplotlib.patches import Circle

    ax.add_patch(Circle((0, 0), _R_TICK_OUT, fill=False, ec=_MUTED, lw=0.5, zorder=3))


def _nice_step(raw: float) -> int:
    exp = 10 ** math.floor(math.log10(max(raw, 1)))
    for m in (1, 2, 2.5, 5, 10):
        if m * exp >= raw:
            return int(m * exp)
    return int(10 * exp)


def _bp_label(pos: int, step: int) -> str:
    if pos >= 1_000_000:
        return f"{pos / 1e6:g} Mb"
    return f"{pos / 1000:g} kb" if step >= 1000 else f"{pos} bp"


def _window_label(width: int) -> str:
    if width >= 1000:
        return f"{width / 1000:.2g} kb" if width < 10_000 else f"{width / 1000:.0f} kb"
    return f"{width} bp"


def _draw_ir_band(ax, regions: Mapping[str, Any], L: int, font: float) -> None:
    from matplotlib.patches import Polygon

    from .ogdraw import REGION_COLORS

    r_in, r_out = _R_IR
    pad = 0.0015 * L  # regions separated by a gap, not by a stroke through the labels
    step = _nice_step(L / 12)
    labelled = list(range(2 * step, L, 2 * step))  # the kb labels just outside the band

    def label_bp(a: float, b: float) -> float:
        """Region middle, or the point inside the region farthest from any kb label."""
        candidates = [(a + b) / 2] + [
            (a + b) / 2 + s * f * (b - a) for f in (0.15, 0.3) for s in (-1, 1)
        ]

        def clearance(bp: float) -> float:
            return min((abs(bp - t) for t in labelled), default=L)

        best = candidates[0]
        if clearance(best) < 0.035 * L:
            best = max(candidates, key=clearance)
        return best

    for name, span in regions.items():
        a = span["start"]
        b = a + span["length"]
        ax.add_patch(
            Polygon(
                _cell(a + pad, b - pad, r_in, r_out, L),
                closed=True,
                fc=REGION_COLORS.get(name, "#dddddd"),
                ec="none",
                zorder=2,
            )
        )
        mid = math.radians(_deg(label_bp(a, b), L))
        deg = math.degrees(mid) % 360
        rot = deg - 90 if deg <= 180 else deg + 90
        r = (r_in + r_out) / 2
        ax.text(
            r * math.cos(mid),
            r * math.sin(mid),
            name,
            rotation=rot,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=font,
            color=_INK,
            zorder=3,
        )


def _heat_scale(counts: Sequence[Sequence[float]]):
    peak = max((max(row) for row in counts if len(row)), default=0)
    vmax = max(1.0, math.ceil(2 * math.log10(float(peak) + 1.0)) / 2)
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    base = matplotlib.colormaps["Blues"]
    cmap = LinearSegmentedColormap.from_list("ov_blues", base(np.linspace(0.12, 1.0, 256)))
    return Normalize(0.0, vmax), cmap


def _draw_heatmap(ax, heat, starts, ends, slot, L: int):
    from matplotlib.collections import PolyCollection

    norm, cmap = _heat_scale(heat["counts"])
    rows = len(heat["samples"])
    h = (slot.r_out - slot.r_in) / rows
    pad = h * 0.12 if rows <= 40 else 0.0
    states = heat.get("state") or [[STATE_SAME] * len(starts) for _ in range(rows)]
    polys, colors = [], []
    for i, (counts, row_states) in enumerate(zip(heat["counts"], states, strict=True)):
        r_out = slot.r_out - i * h
        r_in = r_out - h + pad
        for a, b, n, state in zip(starts, ends, counts, row_states, strict=True):
            if state == STATE_VARIANT:
                color = cmap(norm(math.log10(float(n) + 1.0)))
            else:
                color = _STATE_COLORS.get(state, "#F7F9FB")
            polys.append(_cell(a, b, r_in, r_out, L))
            colors.append(color)
    ax.add_collection(
        PolyCollection(polys, facecolors=colors, edgecolors="none", linewidths=0, zorder=2)
    )
    return norm, cmap


def _draw_sv(ax, sv, starts, ends, slot, L: int) -> None:
    from matplotlib.collections import PolyCollection

    _ring(ax, slot.r_in, slot.r_out, "#FBF3E7")
    top = max(sv) or 1.0
    polys = [
        _cell(a, b, slot.r_in, slot.r_in + (v / top) * (slot.r_out - slot.r_in), L)
        for a, b, v in zip(starts, ends, sv, strict=True)
        if v > 0
    ]
    ax.add_collection(PolyCollection(polys, facecolors=_SV_COLOR, edgecolors="none", zorder=2))


_NUMERAL_GAP_DEG = 4.0  # curves open this far either side of the numeral column


def _curve_runs(mids, values, slot, top: float, L: int, numeral: float):
    """Closed radial curve split into runs that skip the numeral column."""
    span = slot.r_out - slot.r_in
    bp = np.r_[mids, mids[0] + L]
    radii = slot.r_in + (np.r_[values, values[0]] / top) * span
    deg = 90.0 - 360.0 * bp / L
    off = np.abs((deg - numeral + 180.0) % 360.0 - 180.0)
    keep = off > _NUMERAL_GAP_DEG
    ang = np.radians(deg)
    if keep.all():
        return [(radii * np.cos(ang), radii * np.sin(ang), ang)]
    first = int(np.argmin(keep))
    order = (np.arange(len(keep) - 1) + first) % (len(keep) - 1)  # drop the closing duplicate
    runs, current = [], []
    for k in order:
        if keep[k]:
            current.append(k)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    out = []
    for idx in runs:
        if len(idx) < 2:
            continue
        idx = np.asarray(idx)
        out.append((radii[idx] * np.cos(ang[idx]), radii[idx] * np.sin(ang[idx]), ang[idx]))
    return out


def _grid_arcs(ax, slot, numeral: float, fractions=(0.5, 1.0)) -> None:
    from matplotlib.patches import Arc

    for frac in fractions:
        r = slot.r_in + frac * (slot.r_out - slot.r_in)
        ax.add_patch(
            Arc(
                (0, 0),
                2 * r,
                2 * r,
                theta1=numeral + 6,
                theta2=numeral - 6 + 360,
                ec="#C9CCC4",
                lw=0.4,
                ls=(0, (2, 2)),
                zorder=1,
            )
        )


def _draw_diversity(ax, data, starts, ends, slot, L: int) -> dict[str, Any]:
    """Ring of per-site π of SNPs (filled) and indels on one shared radial scale."""
    snp = np.asarray(data["snp_pi"], dtype=float)
    indel = np.asarray(data["indel_pi"], dtype=float)
    _ring(ax, slot.r_in, slot.r_out, "#F5F6F1")
    numeral = data.get("_numeral_deg", _NUMERAL_DEG)
    _grid_arcs(ax, slot, numeral)
    mids = (np.asarray(starts) + np.asarray(ends)) / 2.0
    top = float(max(snp.max(initial=0.0), indel.max(initial=0.0)))
    if top > 0:
        for values, color, lw, fill in (
            (snp, _SNP_COLOR, 1.0, True),
            (indel, _INDEL_COLOR, 0.9, False),
        ):
            if not values.any():
                continue
            for x, y, ang in _curve_runs(mids, values, slot, top, L, numeral):
                if fill:
                    xb, yb = slot.r_in * np.cos(ang), slot.r_in * np.sin(ang)
                    ax.fill(
                        np.r_[x, xb[::-1]],
                        np.r_[y, yb[::-1]],
                        color=color,
                        alpha=0.16,
                        lw=0,
                        zorder=2,
                    )
                ax.plot(x, y, color=color, lw=lw, zorder=3, solid_joinstyle="round")
    # The radial scale (0 at the inner edge, the maximum at the outer edge) is
    # stated in the key: labels inside the ring would sit on the curves.
    return {"top": top, "gc": None}


def _draw_gc(ax, data, starts, ends, slot, L: int) -> tuple[float, float]:
    """Own ring for the reference GC content: inner edge = lowest, outer = highest window."""
    values = np.asarray(data["gc"], dtype=float)
    lo, hi = float(values.min()), float(values.max())
    _ring(ax, slot.r_in, slot.r_out, "#F3F4F6")
    numeral = data.get("_numeral_deg", _NUMERAL_DEG)
    _grid_arcs(ax, slot, numeral, fractions=(0.5,))
    mids = (np.asarray(starts) + np.asarray(ends)) / 2.0
    rng = (hi - lo) or 1.0
    for x, y, _ in _curve_runs(mids, (values - lo) / rng, slot, 1.0, L, numeral):
        ax.plot(x, y, color=_GC_COLOR, lw=0.8, zorder=3, solid_joinstyle="round")
    return lo, hi


def _numeral_angle(data: Mapping[str, Any]) -> float | None:
    """Angle (deg) nearest the top where no gene lies under the I·II badge, else None."""
    L = int(data["genome_length"])
    exons: list[tuple[int, int]] = []
    for row in data["loci"]:
        if row["splicing"] == "trans":
            exons += [(e["start"] - 1, e["end"]) for e in row["exons"]]
        else:  # a cis gene is drawn over its whole span (intron bar included)
            exons.append(
                (min(e["start"] for e in row["exons"]) - 1, max(e["end"] for e in row["exons"]))
            )
    half = 4.0 / 360 * L  # badge half-width (7 pt "I·II" at r = 0.86) in bp

    def free(deg: float) -> bool:
        centre = ((90.0 - deg) / 360.0 * L) % L
        for a, b in exons:
            for shift in (-L, 0, L):
                if a + shift < centre + half and b + shift > centre - half:
                    return False
        return True

    for k in range(0, 360):  # nearest free angle to the top, either way round
        for deg in (_NUMERAL_DEG + 0.5 * k, _NUMERAL_DEG - 0.5 * k):
            if free(deg):
                return deg % 360
    return None  # gene-dense ring: no gap wide enough for the badge


def _sci(value: float) -> str:
    if value == 0:
        return "0"
    exp = math.floor(math.log10(abs(value)))
    if -2 <= exp <= 2:
        return f"{value:.3g}"
    return f"{value / 10**exp:.2g}e{exp}"


def _draw_segments(ax, segments, L: int) -> None:
    from matplotlib.patches import Polygon

    from .pan_graph import CLASS_COLORS

    for i, row in enumerate(segments):
        color = CLASS_COLORS.get(row["class"], CLASS_COLORS["shell"])
        r_in = 0.30 + (i % 3) * 0.03
        ax.add_patch(
            Polygon(
                _cell(row["start"], row["end"], r_in, r_in + 0.026, L),
                closed=True,
                fc=color,
                ec="white",
                lw=0.3,
                zorder=2,
            )
        )


def _draw_center_text(ax, data, font: float) -> None:
    word = _ORGANELLE_WORD.get(data["organelle"], "organelle")
    ax.text(
        0,
        0.05,
        data["species"],
        ha="center",
        va="center",
        fontsize=_PT_HEADING,
        fontstyle="italic",
        color=_INK,
    )
    ax.text(0, -0.03, f"{word} genome", ha="center", va="center", fontsize=font, color=_MUTED)
    ax.text(
        0,
        -0.1,
        f"{data['genome_length']:,} bp",
        ha="center",
        va="center",
        fontsize=font,
        color=_MUTED,
    )


def _draw_bandage_center(ax, data, work: Path, executable, runner) -> None:
    from .gfa_graph import plot_gfa_graph

    plot = plot_gfa_graph(
        Path(data["gfa"]),
        organelle=data["organelle"],
        executable=executable,
        runner=runner,
        color_by="uniform",
        height=1900,
        font_size=40,
        node_width=13,
        edge_width=2.6,
        outline=0.0,
        extra_args=("--unicolpos", "#2A6F97", "--edgecol", "#AAB4BD", "--textcol", "#2B2B2B"),
    )
    if not isinstance(plot, OrganellePlot) or plot.status != "ok":
        raise OrganelleDependencyError(
            code="visualization.missing_bandage",
            message="Bandage could not render the pangenome graph.",
            details={"gfa": data["gfa"]},
        )
    work.mkdir(parents=True, exist_ok=True)
    image_path = plot.save(work / "pan_bandage_center.png")
    import matplotlib.image as mpimg

    image = np.asarray(mpimg.imread(str(image_path)), dtype=float)
    if image.ndim == 3 and image.shape[2] == 3:
        image = np.dstack([image, np.ones(image.shape[:2])])
    image = image.copy()
    image[np.all(image[..., :3] > 0.94, axis=-1), 3] = 0.0
    r = _R_CENTER * 0.92
    inset = ax.inset_axes([-r, -r, 2 * r, 2 * r], transform=ax.transData)
    inset.imshow(image, interpolation="bilinear")
    inset.set_axis_off()


def _draw_numerals(ax, numerals, font: float, deg: float = _NUMERAL_DEG) -> None:
    ang = math.radians(deg)
    for text, r in numerals:
        if not text:
            continue
        ax.text(
            r * math.cos(ang),
            r * math.sin(ang),
            text,
            ha="center",
            va="center",
            fontsize=font,
            fontweight="bold",
            color="white",
            zorder=30,
            bbox={"boxstyle": "round,pad=0.18", "fc": "#37414B", "ec": "none"},
        )


class _Panel:
    """Top-down cursor for the legend column (axes-fraction coordinates)."""

    def __init__(self, fig, ax) -> None:
        self.fig, self.ax = fig, ax
        bbox = ax.get_position()
        self.height_in = bbox.height * fig.get_figheight()
        self.y = 1.0

    def line(self, points: float) -> float:
        return points / 72 / self.height_in

    def heading(self, numeral: str, text: str) -> None:
        self.y -= self.line(9)
        if numeral:
            self.ax.text(
                0.0,
                self.y,
                numeral,
                ha="left",
                va="top",
                fontsize=_PT_BODY,
                fontweight="bold",
                color="white",
                transform=self.ax.transAxes,
                bbox={"boxstyle": "round,pad=0.18", "fc": "#37414B", "ec": "none"},
            )
        self.ax.text(
            0.11 if numeral else 0.0,
            self.y,
            text,
            ha="left",
            va="top",
            fontsize=_PT_HEADING,
            fontweight="bold",
            color=_INK,
            transform=self.ax.transAxes,
        )
        self.y -= self.line(_PT_HEADING * 1.7)

    def note(self, text: str, size: float = _PT_BODY, color: str = _MUTED, x: float = 0.0) -> None:
        lines = text.count("\n") + 1
        self.ax.text(
            x,
            self.y,
            text,
            ha="left",
            va="top",
            fontsize=size,
            color=color,
            transform=self.ax.transAxes,
            linespacing=1.3,
        )
        self.y -= self.line(size * 1.35 * lines + 2)

    def swatch(self, color: str, text: str, *, kind: str = "box", x: float = 0.0, ls="-") -> None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle

        h = self.line(_PT_BODY * 0.95)
        if kind == "box":
            self.ax.add_patch(
                Rectangle(
                    (x, self.y - h),
                    0.05,
                    h,
                    transform=self.ax.transAxes,
                    fc=color,
                    ec="#3A3A3A",
                    lw=0.4,
                    clip_on=False,
                )
            )
        else:
            self.ax.add_line(
                Line2D(
                    [x, x + 0.06],
                    [self.y - h / 2] * 2,
                    transform=self.ax.transAxes,
                    color=color,
                    lw=1.6,
                    ls=ls,
                    clip_on=False,
                )
            )
        self.ax.text(
            x + 0.08,
            self.y - h / 2,
            text,
            ha="left",
            va="center",
            fontsize=_PT_BODY,
            color=_INK,
            transform=self.ax.transAxes,
        )
        self.y -= self.line(_PT_BODY * 1.5)


def _draw_legend_panel(fig, ax, data, heat_scale, div_scale) -> None:
    """Only what reading the rings needs; explanations live in metrics["caption"]."""
    from .pan_graph import CLASS_COLORS, CLASS_LABELS

    panel = _Panel(fig, ax)
    names = _ring_numerals(data)
    heat = data["heatmap"]
    if heat_scale is not None:
        norm, cmap = heat_scale
        starts, ends = data["bins"]["start"], data["bins"]["end"]
        window = round(float(np.median(np.asarray(ends) - np.asarray(starts))))
        panel.heading(names["variants"], f"Variants per genome ({_window_label(window)} windows)")
        cax = fig.add_axes(_panel_rect(ax, 0.0, panel.y - panel.line(8), 0.62, panel.line(8)))
        grad = np.linspace(norm.vmin, norm.vmax, 256).reshape(1, -1)
        cax.imshow(grad, aspect="auto", cmap=cmap, norm=norm, extent=(norm.vmin, norm.vmax, 0, 1))
        cax.set_yticks([])
        ticks = np.arange(0, norm.vmax + 1e-9, 0.5)
        cax.set_xticks(ticks)
        cax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=_PT_TICK)
        cax.tick_params(length=2, width=0.4, pad=1.5)
        for spine in cax.spines.values():
            spine.set_linewidth(0.4)
        ax.text(
            0.66,
            panel.y - panel.line(4),
            "log\N{SUBSCRIPT ONE}\N{SUBSCRIPT ZERO}(n + 1)",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=_PT_BODY,
            color=_MUTED,
        )
        panel.y -= panel.line(22)
        for state in _present_states(data):
            panel.swatch(_STATE_COLORS[state], _STATE_LABELS[state])
        if len(heat["labels"]) <= 30:
            names_list = [f"{i}. {name}" for i, name in enumerate(heat["labels"], 1)]
            half = (len(names_list) + 1) // 2 if len(names_list) > 14 else len(names_list)
            panel.note("rows, outer \N{RIGHTWARDS ARROW} inner")
            y0 = panel.y
            panel.note("\n".join(names_list[:half]), size=_PT_LABEL, color=_INK)
            if names_list[half:]:
                y_left = panel.y
                panel.y = y0
                panel.note("\n".join(names_list[half:]), size=_PT_LABEL, color=_INK, x=0.5)
                panel.y = min(panel.y, y_left)
    if data.get("sv") and any(data["sv"]):
        panel.heading(names["sv"], "Structural variants per window")
        panel.swatch(_SV_COLOR, f"0 \N{EN DASH} {max(data['sv']):g} SV calls")
    if div_scale["top"] > 0:
        title = {"pi": "Nucleotide diversity \N{GREEK SMALL LETTER PI}", "supplied": "Diversity"}
        panel.heading(names["pi"], title.get(data["diversity_kind"], "Calls per genome"))
        panel.swatch(_SNP_COLOR, "SNPs", kind="line")
        if any(data["indel_pi"]):
            panel.swatch(_INDEL_COLOR, "indels", kind="line")
        panel.note(f"inner edge 0, outer edge {_sci(div_scale['top'])}")
    if div_scale["gc"]:
        lo, hi = div_scale["gc"]
        panel.heading(names["gc"], "GC content")
        panel.swatch(_GC_COLOR, f"inner edge {lo:.1%}, outer edge {hi:.1%}", kind="line")
    graph = data.get("graph_overview")
    if data["center_image"] == "ov_graph" and graph:
        panel.heading(names["center"], "Pangenome graph, simplified")
        classes = {b["class"] for b in graph["blocks"]}
        for cls, color in CLASS_COLORS.items():
            if cls in classes:
                panel.swatch(color, CLASS_LABELS[cls])
        full = data.get("_full_graph_file")
        panel.note(
            f"{len(graph['blocks'])} largest-scale blocks"
            + (f";\nfull graph: {full}" if full else ""),
        )
    elif data["center_image"] == "bandage":
        panel.heading(names["center"], "Pangenome graph (Bandage)")
    elif data["center_image"] == "segments":
        panel.heading(names["center"], "Pangenome segments")
        for cls in sorted({s["class"] for s in data["segments"]}):
            panel.swatch(CLASS_COLORS.get(cls, CLASS_COLORS["shell"]), cls)


def _panel_rect(ax, x: float, y: float, w: float, h: float):
    box = ax.get_position()
    return (box.x0 + x * box.width, box.y0 + y * box.height, w * box.width, h * box.height)


_SHORT_LEGEND = {
    "hypothetical reading frames (ycf)": "ycf (hypothetical)",
    "NADH dehydrogenase (ndh)": "NADH dehydrogenase",
    "cytochrome b6/f complex": "cytochrome b6/f",
}


def _short(text: str) -> str:
    """Legend names that fit one half of the panel column."""
    text = _SHORT_LEGEND.get(text, text)
    return text.split(" (")[0] if text.startswith("complex ") else text


def _write_intermediates(work: Path, data: Mapping[str, Any]) -> None:
    work.mkdir(parents=True, exist_ok=True)
    starts, ends = data["bins"]["start"], data["bins"]["end"]
    heat = data["heatmap"]
    if heat["samples"]:
        labels = [f"{a}-{b}" for a, b in zip(starts, ends, strict=True)]
        lines = ["accession\t" + "\t".join(labels)]
        for sample, counts in zip(heat["samples"], heat["counts"], strict=True):
            lines.append(sample + "\t" + "\t".join(f"{float(v):g}" for v in counts))
        (work / "pan_variant_matrix.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["start\tend\tsnp\tindel" + ("\tgc" if data.get("gc") else "")]
    for i, (a, b) in enumerate(zip(starts, ends, strict=True)):
        row = f"{a}\t{b}\t{data['snp_pi'][i]:.6g}\t{data['indel_pi'][i]:.6g}"
        if data.get("gc"):
            row += f"\t{data['gc'][i]:.4f}"
        lines.append(row)
    (work / "pan_diversity.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    graph = data.get("graph")
    if graph:
        cols = ["name", "length", "genomes", "class", "ref_start", "ref_end"]
        rows = ["\t".join(cols)] + [
            "\t".join("" if b.get(c) is None else str(b.get(c)) for c in cols)
            for b in graph["blocks"]
        ]
        (work / "pan_graph_blocks.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        rows = ["source\ttarget\tgenomes"] + [
            f"{link['source']}\t{link['target']}\t{link['genomes']}" for link in graph["links"]
        ]
        (work / "pan_graph_links.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    summary = heat.get("summary") or []
    if summary:
        cols = list(summary[0])
        rows = ["\t".join(cols)] + [
            "\t".join("" if r[c] is None else str(r[c]) for c in cols) for r in summary
        ]
        (work / "pan_missing_summary.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (work / "pan_circular_caption.txt").write_text(
        str(data.get("caption", "")) + "\n", encoding="utf-8"
    )


def _write_full_graph(work: Path, data: Mapping[str, Any]) -> Path | None:
    """The complete graph skeleton, the file the simplified centre refers to."""
    graph, layout = data.get("graph"), data.get("graph_layout")
    if not graph or not layout:
        return None
    from .pan_graph import render_full_graph

    work.mkdir(parents=True, exist_ok=True)
    header = (
        f"{data['title']}: graph skeleton, {len(graph['blocks'])} blocks "
        f">= {_window_label(graph['min_block'])} from {graph['raw_nodes']:,} graph nodes"
    )
    pdf = render_full_graph(work / "pan_graph_full.pdf", graph, layout, header=header)
    render_full_graph(work / "pan_graph_full.png", graph, layout, header=header)
    return pdf


# ------------------------------------------------------------------ tables


def _read_table(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        return []
    try:
        dialect = csv.Sniffer().sniff("\n".join(text[:5]), delimiters="\t,")
    except csv.Error:
        dialect = csv.excel_tab
    return [dict(row) for row in csv.DictReader(text, dialect=dialect)]


def _column(rows: Sequence[Mapping[str, Any]], *candidates: str, required: bool = True) -> str:
    if not rows:
        if required:
            raise OrganelleInputError(
                code="visualization.pan_circular.empty_table", message="input table is empty"
            )
        return ""
    lower = {c.lower(): c for c in rows[0]}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise OrganelleInputError(
            code="visualization.pan_circular.missing_column",
            message=f"missing required column; expected one of {', '.join(candidates)}",
        )
    return ""


def _make_bins(genome_length: int, *, bins: int, window_size: int | None):
    if window_size is None:
        window_size = max(1, math.ceil(genome_length / max(1, bins)))
    starts = list(range(0, genome_length, window_size))
    ends = [min(genome_length, start + window_size) for start in starts]
    return starts, ends


def _bins_from_labels(labels: Sequence[str], genome_length: int):
    starts: list[int] = []
    ends: list[int] = []
    for label in labels:
        clean = str(label).replace(",", "").strip()
        if "-" in clean:
            left, right = clean.split("-", 1)
            starts.append(_safe_int(left, default=0))
            ends.append(_safe_int(right, default=0))
    if len(starts) == len(labels) and all(e > s for s, e in zip(starts, ends, strict=False)):
        return starts, ends
    return _make_bins(genome_length, bins=len(labels), window_size=None)


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


__all__ = ["plot_pan_circular"]
