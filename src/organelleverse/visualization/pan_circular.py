"""Circular pangenome overview plots for organelle genomes."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..core.result import OrganelleResult
from .circular.prepare import (
    DensityData,
    DiversityData,
    GraphImage,
    SegmentData,
    SegmentRow,
    SVData,
)
from .circular.render import plot_circular
from .circular.tracks import (
    DensityRing,
    DiversityRing,
    GeneRing,
    GraphCenter,
    SchematicCenter,
    SVRing,
)
from .ogdraw import classify_gene, plot_ogdraw_map
from .plot_object import OrganellePlot, plot_result


@dataclass(frozen=True)
class PanFeature:
    name: str
    start: int
    end: int
    strand: int
    feature_type: str
    category: str


@dataclass(frozen=True)
class VariantTracks:
    samples: tuple[str, ...]
    bin_starts: tuple[int, ...]
    bin_ends: tuple[int, ...]
    matrix: np.ndarray
    snp_diversity: np.ndarray
    indel_diversity: np.ndarray
    snp_count: int
    indel_count: int
    output_paths: tuple[Path, ...] = ()
    sv_density: np.ndarray | None = None
    sv_count: int = 0


@dataclass(frozen=True)
class PanSegment:
    start: int
    end: int
    label: str
    segment_class: str
    coverage: float | None = None


def plot_pan_circular(
    reference_gbk: str | Path,
    *,
    variants: str | Path | None = None,
    variant_matrix: str | Path | None = None,
    diversity: str | Path | None = None,
    sv_variants: str | Path | None = None,
    pan_segments: str | Path | None = None,
    pangenome_gfa: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    window_size: int | None = None,
    bins: int = 120,
    max_heatmap_rows: int = 220,
    label_limit: int = 70,
    outer_method: str = "ogdraw",
    bandage_executable: str | Path | None = None,
    bandage_runner: Any | None = None,
    bandage_labels: tuple[str, ...] = ("name", "length", "depth"),
    bandage_color_by: str = "depth",
    title: str = "Circular representation of the mitochondrial pan-genome",
    dpi: int = 300,
    figsize: tuple[float, float] = (10.5, 10.5),
    write_intermediates: bool = True,
) -> OrganellePlot:
    """Draw a circular mitochondrial/plastid pangenome overview (compute-only).

    The figure follows the common five-layer pangenome layout: forward-strand
    genes (I), reverse-strand genes (II), accession-level variant heatmap (III),
    SNP/indel diversity curves (IV), and an inner pangenome segment schematic
    or Bandage-rendered GFA graph (V).

    Returns an :class:`OrganellePlot`; ``ov.write(plot, output)`` materializes
    the primary figure at ``output`` and any sidecars/intermediates into a
    writer-managed sibling directory (``output`` with its suffix removed).
    """
    if outer_method not in {"ogdraw", "none", ""}:
        raise ValueError("outer_method must be 'ogdraw' or 'none'.")
    ref = Path(reference_gbk)
    features, genome_length, record_id = _parse_reference_features(ref)
    tracks = _load_or_derive_variant_tracks(
        variants=Path(variants) if variants else None,
        variant_matrix=Path(variant_matrix) if variant_matrix else None,
        diversity=Path(diversity) if diversity else None,
        genome_length=genome_length,
        bins=bins,
        window_size=window_size,
        work_dir=ref.parent,
        write_intermediates=False,
    )
    if sv_variants:
        sv_density, sv_count = _load_sv_density(
            Path(sv_variants), tracks.bin_starts, tracks.bin_ends
        )
        tracks = replace(tracks, sv_density=sv_density, sv_count=sv_count)
    segments = (
        _read_pan_segments(Path(pan_segments), genome_length)
        if pan_segments
        else _derive_pan_segments(features, genome_length)
    )
    meta = dict(metadata or {})
    accession_count = int(meta.get("accessions") or len(tracks.samples) or 0)
    organelle = str(meta.get("organelle", "mito"))

    # Sidecar plots are captured now and materialized by the renderer, so the
    # compute step writes nothing and runs no subprocess.
    outer_plot: OrganellePlot | None = None
    if outer_method == "ogdraw":
        outer_plot = plot_ogdraw_map(
            ref,
            genome_name=str(meta.get("reference") or ref.stem),
            title="",
            dpi=dpi,
            width=8,
            height=8,
        )
    bandage_plot: OrganellePlot | None = None
    if pangenome_gfa:
        from .gfa_graph import plot_gfa_graph

        candidate = plot_gfa_graph(
            Path(pangenome_gfa),
            organelle=organelle,
            executable=bandage_executable,
            runner=bandage_runner,
            labels=bandage_labels,
            color_by="uniform",
            height=1900,
            font_size=40,
            node_width=13,
            edge_width=2.6,
            outline=0.0,
            extra_args=(
                "--unicolpos",
                "#4FA6A0",
                "--edgecol",
                "#AAB4BD",
                "--textcol",
                "#2B2B2B",
                "--outcol",
                "#3C736E",
            ),
        )
        if isinstance(candidate, OrganellePlot) and candidate.status == "ok":
            bandage_plot = candidate

    has_outer = outer_plot is not None
    has_bandage = bandage_plot is not None
    derived_from_variants = bool(variants)

    def _render(path: str | Path) -> Path:
        return _render_pan_circular(
            ref,
            tracks,
            segments,
            meta,
            features,
            genome_length,
            record_id,
            accession_count,
            diversity is not None,
            derived_from_variants,
            outer_plot,
            bandage_plot,
            path,
            title=title,
            dpi=dpi,
            figsize=figsize,
            write_intermediates=write_intermediates,
        )

    flags = ["pan_circular_plot", "gene_structure_tracks", "center_summary"]
    if has_outer:
        flags.append("ogdraw_outer")
    if tracks.samples:
        flags.append("variant_heatmap")
    if np.any(tracks.snp_diversity) or np.any(tracks.indel_diversity):
        flags.append("diversity_curve")
    if tracks.sv_density is not None and np.any(tracks.sv_density):
        flags.append("sv_density_track")
    if has_bandage:
        flags.append("bandage_inner")
    elif segments:
        flags.append("pan_schematic")

    metrics = {
        "genome_length": genome_length,
        "reference": meta.get("reference") or record_id,
        "accessions": accession_count,
        "features": len(features),
        "forward_features": sum(1 for f in features if f.strand >= 0),
        "reverse_features": sum(1 for f in features if f.strand < 0),
        "variant_bins": len(tracks.bin_starts),
        "variant_samples": len(tracks.samples),
        "snp_count": tracks.snp_count,
        "indel_count": tracks.indel_count,
        "sv_count": tracks.sv_count,
        "pan_segments": len(segments),
        "center_image": "bandage" if has_bandage else "none",
        "outer_renderer": "ogdraw" if has_outer else "none",
        "inner_renderer": "bandage" if has_bandage else "schematic",
    }
    return plot_result(
        "plot_pan_circular",
        _render,
        organelle=organelle,
        method="ogdraw_bandage_matplotlib",
        metrics=metrics,
        key_findings=(
            {"metric": "accessions", "value": accession_count},
            {"metric": "genome_length", "value": genome_length},
            {"metric": "snp_count", "value": tracks.snp_count},
            {"metric": "indel_count", "value": tracks.indel_count},
        ),
        flags=tuple(flags),
        summary=(
            f"Pan-genome circular plot for {accession_count or len(tracks.samples)} "
            f"accession(s), {tracks.snp_count} SNP(s), {tracks.indel_count} indel(s)."
        ),
    )


def _render_pan_circular(
    ref: Path,
    tracks: VariantTracks,
    segments: Sequence[PanSegment],
    meta: Mapping[str, Any],
    features: Sequence[PanFeature],
    genome_length: int,
    record_id: str,
    accession_count: int,
    is_true_pi: bool,
    derived_from_variants: bool,
    outer_plot: OrganellePlot | None,
    bandage_plot: OrganellePlot | None,
    output: str | Path,
    *,
    title: str,
    dpi: int,
    figsize: tuple[float, float],
    write_intermediates: bool,
) -> Path:
    """Private path-taking renderer backing :func:`plot_pan_circular`.

    Materializes the primary composite at ``output`` and sidecars/intermediates
    into ``output.with_suffix("")`` (writer-managed). Sidecar plots captured at
    compute time are saved here.
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)

    if outer_plot is not None:
        outer_plot.save(work / "pan_ogdraw_outer.png")
    if write_intermediates and derived_from_variants and tracks.samples:
        matrix_path = work / "pan_variant_matrix.tsv"
        diversity_path = work / "pan_diversity.tsv"
        _write_variant_matrix(
            matrix_path, tracks.samples, tracks.bin_starts, tracks.bin_ends, tracks.matrix
        )
        _write_diversity(
            diversity_path,
            tracks.bin_starts,
            tracks.bin_ends,
            tracks.snp_diversity,
            tracks.indel_diversity,
        )
    bandage_path: Path | None = None
    if bandage_plot is not None:
        bandage_path = bandage_plot.save(work / "pan_bandage_inner.png")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

    _apply_figure_style(plt)

    density_data = DensityData(
        bin_starts=tracks.bin_starts,
        bin_ends=tracks.bin_ends,
        density=tuple(_window_density(tracks).tolist()),
        n_samples=len(tracks.samples),
    )
    has_sv = tracks.sv_density is not None and bool(np.any(tracks.sv_density))
    sv_data = (
        SVData(
            bin_starts=tracks.bin_starts,
            bin_ends=tracks.bin_ends,
            density=tuple(np.asarray(tracks.sv_density, dtype=float).tolist()),
            sv_count=tracks.sv_count,
        )
        if has_sv
        else None
    )
    diversity_data = DiversityData(
        bin_starts=tracks.bin_starts,
        bin_ends=tracks.bin_ends,
        snp=tuple(np.asarray(tracks.snp_diversity, dtype=float).tolist()),
        indel=tuple(np.asarray(tracks.indel_diversity, dtype=float).tolist()),
        is_true_pi=is_true_pi,
    )
    segment_data = SegmentData(
        rows=tuple(
            SegmentRow(
                start=segment.start,
                end=segment.end,
                label=segment.label,
                segment_class=segment.segment_class,
                coverage=segment.coverage,
            )
            for segment in segments
        )
    )

    track_list: list[Any] = [
        GeneRing(strand="forward"),
        GeneRing(strand="reverse"),
        DensityRing(data=density_data),
    ]
    if sv_data is not None:
        track_list.append(SVRing(data=sv_data))
    track_list.append(DiversityRing(data=diversity_data))
    if bandage_path is not None:
        center: Any = GraphCenter(data=GraphImage(bandage_path, "ok"))
    else:
        center = SchematicCenter(data=segment_data)
    track_list.append(center)

    summary_lines = _summary_lines(
        genome_length=genome_length,
        record_id=record_id,
        metadata=meta,
        accession_count=accession_count,
        feature_count=len(features),
        snp_count=tracks.snp_count,
        indel_count=tracks.indel_count,
        pan_segment_count=len(segments),
    )

    plot_circular(
        ref,
        out,
        tracks=track_list,
        title=title,
        figsize=figsize,
        dpi=dpi,
        work_dir=work,
        metadata=meta,
        info_lines=summary_lines,
    )
    return out


def _parse_reference_features(path: Path) -> tuple[list[PanFeature], int, str]:
    from Bio import SeqIO  # type: ignore

    record = SeqIO.read(str(path), "genbank")
    genome_length = len(record.seq)
    features: list[PanFeature] = []
    seen: set[tuple[str, int, int, str]] = set()
    for feat in record.features:
        if feat.type not in {
            "gene",
            "CDS",
            "tRNA",
            "rRNA",
            "tmRNA",
            "ncRNA",
            "misc_RNA",
            "repeat_region",
            "regulatory",
        }:
            continue
        start = int(feat.location.start)
        end = int(feat.location.end)
        if end <= start:
            continue
        name = _feature_name(feat)
        key = (name.lower(), start, end, feat.type)
        if key in seen:
            continue
        seen.add(key)
        strand = int(feat.location.strand or 1)
        category = _feature_category(name, feat.type)
        features.append(PanFeature(name, start, end, strand, feat.type, category))
    features.sort(key=lambda f: (f.start, f.end, f.name))
    return features, genome_length, str(record.id)


def _feature_name(feat: Any) -> str:
    qualifiers = getattr(feat, "qualifiers", {}) or {}
    for key in ("gene", "locus_tag", "product", "note"):
        value = qualifiers.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if value:
            return str(value)
    return str(getattr(feat, "type", "feature"))


def _feature_category(name: str, feature_type: str) -> str:
    lower_type = feature_type.lower()
    if lower_type in {"trna", "tmrna"}:
        return "tRNA"
    if lower_type == "rrna":
        return "rRNA"
    if lower_type == "repeat_region":
        return "ori_rep"
    return classify_gene(name)


def _load_or_derive_variant_tracks(
    *,
    variants: Path | None,
    variant_matrix: Path | None,
    diversity: Path | None,
    genome_length: int,
    bins: int,
    window_size: int | None,
    work_dir: Path,
    write_intermediates: bool,
) -> VariantTracks:
    if variant_matrix:
        base = _read_variant_matrix(variant_matrix, genome_length)
        if diversity:
            snp, indel = _read_diversity(diversity, base.bin_starts, base.bin_ends)
            return VariantTracks(
                base.samples,
                base.bin_starts,
                base.bin_ends,
                base.matrix,
                snp,
                indel,
                int(np.nansum(snp > 0)),
                int(np.nansum(indel > 0)),
            )
        return base
    if variants:
        return _derive_variant_tracks(
            variants,
            genome_length=genome_length,
            bins=bins,
            window_size=window_size,
            work_dir=work_dir,
            write_intermediates=write_intermediates,
        )
    return _empty_variant_tracks(genome_length, bins=bins, window_size=window_size)


def _empty_variant_tracks(
    genome_length: int, *, bins: int, window_size: int | None
) -> VariantTracks:
    starts, ends = _make_bins(genome_length, bins=bins, window_size=window_size)
    zeros = np.zeros(len(starts), dtype=float)
    return VariantTracks(
        (), tuple(starts), tuple(ends), np.zeros((0, len(starts))), zeros, zeros, 0, 0
    )


def _derive_variant_tracks(
    path: Path,
    *,
    genome_length: int,
    bins: int,
    window_size: int | None,
    work_dir: Path,
    write_intermediates: bool,
) -> VariantTracks:
    rows = _read_table(path)
    sample_col = _column(rows, "accession", "sample", "individual", "id")
    pos_col = _column(rows, "position", "pos", "start")
    type_col = _column(rows, "type", "variant_type", "kind", "class")
    starts, ends = _make_bins(genome_length, bins=bins, window_size=window_size)
    samples = tuple(dict.fromkeys(str(row.get(sample_col, "") or "sample") for row in rows))
    sample_index = {sample: i for i, sample in enumerate(samples)}
    matrix = np.zeros((len(samples), len(starts)), dtype=float)
    snp_counts = np.zeros(len(starts), dtype=float)
    indel_counts = np.zeros(len(starts), dtype=float)
    snp_count = 0
    indel_count = 0
    for row in rows:
        pos = _safe_int(row.get(pos_col), default=1) - 1
        idx = min(len(starts) - 1, max(0, pos // max(1, ends[0] - starts[0])))
        sample = str(row.get(sample_col, "") or "sample")
        matrix[sample_index[sample], idx] += 1
        kind = str(row.get(type_col, "SNP")).lower()
        if "indel" in kind or "ins" in kind or "del" in kind:
            indel_counts[idx] += 1
            indel_count += 1
        else:
            snp_counts[idx] += 1
            snp_count += 1
    denom = max(1, len(samples))
    snp = snp_counts / denom
    indel = indel_counts / denom

    output_paths: tuple[Path, ...] = ()
    if write_intermediates:
        matrix_path = work_dir / "pan_variant_matrix.tsv"
        diversity_path = work_dir / "pan_diversity.tsv"
        _write_variant_matrix(matrix_path, samples, starts, ends, matrix)
        _write_diversity(diversity_path, starts, ends, snp, indel)
        output_paths = (matrix_path, diversity_path)
    return VariantTracks(
        samples,
        tuple(starts),
        tuple(ends),
        matrix,
        snp,
        indel,
        snp_count,
        indel_count,
        output_paths,
    )


def _read_variant_matrix(path: Path, genome_length: int) -> VariantTracks:
    rows = _read_table(path)
    if not rows:
        return _empty_variant_tracks(genome_length, bins=1, window_size=None)
    first_col = next(iter(rows[0].keys()))
    bin_labels = list(rows[0].keys())[1:]
    starts, ends = _bins_from_labels(bin_labels, genome_length)
    samples = tuple(str(row.get(first_col, "")) for row in rows)
    matrix = np.array(
        [[_safe_float(row.get(label), default=0.0) for label in bin_labels] for row in rows],
        dtype=float,
    )
    totals = np.nansum(matrix, axis=0) if matrix.size else np.zeros(len(starts))
    denom = max(1, len(samples))
    return VariantTracks(
        samples,
        tuple(starts),
        tuple(ends),
        matrix,
        totals / denom,
        np.zeros(len(starts), dtype=float),
        int(np.nansum(totals)),
        0,
    )


def _read_diversity(
    path: Path,
    bin_starts: Sequence[int],
    bin_ends: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_table(path)
    snp_col = _column(rows, "snp", "snps", "snp_diversity", "pi_snp")
    indel_col = _column(rows, "indel", "indels", "indel_diversity", "pi_indel")
    snp = np.zeros(len(bin_starts), dtype=float)
    indel = np.zeros(len(bin_starts), dtype=float)
    for i, row in enumerate(rows[: len(bin_starts)]):
        snp[i] = _safe_float(row.get(snp_col), default=0.0)
        indel[i] = _safe_float(row.get(indel_col), default=0.0)
    return snp, indel


def _load_sv_density(
    path: Path,
    bin_starts: Sequence[int],
    bin_ends: Sequence[int],
) -> tuple[np.ndarray, int]:
    """Bin structural-variant calls into the heatmap grid.

    Accepts either a precomputed density table (start/end/sv columns) or a raw
    call list with a position column; both are aggregated onto ``bin_starts``.
    """
    density = np.zeros(len(bin_starts), dtype=float)
    if not bin_starts:
        return density, 0
    rows = _read_table(path)
    if not rows:
        return density, 0
    sv_col = _column(rows, "sv", "svs", "sv_density", "count", "n_sv", required=False)
    if sv_col:
        for i, row in enumerate(rows[: len(bin_starts)]):
            density[i] = _safe_float(row.get(sv_col), default=0.0)
        return density, int(np.nansum(density))
    pos_col = _column(rows, "position", "pos", "start", "breakpoint")
    edges = np.asarray([*list(bin_starts), bin_ends[-1]], dtype=float)
    total = 0
    for row in rows:
        pos = _safe_int(row.get(pos_col), default=-1)
        if pos < 0:
            continue
        idx = int(np.searchsorted(edges, pos, side="right")) - 1
        idx = min(len(bin_starts) - 1, max(0, idx))
        density[idx] += 1.0
        total += 1
    return density, total


def _read_pan_segments(path: Path, genome_length: int) -> list[PanSegment]:
    rows = _read_table(path)
    start_col = _column(rows, "start", "pos", "from")
    end_col = _column(rows, "end", "stop", "to")
    label_col = _column(rows, "label", "name", "segment", "id", required=False)
    class_col = _column(rows, "class", "type", "segment_class", "category", required=False)
    coverage_col = _column(rows, "coverage", "frequency", "presence", required=False)
    segments: list[PanSegment] = []
    for i, row in enumerate(rows):
        start = max(0, min(genome_length, _safe_int(row.get(start_col), default=0)))
        end = max(start + 1, min(genome_length, _safe_int(row.get(end_col), default=genome_length)))
        label = str(row.get(label_col, f"s{i + 1}") if label_col else f"s{i + 1}")
        segment_class = str(row.get(class_col, "core") if class_col else "core").lower()
        coverage = _safe_float(row.get(coverage_col), default=np.nan) if coverage_col else np.nan
        segments.append(
            PanSegment(
                start=start,
                end=end,
                label=label,
                segment_class=segment_class,
                coverage=None if math.isnan(float(coverage)) else float(coverage),
            )
        )
    return segments


def _derive_pan_segments(features: Sequence[PanFeature], genome_length: int) -> list[PanSegment]:
    if not features:
        return [PanSegment(0, genome_length, "core genome", "core", 1.0)]
    segments: list[PanSegment] = []
    for i, feat in enumerate(features):
        segment_class = "core" if feat.feature_type in {"gene", "CDS", "tRNA", "rRNA"} else "shell"
        segments.append(
            PanSegment(feat.start, feat.end, feat.name or f"s{i + 1}", segment_class, 1.0)
        )
    return segments


def _render_bandage_inner(
    *,
    pangenome_gfa: Path | None,
    output: Path,
    organelle: str,
    executable: str | Path | None,
    runner: Any | None,
    labels: tuple[str, ...],
    color_by: str,
) -> tuple[Path | None, OrganelleResult | None]:
    if pangenome_gfa is None:
        return None, None
    from .gfa_graph import plot_gfa_graph

    # Soft, uniform palette for the centre graph — teal nodes, muted grey edges,
    # no harsh outlines — instead of Bandage's default black/red depth colouring.
    result = plot_gfa_graph(
        pangenome_gfa,
        output,
        organelle=organelle,
        executable=executable,
        runner=runner,
        labels=labels,
        color_by="uniform",
        height=1900,
        font_size=40,
        node_width=13,
        edge_width=2.6,
        outline=0.0,
        extra_args=(
            "--unicolpos",
            "#4FA6A0",
            "--edgecol",
            "#AAB4BD",
            "--textcol",
            "#2B2B2B",
            "--outcol",
            "#3C736E",
        ),
    )
    if result.status == "ok" and output.exists():
        return output, result
    return None, result


def _window_density(tracks: VariantTracks) -> np.ndarray:
    """Total variants per window across all accessions (column sums)."""
    if tracks.matrix.size == 0:
        return np.zeros(len(tracks.bin_starts), dtype=float)
    return np.asarray(np.nansum(tracks.matrix, axis=0), dtype=float)


def _summary_lines(
    *,
    genome_length: int,
    record_id: str,
    metadata: Mapping[str, Any],
    accession_count: int,
    feature_count: int,
    snp_count: int,
    indel_count: int,
    pan_segment_count: int,
) -> list[str]:
    reference = metadata.get("reference") or record_id
    species = metadata.get("species")
    organelle = metadata.get("organelle", "mitochondrial")
    lines = [
        f"{organelle} pan-genome",
        f"{accession_count} accessions" if accession_count else "accessions: n/a",
        f"Reference: {reference}",
        f"Length: {_format_bp(genome_length)}",
        f"Genes: {feature_count}",
        f"SNPs: {snp_count} | indels: {indel_count}",
        f"Pan segments: {pan_segment_count}",
    ]
    if species:
        lines.insert(1, str(species))
    return lines


def _read_table(path: Path) -> list[dict[str, str]]:
    text = path.read_text().splitlines()
    if not text:
        return []
    sample = "\n".join(text[:5])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
    except csv.Error:
        dialect = csv.excel_tab
    reader = csv.DictReader(text, dialect=dialect)
    return [dict(row) for row in reader]


def _column(rows: Sequence[Mapping[str, Any]], *candidates: str, required: bool = True) -> str:
    if not rows:
        if required:
            raise ValueError("input table is empty")
        return ""
    columns = list(rows[0].keys())
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise ValueError(f"missing required column; expected one of {', '.join(candidates)}")
    return ""


def _make_bins(
    genome_length: int, *, bins: int, window_size: int | None
) -> tuple[list[int], list[int]]:
    if window_size is None:
        window_size = max(1, math.ceil(genome_length / max(1, bins)))
    starts = list(range(0, genome_length, window_size))
    ends = [min(genome_length, start + window_size) for start in starts]
    return starts, ends


def _bins_from_labels(labels: Sequence[str], genome_length: int) -> tuple[list[int], list[int]]:
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


def _write_variant_matrix(
    path: Path,
    samples: Sequence[str],
    starts: Sequence[int],
    ends: Sequence[int],
    matrix: np.ndarray,
) -> None:
    labels = [f"{a}-{b}" for a, b in zip(starts, ends, strict=False)]
    lines = ["accession\t" + "\t".join(labels)]
    for sample, values in zip(samples, matrix, strict=False):
        lines.append(str(sample) + "\t" + "\t".join(f"{float(v):g}" for v in values))
    path.write_text("\n".join(lines) + "\n")


def _write_diversity(
    path: Path,
    starts: Sequence[int],
    ends: Sequence[int],
    snp: np.ndarray,
    indel: np.ndarray,
) -> None:
    lines = ["start\tend\tsnp\tindel"]
    for start, end, s, i in zip(starts, ends, snp, indel, strict=False):
        lines.append(f"{start}\t{end}\t{float(s):.6g}\t{float(i):.6g}")
    path.write_text("\n".join(lines) + "\n")


def _theta(position: float, genome_length: int | float) -> float:
    return (float(position) / max(1.0, float(genome_length))) * 2 * np.pi


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


def _format_bp(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2g} Mb"
    if value >= 1_000:
        return f"{value / 1_000:.3g} kb"
    return f"{value} bp"


def _apply_figure_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.6,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


__all__ = ["plot_pan_circular"]
