"""gbdraw-backed organelle genome diagrams.

This module wraps the public ``gbdraw`` command-line interface.  gbdraw's
internal Python modules move quickly, while the CLI is the documented stable
surface for circular maps, linear gene-structure diagrams, BLAST comparison
tracks, and protein-BLASTP collinearity plots.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Iterable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from ..core.data import OrganelleData
from ..core.genome import OrganelleGenome
from ..core.result import OrganelleResult
from .plot_object import failed_plot_result, plot_result

Runner = Callable[[list[str], dict[str, str]], Any]
PathInput = str | Path

_GBDRAW_CANDIDATES = ("gbdraw",)
_OUTPUT_SUFFIX = {
    "svg": ".svg",
    "interactive-svg": ".interactive.svg",
    "png": ".png",
    "pdf": ".pdf",
    "eps": ".eps",
    "ps": ".ps",
}


def plot_gbdraw(
    inputs: (PathInput | Sequence[PathInput] | OrganelleGenome | OrganelleData),
    *,
    mode: str = "linear",
    organelle: str = "mito",
    executable: PathInput | None = None,
    runner: Runner | None = None,
    input_format: str | None = None,
    fasta: PathInput | Sequence[PathInput] | None = None,
    formats: str | Sequence[str] | None = None,
    title: str = "",
    # Input/comparison data
    blast: PathInput | Sequence[PathInput] | None = None,
    conservation_blast: PathInput | Sequence[PathInput] | None = None,
    depth: PathInput | Sequence[PathInput] | None = None,
    depth_track: PathInput | Sequence[PathInput] | Sequence[Sequence[PathInput]] | None = None,
    # Protein BLASTP / collinearity
    losatp_bin: str | None = None,
    ncbi_blastp_bin: str | None = None,
    losatp_threads: int | str | None = None,
    protein_blastp_mode: str | None = None,
    protein_blastp_max_hits: int | None = None,
    protein_blastp_candidate_limit: int | str | None = None,
    align_orthogroup_feature: str | None = None,
    collinear_search_scope: str | None = None,
    collinear_min_anchors: int | None = None,
    collinear_max_unit_gap: int | None = None,
    collinear_max_diagonal_drift: int | None = None,
    collinear_color_mode: str | None = None,
    pairwise_match_style: str | None = None,
    # BLAST/conservation thresholds
    evalue: float | str | None = None,
    bitscore: float | int | None = None,
    identity: float | int | None = None,
    alignment_length: int | None = None,
    # Core drawing/style
    palette: str | None = None,
    table: PathInput | None = None,
    default_colors: PathInput | None = None,
    nt: str | None = None,
    window: int | None = None,
    step: int | None = None,
    species: str | None = None,
    strain: str | None = None,
    features: str | Sequence[str] | None = None,
    feature_shape: str | Sequence[str] | None = None,
    block_stroke_color: str | None = None,
    block_stroke_width: float | int | None = None,
    axis_stroke_color: str | None = None,
    axis_stroke_width: float | int | None = None,
    line_stroke_color: str | None = None,
    line_stroke_width: float | int | None = None,
    definition_font_size: float | int | None = None,
    definition_line_style: str | Sequence[str] | None = None,
    plot_title_position: str | None = None,
    plot_title_font_size: float | int | None = None,
    label_font_size: float | int | None = None,
    legend: str | None = None,
    separate_strands: bool = False,
    resolve_overlaps: bool = False,
    # Labels and feature visibility
    labels: str | None = None,
    show_labels: str | bool | None = None,
    label_placement: str | None = None,
    label_rendering: str | None = None,
    label_rotation: float | int | None = None,
    linear_label_spacing: float | int | None = None,
    circular_label_spacing: float | int | None = None,
    label_whitelist: PathInput | None = None,
    label_blacklist: PathInput | str | None = None,
    qualifier_priority: PathInput | None = None,
    label_table: PathInput | None = None,
    feature_visibility_table: PathInput | None = None,
    # Linear layout
    track_layout: str | None = None,
    track_axis_gap: str | float | int | None = None,
    linear_track_order: str | Sequence[str] | None = None,
    linear_track_slot: str | Sequence[str] | None = None,
    linear_track_axis_index: int | None = None,
    ruler_on_axis: bool = False,
    align_center: bool = False,
    keep_definition_left_aligned: bool = False,
    record_label: str | Sequence[str] | None = None,
    record_subtitle: str | Sequence[str] | None = None,
    show_replicon: bool = False,
    hide_accession: bool = False,
    hide_length: bool = False,
    feature_height: float | int | None = None,
    gc_height: float | int | None = None,
    comparison_height: float | int | None = None,
    scale_style: str | None = None,
    scale_stroke_color: str | None = None,
    scale_stroke_width: float | int | None = None,
    scale_font_size: float | int | None = None,
    ruler_label_font_size: float | int | None = None,
    ruler_label_color: str | None = None,
    normalize_length: bool = False,
    region: str | Sequence[str] | None = None,
    record_id: str | Sequence[str] | None = None,
    reverse_complement: str | bool | Sequence[str | bool] | None = None,
    # Circular layout
    multi_record_canvas: bool = False,
    multi_record_size_mode: str | None = None,
    multi_record_min_radius_ratio: float | None = None,
    multi_record_column_gap_ratio: float | None = None,
    multi_record_row_gap_ratio: float | None = None,
    multi_record_position: str | Sequence[str] | None = None,
    keep_full_definition_with_plot_title: bool = False,
    center_reserved_radius: float | int | None = None,
    track_type: str | None = None,
    outer_label_x_radius_offset: float | int | None = None,
    outer_label_y_radius_offset: float | int | None = None,
    inner_label_x_radius_offset: float | int | None = None,
    inner_label_y_radius_offset: float | int | None = None,
    scale_interval: float | int | None = None,
    tick_label_font_size: float | int | None = None,
    feature_width: float | int | None = None,
    circular_track_order: str | Sequence[str] | None = None,
    circular_track_slot: str | Sequence[str] | None = None,
    circular_track_axis_index: int | None = None,
    gc_content_width: float | int | None = None,
    gc_content_radius: float | int | None = None,
    gc_skew_width: float | int | None = None,
    gc_skew_radius: float | int | None = None,
    # GC/depth tracks
    show_gc: bool = False,
    show_skew: bool = False,
    suppress_gc: bool = False,
    suppress_skew: bool = False,
    gc_content_mode: str | None = None,
    gc_content_min_percent: float | int | None = None,
    gc_content_max_percent: float | int | None = None,
    gc_content_tick_interval: float | int | None = None,
    gc_content_large_tick_interval: float | int | None = None,
    gc_content_small_tick_interval: float | int | None = None,
    gc_content_tick_font_size: float | int | None = None,
    show_gc_content_axis: bool | None = None,
    show_gc_content_ticks: bool | None = None,
    show_depth: bool = False,
    depth_track_label: str | Sequence[str] | None = None,
    depth_track_color: str | Sequence[str] | None = None,
    depth_track_height: float | int | str | Sequence[float | int | str] | None = None,
    depth_track_large_tick_interval: float | int | str | Sequence[float | int | str] | None = None,
    depth_track_small_tick_interval: float | int | str | Sequence[float | int | str] | None = None,
    depth_track_tick_font_size: float | int | str | Sequence[float | int | str] | None = None,
    depth_color: str | None = None,
    depth_width: float | int | None = None,
    depth_height: float | int | None = None,
    depth_window: int | None = None,
    depth_step: int | None = None,
    share_depth_axis: bool = False,
    depth_min: float | int | None = None,
    depth_max: float | int | None = None,
    depth_log_scale: bool | None = None,
    show_depth_axis: bool | None = None,
    show_depth_ticks: bool | None = None,
    depth_tick_interval: float | int | None = None,
    depth_large_tick_interval: float | int | None = None,
    depth_small_tick_interval: float | int | None = None,
    depth_tick_font_size: float | int | None = None,
    # Circular conservation rings
    conservation_reference: str | None = None,
    conservation_labels: str | Sequence[str] | None = None,
    conservation_colors: str | Sequence[str] | None = None,
    conservation_ring_width: float | int | None = None,
    conservation_ring_gap: float | int | None = None,
    legend_box_size: float | int | None = None,
    legend_font_size: float | int | None = None,
    # Session/export escape hatches
    session: PathInput | None = None,
    save_session: bool = False,
    white_background: bool = True,
    extra_args: Sequence[str] = (),
    _op: str = "plot_gbdraw",
    _extra_flags: Sequence[str] = (),
) -> OrganelleResult:
    """Render a genome diagram with the external ``gbdraw`` backend (compute-only).

    Parameters mirror the documented gbdraw CLI names. The gbdraw subprocess
    runs when the returned plot is materialized via ``ov.write(plot, output)``.
    Use :func:`plot_gene_structure` for a linear gene-layout preset and
    :func:`plot_collinearity` for a collinearity-comparison preset.
    """
    mode = _normalize_mode(mode)
    input_paths, fasta_paths, resolved_organelle, resolved_format = _resolve_inputs(
        inputs,
        fasta=fasta,
        organelle=organelle,
        input_format=input_format,
    )

    exe = _find_gbdraw(executable, runner=runner)
    if exe is None:
        return failed_plot_result(
            _op,
            organelle=resolved_organelle,
            code="visualization.missing_gbdraw",
            message=(
                "gbdraw executable not found. Install the current Bioconda "
                "package with `conda install -c conda-forge -c bioconda gbdraw` "
                "or pass executable=..."
            ),
            method="gbdraw",
            suggested_action={
                "install": "conda install -c conda-forge -c bioconda gbdraw",
                "parameter": "executable",
            },
            flags=("missing_gbdraw",),
            software_versions={"gbdraw": _gbdraw_version()},
        )

    options = {
        # input / comparison data
        "blast": blast,
        "conservation_blast": conservation_blast,
        "depth": depth,
        "depth_track": depth_track,
        # protein BLASTP / collinearity
        "losatp_bin": losatp_bin,
        "ncbi_blastp_bin": ncbi_blastp_bin,
        "losatp_threads": losatp_threads,
        "protein_blastp_mode": protein_blastp_mode,
        "protein_blastp_max_hits": protein_blastp_max_hits,
        "protein_blastp_candidate_limit": protein_blastp_candidate_limit,
        "align_orthogroup_feature": align_orthogroup_feature,
        "collinear_search_scope": collinear_search_scope,
        "collinear_min_anchors": collinear_min_anchors,
        "collinear_max_unit_gap": collinear_max_unit_gap,
        "collinear_max_diagonal_drift": collinear_max_diagonal_drift,
        "collinear_color_mode": collinear_color_mode,
        "pairwise_match_style": pairwise_match_style,
        # thresholds
        "evalue": evalue,
        "bitscore": bitscore,
        "identity": identity,
        "alignment_length": alignment_length,
        # core drawing / style
        "palette": palette,
        "table": table,
        "default_colors": default_colors,
        "nt": nt,
        "window": window,
        "step": step,
        "species": species,
        "strain": strain,
        "features": features,
        "feature_shape": feature_shape,
        "block_stroke_color": block_stroke_color,
        "block_stroke_width": block_stroke_width,
        "axis_stroke_color": axis_stroke_color,
        "axis_stroke_width": axis_stroke_width,
        "line_stroke_color": line_stroke_color,
        "line_stroke_width": line_stroke_width,
        "definition_font_size": definition_font_size,
        "definition_line_style": definition_line_style,
        "plot_title_position": plot_title_position,
        "plot_title_font_size": plot_title_font_size,
        "label_font_size": label_font_size,
        "legend": legend,
        "separate_strands": separate_strands,
        "resolve_overlaps": resolve_overlaps,
        # labels / feature visibility
        "labels": labels,
        "show_labels": show_labels,
        "label_placement": label_placement,
        "label_rendering": label_rendering,
        "label_rotation": label_rotation,
        "linear_label_spacing": linear_label_spacing,
        "circular_label_spacing": circular_label_spacing,
        "label_whitelist": label_whitelist,
        "label_blacklist": label_blacklist,
        "qualifier_priority": qualifier_priority,
        "label_table": label_table,
        "feature_visibility_table": feature_visibility_table,
        # linear layout
        "track_layout": track_layout,
        "track_axis_gap": track_axis_gap,
        "linear_track_order": linear_track_order,
        "linear_track_slot": linear_track_slot,
        "linear_track_axis_index": linear_track_axis_index,
        "ruler_on_axis": ruler_on_axis,
        "align_center": align_center,
        "keep_definition_left_aligned": keep_definition_left_aligned,
        "record_label": record_label,
        "record_subtitle": record_subtitle,
        "show_replicon": show_replicon,
        "hide_accession": hide_accession,
        "hide_length": hide_length,
        "feature_height": feature_height,
        "gc_height": gc_height,
        "comparison_height": comparison_height,
        "scale_style": scale_style,
        "scale_stroke_color": scale_stroke_color,
        "scale_stroke_width": scale_stroke_width,
        "scale_font_size": scale_font_size,
        "ruler_label_font_size": ruler_label_font_size,
        "ruler_label_color": ruler_label_color,
        "normalize_length": normalize_length,
        "region": region,
        "record_id": record_id,
        "reverse_complement": reverse_complement,
        # circular layout
        "multi_record_canvas": multi_record_canvas,
        "multi_record_size_mode": multi_record_size_mode,
        "multi_record_min_radius_ratio": multi_record_min_radius_ratio,
        "multi_record_column_gap_ratio": multi_record_column_gap_ratio,
        "multi_record_row_gap_ratio": multi_record_row_gap_ratio,
        "multi_record_position": multi_record_position,
        "keep_full_definition_with_plot_title": keep_full_definition_with_plot_title,
        "center_reserved_radius": center_reserved_radius,
        "track_type": track_type,
        "outer_label_x_radius_offset": outer_label_x_radius_offset,
        "outer_label_y_radius_offset": outer_label_y_radius_offset,
        "inner_label_x_radius_offset": inner_label_x_radius_offset,
        "inner_label_y_radius_offset": inner_label_y_radius_offset,
        "scale_interval": scale_interval,
        "tick_label_font_size": tick_label_font_size,
        "feature_width": feature_width,
        "circular_track_order": circular_track_order,
        "circular_track_slot": circular_track_slot,
        "circular_track_axis_index": circular_track_axis_index,
        "gc_content_width": gc_content_width,
        "gc_content_radius": gc_content_radius,
        "gc_skew_width": gc_skew_width,
        "gc_skew_radius": gc_skew_radius,
        # GC / depth tracks
        "show_gc": show_gc,
        "show_skew": show_skew,
        "suppress_gc": suppress_gc,
        "suppress_skew": suppress_skew,
        "gc_content_mode": gc_content_mode,
        "gc_content_min_percent": gc_content_min_percent,
        "gc_content_max_percent": gc_content_max_percent,
        "gc_content_tick_interval": gc_content_tick_interval,
        "gc_content_large_tick_interval": gc_content_large_tick_interval,
        "gc_content_small_tick_interval": gc_content_small_tick_interval,
        "gc_content_tick_font_size": gc_content_tick_font_size,
        "show_gc_content_axis": show_gc_content_axis,
        "show_gc_content_ticks": show_gc_content_ticks,
        "show_depth": show_depth,
        "depth_track_label": depth_track_label,
        "depth_track_color": depth_track_color,
        "depth_track_height": depth_track_height,
        "depth_track_large_tick_interval": depth_track_large_tick_interval,
        "depth_track_small_tick_interval": depth_track_small_tick_interval,
        "depth_track_tick_font_size": depth_track_tick_font_size,
        "depth_color": depth_color,
        "depth_width": depth_width,
        "depth_height": depth_height,
        "depth_window": depth_window,
        "depth_step": depth_step,
        "share_depth_axis": share_depth_axis,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_log_scale": depth_log_scale,
        "show_depth_axis": show_depth_axis,
        "show_depth_ticks": show_depth_ticks,
        "depth_tick_interval": depth_tick_interval,
        "depth_large_tick_interval": depth_large_tick_interval,
        "depth_small_tick_interval": depth_small_tick_interval,
        "depth_tick_font_size": depth_tick_font_size,
        # circular conservation rings
        "conservation_reference": conservation_reference,
        "conservation_labels": conservation_labels,
        "conservation_colors": conservation_colors,
        "conservation_ring_width": conservation_ring_width,
        "conservation_ring_gap": conservation_ring_gap,
        "legend_box_size": legend_box_size,
        "legend_font_size": legend_font_size,
        # session / export escape hatches
        "session": session,
        "save_session": save_session,
        "session_output": None,
    }

    def _render(path: str | Path) -> Path:
        return _render_gbdraw(
            exe,
            input_paths,
            fasta_paths,
            resolved_format,
            path,
            mode=mode,
            formats=formats,
            runner=runner,
            white_background=white_background,
            title=title,
            extra_args=extra_args,
            options=options,
        )

    flags = _result_flags(
        mode=mode,
        blast=blast,
        conservation_blast=conservation_blast,
        protein_blastp_mode=protein_blastp_mode,
        depth=depth,
        depth_track=depth_track,
        show_depth=show_depth,
        extra_flags=_extra_flags,
    )
    metrics = {
        "method": "gbdraw",
        "mode": mode,
        "input_format": resolved_format,
        "inputs": len(input_paths),
        "renderer": Path(str(exe)).name,
    }
    return plot_result(
        _op,
        _render,
        organelle=resolved_organelle,
        method="gbdraw",
        metrics=metrics,
        key_findings=(
            {"metric": "mode", "value": mode},
            {"metric": "inputs", "value": len(input_paths)},
            {"metric": "renderer", "value": Path(str(exe)).name},
        ),
        flags=flags,
        summary=f"gbdraw will render {mode} diagram for {len(input_paths)} input file(s).",
        software_versions={"gbdraw": _gbdraw_version()},
    )


def _render_gbdraw(
    executable: str,
    input_paths: tuple[Path, ...],
    fasta_paths: tuple[Path, ...],
    resolved_format: str,
    destination: str | Path,
    *,
    mode: str,
    formats: str | Sequence[str] | None,
    runner: Runner | None,
    white_background: bool,
    title: str,
    extra_args: Sequence[str],
    options: dict[str, Any],
) -> Path:
    """Private path-taking renderer backing :func:`plot_gbdraw`.

    Runs the gbdraw CLI to materialize ``destination``. Writer-time failures
    (nonzero exit or missing output) raise so ``ov.write`` leaves no partial
    destination.
    """
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = out.with_suffix("") if out.suffix else out
    format_tokens = _resolve_formats(out, formats)
    expected_outputs = _expected_output_paths(output_prefix, format_tokens, fallback=out)
    argv = _build_gbdraw_argv(
        executable,
        mode=mode,
        input_paths=input_paths,
        input_format=resolved_format,
        fasta_paths=fasta_paths,
        output_prefix=output_prefix,
        formats=format_tokens,
        title=title,
        extra_args=extra_args,
        **options,
    )
    env = dict(os.environ)
    run_result = _run_gbdraw(argv, env=env, runner=runner)
    return_code = _return_code(run_result)
    if return_code != 0:
        raise RuntimeError(_failure_summary(run_result))
    existing_outputs = tuple(p for p in expected_outputs if p.exists())
    if not existing_outputs:
        raise RuntimeError(f"gbdraw exited successfully but did not create {out}.")
    if white_background:
        _apply_white_background(existing_outputs, output_prefix=output_prefix)
    return existing_outputs[0]


def plot_gene_structure(
    inputs: (PathInput | Sequence[PathInput] | OrganelleGenome | OrganelleData),
    *,
    show_labels: str | bool | None = "all",
    separate_strands: bool = True,
    label_placement: str | None = "above_feature",
    track_layout: str | None = "above",
    **kwargs: Any,
) -> OrganelleResult:
    """Render a linear gene-structure diagram with gbdraw (compute-only).

    This preset keeps genes and labels prominent, making it suitable for
    annotated mitochondrial/chloroplast loci, contigs, and full organelle maps.
    Render with ``ov.write(plot, output)``.
    """
    return plot_gbdraw(
        inputs,
        mode="linear",
        show_labels=show_labels,
        separate_strands=separate_strands,
        label_placement=label_placement,
        track_layout=track_layout,
        _op="plot_gene_structure",
        _extra_flags=("gene_structure_plot",),
        **kwargs,
    )


def plot_collinearity(
    inputs: (PathInput | Sequence[PathInput] | OrganelleGenome | OrganelleData),
    *,
    protein_blastp_mode: str | None = "collinear",
    collinear_search_scope: str | None = "adjacent",
    collinear_color_mode: str | None = "orientation",
    pairwise_match_style: str | None = "ribbon",
    **kwargs: Any,
) -> OrganelleResult:
    """Render a gbdraw linear collinearity/comparison plot (compute-only).

    Provide either precomputed ``blast=...`` outfmt 6/7 files or let gbdraw run
    protein BLASTP/LOSAT through ``protein_blastp_mode``. Render with
    ``ov.write(plot, output)``.
    """
    return plot_gbdraw(
        inputs,
        mode="linear",
        protein_blastp_mode=protein_blastp_mode,
        collinear_search_scope=collinear_search_scope,
        collinear_color_mode=collinear_color_mode,
        pairwise_match_style=pairwise_match_style,
        _op="plot_collinearity",
        _extra_flags=("collinearity_plot",),
        **kwargs,
    )


plot_gene_synteny = plot_collinearity
plot_synteny = plot_gene_synteny  # Backward-compatible alias; prefer ov.viz.gene.synteny.


def _build_gbdraw_argv(
    executable: str,
    *,
    mode: str,
    input_paths: tuple[Path, ...],
    input_format: str,
    fasta_paths: tuple[Path, ...],
    output_prefix: Path,
    formats: tuple[str, ...],
    title: str,
    extra_args: Sequence[str],
    **options: Any,
) -> list[str]:
    argv = [executable, mode]
    if input_format == "gff":
        argv.append("--gff")
        argv.extend(str(p) for p in input_paths)
        argv.append("--fasta")
        argv.extend(str(p) for p in fasta_paths)
    else:
        argv.append("--gbk")
        argv.extend(str(p) for p in input_paths)
    argv.extend(["-o", str(output_prefix), "-f", ",".join(formats)])

    _add_values(argv, "-b", options["blast"])
    _add_values(argv, "--conservation_blast", options["conservation_blast"])
    _add_values(argv, "--depth", options["depth"])
    _add_grouped_values(argv, "--depth_track", options["depth_track"])

    _add_option(argv, "--losatp_bin", options["losatp_bin"])
    _add_option(argv, "--ncbi_blastp_bin", options["ncbi_blastp_bin"])
    _add_option(argv, "--losatp_threads", options["losatp_threads"])
    _add_option(argv, "--protein_blastp_mode", options["protein_blastp_mode"])
    _add_option(argv, "--protein_blastp_max_hits", options["protein_blastp_max_hits"])
    _add_option(argv, "--protein_blastp_candidate_limit", options["protein_blastp_candidate_limit"])
    _add_option(argv, "--align_orthogroup_feature", options["align_orthogroup_feature"])
    _add_option(argv, "--collinear_search_scope", options["collinear_search_scope"])
    _add_option(argv, "--collinear_min_anchors", options["collinear_min_anchors"])
    _add_option(argv, "--collinear_max_unit_gap", options["collinear_max_unit_gap"])
    _add_option(argv, "--collinear_max_diagonal_drift", options["collinear_max_diagonal_drift"])
    _add_option(argv, "--collinear_color_mode", options["collinear_color_mode"])
    _add_option(argv, "--pairwise_match_style", options["pairwise_match_style"])
    _add_option(argv, "--evalue", options["evalue"])
    _add_option(argv, "--bitscore", options["bitscore"])
    _add_option(argv, "--identity", options["identity"])
    _add_option(argv, "--alignment_length", options["alignment_length"])

    _add_option(argv, "--palette", options["palette"])
    _add_option(argv, "--table", options["table"])
    _add_option(argv, "--default_colors", options["default_colors"])
    _add_option(argv, "--nt", options["nt"])
    _add_option(argv, "--window", options["window"])
    _add_option(argv, "--step", options["step"])
    _add_option(argv, "--species", options["species"])
    _add_option(argv, "--strain", options["strain"])
    if options["features"] is not None:
        _add_option(argv, "--features", _comma(options["features"]))
    _add_repeated(argv, "--feature_shape", options["feature_shape"])
    _add_option(argv, "--block_stroke_color", options["block_stroke_color"])
    _add_option(argv, "--block_stroke_width", options["block_stroke_width"])
    _add_option(argv, "--axis_stroke_color", options["axis_stroke_color"])
    _add_option(argv, "--axis_stroke_width", options["axis_stroke_width"])
    _add_option(argv, "--line_stroke_color", options["line_stroke_color"])
    _add_option(argv, "--line_stroke_width", options["line_stroke_width"])
    _add_option(argv, "--definition_font_size", options["definition_font_size"])
    _add_repeated(argv, "--definition_line_style", options["definition_line_style"])
    _add_option(argv, "--plot_title", title or None)
    _add_option(argv, "--plot_title_position", options["plot_title_position"])
    _add_option(argv, "--plot_title_font_size", options["plot_title_font_size"])
    _add_option(argv, "--label_font_size", options["label_font_size"])
    _add_option(argv, "--legend", options["legend"])
    _add_flag(argv, "--separate_strands", options["separate_strands"])
    _add_flag(argv, "--resolve_overlaps", options["resolve_overlaps"])

    _add_optional_value_flag(argv, "--labels", options["labels"])
    _add_optional_value_flag(argv, "--show_labels", _show_labels_value(options["show_labels"]))
    _add_option(argv, "--label_placement", options["label_placement"])
    _add_option(argv, "--label_rendering", options["label_rendering"])
    _add_option(argv, "--label_rotation", options["label_rotation"])
    _add_option(argv, "--linear_label_spacing", options["linear_label_spacing"])
    _add_option(argv, "--circular_label_spacing", options["circular_label_spacing"])
    _add_option(argv, "--label_whitelist", options["label_whitelist"])
    _add_option(argv, "--label_blacklist", options["label_blacklist"])
    _add_option(argv, "--qualifier_priority", options["qualifier_priority"])
    _add_option(argv, "--label_table", options["label_table"])
    _add_option(argv, "--feature_visibility_table", options["feature_visibility_table"])

    _add_option(argv, "--track_layout", options["track_layout"])
    _add_option(argv, "--track_axis_gap", options["track_axis_gap"])
    if options["linear_track_order"] is not None:
        _add_option(argv, "--linear_track_order", _comma(options["linear_track_order"]))
    _add_repeated(argv, "--linear_track_slot", options["linear_track_slot"])
    _add_option(argv, "--linear_track_axis_index", options["linear_track_axis_index"])
    _add_flag(argv, "--ruler_on_axis", options["ruler_on_axis"])
    _add_flag(argv, "--align_center", options["align_center"])
    _add_flag(argv, "--keep_definition_left_aligned", options["keep_definition_left_aligned"])
    _add_repeated(argv, "--record_label", options["record_label"])
    _add_repeated(argv, "--record_subtitle", options["record_subtitle"])
    _add_flag(argv, "--show_replicon", options["show_replicon"])
    _add_flag(argv, "--hide_accession", options["hide_accession"])
    _add_flag(argv, "--hide_length", options["hide_length"])
    _add_option(argv, "--feature_height", options["feature_height"])
    _add_option(argv, "--gc_height", options["gc_height"])
    _add_option(argv, "--comparison_height", options["comparison_height"])
    _add_option(argv, "--scale_style", options["scale_style"])
    _add_option(argv, "--scale_stroke_color", options["scale_stroke_color"])
    _add_option(argv, "--scale_stroke_width", options["scale_stroke_width"])
    _add_option(argv, "--scale_font_size", options["scale_font_size"])
    _add_option(argv, "--ruler_label_font_size", options["ruler_label_font_size"])
    _add_option(argv, "--ruler_label_color", options["ruler_label_color"])
    _add_flag(argv, "--normalize_length", options["normalize_length"])
    _add_repeated(argv, "--region", options["region"])
    _add_repeated(argv, "--record_id", options["record_id"])
    _add_repeated(argv, "--reverse_complement", options["reverse_complement"])

    _add_flag(argv, "--multi_record_canvas", options["multi_record_canvas"])
    _add_option(argv, "--multi_record_size_mode", options["multi_record_size_mode"])
    _add_option(argv, "--multi_record_min_radius_ratio", options["multi_record_min_radius_ratio"])
    _add_option(argv, "--multi_record_column_gap_ratio", options["multi_record_column_gap_ratio"])
    _add_option(argv, "--multi_record_row_gap_ratio", options["multi_record_row_gap_ratio"])
    _add_repeated(argv, "--multi_record_position", options["multi_record_position"])
    _add_flag(
        argv,
        "--keep_full_definition_with_plot_title",
        options["keep_full_definition_with_plot_title"],
    )
    _add_option(argv, "--center_reserved_radius", options["center_reserved_radius"])
    _add_option(argv, "--track_type", options["track_type"])
    _add_option(argv, "--outer_label_x_radius_offset", options["outer_label_x_radius_offset"])
    _add_option(argv, "--outer_label_y_radius_offset", options["outer_label_y_radius_offset"])
    _add_option(argv, "--inner_label_x_radius_offset", options["inner_label_x_radius_offset"])
    _add_option(argv, "--inner_label_y_radius_offset", options["inner_label_y_radius_offset"])
    _add_option(argv, "--scale_interval", options["scale_interval"])
    _add_option(argv, "--tick_label_font_size", options["tick_label_font_size"])
    _add_option(argv, "--feature_width", options["feature_width"])
    if options["circular_track_order"] is not None:
        _add_option(argv, "--circular_track_order", _comma(options["circular_track_order"]))
    _add_repeated(argv, "--circular_track_slot", options["circular_track_slot"])
    _add_option(argv, "--circular_track_axis_index", options["circular_track_axis_index"])
    _add_option(argv, "--gc_content_width", options["gc_content_width"])
    _add_option(argv, "--gc_content_radius", options["gc_content_radius"])
    _add_option(argv, "--gc_skew_width", options["gc_skew_width"])
    _add_option(argv, "--gc_skew_radius", options["gc_skew_radius"])

    _add_flag(argv, "--show_gc", options["show_gc"])
    _add_flag(argv, "--show_skew", options["show_skew"])
    _add_flag(argv, "--suppress_gc", options["suppress_gc"])
    _add_flag(argv, "--suppress_skew", options["suppress_skew"])
    _add_option(argv, "--gc_content_mode", options["gc_content_mode"])
    _add_option(argv, "--gc_content_min_percent", options["gc_content_min_percent"])
    _add_option(argv, "--gc_content_max_percent", options["gc_content_max_percent"])
    _add_option(argv, "--gc_content_tick_interval", options["gc_content_tick_interval"])
    _add_option(argv, "--gc_content_large_tick_interval", options["gc_content_large_tick_interval"])
    _add_option(argv, "--gc_content_small_tick_interval", options["gc_content_small_tick_interval"])
    _add_option(argv, "--gc_content_tick_font_size", options["gc_content_tick_font_size"])
    _add_bool_pair(
        argv, options["show_gc_content_axis"], "--show_gc_content_axis", "--hide_gc_content_axis"
    )
    _add_bool_pair(
        argv, options["show_gc_content_ticks"], "--show_gc_content_ticks", "--hide_gc_content_ticks"
    )
    _add_flag(argv, "--show_depth", options["show_depth"])
    _add_values(argv, "--depth_track_label", options["depth_track_label"])
    _add_values(argv, "--depth_track_color", options["depth_track_color"])
    _add_values(argv, "--depth_track_height", options["depth_track_height"])
    _add_values(
        argv, "--depth_track_large_tick_interval", options["depth_track_large_tick_interval"]
    )
    _add_values(
        argv, "--depth_track_small_tick_interval", options["depth_track_small_tick_interval"]
    )
    _add_values(argv, "--depth_track_tick_font_size", options["depth_track_tick_font_size"])
    _add_option(argv, "--depth_color", options["depth_color"])
    _add_option(argv, "--depth_width", options["depth_width"])
    _add_option(argv, "--depth_height", options["depth_height"])
    _add_option(argv, "--depth_window", options["depth_window"])
    _add_option(argv, "--depth_step", options["depth_step"])
    _add_flag(argv, "--share_depth_axis", options["share_depth_axis"])
    _add_option(argv, "--depth_min", options["depth_min"])
    _add_option(argv, "--depth_max", options["depth_max"])
    _add_bool_pair(argv, options["depth_log_scale"], "--depth_log_scale", "--no_depth_log_scale")
    _add_bool_pair(argv, options["show_depth_axis"], "--show_depth_axis", "--hide_depth_axis")
    _add_bool_pair(argv, options["show_depth_ticks"], "--show_depth_ticks", "--hide_depth_ticks")
    _add_option(argv, "--depth_tick_interval", options["depth_tick_interval"])
    _add_option(argv, "--depth_large_tick_interval", options["depth_large_tick_interval"])
    _add_option(argv, "--depth_small_tick_interval", options["depth_small_tick_interval"])
    _add_option(argv, "--depth_tick_font_size", options["depth_tick_font_size"])

    _add_option(argv, "--conservation_reference", options["conservation_reference"])
    _add_values(argv, "--conservation_labels", options["conservation_labels"])
    _add_values(argv, "--conservation_colors", options["conservation_colors"])
    _add_option(argv, "--conservation_ring_width", options["conservation_ring_width"])
    _add_option(argv, "--conservation_ring_gap", options["conservation_ring_gap"])
    _add_option(argv, "--legend_box_size", options["legend_box_size"])
    _add_option(argv, "--legend_font_size", options["legend_font_size"])

    _add_option(argv, "--session", options["session"])
    _add_flag(argv, "--save_session", options["save_session"])
    _add_option(argv, "--session_output", options["session_output"])
    argv.extend(str(arg) for arg in extra_args)
    return argv


def _resolve_inputs(
    inputs: (PathInput | Sequence[PathInput] | OrganelleGenome | OrganelleData),
    *,
    fasta: PathInput | Sequence[PathInput] | None,
    organelle: str,
    input_format: str | None,
) -> tuple[tuple[Path, ...], tuple[Path, ...], str, str]:
    if isinstance(inputs, OrganelleGenome):
        artifact = inputs.annotation or inputs.sequence
        path = artifact.resolve() if artifact is not None else None
        if path is None:
            raise ValueError("OrganelleGenome needs an annotation or sequence artifact.")
        input_paths = (Path(path),)
        sequence_path = inputs.sequence.resolve() if inputs.sequence is not None else None
        fasta_paths = _paths(fasta or sequence_path)
        resolved_organelle = "mito" if inputs.organelle == "mitochondrion" else "plastid"
    elif isinstance(inputs, OrganelleData):
        candidate = (
            inputs.artifacts.get("gbk")
            or inputs.artifacts.get("genbank")
            or inputs.artifacts.get("annotation")
            or inputs.artifacts.get("gff")
        )
        if candidate is None:
            raise ValueError("OrganelleData needs a gbk/genbank/annotation/gff artifact.")
        input_paths = _paths(candidate.resolve())
        sequence = inputs.artifacts.get("fasta")
        fasta_paths = _paths(fasta or (sequence.resolve() if sequence is not None else None))
        resolved_organelle = str(inputs.metadata.get("organelle", organelle))
    else:
        input_paths = _paths(inputs)
        fasta_paths = _paths(fasta)
        resolved_organelle = organelle

    if not input_paths:
        raise ValueError("plot_gbdraw() needs at least one input file.")
    resolved_format = _infer_input_format(input_paths, input_format)
    if resolved_format == "gff" and not fasta_paths:
        raise ValueError("GFF3 input requires fasta=... for gbdraw.")
    return input_paths, fasta_paths, resolved_organelle, resolved_format


def _paths(value: Any) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        return (Path(value),)
    return tuple(Path(v) for v in value)


def _infer_input_format(paths: tuple[Path, ...], input_format: str | None) -> str:
    if input_format is not None:
        normalized = input_format.lower().strip()
        if normalized in {"gb", "gbk", "genbank"}:
            return "gbk"
        if normalized in {"gff", "gff3"}:
            return "gff"
        raise ValueError("input_format must be 'gbk'/'genbank' or 'gff'/'gff3'.")
    if any(p.suffix.lower() in {".gff", ".gff3"} for p in paths):
        return "gff"
    return "gbk"


def _normalize_mode(mode: str) -> str:
    normalized = mode.lower().strip()
    if normalized not in {"linear", "circular"}:
        raise ValueError("mode must be 'linear' or 'circular'.")
    return normalized


def _resolve_formats(output: Path, formats: str | Sequence[str] | None) -> tuple[str, ...]:
    if formats is None:
        suffix = output.suffix.lower().lstrip(".")
        return (suffix or "svg",)
    if isinstance(formats, str):
        values = [part.strip() for part in formats.split(",")]
    else:
        values = [str(part).strip() for part in formats]
    cleaned = tuple(value for value in values if value)
    return cleaned or ("svg",)


def _expected_output_paths(
    prefix: Path, formats: tuple[str, ...], *, fallback: Path
) -> tuple[Path, ...]:
    if len(formats) == 1 and fallback.suffix:
        return (fallback,)
    return tuple(prefix.with_suffix(_OUTPUT_SUFFIX.get(fmt, f".{fmt}")) for fmt in formats)


def _find_gbdraw(executable: PathInput | None, *, runner: Runner | None) -> str | None:
    if executable is not None:
        exe = str(executable)
        if runner is not None:
            return exe
        if os.sep in exe:
            return exe if Path(exe).exists() else None
        return shutil.which(exe)
    if runner is not None:
        return "gbdraw"
    for candidate in _GBDRAW_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _run_gbdraw(argv: list[str], *, env: dict[str, str], runner: Runner | None) -> Any:
    if runner is not None:
        return runner(argv, env)
    return subprocess.run(argv, check=False, capture_output=True, text=True, env=env)


def _return_code(run_result: Any) -> int:
    if run_result is None:
        return 0
    if isinstance(run_result, int):
        return run_result
    return int(getattr(run_result, "returncode", 0))


def _failure_summary(run_result: Any) -> str:
    stderr = str(getattr(run_result, "stderr", "") or "").strip()
    stdout = str(getattr(run_result, "stdout", "") or "").strip()
    return (stderr or stdout or "gbdraw rendering failed.")[:1000]


def _apply_white_background(paths: tuple[Path, ...], *, output_prefix: Path) -> None:
    svg_candidates = {p for p in paths if p.suffix.lower() == ".svg"}
    sibling_svg = output_prefix.with_suffix(".svg")
    if sibling_svg.exists():
        svg_candidates.add(sibling_svg)
    for svg in svg_candidates:
        _apply_white_svg_background(svg)
    for path in paths:
        if path.suffix.lower() == ".png":
            _flatten_png_on_white(path, source_svg=sibling_svg if sibling_svg.exists() else None)


def _apply_white_svg_background(path: Path) -> None:
    try:
        text = path.read_text()
    except OSError:
        return
    if 'data-organelleverse-bg="white"' in text:
        return
    svg_start = text.find("<svg")
    if svg_start < 0:
        return
    svg_open_end = text.find(">", svg_start)
    if svg_open_end < 0:
        return
    bg = '<rect data-organelleverse-bg="white" width="100%" height="100%" fill="white" />'
    path.write_text(text[: svg_open_end + 1] + bg + text[svg_open_end + 1 :])


def _flatten_png_on_white(path: Path, *, source_svg: Path | None) -> None:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            if image.mode not in {"RGBA", "LA"} and "transparency" not in image.info:
                return
            rgba = image.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, "white")
            canvas.alpha_composite(rgba)
            canvas.convert("RGB").save(path)
            return
    except Exception:
        pass
    if source_svg is None:
        return
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(url=str(source_svg), write_to=str(path), background_color="white")
    except Exception:
        return


def _gbdraw_version() -> str:
    try:
        return metadata.version("gbdraw")
    except metadata.PackageNotFoundError:
        return ""


def _result_flags(
    *,
    mode: str,
    blast: Any,
    conservation_blast: Any,
    protein_blastp_mode: str | None,
    depth: Any,
    depth_track: Any,
    show_depth: bool,
    extra_flags: Sequence[str],
) -> tuple[str, ...]:
    flags = ["gbdraw_rendered", "third_party_renderer", f"{mode}_map"]
    if blast is not None or protein_blastp_mode not in {None, "", "none"}:
        flags.append("comparison_plot")
    if protein_blastp_mode == "collinear":
        flags.append("collinearity_plot")
    if conservation_blast is not None:
        flags.append("conservation_ring")
    if depth is not None or depth_track is not None or show_depth:
        flags.append("depth_track")
    flags.extend(extra_flags)
    return tuple(dict.fromkeys(flags))


def _add_option(argv: list[str], flag: str, value: Any) -> None:
    if value is not None:
        argv.extend([flag, _cli_value(value)])


def _add_flag(argv: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


def _add_bool_pair(argv: list[str], value: bool | None, true_flag: str, false_flag: str) -> None:
    if value is True:
        argv.append(true_flag)
    elif value is False:
        argv.append(false_flag)


def _add_values(argv: list[str], flag: str, values: Any) -> None:
    normalized = _tuple_or_none(values)
    if normalized is None:
        return
    argv.append(flag)
    argv.extend(_cli_value(value) for value in normalized)


def _add_grouped_values(argv: list[str], flag: str, values: Any) -> None:
    if values is None:
        return
    if _is_sequence(values) and values and all(_is_sequence(item) for item in values):
        for group in values:
            _add_values(argv, flag, group)
        return
    _add_values(argv, flag, values)


def _add_repeated(argv: list[str], flag: str, values: Any) -> None:
    normalized = _tuple_or_none(values)
    if normalized is None:
        return
    for value in normalized:
        argv.extend([flag, _cli_value(value)])


def _add_optional_value_flag(argv: list[str], flag: str, value: str | bool | None) -> None:
    if value is None or value is False:
        return
    argv.append(flag)
    if value is not True:
        argv.append(_cli_value(value))


def _tuple_or_none(values: Any) -> tuple[Any, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes, Path)):
        return (values,)
    if isinstance(values, Iterable):
        return tuple(values)
    return (values,)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Path))


def _cli_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _comma(value: str | Sequence[str]) -> str:
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


def _show_labels_value(value: str | bool | None) -> str | bool | None:
    if value is True:
        return "all"
    return value


__all__ = [
    "plot_collinearity",
    "plot_gbdraw",
    "plot_gene_structure",
    "plot_gene_synteny",
]
