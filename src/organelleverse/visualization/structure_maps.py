"""Annotation-backed structure panels and tracks on the existing OGDraw map."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

from organelleverse._public_validation import validated_public

from ..core.errors import OrganelleInputError, OrganelleParameterError
from .features import (
    TRANS_BASIS_TEXT,
    annotation_rows,
    genes_from_rows,
    record_display_name,
    resolve_ir_regions,
)
from .ogdraw import (
    GENE_COLORS,
    MitoGenome,
    _bp_to_deg,
    _draw_arc_block,
    _load_genbank_records,
    _parse_genbank_record,
    draw_mito_map,
)
from .plot_object import OrganellePlot, plot_result
from .style import FONT_PT, publication_style, width_inches

# Kept under their historical names; the logic is shared with the OGDraw map.
_annotations = annotation_rows
_structure_genes = genes_from_rows


def _tracks(ssrs, tandem_repeats, dispersed_repeats, length):
    tracks = []
    for kind, rows in (("SSR", ssrs), ("tandem", tandem_repeats), ("dispersed", dispersed_repeats)):
        if rows is None:
            continue
        intervals = []
        for row in rows:
            if kind == "dispersed":
                spans = [(int(p), int(p) + int(row["length"]) - 1) for p in row["positions"]]
            else:
                spans = [(int(row["start"]), int(row["end"]))]
            for start, end in spans:
                if not 1 <= start <= end <= length:
                    raise OrganelleParameterError(
                        code="visualization.structure.track_coordinates",
                        message="Repeat tracks require 1-based inclusive coordinates within the genome.",
                    )
                intervals.append({"start": start, "end": end})
        tracks.append({"name": kind, "intervals": intervals})
    return tracks


@validated_public('visualization.plot_structure_map')
def plot_structure_map(
    genbank_path: Sequence[str | Path],
    *,
    genome_name: str | None = None,
    show_ir: bool | None = None,
    ssrs: Sequence[Mapping[str, int | float | str | list[int]]] | None = None,
    tandem_repeats: Sequence[Mapping[str, int | float | str | list[int]]] | None = None,
    dispersed_repeats: Sequence[Mapping[str, int | float | str | list[int]]] | None = None,
    dpi: int = 200,
) -> OrganellePlot:
    """Prepare a CPGView/PMGmap-style OGDraw map with exon structure panels.

    ``genbank_path`` may be one file, a file with several records, or a list of files;
    more than one record puts one circle per record on a single figure.

    CDS/tRNA/rRNA join parts supply 1-based inclusive exon coordinates in
    transcript order; gene-only loci remain on the circle but are not called
    exons. Trans-splicing comes from ``/trans_splicing`` on the feature or its
    parent gene; a file without it is inferred from the exon layout (mixed
    strands, a backward jump, or a gap above ``MAX_CIS_INTRON`` = 25 kb, the
    same threshold the annotation writer uses). Each row's ``trans_basis`` says
    which (``"qualifier"``, ``"mixed_strand"``, ``"backward"``, ``"distance"``)
    and inferred panels say so in their title. Every copy is retained. Dashed links
    connect successive parts of a trans-spliced transcript; they do not assign
    individual introns a cis/trans mechanism.

    Pass existing ``compute_repeats(...)["ssrs"]`` and
    ``compute_multiconf(...)["repeats"]`` results as tracks. Tandem rows use
    start/end coordinates from an existing repeat analysis.
    No repeat detection runs here. ``show_ir`` reuses the circular plastome IR
    detector. ``None`` (default) draws the partition for plastid records that have one
    and skips it silently otherwise; ``True`` fails explicitly when it cannot be
    resolved, ``False`` never draws it.

    ``metrics["features"]`` is the exact block data used to render. Rendering
    via ``ov.write`` also exports these blocks to ``<stem>.exons.tsv``.
    """
    try:
        records = [record for _, record in _load_genbank_records(genbank_path)]
    except (OSError, ValueError) as exc:
        raise OrganelleInputError(code="visualization.structure.genbank", message=str(exc)) from exc
    if not records:
        raise OrganelleInputError(
            code="visualization.structure.genbank", message="No GenBank records found."
        )
    if len(records) > 1:
        return _plot_multi_contig_structure_map(
            records,
            genome_name=genome_name,
            show_ir=show_ir,
            ssrs=ssrs,
            tandem_repeats=tandem_repeats,
            dispersed_repeats=dispersed_repeats,
            dpi=dpi,
        )
    record = records[0]
    if not len(record):
        raise OrganelleInputError(
            code="visualization.structure.empty", message="GenBank sequence is empty."
        )
    parsed = _parse_genbank_record(record)
    features = _annotations(record, parsed.organelle)
    if not features:
        raise OrganelleInputError(
            code="visualization.structure.annotations",
            message="No gene, CDS, tRNA or rRNA annotations found.",
        )
    parsed.genes = _structure_genes(features)
    regions = resolve_ir_regions(record, parsed.organelle, show_ir)
    tracks = _tracks(ssrs, tandem_repeats, dispersed_repeats, len(record))
    name = genome_name or record_display_name(record, record.id)

    def render(path):
        return _render_structure_map(Path(path), parsed, features, tracks, regions, name, dpi)

    return plot_result(
        "plot_structure_map",
        render,
        organelle=parsed.organelle,
        method="ogdraw_structure",
        metrics={
            "genome_name": name,
            "genome_length": len(record),
            "organelle": parsed.organelle,
            "dpi": dpi,
            "coordinate_system": "1-based inclusive; IR regions: 0-based start and length",
            "features": features,
            "tracks": tracks,
            "regions": regions,
            "cis_transcripts": sum(f["splicing"] == "cis" for f in features),
            "trans_transcripts": sum(f["splicing"] == "trans" for f in features),
        },
        summary="Organelle structure map and exon panels prepared.",
    )


def _plot_multi_contig_structure_map(
    records,
    *,
    genome_name,
    show_ir,
    ssrs,
    tandem_repeats,
    dispersed_repeats,
    dpi,
) -> OrganellePlot:
    """One circle per contig record on a shared figure, never concatenated."""
    if show_ir:
        raise OrganelleParameterError(
            code="visualization.structure.ir_single_record",
            message="IR partitioning requires a single-record GenBank.",
        )
    if ssrs or tandem_repeats or dispersed_repeats:
        raise OrganelleParameterError(
            code="visualization.structure.track_single_record",
            message="Repeat tracks require a single-record GenBank; contig-local tracks are not assigned.",
        )
    contigs = []
    for record in records:
        if not len(record):
            raise OrganelleInputError(
                code="visualization.structure.empty",
                message=f"GenBank sequence for {record.id} is empty.",
            )
        parsed = _parse_genbank_record(record, gc_window=0)
        features = _annotations(record, parsed.organelle)
        parsed.genes = _structure_genes(features)
        contigs.append(
            {"name": record.name or record.id, "length": len(record), "organelle": parsed.organelle,
             "parsed": parsed, "features": features}
        )
    if not any(contig["features"] for contig in contigs):
        raise OrganelleInputError(
            code="visualization.structure.annotations",
            message="No gene, CDS, tRNA or rRNA annotations found on any contig.",
        )
    rows: list[dict] = []
    for contig in contigs:
        renumbered = []
        for row in contig["features"]:
            renumbered_row = {**row, "contig": contig["name"], "feature_index": len(rows)}
            renumbered.append(renumbered_row)
            rows.append(renumbered_row)
        contig["features"] = renumbered
    name = genome_name or f"{len(records)} contigs"

    def render(path):
        return _render_contig_map_bundle(Path(path), contigs, name, dpi)

    return plot_result(
        "plot_structure_map",
        render,
        organelle=contigs[0]["organelle"],
        method="ogdraw_structure",
        metrics={
            "genome_name": name,
            "genome_length": sum(contig["length"] for contig in contigs),
            "organelle": contigs[0]["organelle"],
            "contig_count": len(contigs),
            "dpi": dpi,
            "coordinate_system": "1-based inclusive within each contig record",
            "features": rows,
            "contigs": [
                {
                    "name": contig["name"],
                    "length": contig["length"],
                    "organelle": contig["organelle"],
                    "features": contig["features"],
                }
                for contig in contigs
            ],
            "cis_transcripts": sum(row["splicing"] == "cis" for row in rows),
            "trans_transcripts": sum(row["splicing"] == "trans" for row in rows),
        },
        summary=f"Per-contig organelle structure maps ({len(contigs)} contigs) prepared.",
    )


def _restore_structure_plot(result):
    """Reattach rendering to a serialized, fully computed structure-map Result."""
    plot = OrganellePlot.model_validate(result.model_dump(mode="python"))
    data = plot.metrics
    if data.get("contigs"):
        contigs = [
            {
                "name": contig["name"],
                "length": contig["length"],
                "organelle": contig["organelle"],
                "parsed": MitoGenome(
                    genes=_structure_genes(contig["features"]),
                    gc=[],
                    genome_length=contig["length"],
                    organelle=contig["organelle"],
                ),
                "features": contig["features"],
            }
            for contig in data["contigs"]
        ]

        def render_multi(path):
            return _render_contig_map_bundle(
                Path(path), contigs, data["genome_name"], data["dpi"]
            )

        plot._renderer = render_multi
        return plot
    parsed = MitoGenome(
        genes=_structure_genes(data["features"]),
        gc=[],
        genome_length=data["genome_length"],
        organelle=data["organelle"],
    )

    def render(path):
        return _render_structure_map(
            Path(path),
            parsed,
            data["features"],
            data["tracks"],
            data["regions"],
            data["genome_name"],
            data["dpi"],
        )

    plot._renderer = render
    return plot


def _draw_structure_tracks(ax, length, features, tracks, regions):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    for region, span in regions.items():
        a, b = span["start"], span["start"] + span["length"]
        color = {"LSC": "#cde8bc", "SSC": "#ffd6a0", "IRa": "#a8d4ec", "IRb": "#a8d4ec"}[region]
        _draw_arc_block(ax, a, b, 0.72, 0.77, length, color, lw=0.3, edge="0.35")
        theta = math.radians(_bp_to_deg((a + b) / 2, length))
        ax.text(
            0.745 * math.cos(theta),
            0.745 * math.sin(theta),
            region,
            fontsize=_FONT_TRACK,
            ha="center",
            va="center",
        )
    for i, track in enumerate(tracks):
        radius = 0.68 - i * 0.045
        color = ("#bd4b77", "#318b74", "#6d61ad")[i]
        for row in track["intervals"]:
            _draw_arc_block(
                ax, row["start"] - 1, row["end"], radius - 0.025, radius, length, color, lw=0
            )
        ax.text(
            -1.48,
            1.18 - i * 0.09,
            f"{track['name']}: {len(track['intervals'])} intervals",
            color=color,
            fontsize=_FONT_TRACK,
        )
    for row in features:
        if row["splicing"] != "trans":
            continue
        for left, right in pairwise(row["exons"]):
            points = []
            for exon in (left, right):
                theta = math.radians(_bp_to_deg((exon["start"] - 1 + exon["end"]) / 2, length))
                radius = 0.862 if exon["strand"] == 1 else 0.82
                points.append((radius * math.cos(theta), radius * math.sin(theta)))
            patch = PathPatch(
                MplPath(
                    [points[0], (0, 0), points[1]], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                ),
                facecolor="none",
                edgecolor="#555555",
                lw=0.6,
                linestyle="--",
                alpha=0.65,
            )
            patch.set_gid(f"splice-{row['feature_index']}-{left['exon']}-{right['exon']}")
            ax.add_patch(patch)


# Printed layout of one structure-map block (millimetres on a 183 mm double column).
_FONT_TRACK = 6.5
_FONT_ROW_GENE = 6.5
_FONT_ROW_SMALL = 5.0
_MM_CIRCLE_W = 96.0
_MM_GUTTER = 3.0
_MM_ROW_LABEL = 17.0
_MM_ROW_H = 6.6
_MM_HEADER = 4.0
_MM_NOTE = 7.0
_MM_BLOCK_GAP = 6.0
_EXON_NOTE = (
    "Coordinates 1-based inclusive. Cis: exons and introns to scale, intron length in bp.\n"
    "Trans: exons to scale; // marks an omitted genomic gap. Inferred: no /trans_splicing in\n"
    "the file; called from mixed exon strands, a backward exon order or a gap > 25 kb."
)
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
# Typographic minus, middle dot and en dash for printed labels.
_MINUS, _DOT, _DASH = "\u2212", "\u00b7", "\u2013"


def _spliced_rows(features):
    return [f for f in features if f["feature_type"] != "gene" and f["splicing"] != "none"]


def _circle_height_mm(legend: bool) -> float:
    from .ogdraw import _XLIM_LEGEND_BELOW, _YLIM, _YLIM_LEGEND_BELOW_BOTTOM, _YLIM_PANEL_BOTTOM

    bottom = _YLIM_LEGEND_BELOW_BOTTOM if legend else _YLIM_PANEL_BOTTOM
    span_x = _XLIM_LEGEND_BELOW[1] - _XLIM_LEGEND_BELOW[0]
    return _MM_CIRCLE_W * (_YLIM[1] - bottom) / span_x


def _block_height_mm(spliced_count: int, legend: bool) -> float:
    rows_h = spliced_count * _MM_ROW_H + (_MM_NOTE if spliced_count else 0.0)
    return _MM_HEADER + max(_circle_height_mm(legend), rows_h)


def _draw_structure_block(
    fig, top_mm, parsed, features, tracks, regions, header, letters, *, legend
):
    """Circle panel and exon-row panel of one record, laid out in printed millimetres.

    Both panel letters and both headers sit on one baseline (figure text, so a
    and b align exactly); every exon row has the same left edge, width and height.
    """
    fig_w = fig.get_figwidth() * 25.4
    fig_h = fig.get_figheight() * 25.4

    def rect(x, y_top, w, h):
        return (x / fig_w, (fig_h - y_top - h) / fig_h, w / fig_w, h / fig_h)

    body_top = top_mm + _MM_HEADER
    ax = fig.add_axes(rect(0.0, body_top, _MM_CIRCLE_W, _circle_height_mm(legend)))
    draw_mito_map(
        parsed,
        genome_name=header,
        ax=ax,
        draw_gc=False,
        draw_center_text=False,
        draw_legend=legend,
        min_label_gap=None,
        fold_minus_labels_out=True,
        legend_below=True,
        inner_tracks=lambda ax, length: _draw_structure_tracks(
            ax, length, features, tracks, regions
        ),
    )
    baseline = 1 - (body_top - 1.0) / fig_h  # 1 mm above the panel bodies
    letter_x = 1.0 / fig_w
    fig.text(letter_x, baseline, letters[0], fontsize=FONT_PT["panel"], fontweight="bold",
             ha="left", va="baseline")
    fig.text(letter_x + 4.0 / fig_w, baseline, header, fontsize=FONT_PT["body"],
             ha="left", va="baseline")
    spliced = _spliced_rows(features)
    if not spliced:
        return
    rows_x = _MM_CIRCLE_W + _MM_GUTTER
    exon_x = rows_x + _MM_ROW_LABEL
    exon_w = fig_w - exon_x - 1.0
    fig.text(rows_x / fig_w, baseline, letters[1], fontsize=FONT_PT["panel"],
             fontweight="bold", ha="left", va="baseline")
    fig.text((rows_x + 4.0) / fig_w, baseline, "Spliced genes: exon structure",
             fontsize=FONT_PT["body"], ha="left", va="baseline")
    for i, row in enumerate(spliced):
        row_ax = fig.add_axes(rect(exon_x, body_top + i * _MM_ROW_H, exon_w, _MM_ROW_H))
        _draw_exon_panel(row_ax, row)
    note_y = 1 - (body_top + len(spliced) * _MM_ROW_H + 1.5) / fig_h
    fig.text(rows_x / fig_w, note_y, _EXON_NOTE, fontsize=_FONT_ROW_SMALL, ha="left",
             va="top", color="0.25", linespacing=1.3)


def _label_exons(ax, row, exons, lengths, starts, total):
    """Coordinates over each exon, on alternating levels; exon numbers when they would collide.

    End exons anchor their label inward so it is not clipped at the panel edge.
    When any two labels on one level would touch (many small exons in a narrow
    row, e.g. nad7), each exon gets only its number and a cis row gets its
    overall span below; per-exon coordinates stay in ``<stem>.exons.tsv``.
    """
    fig = ax.figure
    width_pt = ax.get_position().width * fig.get_figwidth() * 72
    per_pt = total * 1.008 / width_pt

    def placed(texts):
        spans = []
        for i, (text, size) in enumerate(zip(texts, lengths, strict=True)):
            w = len(text) * _FONT_ROW_SMALL * 0.52 * per_pt
            first, last = i == 0, i == len(texts) - 1
            x = starts[i] if first else starts[i] + size if last else starts[i] + size / 2
            lo = x if first else x - w if last else x - w / 2
            spans.append((i % 2, lo, lo + w, x, "left" if first else "right" if last else "center"))
        return spans

    def collide(spans):
        pad = 2 * per_pt
        for level in (0, 1):
            row_spans = sorted((s for s in spans if s[0] == level), key=lambda s: s[1])
            if any(a[2] + pad > b[1] for a, b in pairwise(row_spans)):
                return True
        return any(s[1] < -0.004 * total or s[2] > total * 1.004 for s in spans)

    def strand(e):
        return "+" if e["strand"] == 1 else _MINUS

    texts = [f"{e['exon']} {_DOT} {e['start']:,}{_DASH}{e['end']:,} {strand(e)}" for e in exons]
    spans = placed(texts)
    compact = collide(spans)
    if compact:
        spans = [(0, 0, 0, starts[i] + lengths[i] / 2, "center") for i in range(len(exons))]
        texts = [str(e["exon"]) for e in exons]
    for (level, _lo, _hi, x, ha), text in zip(spans, texts, strict=True):
        ax.text(x, 0.36 if level == 0 else -0.36, text, ha=ha, va="center",
                fontsize=_FONT_ROW_SMALL, zorder=3)
    if compact and row["splicing"] == "cis":
        lo = min(e["start"] for e in exons)
        hi = max(e["end"] for e in exons)
        ax.text(0, -0.36, f"{lo:,}{_DASH}{hi:,} {strand(exons[0])}", ha="left", va="center",
                fontsize=_FONT_ROW_SMALL, zorder=3)


def _draw_exon_panel(ax, row):
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import ScaledTranslation

    exons = row["exons"]
    trans = row["splicing"] == "trans"
    lengths = [e["end"] - e["start"] + 1 for e in exons]
    # Trans panels omit genomic gaps explicitly; cis panels retain true intron lengths.
    gap_lengths = [sum(lengths) * 0.28] * (len(exons) - 1) if trans else row["gaps"]
    total = sum(lengths) + sum(gap_lengths)
    cursor = 0
    ax.plot([0, total], [0, 0], color=".5", lw=0.6, linestyle="--" if trans else "-", zorder=1)
    starts = []
    for i, (exon, size) in enumerate(zip(exons, lengths, strict=True)):
        starts.append(cursor)
        block = Rectangle(
            (cursor, -0.15),
            size,
            0.3,
            facecolor=GENE_COLORS[row["category"]],
            edgecolor="black",
            lw=0.4,
            zorder=2,
        )
        block.set_gid(f"exon-{row['feature_index']}-{exon['exon']}")
        ax.add_patch(block)
        if i < len(gap_lengths):
            ax.text(
                cursor + size + gap_lengths[i] / 2,
                0,
                "//" if trans else f"{gap_lengths[i]:,}",
                ha="center",
                va="center",
                fontsize=_FONT_ROW_SMALL,
                # A tight white box breaks the intron line under the number
                # without covering the coordinate labels above and below.
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
                zorder=2.5,
            )
            cursor += size + gap_lengths[i]
    _label_exons(ax, row, exons, lengths, starts, total)
    ax.set_xlim(-0.004 * total, total * 1.004)  # keep the end exons' outlines inside
    ax.set_ylim(-0.62, 0.62)
    basis = row.get("trans_basis")
    # Annotated trans-splicing is just "trans"; an inferred call says how it was inferred.
    inferred = trans and basis not in (None, "qualifier")
    splicing = f"trans ({TRANS_BASIS_TEXT[basis]})" if inferred else row["splicing"]
    left = ax.transAxes + ScaledTranslation(-3 / 72, 0, ax.figure.dpi_scale_trans)
    ax.text(0, 0.68, row["gene"], transform=left, ha="right", va="center",
            fontsize=_FONT_ROW_GENE, fontstyle="italic")
    ax.text(0, 0.3, "trans (inferred)" if inferred else row["splicing"], transform=left,
            ha="right", va="center", fontsize=_FONT_ROW_SMALL, color="0.35")
    # The full call (with how trans-splicing was inferred) stays on the axes.
    ax.set_label(f"{row['gene']} \u00b7 feature {row['feature_index']} \u00b7 {splicing}")
    ax.axis("off")


def _render_contig_map_bundle(path, contigs, name, dpi):
    """One combined figure plus one standalone figure per contig.

    The writer collects every file the renderer leaves beside ``path``, so a
    ``map.png`` request also materializes ``map.<contig>.png`` per contig, each
    laid out exactly like a single-record structure map.
    """
    combined = _render_multi_contig_structure_map(path, contigs, name, dpi)
    stem = Path(path)
    for contig in contigs:
        _render_structure_map(
            stem.with_name(f"{stem.stem}.{contig['name']}{stem.suffix}"),
            contig["parsed"],
            contig["features"],
            [],
            {},
            f"{name} \u00b7 {contig['name']}",
            dpi,
        )
    return combined


def _structure_figure(heights_mm):
    import matplotlib.pyplot as plt

    height = sum(heights_mm) + _MM_BLOCK_GAP * (len(heights_mm) - 1)
    return plt.figure(figsize=(width_inches("double"), height / 25.4))


def _render_multi_contig_structure_map(path, contigs, name, dpi):
    import csv

    import matplotlib.pyplot as plt

    rows: list[dict] = []
    heights = [
        _block_height_mm(len(_spliced_rows(c["features"])), legend=i == len(contigs) - 1)
        for i, c in enumerate(contigs)
    ]
    with publication_style():
        fig = _structure_figure(heights)
        try:
            top = 0.0
            for index, contig in enumerate(contigs):
                rows.extend(contig["features"])
                header = (
                    f"{name} \u00b7 {contig['name']} \u00b7 {contig['length']:,} bp "
                    f"\u00b7 {contig['organelle']}"
                )
                first = 2 * index % len(_LETTERS)
                _draw_structure_block(
                    fig, top, contig["parsed"], contig["features"], [], {}, header,
                    _LETTERS[first:first + 2],
                    # One shared category legend, under the last circle.
                    legend=index == len(contigs) - 1,
                )
                top += heights[index] + _MM_BLOCK_GAP
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        finally:
            plt.close(fig)
    fields = [
        "contig",
        "feature_index",
        "gene",
        "feature_type",
        "splicing",
        "exon",
        "start",
        "end",
        "strand",
    ]
    with path.with_suffix(".exons.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            for exon in row["exons"]:
                writer.writerow(
                    {**{k: row[k] for k in fields[:4]}, **exon}
                )
    return path


def _render_structure_map(path, parsed, features, tracks, regions, name, dpi):
    import csv

    import matplotlib.pyplot as plt

    header = f"{name} \u00b7 {parsed.genome_length:,} bp \u00b7 {parsed.organelle}"
    with publication_style():
        fig = _structure_figure([_block_height_mm(len(_spliced_rows(features)), legend=True)])
        try:
            _draw_structure_block(
                fig, 0.0, parsed, features, tracks, regions, header, "ab", legend=True
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        finally:
            plt.close(fig)
    with path.with_suffix(".exons.tsv").open("w", newline="") as stream:
        fields = [
            "feature_index",
            "gene",
            "feature_type",
            "splicing",
            "exon",
            "start",
            "end",
            "strand",
        ]
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        for row in features:
            for exon in row["exons"]:
                writer.writerow({**{k: row[k] for k in fields[:4]}, **exon})
    return path
