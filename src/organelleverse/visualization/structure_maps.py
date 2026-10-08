"""Annotation-backed structure panels and tracks on the existing OGDraw map."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

from ..annotation.splice_layout import parts_are_trans_spliced
from ..core.errors import OrganelleInputError, OrganelleParameterError
from .ogdraw import (
    GENE_COLORS,
    Gene,
    MitoGenome,
    _bp_to_deg,
    _draw_arc_block,
    _load_genbank_records,
    _parse_genbank_record,
    classify_gene,
    draw_mito_map,
)
from .plot_object import OrganellePlot, plot_result


def _parents(feature, genes):
    """Match the annotated locus, including duplicated genes and mixed strands."""
    name = feature.qualifiers.get("gene")
    tag = feature.qualifiers.get("locus_tag")
    return [
        gene
        for gene in genes
        if (tag == gene.qualifiers.get("locus_tag") if tag else name == gene.qualifiers.get("gene"))
        and all(
            any(
                p.start >= q.start and p.end <= q.end and p.strand == q.strand
                for q in gene.location.parts
            )
            for p in feature.location.parts
        )
    ]


def _annotations(record, organelle="mito"):
    genes = [f for f in record.features if f.type == "gene"]
    products = [f for f in record.features if f.type in {"CDS", "tRNA", "rRNA"}]
    covered = {id(g) for f in products for g in _parents(f, genes)}
    rows = []
    for index, feature in enumerate(record.features):
        if feature.type not in {"CDS", "tRNA", "rRNA"} and not (
            feature.type == "gene" and id(feature) not in covered
        ):
            continue
        name = feature.qualifiers.get("gene", feature.qualifiers.get("locus_tag", [feature.type]))[
            0
        ]
        parts = list(feature.location.parts)
        if any(p.ref is not None or p.strand not in (-1, 1) for p in parts):
            raise OrganelleInputError(
                code="visualization.structure.location",
                message="Structure maps require local, stranded feature locations.",
            )
        trans = any(
            "trans_splicing" in f.qualifiers for f in [feature, *_parents(feature, genes)]
        ) or (
            feature.type != "gene"
            and parts_are_trans_spliced(
                [(int(p.start), int(p.end), p.strand) for p in parts], len(record)
            )
        )
        # GenBank join order is biological order, including complement and trans-splicing.
        exons = [
            {"exon": i, "start": int(p.start) + 1, "end": int(p.end), "strand": p.strand}
            for i, p in enumerate(parts, 1)
        ]
        gaps = [
            ((b["start"] - a["end"] - 1) if a["strand"] == 1 else (a["start"] - b["end"] - 1))
            % len(record)
            for a, b in pairwise(exons)
        ]
        # A contiguous join across the circular origin is not a splice junction.
        splicing = "trans" if trans else "cis" if any(gaps) else "none"
        if feature.type == "gene":
            splicing = "none"
        rows.append(
            {
                "feature_index": index,
                "gene": name,
                "feature_type": feature.type,
                "category": classify_gene(name, organelle),
                "splicing": splicing,
                "exons": exons,
                "gaps": gaps if splicing == "cis" else [],
            }
        )
    return rows


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
    exons. Trans-splicing comes from the feature or its parent gene qualifier,
    never from a distance threshold. Every copy is retained. Dashed links
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
    regions = {}
    if show_ir is None and parsed.organelle == "plastid":
        from ..comparative.orientation import _quadripartite

        regions = _quadripartite(str(record.seq).upper()) or {}
    elif show_ir:
        from ..comparative.orientation import _quadripartite

        if parsed.organelle != "plastid":
            raise OrganelleParameterError(
                code="visualization.structure.ir_organelle",
                message="IR partitioning requires a plastid record.",
            )
        regions = _quadripartite(str(record.seq).upper())
        if regions is None:
            raise OrganelleInputError(
                code="visualization.structure.ir_unresolved",
                message="No quadripartite plastome structure detected.",
            )
    tracks = _tracks(ssrs, tandem_repeats, dispersed_repeats, len(record))
    name = genome_name or record.id

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


def _structure_genes(features):
    return [
        Gene(
            name=row["gene"],
            start=min(e["start"] for e in row["exons"]),
            end=max(e["end"] for e in row["exons"]),
            strand=row["exons"][0]["strand"],
            has_intron=row["splicing"] != "none",
            category=row["category"],
            exons=[(e["start"], e["end"]) for e in row["exons"]],
            exon_strands=[e["strand"] for e in row["exons"]],
            trans_spliced=row["splicing"] == "trans",
        )
        for row in features
    ]


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
        _draw_arc_block(ax, a, b, 0.72, 0.77, length, color)
        theta = math.radians(_bp_to_deg((a + b) / 2, length))
        ax.text(
            0.745 * math.cos(theta),
            0.745 * math.sin(theta),
            region,
            fontsize=8,
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
            -1.65,
            0.65 - i * 0.09,
            f"{track['name']}: {len(track['intervals'])} intervals",
            color=color,
            fontsize=9,
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
                lw=0.8,
                linestyle="--",
                alpha=0.65,
            )
            patch.set_gid(f"splice-{row['feature_index']}-{left['exon']}-{right['exon']}")
            ax.add_patch(patch)


def _draw_exon_panel(ax, row):
    from matplotlib.patches import Rectangle

    exons = row["exons"]
    trans = row["splicing"] == "trans"
    lengths = [e["end"] - e["start"] + 1 for e in exons]
    # Trans panels omit genomic gaps explicitly; cis panels retain true intron lengths.
    gap_lengths = [sum(lengths) * 0.28] * (len(exons) - 1) if trans else row["gaps"]
    total = sum(lengths) + sum(gap_lengths)
    cursor = 0
    ax.plot([0, total], [0, 0], color=".5", lw=0.8, linestyle="--" if trans else "-")
    for i, (exon, size) in enumerate(zip(exons, lengths, strict=True)):
        block = Rectangle(
            (cursor, -0.15),
            size,
            0.3,
            facecolor=GENE_COLORS[row["category"]],
            edgecolor="black",
            lw=0.5,
        )
        block.set_gid(f"exon-{row['feature_index']}-{exon['exon']}")
        ax.add_patch(block)
        # Coordinate labels sit on alternating levels so small neighboring exons remain readable.
        label = f"{exon['exon']}: {exon['start']:,}-{exon['end']:,} ({'+' if exon['strand'] == 1 else '-'})"
        ax.text(
            cursor + size / 2,
            0.28 if i % 2 == 0 else -0.38,
            label,
            ha="center",
            va="center",
            fontsize=6.5,
        )
        if i < len(gap_lengths):
            ax.text(
                cursor + size + gap_lengths[i] / 2,
                0,
                "//" if trans else f"{gap_lengths[i]:,}",
                ha="center",
                fontsize=6,
                backgroundcolor="white",
            )
            cursor += size + gap_lengths[i]
    ax.set_xlim(-total * 0.09, total * 1.09)
    ax.set_ylim(-0.65, 0.7)
    ax.set_title(
        f"{row['gene']} · feature {row['feature_index']} · {row['splicing']}",
        loc="left",
        fontsize=8,
        fontstyle="italic",
        pad=1,
    )
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
            f"{name} · {contig['name']}",
            dpi,
        )
    return combined


def _render_multi_contig_structure_map(path, contigs, name, dpi):
    import csv

    import matplotlib.pyplot as plt

    rows: list[dict] = []
    heights: list[float] = []
    for contig in contigs:
        spliced = [
            row
            for row in contig["features"]
            if row["feature_type"] != "gene" and row["splicing"] != "none"
        ]
        heights.append(max(4.0, len(spliced) * 0.62 + 1.2))
        rows.extend(contig["features"])
    fig = plt.figure(figsize=(23, max(8.0, sum(heights) * 0.62 + 1.5)))
    try:
        grid = fig.add_gridspec(
            len(contigs), 2, width_ratios=[1.1, 1], height_ratios=heights, wspace=0.15, hspace=0.3
        )
        for index, contig in enumerate(contigs):
            parsed = contig["parsed"]
            ax = fig.add_subplot(grid[index, 0])
            draw_mito_map(
                parsed,
                genome_name=contig["name"],
                ax=ax,
                draw_gc=False,
                draw_center_text=False,
                # The category legend is shared by every contig; drawing it per
                # row would stack three copies over the lower circles.
                draw_legend=index == 0,
                min_label_gap=parsed.genome_length / 150,
                fold_minus_labels_out=True,
            )
            ax.set_title(
                f"{contig['name']}\n{contig['length']:,} bp · {contig['organelle']}", fontsize=13
            )
            spliced = [
                row
                for row in contig["features"]
                if row["feature_type"] != "gene" and row["splicing"] != "none"
            ]
            if spliced:
                panels = grid[index, 1].subgridspec(len(spliced), 1, hspace=0.55)
                for panel_index, row in enumerate(spliced):
                    _draw_exon_panel(fig.add_subplot(panels[panel_index]), row)
        fig.suptitle(
            f"{name} · {len(contigs)} contigs · coordinates are 1-based inclusive within each contig\n"
            "Cis: exon/intron lengths to scale within each row. Trans: exon lengths to scale; // genomic gaps omitted.",
            fontsize=11,
            y=0.985,
        )
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
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

    spliced = [f for f in features if f["feature_type"] != "gene" and f["splicing"] != "none"]
    fig = plt.figure(figsize=(23, max(12, len(spliced) * 0.62 + 1.5)))
    try:
        grid = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.15)
        ax = fig.add_subplot(grid[0, 0])
        draw_mito_map(
            parsed,
            genome_name=name,
            ax=ax,
            draw_gc=False,
            draw_center_text=False,
            min_label_gap=parsed.genome_length / 150,
            fold_minus_labels_out=True,
            inner_tracks=lambda ax, length: _draw_structure_tracks(
                ax, length, features, tracks, regions
            ),
        )
        ax.set_title(f"{name}\n{parsed.genome_length:,} bp · {parsed.organelle}", fontsize=14)
        if spliced:
            panels = grid[0, 1].subgridspec(len(spliced), 1, hspace=0.55)
            for i, row in enumerate(spliced):
                _draw_exon_panel(fig.add_subplot(panels[i]), row)
        fig.suptitle(
            "Exon structure · coordinates are 1-based inclusive\nCis: exon/intron lengths to scale within each row. Trans: exon lengths to scale; // genomic gaps omitted.",
            fontsize=11,
            y=0.98,
        )
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
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
