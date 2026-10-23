"""Circos-style transfer chord diagram: every sequence on one ring, ribbons for transfers.

The recipient genome (mitochondrial chromosomes or contigs, or nuclear
chromosomes) and the donor genome (the plastome, or the mitogenome for NUMTs)
share one ring. Each transferred fragment is a ribbon whose two feet span the
fragment on the recipient and its homologous region on the donor, so ribbon
width is fragment length; ribbons are coloured continuously by identity (or
length) with transparency. Segments carry kb ticks, the plastome its
LSC/IRb/SSC/IRa band, an optional gene ring outside, and the names of genes
carried by the transfers.

Inputs are OrganelleVerse's own transfer results:

* ``transfer.detect_mtpt`` / ``compute_transfer`` fragments: each merged
  fragment lists its HSPs with donor coordinates (``hsps``); one ribbon per HSP,
  so a copy of a plastid inverted repeat links to both IR arms.
* ``transfer.detect_transfers_evidence`` candidates (NUMT/NUPT): one ribbon per
  candidate from ``nuclear_*`` to ``organelle_*``.
* flat link rows with ``recipient_seqid/start/end`` and ``donor_seqid/start/end``.

Compute-only: :func:`plot_transfer_circos` resolves the layout and returns an
:class:`OrganellePlot` whose metrics hold everything the renderer draws;
``ov.write(plot, "fig.svg")`` renders PNG/SVG/PDF and a ``<stem>.links.tsv``.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from organelleverse._public_validation import validated_public

from ..core.errors import OrganelleInputError, OrganelleParameterError
from ..core.frozen import thaw_json
from ..core.result import OrganelleResult
from .plot_object import OrganellePlot, plot_result
from .style import FONT_PT, MAX_HEIGHT_MM, publication_style, width_inches

OP = "plot_transfer_circos"

_GENBANK_SUFFIXES = {".gb", ".gbk", ".gbff", ".genbank", ".gbf"}

#: Recipient / donor kinds implied by each transfer class.
_KINDS = {
    "MTPT": ("mitochondrion", "plastid"),
    "NUPT": ("nuclear", "plastid"),
    "NUMT": ("nuclear", "mitochondrion"),
}
_KIND_TEXT = {
    "MTPT": "Plastid-derived DNA in the mitochondrial genome (MTPT)",
    "NUPT": "Plastid-derived DNA in the nuclear genome (NUPT)",
    "NUMT": "Mitochondrion-derived DNA in the nuclear genome (NUMT)",
}
_SHORT = {"mitochondrion": "mtDNA", "plastid": "cpDNA", "nuclear": "nuclear"}

# Ring geometry in data units. The gene ring IS the OGDraw annotation map's ring
# (same radii, blocks, backbone and radial italic labels, see ``ogdraw``), so a
# chord diagram and ``plot_ogdraw_map`` drawn at the same circle size share their
# outer annotation and type sizes. Outer → inner: segment name, gene labels,
# plus-strand blocks, backbone, minus-strand blocks, sequence band (plastid
# LSC/IRb/SSC/IRa), kb ticks and tangential tick numbers, ribbon feet, ribbons.
#: Segment names sit just outside the gene ring (or outside the names in their
#: window), never beyond _R_NAME_MAX; with no room there they go inside the ring.
_R_NAME_MIN = 0.965
_R_NAME_MAX = 1.33
_R_NAME_INSIDE = 0.63
_R_BAND_OUT = 0.80
_R_BAND_IN = 0.752
_R_TICK_MAJOR = 0.736
_R_TICK_MINOR = 0.743
_R_TICK_LABEL = 0.718
#: Foot track: each transferred region as a solid block inside the tick numbers.
_R_FOOT_OUT = 0.692
_R_FOOT_IN = 0.672
_R_LINK = 0.672
#: The band never recedes further in than this, so the ribbons keep their room.
_BAND_OUT_MIN = 0.46
_LIM = 1.36

#: Gap between neighbouring segments of one genome, and between the two genomes (degrees).
_GAP_DEG = 1.4
_GROUP_GAP_DEG = 5.0
#: A ribbon foot is never drawn narrower than this (degrees) so short links stay visible.
_MIN_FOOT_DEG = 0.5
#: A group whose total length differs from the other's by more than this factor
#: gets half the ring instead of a proportional share (nuclear vs organelle).
_PROPORTIONAL_MAX_RATIO = 8.0
#: Layout search: exhaustive over recipient orders up to this many segments.
_MAX_PERMUTED_SEGMENTS = 5
#: Crossings are counted on at most this many (longest) links.
_MAX_CROSSING_LINKS = 800
#: Printed layout (``visualization.style``): text is in printed points and never
#: scales with the canvas. A figure at least this wide puts its legend in a
#: column right of the ring; a narrower one puts a short legend below it.
_SIDE_LEGEND_MIN_MM = 150.0
#: Wide figures: a right column (zoom inset, colour bar, sequence key) no wider than
#: a quarter of the page, and the gene-class key in columns under the ring.
_RIGHT_COLUMN_MM = 45.0
_COLUMN_GUTTER_MM = 3.0
_RING_MAX_MM = 135.0
_GENE_KEY_COLUMNS_WIDE = 4
_GENE_KEY_COLUMNS_NARROW = 2
#: Narrow figures: zoom inset and keys in a band under the ring.
_ZOOM_BAND_MM = 44.0
_LEGEND_ROW_MM = 3.35  # one 7 pt legend row
#: Gene names: OGDraw's 6.5 pt; a crowded segment first shrinks them to 6 pt,
#: then stops naming tRNAs, then names only transferred genes (>= 5 pt), and
#: says so on the figure.
_FONT_LABEL = 6.5
_FONT_LABEL_CROWDED = 6.5  # names never shrink: crowding means fewer names, not smaller ones
_FONT_TICK = FONT_PT["tick"]
_FONT_REGION = FONT_PT["tick"]
_FONT_SEGMENT = FONT_PT["body"]
_FONT_SEGMENT_SUB = FONT_PT["tick"]
_FONT_LEGEND = FONT_PT["body"]
_LABEL_PLUS = "#111111"
_LABEL_MINUS = "#4B5563"

_KIND_COLORS = {"mitochondrion": "#7C8796", "plastid": "#5E9E6E", "nuclear": "#B49B76"}
_REGION_COLORS = {"LSC": "#8FC48A", "SSC": "#EDC173", "IRa": "#79AEDB", "IRb": "#79AEDB"}
#: Continuous ribbon palette: deep indigo (low identity / short) → magenta → amber.
_CMAP = "plasma"
_CMAP_RANGE = (0.04, 0.86)
_RIBBON_ALPHA = 0.6


# ── public entry point ───────────────────────────────────────────────────────


@validated_public('visualization.plot_transfer_circos')
def plot_transfer_circos(
    transfers: Sequence[
        Mapping[
            str,
            str | int | float | bool | None | Sequence[Mapping[str, str | int | float | bool | None]],
        ]
    ],
    recipient: Sequence[str | Path],
    donor: Sequence[str | Path],
    *,
    transfer_kind: str | None = None,
    color_by: str = "identity",
    min_length: int = 0,
    show_genes: bool = True,
    gene_labels: str = "auto",
    minus_labels: str = "inside",
    gene_label_min_fraction: float = 0.5,
    show_ir: bool | None = None,
    order: str = "auto",
    zoom: str | None = "auto",
    title: str | None = None,
    width: str | float = "double",
    dpi: int = 600,
) -> OrganellePlot:
    """Prepare a circos-style chord diagram of organelle DNA transfers.

    ``transfers`` is a transfer result (``detect_mtpt``, ``compute_transfer``,
    ``detect_transfers_evidence``; an ``OrganelleResult`` or its metrics), its
    ``fragments``/``candidates`` rows, or flat link rows (``recipient_seqid``,
    ``recipient_start``, ``recipient_end``, ``donor_seqid``, ``donor_start``,
    ``donor_end``, optional ``identity``/``strand``). Coordinates are 1-based
    inclusive. ``recipient`` and ``donor`` are GenBank or FASTA files (one or a
    list); every record is one ring segment, its length taken from the file.
    GenBank records also supply the gene ring, drawn exactly like the OGDraw
    annotation map (``plot_ogdraw_map``: same block radii, backbone, italic radial
    labels and type sizes at the same circle size), and a plastid record its
    LSC/IRb/SSC/IRa band (``show_ir``: ``None`` draws it when found, ``True``
    requires it, ``False`` never draws it).

    ``gene_labels``: ``"auto"`` names every gene of a segment at 6.5 pt (6 pt when
    crowded); if that does not fit it drops tRNA names, then names only the genes
    carried by a transfer, and the figure says so. ``"all"``, ``"transferred"`` or
    ``"none"`` force one choice. ``minus_labels="inside"`` (default) is the OGDraw
    layout: minus-strand names read inward inside the ring and the band, ticks
    and ribbons recede to make room; ``"outside"`` puts every name outside, the
    minus strand in grey. Transferred genes are set in bold;
    ``metrics["gene_label_modes"]`` records what each segment got on the
    standalone figure (a composed panel re-decides for its own size).

    The figure is drawn at its printed size in the shared journal style
    (``visualization.style``): ``width`` is ``"double"`` (183 mm, default),
    ``"onehalf"`` (120 mm), ``"single"`` (89 mm) or millimetres; height stays
    within 170 mm; text is in printed points (7 pt body, 6 pt ticks, 6.5 pt gene
    names, nothing below 5 pt). There is no in-figure title: ``title`` (default:
    the transfer class) and ``metrics["caption"]`` are for the figure legend.

    ``zoom`` adds an inset that shows one ribbon at true scale: both feet on
    linear tracks with kb coordinates, a scale bar and the genes around them, so
    a short fragment that is a hairline on the ring can be read. ``"auto"``
    (default) zooms the shortest ribbon, a link id (``"L4"``) picks one, ``None``
    leaves the inset out; ``metrics["zoom"]`` holds what the inset draws.
    The plot also draws as one panel of ``visualization.compose_figure``.

    ``transfer_kind`` (``"MTPT"``, ``"NUPT"``, ``"NUMT"``) is inferred from the
    result when omitted. ``color_by`` is ``"identity"`` or ``"length"``
    (log-scaled); ``min_length`` drops links shorter than that on the recipient.
    A gene counts as transferred when at least ``gene_label_min_fraction`` of its
    exon bases lie inside a ribbon foot. ``order="auto"`` picks the recipient segment
    order and the donor direction with the fewest ribbon crossings (the donor
    may then run counter-clockwise, which its label says); ``"input"`` keeps
    file order and clockwise coordinates.

    When the two genomes differ in total length by more than 8-fold (nuclear vs
    organelle) each takes half the ring and the scales differ; ``metrics["scale"]``
    says which. ``metrics["links"]`` is every drawn ribbon with its coordinates,
    identity, strand and the genes it covers; ``metrics["unlinked"]`` lists
    fragments that carry no donor coordinates (k-mer fallback results).
    """
    if color_by not in {"identity", "length"}:
        raise OrganelleParameterError(
            code="visualization.circos.color_by",
            message="color_by must be 'identity' or 'length'.",
        )
    if order not in {"auto", "input"}:
        raise OrganelleParameterError(
            code="visualization.circos.order", message="order must be 'auto' or 'input'."
        )
    if gene_labels not in {"auto", "all", "transferred", "none"}:
        raise OrganelleParameterError(
            code="visualization.circos.gene_labels",
            message="gene_labels must be 'auto', 'all', 'transferred' or 'none'.",
        )
    if minus_labels not in {"inside", "outside"}:
        raise OrganelleParameterError(
            code="visualization.circos.minus_labels",
            message="minus_labels must be 'inside' or 'outside'.",
        )
    if not 0.0 < float(gene_label_min_fraction) <= 1.0:
        raise OrganelleParameterError(
            code="visualization.circos.label_fraction",
            message="gene_label_min_fraction must be in (0, 1].",
        )
    try:
        width_mm = width_inches(width) * 25.4
    except ValueError as exc:
        raise OrganelleParameterError(code="visualization.circos.width", message=str(exc)) from exc

    links, unlinked, inferred_kind, source = _transfer_links(transfers)
    if transfer_kind is None and inferred_kind is None:
        # Candidate rows name the organelle only by accession: ask the donor file.
        donor_kind = _donor_record_kind(donor)
        if source == "candidates":
            inferred_kind = "NUPT" if donor_kind == "plastid" else "NUMT"
        elif donor_kind == "mitochondrion":
            inferred_kind = "NUMT"
    kind = (transfer_kind or inferred_kind or "MTPT").upper()
    if kind not in _KINDS:
        raise OrganelleParameterError(
            code="visualization.circos.kind",
            message="transfer_kind must be MTPT, NUPT or NUMT.",
        )
    recipient_kind, donor_kind = _KINDS[kind]
    recipients = _load_segments(recipient, "recipient", recipient_kind, show_ir, show_genes)
    donors = _load_segments(donor, "donor", donor_kind, show_ir, show_genes)
    _attach_links(links, recipients, donors)
    _attach_unlinked(unlinked, recipients)

    dropped = [link for link in links if link["length"] < min_length]
    links = [link for link in links if link["length"] >= min_length]
    if not links:
        raise OrganelleInputError(
            code="visualization.circos.no_links",
            message=(
                "No transfer has donor coordinates to draw"
                + (f" at >= {min_length} bp" if dropped else "")
                + "; k-mer fallback fragments carry none, rerun with LOSAT or blastn."
            ),
        )
    recipients = _keep_segments(recipients, links, "recipient")
    donors = _keep_segments(donors, links, "donor")
    for index, link in enumerate(sorted(links, key=_link_sort_key), 1):
        link["id"] = f"L{index}"
    links.sort(key=lambda link: int(link["id"][1:]))
    _annotate_link_genes(links, recipients + donors)

    layout = _choose_layout(recipients, donors, links, order)
    candidates = (
        _gene_label_candidates(layout["segments"], links, gene_label_min_fraction)
        if gene_labels != "none"
        else []
    )
    zoom_spec = _zoom_spec(links, layout["segments"], zoom)
    ring_mm = _ring_mm(width_mm)
    labels, label_modes, label_notes, minus_inside = _fit_labels(
        layout["segments"], candidates, gene_labels, ring_mm / 25.4 / (2 * _LIM),
        minus_labels == "inside",
    )
    values = [link["identity"] if color_by == "identity" else link["length"] for link in links]
    known = [v for v in values if v is not None]
    if color_by == "identity":
        low = min(known) if known else 80.0
        color_range = [math.floor(min(low, 100.0) / 5) * 5, 100.0]
    else:
        color_range = [float(min(known)), float(max(known))]
    species = next((s["species"] for s in recipients + donors if s.get("species")), "")
    linked_bp = sum(link["length"] for link in links)
    resolved_title = title if title is not None else _KIND_TEXT[kind]
    categories = list(dict.fromkeys(g["category"] for seg in layout["segments"] for g in seg["genes"]))
    metrics = {
        "transfer_kind": kind,
        "title": resolved_title,
        "caption": _caption(
            resolved_title, species, kind, len(links), linked_bp, color_by, label_notes, zoom_spec,
            any(row.get("transferred") for row in labels),
        ),
        "panel_header": " · ".join(
            x for x in (species, kind, f"{len(links)} ribbons") if x
        ),
        "species": species,
        "color_by": color_by,
        "color_range": color_range,
        "cmap": _CMAP,
        "min_length": int(min_length),
        "scale": layout["scale"],
        "order": order,
        "crossings": layout["crossings"],
        "crossings_input_order": layout["crossings_input_order"],
        "segments": [_public_segment(s) for s in layout["segments"]],
        "links": links,
        "link_count": len(links),
        "linked_recipient_bp": linked_bp,
        "dropped_below_min_length": len(dropped),
        "unlinked": unlinked,
        "gene_label_request": gene_labels,
        "minus_labels": minus_labels,
        "minus_labels_drawn": "inside" if minus_inside else "outside",
        "gene_label_candidates": candidates,
        "gene_labels": labels,
        "gene_label_modes": label_modes,
        "gene_label_notes": label_notes,
        "width_mm": round(width_mm, 2),
        "ring_mm": round(ring_mm, 2),
        "height_mm": round(_height_mm(width_mm, ring_mm, len(categories)), 2),
        "gene_key": _gene_key_fits(width_mm, ring_mm, len(categories)),
        "zoom": zoom_spec,
        "dpi": int(dpi),
        "coordinate_system": "1-based inclusive; regions: 0-based start and length",
    }
    return _plot(metrics)


def transfer_links(
    transfers: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Normalize a transfer result into ``(links, unlinked, transfer_kind)``.

    ``links`` are flat rows with recipient and donor coordinates (1-based
    inclusive), ``identity`` (percent or ``None``) and ``strand``; ``unlinked``
    are fragments without donor coordinates; ``transfer_kind`` is ``"MTPT"``,
    ``"NUPT"``, ``"NUMT"`` or ``None`` when the input does not say.
    """
    links, unlinked, kind, _source = _transfer_links(transfers)
    return links, unlinked, kind


def _transfer_links(
    transfers: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, str]:
    """:func:`transfer_links` plus the row family (fragments, candidates, links)."""
    kind: str | None = None
    source = "links"
    rows: Any = transfers
    if isinstance(transfers, OrganelleResult):
        metrics = cast(Mapping[str, Any], thaw_json(transfers.metrics))
        if transfers.operation_id.endswith(("detect_mtpt", "compute_transfer")):
            kind = "MTPT"
        elif "candidates" in metrics:
            kind = _organelle_kind(str(metrics.get("organelle") or transfers.scope))
        rows = metrics
    if isinstance(rows, Mapping):
        observed = rows.get("observed_metrics")
        if isinstance(observed, Mapping):
            rows = observed
        if "fragments" in rows:
            kind = kind or "MTPT"
            rows = rows["fragments"]
        elif "candidates" in rows:
            kind = kind or _organelle_kind(str(rows.get("organelle", "")))
            rows = rows["candidates"]
        elif "links" in rows:
            rows = rows["links"]
        else:
            raise OrganelleInputError(
                code="visualization.circos.transfers",
                message="transfers must hold fragments, candidates or links.",
            )
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise OrganelleInputError(
            code="visualization.circos.transfers",
            message="transfers must be a transfer result or a list of rows.",
        )
    links: list[dict[str, Any]] = []
    unlinked: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise OrganelleInputError(
                code="visualization.circos.transfers",
                message=f"transfer row {index} is not a mapping.",
            )
        row = dict(raw)
        if "nuclear_seqid" in row and "organelle_seqid" in row:
            kind = kind or _organelle_kind(str(row.get("organelle_seqid", "")))
            source = "candidates"
            links.append(
                _link(
                    row["nuclear_seqid"], row["nuclear_start"], row["nuclear_end"],
                    row["organelle_seqid"], row["organelle_start"], row["organelle_end"],
                    row.get("identity"), row.get("strand"), index,
                )
            )
        elif "recipient_seqid" in row:
            links.append(
                _link(
                    row["recipient_seqid"], row["recipient_start"], row["recipient_end"],
                    row["donor_seqid"], row["donor_start"], row["donor_end"],
                    row.get("identity"), row.get("strand"), index,
                )
            )
        elif row.get("hsps"):
            kind = kind or "MTPT"
            source = "fragments"
            for hsp in row["hsps"]:
                links.append(
                    _link(
                        row.get("seqid", ""), hsp["start"], hsp["end"],
                        hsp.get("source_seqid", ""), hsp["source_start"], hsp["source_end"],
                        hsp.get("identity"), hsp.get("strand"), index,
                    )
                )
        elif "start" in row and "end" in row:
            source = "fragments"
            unlinked.append(
                {
                    "fragment": index,
                    "recipient_seqid": str(row.get("seqid", "")),
                    "recipient_start": int(row["start"]),
                    "recipient_end": int(row["end"]),
                    "length": int(row["end"]) - int(row["start"]) + 1,
                }
            )
        else:
            raise OrganelleInputError(
                code="visualization.circos.transfers",
                message=f"transfer row {index} has no recognizable coordinates.",
            )
    return links, unlinked, kind, source


def _link(rseq, rstart, rend, dseq, dstart, dend, identity, strand, fragment) -> dict[str, Any]:
    r0, r1 = sorted((int(rstart), int(rend)))
    d0, d1 = sorted((int(dstart), int(dend)))
    return {
        "recipient_seqid": str(rseq),
        "recipient_start": r0,
        "recipient_end": r1,
        "donor_seqid": str(dseq),
        "donor_start": d0,
        "donor_end": d1,
        "length": r1 - r0 + 1,
        "identity": None if identity in (None, "") else round(float(identity), 2),
        "strand": "-" if str(strand) == "-" else "+",
        "fragment": fragment,
    }


def _organelle_kind(text: str) -> str | None:
    lowered = text.lower()
    if any(token in lowered for token in ("plastid", "chloroplast", "cpdna", "plastome")):
        return "NUPT"
    if any(token in lowered for token in ("mitochondri", "mtdna", "mitogenome")):
        return "NUMT"
    return None


def _donor_record_kind(value: Any) -> str | None:
    """``plastid`` / ``mitochondrion`` from the first GenBank donor record that says, else ``None``."""
    from Bio import SeqIO  # type: ignore

    for path in _paths(value):
        if path.exists() and _is_genbank(path):
            for record in SeqIO.parse(str(path), "genbank"):
                kind = _record_kind(record, "")
                if kind:
                    return kind
    return None


def _link_sort_key(link: Mapping[str, Any]) -> tuple:
    return (link["recipient_seqid"], link["recipient_start"], link["donor_seqid"], link["donor_start"])


# ── sequences ────────────────────────────────────────────────────────────────


def _paths(value: Any) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    paths = [Path(v) for v in value]
    if not paths:
        raise OrganelleInputError(
            code="visualization.circos.sequences", message="No sequence files given."
        )
    return paths


def _is_genbank(path: Path) -> bool:
    if path.suffix.lower() in _GENBANK_SUFFIXES:
        return True
    with open(path, "rb") as handle:
        head = handle.read(512).lstrip()
    return head.startswith(b"LOCUS")


def _load_segments(
    value: Any, role: str, default_kind: str, show_ir: bool | None, show_genes: bool
) -> list[dict[str, Any]]:
    """One ring segment per record: name, length, kind, genes and plastid regions."""
    from .features import annotation_rows, record_display_name, resolve_ir_regions

    segments: list[dict[str, Any]] = []
    for path in _paths(value):
        if not path.exists():
            raise OrganelleInputError(
                code="visualization.circos.sequences", message=f"Sequence file not found: {path}"
            )
        if not _is_genbank(path):
            from .._losat import fasta_sequence_lengths

            for seqid, length in fasta_sequence_lengths(path).items():
                segments.append(_segment(seqid, [seqid], length, default_kind, role))
            continue
        from Bio import SeqIO  # type: ignore

        for record in SeqIO.parse(str(path), "genbank"):
            kind = _record_kind(record, default_kind)
            aliases = [record.id, record.name, record.id.split(".")[0]]
            segment = _segment(record.id, aliases, len(record), kind, role)
            segment["description"] = str(record.description or "")
            segment["species"] = record_display_name(record, "")
            if show_genes and kind != "nuclear":
                organelle = "plastid" if kind == "plastid" else "mito"
                segment["genes"] = [
                    {
                        "gene": row["gene"],
                        "category": row["category"],
                        "splicing": row["splicing"],
                        "exons": [[e["start"], e["end"], e["strand"]] for e in row["exons"]],
                    }
                    for row in annotation_rows(record, organelle, strict=False)
                ]
            if kind == "plastid" and show_ir is not False:
                segment["regions"] = resolve_ir_regions(record, "plastid", show_ir)
            elif show_ir and role == "donor" and kind != "plastid":
                raise OrganelleParameterError(
                    code="visualization.circos.ir_organelle",
                    message="show_ir=True needs a plastid record.",
                )
            segments.append(segment)
    if not segments:
        raise OrganelleInputError(
            code="visualization.circos.sequences", message=f"No {role} sequence records found."
        )
    _name_segments(segments)
    return segments


def _segment(seqid: str, aliases: list[str], length: int, kind: str, role: str) -> dict[str, Any]:
    if length <= 0:
        raise OrganelleInputError(
            code="visualization.circos.sequences", message=f"Sequence {seqid} is empty."
        )
    return {
        "seqid": seqid,
        "aliases": sorted({a for a in aliases if a}),
        "length": int(length),
        "kind": kind,
        "role": role,
        "genes": [],
        "regions": {},
        "species": "",
        "description": "",
    }


def _record_kind(record, default_kind: str) -> str:
    source = next((f for f in record.features if f.type == "source"), None)
    organelle = " ".join(source.qualifiers.get("organelle", [])) if source is not None else ""
    text = f"{organelle} {record.description}".lower()
    if "plastid" in text or "chloroplast" in text:
        return "plastid"
    if "mitochondri" in text:
        return "mitochondrion"
    return default_kind


_CHROMOSOME = re.compile(r"\b(?:chromosome|chr)\s*([A-Za-z0-9]+)", re.IGNORECASE)


def _name_segments(segments: list[dict[str, Any]]) -> None:
    """Short ring labels: ``mtDNA`` / ``cpDNA`` alone, ``mt chr 2`` for several."""
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        by_kind.setdefault(segment["kind"], []).append(segment)
    for kind, group in by_kind.items():
        short = _SHORT[kind]
        for index, segment in enumerate(group, 1):
            if kind == "nuclear":
                segment["label"] = segment["seqid"]
                continue
            if len(group) == 1:
                segment["label"] = short
                continue
            match = _CHROMOSOME.search(segment["description"])
            number = match.group(1) if match else str(index)
            segment["label"] = f"{short[:2]} chr {number}"


def _alias_index(segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for segment in segments:
        for alias in segment["aliases"]:
            index.setdefault(alias, segment)
    return index


def _resolve(seqid: str, segments: list[dict[str, Any]], role: str) -> dict[str, Any]:
    index = _alias_index(segments)
    for key in (seqid, seqid.split(".")[0], seqid.split()[0] if seqid.split() else seqid):
        if key in index:
            return index[key]
    if len(segments) == 1 and not seqid:
        return segments[0]
    raise OrganelleInputError(
        code="visualization.circos.seqid",
        message=f"{role} sequence {seqid!r} is not in the {role} files "
        f"({', '.join(s['seqid'] for s in segments)}).",
    )


def _attach_links(links, recipients, donors) -> None:
    for link in links:
        target = _resolve(link["recipient_seqid"], recipients, "recipient")
        source = _resolve(link["donor_seqid"], donors, "donor")
        link["recipient_seqid"], link["donor_seqid"] = target["seqid"], source["seqid"]
        for seg, lo, hi in ((target, "recipient_start", "recipient_end"), (source, "donor_start", "donor_end")):
            if link[hi] > seg["length"] or link[lo] < 1:
                raise OrganelleInputError(
                    code="visualization.circos.coordinates",
                    message=f"Link {link[lo]}..{link[hi]} lies outside {seg['seqid']} "
                    f"({seg['length']} bp).",
                )


def _attach_unlinked(unlinked, recipients) -> None:
    for row in unlinked:
        with contextlib.suppress(OrganelleInputError):
            row["recipient_seqid"] = _resolve(row["recipient_seqid"], recipients, "recipient")["seqid"]


def _keep_segments(segments, links, role: str) -> list[dict[str, Any]]:
    """Organelle records are all kept; nuclear scaffolds without links that are tiny are dropped."""
    used = {link[f"{role}_seqid"] for link in links}
    if not segments or segments[0]["kind"] != "nuclear":
        return segments
    longest = max(s["length"] for s in segments)
    return [s for s in segments if s["seqid"] in used or s["length"] >= 0.05 * longest]


# ── genes ────────────────────────────────────────────────────────────────────


def _gene_coverage(gene: Mapping[str, Any], lo: int, hi: int) -> tuple[int, int]:
    total = covered = 0
    for start, end, _strand in gene["exons"]:
        total += end - start + 1
        covered += max(0, min(end, hi) - max(start, lo) + 1)
    return covered, total


def _annotate_link_genes(links, segments) -> None:
    by_id = {s["seqid"]: s for s in segments}
    for link in links:
        for role in ("recipient", "donor"):
            segment = by_id[link[f"{role}_seqid"]]
            lo, hi = link[f"{role}_start"], link[f"{role}_end"]
            hits = []
            for gene in segment["genes"]:
                covered, total = _gene_coverage(gene, lo, hi)
                if covered:
                    hits.append({"gene": gene["gene"], "fraction": round(covered / total, 3)})
            link[f"{role}_genes"] = hits


def _bp_gene(gene: Mapping[str, Any]):
    """An ``ogdraw.Gene`` in base pairs, for OGDraw's label anchors."""
    from .ogdraw import Gene

    exons = gene["exons"]
    return Gene(
        name=gene["gene"],
        start=min(e[0] for e in exons),
        end=max(e[1] for e in exons),
        strand=exons[0][2],
        has_intron=gene.get("splicing", "none") != "none",
        exons=[(e[0], e[1]) for e in exons],
        category=gene["category"],
        trans_spliced=gene.get("splicing") == "trans",
        exon_strands=[e[2] for e in exons],
    )


def _label_points(gene: Mapping[str, Any]) -> list[tuple[float, str, int]]:
    """``(anchor_bp, name, strand)`` exactly as the OGDraw map labels this gene."""
    from .ogdraw import _gene_label_points, _is_trans_spliced

    g = _bp_gene(gene)
    trans = _is_trans_spliced(g)
    return [
        (bp, name, g.exon_strands[i] if trans else g.strand)
        for i, (bp, name) in enumerate(_gene_label_points(g))
    ]


def _gene_label_candidates(segments, links, min_fraction: float) -> list[dict[str, Any]]:
    """Every OGDraw label point of every gene, flagged when a ribbon foot carries it.

    A gene is transferred when >= ``min_fraction`` of its exon bases lie in one
    foot; a trans-spliced gene's per-exon labels count only inside that foot.
    """
    rows: list[dict[str, Any]] = []
    for segment in segments:
        feet = [
            (link[f"{role}_start"], link[f"{role}_end"])
            for link in links
            for role in ("recipient", "donor")
            if link[f"{role}_seqid"] == segment["seqid"]
        ]
        for gene in segment["genes"]:
            carried = any(
                total and covered / total >= min_fraction
                for covered, total in (_gene_coverage(gene, lo, hi) for lo, hi in feet)
            )
            trans = gene.get("splicing") == "trans"
            for bp, name, strand in _label_points(gene):
                in_foot = any(lo <= bp <= hi for lo, hi in feet)
                rows.append(
                    {
                        "seqid": segment["seqid"],
                        "gene": name,
                        "position": round(bp, 1),
                        "strand": strand,
                        "category": gene["category"],
                        "transferred": bool(carried and (in_foot or not trans)),
                    }
                )
    rows.sort(key=lambda row: (row["seqid"], row["position"]))
    return rows


def _label_font(span_deg: float, unit_in: float, count: int, radius: float | None = None) -> float:
    """Point size at which ``count`` names fill a segment's arc on a label ring."""
    from .ogdraw import _LABEL_LINE, _MAX_LABEL_FILL, _R_LABEL_OUT

    if not count:
        return _FONT_LABEL
    arc_in = math.radians(span_deg) * (radius or _R_LABEL_OUT) * unit_in
    fits = _MAX_LABEL_FILL * arc_in / (_FONT_LABEL / 72 * _LABEL_LINE)
    return _FONT_LABEL * min(1.0, fits / count)


def _choose_labels(segments, candidates, mode: str, unit_in: float, minus_inside: bool = True):
    """Names to draw on a ring of ``unit_in`` inches per unit: rows, mode per segment, notes.

    With ``minus_inside`` the plus strand is fitted on the outer label ring and the
    minus strand on the inner one (OGDraw), and the tighter of the two sets the size.
    """
    from .ogdraw import _R_LABEL_IN, _R_LABEL_OUT

    def fit(span: float, picked) -> float:
        if not minus_inside:
            return _label_font(span, unit_in, len(picked))
        plus = sum(1 for r in picked if r["strand"] != -1)
        return min(
            _label_font(span, unit_in, plus, _R_LABEL_OUT),
            _label_font(span, unit_in, len(picked) - plus, _R_LABEL_IN),
        )

    rows: list[dict[str, Any]] = []
    modes: dict[str, str] = {}
    notes: list[str] = []
    fonts: list[float] = []
    for segment in segments:
        own = [row for row in candidates if row["seqid"] == segment["seqid"]]
        if mode == "none" or not own:
            modes[segment["seqid"]] = "none"
            continue
        tiers = {
            "auto": ("all", "no_trna", "transferred", "transferred_once"),
            "all": ("all",),
            "transferred": ("transferred", "transferred_once"),
        }[mode]
        span = segment["end_deg"] - segment["start_deg"]
        for tier in tiers:
            transferred = [r for r in own if r["transferred"]]
            picked = {
                "all": own,
                "no_trna": [r for r in own if r["category"] != "tRNA" or r["transferred"]],
                "transferred": transferred,
                "transferred_once": list({r["gene"]: r for r in reversed(transferred)}.values())[::-1],
            }[tier]
            font = fit(span, picked)
            if font >= _FONT_LABEL_CROWDED or tier == tiers[-1]:
                break
        modes[segment["seqid"]] = tier
        fonts.append(max(FONT_PT["min"], font))
        rows += [dict(row) for row in picked]
        name = segment.get("label") or segment["seqid"]
        if tier == "no_trna":
            notes.append(f"{name}: tRNA genes are drawn but not named")
        elif tier == "transferred" and mode == "auto":
            notes.append(f"{name}: only genes carried by a transfer are named")
        elif tier == "transferred_once":
            notes.append(f"{name}: only genes carried by a transfer are named, each name once")
    # names stay at full size (crowding is handled by naming fewer genes)
    font = _FONT_LABEL
    for row in rows:
        row["font_pt"] = font
    return rows, modes, notes


def _ring_mm(width_mm: float) -> float:
    if width_mm >= _SIDE_LEGEND_MIN_MM:
        return min(width_mm - _RIGHT_COLUMN_MM - _COLUMN_GUTTER_MM, _RING_MAX_MM)
    return min(width_mm, MAX_HEIGHT_MM - _ZOOM_BAND_MM)


def _gene_key_mm(n_categories: int, columns: int) -> float:
    """Height of the gene-class key (title row plus entries) in millimetres."""
    if not n_categories:
        return 0.0
    return (math.ceil(n_categories / columns) + 1.6) * _LEGEND_ROW_MM


def _layout_mm(width_mm: float, ring_mm: float, n_categories: int) -> tuple[float, float]:
    """(height without the gene key, gene key height) in millimetres."""
    if width_mm >= _SIDE_LEGEND_MIN_MM:
        return ring_mm + 2.0, _gene_key_mm(n_categories, _GENE_KEY_COLUMNS_WIDE)
    return ring_mm + _ZOOM_BAND_MM, _gene_key_mm(n_categories, _GENE_KEY_COLUMNS_NARROW)


def _gene_key_fits(width_mm: float, ring_mm: float, n_categories: int) -> bool:
    body, key = _layout_mm(width_mm, ring_mm, n_categories)
    return bool(n_categories) and body + key <= MAX_HEIGHT_MM


def _height_mm(width_mm: float, ring_mm: float, n_categories: int) -> float:
    body, key = _layout_mm(width_mm, ring_mm, n_categories)
    return min(MAX_HEIGHT_MM, body + (key if body + key <= MAX_HEIGHT_MM else 0.0))


def _caption(
    title: str, species: str, kind: str, count: int, linked_bp: int, color_by: str,
    label_notes=(), zoom=None, bold=False,
) -> str:
    where = f" in {species}" if species else ""
    colour = "identity (%)" if color_by == "identity" else "fragment length (log scale)"
    parts = [
        f"{title}{where}. Each ribbon links a transferred fragment on the recipient genome to its "
        f"homologous region on the donor ({count} ribbons, {linked_bp / 1000:.1f} kb on the "
        f"recipient); ribbon width is fragment length and colour is {colour}.",
        "Outer ring: genes as in the OGDraw map (plus strand outside, minus strand inside); "
        "the plastome band shows LSC, IRb, SSC and IRa.",
    ]
    if bold:
        parts.append("Gene names in bold are carried by a transfer.")
    parts += [f"{note}." for note in label_notes]
    if zoom:
        r, d = zoom["recipient"], zoom["donor"]
        parts.append(
            f"Inset: ribbon {zoom['link']} at true scale ({zoom['length']} bp, "
            f"{zoom['identity']}% identity, {'inverted' if zoom['strand'] == '-' else 'direct'}), "
            f"{r['label']} {r['frag'][0]:,}-{r['frag'][1]:,} and "
            f"{d['label']} {d['frag'][0]:,}-{d['frag'][1]:,}."
        )
    return " ".join(parts)


def _zoom_spec(links, segments, zoom: str | None):
    """What the true-scale inset draws: one ribbon, both windows, the genes in them."""
    if zoom is None or not links:
        return None
    if zoom == "auto":
        link = min(links, key=lambda lk: (lk["length"], int(lk["id"][1:])))
    else:
        link = next((lk for lk in links if lk["id"] == zoom), None)
        if link is None:
            raise OrganelleParameterError(
                code="visualization.circos.zoom",
                message=f"zoom must be 'auto', None or a link id; {zoom!r} is not drawn.",
            )
    by_id = {seg["seqid"]: seg for seg in segments}
    longest = max(link["length"], link["donor_end"] - link["donor_start"] + 1)
    span = int(_nice_step(max(5 * longest, 500), 1.0))  # the fragment fills >= 1/10 of the window

    def window(role: str) -> dict[str, Any]:
        seg = by_id[link[f"{role}_seqid"]]
        lo, hi = link[f"{role}_start"], link[f"{role}_end"]
        start = max(1, int((lo + hi) / 2 - span / 2))
        end = min(seg["length"], start + span - 1)
        start = max(1, end - span + 1)
        genes = []
        for gene in seg["genes"]:
            exons = [[max(a, start), min(b, end), st] for a, b, st in gene["exons"] if b >= start and a <= end]
            if exons:
                genes.append({"gene": gene["gene"], "category": gene["category"], "exons": exons})
        return {
            "seqid": seg["seqid"], "label": seg.get("label") or seg["seqid"], "kind": seg["kind"],
            "start": start, "end": end, "frag": [lo, hi], "genes": genes,
        }

    return {
        "link": link["id"], "length": link["length"], "identity": link["identity"],
        "strand": link["strand"], "span_bp": span,
        "recipient": window("recipient"), "donor": window("donor"),
    }


# ── layout ───────────────────────────────────────────────────────────────────


def _layout(ordered: list[dict[str, Any]], reversed_ids: set[str], scale: str) -> list[dict[str, Any]]:
    """Clockwise start/end angles (degrees from 12 o'clock) for segments in ring order."""
    groups = [r["role"] for r in ordered]
    n_group_gaps = sum(1 for a, b in zip(groups, groups[1:] + groups[:1], strict=False) if a != b)
    n_gaps = len(ordered) - n_group_gaps
    available = 360.0 - n_group_gaps * _GROUP_GAP_DEG - n_gaps * _GAP_DEG
    totals: dict[str, int] = {}
    for segment in ordered:
        totals[segment["role"]] = totals.get(segment["role"], 0) + segment["length"]
    grand = sum(totals.values())
    placed = []
    angle = _GROUP_GAP_DEG / 2 if n_group_gaps else _GAP_DEG / 2
    for i, segment in enumerate(ordered):
        if scale == "proportional":
            span = available * segment["length"] / grand
        else:
            span = available / len(totals) * segment["length"] / totals[segment["role"]]
        placed.append(
            {**segment, "start_deg": angle, "end_deg": angle + span, "reversed": segment["seqid"] in reversed_ids}
        )
        following = ordered[(i + 1) % len(ordered)]
        angle += span + (_GROUP_GAP_DEG if following["role"] != segment["role"] else _GAP_DEG)
    return placed


def _pos_deg(segment: Mapping[str, Any], x: float) -> float:
    """Clockwise degrees of a 0-based edge coordinate ``x`` in [0, length] (base i spans [i-1, i])."""
    fraction = x / segment["length"]
    if segment["reversed"]:
        fraction = 1.0 - fraction
    return segment["start_deg"] + fraction * (segment["end_deg"] - segment["start_deg"])


def _span_deg(segment: Mapping[str, Any], lo: int, hi: int) -> tuple[float, float]:
    """Clockwise [a, b] (a <= b) covering 1-based bases lo..hi."""
    a, b = _pos_deg(segment, lo - 1), _pos_deg(segment, hi)
    return (a, b) if a <= b else (b, a)


def _mid_deg(segment: Mapping[str, Any], lo: float, hi: float) -> float:
    return _pos_deg(segment, (lo - 1 + hi) / 2)


def _foot(segment: Mapping[str, Any], lo: int, hi: int) -> tuple[float, float]:
    """Clockwise [a, b] (a < b) covering bases lo..hi, widened to ``_MIN_FOOT_DEG``."""
    a, b = _span_deg(segment, lo, hi)
    if b - a < _MIN_FOOT_DEG:
        mid = (a + b) / 2
        a, b = mid - _MIN_FOOT_DEG / 2, mid + _MIN_FOOT_DEG / 2
    return a, b


def _crossings(placed: list[dict[str, Any]], links: list[dict[str, Any]]) -> int:
    import numpy as np

    by_id = {s["seqid"]: s for s in placed}
    sample = sorted(links, key=lambda link: -link["length"])[:_MAX_CROSSING_LINKS]
    if len(sample) < 2:
        return 0
    p = np.array(
        [_mid_deg(by_id[lk["recipient_seqid"]], lk["recipient_start"], lk["recipient_end"]) for lk in sample]
    )
    q = np.array([_mid_deg(by_id[lk["donor_seqid"]], lk["donor_start"], lk["donor_end"]) for lk in sample])
    lo, hi = np.minimum(p, q), np.maximum(p, q)
    inside_lo = (lo[None, :] > lo[:, None]) & (lo[None, :] < hi[:, None])
    inside_hi = (hi[None, :] > lo[:, None]) & (hi[None, :] < hi[:, None])
    cross = inside_lo ^ inside_hi
    return int(np.triu(cross, 1).sum())


def _scale(recipients, donors) -> str:
    a = sum(s["length"] for s in recipients)
    b = sum(s["length"] for s in donors)
    return "proportional" if max(a, b) / min(a, b) <= _PROPORTIONAL_MAX_RATIO else "per_group"


def _choose_layout(recipients, donors, links, order: str) -> dict[str, Any]:
    scale = _scale(recipients, donors)
    base = _layout(recipients + donors, set(), scale)
    base_crossings = _crossings(base, links)
    best = (base_crossings, 0, base)
    if order == "auto":
        if len(recipients) <= _MAX_PERMUTED_SEGMENTS:
            orders = [list(p) for p in itertools.permutations(recipients)]
        else:
            orders = [recipients, _barycentre_order(recipients, donors, links, scale)]
        donor_ids = {s["seqid"] for s in donors}
        for rank, (recipient_order, flip) in enumerate(itertools.product(orders, (False, True))):
            placed = _layout(recipient_order + donors, donor_ids if flip else set(), scale)
            crossings = _crossings(placed, links)
            if (crossings, rank) < best[:2]:
                best = (crossings, rank, placed)
    return {
        "segments": best[2],
        "crossings": best[0],
        "crossings_input_order": base_crossings,
        "scale": scale,
    }


def _barycentre_order(recipients, donors, links, scale) -> list[dict[str, Any]]:
    placed = {s["seqid"]: s for s in _layout(recipients + donors, set(), scale)}
    weight: dict[str, list[float]] = {}
    for link in links:
        deg = _mid_deg(placed[link["donor_seqid"]], link["donor_start"], link["donor_end"])
        weight.setdefault(link["recipient_seqid"], []).append(deg)
    # Recipients whose links land late on the donor go first (ribbons nest, not cross).
    return sorted(
        recipients,
        key=lambda s: -sum(weight[s["seqid"]]) / len(weight[s["seqid"]]) if s["seqid"] in weight else 0.0,
    )


def _public_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "seqid", "label", "kind", "role", "length", "start_deg", "end_deg",
        "reversed", "regions", "genes", "species",
    )
    out = {key: segment[key] for key in keys}
    out["start_deg"] = round(out["start_deg"], 4)
    out["end_deg"] = round(out["end_deg"], 4)
    return out


# ── plot object ──────────────────────────────────────────────────────────────


def _plot(metrics: Mapping[str, Any]) -> OrganellePlot:
    kind = metrics["transfer_kind"]
    count = metrics["link_count"]
    plot = plot_result(
        OP,
        lambda path: _render_transfer_circos(metrics, path),
        organelle="mito" if kind == "MTPT" else "nuclear",
        metrics=metrics,
        key_findings=(
            {"metric": "links", "value": count},
            {"metric": "linked_recipient_bp", "value": metrics["linked_recipient_bp"], "unit": "bp"},
            {"metric": "crossings", "value": metrics["crossings"]},
        ),
        flags=("transfer_circos",),
        summary=(
            f"{kind} chord diagram prepared: {count} ribbon(s), "
            f"{metrics['linked_recipient_bp']} bp on the recipient, {metrics['crossings']} crossing(s)."
        ),
        method="matplotlib_circos",
    )
    plot._panel = _panel_drawer(metrics)
    return plot


def _restore_transfer_circos(result: OrganelleResult) -> OrganellePlot:
    """Reattach rendering to a serialized, fully computed chord-diagram Result."""
    plot = OrganellePlot.model_validate(result.model_dump(mode="python"))
    metrics = cast(Mapping[str, Any], thaw_json(plot.metrics))
    plot._renderer = lambda path: _render_transfer_circos(metrics, path)
    plot._panel = _panel_drawer(metrics)
    return plot


def _panel_drawer(metrics: Mapping[str, Any]):
    """Draw the ring into one panel of ``compose_figure`` (labels re-fitted to the panel)."""
    from .plot_object import PanelDrawer

    def draw(ax, legend: bool, label_pt: float | None = None):
        _draw_ring(ax, metrics, label_pt=label_pt)  # label notes go to the caption
        handles = _legend_handles(metrics)
        if legend:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                      fontsize=_FONT_LEGEND, frameon=False, handlelength=1.0, borderaxespad=0)
            return []
        return handles

    def label_pt(ax) -> float:
        """Gene-name size this panel would use on ``ax``."""
        from .ogdraw import _inches_per_unit

        _setup_axes(ax)
        rows, _modes, _notes, _inside = _fit_labels(
            metrics["segments"], metrics.get("gene_label_candidates") or [],
            metrics.get("gene_label_request", "auto"), _inches_per_unit(ax),
            metrics.get("minus_labels", "inside") == "inside",
        )
        return float(rows[0]["font_pt"]) if rows else _FONT_LABEL

    return PanelDrawer(
        draw, aspect=1.0, header=str(metrics.get("panel_header", "")), label_pt=label_pt
    )


# ── rendering ────────────────────────────────────────────────────────────────


def _math_deg(cw: float) -> float:
    return 90.0 - cw


def _xy(radius: float, cw: float) -> tuple[float, float]:
    t = math.radians(_math_deg(cw))
    return radius * math.cos(t), radius * math.sin(t)


def _arc_inches(span_deg: float, radius: float, unit_in: float) -> float:
    """Printed length (inches) of an arc of ``span_deg`` at ``radius``."""
    return math.radians(span_deg) * radius * unit_in


def _arc_points(radius: float, a: float, b: float, step: float = 0.5) -> list[tuple[float, float]]:
    n = max(2, int(abs(b - a) / step) + 1)
    return [_xy(radius, a + (b - a) * i / (n - 1)) for i in range(n)]


def _ribbon_path(a: tuple[float, float], b: tuple[float, float], radius: float):
    """Untwisted ribbon: foot A clockwise, Bézier across, foot B clockwise, Bézier back."""
    from matplotlib.path import Path as MplPath

    def control(p: float, q: float) -> tuple[float, float]:
        gap = abs((q - p + 180) % 360 - 180)  # 0..180
        mid = p + ((q - p + 180) % 360 - 180) / 2
        return _xy(radius * 0.72 * (1 - gap / 180) ** 1.6, mid)

    verts: list[tuple[float, float]] = []
    codes: list[int] = []
    arc_a = _arc_points(radius, a[0], a[1])
    verts.append(arc_a[0])
    codes.append(MplPath.MOVETO)
    verts.extend(arc_a[1:])
    codes.extend([MplPath.LINETO] * (len(arc_a) - 1))
    verts.extend([control(a[1], b[0]), _xy(radius, b[0])])
    codes.extend([MplPath.CURVE3, MplPath.CURVE3])
    arc_b = _arc_points(radius, b[0], b[1])
    verts.extend(arc_b[1:])
    codes.extend([MplPath.LINETO] * (len(arc_b) - 1))
    verts.extend([control(b[1], a[0]), arc_a[0]])
    codes.extend([MplPath.CURVE3, MplPath.CURVE3])
    verts.append(arc_a[0])
    codes.append(MplPath.CLOSEPOLY)
    return MplPath(verts, codes)


def _nice_step(span_bp: float, min_count: float) -> float:
    """Smallest 1-2-5 step that gives at most ``min_count`` intervals over ``span_bp``."""
    raw = span_bp / max(min_count, 1.0)
    exponent = math.floor(math.log10(raw)) if raw > 0 else 0
    for mult in (1, 2, 5, 10):
        step = mult * 10**exponent
        if step >= raw:
            return float(step)
    return float(10 ** (exponent + 1))


def _tick_text(bp: float, unit: str) -> str:
    value = bp / (1e6 if unit == "Mb" else 1e3)
    text = f"{value:.1f}" if value % 1 else f"{int(value)}"
    return text


def _render_transfer_circos(metrics: Mapping[str, Any], output: str | Path) -> Path:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    with publication_style():
        return _draw_figure(metrics, Path(output), plt)


def _gene_category_rows(segments) -> list[str]:
    return list(dict.fromkeys(g["category"] for seg in segments for g in seg.get("genes") or []))


def _colours(metrics: Mapping[str, Any]):
    """(cmap, colour(value) -> rgb) for the ribbon colour scale."""
    from matplotlib import colormaps  # type: ignore
    from matplotlib.colors import LogNorm, Normalize  # type: ignore

    cmap = colormaps[metrics.get("cmap", _CMAP)]
    lo_c, hi_c = _CMAP_RANGE
    v0, v1 = metrics["color_range"]
    if metrics["color_by"] == "length":
        norm = LogNorm(vmin=max(1.0, v0), vmax=max(v1, v0 + 1), clip=True)
    else:
        norm = Normalize(vmin=v0, vmax=max(v1, v0 + 1e-9), clip=True)

    def colour(value: float | None) -> tuple[float, float, float]:
        if value is None:
            return (0.55, 0.55, 0.6)
        return cmap(lo_c + (hi_c - lo_c) * float(norm(value)))[:3]

    return cmap, colour


def _draw_figure(metrics: Mapping[str, Any], output: Path, plt) -> Path:
    """Ring top-left; wide figures put the zoom inset, colour bar and sequence key in a
    right column (<= 1/4 of the width) and the gene-class key in columns under the ring;
    narrow figures put inset and keys in a band under the ring."""
    width_mm, ring_mm, height_mm = metrics["width_mm"], metrics["ring_mm"], metrics["height_mm"]
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))

    def rect(x_mm: float, top_mm: float, w_mm: float, h_mm: float) -> tuple[float, float, float, float]:
        return (x_mm / width_mm, 1 - (top_mm + h_mm) / height_mm, w_mm / width_mm, h_mm / height_mm)

    ax = fig.add_axes(rect(0.0, 0.0, ring_mm, ring_mm))
    _draw_ring(ax, metrics, metrics.get("gene_labels") or [])
    zoom = metrics.get("zoom")
    if width_mm >= _SIDE_LEGEND_MIN_MM:
        x = ring_mm + _COLUMN_GUTTER_MM
        col = width_mm - x
        top = 4.0
        if zoom:
            _draw_zoom(fig.add_axes(rect(x, top + 4.0, col, 46.0)), metrics)
            top += 56.0
        _colour_bar(fig, metrics, rect(x, top + 4.0, col * 0.85, 2.6))
        _sequence_key(fig, metrics, rect(x, top + 12.0, col, 0)[:2], ncol=1)
        key_top, columns = ring_mm + 2.0, _GENE_KEY_COLUMNS_WIDE
    else:
        top = ring_mm + 2.0
        if zoom:
            _draw_zoom(fig.add_axes(rect(4.0, top + 4.0, width_mm * 0.52, _ZOOM_BAND_MM - 10.0)), metrics)
        x = width_mm * 0.6
        _colour_bar(fig, metrics, rect(x, top + 4.0, width_mm - x - 3.0, 2.4))
        _sequence_key(fig, metrics, rect(x, top + 12.0, 0, 0)[:2], ncol=1)
        key_top, columns = ring_mm + _ZOOM_BAND_MM, _GENE_KEY_COLUMNS_NARROW
    if metrics.get("gene_key", True):
        _gene_key(fig, metrics, rect(2.0, key_top, 0, 0)[:2], columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(metrics.get("dpi", 600)), facecolor="white")
    plt.close(fig)
    _write_links_tsv(output.with_name(f"{output.stem}.links.tsv"), metrics)
    return output


def _rings(shift: float) -> dict[str, float]:
    """Inner track radii moved inward by ``-shift`` (band, ticks, feet, ribbons)."""
    return {
        "band_out": _R_BAND_OUT + shift,
        "band_in": _R_BAND_IN + shift,
        "tick_major": _R_TICK_MAJOR + shift,
        "tick_minor": _R_TICK_MINOR + shift,
        "tick_label": _R_TICK_LABEL + shift,
        "foot_out": _R_FOOT_OUT + shift,
        "foot_in": _R_FOOT_IN + shift,
        "link": _R_LINK + shift,
        "name_inside": _R_NAME_INSIDE + shift,
    }


def _inner_band_out(labels, unit_in: float, minus_inside: bool) -> float:
    """Outer band radius at which inward minus-strand names clear the band (unclamped)."""
    from .ogdraw import _R_LABEL_IN

    inner = [r for r in labels if minus_inside and r.get("strand") == -1]
    if not inner:
        return _R_BAND_OUT
    reach = max(
        _text_inches(f" {r['gene']} ", float(r.get("font_pt", _FONT_LABEL)), bool(r.get("transferred")))
        for r in inner
    ) / unit_in
    return min(_R_BAND_OUT, _R_LABEL_IN - reach - 0.02)


def _inner_shift(labels, unit_in: float, minus_inside: bool) -> float:
    """How far the band must move in so inward minus-strand names clear it."""
    return max(_BAND_OUT_MIN, _inner_band_out(labels, unit_in, minus_inside)) - _R_BAND_OUT


def _fit_labels(segments, candidates, mode: str, unit_in: float, minus_inside: bool):
    """:func:`_choose_labels` plus the side minus-strand names end on.

    Inward names need room between the gene ring and the band; when the ring is
    too small for that (the band would have to pass ``_BAND_OUT_MIN``) every name
    goes outside instead, minus strand in grey.
    """
    rows, modes, notes = _choose_labels(segments, candidates, mode, unit_in, minus_inside)
    if minus_inside and _inner_band_out(rows, unit_in, True) < _BAND_OUT_MIN:
        rows, modes, notes = _choose_labels(segments, candidates, mode, unit_in, False)
        minus_inside = False
    return rows, modes, notes, minus_inside


def _setup_axes(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-_LIM, _LIM)
    ax.set_ylim(-_LIM, _LIM)
    ax.axis("off")


def _draw_ring(ax, metrics: Mapping[str, Any], labels=None, label_pt: float | None = None) -> list[str]:
    """Ribbons, segments, gene ring and names into ``ax``; returns the label notes.

    ``labels=None`` re-fits the gene names to this axes (composed panels);
    ``label_pt`` caps their size so a composed figure has one gene-name size.
    """
    from matplotlib.patches import PathPatch, Wedge  # type: ignore

    from .ogdraw import GENE_COLORS, _inches_per_unit

    _setup_axes(ax)
    unit_in = _inches_per_unit(ax)
    segments = metrics["segments"]
    by_id = {seg["seqid"]: seg for seg in segments}
    notes = list(metrics.get("gene_label_notes") or [])
    minus_inside = metrics.get("minus_labels_drawn", metrics.get("minus_labels", "inside")) == "inside"
    if labels is None:
        labels, _modes, notes, minus_inside = _fit_labels(
            segments, metrics.get("gene_label_candidates") or [],
            metrics.get("gene_label_request", "auto"), unit_in,
            metrics.get("minus_labels", "inside") == "inside",
        )
    if label_pt is not None:
        labels = [dict(row, font_pt=min(float(row.get("font_pt", _FONT_LABEL)), label_pt)) for row in labels]
    rings = _rings(_inner_shift(labels, unit_in, minus_inside))
    _cmap, colour = _colours(metrics)

    # ribbons: long ones first so short ones stay on top
    for link in sorted(metrics["links"], key=lambda lk: -lk["length"]):
        a = _foot(by_id[link["recipient_seqid"]], link["recipient_start"], link["recipient_end"])
        b = _foot(by_id[link["donor_seqid"]], link["donor_start"], link["donor_end"])
        face = colour(link["identity"] if metrics["color_by"] == "identity" else link["length"])
        zoomed = bool(metrics.get("zoom")) and metrics["zoom"]["link"] == link["id"]
        ax.add_patch(
            PathPatch(
                _ribbon_path(a, b, rings["link"]),
                facecolor=(*face, _RIBBON_ALPHA),
                edgecolor="black" if zoomed else (*face, min(1.0, _RIBBON_ALPHA + 0.25)),
                lw=0.6 if zoomed else 0.3,
                zorder=2.2 if zoomed else 2,
                gid=link["id"],
            )
        )
        if zoomed:
            for x0, x1 in (a, b):
                _tangential_text(ax, rings["foot_in"] - 0.045, (x0 + x1) / 2, link["id"],
                                 _FONT_LABEL, weight="bold", zorder=5)
        for x0, x1 in (a, b):
            ax.add_patch(
                Wedge((0, 0), rings["foot_out"], _math_deg(x1), _math_deg(x0),
                      width=rings["foot_out"] - rings["foot_in"], facecolor=face, edgecolor="white",
                      lw=0.2, zorder=2.6)
            )
    for seg in segments:
        _draw_segment(ax, seg, unit_in, Wedge, GENE_COLORS, rings)
    placed = _draw_gene_labels(ax, labels, by_id, unit_in, minus_inside)
    for seg in segments:
        _draw_segment_name(ax, seg, unit_in, placed, rings)
    return notes


#: Virtual genome length handed to the OGDraw drawers: one ring degree = 1000 units,
#: so their ``_bp_to_deg(x, L)`` is our clockwise angle and their 1-unit offsets vanish.
_VIRTUAL = 360_000.0


def _virtual_gene(seg: Mapping[str, Any], gene: Mapping[str, Any]):
    """The gene in ring units for ``ogdraw._draw_gene_blocks`` (exons ascending on the ring)."""
    from .ogdraw import Gene

    spans = []
    for start, end, strand in gene["exons"]:
        x0, x1 = _span_deg(seg, start, end)
        if x1 - x0 < 0.15:  # keep a tRNA visible on a long nuclear-scale ring
            mid = (x0 + x1) / 2
            x0, x1 = mid - 0.075, mid + 0.075
        spans.append((x0 * 1000 + 1, x1 * 1000, strand))
    spans.sort()
    return Gene(
        name=gene["gene"],
        start=spans[0][0],
        end=max(e[1] for e in spans),
        strand=1,  # exons are ascending on the ring; blocks still sit on their own strand
        has_intron=gene.get("splicing", "none") != "none",
        exons=[(a, b) for a, b, _st in spans],
        category=gene["category"],
        trans_spliced=gene.get("splicing") == "trans",
        exon_strands=[st for _a, _b, st in spans],
    )


def _draw_segment(ax, seg, unit_in, Wedge, gene_colors, rings) -> None:
    from .ogdraw import (
        _R_BACKBONE,
        _R_MINUS_BLOCK_IN,
        _R_MINUS_BLOCK_OUT,
        _R_PLUS_BLOCK_IN,
        _R_PLUS_BLOCK_OUT,
        _draw_gene_blocks,
    )

    a, b = seg["start_deg"], seg["end_deg"]
    length = seg["length"]
    band = rings["band_out"] - rings["band_in"]
    ax.add_patch(
        Wedge((0, 0), rings["band_out"], _math_deg(b), _math_deg(a), width=band,
              facecolor=_KIND_COLORS[seg["kind"]], edgecolor="none", zorder=3)
    )
    for name, span in (seg.get("regions") or {}).items():
        start, size = span["start"], span["length"]
        pieces = [(start, min(start + size, length))]
        if start + size > length:
            pieces.append((0, start + size - length))
        for p0, p1 in pieces:
            x0, x1 = sorted((_pos_deg(seg, p0), _pos_deg(seg, p1)))
            ax.add_patch(
                Wedge((0, 0), rings["band_out"], _math_deg(x1), _math_deg(x0), width=band,
                      facecolor=_REGION_COLORS.get(name, "#cccccc"), edgecolor="white",
                      lw=0.5, zorder=3.1)
            )
        mid = (start + size / 2) % length
        span_deg = size / length * (b - a)
        if _arc_inches(span_deg, rings["band_out"], unit_in) > 0.3:
            _tangential_text(ax, (rings["band_in"] + rings["band_out"]) / 2, _pos_deg(seg, mid), name,
                             _FONT_REGION, color="0.1", zorder=3.3)
    ax.add_patch(
        Wedge((0, 0), rings["band_out"], _math_deg(b), _math_deg(a), width=band,
              facecolor="none", edgecolor="0.25", lw=0.5, zorder=3.2)
    )

    # OGDraw gene ring: backbone, plus-strand blocks outside, minus-strand inside
    if seg.get("genes"):
        ax.add_patch(
            Wedge((0, 0), _R_BACKBONE + 0.002, _math_deg(b), _math_deg(a), width=0.008,
                  facecolor="0.2", edgecolor="none", zorder=0)
        )
        for gene in seg["genes"]:
            g = _virtual_gene(seg, gene)
            plus = g.exon_strands[0] != -1
            r_in, r_out = (
                (_R_PLUS_BLOCK_IN, _R_PLUS_BLOCK_OUT) if plus else (_R_MINUS_BLOCK_IN, _R_MINUS_BLOCK_OUT)
            )
            _draw_gene_blocks(ax, g, gene_colors, _VIRTUAL, r_in, r_out)

    # kb ticks on the inner edge of the band, numbers set along the arc
    unit = "Mb" if length >= 2_000_000 else "kb"
    pitch_in = _FONT_TICK / 72 * 5.2
    major = _nice_step(length, max(1.0, _arc_inches(b - a, rings["tick_label"], unit_in) / pitch_in))
    per_major = 4 if int(f"{major:e}"[0]) == 2 else 5
    minor = major / per_major
    for i in range(int(length // minor) + 1):
        tick = i * minor
        is_major = i % per_major == 0
        deg = _pos_deg(seg, tick)
        r_tip = rings["tick_major"] if is_major else rings["tick_minor"]
        x0, y0 = _xy(rings["band_in"], deg)
        x1, y1 = _xy(r_tip, deg)
        ax.plot([x0, x1], [y0, y1], color="0.25", lw=0.5 if is_major else 0.35,
                solid_capstyle="butt", zorder=3)
        # never label the clockwise end (it meets the next segment's first number);
        # the unit rides on the clockwise-first number
        if seg["reversed"]:
            labelled, cw_first = tick > 0.45 * major, tick == major * int(length // major)
        else:
            labelled, cw_first = tick == 0 or length - tick > 0.45 * major, tick == 0
        if is_major and labelled:
            text = _tick_text(tick, unit) + (f" {unit}" if cw_first else "")
            _tangential_text(ax, rings["tick_label"], deg, text, _FONT_TICK, color="0.25", zorder=3)


def _text_inches(text: str, font_pt: float, bold: bool = False, italic: bool = True) -> float:
    """Printed width of ``text`` in the journal font (measured, not estimated)."""
    from .style import text_width_pt

    return text_width_pt(text, round(float(font_pt), 2), bool(bold), bool(italic)) / 72


def _wrap(text: str, width_in: float, font_pt: float) -> str:
    """``text`` wrapped to lines no wider than ``width_in`` at ``font_pt``."""
    import textwrap

    chars = max(12, int(width_in / (font_pt * 0.53 / 72)))
    return "\n".join(textwrap.wrap(text, chars))


def _draw_segment_name(ax, seg, unit_in, placed, rings) -> None:
    """Two-line name in the segment's emptiest label window, outside any names there.

    ``placed`` is ``[(seqid, angle, outer_radius), ...]`` for the drawn gene names.
    """
    a, b = seg["start_deg"], seg["end_deg"]
    length = seg["length"]
    label = seg.get("label") or seg["seqid"]
    size_text = f"{length / 1e6:.2f} Mb" if length >= 2_000_000 else f"{length / 1000:.1f} kb"
    main = f"{label} (rev.)" if seg["reversed"] else label
    sub = f"{seg['seqid']} · {size_text}"
    line = (_FONT_SEGMENT + _FONT_SEGMENT_SUB) / 2 * 1.3 / 72 / unit_in
    width_in = max(_text_inches(main, _FONT_SEGMENT, True, False), _text_inches(sub, _FONT_SEGMENT_SUB, False, False))
    names = [(angle, outer) for seqid, angle, outer in placed]

    def window(mid: float, radius: float) -> tuple[float, float]:
        # the name's own half-width plus one gene-name height: a radial name just
        # outside the window still occupies that much arc beside its anchor
        half = math.degrees((width_in / 2 + _FONT_LABEL / 72) / (radius * unit_in)) + 1.0
        return mid - half, mid + half

    best = None
    margin = min(0.1 * (b - a), 4.0)
    steps = max(1, int((b - a - 2 * margin) / 0.5))
    for k in range(steps + 1):
        mid = a + margin + (b - a - 2 * margin) * k / steps
        lo, hi = window(mid, _R_NAME_MIN + line)
        crowd = [outer for angle, outer in names if lo <= angle <= hi]
        reach = max(crowd, default=0.0)
        key = (reach, len(crowd), abs(mid - (a + b) / 2))
        if best is None or key < best[0]:
            best = (key, mid, reach)
    _key, mid, reach = best
    inner = max(_R_NAME_MIN, reach + 0.03) + line / 2
    outer = inner + line
    halo = {}
    if outer + line / 2 > _R_NAME_MAX:
        # no free window inside the axes: set the name inside the ring, under the
        # tick numbers, at the segment's middle, with a white halo over ribbons
        import matplotlib.patheffects as effects  # type: ignore

        mid = (a + b) / 2
        outer = rings["name_inside"]
        inner = outer - line
        halo = {"path_effects": [effects.withStroke(linewidth=1.8, foreground="white")]}
    upper = _upper(mid)
    _tangential_text(ax, outer if upper else inner, mid, main, _FONT_SEGMENT, weight="bold",
                     color=_shade(_KIND_COLORS[seg["kind"]], 0.55), zorder=5, **halo)
    _tangential_text(ax, inner if upper else outer, mid, sub, _FONT_SEGMENT_SUB, color="0.3",
                     zorder=5, **halo)


def _upper(cw: float) -> bool:
    deg = cw % 360
    return not (90 < deg < 270)


def _tangential_text(ax, radius, cw, text, size, **kw) -> None:
    x, y = _xy(radius, cw)
    deg = cw % 360
    rotation = -deg if _upper(cw) else 180 - deg
    ax.text(x, y, text, rotation=rotation, rotation_mode="anchor", ha="center", va="center",
            fontsize=size, **kw)


def _shade(hex_color: str, factor: float) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb  # type: ignore

    r, g, b = to_rgb(hex_color)
    return r * factor, g * factor, b * factor


def _draw_gene_labels(ax, labels, by_id, unit_in, minus_inside: bool = True) -> list[tuple[str, float, float]]:
    """OGDraw radial labels with leaders: plus strand outward outside the ring and,
    with ``minus_inside``, minus strand inward inside it (else outside, in grey).

    Each side is spread per segment at the font's own pitch on its ring, then once
    globally so neighbours across a segment gap do not overprint. Returns the
    outer names as ``(seqid, angle, outer_radius)`` for placing segment names.
    """
    from .ogdraw import (
        _LABEL_LINE,
        _R_LABEL_IN,
        _R_LABEL_OUT,
        _R_MINUS_BLOCK_IN,
        _R_MINUS_BLOCK_OUT,
        _R_PLUS_BLOCK_OUT,
        _SPREAD_PASSES,
        _draw_label,
        spread_labels,
    )

    def pitch(font_pt: float, radius: float) -> float:
        return 360.0 * (font_pt / 72 * _LABEL_LINE) / (2 * math.pi * radius * unit_in)

    sides: dict[str, dict[str, list[Mapping[str, Any]]]] = {"out": {}, "in": {}}
    for row in labels:
        side = "in" if minus_inside and row.get("strand") == -1 else "out"
        sides[side].setdefault(row["seqid"], []).append(row)
    placed_out: list[tuple[str, float, float]] = []
    for side, by_segment in sides.items():
        radius = _R_LABEL_OUT if side == "out" else _R_LABEL_IN
        entries = []
        for seqid, rows in by_segment.items():
            seg = by_id[seqid]
            font = float(rows[0].get("font_pt", _FONT_LABEL))
            anchors = [_pos_deg(seg, row["position"] - 0.5) for row in rows]
            spread, _moved = spread_labels(anchors, 360.0, pitch(font, radius), _SPREAD_PASSES)
            entries += [(at, anchor, row, font) for row, anchor, at in zip(rows, anchors, spread, strict=True)]
        if not entries:
            continue
        final, _moved = spread_labels(
            [e[0] for e in entries], 360.0, pitch(min(e[3] for e in entries), radius), _SPREAD_PASSES
        )
        for (_at, anchor, row, font), at in zip(entries, final, strict=True):
            bold = bool(row.get("transferred"))
            weight = "bold" if bold else "normal"
            if side == "in":
                _draw_label(ax, at, anchor, _R_MINUS_BLOCK_IN, _R_LABEL_IN, row["gene"], font / 11,
                            360.0, outward=False, color=_LABEL_PLUS, weight=weight)
                continue
            reach = _R_LABEL_OUT + _text_inches(f" {row['gene']} ", font, bold) / unit_in
            placed_out.append((row["seqid"], at % 360, reach))
            plus = row.get("strand", 1) != -1
            _draw_label(
                ax, at, anchor, _R_PLUS_BLOCK_OUT if plus else _R_MINUS_BLOCK_OUT, _R_LABEL_OUT,
                row["gene"], font / 11, 360.0, outward=True,
                color=_LABEL_PLUS if plus or bold else _LABEL_MINUS, weight=weight,
            )
    return placed_out


def _category_text() -> dict[str, str]:
    from .ogdraw import _LEGEND, _PLASTID_LEGEND

    text = dict(_PLASTID_LEGEND)
    text.update(dict(_LEGEND))  # complex I covers nad and ndh alike
    return text


def _sequence_handles(metrics: Mapping[str, Any]) -> list[Any]:
    from matplotlib.patches import Patch  # type: ignore

    segments = metrics["segments"]
    names = {"mitochondrion": "Mitochondrial genome", "plastid": "Plastid genome",
             "nuclear": "Nuclear genome"}
    kinds = list(dict.fromkeys(seg["kind"] for seg in segments))
    handles = [Patch(facecolor=_KIND_COLORS[k], edgecolor="0.25", lw=0.4, label=names[k]) for k in kinds]
    if any(seg.get("regions") for seg in segments):
        handles += [
            Patch(facecolor=_REGION_COLORS["LSC"], edgecolor="0.25", lw=0.4, label="LSC"),
            Patch(facecolor=_REGION_COLORS["SSC"], edgecolor="0.25", lw=0.4, label="SSC"),
            Patch(facecolor=_REGION_COLORS["IRa"], edgecolor="0.25", lw=0.4, label="IRa / IRb"),
        ]
    return handles


def _gene_handles(metrics: Mapping[str, Any]) -> list[Any]:
    from matplotlib.patches import Patch  # type: ignore

    from .ogdraw import GENE_COLORS

    text = _category_text()
    categories = _gene_category_rows(metrics["segments"])
    return [
        Patch(facecolor=GENE_COLORS.get(c, "#bbbbbb"), edgecolor="black", linewidth=0.4,
              label=text.get(c, c))
        for c in sorted(categories, key=lambda c: (c == "other", text.get(c, c).lower()))
    ]


def _legend_handles(metrics: Mapping[str, Any]) -> list[Any]:
    """Handles a composed figure merges: ribbon colour samples, sequences, gene classes."""
    from matplotlib.patches import Patch  # type: ignore

    _cmap, colour = _colours(metrics)
    v0, v1 = metrics["color_range"]
    unit = "%" if metrics["color_by"] == "identity" else " bp"
    samples = [v0, (v0 + v1) / 2, v1]
    ribbons = [
        Patch(facecolor=(*colour(v), _RIBBON_ALPHA), edgecolor=colour(v), lw=0.4,
              label=f"Ribbon {v:g}{unit}" if unit == "%" else f"Ribbon {v:,.0f}{unit}")
        for v in samples
    ]
    return ribbons + _sequence_handles(metrics) + _gene_handles(metrics)


def _colour_bar(fig, metrics, rect) -> None:
    import numpy as np

    cmap, _colour = _colours(metrics)
    cax = fig.add_axes(rect)
    lo_c, hi_c = _CMAP_RANGE
    cax.imshow(np.linspace(lo_c, hi_c, 256)[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=1,
               extent=(0, 1, 0, 1), alpha=min(1.0, _RIBBON_ALPHA + 0.25))
    cax.set_yticks([])
    v0, v1 = metrics["color_range"]
    if metrics["color_by"] == "identity":
        ticks = np.linspace(v0, v1, 5 if round(v1 - v0) % 4 == 0 else 3)
        cax.set_xticks([(t - v0) / max(v1 - v0, 1e-9) for t in ticks])
        cax.set_xticklabels([f"{t:g}" for t in ticks])
        title = "Ribbon colour: identity (%)"
    else:
        logs = np.linspace(math.log10(max(v0, 1)), math.log10(max(v1, v0 + 1)), 4)
        cax.set_xticks([(x - logs[0]) / max(logs[-1] - logs[0], 1e-9) for x in logs])
        cax.set_xticklabels([f"{10 ** x:,.0f}" for x in logs])
        title = "Ribbon colour: length (bp, log)"
    cax.set_title(title, fontsize=_FONT_LEGEND, loc="left", pad=2)
    cax.tick_params(labelsize=FONT_PT["tick"], length=2, width=0.5, pad=1.5)
    for spine in cax.spines.values():
        spine.set_linewidth(0.5)


def _sequence_key(fig, metrics, anchor, ncol: int = 1) -> None:
    fig.legend(handles=_sequence_handles(metrics), loc="upper left", bbox_to_anchor=anchor,
               ncol=ncol, frameon=False, fontsize=_FONT_LEGEND, handlelength=1.4,
               handleheight=0.9, borderaxespad=0, borderpad=0, labelspacing=0.35,
               columnspacing=1.0)


def _gene_key(fig, metrics, anchor, columns: int) -> None:
    """Gene classes drawn in the ring (only those present), in columns under it."""
    genes = _gene_handles(metrics)
    if not genes:
        return
    inside = metrics.get("minus_labels_drawn", "inside") == "inside"
    fig.legend(handles=genes, loc="upper left", bbox_to_anchor=anchor, ncol=columns,
               frameon=False, fontsize=_FONT_LEGEND, handlelength=1.0, handleheight=0.8,
               borderaxespad=0, borderpad=0, labelspacing=0.3, columnspacing=1.2,
               title="Gene ring: + strand outside, \u2212 strand inside" if inside
               else "Gene ring: + strand outside, \u2212 strand names grey",
               title_fontsize=_FONT_LEGEND, alignment="left")


def _draw_zoom(ax, metrics) -> None:
    """True-scale view of one ribbon: recipient track above, donor below, kb axes, scale bar."""
    from matplotlib.patches import Polygon, Rectangle  # type: ignore

    from .ogdraw import GENE_COLORS

    zoom = metrics["zoom"]
    _cmap, colour = _colours(metrics)
    face = colour(zoom["identity"] if metrics["color_by"] == "identity" else zoom["length"])
    span = zoom["span_bp"]
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1.2, 2.1)
    ax.axis("off")
    tracks = {"recipient": 1.0, "donor": 0.0}

    def x_of(win, bp: float) -> float:
        return (bp - win["start"]) / span

    for role, y in tracks.items():
        win = zoom[role]
        up = 1 if role == "recipient" else -1  # ticks and genes face away from the ribbon
        ax.add_patch(Rectangle((0, y - 0.025), x_of(win, win["end"] + 1), 0.05,
                               facecolor="0.6", edgecolor="none", zorder=1))
        # kb axis
        step = _nice_step(span, 4.0)
        ticks = []
        tick = math.ceil((win["start"] - 1) / step) * step
        while tick <= win["end"]:
            ticks.append(tick)
            tick += step
        for k, tick in enumerate(ticks):
            xt = x_of(win, tick + 0.5)
            ax.plot([xt, xt], [y + up * 0.05, y + up * 0.11], color="0.3", lw=0.5)
            text = _zoom_kb(tick, step) + (" kb" if k == len(ticks) - 1 else "")
            ax.text(xt, y + up * 0.15, text, ha="center" if k < len(ticks) - 1 else "right",
                    va="bottom" if up > 0 else "top", fontsize=FONT_PT["tick"], color="0.25")
        # genes in the window; names in up to two staggered rows, never overprinting
        unit_in = ax.get_position().width * ax.figure.get_figwidth() / 1.04
        row_end = [-1.0, -1.0]
        for gene in sorted(win["genes"], key=lambda g: min(e[0] for e in g["exons"])):
            for a, b, _st in gene["exons"]:
                ax.add_patch(Rectangle((x_of(win, a), y + up * 0.42 - 0.06), x_of(win, b + 1) - x_of(win, a),
                                       0.12, facecolor=GENE_COLORS.get(gene["category"], "#bbbbbb"),
                                       edgecolor="0.3", lw=0.3, zorder=2))
            mid = sum(x_of(win, (a + b) / 2) for a, b, _st in gene["exons"]) / len(gene["exons"])
            half = _text_inches(gene["gene"], _FONT_LABEL) / unit_in / 2 + 0.01
            mid = min(1.02 - half, max(-0.02 + half, mid))
            row = next((k for k in (0, 1) if mid - half > row_end[k]), None)
            if row is None:
                continue  # block drawn, name left out rather than overprinted
            row_end[row] = mid + half
            ax.text(mid, y + up * (0.55 + 0.17 * row), gene["gene"], ha="center",
                    va="bottom" if up > 0 else "top", fontsize=_FONT_LABEL, fontstyle="italic")
        # the transferred fragment at its true length
        f0, f1 = win["frag"]
        ax.add_patch(Rectangle((x_of(win, f0), y - 0.07), x_of(win, f1 + 1) - x_of(win, f0), 0.14,
                               facecolor=face, edgecolor="black", lw=0.4, zorder=3))
        # track name at the left end, on the ribbon side (accessions are on the ring)
        ax.text(0.0, y - up * 0.1, win["label"], ha="left",
                va="top" if up > 0 else "bottom", fontsize=FONT_PT["tick"], color="0.3")
    r, d = zoom["recipient"], zoom["donor"]
    r0, r1 = x_of(r, r["frag"][0]), x_of(r, r["frag"][1] + 1)
    d0, d1 = x_of(d, d["frag"][0]), x_of(d, d["frag"][1] + 1)
    if zoom["strand"] == "-":
        d0, d1 = d1, d0  # an inverted copy twists the ribbon
    ax.add_patch(Polygon([(r0, 0.93), (r1, 0.93), (d1, 0.07), (d0, 0.07)], closed=True,
                         facecolor=(*face, _RIBBON_ALPHA), edgecolor=(*face, 0.9), lw=0.4, zorder=2))
    # scale bar and caption line
    bar = _nice_step(span / 5, 1.0)
    ax.plot([0.0, bar / span], [-1.1, -1.1], color="black", lw=1.0, solid_capstyle="butt")
    ax.text(bar / span + 0.02, -1.1, f"{int(bar):,} bp", ha="left", va="center", fontsize=FONT_PT["tick"])
    kind = "inverted" if zoom["strand"] == "-" else "direct"
    ax.set_title(f"{zoom['link']} at true scale", loc="left", fontsize=_FONT_LEGEND, pad=2,
                 fontweight="bold")
    ax.text(0.0, 1.0, f"{zoom['length']} bp · {zoom['identity']}% identity · {kind}",
            transform=ax.transAxes, ha="left", va="top", fontsize=FONT_PT["tick"], color="0.25")


def _zoom_kb(bp: float, step: float) -> str:
    decimals = 0 if step >= 1000 else 1 if step >= 100 else 2 if step >= 10 else 3
    return f"{bp / 1000:.{decimals}f}"


def _write_links_tsv(path: Path, metrics: Mapping[str, Any]) -> None:
    columns = (
        "id", "recipient_seqid", "recipient_start", "recipient_end", "donor_seqid",
        "donor_start", "donor_end", "length", "identity", "strand", "fragment",
    )
    lines = ["\t".join((*columns, "recipient_genes", "donor_genes"))]
    for link in metrics["links"]:
        genes = [
            ",".join(f"{g['gene']}:{g['fraction']:g}" for g in link.get(f"{role}_genes", []))
            for role in ("recipient", "donor")
        ]
        lines.append("\t".join([*("" if link[c] is None else str(link[c]) for c in columns), *genes]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["plot_transfer_circos", "transfer_links"]
