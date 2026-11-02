"""Linear collinearity figure drawn by OrganelleVerse (no gbdraw).

Genomes are stacked as horizontal tracks with their genes coloured by
functional class (the OGDraw palette) and the plastid IR shaded. Adjacent
tracks are joined by ribbons:

* ``links="genes"`` (default) - homologous genes are matched by their
  standardised organelle gene names (``orf*`` and unnamed genes are left out,
  trans-spliced genes are drawn but not linked). Runs of at least
  ``min_block_genes`` linked genes that keep their order and orientation in
  both genomes merge into one collinear block, so a rearrangement shows as a
  block boundary and an inversion as a crossed (blue) ribbon.
* ``alignments=[table, ...]`` - one BLAST tabular (outfmt 6) table per adjacent
  pair, computed elsewhere (e.g. on the compute server); each hit becomes a
  ribbon, filtered by ``min_length`` and ``min_identity``.

Circular genomes can be rotated to start at one gene (``rotate_to``) and
reverse-complemented to follow the previous track (``orient="match_previous"``);
both are recorded per genome in ``metrics``. Everything drawn is computed into
``metrics``; drawing reads ``metrics`` only, so the figure redraws through
``ov.write`` and composes with ``compose_figure``.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..core.errors import OrganelleInputError, OrganelleParameterError
from .plot_object import OrganellePlot, PanelDrawer, plot_result
from .style import FONT_PT, JOURNAL_WIDTHS_MM, MAX_HEIGHT_MM, publication_rc, text_width_pt

_PT_PER_MM = 72.0 / 25.4
_INK = "#1F2328"
_GREY = "#8C939D"
_BASE = "#9AA0A6"
#: Okabe-Ito vermillion / blue, as in ``plot_synteny_matrix``.
_FORWARD, _INVERTED = "#D55E00", "#0072B2"
_IR_FILL = "#D5DAE1"  # grey, so it never reads as the blue of an inverted link
_GENE_H = 4.0  # pt, one strand row
_TRACK_H = 2 * _GENE_H + 3.0
_SKIP_PREFIXES = ("orf", "hypothetical", "unknown")
_LOCUS_TAG = re.compile(r"\d{3,}$")

COLLINEARITY_OPERATIONS = frozenset({"visualization.plot_collinearity"})


# ── reading genomes ───────────────────────────────────────────────────────────


def _paths(inputs: Any) -> list[Path]:
    if isinstance(inputs, (str, Path)):
        items = [inputs]
    elif isinstance(inputs, Sequence):
        items = list(inputs)
    else:
        items = [inputs]
    paths = []
    for item in items:
        path = getattr(item, "path", item)
        if not isinstance(path, (str, Path)):
            raise OrganelleInputError(
                code="visualization.collinearity.input",
                message="give GenBank files (use method='gbdraw' for other inputs)",
            )
        path = Path(path)
        if not path.is_file():
            raise OrganelleInputError(
                code="visualization.collinearity.missing", message=f"GenBank file not found: {path}"
            )
        paths.append(path)
    if len(paths) < 2:
        raise OrganelleParameterError(
            code="visualization.collinearity.count", message="collinearity needs at least two genomes"
        )
    return paths


#: Synonyms seen in RefSeq organelle annotations -> one name per gene.
_SYNONYMS = {
    "pbf1": "psbn", "pafi": "ycf3", "paf1": "ycf3", "pafii": "ycf4", "paf2": "ycf4",
    "ycf9": "psbz", "ycf10": "cema", "ycf5": "ccsa", "ycf6": "petn", "ycf7": "petl",
    "cytb": "cob", "rrns": "rrn18", "rrnl": "rrn26", "rrn18s": "rrn18", "rrn26s": "rrn26",
    "rrn16s": "rrn16", "rrn23s": "rrn23", "rrn5s": "rrn5", "rrn4.5s": "rrn4.5",
}
_AMINO = {
    "ala": "a", "arg": "r", "asn": "n", "asp": "d", "cys": "c", "gln": "q", "glu": "e",
    "gly": "g", "his": "h", "ile": "i", "leu": "l", "lys": "k", "met": "m", "fmet": "fm",
    "phe": "f", "pro": "p", "ser": "s", "thr": "t", "trp": "w", "tyr": "y", "val": "v",
    "sec": "u",
}
_TRNA = re.compile(r"^trn([a-z]{1,2})\s*(?:[-(]\s*([acgtu]{3})\)?)?$")
_COPY_SUFFIX = re.compile(r"\s+(?:i{1,3}|iv|[0-9])$")


def _gene_key(name: str) -> str | None:
    """A comparable gene key, or ``None`` for orfs and locus-tag style names."""
    key = " ".join(str(name).split()).lower()
    key = re.split(r"[;,]", key)[0].strip()  # "trnd-guc; trnd(gtc)" -> first name
    key = _COPY_SUFFIX.sub("", key)
    # orfs and locus-tag style names ("brdic_r001") are not gene names
    if not key or key.startswith(_SKIP_PREFIXES) or _LOCUS_TAG.search(key):
        return None
    match = _TRNA.match(key)
    if match:
        amino, anticodon = match.groups()
        return f"trn{amino}-{anticodon.replace('t', 'u')}" if anticodon else f"trn{amino}"
    return _SYNONYMS.get(key, key)


def _name_from_product(feature) -> str | None:
    """``tRNA-His`` -> ``trnH``, ``16S ribosomal RNA`` -> ``rrn16`` (for features without /gene)."""
    product = " ".join(feature.qualifiers.get("product", [""])).strip()
    if feature.type == "tRNA":
        match = re.match(r"tRNA-([A-Za-z]+)", product)
        amino = _AMINO.get(match.group(1).lower()) if match else None
        return f"trn{amino.upper() if len(amino) == 1 else 'fM'}" if amino else None
    if feature.type == "rRNA":
        match = re.match(r"([0-9.]+)\s*S", product)
        return f"rrn{match.group(1)}" if match else None
    return None


def _name_unnamed_rnas(record) -> None:
    """Give tRNA/rRNA features that carry only a locus tag a gene name from their product."""
    for feature in record.features:
        if feature.type in {"tRNA", "rRNA"} and "gene" not in feature.qualifiers:
            name = _name_from_product(feature)
            if name:
                feature.qualifiers["gene"] = [name]


def _trna_base(key: str) -> str:
    """``trnh-gug`` -> ``trnh``: the amino-acid level of a tRNA name."""
    return key.split("-")[0] if key.startswith("trn") else key


def _read_genome(path: Path, name: str | None, show_ir: bool | None) -> dict[str, Any]:
    """One genome: its records laid end to end (contigs), genes, IR regions."""
    from Bio import SeqIO  # type: ignore

    from .features import record_display_name, resolve_ir_regions
    from .ogdraw import _parse_genbank_record, record_organelle

    records = list(SeqIO.parse(str(path), "genbank"))
    if not records:
        raise OrganelleInputError(
            code="visualization.collinearity.empty", message=f"no GenBank record in {path}"
        )
    organelle = record_organelle(records[0])
    gap = 0
    total = sum(len(r.seq) for r in records)
    spacer = int(total * 0.01) if len(records) > 1 else 0
    contigs, genes, regions = [], [], []
    for record in records:
        offset = gap
        length = len(record.seq)
        contigs.append({"id": record.id, "name": record.name, "start": offset, "end": offset + length})
        _name_unnamed_rnas(record)
        parsed = _parse_genbank_record(record, gc_window=0)
        for gene in parsed.genes:
            trans = bool(gene.trans_spliced) or (
                len(gene.exons) > 1 and abs(gene.end - gene.start) > length / 2
            )
            parts = gene.exons if trans and gene.exons else [(gene.start, gene.end)]
            for a, b in parts:
                genes.append({
                    "name": gene.name,
                    "key": _gene_key(gene.name),
                    "trans": trans,
                    "start": offset + min(a, b) - 1,
                    "end": offset + max(a, b),
                    "strand": 1 if gene.strand >= 0 else -1,
                    "category": gene.category,
                })
        if len(records) == 1:
            for region, span in resolve_ir_regions(record, organelle, show_ir).items():
                start = int(span["start"])
                regions.append({"name": region, "start": start, "end": start + int(span["length"])})
        gap = offset + length + spacer
    return {
        "name": name or record_display_name(records[0], path.stem),
        "source": path.name,
        "organelle": organelle,
        "length": gap - spacer if len(records) > 1 else gap,
        "contigs": contigs,
        "genes": sorted(genes, key=lambda g: (g["start"], g["end"])),
        "regions": regions,
        "rotated_by": 0,
        "reverse_complemented": False,
    }


# ── rotation and orientation ──────────────────────────────────────────────────


def _anchor_strand(genome: dict[str, Any], gene: str) -> int | None:
    key = _gene_key(gene) or gene.lower()
    hits = [g for g in genome["genes"] if (g["key"] or g["name"].lower()) == key]
    return hits[0]["strand"] if hits else None


def _rotate(genome: dict[str, Any], gene: str) -> bool:
    """Start a single-record circular genome at the first copy of ``gene``."""
    if len(genome["contigs"]) != 1:
        return False
    key = _gene_key(gene) or gene.lower()
    hits = [g for g in genome["genes"] if (g["key"] or g["name"].lower()) == key]
    if not hits:
        return False
    length = genome["length"]
    anchor = hits[0]
    # rotation only: the strand stays as annotated (a plastome starts with the
    # minus-strand psbA); ``orient`` decides reverse complements
    offset = anchor["start"]

    def shift(start: int, end: int) -> list[tuple[int, int]]:
        a, b = (start - offset) % length, (end - offset) % length
        if b == 0:
            b = length
        return [(a, b)] if a < b else [(a, length), (0, b)]

    genes = []
    for g in genome["genes"]:
        spans = shift(g["start"], g["end"])
        # a gene cut by the new origin keeps its larger piece
        a, b = max(spans, key=lambda s: s[1] - s[0])
        genes.append({**g, "start": a, "end": b})
    regions = [
        {**r, "start": a, "end": b} for r in genome["regions"] for a, b in shift(r["start"], r["end"])
    ]
    genome.update(genes=sorted(genes, key=lambda g: (g["start"], g["end"])), regions=regions,
                  rotated_by=offset)
    return True


def _flip(genome: dict[str, Any]) -> None:
    length = genome["length"]
    genome["genes"] = sorted(
        ({**g, "start": length - g["end"], "end": length - g["start"], "strand": -g["strand"]}
         for g in genome["genes"]),
        key=lambda g: (g["start"], g["end"]),
    )
    genome["regions"] = [
        {**r, "start": length - r["end"], "end": length - r["start"]} for r in genome["regions"]
    ]
    genome["contigs"] = [
        {**c, "start": length - c["end"], "end": length - c["start"]} for c in reversed(genome["contigs"])
    ]
    genome["reverse_complemented"] = not genome["reverse_complemented"]


# ── links and collinear blocks ────────────────────────────────────────────────


def _gene_links(
    a: dict[str, Any], b: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    """Pair same-named genes; copies (IR) pair by relative position.

    Returns the links, the gene keys found only in ``a`` and only in ``b``, and
    the trans-spliced gene keys that were left unlinked.
    """
    # when one annotation names tRNAs without anticodons, match them by amino acid
    def bare(genome):
        return any(g["key"] and g["key"].startswith("trn") and "-" not in g["key"] for g in genome["genes"])

    loose = bare(a) or bare(b)

    def key(g):
        return _trna_base(g["key"]) if loose else g["key"]

    # a gene trans-spliced in either genome is drawn but not linked: its exons
    # cannot be paired one to one from names alone
    trans = {key(g) for genome in (a, b) for g in genome["genes"] if g["key"] and g["trans"]}
    by_a: dict[str, list[dict[str, Any]]] = {}
    by_b: dict[str, list[dict[str, Any]]] = {}
    for g in a["genes"]:
        if g["key"] and key(g) not in trans:
            by_a.setdefault(key(g), []).append(g)
    for g in b["genes"]:
        if g["key"] and key(g) not in trans:
            by_b.setdefault(key(g), []).append(g)
    links = []
    for key in sorted(set(by_a) & set(by_b)):
        left, right = by_a[key], by_b[key]
        pool = list(right)
        for g in sorted(left, key=lambda x: x["start"]):
            if not pool:
                break
            rel = g["start"] / a["length"]
            h = min(pool, key=lambda x: abs(x["start"] / b["length"] - rel))
            pool.remove(h)
            links.append({
                "gene": g["name"],
                "a_start": g["start"], "a_end": g["end"],
                "b_start": h["start"], "b_end": h["end"],
                "inverted": g["strand"] != h["strand"],
            })
    return (sorted(links, key=lambda l: l["a_start"]), sorted(set(by_a) - set(by_b)),
            sorted(set(by_b) - set(by_a)), sorted(trans))


def _blocks(links: list[dict[str, Any]], min_genes: int) -> list[dict[str, Any]]:
    """Merge runs that keep order and orientation in both genomes."""
    if not links:
        return []
    rank_b = {id(l): i for i, l in enumerate(sorted(links, key=lambda l: l["b_start"]))}
    runs: list[list[dict[str, Any]]] = [[links[0]]]
    for link in links[1:]:
        prev = runs[-1][-1]
        step = rank_b[id(link)] - rank_b[id(prev)]
        same = link["inverted"] == prev["inverted"] and step == (-1 if link["inverted"] else 1)
        if same:
            runs[-1].append(link)
        else:
            runs.append([link])
    blocks = []
    for k, run in enumerate(runs):
        if len(run) < min_genes:
            continue
        for link in run:
            link["block"] = len(blocks)
        blocks.append({
            "a_start": min(l["a_start"] for l in run), "a_end": max(l["a_end"] for l in run),
            "b_start": min(l["b_start"] for l in run), "b_end": max(l["b_end"] for l in run),
            "inverted": run[0]["inverted"], "n_genes": len(run),
            "first_gene": run[0]["gene"], "last_gene": run[-1]["gene"],
        })
    return blocks


def _contig_offsets(genome: dict[str, Any]) -> dict[str, tuple[int, int, bool]]:
    out = {}
    for c in genome["contigs"]:
        for ident in {c["id"], c["name"], c["id"].split(".")[0]}:
            out[ident] = (c["start"], c["end"], genome["reverse_complemented"])
    return out


def _alignment_links(path: Path, a: dict[str, Any], b: dict[str, Any], min_length: int,
                     min_identity: float) -> tuple[list[dict[str, Any]], int]:
    """BLAST outfmt 6 hits (query = upper track) as links; returns (links, skipped)."""
    links, skipped = [], 0
    offsets_a, offsets_b = _contig_offsets(a), _contig_offsets(b)

    def place(genome, offsets, ident, s, e):
        hit = offsets.get(ident) or offsets.get(ident.split(".")[0])
        if hit is None:
            return None
        start, end, flipped = hit
        lo, hi = min(s, e) - 1, max(s, e)
        if flipped:  # coordinates in the file are of the original orientation
            span = end - start
            lo, hi = span - hi, span - lo
        rot = genome["rotated_by"] if len(genome["contigs"]) == 1 else 0
        length = genome["length"]
        lo, hi = (lo - rot) % length, (hi - rot) % length or length
        if hi <= lo:
            hi = length
        return start + lo, start + hi

    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or row[0].startswith("#") or len(row) < 10:
                continue
            try:
                ident, length = float(row[2]), int(float(row[3]))
                qs, qe, ss, se = (int(float(v)) for v in row[6:10])
            except ValueError:
                skipped += 1
                continue
            if length < min_length or ident < min_identity:
                continue
            pa, pb = place(a, offsets_a, row[0], qs, qe), place(b, offsets_b, row[1], ss, se)
            if pa is None or pb is None:
                skipped += 1
                continue
            inverted = (qe < qs) != (se < ss)
            if a["reverse_complemented"] != b["reverse_complemented"]:
                inverted = not inverted
            links.append({"gene": None, "a_start": pa[0], "a_end": pa[1], "b_start": pb[0],
                          "b_end": pb[1], "inverted": inverted, "identity": ident})
    return sorted(links, key=lambda l: l["a_start"]), skipped


def _inverted_share(links: list[dict[str, Any]]) -> float:
    total = sum(l["a_end"] - l["a_start"] for l in links) or 1
    return sum(l["a_end"] - l["a_start"] for l in links if l["inverted"]) / total


# ── public function ───────────────────────────────────────────────────────────


def plot_collinearity_ov(
    inputs: Any,
    *,
    names: Sequence[str] | None = None,
    alignments: Sequence[str | Path] | None = None,
    min_block_genes: int = 2,
    min_length: int = 200,
    min_identity: float = 0.0,
    rotate_to: str | None = None,
    orient: str = "as_given",
    scale: str = "shared",
    gene_labels: str | Sequence[str] = "none",
    show_ir: bool | None = None,
    width: str | float = "double",
    dpi: int = 600,
) -> OrganellePlot:
    """Prepare a linear collinearity figure drawn by OrganelleVerse.

    Parameters
    ----------
    inputs : two or more GenBank files, top track first (a multi-record file is
        one genome with its contigs end to end)
    names : track names (default: the species named in each record)
    alignments : one BLAST outfmt 6 table per adjacent pair (query = upper
        track); replaces the gene links with alignment ribbons
    min_block_genes : linked genes in a row that form a collinear block
    min_length, min_identity : filters for alignment hits (bp, percent)
    rotate_to : start every single-record circular genome at this gene (the
        strand is kept; use ``orient`` to reverse-complement)
    orient : ``"as_given"``; ``"anchor"`` (with ``rotate_to``: reverse-complement
        every genome whose anchor gene lies on the other strand than in the first
        genome - for plastomes ``rotate_to="psbA", orient="anchor"`` gives the
        usual orientation); or ``"match_previous"`` (reverse-complement a genome
        when most of its linked length is inverted against the track above)
    scale : ``"shared"`` (one bp scale, the longest genome spans the width) or
        ``"fit"`` (every track spans the width)
    gene_labels : ``"none"``, ``"all"`` (thinned so labels never collide) or a
        list of gene names to label in every track
    show_ir : shade the plastid IR (``None`` = when found)
    width : ``"single"`` / ``"onehalf"`` / ``"double"`` or millimetres
    """
    paths = _paths(inputs)
    if names is not None and len(names) != len(paths):
        raise OrganelleParameterError(
            code="visualization.collinearity.names", message="give one name per genome"
        )
    if orient not in {"as_given", "match_previous", "anchor"}:
        raise OrganelleParameterError(
            code="visualization.collinearity.orient",
            message="orient is 'as_given', 'match_previous' or 'anchor'",
        )
    if orient == "anchor" and not rotate_to:
        raise OrganelleParameterError(
            code="visualization.collinearity.orient_anchor",
            message="orient='anchor' needs rotate_to (the anchor gene)",
        )
    if scale not in {"shared", "fit"}:
        raise OrganelleParameterError(
            code="visualization.collinearity.scale", message="scale is 'shared' or 'fit'"
        )
    if alignments is not None and len(alignments) != len(paths) - 1:
        raise OrganelleParameterError(
            code="visualization.collinearity.alignments",
            message="give one alignment table per adjacent pair of genomes",
        )
    genomes = [_read_genome(p, names[i] if names else None, show_ir) for i, p in enumerate(paths)]
    not_rotated = []
    if rotate_to:
        not_rotated = [g["name"] for g in genomes if not _rotate(g, rotate_to)]
        if orient == "anchor":
            # same strand for the anchor gene as in the first genome, then restart at it
            first = _anchor_strand(genomes[0], rotate_to)
            for g in genomes[1:]:
                strand = _anchor_strand(g, rotate_to)
                if first is not None and strand is not None and strand != first and len(g["contigs"]) == 1:
                    _flip(g)
                    rotated = g["rotated_by"]
                    _rotate(g, rotate_to)
                    g["rotated_by"] = (rotated + g["rotated_by"]) % g["length"]

    pairs = []
    for i in range(len(genomes) - 1):
        a, b = genomes[i], genomes[i + 1]
        links, only_a, only_b, trans = _gene_links(a, b)
        if orient == "match_previous" and links and _inverted_share(links) > 0.5:
            _flip(b)
            links, only_a, only_b, trans = _gene_links(a, b)
        if alignments is not None:
            links, skipped = _alignment_links(Path(alignments[i]), a, b, min_length, min_identity)
            blocks = []
        else:
            skipped = 0
            blocks = _blocks(links, max(1, int(min_block_genes)))
        pairs.append({
            "upper": i, "lower": i + 1,
            "source": "alignment" if alignments is not None else "genes",
            "links": links,
            "blocks": blocks,
            "n_links": len(links),
            "n_inverted_links": sum(l["inverted"] for l in links),
            "n_blocks": len(blocks),
            "n_inverted_blocks": sum(b_["inverted"] for b_ in blocks),
            "inverted_share": round(_inverted_share(links), 4) if links else 0.0,
            "only_upper": only_a,
            "only_lower": only_b,
            "trans_spliced_not_linked": trans,
            "skipped_rows": skipped,
        })

    labels: Any = gene_labels if isinstance(gene_labels, str) else sorted({_gene_key(n) or n.lower() for n in gene_labels})
    if isinstance(labels, str) and labels not in {"none", "all"}:
        raise OrganelleParameterError(
            code="visualization.collinearity.labels",
            message="gene_labels is 'none', 'all' or a list of gene names",
        )
    metrics: dict[str, Any] = {
        "genomes": genomes,
        "pairs": pairs,
        "scale": scale,
        "gene_labels": labels,
        "orient": orient,
        "rotate_to": rotate_to,
        "not_rotated": not_rotated,
        "min_block_genes": int(min_block_genes),
        "link_rule": (
            "BLAST outfmt 6 hits, query = upper track" if alignments is not None
            else "same standardised gene name (tRNAs by amino acid when one annotation omits "
                 "anticodons); orf, locus-tag and unnamed genes and trans-spliced exons not linked; "
                 "IR copies paired by relative position"
        ),
        "coordinates": "0-based start, end exclusive, after any rotation / reverse complement",
        "dpi": int(dpi),
    }
    _size(metrics, width)
    flags = []
    if not_rotated:
        flags.append("rotation_not_applied")
    if any(g["reverse_complemented"] for g in genomes):
        flags.append("reverse_complemented")
    if any(p["n_inverted_blocks"] or (p["source"] == "alignment" and p["n_inverted_links"]) for p in pairs):
        flags.append("inversion")
    if any(p["skipped_rows"] for p in pairs):
        flags.append("alignment_rows_skipped")
    if metrics["height_mm"] > MAX_HEIGHT_MM:
        flags.append("taller_than_page")
    organelles = {g["organelle"] for g in genomes}
    summary = "; ".join(
        f"{genomes[p['upper']]['name']} vs {genomes[p['lower']]['name']}: "
        f"{p['n_links']} links, {p['n_blocks']} blocks ({p['n_inverted_blocks']} inverted)"
        for p in pairs
    )
    return _with_panel(plot_result(
        "plot_collinearity",
        lambda path: _render(Path(path), metrics),
        organelle=organelles.pop() if len(organelles) == 1 else "mixed",
        method="ov_collinearity",
        metrics=metrics,
        flags=tuple(flags),
        summary=f"Collinearity of {len(genomes)} genomes prepared ({summary}).",
    ), metrics)


# ── layout ────────────────────────────────────────────────────────────────────


def _label_set(m: Mapping[str, Any], genome: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels = m["gene_labels"]
    if labels == "none":
        return []
    if labels == "all":
        return [g for g in genome["genes"] if g["category"] != "tRNA"]
    wanted = set(labels)
    return [g for g in genome["genes"] if (g["key"] or g["name"].lower()) in wanted]


def _label_room(m: Mapping[str, Any]) -> float:
    names = [g["name"] for genome in m["genomes"] for g in _label_set(m, genome)]
    if not names:
        return 0.0
    return min(max(text_width_pt(n, FONT_PT["label"], italic=True) for n in names), 40.0) + 3.0


def _margins(m: Mapping[str, Any]) -> tuple[float, float]:
    left = max(text_width_pt(g["name"], FONT_PT["label"], italic=True) for g in m["genomes"]) + 6.0
    right = max(text_width_pt(_kb(g["length"]), FONT_PT["tick"]) for g in m["genomes"]) + 6.0
    return left, right


def _gap(m: Mapping[str, Any]) -> float:
    return 26.0 + _label_room(m)


def _legend_entries(m: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    from .ogdraw import legend_rows

    entries = [("ribbon", _FORWARD, "Same orientation"), ("ribbon", _INVERTED, "Inverted")]
    if any(g["regions"] for g in m["genomes"]):
        entries.append(("patch", _IR_FILL, "Inverted repeat (IR)"))
    from .ogdraw import GENE_COLORS

    present = {g["category"] for genome in m["genomes"] for g in genome["genes"]}
    seen = set()
    for organelle in sorted({g["organelle"] for g in m["genomes"]}):
        for key, label in legend_rows(organelle):
            if key in present and key not in seen:
                seen.add(key)
                entries.append(("patch", GENE_COLORS.get(key, "#DA70D6"), label))
    return entries


def _legend_band(entries, width_pt: float) -> tuple[float, int]:
    if not entries:
        return 0.0, 1
    item = max(text_width_pt(e[2], FONT_PT["tick"]) for e in entries) + 14.0
    ncol = max(1, min(len(entries), int(width_pt // item)))
    return math.ceil(len(entries) / ncol) * 8.5 + 5.0, ncol


def _size(m: dict[str, Any], width: str | float) -> None:
    total = (JOURNAL_WIDTHS_MM[width] if isinstance(width, str) else float(width)) * _PT_PER_MM
    n = len(m["genomes"])
    top = 4.0 + _label_room(m)
    body = top + n * _TRACK_H + (n - 1) * _gap(m) + 16.0
    band, _ = _legend_band(_legend_entries(m), total)
    m["width_mm"] = round(total / _PT_PER_MM, 2)
    m["height_mm"] = round((body + band) / _PT_PER_MM, 2)


def _kb(length: float) -> str:
    return f"{length / 1e6:.2f} Mb" if length >= 1e6 else f"{length / 1e3:.1f} kb"


def _scale_length(span_bp: float) -> int:
    limit = span_bp * 0.15
    base = 10 ** math.floor(math.log10(limit))
    return int(max(k * base for k in (1, 2, 5) if k * base <= limit))


# ── drawing ───────────────────────────────────────────────────────────────────


def _ribbon(xa1, xa2, ya, xb1, xb2, yb, inverted):
    from matplotlib.path import Path as MPath  # type: ignore

    ym = (ya + yb) / 2.0
    end_b, start_b = (xb1, xb2) if inverted else (xb2, xb1)
    vertices = [
        (xa1, ya), (xa2, ya),
        (xa2, ym), (end_b, ym), (end_b, yb),
        (start_b, yb),
        (start_b, ym), (xa1, ym), (xa1, ya),
        (xa1, ya),
    ]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    return MPath(vertices, codes)


def _draw(ax: Any, m: Mapping[str, Any], legend: bool) -> list[Any]:
    from matplotlib.collections import LineCollection  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore
    from matplotlib.patches import Patch, PathPatch, Rectangle  # type: ignore

    from .ogdraw import GENE_COLORS

    fig = ax.figure
    box = ax.get_position()
    width, height = box.width * fig.get_figwidth() * 72.0, box.height * fig.get_figheight() * 72.0
    entries = _legend_entries(m)
    band, ncol = _legend_band(entries, width) if legend else (0.0, 1)
    ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    genomes = m["genomes"]
    left, right = _margins(m)
    track_w = width - left - right
    longest = max(g["length"] for g in genomes)
    room = _label_room(m)
    gap = (height - band - 4.0 - room - 16.0 - len(genomes) * _TRACK_H) / max(len(genomes) - 1, 1)
    gap = max(gap, 10.0)

    def scale(g) -> float:
        return track_w / (longest if m["scale"] == "shared" else g["length"])

    def base_y(i: int) -> float:
        return 4.0 + room + i * (_TRACK_H + gap) + _TRACK_H / 2.0

    # ribbons first, under the tracks
    for pair in m["pairs"]:
        a, b = genomes[pair["upper"]], genomes[pair["lower"]]
        sa, sb = scale(a), scale(b)
        ya, yb = base_y(pair["upper"]) + _TRACK_H / 2.0, base_y(pair["lower"]) - _TRACK_H / 2.0
        # one ribbon per link: a collinear block reads as a run of parallel
        # ribbons, and the genes inside it stay visible; blocks are counted in metrics
        shapes = [(l, 0.42 if l.get("block") is not None or pair["source"] == "alignment" else 0.3)
                  for l in pair["links"]]
        for item, alpha in shapes:
            path = _ribbon(left + item["a_start"] * sa, left + item["a_end"] * sa, ya,
                           left + item["b_start"] * sb, left + item["b_end"] * sb, yb, item["inverted"])
            colour = _INVERTED if item["inverted"] else _FORWARD
            ax.add_patch(PathPatch(path, facecolor=colour, edgecolor="none", alpha=alpha, zorder=1))

    for i, g in enumerate(genomes):
        s, y = scale(g), base_y(i)
        for region in g["regions"]:
            if region["name"].startswith("IR"):
                ax.add_patch(Rectangle((left + region["start"] * s, y - _TRACK_H / 2.0),
                                       (region["end"] - region["start"]) * s, _TRACK_H,
                                       facecolor=_IR_FILL, edgecolor="none", alpha=0.55, zorder=2))
        segs = [[(left + c["start"] * s, y), (left + c["end"] * s, y)] for c in g["contigs"]]
        ax.add_collection(LineCollection(segs, colors=_BASE, linewidths=0.6, zorder=3))
        for gene in g["genes"]:
            x0, x1 = left + gene["start"] * s, left + gene["end"] * s
            y0 = y - _GENE_H if gene["strand"] > 0 else y
            ax.add_patch(Rectangle((x0, y0), max(x1 - x0, 0.35), _GENE_H,
                                   facecolor=GENE_COLORS.get(gene["category"], "#DA70D6"),
                                   edgecolor="none", zorder=4))
        ax.text(left - 4.0, y, g["name"], ha="right", va="center", fontsize=FONT_PT["label"],
                fontstyle="italic", color=_INK)
        ax.text(left + g["length"] * s + 4.0, y, _kb(g["length"]), ha="left", va="center",
                fontsize=FONT_PT["tick"], color=_GREY)
        last = -1e9
        for gene in sorted(_label_set(m, g), key=lambda x: x["start"]):
            xm = left + (gene["start"] + gene["end"]) / 2.0 * s
            if xm - last < FONT_PT["label"] * 1.15:
                continue
            last = xm
            ax.text(xm, y - _TRACK_H / 2.0 - 1.0, gene["name"], rotation=90, ha="center", va="bottom",
                    fontsize=FONT_PT["label"], fontstyle="italic", color=_INK, zorder=5)

    scale_bp = longest if m["scale"] == "shared" else genomes[-1]["length"]
    bar = _scale_length(scale_bp)
    bar_y = base_y(len(genomes) - 1) + _TRACK_H / 2.0 + 8.0
    s_last = scale(genomes[-1])
    x_end = left + track_w
    ax.plot([x_end - bar * s_last, x_end], [bar_y, bar_y], color=_INK, lw=0.65, solid_capstyle="butt")
    ax.text(x_end - bar * s_last / 2.0, bar_y + 1.5, _kb(bar).replace(".0 kb", " kb"), ha="center",
            va="top", fontsize=FONT_PT["tick"], color=_INK)

    handles = []
    for kind, colour, label in entries:
        if kind == "ribbon":
            handles.append(Patch(facecolor=colour, edgecolor="none", alpha=0.45, label=label))
        else:
            handles.append(Patch(facecolor=colour, edgecolor="none", label=label))
    if legend:
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=ncol,
                  frameon=False, fontsize=FONT_PT["tick"], handlelength=1.1, handleheight=0.8,
                  handletextpad=0.4, columnspacing=1.0, borderaxespad=0.2, labelspacing=0.35)
        return []
    _ = Line2D  # imported for symmetry with the other drawers
    return handles


def _render(output: Path, m: Mapping[str, Any]) -> Path:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(publication_rc()):
        fig = plt.figure(figsize=(m["width_mm"] / 25.4, m["height_mm"] / 25.4))
        try:
            ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
            _draw(ax, m, True)
            fig.savefig(str(output), dpi=int(m.get("dpi", 600)), facecolor="white")
        finally:
            plt.close(fig)
    return output


def _with_panel(plot: OrganellePlot, metrics: Mapping[str, Any]) -> OrganellePlot:
    plot._panel = PanelDrawer(
        lambda ax, legend: _draw(ax, metrics, legend),
        float(metrics["height_mm"]) / float(metrics["width_mm"]),
        "Collinearity",
    )
    return plot


def restore_collinearity_plot(result) -> OrganellePlot:
    """Reattach rendering to a serialized OV collinearity Result; nothing is recomputed."""
    from ..core.frozen import thaw_json

    plot = OrganellePlot.model_validate(result.model_dump(mode="python"))
    metrics = thaw_json(plot.metrics)
    plot._renderer = lambda path: _render(Path(path), metrics)
    return _with_panel(plot, metrics)


__all__ = ["COLLINEARITY_OPERATIONS", "plot_collinearity_ov", "restore_collinearity_plot"]
