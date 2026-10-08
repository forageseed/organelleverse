"""mVISTA-style whole-genome sequence identity against one reference genome.

Each query genome is anchored to the reference with ``mappy`` (minimap2),
and every collinear block is then aligned exactly with an ``edlib`` global
alignment. Per-window identity is reported along reference coordinates and
an optional GenBank annotation of the reference drives the mVISTA-style
category coloring (coding / UTR / intron / non-coding gene / intergenic).

Alignment strategy (anchor-then-align, the same two-phase idea as
Shuffle-LAGAN/mVISTA):

1. ``mappy`` maps the query to the reference. The two inverted-repeat copies
   of a plastome cross-map to both reference IRs, and some lineages carry
   inversions or whole-genome reverse complements; the anchor blocks carry
   the strand and order information that makes those cases explicit.
2. Hits are ranked by mapq and matching length; overlaps are
   trimmed along their CIGAR on both genomes. Each reference AND query
   position belongs to at most one block, including the two IR copies.
   Unused query components are re-anchored until no new block is found;
   each iteration strictly consumes new bases, and accepted blocks stay fixed.
3. Each accepted block and each unambiguous collinear gap/tail is aligned
   globally with ``edlib`` (unit-cost edit distance). Unresolved gaps at
   rearrangement breakpoints remain unaligned and have covered=0. This is
   an anchored blockwise global alignment, not Shuffle-LAGAN or a global
   optimum over all possible rearrangements.

Identity definition, per reference window:
``identity = 100 * identical_reference_bases / window_span`` where a
reference position counts as identical when an edlib ``=`` column places the
same base from the query there; mismatches (``X``), deletions (``D``) and
unaligned reference positions count as non-identical. Insertions relative to
the reference (``I``) consume no reference position and do not enter the
window denominator. ``covered`` reports how many window positions received
an alignment column (``=`` + ``X`` + ``D``) so unaligned windows are
distinguishable from divergent ones.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .._bio import read_fasta
from .._sequtil import reverse_complement
from ..core.errors import OrganelleDependencyError, OrganelleInputError

#: mVISTA default sliding-window size along the reference, in bp.
DEFAULT_WINDOW_SIZE = 100
#: mVISTA y-axis display floor: windows below this identity are clipped.
IDENTITY_FLOOR = 50.0
#: mVISTA default: windows at or above this identity are "conserved".
CONSERVED_THRESHOLD = 70.0

_CIGAR_RUN = re.compile(r"(\d+)([MXID=])")
_MISSING_MESSAGE = (
    "{package!r} is required for genome identity computation. "
    "Install it with: pip install {package} (or pip install 'organelleverse[align]'); "
    "no Windows wheel is published, so use Linux, macOS or WSL."
)


def _require_mappy():
    try:
        import mappy as mp  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch tests
        raise OrganelleDependencyError(
            code="dependency.mappy",
            message=_MISSING_MESSAGE.format(package="mappy"),
            details={"package": "mappy", "install": "pip install mappy", "extra": "align"},
        ) from exc
    return mp


def _require_edlib():
    try:
        import edlib  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch tests
        raise OrganelleDependencyError(
            code="dependency.edlib",
            message=_MISSING_MESSAGE.format(package="edlib"),
            details={"package": "edlib", "install": "pip install edlib", "extra": "align"},
        ) from exc
    return edlib


# ---------------------------------------------------------------------------
# Anchoring (mappy / minimap2)
# ---------------------------------------------------------------------------


def _anchor_blocks(
    reference: str,
    query: str,
    *,
    preset: str,
    reference_unique: bool = True,
    include_cigar: bool = False,
) -> list[dict[str, Any]]:
    """Select one-to-one blocks from minimap2 alignments.

    Prefer mapq, then matching length. Trim overlaps along the actual CIGAR
    on BOTH genomes, retaining the unused portions of IR/inversion hits.
    No reference or query base may occur in two accepted blocks. Equal-score
    repeat copies are resolved deterministically by reference/query start;
    this is a coordinate assignment, not evidence of repeat-copy orthology.
    Re-anchor unused query components until no new bases can be assigned.
    With ``reference_unique=False``, only after this one-to-one phase is
    exhausted, map remaining query bases against already used reference
    positions to expose extra copies. Query positions remain unique.
    ``include_cigar`` retains trimmed M/I/D runs for structural evidence;
    M denotes a paired column, not necessarily an identical base.
    """
    mp = _require_mappy()
    aligner = mp.Aligner(seq=reference, preset=preset)
    reference_used = np.zeros(len(reference), dtype=bool)
    query_used = np.zeros(len(query), dtype=bool)
    blocks: list[dict[str, Any]] = []
    unique_phase = True
    query_intervals = [(0, len(query))]
    while query_intervals:
        hits = sorted(
            (
                (hit, start)
                for start, end in query_intervals
                for hit in aligner.map(query[start:end])
            ),
            key=lambda item: (
                -item[0].mapq,
                -item[0].mlen,
                not item[0].is_primary,
                item[0].r_st,
                item[1] + item[0].q_st,
                -item[0].strand,
            ),
        )
        previous_count = len(blocks)
        for hit, query_offset in hits:
            # CIGAR columns in increasing reference order; -1 denotes a gap.
            reference_columns: list[int] = []
            query_columns: list[int] = []
            r = hit.r_st
            q = query_offset + (hit.q_st if hit.strand == 1 else hit.q_en - 1)
            for length, op in hit.cigar:
                reference_columns.extend(range(r, r + length) if op != 1 else [-1] * length)
                query_columns.extend(
                    range(q, q + hit.strand * length, hit.strand) if op != 2 else [-1] * length
                )
                if op != 1:
                    r += length
                if op != 2:
                    q += hit.strand * length
            rs = np.asarray(reference_columns)
            qs = np.asarray(query_columns)
            available = np.ones(len(rs), dtype=bool)
            if unique_phase:
                available &= ~((rs >= 0) & reference_used[np.maximum(rs, 0)])
            available &= ~((qs >= 0) & query_used[np.maximum(qs, 0)])
            edges = np.diff(np.concatenate(([False], available, [False])).astype(int))
            for first, last in zip(
                np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
            ):
                # End each retained anchor on a column containing both bases.
                paired = np.flatnonzero((rs[first:last] >= 0) & (qs[first:last] >= 0))
                if not len(paired):
                    continue
                left, right = first + paired[0], first + paired[-1] + 1
                reference_start, reference_end = int(rs[left]), int(rs[right - 1]) + 1
                query_start = int(min(qs[left], qs[right - 1]))
                query_end = int(max(qs[left], qs[right - 1])) + 1
                reference_used[reference_start:reference_end] = True
                query_used[query_start:query_end] = True
                blocks.append(
                    {
                        "reference_start": reference_start,
                        "reference_end": reference_end,
                        "query_start": query_start,
                        "query_end": query_end,
                        "strand": int(hit.strand),
                        "mapq": int(hit.mapq),
                    }
                )
                if include_cigar:
                    cigar = []
                    for rr, qq in zip(rs[left:right], qs[left:right], strict=True):
                        op = "I" if rr < 0 else "D" if qq < 0 else "M"
                        if cigar and cigar[-1][1] == op:
                            cigar[-1][0] += 1
                        else:
                            cigar.append([1, op])
                    blocks[-1]["cigar"] = cigar
        if len(blocks) == previous_count:
            if not reference_unique and unique_phase:
                unique_phase = False
                continue
            break
        # Full-query chaining can hide a free IR copy behind a longer hit
        # to the other copy. Re-map only the remaining query components;
        # accepted bases never change. Stop when no new anchor is accepted.
        edges = np.diff(np.concatenate(([False], ~query_used, [False])).astype(int))
        query_intervals = list(
            zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
        )
    return sorted(blocks, key=lambda block: block["reference_start"])


# ---------------------------------------------------------------------------
# Global alignment of one piece (edlib) into reference-coordinate events
# ---------------------------------------------------------------------------


def _accumulate_piece_events(
    reference_piece: str,
    query_piece: str,
    reference_offset: int,
    matches: np.ndarray,
    mismatches: np.ndarray,
    deletions: np.ndarray,
) -> int:
    """Align one piece globally and tally per-reference-position events.

    ``edlib``'s extended CIGAR distinguishes ``=`` from ``X``; empty query
    pieces count every reference base as a deletion. Returns the number of
    edit operations (unused, for symmetry with older callers).
    """
    edlib = _require_edlib()
    size = len(reference_piece)
    if size == 0:
        return 0
    if len(query_piece) == 0:
        deletions[reference_offset : reference_offset + size] += 1
        return size
    result = edlib.align(query_piece, reference_piece, mode="NW", task="path")
    cigar = result["cigar"]
    if not cigar:  # pragma: no cover - edlib returns a CIGAR for non-empty pieces
        raise OrganelleInputError(
            code="identity.alignment_failed",
            message="edlib returned no CIGAR for a non-empty alignment piece",
            details={"reference_offset": reference_offset, "size": size},
        )
    position = reference_offset
    for length_text, op in _CIGAR_RUN.findall(cigar):
        length = int(length_text)
        if op == "=":
            matches[position : position + length] += 1
            position += length
        elif op == "X":
            mismatches[position : position + length] += 1
            position += length
        elif op == "D":
            deletions[position : position + length] += 1
            position += length
        # "I" consumes query bases only; reference positions are unaffected.
    if position != reference_offset + size:  # pragma: no cover - CIGAR/ref length invariant
        raise OrganelleInputError(
            code="identity.alignment_invariant",
            message="CIGAR reference extent does not match the reference piece",
            details={"expected_end": reference_offset + size, "actual_end": position},
        )
    return int(result["editDistance"])


def _oriented_query_piece(query: str, block: dict[str, int]) -> str:
    piece = query[block["query_start"] : block["query_end"]]
    return piece if block["strand"] == 1 else reverse_complement(piece)


def _piece_plan(
    reference: str,
    query: str,
    blocks: Sequence[dict[str, int]],
) -> list[tuple[str, str, int]]:
    """Globally align anchors and unambiguous collinear gaps/tails.

    A gap is alignable only when its flanking anchors have the same strand,
    occur in that order on the query, and contain no other accepted anchor.
    Tails extend only to the corresponding free query end. Rearrangement
    breakpoints without such a correspondence remain unaligned (covered=0),
    rather than reusing query bases or fabricating deletions.
    """
    pieces = [
        (
            reference[b["reference_start"] : b["reference_end"]],
            _oriented_query_piece(query, b),
            b["reference_start"],
        )
        for b in blocks
    ]
    query_order = sorted(blocks, key=lambda b: b["query_start"])
    ranks = {b["reference_start"]: i for i, b in enumerate(query_order)}
    for left, right in pairwise(blocks):
        strand = left["strand"]
        if right["strand"] != strand:
            continue
        if ranks[right["reference_start"]] - ranks[left["reference_start"]] != strand:
            continue
        low, high = (
            (left["query_end"], right["query_start"])
            if strand == 1
            else (right["query_end"], left["query_start"])
        )
        gap_query = query[low:high]
        if strand == -1:
            gap_query = reverse_complement(gap_query)
        pieces.append(
            (
                reference[left["reference_end"] : right["reference_start"]],
                gap_query,
                left["reference_end"],
            )
        )
    if blocks:
        first, last = blocks[0], blocks[-1]
        if first["strand"] == 1 and ranks[first["reference_start"]] == 0:
            pieces.append((reference[: first["reference_start"]], query[: first["query_start"]], 0))
        elif first["strand"] == -1 and ranks[first["reference_start"]] == len(blocks) - 1:
            pieces.append(
                (
                    reference[: first["reference_start"]],
                    reverse_complement(query[first["query_end"] :]),
                    0,
                )
            )
        if last["strand"] == 1 and ranks[last["reference_start"]] == len(blocks) - 1:
            pieces.append(
                (
                    reference[last["reference_end"] :],
                    query[last["query_end"] :],
                    last["reference_end"],
                )
            )
        elif last["strand"] == -1 and ranks[last["reference_start"]] == 0:
            pieces.append(
                (
                    reference[last["reference_end"] :],
                    reverse_complement(query[: last["query_start"]]),
                    last["reference_end"],
                )
            )
    return sorted((piece for piece in pieces if piece[0]), key=lambda piece: piece[2])


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def _window_rows(
    sample: str,
    matches: np.ndarray,
    mismatches: np.ndarray,
    deletions: np.ndarray,
    reference_length: int,
    window_size: int,
    step: int,
) -> list[dict[str, Any]]:
    cumulative_matches = np.concatenate(([0], np.cumsum(matches)))
    cumulative_mismatches = np.concatenate(([0], np.cumsum(mismatches)))
    cumulative_deletions = np.concatenate(([0], np.cumsum(deletions)))
    rows: list[dict[str, Any]] = []
    for start in range(0, reference_length, step):
        end = min(start + window_size, reference_length)
        span = end - start
        match_count = int(cumulative_matches[end] - cumulative_matches[start])
        mismatch_count = int(cumulative_mismatches[end] - cumulative_mismatches[start])
        deletion_count = int(cumulative_deletions[end] - cumulative_deletions[start])
        rows.append(
            {
                "sample": sample,
                "start": start,
                "end": end,
                "matches": match_count,
                "mismatches": mismatch_count,
                "deletions": deletion_count,
                "covered": match_count + mismatch_count + deletion_count,
                "identity": round(100.0 * match_count / span, 4),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# GenBank annotation -> mVISTA feature categories
# ---------------------------------------------------------------------------

_FEATURE_CATEGORY_BY_TYPE = {
    "cds": "cds",
    "trna": "nc_gene",
    "rrna": "nc_gene",
    "ncrna": "nc_gene",
    "misc_rna": "nc_gene",
    "tmrna": "nc_gene",
    "5'utr": "utr",
    "3'utr": "utr",
    "utr": "utr",
    "intron": "intron",
}
_ARROW_FEATURE_TYPES = frozenset(_FEATURE_CATEGORY_BY_TYPE)


def load_reference_features(
    genbank_path: str | Path, *, reference_sequence: str | None = None
) -> list[dict[str, Any]]:
    """Load reference gene features from a GenBank file as flat rows.

    One row per location part (so every value is a JSON primitive and the
    plotting side can rebuild multi-exon arrows via ``key``). ``category``
    is one of ``cds`` / ``utr`` / ``intron`` / ``nc_gene``; features of
    other types (``gene``, ``exon``, ``source``, ...) are skipped because
    they duplicate the CDS/RNA features used for the gene arrows.
    """
    from ..annotation.genbank import parse_genbank

    path = Path(genbank_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    document = parse_genbank(path)
    if len(document.records) != 1:
        raise OrganelleInputError(
            code="identity.reference_annotation",
            message="reference GenBank must contain exactly one record",
            details={"records": len(document.records), "path": str(path)},
        )
    record = document.records[0]
    if reference_sequence is not None and record.sequence.upper() != reference_sequence.upper():
        raise OrganelleInputError(
            code="identity.reference_annotation_mismatch",
            message="reference GenBank sequence must match the reference FASTA in the same orientation",
            details={"path": str(path)},
        )
    rows: list[dict[str, Any]] = []
    for feature in record.features:
        category = _FEATURE_CATEGORY_BY_TYPE.get(feature.type.casefold())
        if category is None:
            continue
        name = (
            feature.qualifier_values("gene")
            or feature.qualifier_values("product")
            or feature.qualifier_values("locus_tag")
            or (feature.feature_id,)
        )[0]
        strand = 1 if all(part.strand == 1 for part in feature.parts) else -1
        key = feature.feature_id
        parts = sorted(feature.parts, key=lambda part: part.start)
        for part_index, part in enumerate(parts):
            rows.append(
                {
                    "key": str(key),
                    "name": str(name),
                    "start": int(part.start),
                    "end": int(part.end),
                    "strand": int(strand),
                    "category": category,
                    "part_index": part_index,
                }
            )
    rows.sort(key=lambda row: (row["start"], row["end"], row["key"]))
    return rows


# ---------------------------------------------------------------------------
# Public compute
# ---------------------------------------------------------------------------


def _resolve_output_path(output: str | Path, default_name: str) -> Path:
    path = Path(output)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name


def _write_window_table(rows: Sequence[dict[str, Any]], output: str | Path) -> Path:
    path = _resolve_output_path(output, "genome_identity_windows.tsv")
    header = (
        "sample\treference_start\treference_end\tmatches\tmismatches\t"
        "deletions\tcovered\tidentity_percent\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['sample']}\t{row['start']}\t{row['end']}\t{row['matches']}\t"
            f"{row['mismatches']}\t{row['deletions']}\t{row['covered']}\t{row['identity']}\n"
        )
    path.write_text("".join(lines))
    return path


def compute_genome_identity(
    reference_fasta: str | Path,
    query_fastas: Sequence[str | Path],
    *,
    reference_genbank: str | Path | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int | None = None,
    preset: str = "asm10",
    table_output: str | Path | None = None,
) -> dict[str, Any]:
    """Compute mVISTA-style per-window identity of queries against one reference.

    ``reference_fasta`` and each query FASTA must contain exactly one
    record (a complete organelle genome). ``reference_genbank`` optionally provides the gene
    features for the mVISTA category track. Windows tile the reference every
    ``step`` bp (``step=None`` → non-overlapping ``window_size`` windows, the
    default here is 100 bp). ``preset`` is the minimap2 preset used for
    anchoring (``asm10`` fits same-family plastomes; ``asm5``/``asm20`` for
    closer/more diverged pairs). When ``table_output`` is given, the per-
    window table is also written as TSV (file path, or directory receiving
    ``genome_identity_windows.tsv``).

    mVISTA defaults honored here and by :func:`plot_genome_identity`:
    window size 100 bp, y-axis display floor ``IDENTITY_FLOOR`` = 50%, and
    ``CONSERVED_THRESHOLD`` = 70% marking conserved sequence.

    Returns a JSON-safe dict with ``reference_name``, ``reference_length``,
    ``samples``, per-window ``windows`` rows (``sample``/``start``/``end``/
    ``matches``/``mismatches``/``deletions``/``covered``/``identity`` percent),
    per-block ``segments``, and reference ``features`` rows (empty unless a
    GenBank was given).
    """
    if window_size < 1:
        raise OrganelleInputError(
            code="identity.window_size",
            message="window_size must be >= 1",
            details={"window_size": window_size},
        )
    if step is not None and step < 1:
        raise OrganelleInputError(
            code="identity.step",
            message="step must be >= 1",
            details={"step": step},
        )
    if not query_fastas:
        raise OrganelleInputError(
            code="identity.no_queries",
            message="compute_genome_identity requires at least one query FASTA",
        )
    resolved_step = step if step is not None else window_size

    reference_records = read_fasta(reference_fasta)
    if len(reference_records) != 1:
        raise OrganelleInputError(
            code="identity.reference_fasta",
            message="reference FASTA must contain exactly one record",
            details={"records": len(reference_records), "path": str(reference_fasta)},
        )
    reference_name, reference = reference_records[0]
    reference = reference.upper()
    reference_length = len(reference)

    features = (
        load_reference_features(reference_genbank, reference_sequence=reference)
        if reference_genbank is not None
        else []
    )

    samples: list[str] = []
    windows: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for query_path in query_fastas:
        path = Path(query_path)
        query_records = read_fasta(path)
        if len(query_records) != 1:
            raise OrganelleInputError(
                code="identity.query_fasta",
                message="each query FASTA must contain exactly one record",
                details={"records": len(query_records), "path": str(path)},
            )
        sample = path.stem
        if sample in samples:
            raise OrganelleInputError(
                code="identity.duplicate_sample",
                message="query FASTA file stems must be unique",
                details={"sample": sample},
            )
        query = query_records[0][1].upper()
        blocks = _anchor_blocks(reference, query, preset=preset)
        if not blocks:
            raise OrganelleInputError(
                code="identity.no_alignment",
                message="query genome has no minimap2 alignment to the reference",
                details={"sample": sample, "path": str(path)},
            )
        samples.append(sample)
        matches = np.zeros(reference_length, dtype=np.int64)
        mismatches = np.zeros(reference_length, dtype=np.int64)
        deletions = np.zeros(reference_length, dtype=np.int64)
        for reference_piece, query_piece, offset in _piece_plan(reference, query, blocks):
            _accumulate_piece_events(
                reference_piece, query_piece, offset, matches, mismatches, deletions
            )
        windows.extend(
            _window_rows(
                sample, matches, mismatches, deletions, reference_length, window_size, resolved_step
            )
        )
        for block in blocks:
            span = block["reference_end"] - block["reference_start"]
            block_matches = int(matches[block["reference_start"] : block["reference_end"]].sum())
            block_mismatches = int(
                mismatches[block["reference_start"] : block["reference_end"]].sum()
            )
            block_deletions = int(
                deletions[block["reference_start"] : block["reference_end"]].sum()
            )
            segments.append(
                {
                    "sample": sample,
                    "reference_start": block["reference_start"],
                    "reference_end": block["reference_end"],
                    "query_start": block["query_start"],
                    "query_end": block["query_end"],
                    "strand": block["strand"],
                    "mapq": block["mapq"],
                    "length": span,
                    "matches": block_matches,
                    "mismatches": block_mismatches,
                    "deletions": block_deletions,
                    "identity": round(100.0 * block_matches / span, 4),
                }
            )

    table_path: str | None = None
    if table_output is not None:
        table_path = str(_write_window_table(windows, table_output))
    return {
        "reference_name": str(reference_name),
        "reference_length": reference_length,
        "window_size": window_size,
        "step": resolved_step,
        "preset": preset,
        "samples": samples,
        "windows": windows,
        "segments": segments,
        "features": features,
        "table": table_path,
    }


__all__ = [
    "CONSERVED_THRESHOLD",
    "DEFAULT_WINDOW_SIZE",
    "IDENTITY_FLOOR",
    "compute_genome_identity",
    "load_reference_features",
]
