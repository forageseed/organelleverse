"""Typed cores for transfer (OmicVerse-style data-in/data-out).

``compute_transfer`` finds organelle-derived DNA (``source``) inside another
genome (``target``) by local alignment: LOSAT (vendored Rust BLAST) first, NCBI
``blastn`` second. Both run the sensitive ``blastn`` task (word size 11) because
transferred copies diverge: on real rice and Arabidopsis plastid-to-mitochondrion
transfers an exact-k-mer-run scan recovered only 59% / 29% of the bases found by
BLAST (recall falls from 93% for >=99% identity copies to 21% / 0% for 80-90%
identity copies), while megablast (word 28) lost 29% of the 80-90% identity
bases. The exact-run scan is kept only as an explicit last resort when no
aligner is installed; it is flagged by ``backend == "kmer_fallback"``.

Fragments are 1-based inclusive in the coordinates of each target record
(``seqid``), merged within a record when they overlap or touch.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .._bio import read_fasta
from .._sequtil import reverse_complement

#: ``blastn`` task word size. The NCBI default for the sensitive task; reaches
#: the 80-90% identity copies that megablast (28) misses.
WORD_SIZE = 11
DEFAULT_MIN_IDENTITY = 80.0
DEFAULT_EVALUE = 1e-5

_NCBI_FIELDS = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"


def _fasta_lengths(path: str | Path) -> dict[str, int]:
    from .._losat import fasta_sequence_lengths

    return fasta_sequence_lengths(path)


def _losat_rows(source: Path, target: Path, evalue: float) -> list[list[str]]:
    from .._losat import run_losat

    return run_losat(
        "blastn",
        source,
        target,
        task="blastn",
        word_size=WORD_SIZE,
        evalue=evalue,
        stage="transfer.compute_transfer",
    )


def _ncbi_rows(source: Path, target: Path, evalue: float, blastn: str) -> list[list[str]]:
    # ``-subject`` needs no makeblastdb; organelle-sized subjects are fine.
    done = subprocess.run(
        [
            blastn,
            "-task",
            "blastn",
            "-word_size",
            str(WORD_SIZE),
            "-dust",
            "no",
            "-evalue",
            str(evalue),
            "-query",
            str(source),
            "-subject",
            str(target),
            "-outfmt",
            _NCBI_FIELDS,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or f"blastn exited {done.returncode}")
    return [line.split("\t") for line in done.stdout.splitlines() if line.strip()]


def _hits(rows: list[list[str]], *, min_identity: float, min_len: int) -> list[tuple]:
    """Keep HSPs passing the thresholds as ``(seqid, lo, hi, identity, strand)`` on the target."""
    hits = []
    for cells in rows:
        if len(cells) < 12:
            continue
        try:
            identity, length = float(cells[2]), int(cells[3])
            sstart, send = int(cells[8]), int(cells[9])
        except ValueError:
            continue
        if identity < min_identity or length < min_len:
            continue
        strand = "+" if sstart <= send else "-"
        hits.append((cells[1], min(sstart, send), max(sstart, send), identity, strand))
    return hits


def _merge_hits(hits: list[tuple]) -> list[dict]:
    """Union overlapping/touching target intervals per record.

    Only the target coordinates are merged, so the result never claims donor
    coverage; identity is the span-weighted mean of the merged HSPs and the
    strand is the one covering more of the merged bases (``+`` donor-forward).
    """
    by_seq: dict[str, list[tuple]] = {}
    for hit in hits:
        by_seq.setdefault(hit[0], []).append(hit)
    fragments: list[dict] = []
    for seqid, group in by_seq.items():
        current: dict | None = None
        for _, lo, hi, identity, strand in sorted(group, key=lambda h: (h[1], h[2])):
            span = hi - lo + 1
            if current is not None and lo <= current["end"] + 1:
                current["end"] = max(current["end"], hi)
            else:
                if current is not None:
                    fragments.append(current)
                current = {
                    "seqid": seqid,
                    "start": lo,
                    "end": hi,
                    "_ident": 0.0,
                    "_span": 0,
                    "_strand": {"+": 0, "-": 0},
                }
            current["_ident"] += identity * span
            current["_span"] += span
            current["_strand"][strand] += span
        if current is not None:
            fragments.append(current)
    for f in fragments:
        f["length"] = f["end"] - f["start"] + 1
        f["identity"] = round(f.pop("_ident") / f.pop("_span"), 2)
        strands = f.pop("_strand")
        f["strand"] = "-" if strands["-"] > strands["+"] else "+"
    return fragments


def alignment_transfer(
    target_fasta: str | Path,
    source_fasta: str | Path,
    *,
    min_len: int = 100,
    min_identity: float = DEFAULT_MIN_IDENTITY,
    evalue: float = DEFAULT_EVALUE,
    target_lengths: dict[str, int] | None = None,
) -> dict | None:
    """Align ``source`` against ``target``; ``None`` when no aligner could run.

    Returns ``{"fragments", "backend", "attempted", "target_bp", "target_records"}``.
    ``target_lengths`` (record id -> length) skips re-scanning a large target. LOSAT is
    tried first, NCBI ``blastn`` second; a backend that is missing or fails is
    recorded in ``attempted`` and the next one is tried.
    """
    from .._losat import resolve_losat
    from ..core.errors import OrganelleDependencyError, OrganelleExecutionError

    target, source = Path(target_fasta), Path(source_fasta)
    attempted: list[str] = []
    rows: list[list[str]] | None = None
    backend = ""
    if resolve_losat() is not None:
        attempted.append("losat")
        try:
            rows = _losat_rows(source, target, evalue)
            backend = "losat"
        except (OrganelleDependencyError, OrganelleExecutionError):
            rows = None
    if rows is None:
        blastn = shutil.which("blastn")
        if blastn:
            attempted.append("ncbi_blastn")
            try:
                rows = _ncbi_rows(source, target, evalue, blastn)
                backend = "ncbi_blastn"
            except (RuntimeError, OSError):
                rows = None
    if rows is None:
        return None
    fragments = _merge_hits(_hits(rows, min_identity=min_identity, min_len=min_len))
    lengths = target_lengths if target_lengths is not None else _fasta_lengths(target)
    return {
        "fragments": fragments,
        "backend": backend,
        "attempted": attempted,
        "target_bp": sum(lengths.values()),
        "target_records": len(lengths),
    }


def compute_transfer(
    target_fasta: str | Path,
    source_fasta: str | Path,
    k: int = 31,
    min_len: int = 100,
    min_identity: float = DEFAULT_MIN_IDENTITY,
) -> dict:
    """Detect source-derived transfer fragments in the target. Returns dict.

    Local alignment (LOSAT, then NCBI ``blastn``; word size 11, e-value 1e-5)
    keeping HSPs with >= ``min_len`` alignment columns and >= ``min_identity``
    percent identity on either strand. Reports fragment count, total bp,
    per-strand bp, the fraction of the target covered, and ``backend``. Each
    fragment carries ``seqid``, 1-based inclusive ``start``/``end`` in that
    target record, ``strand`` (``-`` = integrated as reverse complement) and
    ``identity``. ``k`` only applies to the exact-k-mer last-resort fallback
    used when neither aligner is installed (``backend == "kmer_fallback"``).
    """
    found = alignment_transfer(target_fasta, source_fasta, min_len=min_len, min_identity=min_identity)
    if found is None:
        result = _compute_transfer_kmer(target_fasta, source_fasta, k, min_len)
        result["backend"] = "kmer_fallback"
        return result
    fragments = found["fragments"]
    total_bp = sum(f["length"] for f in fragments)
    reverse_bp = sum(f["length"] for f in fragments if f["strand"] == "-")
    return {
        "fragment_count": len(fragments),
        "total_bp": total_bp,
        "direct_strand_bp": total_bp - reverse_bp,
        "reverse_strand_bp": reverse_bp,
        "fraction": round(total_bp / found["target_bp"], 6) if found["target_bp"] else 0.0,
        "fragments": fragments,
        "backend": found["backend"],
    }


def _compute_transfer_kmer(
    target_fasta: str | Path, source_fasta: str | Path, k: int = 31, min_len: int = 100
) -> dict:
    """Last resort without an aligner: dual-strand contiguous exact k-mer runs.

    Only near-identical copies survive (see the module docstring), so a result
    from this path under-reports diverged transfers.
    """
    target = "".join(s.upper() for _, s in read_fasta(Path(target_fasta)))
    source = "".join(s.upper() for _, s in read_fasta(Path(source_fasta)))
    if len(target) < k or len(source) < k:
        return {
            "fragment_count": 0,
            "total_bp": 0,
            "direct_strand_bp": 0,
            "reverse_strand_bp": 0,
            "fraction": 0.0,
            "fragments": [],
        }

    # Forward + reverse-complement k-mer index of the source.
    forward_index: set[str] = set()
    rev_index: set[str] = set()
    for i in range(len(source) - k + 1):
        km = source[i : i + k]
        if "N" in km:
            continue
        forward_index.add(km)
        rev_index.add(reverse_complement(km))
    combined = forward_index | rev_index

    fragments = []
    run_start = None
    last_match_end = 0
    n = len(target)
    for i in range(n - k + 1):
        km = target[i : i + k]
        if km in combined:
            if run_start is None:
                run_start = i
            last_match_end = i + k
        elif run_start is not None:
            run_len = last_match_end - run_start
            if run_len >= min_len:
                fragments.append({"start": run_start + 1, "end": last_match_end, "length": run_len})
            run_start = None
    if run_start is not None:
        run_len = last_match_end - run_start
        if run_len >= min_len:
            fragments.append({"start": run_start + 1, "end": last_match_end, "length": run_len})

    direct_bp = reverse_bp = 0
    for f in fragments:
        head = target[f["start"] - 1 : f["start"] - 1 + k]
        if head in forward_index:
            f["strand"] = "+"
            direct_bp += f["length"]
        elif head in rev_index:
            f["strand"] = "-"
            reverse_bp += f["length"]
        else:
            f["strand"] = "?"
            direct_bp += f["length"]

    total_bp = sum(f["length"] for f in fragments)
    return {
        "fragment_count": len(fragments),
        "total_bp": total_bp,
        "direct_strand_bp": direct_bp,
        "reverse_strand_bp": reverse_bp,
        "fraction": round(total_bp / len(target), 6) if target else 0,
        "fragments": fragments,
    }
