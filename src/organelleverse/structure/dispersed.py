"""Dispersed (interspersed) repeats from a whole-genome self-alignment.

The genome is aligned against itself with minimap2 (``-x asm20 -DP --eqx -c``,
the setting of Brown et al. 2026 for 300+ plant mitogenomes), or with LOSAT /
NCBI blastn when minimap2 is absent. Hits are kept when the alignment is
>= ``min_length`` columns at >= ``min_identity`` percent identity; a hit of a
sequence onto itself within ``diagonal_band`` bp of the diagonal on the same
strand is the trivial self-match and is dropped. The query and target sides of every kept hit are
merged per sequence, and the repeat fraction is the merged base count over
the genome length.

The filtering and merging are pure functions (:func:`filter_self_hits`,
:func:`repeat_landscape`) so they are testable without an aligner.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from organelleverse._public_validation import validated_public

from .._bio import read_fasta
from ..core.errors import OrganelleDependencyError, OrganelleInputError, OrganelleParameterError
from ..core.result import Finding, OrganelleResult
from ..sequences import Sequences, as_path
from .structure import _OPERATION_VERSION, _provenance, _sha256_file

__all__ = ["SelfHit", "dispersed_repeats", "filter_self_hits", "repeat_landscape"]

# Repeat length classes used for plant mitogenome recombination: short repeats
# rarely recombine, intermediate ones do so ectopically, large ones (> 1 kb)
# recombine frequently and give equimolar isoforms.
_LENGTH_CLASSES = ((100, 1000, "100-1000"), (1000, 5000, "1000-5000"), (5000, None, ">=5000"))
_MAX_PAIRS_REPORTED = 20_000


@dataclass(frozen=True)
class SelfHit:
    """One self-alignment hit, 1-based inclusive on both sides."""

    query: str
    qstart: int
    qend: int
    target: str
    tstart: int
    tend: int
    strand: str  # "+" direct, "-" inverted
    identity: float  # percent
    length: int  # alignment columns
    mapq: int | None = None


def filter_self_hits(
    hits: Iterable[SelfHit],
    *,
    min_length: int,
    min_identity: float,
    min_mapq: int | None,
    diagonal_band: int,
) -> list[SelfHit]:
    """Keep real repeat hits and collapse reciprocal duplicates (A->B and B->A)."""
    kept: dict[tuple, SelfHit] = {}
    for hit in hits:
        if hit.length < min_length or hit.identity < min_identity:
            continue
        if min_mapq is not None and hit.mapq is not None and hit.mapq < min_mapq:
            continue
        if (
            hit.query == hit.target
            and hit.strand == "+"
            and abs(hit.qstart - hit.tstart) <= diagonal_band
        ):
            continue
        a = (hit.query, hit.qstart, hit.qend)
        b = (hit.target, hit.tstart, hit.tend)
        key = (min(a, b), max(a, b), hit.strand)
        if key not in kept or hit.length > kept[key].length:
            kept[key] = hit
    return sorted(kept.values(), key=lambda h: (-h.length, h.query, h.qstart))


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def repeat_landscape(hits: Sequence[SelfHit], lengths: dict[str, int]) -> dict[str, object]:
    """Merged repeat bases per sequence and genome-wide, plus pair summaries."""
    spans: dict[str, list[tuple[int, int]]] = {name: [] for name in lengths}
    for hit in hits:
        spans.setdefault(hit.query, []).append((hit.qstart, hit.qend))
        spans.setdefault(hit.target, []).append((min(hit.tstart, hit.tend), max(hit.tstart, hit.tend)))
    per_sequence = []
    repeat_bases = 0
    for name, length in lengths.items():
        bases = sum(end - start + 1 for start, end in _merge(spans.get(name, [])))
        repeat_bases += bases
        per_sequence.append(
            {
                "sequence_id": name,
                "length": length,
                "repeat_bases": bases,
                "repeat_fraction": bases / length if length else 0.0,
            }
        )
    genome_length = sum(lengths.values())
    classes = {label: 0 for _, _, label in _LENGTH_CLASSES}
    for hit in hits:
        for low, high, label in _LENGTH_CLASSES:
            if hit.length >= low and (high is None or hit.length < high):
                classes[label] += 1
    return {
        "genome_length": genome_length,
        "repeat_bases": repeat_bases,
        "repeat_fraction": repeat_bases / genome_length if genome_length else 0.0,
        "n_repeat_pairs": len(hits),
        "n_direct": sum(1 for hit in hits if hit.strand == "+"),
        "n_inverted": sum(1 for hit in hits if hit.strand == "-"),
        "longest_repeat": max((hit.length for hit in hits), default=0),
        "pairs_by_length": classes,
        "per_sequence": per_sequence,
    }


def _parse_paf(lines: Iterable[str]) -> Iterable[SelfHit]:
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 12:
            continue
        qlen_cols = int(fields[10])
        matches = int(fields[9])
        strand = fields[4]
        tstart, tend = int(fields[7]) + 1, int(fields[8])
        yield SelfHit(
            query=fields[0],
            qstart=int(fields[2]) + 1,
            qend=int(fields[3]),
            target=fields[5],
            tstart=tstart,
            tend=tend,
            strand=strand,
            identity=100.0 * matches / qlen_cols if qlen_cols else 0.0,
            length=qlen_cols,
            mapq=int(fields[11]),
        )


def _parse_blast(rows: Iterable[Sequence[str]]) -> Iterable[SelfHit]:
    for row in rows:
        qstart, qend = int(row[6]), int(row[7])
        sstart, send = int(row[8]), int(row[9])
        strand = "+" if sstart <= send else "-"
        yield SelfHit(
            query=row[0],
            qstart=min(qstart, qend),
            qend=max(qstart, qend),
            target=row[1],
            tstart=min(sstart, send),
            tend=max(sstart, send),
            strand=strand,
            identity=float(row[2]),
            length=int(row[3]),
        )


def _run_minimap2(executable: str, fasta: Path, threads: int) -> tuple[list[SelfHit], list[str]]:
    argv = [executable, "-x", "asm20", "-DP", "--eqx", "-c", "-t", str(threads), str(fasta), str(fasta)]
    process = subprocess.run(argv, capture_output=True, text=True)
    if process.returncode != 0:
        raise OrganelleInputError(
            code="structure.minimap2_failed",
            message=process.stderr.strip()[-1000:] or "minimap2 failed",
            details={"argv": argv},
        )
    return list(_parse_paf(process.stdout.splitlines())), argv


def _run_blastn(executable: str, fasta: Path) -> tuple[list[SelfHit], list[str]]:
    argv = [
        executable, "-task", "blastn", "-query", str(fasta), "-subject", str(fasta),
        "-evalue", "1e-10", "-outfmt", "6", "-dust", "no",
    ]
    process = subprocess.run(argv, capture_output=True, text=True)
    if process.returncode != 0:
        raise OrganelleInputError(
            code="structure.blastn_failed",
            message=process.stderr.strip()[-1000:] or "blastn failed",
            details={"argv": argv},
        )
    rows = [line.split("\t") for line in process.stdout.splitlines() if line.strip()]
    return list(_parse_blast(rows)), argv


def _run_losat(fasta: Path) -> tuple[list[SelfHit], list[str]]:
    from .._losat import run_losat

    rows = run_losat("blastn", fasta, fasta, evalue=1e-10, task="blastn", word_size=11)
    return list(_parse_blast(rows)), ["losat", "blastn", "-task", "blastn", "-word_size", "11"]


@validated_public("structure.dispersed_repeats")
def dispersed_repeats(
    genome_fasta: str | Path | Sequences,
    *,
    min_length: int = 100,
    min_identity: float = 90.0,
    min_mapq: int = 0,
    diagonal_band: int = 50,
    backend: Literal["auto", "minimap2", "losat", "blastn"] = "auto",
    threads: int = 1,
) -> OrganelleResult:
    """Fraction of a genome in dispersed repeats, from a whole-genome self-alignment.

    ``backend="auto"`` tries minimap2 (the published setting), then LOSAT, then
    NCBI blastn; a named backend that is not installed is an error, never a
    silent switch. ``min_mapq`` applies to minimap2 only and is off by
    default: in a self-alignment the two copies of a repeat are each other's
    equal best hit, so minimap2 gives every repeat MAPQ 0 and the published
    "MAPQ > 0" filter would discard them all.

    Metrics: ``repeat_fraction`` and ``repeat_bases`` genome-wide and per
    sequence, direct/inverted pair counts, pairs by length class (100-1000,
    1000-5000, >=5000 bp) and ``repeats`` rows (longest first; ``positions``
    and ``length`` match ``visualization.plot_structure_map``'s
    ``dispersed_repeats`` track).
    """
    if min_length < 1 or not 0 < min_identity <= 100 or diagonal_band < 0 or threads < 1:
        raise OrganelleParameterError(
            code="structure.dispersed_parameters",
            message="need min_length >= 1, 0 < min_identity <= 100, diagonal_band >= 0, threads >= 1",
        )
    genome_fasta = as_path(genome_fasta)
    fasta = Path(genome_fasta)
    started_at = datetime.now(UTC)
    records = read_fasta(fasta)
    if not records:
        raise OrganelleInputError(code="structure.empty_fasta", message="the FASTA has no sequences")
    lengths = {name: len(sequence) for name, sequence in records}

    candidates = ["minimap2", "losat", "blastn"] if backend == "auto" else [backend]
    used = None
    raw: list[SelfHit] = []
    argv: list[str] = []
    for name in candidates:
        if name == "minimap2" and (executable := shutil.which("minimap2")):
            raw, argv = _run_minimap2(executable, fasta, threads)
        elif name == "blastn" and (executable := shutil.which("blastn")):
            raw, argv = _run_blastn(executable, fasta)
        elif name == "losat":
            from .._losat import resolve_losat

            if resolve_losat() is None:
                continue
            # LOSAT rejects lowercase-only input quirks; feed an uppercased copy.
            with tempfile.TemporaryDirectory(prefix="ov_dispersed_") as tmp:
                upper = Path(tmp) / "genome.fa"
                upper.write_text("".join(f">{n}\n{s.upper()}\n" for n, s in records))
                raw, argv = _run_losat(upper)
        else:
            continue
        used = name
        break
    if used is None:
        raise OrganelleDependencyError(
            code="structure.dispersed_backend_missing",
            message=f"no self-alignment backend available ({', '.join(candidates)})",
            details={"tried": candidates},
            suggested_action={"install": "mamba install -c bioconda minimap2"},
        )

    hits = filter_self_hits(
        raw,
        min_length=min_length,
        min_identity=min_identity,
        min_mapq=min_mapq if used == "minimap2" else None,
        diagonal_band=diagonal_band,
    )
    landscape = repeat_landscape(hits, lengths)
    rows = [
        {
            "sequence_id": hit.query,
            "start": hit.qstart,
            "end": hit.qend,
            "partner_sequence_id": hit.target,
            "partner_start": hit.tstart,
            "partner_end": hit.tend,
            "orientation": "direct" if hit.strand == "+" else "inverted",
            "identity": round(hit.identity, 2),
            "length": min(hit.qend - hit.qstart + 1, hit.tend - hit.tstart + 1),
            "alignment_length": hit.length,
            "positions": [hit.qstart, hit.tstart],
        }
        for hit in hits[:_MAX_PAIRS_REPORTED]
    ]
    flags = []
    if len(hits) > _MAX_PAIRS_REPORTED:
        flags.append("repeat_rows_truncated")
    if len(records) > 1:
        flags.append("multi_sequence_genome")
    fraction = float(landscape["repeat_fraction"])
    return OrganelleResult(
        operation_id="structure.dispersed_repeats",
        operation_version=_OPERATION_VERSION,
        scope="mitochondrion",
        status="ok",
        summary_text=(
            f"{fraction * 100:.2f}% of {landscape['genome_length']:,} bp in dispersed repeats "
            f"({landscape['n_repeat_pairs']} pairs >= {min_length} bp at >= {min_identity:g}% "
            f"identity; longest {landscape['longest_repeat']:,} bp; {used})."
        ),
        metrics={**landscape, "backend": used, "repeats": rows},
        findings=(
            Finding(code="structure.repeat_fraction", metric="repeat_fraction", value=round(fraction, 6)),
            Finding(
                code="structure.repeat_pairs", metric="n_repeat_pairs", value=landscape["n_repeat_pairs"]
            ),
        ),
        flags=tuple(flags),
        provenance=_provenance(
            "structure.dispersed_repeats",
            parameters={
                "min_length": min_length,
                "min_identity": min_identity,
                "min_mapq": min_mapq,
                "diagonal_band": diagonal_band,
                "backend": backend,
                "argv": argv,
            },
            input_artifact_hashes=(_sha256_file(fasta),),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
    )
