"""Native mitochondrial rRNA annotation from packaged references."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO
from Bio.Align import PairwiseAligner

from .db import DBManager
from .rrna import RawRRNA

_DNA_COMPLEMENT = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")


@dataclass(frozen=True)
class RRNAReference:
    gene_name: str
    rrna_type: str
    sequence: str


def annotate_rrna_native(
    fasta_path: Path,
    output_dir: Path,
    db_manager: DBManager,
    *,
    min_identity: float = 70.0,
    min_coverage: float = 50.0,
) -> list[RawRRNA]:
    """Detect mitochondrial rRNAs using packaged references without external tools."""
    output_dir.mkdir(parents=True, exist_ok=True)
    from .fasta import load_fasta

    try:
        genome = load_fasta(
            fasta_path
        ).sequence  # contigs joined by 200 N, the pipeline's coordinate space
    except ValueError:  # no records
        return []
    references = _load_rrna_references(db_manager.rrna_ref_dir)
    hits: list[RawRRNA] = []
    for ref in references:
        hits.extend(
            _scan_one_reference(
                genome,
                ref,
                strand=1,
                min_identity=min_identity,
                min_coverage=min_coverage,
            )
        )
        hits.extend(
            _scan_one_reference(
                _reverse_complement(genome),
                ref,
                strand=-1,
                min_identity=min_identity,
                min_coverage=min_coverage,
                genome_length=len(genome),
            )
        )

    return _deduplicate_hits(hits)


def _load_rrna_references(ref_dir: Path) -> list[RRNAReference]:
    if not ref_dir.exists():
        return []

    references: list[RRNAReference] = []
    for path in sorted(ref_dir.glob("*.fasta")):
        rrna_type = _rrna_type_from_name(path.stem)
        if rrna_type is None:
            continue
        gene_name = f"rrn{rrna_type.removesuffix('S')}"
        for record in SeqIO.parse(str(path), "fasta"):
            sequence = _clean_sequence(str(record.seq))
            if len(sequence) >= 50:
                references.append(RRNAReference(gene_name, rrna_type, sequence))
    return references


def _scan_one_reference(
    subject: str,
    reference: RRNAReference,
    *,
    strand: int,
    min_identity: float,
    min_coverage: float,
    genome_length: int | None = None,
) -> list[RawRRNA]:
    kmer_size = _kmer_size_for(reference.rrna_type, len(reference.sequence))
    diagonal_matches = _seed_diagonal_matches(subject, reference.sequence, kmer_size)
    diagonal_counts = Counter(
        {diagonal: len(matches) for diagonal, matches in diagonal_matches.items()}
    )
    if not diagonal_counts:
        return []

    min_seeds = _minimum_seed_count(reference.rrna_type, len(reference.sequence))
    hits: list[RawRRNA] = []
    processed_clusters: list[int] = []
    cluster_span = _cluster_span_for(reference.rrna_type)
    for diagonal, seed_count in diagonal_counts.most_common(80):
        if seed_count < min_seeds:
            break
        if any(abs(diagonal - center) <= cluster_span for center in processed_clusters):
            continue
        cluster_diagonals = [
            value
            for value, matches in diagonal_matches.items()
            if abs(value - diagonal) <= cluster_span and len(matches) >= 1
        ]
        cluster_matches = [
            match for value in cluster_diagonals for match in diagonal_matches[value]
        ]
        seed_coverage = _query_seed_coverage(cluster_matches, kmer_size, len(reference.sequence))
        if seed_coverage < min_coverage:
            continue
        processed_clusters.append(diagonal)

        scored = _refine_cluster_with_alignment(subject, reference.sequence, cluster_diagonals)
        if scored is None:
            continue
        start, end, identity, coverage, alignment_score = scored
        if identity < min_identity or coverage < min_coverage:
            continue
        if strand == -1:
            if genome_length is None:
                genome_length = len(subject)
            start, end = genome_length - end + 1, genome_length - start + 1
        hits.append(
            RawRRNA(
                gene_name=reference.gene_name,
                start=start,
                end=end,
                strand=strand,
                score=alignment_score,
                rrna_type=reference.rrna_type,
                source_tool="Native",
            )
        )
    return _deduplicate_hits(hits)


def _seed_diagonal_counts(subject: str, query: str, kmer_size: int) -> Counter[int]:
    return Counter(
        {
            diagonal: len(matches)
            for diagonal, matches in _seed_diagonal_matches(subject, query, kmer_size).items()
        }
    )


def _seed_diagonal_matches(
    subject: str, query: str, kmer_size: int
) -> dict[int, list[tuple[int, int]]]:
    query_index: dict[str, list[int]] = {}
    for q_pos in range(0, len(query) - kmer_size + 1):
        kmer = query[q_pos : q_pos + kmer_size]
        if "N" in kmer:
            continue
        query_index.setdefault(kmer, []).append(q_pos)

    matches_by_diagonal: dict[int, list[tuple[int, int]]] = {}
    for s_pos in range(0, len(subject) - kmer_size + 1):
        kmer = subject[s_pos : s_pos + kmer_size]
        for q_pos in query_index.get(kmer, []):
            matches_by_diagonal.setdefault(s_pos - q_pos, []).append((s_pos, q_pos))
    return matches_by_diagonal


def _score_diagonal(
    subject: str, query: str, diagonal: int
) -> tuple[int, int, float, float] | None:
    q_start = max(0, -diagonal)
    s_start = max(0, diagonal)
    length = min(len(query) - q_start, len(subject) - s_start)
    if length <= 0:
        return None

    subject_window = subject[s_start : s_start + length]
    query_window = query[q_start : q_start + length]
    matches = sum(1 for a, b in zip(subject_window, query_window, strict=True) if a == b)
    identity = matches / length * 100.0
    coverage = length / len(query) * 100.0
    return s_start + 1, s_start + length, identity, coverage


def _refine_cluster_with_alignment(
    subject: str,
    query: str,
    cluster_diagonals: list[int],
    padding: int = 250,
) -> tuple[int, int, float, float, float] | None:
    if not cluster_diagonals:
        return None
    window_start = max(0, min(cluster_diagonals) - padding)
    window_end = min(len(subject), max(cluster_diagonals) + len(query) + padding)
    if window_end <= window_start:
        return None

    window = subject[window_start:window_end]
    aligner = _new_rrna_aligner()
    alignments = aligner.align(query, window)
    if not alignments:
        return None
    alignment = alignments[0]
    coordinates = alignment.coordinates
    target_start = int(coordinates[0].min())
    target_end = int(coordinates[0].max())
    subject_start = int(coordinates[1].min())
    subject_end = int(coordinates[1].max())
    if target_end <= target_start or subject_end <= subject_start:
        return None

    matches = 0
    aligned_bases = 0
    for idx in range(coordinates.shape[1] - 1):
        q0, q1 = int(coordinates[0, idx]), int(coordinates[0, idx + 1])
        s0, s1 = int(coordinates[1, idx]), int(coordinates[1, idx + 1])
        if q1 <= q0 or s1 <= s0:
            continue
        segment_len = min(q1 - q0, s1 - s0)
        matches += sum(
            1 for offset in range(segment_len) if query[q0 + offset] == window[s0 + offset]
        )
        aligned_bases += segment_len

    if aligned_bases == 0:
        return None
    identity = matches / aligned_bases * 100.0
    coverage = (target_end - target_start) / len(query) * 100.0
    return (
        window_start + subject_start + 1,
        window_start + subject_end,
        identity,
        coverage,
        float(alignment.score),
    )


def _query_seed_coverage(
    matches: list[tuple[int, int]], kmer_size: int, query_length: int
) -> float:
    intervals = sorted((q_pos, min(query_length, q_pos + kmer_size)) for _, q_pos in matches)
    if not intervals:
        return 0.0
    covered = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    covered += current_end - current_start
    return covered / query_length * 100.0


def _deduplicate_hits(hits: list[RawRRNA]) -> list[RawRRNA]:
    kept: list[RawRRNA] = []
    for hit in sorted(hits, key=lambda h: (h.gene_name, -h.score, h.start, h.end)):
        if any(_same_locus(hit, existing) for existing in kept):
            continue
        kept.append(hit)
    return sorted(kept, key=lambda h: (h.start, h.end, h.gene_name, h.strand))


def _same_locus(a: RawRRNA, b: RawRRNA) -> bool:
    if a.gene_name != b.gene_name or a.strand != b.strand:
        return False
    overlap = max(0, min(a.end, b.end) - max(a.start, b.start) + 1)
    shorter = min(a.end - a.start + 1, b.end - b.start + 1)
    return shorter > 0 and overlap / shorter >= 0.5


def _kmer_size_for(rrna_type: str, length: int) -> int:
    if rrna_type == "5S":
        return min(15, max(9, length // 8))
    return 17


def _cluster_span_for(rrna_type: str) -> int:
    if rrna_type == "5S":
        return 80
    return 900


def _minimum_seed_count(rrna_type: str, length: int) -> int:
    if rrna_type == "5S":
        return 2
    return max(4, min(12, length // 350))


def _new_rrna_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    return aligner


def _rrna_type_from_name(name: str) -> str | None:
    lower = name.lower()
    if "rrn26" in lower or "26s" in lower or "lsu" in lower:
        return "26S"
    if "rrn18" in lower or "18s" in lower or "ssu" in lower:
        return "18S"
    if "rrn5" in lower or "5s" in lower:
        return "5S"
    return None


def _clean_sequence(seq: str) -> str:
    # Ambiguity codes become N: dropping them would shift every later coordinate.
    return "".join(
        base if base in "ACGT" else "N" for base in seq.upper().replace("U", "T") if not base.isspace()
    )


def _reverse_complement(seq: str) -> str:
    return seq.translate(_DNA_COMPLEMENT)[::-1]
