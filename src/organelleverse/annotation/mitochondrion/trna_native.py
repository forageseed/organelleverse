"""Native mitochondrial tRNA annotation from packaged references."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from Bio import SeqIO
from Bio.Align import PairwiseAligner

from organelleverse.annotation.trna_core.candidates import scan_candidate_windows
from organelleverse.annotation.trna_core.covariance import best_candidate_features, score_candidate
from organelleverse.annotation.trna_core.debug import write_debug_candidates
from organelleverse.annotation.trna_core.filters import apply_overcall_density_filter
from organelleverse.annotation.trna_core.models import TRNACall, TRNAFeatureSet

from .db import DBManager
from .trna import RawTRNA, _parse_trna_name, _standardize_trna_name

if TYPE_CHECKING:
    from organelleverse.annotation.trna_core.models import TRNACandidate

_DNA_COMPLEMENT = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")


def _merged_genome(fasta_path: Path) -> str:
    """The pipeline's genome string: contigs joined by 200 N, as ``load_fasta`` builds it.

    Every later stage and writer reads coordinates in this space; concatenating the
    records without the spacers put tRNAs on the second and later contigs 200 x k bp
    to the left of where they belong.
    """
    from .fasta import load_fasta

    try:
        return load_fasta(fasta_path).sequence
    except ValueError:  # no records
        return ""


@dataclass(frozen=True)
class TRNAReference:
    gene_name: str
    anticodon: str
    amino_acid: str
    sequence: str


# A plant mitochondrial tRNA call must look like a whole mature tRNA. In seven
# RefSeq mitogenomes every annotated tRNA found was 70-88 nt and scored >=37
# bits, while 21 of 51 extra calls were truncated fragments (43-63 nt) or
# weak/misaligned hits (<30 bits, or 120 nt). Full-length plastid-derived tRNAs
# RefSeq leaves unannotated pass these bounds.
_MATURE_TRNA_LENGTH = (65, 95)
_MIN_REPORTED_BITS = 30.0


def annotate_trna_cm(
    fasta_path: Path,
    output_dir: Path,
    db_manager: DBManager | None = None,
    *,
    min_bits: float = 20.0,
) -> list[RawTRNA]:
    """Native tRNA annotation via the covariance-model genome scanner.

    Uses the packaged (plant-mito) CM + HMM filter for coordinate-exact
    detection, then the CYK traceback to read the anticodon for gene naming.
    Unnamed hits (atypical/divergent tRNAs where the anticodon columns are not
    aligned) are kept with a generic ``tRNA`` name rather than a wrong one.
    """
    from ..cmsearch.genome import default_cm_path, find_trnas
    from ..cmsearch.model import parse_cm
    from ..cmsearch.naming import name_trna

    genome = _merged_genome(fasta_path).replace("U", "T")
    if not genome:
        return []
    model = parse_cm(default_cm_path())

    hits = find_trnas(genome, cm_model=model, min_bits=min_bits)
    results: list[RawTRNA] = []
    for hit in hits:
        length = hit.end - hit.start + 1
        if hit.score < _MIN_REPORTED_BITS or not (
            _MATURE_TRNA_LENGTH[0] <= length <= _MATURE_TRNA_LENGTH[1]
        ):
            continue
        mature = genome[hit.start - 1 : hit.end]
        if hit.strand == -1:
            mature = mature.translate(_DNA_COMPLEMENT)[::-1]
        named = name_trna(model, mature)
        if named is not None:
            anticodon, amino_acid, gene = named
        else:
            anticodon, amino_acid, gene = "", "", "tRNA"
        results.append(
            RawTRNA(
                gene_name=gene,
                start=hit.start,
                end=hit.end,
                strand=hit.strand,
                anticodon=anticodon.upper(),
                amino_acid=amino_acid,
                score=hit.score,
                source="NativeCM",
            )
        )
    return results


def annotate_trna_native(
    fasta_path: Path,
    output_dir: Path,
    db_manager: DBManager,
    *,
    threads: int = 1,
    min_identity: float = 70.0,
    min_coverage: float = 85.0,
    min_seed_hits: int = 3,
) -> list[RawTRNA]:
    """Detect tRNAs using packaged references without BLAST or external tools."""
    del threads
    output_dir.mkdir(parents=True, exist_ok=True)
    genome = _merged_genome(fasta_path)
    if not genome:
        return []
    references = _load_trna_references(db_manager.trna_ref_dir)
    if not references:
        return []

    forward_index = _build_kmer_index(genome, 12)
    reverse = _reverse_complement(genome)
    reverse_index = _build_kmer_index(reverse, 12)

    hits: list[RawTRNA] = []
    for reference in references:
        hits.extend(
            _scan_reference(
                genome,
                forward_index,
                reference,
                strand=1,
                min_identity=min_identity,
                min_coverage=min_coverage,
                min_seed_hits=min_seed_hits,
            )
        )
        hits.extend(
            _scan_reference(
                reverse,
                reverse_index,
                reference,
                strand=-1,
                min_identity=min_identity,
                min_coverage=min_coverage,
                min_seed_hits=min_seed_hits,
                genome_length=len(genome),
            )
        )

    return _deduplicate_trna_hits(hits)


def annotate_trna_cleanroom_experimental(
    fasta_path: Path,
    output_dir: Path,
    db_manager: DBManager,
    *,
    threads: int = 1,
    debug: bool = False,
) -> list[RawTRNA]:
    """Run the experimental clean-room tRNA engine without changing defaults."""
    del threads
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        return []

    genome = "".join(str(record.seq).upper().replace("U", "T") for record in records)
    if len(genome) <= 100_000:
        candidates = scan_candidate_windows(genome)
    else:
        candidates = _cleanroom_candidates_from_native_hits(
            fasta_path, output_dir, db_manager, genome
        )
    scored = []
    for candidate in candidates:
        score = score_candidate(candidate)
        features = best_candidate_features(candidate)
        scored.append((candidate, score, features))

    features_by_candidate = {candidate: features for candidate, _, features in scored}
    decisions = apply_overcall_density_filter(scored)
    calls = [
        TRNACall(
            candidate=candidate,
            features=features_by_candidate.get(candidate) or TRNAFeatureSet(anticodon_offset=None),
            score=score,
            decision=decision,
        )
        for candidate, score, decision in decisions
    ]
    if debug:
        write_debug_candidates(output_dir / "cleanroom_trna_candidates.tsv", calls)

    hits = []
    for call in calls:
        if not call.decision.passed:
            continue
        candidate = call.candidate
        hits.append(
            RawTRNA(
                gene_name=candidate.gene_name,
                start=candidate.start,
                end=candidate.end,
                strand=candidate.strand,
                anticodon=candidate.anticodon.upper(),
                amino_acid=candidate.amino_acid,
                score=call.score.total,
                source="NativeCleanRoom",
                intron_start=candidate.intron_start,
                intron_end=candidate.intron_end,
            )
        )
    return _deduplicate_trna_hits(hits)


def _cleanroom_candidates_from_native_hits(
    fasta_path: Path,
    output_dir: Path,
    db_manager: DBManager,
    genome: str,
) -> list[TRNACandidate]:
    from organelleverse.annotation.trna_core.models import TRNACandidate

    raw_hits = annotate_trna_native(fasta_path, output_dir / "native_seed", db_manager)
    candidates = []
    for hit in raw_hits:
        sequence = _extract_genome_span(genome, hit.start, hit.end, hit.strand)
        if not sequence:
            continue
        try:
            candidates.append(
                TRNACandidate(
                    sequence=sequence,
                    start=hit.start,
                    end=hit.end,
                    strand=hit.strand,
                    anticodon=hit.anticodon,
                    amino_acid=hit.amino_acid,
                    source="native_cleanroom_seeded",
                )
            )
        except ValueError:
            continue
    return candidates


def _extract_genome_span(genome: str, start: int, end: int, strand: int) -> str:
    if start < 1 or end < start or end > len(genome):
        return ""
    fragment = genome[start - 1 : end]
    if strand < 0:
        return _reverse_complement(fragment)
    return fragment


def _load_trna_references(ref_dir: Path) -> list[TRNAReference]:
    if not ref_dir.exists():
        return []

    references: list[TRNAReference] = []
    for path in sorted(list(ref_dir.glob("*.fasta")) + list(ref_dir.glob("*.fa"))):
        for record in SeqIO.parse(str(path), "fasta"):
            sequence = _clean_sequence(str(record.seq))
            if not 45 <= len(sequence) <= 120:
                continue
            amino_acid, anticodon_raw = _parse_trna_name(record.id)
            anticodon = anticodon_raw.upper().replace("U", "T")
            if "N" in anticodon:
                continue
            references.append(
                TRNAReference(
                    gene_name=_standardize_trna_name(amino_acid, anticodon),
                    anticodon=anticodon,
                    amino_acid=amino_acid,
                    sequence=sequence,
                )
            )
    return references


def _scan_reference(
    subject: str,
    subject_index: dict[str, list[int]],
    reference: TRNAReference,
    *,
    strand: int,
    min_identity: float,
    min_coverage: float,
    min_seed_hits: int,
    genome_length: int | None = None,
) -> list[RawTRNA]:
    kmer_size = 12
    counts = _seed_diagonal_counts(subject_index, reference.sequence, kmer_size)
    if not counts:
        return []

    hits: list[RawTRNA] = []
    processed: list[int] = []
    for diagonal, seed_count in counts.most_common(10):
        if seed_count < min_seed_hits:
            break
        if any(abs(diagonal - prev) <= 30 for prev in processed):
            continue
        processed.append(diagonal)
        cluster = [
            value for value, count in counts.items() if abs(value - diagonal) <= 30 and count >= 1
        ]
        scored = _refine_trna_alignment(subject, reference.sequence, cluster)
        if scored is None:
            continue
        start, end, identity, coverage, score = scored
        length = end - start + 1
        if identity < min_identity or coverage < min_coverage or not 45 <= length <= 130:
            continue
        if strand == -1:
            if genome_length is None:
                genome_length = len(subject)
            start, end = genome_length - end + 1, genome_length - start + 1
        hits.append(
            RawTRNA(
                gene_name=reference.gene_name,
                start=start,
                end=end,
                strand=strand,
                anticodon=reference.anticodon,
                amino_acid=reference.amino_acid,
                score=score,
                source="Native",
            )
        )
    return hits


def _build_kmer_index(sequence: str, kmer_size: int) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for pos in range(0, len(sequence) - kmer_size + 1):
        kmer = sequence[pos : pos + kmer_size]
        if "N" in kmer:
            continue
        index.setdefault(kmer, []).append(pos)
    return {kmer: positions for kmer, positions in index.items() if len(positions) <= 500}


def _seed_diagonal_counts(
    subject_index: dict[str, list[int]], query: str, kmer_size: int
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for q_pos in range(0, len(query) - kmer_size + 1, 2):
        kmer = query[q_pos : q_pos + kmer_size]
        if "N" in kmer:
            continue
        for s_pos in subject_index.get(kmer, []):
            counts[s_pos - q_pos] += 1
    return counts


def _refine_trna_alignment(
    subject: str,
    query: str,
    cluster_diagonals: list[int],
    padding: int = 20,
) -> tuple[int, int, float, float, float] | None:
    if not cluster_diagonals:
        return None
    window_start = max(0, min(cluster_diagonals) - padding)
    window_end = min(len(subject), max(cluster_diagonals) + len(query) + padding)
    if window_end <= window_start:
        return None

    window = subject[window_start:window_end]
    aligner = _new_trna_aligner()
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


def _deduplicate_trna_hits(hits: list[RawTRNA]) -> list[RawTRNA]:
    kept: list[RawTRNA] = []
    for hit in sorted(hits, key=lambda item: (item.score, item.end - item.start + 1), reverse=True):
        if any(_same_locus(hit, existing) for existing in kept):
            continue
        kept.append(hit)
    return sorted(kept, key=lambda item: (item.start, item.end, item.gene_name))


def _same_locus(a: RawTRNA, b: RawTRNA) -> bool:
    overlap = max(0, min(a.end, b.end) - max(a.start, b.start) + 1)
    shorter = min(a.end - a.start + 1, b.end - b.start + 1)
    return shorter > 0 and overlap / shorter >= 0.5


@lru_cache(maxsize=1)
def _new_trna_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -4
    aligner.extend_gap_score = -1
    return aligner


def _clean_sequence(seq: str) -> str:
    # Ambiguity codes become N: dropping them would shift every later coordinate.
    return "".join(
        base if base in "ACGT" else "N" for base in seq.upper().replace("U", "T") if not base.isspace()
    )


def _reverse_complement(seq: str) -> str:
    return seq.translate(_DNA_COMPLEMENT)[::-1]
