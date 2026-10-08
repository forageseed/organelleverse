"""Reference-protein-validated assembly of trans-spliced mito CDS (nad1/nad2/nad5).

The heuristic ``merge_exons_to_gene`` selects exons by proximity/strand, which
fails for genuinely trans-spliced genes whose exons are scattered across the
genome on both strands — e.g. nad5 exon 3 is a 22 bp minus-strand exon ~28 kb
from exon 2. This module instead treats exon selection as a small combinatorial
search: for each exon number take the best few blastn candidates (plus
expected-length trims), splice every combination in transcription (exon-number)
order — each exon reverse-complemented on its own strand — translate, and keep
the combination whose protein best matches a reference and forms a clean ORF.

A quality gate (near-reference length + high identity) means the result is only
adopted when the assembly is confident; otherwise the caller falls back to the
existing merge, so divergent taxa never regress.
"""

from __future__ import annotations

import contextlib
import dataclasses
import statistics
from itertools import product
from pathlib import Path

from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq

from .models.gene import ExonRecord, GeneAnnotation, Strand

_AL = PairwiseAligner()
_AL.mode = "local"
_AL.open_gap_score = -11
_AL.extend_gap_score = -1
with contextlib.suppress(Exception):  # pragma: no cover
    _AL.substitution_matrix = substitution_matrices.load("BLOSUM62")

_AA = set("ACDEFGHIKLMNPQRSTVWY")
_TOPK = 3
_MAX_COMBOS = 20000


def _san(p: str) -> str:
    return "".join(c if c in _AA else "X" for c in p)


def _score(ref: str, prot: str) -> float:
    try:
        return _AL.score(ref, prot)
    except Exception:
        return float("-inf")


def _load_ref_proteins(ref_dir: Path, gene_name: str) -> list[str]:
    ref_file = ref_dir / f"{gene_name}.Protein.fasta"
    if not ref_file.exists():
        return []
    return [_san(str(rec.seq).rstrip("*")) for rec in SeqIO.parse(str(ref_file), "fasta")]


def _candidates(exon_hits, n: int):
    """Per-exon candidate (start, end, strand) list with expected-length trims."""
    cand = {}
    for en in range(1, n + 1):
        hits = exon_hits.get(en, [])
        if not hits:
            return None
        scored = {}
        for h in sorted(hits, key=lambda x: -(_len_ratio(x) * x[3])):
            s, e = min(h[0], h[1]), max(h[0], h[1])
            exp = h[5] or (e - s + 1)
            key = (s // 15, h[2])
            if key not in scored:
                scored[key] = (s, e, h[2], exp)
        variants = []
        for s, e, strand, exp in list(scored.values())[:_TOPK]:
            variants.append((s, e, strand))
            if (e - s + 1) > exp + 3:  # over-long: trim to expected from each end
                variants.append((s, s + exp - 1, strand))
                variants.append((e - exp + 1, e, strand))
        cand[en] = variants
    return cand


def _len_ratio(h) -> float:
    exp = h[5] or h[4]
    lo, hi = min(h[4], exp), max(h[4], exp)
    return lo / hi if hi else 0.0


def _spliced(genome: str, combo) -> str:
    parts = []
    for a, b, strand in combo:
        seg = genome[a - 1 : b]
        parts.append(str(Seq(seg).reverse_complement()) if int(strand) == -1 else seg)
    return "".join(parts)


def assemble_exons_by_protein(
    gene_name: str, genome, exon_hits, expected_exons: int, ref_dir: Path
) -> GeneAnnotation | None:
    """Return a protein-validated trans-spliced GeneAnnotation, or None to fall back."""
    ref_proteins = _load_ref_proteins(ref_dir, gene_name)
    if not ref_proteins or expected_exons < 2:
        return None
    cand = _candidates(exon_hits, expected_exons)
    if cand is None:
        return None
    combos = 1
    for en in cand:
        combos *= len(cand[en])
    if combos > _MAX_COMBOS:
        return None
    gseq = genome.sequence.upper()
    # A few representative references keep the inner loop fast; the full set gates.
    score_refs = sorted(ref_proteins, key=len, reverse=True)[:3]

    best = None
    for combo in product(*[cand[en] for en in range(1, expected_exons + 1)]):
        prot = _san(str(Seq(_spliced(gseq, combo)).translate(table=11)))
        if "*" in prot[:-1]:
            continue
        sc = max(_score(rp, prot) for rp in score_refs)
        total = sum(b - a + 1 for a, b, _ in combo)
        if best is None or (sc, -total) > (best[0], -best[3]):
            best = (sc, combo, prot, total)
    if best is None:
        return None
    _, combo, prot, _ = best

    # Quality gate: near-reference length AND high identity to a reference.
    ref_len = statistics.median(len(r) for r in ref_proteins)
    prot_core = prot.rstrip("*")
    if not (0.9 * ref_len <= len(prot_core) <= 1.1 * ref_len):
        return None
    ref_best = max(ref_proteins, key=lambda rp: _score(rp, prot))
    self_score = _score(ref_best, ref_best)
    if self_score <= 0 or _score(ref_best, prot) < 0.6 * self_score:
        return None

    exons = []
    cumulative = 0
    for i, (a, b, strand) in enumerate(combo, 1):
        exons.append(
            ExonRecord(start=a, end=b, strand=Strand(int(strand)), number=i, phase=cumulative % 3)
        )
        cumulative += b - a + 1
    return GeneAnnotation(
        gene_name=gene_name,
        gene_type="CDS",
        exons=exons,
        strand=Strand(int(combo[0][2])),
        notes=[f"Assembled {expected_exons} exons by reference-protein validation"],
        source_method="BLAST",
    )


#: Flank added on each side of a blastn exon hit before solving, and the N
#: spacer between windows. The DP treats "flank + spacer + flank" as an intron,
#: so the spacer must exceed the solver's minimum intron length.
_SOLVER_FLANK = 30
_SOLVER_SPACER = 400
#: Intron-opening cost for the joined windows. The DP's default (-19.4) is the
#: prior of an intron at an arbitrary position; here every spacer is a known
#: intron, and at that price the 21-22 bp nad5 exon 3 (7 residues) was never
#: worth a second intron. Between -12 and -8 monocot nad5 keeps all five exons
#: (protein 99.0% identical to RefSeq, against 98.7% without exon 3); a window
#: is only an exon hit +-30 bp, too short to hide a >=400 bp false intron.
_SOLVER_INTRON_OPEN = -10.0


def assemble_exons_by_solver(
    gene_name: str, genome, exon_hits, expected_exons: int, ref_dir: Path
) -> GeneAnnotation | None:
    """Place the exact splice sites of scattered exons with the reading-frame DP.

    blastn hit ends are routinely a base or two off the true exon ends; spliced
    that way, monocot nad5 (exon 1 of 231 bp and exon 3 of 21 bp against 230/22
    bp references) reads out of frame and the combinatorial assembly rejects
    every combination. Here each exon's best hit is widened by a flank, turned
    to its own strand, and the windows are joined in exon order with an N
    spacer, which the DP must treat as an intron. Solving that sequence against
    a reference protein puts every boundary on the codon, whatever the
    distance or strand between exons.
    """
    from ..solver.vector import solve_vector
    from .solver_models import _homologous, _profile

    ref_proteins = _load_ref_proteins(ref_dir, gene_name)
    if not ref_proteins or expected_exons < 2:
        return None
    gseq = genome.sequence.upper()
    pieces: list[str] = []
    origin: list[tuple[int, int] | None] = []  # (genome position 1-based, strand) per base
    for exon_number in range(1, expected_exons + 1):
        hits = exon_hits.get(exon_number)
        if not hits:
            # Exon counts are angiosperm counts (moss nad5 has four exons); the
            # protein gates below decide whether the windows found make a gene.
            continue
        # Identity x aligned length, not closeness to the expected length: the
        # 22 bp nad5 exon 3 reference carries a 60 bp flank, so its true hit is
        # 82 bp long and a length-ratio score prefers a spurious short hit.
        best = max(hits, key=lambda h: h[3] * h[4])
        lo = max(1, min(best[0], best[1]) - _SOLVER_FLANK)
        hi = min(len(gseq), max(best[0], best[1]) + _SOLVER_FLANK)
        strand = int(best[2])
        positions = list(range(lo, hi + 1))
        window = gseq[lo - 1 : hi]
        if strand == -1:
            window = str(Seq(window).reverse_complement())
            positions.reverse()
        if pieces:
            pieces.append("N" * _SOLVER_SPACER)
            origin.extend([None] * _SOLVER_SPACER)
        pieces.append(window)
        origin.extend((pos, strand) for pos in positions)
    if len(pieces) < 3:  # at least two windows and a spacer
        return None
    joined = "".join(pieces)

    ref_len = statistics.median(len(r) for r in ref_proteins)
    best_solution = None
    for protein in sorted(ref_proteins, key=lambda r: abs(len(r) - ref_len))[:3]:
        profile = dataclasses.replace(_profile(gene_name, protein), intron_open=_SOLVER_INTRON_OPEN)
        solution = solve_vector(profile, joined, min_intron=_SOLVER_SPACER)
        if solution is not None and (
            best_solution is None or solution.score > best_solution[0].score
        ):
            best_solution = (solution, protein)
    if best_solution is None:
        return None
    solution, protein = best_solution

    exons: list[tuple[int, int, int]] = []
    for exon in solution.exons:
        ends = [origin[exon.start - 1], origin[exon.end - 1]]
        span = origin[exon.start - 1 : exon.end]
        if any(o is None for o in span) or len({o[1] for o in span}) != 1:
            return None  # an exon may not run into the spacer
        (a, strand), (b, _) = ends
        exons.append((min(a, b), max(a, b), strand))
    cds = "".join(
        gseq[a - 1 : b] if strand == 1 else str(Seq(gseq[a - 1 : b]).reverse_complement())
        for a, b, strand in exons
    )
    peptide = str(Seq(cds).translate()).rstrip("*")
    if not 0.9 * ref_len <= len(peptide) <= 1.1 * ref_len or not _homologous(peptide, protein):
        return None

    records, cumulative = [], 0
    for number, (a, b, strand) in enumerate(exons, 1):
        records.append(
            ExonRecord(start=a, end=b, strand=Strand(strand), number=number, phase=cumulative % 3)
        )
        cumulative += b - a + 1
    return GeneAnnotation(
        gene_name=gene_name,
        gene_type="CDS",
        exons=records,
        strand=Strand(exons[0][2]),
        notes=[f"Assembled {len(records)} exons; splice sites placed by the reading-frame DP"],
        source_method="BLAST",
    )


def length_is_off(annotation: GeneAnnotation | None, ref_dir: Path) -> bool:
    """True when an existing CDS is clearly incomplete or overgrown.

    The DP assembly only replaces an annotation outside 80-125% of the median
    reference length (rice nad5 1200 of ~2000 bp, moss nad2 2013 of ~1470 bp).
    Real lengths vary across lineages -- moss nad1 is 9% longer than the
    angiosperm references -- so a plausible one is kept, as the other
    fallbacks keep it.
    """
    if annotation is None:
        return True
    proteins = _load_ref_proteins(ref_dir, annotation.gene_name)
    if not proteins:
        return False
    expected = 3 * (statistics.median(len(p) for p in proteins) + 1)
    length = sum(e.end - e.start + 1 for e in annotation.exons)
    return not 0.8 * expected <= length <= 1.25 * expected
