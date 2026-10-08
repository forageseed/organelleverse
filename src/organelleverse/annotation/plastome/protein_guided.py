"""Protein-guided CDS boundary refinement (EXPERIMENTAL — not wired into the
default pipeline).

Aligning the reference protein to the target's 3-frame translation pins the exact
reading frame and amino-acid extent, from which codon-exact start/stop follow;
it is edit-tolerant (C->U editing changes the codon but the alignment still
scores). With a good anchor it reaches ~0.92 exact single-exon CDS recall vs
GenBank — better than the nucleotide heuristic's ~0.86.

However, when driven from imprecise BLAST anchors it can override a correct
heuristic call with a different clean-but-wrong ORF, regressing end-to-end
recall. It is therefore kept as a validated building block but NOT enabled in
``PlastomeAnnotationPipeline`` until anchor selection / confidence gating is
robust. The default CDS refinement remains the nucleotide heuristic in
``cds_refine.py``.
"""

from __future__ import annotations

import contextlib

from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq

_STOP = {"TAA", "TAG", "TGA"}
_START = ("ATG", "ACG", "GTG")

# Local alignment finds a reliable conserved core (and thus the reading frame)
# even when the reference protein differs in length; the exact start/stop are
# then found by walking in-frame to the codons.
_ALIGNER = PairwiseAligner()
_ALIGNER.mode = "local"
_ALIGNER.open_gap_score = -11
_ALIGNER.extend_gap_score = -1
with contextlib.suppress(Exception):  # pragma: no cover - matrix always ships with Biopython
    _ALIGNER.substitution_matrix = substitution_matrices.load("BLOSUM62")


_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _san(prot: str) -> str:
    """Map any non-standard residue (stops, ambiguity codes) to X for scoring."""
    return "".join(c if c in _AA else "X" for c in prot)


def _score(a: str, b: str) -> float:
    try:
        return _ALIGNER.score(a, b)
    except Exception:
        return float("-inf")


def _codon(region: str, offset: int, pos: int) -> str:
    i = pos - offset
    return region[i : i + 3] if i >= 0 and i + 3 <= len(region) else ""


def _refine_forward(region: str, ref_protein: str, offset: int, genome_len: int, max_ext: int):
    """Return (genome_start, genome_end) 1-based for a forward-strand CDS.

    ``region`` starts at genome position ``offset`` (1-based).
    """
    best = None
    for frame in range(3):
        prot = _san(str(Seq(region[frame:]).translate(table=11)))
        if not prot:
            continue
        score = _score(ref_protein, prot)
        if best is None or score > best[0]:
            best = (score, frame, prot)
    if best is None:
        return None
    _, frame, prot = best
    try:
        aln = _ALIGNER.align(ref_protein, prot)[0]
    except Exception:
        return None
    r_start = aln.aligned[0][0][0]  # first aligned reference-protein residue
    t_start = aln.aligned[1][0][0]  # first aligned target-protein index
    t_end = aln.aligned[1][-1][1]  # one past last aligned index
    core_start = offset + frame + t_start * 3  # 1-based genome pos of core 5'
    core_end = offset + frame + t_end * 3 - 1  # core 3' (last aligned aa base)

    # Expected N-terminus: the reference N-term is r_start residues before the
    # aligned core. Snap to the start codon CLOSEST to that expected position.
    expected_start = core_start - r_start * 3
    gstart = expected_start
    best_k = None
    for k in range(-6, 7):
        s = expected_start + k * 3
        if s < 1 or s + 2 > genome_len or s < offset:
            continue
        if _codon(region, offset, s) in _START and (best_k is None or abs(k) < abs(best_k)):
            best_k, gstart = k, s

    # Walk downstream in-frame from the core to the first stop codon (inclusive).
    gend = core_end
    p = core_end + 1
    limit = min(genome_len - 2, core_end + max_ext)
    while p <= limit:
        c = _codon(region, offset, p)
        if c in _STOP:
            gend = p + 2
            break
        p += 3
    return gstart, gend


def refine_cds_by_protein(
    genome: str,
    start: int,
    end: int,
    strand: int,
    ref_protein: str,
    *,
    flank: int = 90,
    max_ext: int = 60,
) -> tuple[int, int] | None:
    """Codon-exact (start, end) 1-based from a reference protein, or None."""
    if not ref_protein:
        return None
    ref_protein = _san(ref_protein.rstrip("*"))
    L = len(genome)
    a = max(1, start - flank)
    b = min(L, end + flank)
    if strand == -1:
        rc = str(Seq(genome).reverse_complement())
        ra, rb = L - b + 1, L - a + 1
        res = _refine_forward(rc[ra - 1 : rb], ref_protein, ra, L, max_ext)
        if res is None:
            return None
        gs, ge = res
        return L - int(ge) + 1, L - int(gs) + 1
    res = _refine_forward(genome[a - 1 : b], ref_protein, a, L, max_ext)
    return (int(res[0]), int(res[1])) if res is not None else None


def refine_cds_by_best_protein(
    genome: str,
    start: int,
    end: int,
    strand: int,
    ref_proteins: list[str],
    *,
    max_refs: int = 8,
) -> tuple[int, int] | None:
    """Protein-guided boundaries using the best-scoring reference ortholog.

    Aligning the closest ortholog (highest score) gives the most precise
    boundary. Returns codon-exact (start, end), or None.
    """
    if not ref_proteins:
        return None
    L = len(genome)
    a = max(1, start - 90)
    b = min(L, end + 90)
    window = genome[a - 1 : b] if strand == 1 else str(Seq(genome[a - 1 : b]).reverse_complement())
    frames = [_san(str(Seq(window[f:]).translate(table=11))) for f in range(3)]

    uniq = list(dict.fromkeys(ref_proteins))[:max_refs]
    best_p, best_score, best_len = None, None, None
    for p in uniq:
        p = _san(p.rstrip("*"))
        if not p:
            continue
        score = max(_score(p, fr) for fr in frames)
        if best_score is None or score > best_score:
            best_score, best_p, best_len = score, p, len(p)
    if best_p is None:
        return None
    res = refine_cds_by_protein(genome, start, end, strand, best_p)
    if res is None:
        return None
    ns, ne = res
    # Validation gate: only trust a protein-guided call that yields a clean ORF
    # (start codon, terminal stop, no internal stop) of a length close to the
    # reference. Otherwise the caller keeps the heuristic result.
    if not _is_clean_orf(genome, ns, ne, strand, best_len):
        return None
    return ns, ne


def _is_clean_orf(genome: str, start: int, end: int, strand: int, ref_aa_len: int) -> bool:
    if end <= start or (end - start + 1) % 3 != 0:
        return False
    cds = genome[start - 1 : end]
    if strand == -1:
        cds = str(Seq(cds).reverse_complement())
    cds = cds.upper()
    if cds[:3] not in _START or cds[-3:] not in _STOP:
        return False
    codons = [cds[i : i + 3] for i in range(0, len(cds) - 3, 3)]
    if any(c in _STOP for c in codons):
        return False
    aa_len = len(cds) // 3 - 1  # excluding stop
    return not ref_aa_len or 0.7 * ref_aa_len <= aa_len <= 1.3 * ref_aa_len


def refine_single_exon_cds_features(genome, features, gene_proteins):
    """Refine single-exon CDS features in place using protein-guided boundaries.

    ``gene_proteins`` maps a lower-case gene name to reference protein strings.
    Features whose gene has no reference protein, or where alignment fails, are
    left for the heuristic refiner. Returns the set of feature ids refined.
    """
    from Bio.SeqFeature import FeatureLocation

    refined: set[int] = set()
    for feat in features:
        if feat.type != "CDS" or len(feat.location.parts) != 1:
            continue
        gene = (feat.qualifiers.get("gene", [""])[0] or "").lower()
        proteins = gene_proteins.get(gene)
        if not proteins:
            continue
        strand = feat.location.strand or 1
        start = int(feat.location.start) + 1
        end = int(feat.location.end)
        res = refine_cds_by_best_protein(genome, start, end, strand, proteins)
        if res is None:
            continue
        ns, ne = res
        if ne > ns:
            feat.location = FeatureLocation(ns - 1, ne, strand=strand)
            refined.add(id(feat))
    return refined
