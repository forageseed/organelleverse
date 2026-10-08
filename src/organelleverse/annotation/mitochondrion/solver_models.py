"""Protein-guided gene models for mitochondrial genes the primary path missed.

The primary path (HMM, BLAST fallback, reference exon merging) encodes
angiosperm gene structures: cox1 has at most two exons there, nad9 none. In
the moss Physcomitrium cox1 has five exons, nad9 two and atp9 four (two of
them 9 and 22 bp), and all three were lost. The reading-frame DP in
``annotation/solver`` has no notion of an expected exon count, so it can build
those structures from a reference protein alone.

For each still-missing gene: tblastn places the reference proteins, hits on
one strand that lie within an intron's reach of each other form a candidate
locus, the reference with the best summed bit score at that locus becomes a
BLOSUM62 profile, and the DP solves the locus window. A model is kept only if
it covers most of the reference protein and does not sit on another gene.

Trans-spliced genes (nad1, nad2, nad5) are not modelled here: their pieces lie
on different loci, which a single window cannot express.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Seq import Seq

from organelleverse._losat import losat_from_tool_paths, run_losat
from organelleverse.core.external import run_external

from ..execution import CommandRunner
from ..solver.dp import AMINO_ACIDS, Profile
from ..solver.vector import solve_vector
from .codon_contract import allowed_start_codons
from .models.gene import ExonRecord, GeneAnnotation, Strand
from .models.genome import GenomeSequence

logger = logging.getLogger(__name__)

TRANS_SPLICED = frozenset({"nad1", "nad2", "nad5", "rps12"})

#: The DP's default 400 bp floor forced both splice sites of a 395 bp
#: Physcomitrium cox1 intron 2-4 bp off. Lower floors let the DP invent the
#: cheapest legal intron to skip a stop codon or reach the profile's end: at
#: 250 bp, five invented introns in rice, maize, Medicago and Salix models were
#: 250-312 bp, while the shortest of 84 RefSeq cis-introns in seven mitogenomes
#: was 395 bp.
_MIN_INTRON = 380
_MAX_INTRON = 6000
#: Flank added around the tblastn hits of a locus, enough for an unmatched
#: terminal exon of a few dozen codons.
_WINDOW_FLANK = 1500
#: A model must span this share of its reference protein; fragments and
#: pseudogene remnants do not.
_MIN_PROTEIN_COVERAGE = 0.6
_MIN_HIT_BITS = 30.0
_BLOSUM62 = substitution_matrices.load("BLOSUM62")


#: An intron whose in-frame translation aligns to the reference protein this
#: well is skipped coding sequence, not an intron: with an in-frame stop in an
#: intronless rice cox1 the DP "spliced out" 381 bp of cox1 to avoid it.
_CODING_INTRON_SCORE = 100.0
_LOCAL = PairwiseAligner(mode="local", open_gap_score=-11, extend_gap_score=-1)
_LOCAL.substitution_matrix = _BLOSUM62


def _intron_codes_protein(intron: str, protein: str) -> bool:
    for frame in range(3):
        coding = intron[frame : frame + (len(intron) - frame) // 3 * 3]
        peptide = str(Seq(coding).translate()).replace("*", "X")
        if peptide and _LOCAL.score(peptide, protein) >= _CODING_INTRON_SCORE:
            return True
    return False


#: A model protein must align to its reference over most of its length. Correct
#: models (moss cox1, nad9, atp9) align at 82-94% identity over >=97% of their
#: length; an ORF the DP stitched together elsewhere after the real maize nad6
#: carried an in-frame stop aligned over 35%. Identity is kept permissive
#: because genuine atp4/atp8 sit at 42-45% against angiosperm references.
_MIN_ALIGNED_SHARE = 0.7
_MIN_IDENTITY = 0.4


def _homologous(model_protein: str, protein: str) -> bool:
    alignment = _LOCAL.align(model_protein, protein)[0]
    query, target = alignment.aligned
    span = sum(end - start for start, end in query)
    same = sum(
        model_protein[qs + k] == protein[ts + k]
        for (qs, qe), (ts, _) in zip(query, target, strict=True)
        for k in range(qe - qs)
    )
    return span >= _MIN_ALIGNED_SHARE * len(model_protein) and same >= _MIN_IDENTITY * span


@dataclass
class _Hit:
    ref: int
    start: int  # 1-based genome, start <= end
    end: int
    strand: int
    bits: float


def _profile(name: str, protein: str) -> Profile:
    scores = [
        [float(_BLOSUM62[a][b]) if a in AMINO_ACIDS else -1.0 for b in AMINO_ACIDS] for a in protein
    ]
    return Profile(name=name, scores=scores)


def _reference_proteins(ref_dir: Path, gene: str) -> list[str]:
    for name in (gene, gene.lower()):
        path = ref_dir / f"{name}.Protein.fasta"
        if path.exists():
            proteins = [
                str(r.seq).rstrip("*").replace("*", "X") for r in SeqIO.parse(path, "fasta")
            ]
            return [p for p in proteins if p]
    return []


def _search(
    proteins: list[str],
    genome_fasta: Path,
    workdir: Path,
    tool_paths: Mapping[str, str] | None,
    command_runner: CommandRunner | None,
) -> list[_Hit]:
    query = workdir / "query.faa"
    # References repeat IDs with different sequences; number them instead.
    query.write_text("".join(f">r{n}\n{p}\n" for n, p in enumerate(proteins)))
    losat = losat_from_tool_paths(tool_paths)
    if losat is not None:
        rows = run_losat(
            "tblastn",
            query,
            genome_fasta,
            evalue=1e-5,
            seg=False,
            command_runner=command_runner,
            executable=losat,
            stage="pcg_solver_models:tblastn",
        )
    else:
        tblastn = tool_paths.get("tblastn") if tool_paths is not None else shutil.which("tblastn")
        if not tblastn:
            return []
        completed = run_external(
            [
                tblastn,
                "-query",
                str(query),
                "-subject",
                str(genome_fasta),
                "-outfmt",
                "6",
                "-evalue",
                "1e-5",
                "-seg",
                "no",
            ],
            code="tblastn_failed",
            tool="tblastn",
        )
        rows = [line.split("\t") for line in completed.stdout.splitlines() if line.strip()]
    hits = []
    for row in rows:
        if len(row) < 12 or float(row[11]) < _MIN_HIT_BITS:
            continue
        s, e = int(row[8]), int(row[9])
        hits.append(
            _Hit(int(row[0][1:]), min(s, e), max(s, e), 1 if s <= e else -1, float(row[11]))
        )
    return hits


def _loci(hits: list[_Hit]) -> list[list[_Hit]]:
    """Group hits on one strand that lie within an intron's reach of each other."""
    loci: list[list[_Hit]] = []
    for strand in (1, -1):
        current: list[_Hit] = []
        for hit in sorted((h for h in hits if h.strand == strand), key=lambda h: h.start):
            if current and hit.start - max(h.end for h in current) > _MAX_INTRON:
                loci.append(current)
                current = []
            current.append(hit)
        if current:
            loci.append(current)
    return loci


def _best_reference(locus: list[_Hit]) -> tuple[int, float]:
    totals: dict[int, float] = {}
    for hit in locus:
        totals[hit.ref] = totals.get(hit.ref, 0.0) + hit.bits
    ref = max(totals, key=lambda r: totals[r])
    return ref, totals[ref]


def _is_fragment(ann: GeneAnnotation, ref_dir: Path) -> bool:
    """A CDS far shorter than its own reference proteins is a remnant, not a gene."""
    proteins = _reference_proteins(ref_dir, ann.gene_name)
    if not proteins:
        return False
    typical = sorted(len(p) for p in proteins)[len(proteins) // 2]
    return sum(e.end - e.start + 1 for e in ann.exons) < _MIN_PROTEIN_COVERAGE * 3 * typical


def _overlapping(exons: list[tuple[int, int]], name: str, others: list[GeneAnnotation]):
    """Other-gene CDS sharing more than a fifth of their length with ``exons``."""
    hits = []
    for ann in others:
        if ann.gene_type != "CDS" or ann.gene_name.lower() == name.lower():
            continue
        shared = sum(
            max(0, min(b, e.end) - max(a, e.start) + 1) for a, b in exons for e in ann.exons
        )
        if shared > 0.2 * sum(e.end - e.start + 1 for e in ann.exons):
            hits.append(ann)
    return hits


def model_missing_genes(
    genome: GenomeSequence,
    db_manager,
    missing_genes: list[str],
    existing: list[GeneAnnotation],
    *,
    tool_paths: Mapping[str, str] | None = None,
    command_runner: CommandRunner | None = None,
) -> tuple[list[GeneAnnotation], list[GeneAnnotation]]:
    """Solver models for ``missing_genes`` (cis genes only); see module docstring.

    Returns ``(models, displaced)``: the new models and the fragmentary
    annotations of other genes they overlap, which the caller removes.
    """
    ref_dir = getattr(db_manager, "blast_ref_dir", None)
    if ref_dir is None or not Path(ref_dir).is_dir():
        return [], []
    sequence = genome.sequence.upper()
    length = len(sequence)
    models: list[GeneAnnotation] = []
    displaced: list[GeneAnnotation] = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        genome_fasta = workdir / "genome.fasta"
        genome_fasta.write_text(f">{genome.seqid}\n{sequence}\n")
        for gene in missing_genes:
            if gene.lower() in TRANS_SPLICED:
                continue
            proteins = _reference_proteins(Path(ref_dir), gene)
            if not proteins:
                continue
            loci = _loci(_search(proteins, genome_fasta, workdir, tool_paths, command_runner))
            ranked = sorted(loci, key=lambda locus: -_best_reference(locus)[1])
            for locus in ranked[:3]:
                ref, _ = _best_reference(locus)
                protein = proteins[ref]
                lo = max(0, min(h.start for h in locus) - 1 - _WINDOW_FLANK)
                hi = min(length, max(h.end for h in locus) + _WINDOW_FLANK)
                strand = locus[0].strand
                window = sequence[lo:hi]
                coding = window if strand == 1 else str(Seq(window).reverse_complement())
                solution = solve_vector(
                    _profile(gene, protein), coding, min_intron=_MIN_INTRON, max_intron=_MAX_INTRON
                )
                if solution is None or not solution.exons:
                    continue
                if solution.cds_length < _MIN_PROTEIN_COVERAGE * 3 * len(protein):
                    continue
                if strand == 1:
                    spans = [(lo + e.start, lo + e.end) for e in solution.exons]
                else:
                    spans = [(hi - e.end + 1, hi - e.start + 1) for e in solution.exons]
                spans.sort(key=lambda s: s[0], reverse=strand == -1)  # transcription order
                introns = [coding[a.end : b.start - 1] for a, b in pairwise(solution.exons)]
                if any(_intron_codes_protein(intron, protein) for intron in introns):
                    continue
                cds = "".join(coding[e.start - 1 : e.end] for e in solution.exons)
                if not _homologous(str(Seq(cds).translate()).rstrip("*"), protein):
                    continue
                clashes = _overlapping(spans, gene, existing + models)
                # A complete gene keeps its place; a remnant under the model
                # (moss nad9 carried a 240 bp "nad6") is replaced by it.
                if any(not _is_fragment(a, Path(ref_dir)) for a in clashes):
                    continue
                displaced.extend(clashes)
                exons, done = [], 0
                for number, (a, b) in enumerate(spans, 1):
                    exons.append(
                        ExonRecord(
                            start=a,
                            end=b,
                            strand=Strand.PLUS if strand == 1 else Strand.MINUS,
                            number=number,
                            phase=done % 3,
                        )
                    )
                    done += b - a + 1
                start_codon = coding[solution.exons[0].start - 1 : solution.exons[0].start + 2]
                # The initiator is not verified when it is not one the CDS contract
                # accepts (moss nad9 starts at GTG); mark the 5' end partial instead
                # of asserting it or rejecting a model whose body is correct.
                unverified_start = start_codon not in allowed_start_codons(
                    gene, allow_rna_editing=True
                )
                notes = [
                    f"protein-guided model ({len(exons)} exons) from a {len(protein)} aa reference"
                ]
                if unverified_start:
                    notes.append(f"initiator {start_codon} not verified; 5' end marked partial")
                models.append(
                    GeneAnnotation(
                        gene_name=gene,
                        is_partial_5prime=unverified_start,
                        gene_type="CDS",
                        exons=exons,
                        strand=Strand.PLUS if strand == 1 else Strand.MINUS,
                        source_method="solver_model",
                        score=solution.score,
                        confidence=solution.score,
                        notes=notes,
                    )
                )
                logger.info(
                    "Solver model: %s %d exons, %d bp, score %.1f",
                    gene,
                    len(exons),
                    solution.cds_length,
                    solution.score,
                )
                break
    return models, displaced
