"""Native tRNA/rRNA features for the plastome backend.

Replaces BLAST-transferred tRNA/rRNA with the native covariance-model tRNA
engine (plastid CM + HMM + intron recovery + anticodon naming) and the native
pyhmmer rRNA detector — no external bioinformatics program.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from ..cmsearch.genome import default_plastid_cm_path, default_plastid_hmm_path, find_trnas
from ..cmsearch.introns import splice_trna_in_span
from ..cmsearch.model import parse_cm
from ..cmsearch.models import CMHit
from ..cmsearch.naming import name_trna
from ..cmsearch.search import _resolve_overlaps
from .rrna_native import annotate_plastid_rrna

_RC = str.maketrans("ACGTacgt", "TGCAtgca")

_RRNA_PRODUCT = {
    "16S": "16S ribosomal RNA",
    "23S": "23S ribosomal RNA",
    "4.5S": "4.5S ribosomal RNA",
    "5S": "5S ribosomal RNA",
}
_RRNA_GENE = {"16S": "rrn16", "23S": "rrn23", "4.5S": "rrn4.5", "5S": "rrn5"}

# Single-letter -> NCBI three-letter code for standard tRNA product names
# (``tRNA-His`` rather than ``tRNA-H``). fMet is the plastid initiator.
_AA1_TO_3 = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "Q": "Gln",
    "E": "Glu",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
    "fM": "fMet",
}


def _trna_product(aa: str) -> str:
    return f"tRNA-{_AA1_TO_3.get(aa, aa)}"


def _rc(seq: str) -> str:
    return seq.translate(_RC)[::-1]


@dataclass(frozen=True)
class IntronTRNAHint:
    """Approximate gene span (1-based, forward) of an intron tRNA from a reference."""

    start: int
    end: int
    strand: int
    exon_lengths: tuple[int, int]  # (5' exon, 3' exon) in transcription order
    amino_acid: str | None = None  # expected identity, e.g. "K" for trnK-UUU


def _hit_from_splice(spliced, model) -> CMHit:
    return CMHit(
        start=spliced.exon1_start,
        end=spliced.exon2_end,
        strand=spliced.strand,
        score=spliced.score,
        cm_from=1,
        cm_to=model.clen,
        exons=(
            (spliced.exon1_start, spliced.exon1_end),
            (spliced.exon2_start, spliced.exon2_end),
        ),
    )


def _hint_hits(seq: str, hints, hits: list[CMHit], model, min_bits: float) -> list[CMHit]:
    """Splice intron tRNAs at reference-transferred spans the HMM pairing missed."""
    extra: list[CMHit] = []
    for hint in hints:
        if any(h.exons and not (h.end < hint.start or h.start > hint.end) for h in hits):
            continue
        spliced = splice_trna_in_span(
            seq,
            hint.start,
            hint.end,
            hint.strand,
            hint.exon_lengths,
            model,
            min_bits=min_bits,
            expected_aa=hint.amino_acid,
        )
        if spliced is None:
            continue
        extra.append(_hit_from_splice(spliced, model))
    return extra


def _spliced_mature(seq: str, hit: CMHit) -> str:
    mature = "".join(seq[s - 1 : e] for s, e in hit.exons)
    return mature if hit.strand >= 0 else _rc(mature)


def _expected_identity(seq: str, hit: CMHit, hints, model) -> str | None:
    """Amino acid of a spliced hit: from an overlapping reference span, else read.

    A splice shifted by two nt can read another valid anticodon (trnV-UAC's
    loop CGUUUACAC reads UUU, i.e. trnK), so a reference identity wins.
    """
    for hint in hints:
        same_strand = hint.strand == hit.strand
        if same_strand and hint.amino_acid and not (hint.end < hit.start or hint.start > hit.end):
            return hint.amino_acid
    named = name_trna(model, _spliced_mature(seq, hit))
    return named[1] if named else None


def _resplice_with_reference_lengths(
    seq: str,
    hits: list[CMHit],
    exon_lengths: Mapping[str, tuple[int, int]],
    hints,
    model,
    min_bits,
) -> list[CMHit]:
    """Re-place splice sites of HMM-paired intron tRNAs using reference exon lengths.

    Pair recovery chooses splice sites by CM score alone; with the lengths the
    same rule as for hinted spans applies, so both paths agree.
    """
    out: list[CMHit] = []
    for h in hits:
        amino_acid = _expected_identity(seq, h, hints, model) if h.exons else None
        lengths = exon_lengths.get(amino_acid) if amino_acid else None
        spliced = (
            splice_trna_in_span(
                seq,
                h.start,
                h.end,
                h.strand,
                lengths,
                model,
                min_bits=min_bits,
                expected_aa=amino_acid,
            )
            if lengths
            else None
        )
        if spliced is None:
            out.append(h)
            continue
        out.append(_hit_from_splice(spliced, model))
    return out


#: Exon seeds: a match may differ from a reference exon at this share of its
#: bases (2 of trnG-UCC's 23 nt 5' exon), and the two exons of one tRNA lie
#: this far apart (intron length, as for HMM window pairing).
_SEED_MISMATCH_SHARE = 0.1
_SEED_INTRON = (100, 3000)
#: A 5' exon seed inside a complete tRNA is that tRNA's own 5' end (trnG-GCC
#: starts like trnG-UCC exon 1), not an exon.
_COMPLETE_TRNA_NT = 60
#: Seeds are probed by the exon ends next to the intron (anticodon arm): a
#: whole-exon Hamming match fails on a single T-arm indel (Marchantia trnG-UCC
#: exon 2 is 47 nt against the references' 48-49). Local CYK sets the ends.
_SEED_PROBE_NT = 24
#: Reference exons this far from the family's median length are annotation
#: slips (a 130 nt trnK-UUU "exon 1" running into the intron), not seeds.
_SEED_LENGTH_SLACK = 5

_BASES = {b: n for n, b in enumerate("ACGT")}


def _encode(text: str) -> np.ndarray:
    return np.array([_BASES.get(c, 4) for c in text], dtype=np.int8)


def _near_matches(genome: np.ndarray, probe: str) -> list[int]:
    """0-based starts where ``probe`` matches ``genome`` with few mismatches."""
    n = len(probe)
    if n == 0 or len(genome) < n:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(genome, n)
    mismatches = (windows != _encode(probe)).sum(axis=1)
    return np.nonzero(mismatches <= int(_SEED_MISMATCH_SHARE * n))[0].tolist()


def _seed_hints(
    seq: str,
    exon_seeds: Mapping[str, Sequence[tuple[str, str]]],
    exon_lengths: Mapping[str, tuple[int, int]],
    hits: list[CMHit],
) -> list[IntronTRNAHint]:
    """Intron-tRNA spans from near-exact matches of reference 5' and 3' exons.

    The genome-wide HMM filter cannot anchor a 23 nt exon (trnG-UCC exon 1),
    and without a transferred gene span nothing pointed at the tRNA; its 3'
    exon was then reported alone as a 43-51 nt "mature" tRNA. Reference exons
    are conserved enough to find by sequence: 5' and 3' exon matches in
    transcription order, an intron's length apart, bound the gene.
    """
    complete = [h for h in hits if not h.exons and h.end - h.start + 1 >= _COMPLETE_TRNA_NT]
    hints: list[IntronTRNAHint] = []
    length = len(seq)
    for strand, text in ((1, seq), (-1, str(Seq(seq).reverse_complement()))):
        genome = _encode(text)
        for family, pairs in exon_seeds.items():
            if family not in exon_lengths:
                continue
            five_len, three_len = exon_lengths[family]
            pairs = [
                (five, three)
                for five, three in pairs
                if abs(len(five) - five_len) <= _SEED_LENGTH_SLACK
                and abs(len(three) - three_len) <= _SEED_LENGTH_SLACK
            ]
            ends5 = {
                m + len(five[-_SEED_PROBE_NT:])
                for five, _ in pairs
                for m in _near_matches(genome, five[-_SEED_PROBE_NT:])
            }
            threes = {
                (m, len(three))
                for _, three in pairs
                for m in _near_matches(genome, three[:_SEED_PROBE_NT])
            }
            for end5 in sorted(ends5):
                start = max(1, end5 - five_len + 1)
                partners = sorted(
                    (m3, n3) for m3, n3 in threes if _SEED_INTRON[0] <= m3 - end5 <= _SEED_INTRON[1]
                )
                if not partners:
                    continue
                m3, n3 = partners[0]  # the nearest 3' exon
                end = m3 + n3
                lo, hi = (start, end) if strand == 1 else (length - end + 1, length - start + 1)
                five_span = (lo, lo + five_len - 1) if strand == 1 else (hi - five_len + 1, hi)
                if any(
                    h.strand == strand and not (h.end < five_span[0] or h.start > five_span[1])
                    for h in complete
                ):
                    continue
                hints.append(
                    IntronTRNAHint(
                        start=lo,
                        end=hi,
                        strand=strand,
                        exon_lengths=exon_lengths[family],
                        amino_acid=family,
                    )
                )
    return hints


def native_trna_features(
    seq: str,
    *,
    min_bits: float = 20.0,
    intron_hints: Sequence[IntronTRNAHint] = (),
    intron_exon_lengths: Mapping[str, tuple[int, int]] | None = None,
    intron_exon_seeds: Mapping[str, Sequence[tuple[str, str]]] | None = None,
) -> list[SeqFeature]:
    """Detect and name plastid tRNAs; return them as GenBank ``tRNA`` features.

    Intron tRNAs are emitted as spliced ``join`` locations and named from their
    spliced mature sequence. ``intron_exon_lengths`` (reference (5', 3') exon
    lengths by amino acid) place splice sites of HMM-paired intron tRNAs;
    ``intron_hints`` recover those the HMM could not anchor.
    """
    model = parse_cm(default_plastid_cm_path())
    seq = seq.upper()
    hits = find_trnas(
        seq,
        cm_model=model,
        hmm_path=default_plastid_hmm_path(),
        min_bits=min_bits,
        recover_intron_pairs=True,
    )
    if intron_exon_lengths:
        hits = _resplice_with_reference_lengths(
            seq, hits, intron_exon_lengths, intron_hints, model, min_bits
        )
    if intron_exon_seeds and intron_exon_lengths:
        intron_hints = [
            *intron_hints,
            *_seed_hints(seq, intron_exon_seeds, intron_exon_lengths, hits),
        ]
    if intron_hints:
        hits = _resolve_overlaps(hits + _hint_hits(seq, intron_hints, hits, model, min_bits))
    features: list[SeqFeature] = []
    for h in hits:
        strand = 1 if h.strand >= 0 else -1
        if h.exons:
            mature = _spliced_mature(seq, h)
            parts = [FeatureLocation(s - 1, e, strand=strand) for s, e in h.exons]
            if strand == -1:
                parts.reverse()
            location = CompoundLocation(parts)
        else:
            mature = seq[h.start - 1 : h.end]
            location = FeatureLocation(h.start - 1, h.end, strand=strand)
            if strand == -1:
                mature = _rc(mature)
        named = name_trna(model, mature)
        quals: dict[str, list[str]] = {"product": ["tRNA"]}
        if named is not None:
            anticodon, aa, _ = named
            quals = {
                "gene": [f"trn{aa}-{anticodon.replace('T', 'U')}"],
                "product": [_trna_product(aa)],
                "note": [f"anticodon:{anticodon}"],
            }
        if h.exons:
            quals.setdefault("note", []).append("intron-containing tRNA")
        features.append(SeqFeature(location, type="tRNA", qualifiers=quals))
    return features


def native_rrna_features(seq: str) -> list[SeqFeature]:
    """Detect plastid rRNAs; return them as GenBank ``rRNA`` features."""
    features: list[SeqFeature] = []
    for h in annotate_plastid_rrna(seq):
        features.append(
            SeqFeature(
                FeatureLocation(h.start - 1, h.end, strand=h.strand),
                type="rRNA",
                qualifiers={
                    "gene": [_RRNA_GENE.get(h.rrna_type, h.gene_name)],
                    "product": [_RRNA_PRODUCT.get(h.rrna_type, f"{h.rrna_type} ribosomal RNA")],
                },
            )
        )
    return features


def native_trna_rrna_features(
    seq: str,
    *,
    intron_hints: Sequence[IntronTRNAHint] = (),
    intron_exon_lengths: Mapping[str, tuple[int, int]] | None = None,
    intron_exon_seeds: Mapping[str, Sequence[tuple[str, str]]] | None = None,
) -> list[SeqFeature]:
    """All native plastid tRNA + rRNA features for one plastome sequence."""
    trnas = native_trna_features(
        seq,
        intron_hints=intron_hints,
        intron_exon_lengths=intron_exon_lengths,
        intron_exon_seeds=intron_exon_seeds,
    )
    return trnas + native_rrna_features(seq)
