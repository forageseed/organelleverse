"""Per-exon coverage of whole-product sequence alignments; not a ranking.

A high local product score is not evidence that every exon has correspondence.
Split codons belong to both adjacent exon domains. Identities, uncovered ends,
translation-domain excess and ambiguous alignments are retained explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

from Bio.SeqFeature import SeqFeature

from .splice_refine import _AL, _san


@dataclass(frozen=True)
class CodingExonDomain:
    exon_index: int
    coding_interval: tuple[int, int]
    protein_residues: tuple[int, ...]
    coding_bases_outside_translation: int


@dataclass(frozen=True)
class ExonSequenceCorrespondence:
    source_id: str
    target_id: str
    status: str
    alignment_count: int | None
    source_domains: tuple[CodingExonDomain, ...]
    target_domains: tuple[CodingExonDomain, ...]
    residue_pairs: tuple[tuple[int, int], ...]
    uncovered_source_residues: tuple[tuple[int, ...], ...]
    uncovered_target_residues: tuple[tuple[int, ...], ...]
    exon_pair_residue_counts: tuple[tuple[int, ...], ...]
    unequal_residue_pairs: tuple[tuple[int, int], ...]
    source_residues_outside_coding_dna: tuple[int, ...]
    target_residues_outside_coding_dna: tuple[int, ...]


def coding_exon_domains(feature: SeqFeature, protein: str) -> tuple[CodingExonDomain, ...]:
    """Domains in original biological part order, independent of genomic strand.

    The caller supplies its actual comparable translation. DNA outside that
    translation (including a terminal stop or a partial codon) is reported,
    never converted into an aligned residue or silently discarded.
    """
    start = int(feature.qualifiers.get("codon_start", ["1"])[0])
    if start not in (1, 2, 3):
        raise ValueError("codon_start must be 1, 2 or 3")
    translated_end = 3 * len(protein.rstrip("*"))
    position = 1 - start
    domains = []
    for i, part in enumerate(feature.location.parts, 1):
        left, right = position, position + len(part)
        lo, hi = max(0, left), min(translated_end, right)
        residues = tuple(range(lo // 3, (hi + 2) // 3)) if hi > lo else ()
        excess = max(0, right - max(left, translated_end, 0))
        domains.append(CodingExonDomain(i, (left, right), residues, excess))
        position = right
    return tuple(domains)


def compare_exon_sequences(
    source_id: str,
    source_feature: SeqFeature,
    source_protein: str,
    target_id: str,
    target_feature: SeqFeature,
    target_protein: str,
) -> ExonSequenceCorrespondence:
    """Record each exon domain under one unique optimal product alignment.

    Unequal aligned residues remain correspondences, not identical bases or
    proof of homology. No exon coverage threshold, vote or promotion decision
    is applied. Multiple alignments leave ALL domains explicitly unresolved.
    """
    if (
        not source_id
        or not target_id
        or not source_protein.rstrip("*")
        or not target_protein.rstrip("*")
    ):
        raise ValueError("Sequence correspondence requires identities and translations")
    source, target = _san(source_protein.rstrip("*")), _san(target_protein.rstrip("*"))
    sd, td = (
        coding_exon_domains(source_feature, source),
        coding_exon_domains(target_feature, target),
    )
    source_domain = {r for d in sd for r in d.protein_residues}
    target_domain = {r for d in td for r in d.protein_residues}
    source_outside = tuple(i for i in range(len(source)) if i not in source_domain)
    target_outside = tuple(i for i in range(len(target)) if i not in target_domain)
    alignments = _AL.align(source, target)
    try:
        count = len(alignments)
    except OverflowError:
        count = None
    if count != 1:
        return ExonSequenceCorrespondence(
            source_id,
            target_id,
            "no_alignment" if count == 0 else "ambiguous_alignment",
            count,
            sd,
            td,
            (),
            (),
            (),
            (),
            (),
            source_outside,
            target_outside,
        )
    pairs = tuple(
        (int(a), int(b)) for a, b in zip(*alignments[0].indices, strict=True) if a >= 0 and b >= 0
    )
    ss, ts = {a for a, _ in pairs}, {b for _, b in pairs}
    return ExonSequenceCorrespondence(
        source_id,
        target_id,
        "unique_alignment",
        1,
        sd,
        td,
        pairs,
        tuple(tuple(r for r in d.protein_residues if r not in ss) for d in sd),
        tuple(tuple(r for r in d.protein_residues if r not in ts) for d in td),
        tuple(
            tuple(
                sum(a in x.protein_residues and b in y.protein_residues for a, b in pairs)
                for y in td
            )
            for x in sd
        ),
        tuple((a, b) for a, b in pairs if source[a] != target[b]),
        source_outside,
        target_outside,
    )
