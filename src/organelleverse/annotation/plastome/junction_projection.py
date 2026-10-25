"""Homologous coding-junction evidence, independent of candidate ranking.

Coordinates are offsets in the biologically ordered mature CDS after
codon_start, not genomic coordinates. References remain individually identified;
these records are evidence, not votes or a calibrated promotion score.
"""

from __future__ import annotations

from dataclasses import dataclass

from Bio.SeqFeature import SeqFeature

from .splice_refine import _AL, _san


@dataclass(frozen=True)
class JunctionProduct:
    product_id: str
    protein: str
    junctions: tuple[int, ...]


@dataclass(frozen=True)
class JunctionProjection:
    source_id: str
    target_id: str
    status: str
    alignment_count: int | None
    matched_pairs: tuple[tuple[int, int], ...]
    matched_translation_ends: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairedJunctionEvidence:
    reference_id: str
    current: JunctionProjection
    candidate: JunctionProjection
    comparable: bool
    gained_source_junctions: tuple[int, ...]
    lost_source_junctions: tuple[int, ...]
    gained_translation_ends: tuple[str, ...] = ()
    lost_translation_ends: tuple[str, ...] = ()


def coding_junctions(feature: SeqFeature) -> tuple[int, ...]:
    """Return cumulative CDS cuts in biological part order, on either strand."""
    codon_start = int(feature.qualifiers.get("codon_start", ["1"])[0])
    if codon_start not in (1, 2, 3):
        raise ValueError("codon_start must be 1, 2 or 3")
    position = 1 - codon_start
    cuts = []
    for part in feature.location.parts[:-1]:
        position += len(part)
        cuts.append(position)
    return tuple(cuts)


def junction_product(product_id: str, feature: SeqFeature, protein: str) -> JunctionProduct:
    """Use the caller's source-product identity and comparable translation."""
    if not product_id or not protein.rstrip("*"):
        raise ValueError("Junction evidence requires a product identity and translation")
    return JunctionProduct(product_id, _san(protein.rstrip("*")), coding_junctions(feature))


def _matched_junctions(source, target, indices) -> tuple[tuple[int, int], ...]:
    mapping = {
        int(s): (int(t), column)
        for column, (s, t) in enumerate(zip(*indices, strict=True))
        if s >= 0 and t >= 0
    }
    pairs = []
    for cut in source:
        if cut <= 0:
            continue
        codon, phase = divmod(cut, 3)
        target_cut = None
        if phase and codon in mapping:
            target_cut = 3 * mapping[codon][0] + phase
        elif not phase and codon - 1 in mapping and codon in mapping:
            left, left_column = mapping[codon - 1]
            right, right_column = mapping[codon]
            if right == left + 1 and right_column == left_column + 1:
                target_cut = 3 * right
        if target_cut is not None and target_cut in target:
            pairs.append((cut, target_cut))
    return tuple(pairs)


def project_junctions(source: JunctionProduct, target: JunctionProduct) -> JunctionProjection:
    """Project cuts only for a unique optimal protein alignment.

    A split codon needs the same within-codon phase. A cut between codons
    needs consecutive aligned flanks without an intervening alignment gap.
    Multiple optimal alignments stay unresolved, including count overflow.
    Protein-end correspondence is reported separately from junctions; it does
    not infer translation starts or prove exact genomic boundaries.
    """
    alignments = _AL.align(source.protein, target.protein)
    try:
        count = len(alignments)
    except OverflowError:
        count = None
    if count != 1:
        status = "no_alignment" if count == 0 else "ambiguous_alignment"
        return JunctionProjection(source.product_id, target.product_id, status, count, ())
    indices = alignments[0].indices
    pairs = _matched_junctions(source.junctions, target.junctions, indices)
    residue_pairs = set(zip(*(row.tolist() for row in indices), strict=True))
    ends = tuple(
        name
        for name, pair in (
            ("N", (0, 0)),
            ("C", (len(source.protein) - 1, len(target.protein) - 1)),
        )
        if pair in residue_pairs
    )
    return JunctionProjection(
        source.product_id, target.product_id, "unique_alignment", 1, pairs, ends
    )


def compare_junction_evidence(
    current: JunctionProduct,
    candidate: JunctionProduct,
    references: tuple[JunctionProduct, ...],
) -> tuple[PairedJunctionEvidence, ...]:
    """Compare on the same identifiable sources; keep uncomparable rows explicit.

    Coordinate-distinct copies must have distinct caller-supplied identities,
    even when their protein and coding cuts are identical. No copy is collapsed
    or counted as independent biological validation here.
    """
    identities = [r.product_id for r in references]
    if len(set(identities)) != len(identities):
        raise ValueError("Reference product identities must be unique")
    evidence = []
    for reference in references:
        old = project_junctions(reference, current)
        new = project_junctions(reference, candidate)
        comparable = old.status == new.status == "unique_alignment"
        old_cuts = {a for a, _ in old.matched_pairs}
        new_cuts = {a for a, _ in new.matched_pairs}
        evidence.append(
            PairedJunctionEvidence(
                reference.product_id,
                old,
                new,
                comparable,
                tuple(sorted(new_cuts - old_cuts)) if comparable else (),
                tuple(sorted(old_cuts - new_cuts)) if comparable else (),
                tuple(sorted(set(new.matched_translation_ends) - set(old.matched_translation_ends)))
                if comparable
                else (),
                tuple(sorted(set(old.matched_translation_ends) - set(new.matched_translation_ends)))
                if comparable
                else (),
            )
        )
    return tuple(evidence)
