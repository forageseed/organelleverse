"""Complete reference-product exon assemblies with copy-preserving selection.

Candidate enumeration preserves reference biological exon order and its colinear
blocks. It does not infer trans-splicing or resolve circular-origin geometry.
Protein score compares a candidate with its source product; reference score
calibration and pipeline promotion are separate from candidate construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, pairwise, product

from Bio.Data.CodonTable import TranslationError
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from .models import BlastHit, ReferenceQuery
from .references import exon_frame_offset
from .splice_refine import _san, _score


def protein_extended_parts(items: list[tuple[ReferenceQuery, BlastHit]]) -> list[FeatureLocation]:
    """Restore each exon's unqueried DNA and split-codon bases on its strand."""
    max_ext = 3000
    parts = []
    for query, hit in items:
        start, end = hit.start, hit.end
        if hit.qlen and hit.qstart and hit.qend:
            offset = exon_frame_offset(query.reference_feature)
            n5 = min(max_ext, offset + max(0, hit.qstart - 1) * 3)
            n3 = min(max_ext, max(0, len(query.reference_feature.sequence) - offset - hit.qend * 3))
            if hit.strand == 1:
                start, end = max(0, start - n5), end + n3
            else:
                start, end = max(0, start - n3), end + n5
        parts.append(FeatureLocation(start, end, strand=hit.strand))
    return parts


@dataclass(frozen=True)
class ExonModel:
    gene: str
    reference_product: str
    template: SeqFeature
    location: CompoundLocation
    mature_dna: str
    protein: str
    protein_score: float


def _colinear(a: FeatureLocation, b: FeatureLocation) -> bool:
    return a.strand == b.strand and (b.start >= a.end if a.strand == 1 else b.end <= a.start)


def assemble_exon_models(
    sequence: str,
    queries: tuple[ReferenceQuery, ...],
    hits: dict[str, list[BlastHit]],
    reference_sequences: dict[str, str],
) -> list[ExonModel]:
    """Enumerate complete products from supplied hits, never mix reference IDs.

    No candidate is generated for a missing source exon, overlapping target
    exons, a broken reference-colinear block, or a target assembly without a complete coding sequence.
    All coordinate-distinct alternatives remain available to selection.
    """
    groups: dict[str, list[ReferenceQuery]] = {}
    for query in queries:
        feature = query.reference_feature
        if feature.feature_type == "CDS_exon":
            groups.setdefault(feature.feature_id.rsplit(":exon", 1)[0], []).append(query)
    models = []
    target = Seq(sequence)
    for root, group in groups.items():
        ordered = sorted(group, key=lambda q: q.reference_feature.exon_index)
        first = ordered[0].reference_feature
        expected = first.exon_count
        if expected < 2 or [q.reference_feature.exon_index for q in ordered] != list(
            range(1, expected + 1)
        ):
            continue
        template = first.feature
        source_parts = template.location.parts
        cis = [_colinear(a, b) for a, b in pairwise(source_parts)]
        phase = int(template.qualifiers.get("codon_start", ["1"])[0]) - 1
        table = int(template.qualifiers.get("transl_table", ["11"])[0])
        source_dna = template.extract(Seq(reference_sequences[first.reference_name]))
        source_protein = str(source_dna[phase:].translate(table=table)).rstrip("*")
        for combination in product(*(hits.get(q.query_id, []) for q in ordered)):
            parts = protein_extended_parts(list(zip(ordered, combination, strict=True)))
            if any(p.end > len(target) for p in parts):
                continue
            if any(
                required and not _colinear(a, b)
                for required, a, b in zip(cis, parts[:-1], parts[1:], strict=True)
            ):
                continue
            if any(max(a.start, b.start) < min(a.end, b.end) for a, b in combinations(parts, 2)):
                continue
            location = CompoundLocation(parts)
            dna = str(location.extract(target))
            coding = dna[phase:]
            # Exon coverage alone is not a complete CDS: require the genetic
            # code's start, terminal stop, reading frame and no internal stops.
            try:
                translated = str(Seq(coding).translate(table=table, cds=True))
            except TranslationError:
                continue
            # A complete CDS normalizes an allowed initiation codon to M.
            # Keep the model's existing terminal-star representation.
            protein = translated + "*"
            models.append(
                ExonModel(
                    first.gene,
                    root,
                    template,
                    location,
                    dna,
                    protein,
                    _score(_san(source_protein), _san(protein.rstrip("*"))),
                )
            )
    return models


def select_mature_model_copies(models: list[ExonModel]) -> list[ExonModel]:
    """Keep every physical top-score placement only for one mature DNA class.

    Distinct DNA sequences tied at the highest score remain unresolved, even
    when their translated proteins are identical. Return no selected model.
    """
    if not models:
        return []
    highest = max(m.protein_score for m in models)
    best = [m for m in models if m.protein_score == highest]
    if any(m.mature_dna != best[0].mature_dna for m in best[1:]):
        return []
    selected = []
    for model in best:
        if not any(model.location == m.location for m in selected):
            selected.append(model)
    return selected
