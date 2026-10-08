"""Rebuild plastome ``gene`` features from the final CDS/tRNA/rRNA features.

Gene features used to be transferred from the reference once, before CDS
refinement moved coding boundaries and before native detection replaced every
tRNA/rRNA. Their names and spans then disagreed with the products (e.g. a
Triticum ``trnK`` gene beside a native ``trnK-UUU`` tRNA; a gene span that no
longer contains its refined CDS), so each locus appeared twice downstream.
Each named product now gets a gene feature derived from its own location.
"""

from __future__ import annotations

import re

from Bio.SeqFeature import FeatureLocation, SeqFeature

_PRODUCT_TYPES = ("CDS", "tRNA", "rRNA")
# tRNA/rRNA come from native detection; transferred gene features for them are
# never kept, whatever naming the reference used (trnK, tRNA-Lys, 16S rRNA).
_RNA_GENE = re.compile(r"^(trn|rrn|trna|rrna|\d+(\.\d+)?s\b)", re.IGNORECASE)


# Some references name the parts of trans-spliced rps12 as separate genes.
_PART_PREFIX = re.compile(r"^[35]'\s*-?\s*")


def _name(feature: SeqFeature) -> str:
    return feature.qualifiers.get("gene", [""])[0]


def _base_name(name: str) -> str:
    """Gene name without a 5'/3' part marker ("3' rps12" -> "rps12")."""
    return _PART_PREFIX.sub("", name)


def _spans_origin(parts, length: int | None) -> bool:
    """True when consecutive parts of a cis gene wrap across the circular origin."""
    if length is None or len(parts) < 2:
        return False
    starts = [int(p.start) for p in parts]
    ordered = (
        starts == sorted(starts)
        if parts[0].strand != -1
        else starts == sorted(starts, reverse=True)
    )
    return not ordered


def _gene_location(product: SeqFeature, length: int | None):
    parts = list(product.location.parts)
    strands = {p.strand for p in parts}
    if len(parts) > 1 and (
        "trans_splicing" in product.qualifiers or len(strands) > 1 or _spans_origin(parts, length)
    ):
        return product.location
    return FeatureLocation(
        min(int(p.start) for p in parts), max(int(p.end) for p in parts), strand=parts[0].strand
    )


def _overlaps(a: SeqFeature, b: SeqFeature) -> bool:
    return any(
        pa.strand == pb.strand and int(pa.start) < int(pb.end) and int(pb.start) < int(pa.end)
        for pa in a.location.parts
        for pb in b.location.parts
    )


def rebuild_gene_features(
    features: list[SeqFeature], length: int | None = None
) -> list[SeqFeature]:
    """Return ``features`` with gene features regenerated from the products.

    A transferred gene feature survives only when it has no product: not an
    RNA gene, not overlapping a same-named product, and either pseudo or of a
    name no product carries (pseudogene fragments such as the IR copy of ycf1).
    """
    products = [f for f in features if f.type in _PRODUCT_TYPES and _name(f)]
    product_names = {_name(f) for f in products}
    kept: list[SeqFeature] = []
    for feature in features:
        if feature.type != "gene":
            kept.append(feature)
            continue
        name = _base_name(_name(feature))
        if not name or _RNA_GENE.match(name):
            continue
        if any(_name(p) == name and _overlaps(feature, p) for p in products):
            continue
        if name in product_names and "pseudo" not in feature.qualifiers:
            continue
        kept.append(feature)
    for product in products:
        qualifiers: dict[str, list[str]] = {"gene": [_name(product)]}
        for key in ("pseudo", "trans_splicing"):
            if key in product.qualifiers:
                qualifiers[key] = list(product.qualifiers[key])
        kept.append(SeqFeature(_gene_location(product, length), type="gene", qualifiers=qualifiers))
    kept.sort(key=lambda f: (int(f.location.start), int(f.location.end), f.type != "gene", f.type))
    return kept
