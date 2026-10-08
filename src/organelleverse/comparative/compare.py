"""Comparative genomics — self-contained implementations.

These functions compare multiple annotated organelle genomes by their gene
content and gene order using the canonical annotation document.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from ..annotation.genbank import parse_genbank
from ..annotation.models import AnnotationRecord
from ..core.frozen import FrozenMap, thaw_json
from ..core.genome import OrganelleGenome
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult, ResultScope

_SUITE = "comparative"
_OPERATION_VERSION = "1.0"
_METHOD = "organelleverse"

_SCOPE_BY_ORGANELLE: dict[str, ResultScope] = {
    "mitochondrion": "mitochondrion",
    "mito": "mitochondrion",
    "plastid": "plastid",
    "chloro": "plastid",
    "chloroplast": "plastid",
}

# Canonical alias map for normalizing common organelle gene-name variants.
_GENE_ALIASES = {
    "nad1": {"nad1", "nu1", "mt-nad1"},
    "nad2": {"nad2", "nu2"},
    "cox1": {"cox1", "coxi", "mt-cox1"},
    "cob": {"cob", "cytb"},
    "atp1": {"atp1", "atpa"},
    "rbcL": {"rbcl", "rubisco"},
    "matK": {"matk", "maturasek"},
}


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance(op: str, parameters: Mapping[str, Any]) -> ResultProvenance:
    """Build canonical provenance for one ``comparative`` operation."""
    package_version = _package_version()
    return ResultProvenance(
        operation_id=f"{_SUITE}.{op}",
        operation_version=_OPERATION_VERSION,
        package_version=package_version,
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=_sha256_json(dict(parameters)),
        actual_backend=_METHOD,
        attempted_backends=(_METHOD,),
        software_versions=FrozenMap({"organelleverse": package_version}),
    )


def _scope(organelle: str | None) -> ResultScope:
    if organelle is None:
        return "none"
    return _SCOPE_BY_ORGANELLE.get(str(organelle).casefold(), "none")


def _genomes_scope(genomes: list[OrganelleGenome]) -> ResultScope:
    return _scope(genomes[0].organelle) if genomes else "none"


def _failed(op: str, *, scope: ResultScope, message: str, code: str) -> OrganelleResult:
    return OrganelleResult(
        operation_id=f"{_SUITE}.{op}",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="failed",
        summary_text=message,
        errors=(ErrorDetail(code=code, message=message),),
    )


def _result_metrics(result: OrganelleResult | Mapping[str, Any]) -> Mapping[str, Any]:
    """Return plain-JSON metrics from a canonical result or a raw mapping."""
    if isinstance(result, OrganelleResult):
        return cast(Mapping[str, Any], thaw_json(result.metrics))
    return dict(result)


def _annotation_path(genome: OrganelleGenome) -> Path | None:
    return genome.annotation.resolve() if genome.annotation is not None else None


def synteny(
    genomes: list[OrganelleGenome],
) -> OrganelleResult:
    """Detect collinear gene-order blocks across ≥2 annotated genomes.

    Compares the ordered list of gene names per genome and reports shared
    contiguous runs (synteny blocks). Returns block count + a Jaccard order
    similarity.
    """
    if len(genomes) < 2:
        return _failed(
            "synteny",
            scope=_genomes_scope(genomes),
            message="synteny() requires ≥2 genomes.",
            code="too_few_genomes",
        )
    orders = [_gene_order(g) for g in genomes]
    blocks = _shared_blocks(orders)
    sim = _order_jaccard(orders)
    obs = {
        "synteny_blocks": len(blocks),
        "order_jaccard": round(sim, 4),
        "genome_count": len(genomes),
        "blocks": [list(block) for block in blocks],
    }
    return OrganelleResult(
        operation_id=f"{_SUITE}.synteny",
        operation_version=_OPERATION_VERSION,
        scope=_genomes_scope(genomes),
        status="ok",
        summary_text=f"Found {len(blocks)} collinear blocks; order Jaccard {sim:.3f}.",
        metrics=FrozenMap(obs),
        findings=(
            Finding(code="synteny_blocks", metric="synteny_blocks", value=len(blocks)),
            Finding(code="order_jaccard", metric="order_jaccard", value=round(sim, 4)),
        ),
        flags=(),
        artifacts=(),
        provenance=_provenance("synteny", {"genome_count": len(genomes)}),
    )


def compare_genes(
    genomes: list[OrganelleGenome],
) -> OrganelleResult:
    """Compare gene presence across species (intersection / union / differences)."""
    if not genomes:
        return _failed(
            "compare_genes",
            scope="none",
            message="compare_genes() requires >=1 genome.",
            code="no_genomes",
        )
    sets = [_gene_set(g) for g in genomes]
    intersection = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    accessions = [_genome_label(g, i) for i, g in enumerate(genomes)]
    obs = {
        "intersection": len(intersection),
        "union": len(union),
        "genome_count": len(genomes),
        "shared_genes": sorted(intersection),
        "all_genes": sorted(union),
        "per_genome": {accessions[i]: sorted(genes) for i, genes in enumerate(sets)},
    }
    return OrganelleResult(
        operation_id=f"{_SUITE}.compare_genes",
        operation_version=_OPERATION_VERSION,
        scope=_genomes_scope(genomes),
        status="ok",
        summary_text=f"{len(intersection)} shared genes; {len(union)} total (union).",
        metrics=FrozenMap(obs),
        findings=(
            Finding(code="shared_genes", metric="shared_genes", value=len(intersection)),
            Finding(code="total_genes", metric="total_genes", value=len(union)),
        ),
        flags=(),
        artifacts=(),
        provenance=_provenance("compare_genes", {"genome_count": len(genomes)}),
    )


def gene_table(
    genomes: list[OrganelleGenome],
) -> OrganelleResult:
    """Generate a gene presence/absence matrix across genomes."""
    if not genomes:
        return _failed(
            "gene_table",
            scope="none",
            message="gene_table() requires >=1 genome.",
            code="no_genomes",
        )
    sets = [_gene_set(g) for g in genomes]
    all_genes = sorted(set.union(*sets)) if sets else []
    matrix = {gene: [int(gene in s) for s in sets] for gene in all_genes}
    accessions = [f"g{i}" for i in range(len(genomes))]
    obs = {"rows": len(all_genes), "cols": len(genomes), "matrix": matrix, "accessions": accessions}
    return OrganelleResult(
        operation_id=f"{_SUITE}.gene_table",
        operation_version=_OPERATION_VERSION,
        scope=_genomes_scope(genomes),
        status="ok",
        summary_text=f"Gene PAV matrix: {len(all_genes)} genes x {len(genomes)} genomes.",
        metrics=FrozenMap(obs),
        findings=(Finding(code="matrix_rows", metric="matrix_rows", value=len(all_genes)),),
        flags=(),
        artifacts=(),
        provenance=_provenance("gene_table", {"genome_count": len(genomes)}),
    )


def write_gene_table(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write the comparative gene presence/absence table from ``gene_table()``."""
    metrics = _result_metrics(result)
    matrix = dict(metrics.get("matrix", {}))
    accessions = list(metrics.get("accessions", ()))
    path = _resolve_output_path(output, "gene_pav_matrix.tsv")
    lines = ["gene\t" + "\t".join(str(a) for a in accessions)]
    for gene in sorted(matrix):
        lines.append(str(gene) + "\t" + "\t".join(map(str, matrix[gene])))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_synteny(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write shared synteny blocks from ``synteny()``."""
    metrics = _result_metrics(result)
    blocks = list(metrics.get("blocks", ()))
    path = _resolve_output_path(output, "synteny_blocks.tsv")
    lines = ["block_id\tgenes"]
    for i, block in enumerate(blocks, 1):
        lines.append(f"block_{i}\t{','.join(str(gene) for gene in block)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_gene_comparison(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write gene-content comparison rows from ``compare_genes()``."""
    metrics = _result_metrics(result)
    per_genome = metrics.get("per_genome", {})
    if not isinstance(per_genome, Mapping):
        per_genome = {}
    all_genes = list(metrics.get("all_genes", ()))
    path = _resolve_output_path(output, "gene_comparison.tsv")
    labels = list(per_genome)
    lines = ["gene\tshared\t" + "\t".join(labels)]
    shared = set(metrics.get("shared_genes", ()))
    genome_sets = {label: set(genes) for label, genes in per_genome.items()}
    for gene in all_genes:
        lines.append(
            str(gene)
            + f"\t{int(gene in shared)}"
            + "".join(f"\t{int(gene in genome_sets[label])}" for label in labels)
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def compare_genomes(
    genomes: list[OrganelleGenome],
) -> OrganelleResult:
    """Compare genome structure: sizes, gene counts, GC."""
    if not genomes:
        return _failed(
            "compare_genomes",
            scope="none",
            message="compare_genomes() requires >=1 genome.",
            code="no_genomes",
        )
    stats = []
    for i, g in enumerate(genomes):
        annotation_path = _annotation_path(g)
        records = parse_genbank(annotation_path).records if annotation_path is not None else ()
        total = sum(len(record.sequence) for record in records)
        n_genes = sum(
            1
            for record in records
            for feature in record.features
            if feature.type.casefold() == "cds"
        )
        gc = _gc(records)
        stats.append(
            {"genome": _genome_label(g, i), "length": total, "genes": n_genes, "gc": round(gc, 4)}
        )
    return OrganelleResult(
        operation_id=f"{_SUITE}.compare_genomes",
        operation_version=_OPERATION_VERSION,
        scope=_genomes_scope(genomes),
        status="ok",
        summary_text=f"Compared {len(genomes)} genomes (length/genes/GC).",
        metrics=FrozenMap({"genomes": stats}),
        findings=tuple(
            Finding(code="genome_length", metric="genome_length", value=s["length"]) for s in stats
        ),
        flags=(),
        artifacts=(),
        provenance=_provenance("compare_genomes", {"genome_count": len(genomes)}),
    )


def write_genome_comparison(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write genome-level comparison rows from ``compare_genomes()``."""
    metrics = _result_metrics(result)
    rows = list(metrics.get("genomes", ()))
    path = _resolve_output_path(output, "genome_comparison.tsv")
    lines = ["genome\tlength\tgenes\tgc"]
    for row in rows:
        lines.append(
            f"{row.get('genome', '')}\t{row.get('length', '')}\t"
            f"{row.get('genes', '')}\t{row.get('gc', '')}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def normalize_genes(gene_names: list[str]) -> dict[str, str]:
    """Normalize organelle gene-name variants to canonical names.

    Returns {input_name: canonical_name}. Unrecognized names map to themselves
    (lower-cased). This is the pure-Python equivalent of a gene-name alias DB.
    """
    out: dict[str, str] = {}
    for name in gene_names:
        key = name.strip().lower()
        canonical = key
        for canon, aliases in _GENE_ALIASES.items():
            if key in aliases or key == canon.lower():
                canonical = canon
                break
        out[name] = canonical
    return out


# -- helpers --------------------------------------------------------------


def _resolve_output_path(output: str | Path, default_name: str) -> Path:
    path = Path(output)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name


def _gene_order(g: OrganelleGenome) -> list[str]:
    annotation_path = _annotation_path(g)
    records = parse_genbank(annotation_path).records if annotation_path is not None else ()
    order: list[str] = []
    for record in records:
        features_sorted = sorted(
            record.features,
            key=lambda feature: min(part.start for part in feature.parts),
        )
        for feature in features_sorted:
            if feature.type.casefold() != "cds":
                continue
            values = feature.qualifier_values("gene")
            order.append(values[0].lower() if values else "")
    return order


def _gene_set(g: OrganelleGenome) -> set[str]:
    return set(_gene_order(g))


def _genome_label(g: OrganelleGenome, index: int) -> str:
    return g.metadata.species or g.metadata.accession or f"g{index}"


def _shared_blocks(orders: list[list[str]]) -> list[tuple[str, ...]]:
    """Find contiguous runs shared by all gene-order lists (length ≥2)."""
    if not orders:
        return []
    sets_of_pairs = []
    for order in orders:
        pairs = {(order[i], order[i + 1]) for i in range(len(order) - 1)}
        sets_of_pairs.append(pairs)
    common = set.intersection(*sets_of_pairs) if sets_of_pairs else set()
    return sorted(common)


def _order_jaccard(orders: list[list[str]]) -> float:
    if len(orders) < 2:
        return 1.0
    s1 = {(orders[0][i], orders[0][i + 1]) for i in range(len(orders[0]) - 1)}
    s2 = {(orders[1][i], orders[1][i + 1]) for i in range(len(orders[1]) - 1)}
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0


def _gc(records: tuple[AnnotationRecord, ...]) -> float:
    total = sum(len(record.sequence) for record in records)
    if not total:
        return 0.0
    gc = sum(
        record.sequence.upper().count("G") + record.sequence.upper().count("C")
        for record in records
    )
    return gc / total
