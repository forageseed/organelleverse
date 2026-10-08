"""Annotation evidence audit and exact shared-node correspondence across paths."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from .annotation_projection import Annotation, normalize_annotations, project_annotations


def validate_gene_synonyms(synonyms: Mapping[str, str]) -> dict[str, str]:
    """Require explicit terminal canonical labels; never guess biological synonyms."""
    for source, target in synonyms.items():
        if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
            raise ValueError("Gene synonyms require nonempty string names")
        if target in synonyms and synonyms[target] != target:
            raise ValueError(
                "Gene synonym targets must be terminal canonical names; chains/cycles forbidden"
            )
    return dict(synonyms)


def audit_annotations(
    annotations: Sequence[Annotation],
    *,
    path_samples: Mapping[str, str],
    annotated_samples: Sequence[str],
    expected_genes: Sequence[str] = (),
    gene_synonyms: Mapping[str, str] | None = None,
) -> dict:
    """Report coverage of a user-declared expected set, not biological completeness.

    Missing annotation input is distinct from supplied annotation with zero
    selected gene features. Expected labels must be supplied explicitly: lineage
    gene content and annotation omissions cannot be inferred from a graph.
    """
    synonyms = validate_gene_synonyms(gene_synonyms or {})
    normalized = normalize_annotations(annotations, gene_synonyms=synonyms)
    samples = sorted(set(path_samples.values()))
    supplied = set(annotated_samples)
    if supplied - set(samples):
        raise ValueError("Annotated sample is absent from the graph")
    if len(set(expected_genes)) != len(expected_genes) or any(not x for x in expected_genes):
        raise ValueError("Expected genes must be unique nonempty names")
    expected = {synonyms.get(gene, gene) for gene in expected_genes}
    by_sample = defaultdict(list)
    for annotation in normalized:
        if annotation.path not in path_samples:
            raise ValueError(f"Annotation path absent from graph: {annotation.path}")
        sample = path_samples[annotation.path]
        if sample not in supplied:
            raise ValueError("Annotation records contradict missing-input sample status")
        by_sample[sample].append(annotation)
    rows = []
    for sample in samples:
        records = by_sample[sample]
        genes = {item.gene for item in records}
        rows.append(
            {
                "sample": sample,
                "annotation_status": "supplied" if sample in supplied else "missing",
                "feature_count": len(records) if sample in supplied else None,
                "annotated_path_count": len({item.path for item in records}),
                "graph_path_count": sum(value == sample for value in path_samples.values()),
                "expected_gene_count": len(expected),
                "expected_observed": sorted(expected & genes) if sample in supplied else None,
                "expected_not_observed": sorted(expected - genes) if sample in supplied else None,
                "observed_fraction": (
                    len(expected & genes) / len(expected)
                    if expected and sample in supplied
                    else None
                ),
            }
        )
    return {
        "rows": rows,
        "gene_synonyms": synonyms,
        "expected_genes": sorted(expected),
        "interpretation": "Coverage of caller-declared expected genes in supplied annotations; not verified biological absence or genome completeness.",
    }


def untangle_annotations(
    annotations: Sequence[Annotation],
    path_steps: Mapping[str, Sequence[Mapping]],
    *,
    reference_paths: Sequence[str],
) -> dict:
    """Map every source node interval to every occurrence on selected paths.

    This exact shared-node relation preserves repeated occurrences and strand;
    it makes no collinearity/gap-merging, homology, complete-gene or copy calls.
    It is not ODGI's approximate path untangle algorithm.
    """
    if (
        len(set(reference_paths)) != len(reference_paths)
        or set(reference_paths) - path_steps.keys()
    ):
        raise ValueError("Reference paths must be unique existing graph paths")
    index = defaultdict(list)
    for path in reference_paths:
        for step_index, step in enumerate(path_steps[path]):
            length = int(step.get("node_length", int(step["end"]) - int(step["start"])))
            if int(step["end"]) - int(step["start"]) != length or step["orientation"] not in {
                "+",
                "-",
            }:
                raise ValueError("Untangle requires exact full oriented node spans")
            index[str(step["node"])].append((path, step_index, step, length))
    rows = []
    for fragment in project_annotations(annotations, path_steps):
        for path, step_index, step, length in index[fragment["node"]]:
            start, end = fragment["node_start"], fragment["node_end"]
            strand = fragment["strand"]
            if step["orientation"] == "-":
                start, end = length - end, length - start
                if strand != ".":
                    strand = "+" if strand == "-" else "-"
            rows.append(
                {
                    "source_path": fragment["path"],
                    "target_path": path,
                    "gene": fragment["gene"],
                    "locus_id": fragment["locus_id"],
                    "source_part": fragment["part"],
                    "source_step": fragment["step"],
                    "source_start": fragment["path_start"],
                    "source_end": fragment["path_end"],
                    "target_step": step_index,
                    "node": fragment["node"],
                    "node_start": fragment["node_start"],
                    "node_end": fragment["node_end"],
                    "target_start": int(step["start"]) + start,
                    "target_end": int(step["start"]) + end,
                    "target_strand": strand,
                }
            )
    return {
        "method": "exact_shared_node_occurrences",
        "coordinate_system": "zero_based_half_open",
        "rows": rows,
        "reference_paths": list(reference_paths),
        "interpretation": "Every shared-node occurrence is retained; mappings do not establish transferred complete genes or biological copy number.",
    }
