"""Exact-coordinate annotation normalization and graph-node projection.

Coordinates are zero-based half-open. Projection follows named graph paths;
it does not infer homology, repair annotations, or transfer genes between paths.
Compound features retain separate parts, including circular-origin joins.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from Bio import SeqIO
from Bio.SeqFeature import ExactPosition


@dataclass(frozen=True)
class Annotation:
    """One biological locus; parts do not increase copy count."""

    path: str
    gene: str
    locus_id: str
    parts: tuple[tuple[int, int], ...]
    strand: str = "+"
    feature_type: str = "gene"
    part_strands: tuple[str, ...] | None = None
    source_locus_tag: str | None = None

    def strand_for_part(self, index: int) -> str:
        """Preserve biological strand separately for trans-spliced feature parts."""
        return self.part_strands[index] if self.part_strands is not None else self.strand

    def __post_init__(self) -> None:
        if not self.path or not self.gene or not self.locus_id:
            raise ValueError("Annotation path, gene and locus ID must be nonempty")
        if self.strand not in {"+", "-", "."}:
            raise ValueError("Annotation strand must be +, - or .")
        if self.part_strands is not None and (
            len(self.part_strands) != len(self.parts)
            or any(strand not in {"+", "-", "."} for strand in self.part_strands)
        ):
            raise ValueError("Every annotation part requires a valid corresponding strand")
        if not self.parts or any(start < 0 or end <= start for start, end in self.parts):
            raise ValueError("Annotation parts must be nonempty zero-based half-open intervals")
        ordered = sorted(self.parts)
        if any(left[1] > right[0] for left, right in itertools.pairwise(ordered)):
            raise ValueError("Parts of one annotation must not overlap")


def normalize_annotations(
    annotations: Sequence[Annotation],
    *,
    path_mapping: Mapping[str, str] | None = None,
    gene_synonyms: Mapping[str, str] | None = None,
) -> list[Annotation]:
    """Apply explicit, case-sensitive mappings without guessing path identities."""
    result = [
        replace(
            annotation,
            path=(path_mapping or {}).get(annotation.path, annotation.path),
            gene=(gene_synonyms or {}).get(annotation.gene, annotation.gene),
        )
        for annotation in annotations
    ]
    keys = [(annotation.path, annotation.locus_id) for annotation in result]
    if len(set(keys)) != len(keys):
        raise ValueError("Locus IDs must be unique within each path")
    return result


def read_bed(path: str | Path) -> list[Annotation]:
    """Read BED6; every data row is a locus, preserving repeated gene names."""
    result = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line or line.startswith(("#", "track ", "browser ")):
            continue
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"BED line {line_number}: expected BED6")
        chrom, start, end, gene, _score, strand = fields
        result.append(
            Annotation(chrom, gene, f"bed:{line_number}", ((int(start), int(end)),), strand)
        )
    return normalize_annotations(result)


def read_gff(path: str | Path, *, feature_type: str = "gene") -> list[Annotation]:
    """Read GFF3 features of the declared type, retaining multipart shared IDs."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if line == "##FASTA":
            break
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 9:
            raise ValueError(f"GFF line {line_number}: expected nine columns")
        chrom, _source, kind, start, end, _score, strand, _phase, raw_attributes = fields
        if kind != feature_type:
            continue
        attributes = {}
        for item in raw_attributes.split(";"):
            if item:
                key, value = item.split("=", 1)
                attributes[unquote(key)] = unquote(value)
        locus = attributes.get("ID")
        if not locus:
            raise ValueError(f"GFF line {line_number}: selected feature requires ID")
        gene = attributes.get("gene") or attributes.get("Name") or locus
        key = (chrom, locus)
        item = grouped.setdefault(key, {"gene": gene, "strand": strand, "parts": []})
        if item["gene"] != gene or item["strand"] != strand:
            raise ValueError(f"GFF locus {locus}: inconsistent gene name or strand")
        item["parts"].append((int(start) - 1, int(end)))
    return normalize_annotations(
        [
            Annotation(
                chrom, item["gene"], locus, tuple(item["parts"]), item["strand"], feature_type
            )
            for (chrom, locus), item in grouped.items()
        ]
    )


def read_genbank(path: str | Path, *, feature_type: str = "gene") -> list[Annotation]:
    """Read one explicit GenBank feature type, preserving split loci and strand.

    Mixed-strand/trans-spliced joins preserve each part strand and remain one
    source feature. Record-local feature ordinals identify features uniquely;
    repeated source locus tags are retained as metadata, never used to merge.
    Fuzzy positions and remote features require explicit normalization.
    """
    result = []
    for record in SeqIO.parse(str(path), "genbank"):
        for index, feature in enumerate(record.features):
            if feature.type != feature_type:
                continue
            if feature.location is None:
                raise ValueError(f"GenBank {record.id}: selected feature has no location")
            parts = feature.location.parts
            if any(
                part.ref is not None
                or not isinstance(part.start, ExactPosition)
                or not isinstance(part.end, ExactPosition)
                for part in parts
            ):
                raise ValueError(
                    "GenBank exact local coordinates are required for graph projection"
                )
            strands = {part.strand for part in parts}
            gene = (
                feature.qualifiers.get("gene")
                or feature.qualifiers.get("locus_tag")
                or [f"{feature_type}:{index}"]
            )[0]
            source_locus_tag = (feature.qualifiers.get("locus_tag") or [None])[0]
            locus = f"genbank:{index}"
            part_strands = tuple(
                {1: "+", -1: "-", None: ".", 0: "."}[part.strand] for part in parts
            )
            strand = part_strands[0] if len(strands) == 1 else "."
            result.append(
                Annotation(
                    record.id,
                    gene,
                    locus,
                    tuple((int(part.start), int(part.end)) for part in parts),
                    strand,
                    feature_type,
                    part_strands=part_strands if len(strands) > 1 else None,
                    source_locus_tag=source_locus_tag,
                )
            )
    return normalize_annotations(result)


def _union_length(intervals: Sequence[tuple[int, int]]) -> int:
    total = 0
    current_end = -1
    for start, end in sorted(intervals):
        total += max(0, end - max(start, current_end))
        current_end = max(current_end, end)
    return total


def project_annotations(
    annotations: Sequence[Annotation],
    path_steps: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Project exact path intervals onto every intersected oriented graph node.

    Each step provides node, orientation, start, end and optional node_length.
    Overlap-aware step spans may overlap: both node mappings are preserved.
    """
    annotations = normalize_annotations(annotations)
    fragments = []
    for annotation in annotations:
        if annotation.path not in path_steps:
            raise ValueError(f"Annotation path absent from graph: {annotation.path}")
        steps = path_steps[annotation.path]
        for part_index, (start, end) in enumerate(annotation.parts):
            covered = []
            for step_index, step in enumerate(steps):
                step_start, step_end = int(step["start"]), int(step["end"])
                length = int(step.get("node_length", step_end - step_start))
                orientation = step["orientation"]
                if length <= 0 or step_end - step_start != length or orientation not in {"+", "-"}:
                    raise ValueError("Projection requires exact, full oriented node spans")
                overlap_start, overlap_end = max(start, step_start), min(end, step_end)
                if overlap_start >= overlap_end:
                    continue
                covered.append((overlap_start, overlap_end))
                local_start, local_end = overlap_start - step_start, overlap_end - step_start
                if orientation == "-":
                    local_start, local_end = length - local_end, length - local_start
                node_strand = annotation.strand_for_part(part_index)
                if orientation == "-" and node_strand != ".":
                    node_strand = "+" if node_strand == "-" else "-"
                fragments.append(
                    {
                        "path": annotation.path,
                        "gene": annotation.gene,
                        "locus_id": annotation.locus_id,
                        "source_locus_tag": annotation.source_locus_tag,
                        "part": part_index,
                        "step": step_index,
                        "node": str(step["node"]),
                        "orientation": orientation,
                        "path_start": overlap_start,
                        "path_end": overlap_end,
                        "node_start": local_start,
                        "node_end": local_end,
                        "strand": node_strand,
                    }
                )
            if _union_length(covered) != end - start:
                raise ValueError(
                    f"Annotation {annotation.path}/{annotation.locus_id} has unrepresented coordinates"
                )
    return fragments


def annotation_tables(
    annotations: Sequence[Annotation],
    path_bounds: Mapping[str, tuple[int, int]],
    *,
    path_samples: Mapping[str, str] | None = None,
    bin_size: int = 1000,
) -> dict[str, Any]:
    """Gene copy/PAV, arrow-part and union-coverage tables from normalized loci.

    Unannotated sequence is not evidence of gene absence: zeros indicate no
    supplied annotation and require independent annotation completeness QC.
    Sample grouping must be explicit; default samples are graph paths.
    """
    if isinstance(bin_size, bool) or not isinstance(bin_size, int) or bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")
    annotations = normalize_annotations(annotations)
    if any(start < 0 or end <= start for start, end in path_bounds.values()):
        raise ValueError("Path bounds must be nonempty zero-based half-open intervals")
    samples_by_path = (
        dict(path_samples) if path_samples is not None else {path: path for path in path_bounds}
    )
    if set(samples_by_path) != set(path_bounds) or any(
        not sample for sample in samples_by_path.values()
    ):
        raise ValueError("path_samples must name every graph path exactly once")
    genes = sorted({annotation.gene for annotation in annotations})
    samples = list(dict.fromkeys(samples_by_path.values()))
    counts: dict[tuple[str, str], int] = defaultdict(int)
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    arrows = []
    for annotation in annotations:
        if annotation.path not in path_bounds:
            raise ValueError(f"Annotation path absent from graph: {annotation.path}")
        bound_start, bound_end = path_bounds[annotation.path]
        if any(start < bound_start or end > bound_end for start, end in annotation.parts):
            raise ValueError(f"Annotation outside graph path bounds: {annotation.locus_id}")
        sample = samples_by_path[annotation.path]
        counts[(annotation.gene, sample)] += 1
        intervals[annotation.path].extend(annotation.parts)
        for part, (start, end) in enumerate(annotation.parts):
            arrows.append(
                {
                    "path": annotation.path,
                    "sample": sample,
                    "gene": annotation.gene,
                    "locus_id": annotation.locus_id,
                    "source_locus_tag": annotation.source_locus_tag,
                    "part": part,
                    "start": start,
                    "end": end,
                    "strand": annotation.strand_for_part(part),
                }
            )
    bins = []
    for path, (bound_start, bound_end) in path_bounds.items():
        for start in range(bound_start, bound_end, bin_size):
            end = min(start + bin_size, bound_end)
            covered = _union_length(
                [
                    (max(start, left), min(end, right))
                    for left, right in intervals[path]
                    if left < end and right > start
                ]
            )
            bins.append(
                {
                    "path": path,
                    "start": start,
                    "end": end,
                    "annotated_bp": covered,
                    "annotated_fraction": covered / (end - start),
                }
            )
    copies = [[counts[(gene, sample)] for sample in samples] for gene in genes]
    return {
        "genes": genes,
        "samples": samples,
        "copy_matrix": copies,
        "pav_matrix": [[int(value > 0) for value in row] for row in copies],
        "gene_arrows": arrows,
        "bin_coverage": bins,
        "interpretation": (
            "Counts measure supplied feature records, grouping compound parts as one copy. "
            "Separately annotated fragments are not inferred to form one trans-spliced gene. "
            "Zero indicates no supplied annotation, not validated biological absence. "
            "Bin coverage is the union of annotated bases, not sequencing depth."
        ),
    }


def annotate_graph(
    gfa_path: str | Path,
    annotations_path: str | Path,
    *,
    bin_size: int = 1000,
    annotation_format: str | None = None,
    path_mapping: Mapping[str, str] | None = None,
    gene_synonyms: Mapping[str, str] | None = None,
    feature_type: str = "gene",
    transforms: Mapping[str, Any] | None = None,
    expected_genes: Sequence[str] = (),
    reference_paths: Sequence[str] = (),
) -> dict[str, Any]:
    """Normalize and exactly project BED6, GFF3 or GenBank against a GFA."""
    from organelleverse.pangenome.graph import load_gfa, path_steps

    from .annotation_audit import audit_annotations, untangle_annotations, validate_gene_synonyms

    gene_synonyms = validate_gene_synonyms(gene_synonyms or {})
    source = Path(annotations_path)
    kind = (annotation_format or source.suffix.lstrip(".")).lower()
    if kind in {"bed", "bed6"}:
        annotations = read_bed(source)
    elif kind in {"gff", "gff3"}:
        annotations = read_gff(source, feature_type=feature_type)
    elif kind in {"gb", "gbk", "genbank", "gbff"}:
        annotations = read_genbank(source, feature_type=feature_type)
    else:
        raise ValueError("Annotation format must be BED6, GFF3 or GenBank")
    annotations = normalize_annotations(
        annotations, path_mapping=path_mapping, gene_synonyms=gene_synonyms
    )
    if transforms:
        annotations = normalize_annotations(
            [
                transforms[feature.path].annotation(feature)
                if feature.path in transforms
                else feature
                for feature in annotations
            ]
        )
    graph = load_gfa(gfa_path)
    for annotation in annotations:
        if annotation.path not in graph.paths:
            raise ValueError(f"Annotation path absent from graph: {annotation.path}")
        record = graph.paths[annotation.path]
        if record.kind == "W" and record.start is None:
            raise ValueError("Annotation projection requires a declared GFA walk start coordinate")
    steps = path_steps(gfa_path)
    bounds = {
        path: (min(step["start"] for step in rows), max(step["end"] for step in rows))
        for path, rows in steps.items()
    }
    samples = {path: graph.paths[path].sample for path in steps}
    result = annotation_tables(annotations, bounds, path_samples=samples, bin_size=bin_size)
    result["projection"] = project_annotations(annotations, steps)
    result["path_bounds"] = bounds
    result["annotation_count"] = len(annotations)
    result["annotation_format"] = kind
    result["coordinate_policy"] = (
        "declared source-to-graph transforms" if transforms else "graph path coordinates"
    )
    result["annotation_audit"] = audit_annotations(
        annotations,
        path_samples=samples,
        annotated_samples=sorted({samples[feature.path] for feature in annotations}),
        expected_genes=expected_genes,
        gene_synonyms=gene_synonyms,
    )
    if reference_paths:
        result["untangle"] = untangle_annotations(
            annotations, steps, reference_paths=reference_paths
        )
    return result


def annotate_project_graph(
    gfa_path: str | Path,
    manifest: Mapping[str, Any],
    input_root: str | Path,
    *,
    bin_size: int = 1000,
    gene_synonyms: Mapping[str, str] | None = None,
    feature_type: str = "gene",
    expected_genes: Sequence[str] = (),
    reference_paths: Sequence[str] = (),
) -> dict[str, Any]:
    """Project preserved project GenBank loci using per-sample identity maps.

    Mapping is applied separately per source, so record IDs repeated between
    samples never collide. Exact sequence equality is required: an unrecorded
    rotation, reverse complement or edited path cannot reuse source coordinates.
    Recorded molecule normalization is applied to both sequence and annotations.
    An audit distinguishes missing annotation input from zero selected features.
    """
    from organelleverse.pangenome.graph import load_gfa, path_sequences, path_steps

    from .annotation_audit import audit_annotations, untangle_annotations, validate_gene_synonyms
    from .normalization import MoleculeTransform

    gene_synonyms = validate_gene_synonyms(gene_synonyms or {})

    root = Path(input_root).resolve()
    graph = load_gfa(gfa_path)
    sequences = path_sequences(gfa_path)
    annotations: list[Annotation] = []
    source_records: list[dict[str, Any]] = []
    sample_names: set[str] = set()
    mapped_paths: set[str] = set()
    supplied_samples = []
    for sample in manifest.get("samples", []):
        sample_name = sample["name"]
        if sample_name in sample_names:
            raise ValueError(f"Duplicate project sample name: {sample_name}")
        sample_names.add(sample_name)
        if not sample.get("annotation"):
            continue
        supplied_samples.append(sample_name)
        source = (root / sample["annotation"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Project annotation must be inside the declared input snapshot")
        molecules = sample["molecules"]
        mapping = {molecule["source_id"]: molecule["path"] for molecule in molecules}
        if len(mapping) != len(molecules) or len(set(mapping.values())) != len(mapping):
            raise ValueError(f"Project sample {sample_name} has ambiguous molecule identities")
        if mapped_paths.intersection(mapping.values()):
            raise ValueError("Project samples map to the same graph path")
        mapped_paths.update(mapping.values())
        records = list(SeqIO.parse(str(source), "genbank"))
        record_ids = [record.id for record in records]
        if len(set(record_ids)) != len(record_ids) or set(record_ids) != set(mapping):
            raise ValueError(f"Project GenBank records do not match molecule map: {sample_name}")
        molecule_metadata = {molecule["source_id"]: molecule for molecule in molecules}
        transforms = {}
        for record in records:
            path = mapping[record.id]
            if path not in graph.paths:
                raise ValueError(f"Mapped annotation path absent from graph: {path}")
            graph_path = graph.paths[path]
            if graph_path.sample != sample_name:
                raise ValueError(f"Mapped graph path sample disagrees with project: {path}")
            if graph_path.kind == "W" and graph_path.start != 0:
                raise ValueError(
                    "Whole-record GenBank projection requires a graph path starting at zero"
                )
            metadata = molecule_metadata[record.id]
            policy = metadata.get("normalization", {})
            transform = MoleculeTransform(
                source_path=record.id,
                output_path=path,
                length=len(record.seq),
                topology=metadata.get("topology", "unknown"),
                orientation=policy.get("orientation", "+"),
                origin=policy.get("origin", 0),
            )
            transforms[record.id] = transform
            if transform.sequence(str(record.seq).upper()) != sequences[path]:
                raise ValueError(
                    f"Mapped graph path sequence differs from original GenBank record: {path}"
                )
            source_records.append(
                {
                    "sample": sample_name,
                    "source_id": record.id,
                    "path": path,
                    "length": len(record.seq),
                    "sequence_coordinate_check": (
                        "exact_sequence_match_after_declared_transform"
                        if transform.orientation != "+" or transform.origin
                        else "exact_sequence_match"
                    ),
                    "normalization": {
                        "orientation": transform.orientation,
                        "origin": transform.origin,
                    },
                    "coordinate_mapping": transform.coordinate_rows(),
                }
            )
        annotations.extend(
            normalize_annotations(
                [
                    transforms[feature.path].annotation(feature)
                    for feature in read_genbank(source, feature_type=feature_type)
                ],
                gene_synonyms=gene_synonyms,
            )
        )
    steps = path_steps(gfa_path)
    bounds = {
        path: (min(step["start"] for step in rows), max(step["end"] for step in rows))
        for path, rows in steps.items()
    }
    samples = {path: graph.paths[path].sample for path in steps}
    result = annotation_tables(annotations, bounds, path_samples=samples, bin_size=bin_size)
    result["projection"] = project_annotations(annotations, steps)
    result["path_bounds"] = bounds
    result["annotation_count"] = len(annotations)
    result["annotation_format"] = "project_genbank"
    result["source_annotation_records"] = source_records
    result["annotation_audit"] = audit_annotations(
        annotations,
        path_samples=samples,
        annotated_samples=supplied_samples,
        expected_genes=expected_genes,
        gene_synonyms=gene_synonyms,
    )
    if reference_paths:
        result["untangle"] = untangle_annotations(
            annotations, steps, reference_paths=reference_paths
        )
    result["sample_annotation_counts"] = {
        sample: sum(samples[annotation.path] == sample for annotation in annotations)
        for sample in result["samples"]
    }
    return result
