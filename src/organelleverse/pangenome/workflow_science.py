"""Workflow adapters for explicit sequence and annotation analyses."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .graph import load_gfa
from .workflow_models import WorkflowRequest


def phylogeny_stage(graph_path: Path, directory: Path, request: WorkflowRequest):
    if request.tree_mode == "none":
        return [], "Tree inference was disabled in the submitted parameters."
    outputs = []
    if request.tree_mode == "msa":
        from .sequence_phylogeny import RaxmlOptions, alignment_records, run_raxml_ng

        manifest = json.loads((directory / "inputs.json").read_text())
        records, _ = alignment_records(
            manifest["alignment"],
            format=request.msa_format,
            source_mapping=request.msa_taxon_mapping,
        )
        graph = load_gfa(graph_path)
        cohort = (
            set(graph.paths)
            if request.msa_taxon_unit == "path"
            else {path.sample for path in graph.paths.values()}
        )
        aligned = {name for name, _ in records}
        if not aligned <= cohort:
            raise ValueError(
                f"Aligned taxa are absent from graph {request.msa_taxon_unit}s: {sorted(aligned - cohort)}"
            )
        with tempfile.TemporaryDirectory(prefix="organelleverse-raxml-") as work:
            tree = run_raxml_ng(
                manifest["alignment"],
                work,
                format=request.msa_format,
                source_mapping=request.msa_taxon_mapping,
                options=RaxmlOptions(
                    model=request.raxml_model,
                    bootstrap_replicates=request.bootstrap_replicates,
                    seed=request.msa_seed,
                    threads=request.threads,
                    parsimony_starts=request.tree_parsimony_starts,
                    random_starts=request.tree_random_starts,
                ),
            )
            from ..core.artifacts import ArtifactRef

            prepared = Path(work) / "alignment.fasta"
            prepared_ref = ArtifactRef.from_path(
                prepared, kind="sequence_alignment", format="fasta"
            )
            source_ref = ArtifactRef.from_path(
                Path(manifest["alignment"]), kind="sequence_alignment", format=request.msa_format
            )
            files = tree.pop("files")
            if prepared_ref.sha256 == source_ref.sha256:
                # The managed contract references trusted inputs instead of
                # re-publishing identical bytes as a newly produced artifact.
                files.remove(str(prepared))
                tree["prepared_alignment"] = source_ref.model_dump(mode="json")
            else:
                tree["prepared_alignment_file"] = "sequence-phylogeny/alignment.fasta"
            outputs.extend(_move_outputs(files, directory / "sequence-phylogeny"))
        tree.update(
            unit=request.msa_taxon_unit,
            cohort_taxa=sorted(cohort),
            omitted_cohort_taxa=sorted(cohort - aligned),
        )
    else:
        from .pav_store import load_pav_metadata
        from .streaming_phylogeny import stored_node_pav_tree

        if not (directory / "pav-metadata.json").exists():
            return [], "Graph contains no sample paths; a PAV tree cannot be inferred."
        if len(load_pav_metadata(directory).samples) < 2:
            return [], "At least two samples are required for a PAV similarity tree."
        tree = stored_node_pav_tree(
            directory,
            bootstrap_method=request.bootstrap_method,
            bootstrap_replicates=request.bootstrap_replicates,
            seed=request.seed,
            min_replicates=request.adaptive_min_replicates,
            max_replicates=request.adaptive_max_replicates,
            batch_size=request.adaptive_batch_size,
            convergence_threshold=request.adaptive_convergence_threshold,
        )
        tree.update(mode="pav", unit="sample")
    path = directory / "tree.json"
    path.write_text(json.dumps(tree, indent=2, allow_nan=False) + "\n")
    return [path, *outputs], None


def annotation_stage(graph_path: Path, directory: Path, request: WorkflowRequest):
    from .annotation_projection import annotate_graph, annotate_project_graph
    from .normalization import MoleculeTransform

    manifest = json.loads((directory / "inputs.json").read_text())
    options = dict(
        bin_size=request.window_bp,
        gene_synonyms=request.gene_synonyms,
        expected_genes=request.expected_genes,
        reference_paths=request.annotation_reference_paths,
    )
    if manifest.get("annotations"):
        transforms = {}
        for sample in manifest["samples"]:
            for molecule in sample["molecules"]:
                policy = molecule.get("normalization", {})
                path = molecule["path"]
                transforms[path] = MoleculeTransform(
                    path,
                    path,
                    molecule["length"],
                    molecule["topology"],
                    orientation=policy.get("orientation", "+"),
                    origin=policy.get("origin", 0),
                )
        result = annotate_graph(
            graph_path, Path(manifest["annotations"]), transforms=transforms, **options
        )
    elif any(sample.get("annotation") for sample in manifest["samples"]):
        result = annotate_project_graph(
            graph_path, manifest, Path(manifest["snapshot_directory"]), **options
        )
    else:
        if request.annotation_reference_paths or request.annotation_untangle == "odgi":
            raise ValueError("Requested annotation untangling requires supplied annotations")
        return [], "No graph-path or dataset GenBank annotation file supplied."
    outputs = []
    if request.annotation_untangle == "odgi":
        from .annotation_untangle import annotations_from_result, run_odgi_untangle

        with tempfile.TemporaryDirectory(prefix="organelleverse-untangle-") as work:
            untangle = run_odgi_untangle(
                graph_path,
                annotations_from_result(result),
                request.annotation_reference_paths,
                work,
                threads=request.threads,
            )
            outputs.extend(_move_outputs(untangle.pop("files"), directory / "annotation-untangle"))
        result["odgi_untangle"] = untangle
    from ..core.artifacts import ArtifactRef

    result["graph_ref"] = ArtifactRef.from_path(
        graph_path, kind="pangenome_graph", format="gfa"
    ).model_dump(mode="json")
    path = directory / "annotation.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return [path, *outputs], None


def _move_outputs(files, target):
    target.mkdir(exist_ok=True)
    outputs = []
    for source in map(Path, files):
        destination = target / source.name
        shutil.move(str(source), destination)
        outputs.append(destination)
    return outputs
