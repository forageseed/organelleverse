"""Canonical materialization for phylogeny results.

This is the suite half of the single publication boundary: it has the same
shape as ``organelleverse.annotation.writer.materialize_result`` and
``organelleverse.quality_control.writer.materialize_result``, so
``organelleverse.write()`` can dispatch ``operation_id.startswith("phylogeny.")``
here exactly as it already does for the released suites.

The file writers themselves (:func:`~.phylo.write_alignment`,
:func:`~.phylo.write_shared_genes`,
:func:`~.hapnet.write_haplotype_network`) are unchanged; this module only wraps
their outputs in content-addressed ``ArtifactRef`` values and returns one
canonical ``phylogeny.write`` result.
"""

from __future__ import annotations

from pathlib import Path

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from ._results import OPERATION_VERSION, operation_id

_FORMATS: dict[str, tuple[str, str]] = {
    ".fa": ("fasta", "text/plain"),
    ".fasta": ("fasta", "text/plain"),
    ".tsv": ("tsv", "text/tab-separated-values"),
    ".dot": ("dot", "text/vnd.graphviz"),
    ".png": ("png", "image/png"),
    ".svg": ("svg", "image/svg+xml"),
    ".pdf": ("pdf", "application/pdf"),
    ".newick": ("newick", "text/plain"),
    ".treefile": ("newick", "text/plain"),
}

_HAPNET_KINDS = {
    "haplotypes": "haplotype_table",
    "dot": "haplotype_network_dot",
    "image": "haplotype_network_figure",
}

__all__ = ["materialize_result"]


def _artifact(path: Path, *, kind: str) -> ArtifactRef:
    format_name, media_type = _FORMATS.get(path.suffix.lower(), ("text", "text/plain"))
    return ArtifactRef.from_path(path, kind=kind, format=format_name, media_type=media_type)


def _materialized_paths(result: OrganelleResult, output: Path) -> dict[str, tuple[Path, str]]:
    """Run the suite writer for this operation and label every produced file."""

    if result.operation_id == operation_id("align"):
        from .phylo import write_alignment

        return {"alignment": (write_alignment(result, output), "alignment")}

    if result.operation_id == operation_id("extract_shared_genes"):
        from .phylo import write_shared_genes

        paths = write_shared_genes(result, output)
        return {gene: (path, "shared_gene_alignment") for gene, path in paths.items()}

    if result.operation_id == operation_id("haplotype_network"):
        from .hapnet import write_haplotype_network

        paths = write_haplotype_network(result, output)
        return {key: (path, _HAPNET_KINDS[key]) for key, path in paths.items()}

    raise OrganelleInputError(
        code="output.unsupported_operation",
        message="phylogeny.write accepts align, extract_shared_genes, or haplotype_network results",
        details={"operation_id": result.operation_id},
    )


def materialize_result(result: OrganelleResult, output: str | Path) -> OrganelleResult:
    """Write a computed phylogeny result to ``output`` and describe the files.

    ``build_tree`` and ``trim_alignment`` already carry their outputs as
    artifacts (the external tool writes them), so they are not re-materialized
    here; only the compute-in-memory operations have a writer.
    """

    from organelleverse.visualization.plot_object import OrganellePlot

    if isinstance(result, OrganellePlot) and result.operation_id == "phylogeny.render_network":
        return result.materialize(output)
    if result.status == "failed":
        raise OrganelleInputError(
            code="output.failed_result",
            message="phylogeny.write requires a non-failed phylogeny result",
            details={"operation_id": result.operation_id},
        )
    produced = _materialized_paths(result, Path(output))
    artifacts = tuple(_artifact(path, kind=kind) for _key, (path, kind) in sorted(produced.items()))
    return OrganelleResult(
        operation_id="phylogeny.write",
        operation_version=OPERATION_VERSION,
        scope=result.scope,
        status=result.status,
        summary_text=f"Materialized {len(artifacts)} phylogeny files.",
        metrics={
            "file_count": len(artifacts),
            "source_operation_id": result.operation_id,
            "source_result_object_id": result.object_id,
            "source_result_status": result.status,
        },
        flags=tuple(dict.fromkeys(("phylogeny_materialized", *result.flags))),
        artifacts=artifacts,
        provenance=result.provenance,
    )
