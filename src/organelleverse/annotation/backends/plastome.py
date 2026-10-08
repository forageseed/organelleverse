"""Released plastome (chloroplast) annotation adapter."""

from __future__ import annotations

from pathlib import Path

from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleError,
    OrganelleExecutionError,
)
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome

from ..execution import CommandRunner, resolve_required_tools
from ..genbank import parse_genbank
from ..plastome.pipeline import PlastomeAnnotationPipeline
from ..plastome.tools import _resolve_losat
from .base import AnnotationRequest, BackendRun
from .mitochondrion import _hash_directory


def _resolve_search_tools(
    runner: CommandRunner,
) -> tuple[dict[str, str], dict[str, str]]:
    """Prefer LOSAT for every plastome search; fall back to NCBI BLAST+.

    Returns ``(tool_paths, software_versions)``.
    """
    losat = _resolve_losat(None)
    if losat is not None:
        resolved = resolve_required_tools(("losat",), runner=runner, paths={"losat": losat})
        tool = resolved[0]
        return {"losat": tool.path}, {tool.name: tool.version}
    resolved = resolve_required_tools(("blastn", "makeblastdb", "tblastn"), runner=runner)
    tool_paths = {tool.name: tool.path for tool in resolved}
    software_versions = {tool.name: tool.version for tool in resolved}
    return tool_paths, software_versions


class PlastomeBackend:
    """Strict released boundary around the plastome scientific pipeline."""

    name = "plastome"
    organelle_types = ("plastid",)

    def run(
        self,
        genome: OrganelleGenome,
        request: AnnotationRequest,
        scratch: Path,
    ) -> BackendRun:
        if genome.sequence is None:
            raise OrganelleExecutionError(
                code="backend_output_invalid",
                message="plastome backend received no sequence artifact",
            )
        runner = CommandRunner(log_dir=scratch / "command-evidence")
        tool_paths, software_versions = _resolve_search_tools(runner)

        input_dir = scratch / "backend-input"
        input_dir.mkdir(parents=True, exist_ok=True)
        sample_name = (
            genome.metadata.accession.strip()
            or genome.metadata.species.strip().replace(" ", "_")
            or "plastome_sample"
        )
        sample_fasta = input_dir / f"{sample_name}.fasta"
        sample_fasta.write_text(Path(genome.sequence.resolve()).read_text())

        output_dir = scratch / "backend-output"
        pipeline = PlastomeAnnotationPipeline(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            blastn_path=tool_paths.get("blastn"),
            makeblastdb_path=tool_paths.get("makeblastdb"),
            tblastn_path=tool_paths.get("tblastn"),
            losat_path=tool_paths.get("losat"),
            native_trna_rrna="trna" in request.stages or "rrna" in request.stages,
            refine_cds="pcg" in request.stages,
            detect_ir=True,
            threads=request.threads,
        )
        if not pipeline.verify_dependencies():
            raise OrganelleDependencyError(
                code="dependency_missing",
                message="plastome annotation dependencies are unavailable",
                details={"missing": pipeline.dependency_issues},
            )
        pipeline.run_pipeline()
        stats = pipeline.annotation_results.get(sample_name)
        if stats is None or not stats.annotation_success:
            failure = pipeline.failed_samples.get(sample_name, "plastome pipeline produced no annotation")
            raise OrganelleExecutionError(
                code="backend_output_invalid",
                message="plastome pipeline did not produce a successful annotation",
                details={"reason": failure},
            )
        genbank_path = stats.output_genbank
        if genbank_path is None or not genbank_path.is_file():
            raise OrganelleExecutionError(
                code="backend_output_invalid",
                message="plastome pipeline did not produce GenBank output",
            )
        try:
            parsed = parse_genbank(genbank_path)
        except OrganelleError as error:
            raise OrganelleExecutionError(
                code="backend_output_invalid",
                message="plastome GenBank output is malformed",
                details={"parser_error": error.code},
            ) from error
        document = parsed.evolve(
            backend=self.name,
            requested_stages=request.stages,
            completed_stages=request.stages,
            source_metadata=FrozenMap(
                {
                    **dict(parsed.source_metadata),
                    "species": genome.metadata.species,
                    "accession": genome.metadata.accession,
                    "genetic_code": genome.metadata.genetic_code,
                    "unannotated_genes": tuple(stats.unannotated_genes),
                    "pipeline_warnings": tuple(stats.warnings),
                }
            ),
        )
        database_hashes = {
            "plastome_references": _hash_directory(pipeline.reference_dir),
        }
        logs = (runner.log_path,) if runner.log_path is not None else ()
        return BackendRun(
            document=document,
            commands=runner.records,
            software_versions=software_versions,
            database_hashes=database_hashes,
            logs=logs,
        )
