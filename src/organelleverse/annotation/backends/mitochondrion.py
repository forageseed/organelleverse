"""Released mitochondrial annotation adapter."""

from __future__ import annotations

import shutil

import hashlib
from pathlib import Path

from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleError,
    OrganelleExecutionError,
    OrganelleParameterError,
)
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.genome import OrganelleGenome, OrganelleType

from ..._losat import resolve_losat
from ..execution import CommandRunner, resolve_required_tools
from ..genbank import parse_genbank
from ..contigs import (
    apply_molecule_headers,
    contig_layout,
    molecule_headers,
    split_document_by_contigs,
)
from ..mitochondrion.db import DBManager
from ..mitochondrion.pipeline import MitochondrialAnnotationPipeline
from .base import AnnotationRequest, BackendRun


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_hash_file(item)))
    return digest.hexdigest()


_NCBI_SEARCH_TOOLS = ("blastn", "makeblastdb", "tblastn")


def _resolve_search_tools(runner: CommandRunner) -> tuple[dict[str, str], dict[str, str]]:
    """Prefer LOSAT for mitochondrial sequence searches; NCBI BLAST+ is the fallback.

    With LOSAT available, NCBI executables are still resolved when present so a
    stage that has no LOSAT path keeps working, but none of them is required.
    Without LOSAT (or with ``ORG_VERSE_LOSAT_BIN=ncbi``) all three NCBI tools are
    required, exactly as before. Returns ``(tool_paths, software_versions)``.
    """
    losat = resolve_losat()
    if losat is None:
        resolved = resolve_required_tools(_NCBI_SEARCH_TOOLS, runner=runner)
    else:
        resolved = resolve_required_tools(("losat",), runner=runner, paths={"losat": losat})
        available = tuple(name for name in _NCBI_SEARCH_TOOLS if shutil.which(name))
        if available:
            resolved = resolved + resolve_required_tools(available, runner=runner)
    tool_paths = {tool.name: tool.path for tool in resolved}
    software_versions = {tool.name: tool.version for tool in resolved}
    return tool_paths, software_versions


class MitochondrionBackend:
    """Strict released boundary around the mitochondrial scientific pipeline."""

    name = "mitochondrion"
    organelle_types: tuple[OrganelleType, ...] = ("mitochondrion",)

    def run(
        self,
        genome: OrganelleGenome,
        request: AnnotationRequest,
        scratch: Path,
    ) -> BackendRun:
        species = genome.metadata.species.strip()
        if not species:
            raise OrganelleParameterError(
                code="unsupported_annotation_scope",
                message="released mitochondrial annotation requires species metadata",
                details={"missing_metadata": ["species"]},
                suggested_action={"provide_metadata": ["species"]},
            )
        genus = species.split(maxsplit=1)[0].casefold()
        if "trna" in request.stages and genus == "pinus":
            raise OrganelleParameterError(
                code="unsupported_annotation_scope",
                message="released Pinus tRNA annotation is outside the validated scope",
                details={"species": species, "unsupported_stage": "trna"},
                suggested_action={"omit_stage": "trna"},
            )

        database = DBManager()
        database_issues = database.verify()
        if database_issues:
            raise OrganelleDependencyError(
                code="dependency_missing",
                message="packaged mitochondrial annotation database is incomplete",
                details={"database_issues": database_issues},
            )

        runner = CommandRunner(log_dir=scratch / "command-evidence")
        tool_paths, search_tool_versions = _resolve_search_tools(runner)

        if genome.sequence is None:
            raise OrganelleExecutionError(
                code="backend_output_invalid",
                message="mitochondrial backend received no sequence artifact",
            )
        output_dir = scratch / "backend-output"
        sample_name = genome.metadata.accession.strip() or species.replace(" ", "_")
        pipeline = MitochondrialAnnotationPipeline(
            input_fasta=genome.sequence.resolve(),
            output_dir=output_dir,
            name=sample_name,
            threads=request.threads,
            call_pcg="pcg" in request.stages,
            call_trna="trna" in request.stages,
            call_rrna="rrna" in request.stages,
            trna_engine="native",
            rrna_engine="pyhmmer",
            tool_paths=tool_paths,
            command_runner=runner,
        )
        stats = pipeline.run_pipeline()
        if stats is None:
            raise OrganelleExecutionError(
                code="backend_output_invalid",
                message="mitochondrial pipeline did not produce annotation statistics",
                details={"dependency_issues": pipeline.dependency_issues},
            )
        genbank_path = next(
            (path for path in stats.output_paths if path.suffix.casefold() in {".gb", ".gbk"}),
            None,
        )
        if genbank_path is None or not genbank_path.is_file():
            raise OrganelleExecutionError(
                code="backend_output_invalid",
                message="mitochondrial pipeline did not produce GenBank output",
            )
        try:
            parsed = parse_genbank(genbank_path)
        except OrganelleError as error:
            raise OrganelleExecutionError(
                code="backend_output_invalid",
                message="mitochondrial GenBank output is malformed",
                details={"parser_error": error.code},
            ) from error
        document = parsed.evolve(
            backend=self.name,
            requested_stages=request.stages,
            completed_stages=request.stages,
            source_metadata=FrozenMap(
                {
                    **dict(parsed.source_metadata),
                    "species": species,
                    "accession": genome.metadata.accession,
                    "genetic_code": genome.metadata.genetic_code,
                    "missing_core_genes": stats.missing_core_genes,
                    "rejected_cds_candidates": tuple(
                        {
                            "gene_name": candidate.gene_name,
                            "start": candidate.start,
                            "end": candidate.end,
                            "strand": candidate.strand,
                            "parts": tuple(
                                {"start": start, "end": end, "strand": strand}
                                for start, end, strand in candidate.parts
                            ),
                            "issue_codes": candidate.issue_codes,
                            "issue_messages": candidate.issue_messages,
                        }
                        for candidate in stats.rejected_cds_candidates
                    ),
                }
            ),
        )
        # The pipeline annotates contigs joined by 200 N; hand back one record per contig.
        document = split_document_by_contigs(document, contig_layout(genome.sequence.resolve()))
        # ovasm molecules say whether they are circular; GenBank LOCUS lines should too
        document = apply_molecule_headers(document, molecule_headers(genome.sequence.resolve()))
        database_hashes = {
            "mitochondrion_hmm": _hash_file(database.combined_hmm),
            "mitochondrion_gene_info": _hash_directory(database.gene_info_dir),
        }
        if "pcg" in request.stages:
            database_hashes["mitochondrion_pcg_blast_refs"] = _hash_directory(
                database.blast_ref_dir
            )
            database_hashes["mitochondrion_exon_refs"] = _hash_directory(database.exon_ref_dir)
        if "trna" in request.stages:
            from ..cmsearch.genome import default_cm_path, default_hmm_path

            database_hashes["mitochondrion_trna_cm"] = _hash_file(default_cm_path())
            database_hashes["mitochondrion_trna_hmm"] = _hash_file(default_hmm_path())
        if "rrna" in request.stages:
            database_hashes["mitochondrion_rrna_refs"] = _hash_directory(database.rrna_ref_dir)
        logs = (runner.log_path,) if runner.log_path is not None else ()
        return BackendRun(
            document=document,
            commands=runner.records,
            software_versions=search_tool_versions,
            database_hashes=database_hashes,
            logs=logs,
        )
