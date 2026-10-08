"""PMAT2 v2.1.5 ``autoMito`` adapter."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import BinaryIO

from organelleverse.assembly.backends.base import (
    AdapterContext,
    AssemblyCommand,
    BackendOutput,
    NormalizedAssemblyOutputs,
    RawAssemblyOutputs,
)
from organelleverse.assembly.continuation import DirectoryManifest, DirectoryManifestFile
from organelleverse.assembly.contracts import PmatParameters
from organelleverse.assembly.data_contract import validate_pmat_assembly_data
from organelleverse.assembly.normalization import (
    normalize_fasta,
    normalize_gfa,
    validate_fasta_against_gfa,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError

__all__ = ["PmatAdapter"]


class PmatAdapter:
    """Build one safe, fully explicit PMAT2 ``autoMito`` invocation."""

    backend_id = "pmat"

    def preflight(self, context: AdapterContext) -> None:
        if context.route.selected_backend != self.backend_id:
            raise _unsupported("PMAT adapter selected for a non-PMAT route")
        if context.environment.backend_id != self.backend_id:
            raise _unsupported("PMAT requires a verified PMAT environment")
        validate_pmat_assembly_data(context.request.data)
        provided = context.request.backend_parameters
        if provided is not None and not isinstance(provided, PmatParameters):
            raise _unsupported("backend_parameters discriminator does not match PMAT")

        formats = {
            _sequence_format(context.input_artifacts[library.reads_artifact])
            for library in context.payload.long_libraries
        }
        if len(formats) != 1:
            raise _unsupported("PMAT libraries must use one sequence file format")

        correction_task = context.effective_backend_parameters["correction_task"]
        software = context.effective_backend_parameters["correction_software"]
        auxiliary = context.payload.auxiliary
        if correction_task == "skip" and auxiliary.correction_config_artifact is not None:
            raise _unsupported("PMAT correction config cannot be used when correction is skipped")
        if software == "canu" and auxiliary.correction_config_artifact is not None:
            raise _unsupported("PMAT correction config is only consumed by NextDenovo")
        if correction_task == "run":
            _require_tool(context, "canu", auxiliary.canu_executable_artifact)
            if software == "nextdenovo":
                _require_tool(
                    context,
                    "nextdenovo",
                    auxiliary.nextdenovo_executable_artifact,
                )

    def build_command(self, context: AdapterContext) -> AssemblyCommand:
        self.preflight(context)
        params = context.effective_backend_parameters
        input_path = _materialize_reads(context)
        output_path = context.workspace / "backend" / self.backend_id
        output_path.mkdir(parents=True, exist_ok=True)

        stable = [
            "PMAT",
            "autoMito",
            "-i",
            "role://workspace/input",
            "-o",
            "role://workspace/output",
            "-t",
            str(params["seqtype"]),
            "-k",
            str(params["kmer_size"]),
        ]
        resolved = [
            str(context.environment.require_executable("pmat")),
            "autoMito",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-t",
            str(params["seqtype"]),
            "-k",
            str(params["kmer_size"]),
        ]
        genome_size = params["genome_size"]
        if genome_size is not None:
            _extend_both(stable, resolved, "-g", str(genome_size), str(genome_size))
        _extend_both(
            stable,
            resolved,
            "-p",
            "1" if params["correction_task"] == "run" else "0",
            "1" if params["correction_task"] == "run" else "0",
        )
        for flag, key in (("-G", "target"), ("-x", "taxon"), ("-S", "correction_software")):
            value = str(params[key])
            _extend_both(stable, resolved, flag, value, value)

        if params["correction_task"] == "run":
            _append_tool(
                stable,
                resolved,
                context,
                flag="-C",
                name="canu",
                artifact_role=context.payload.auxiliary.canu_executable_artifact,
            )
            if params["correction_software"] == "nextdenovo":
                _append_tool(
                    stable,
                    resolved,
                    context,
                    flag="-N",
                    name="nextdenovo",
                    artifact_role=context.payload.auxiliary.nextdenovo_executable_artifact,
                )
                config_role = context.payload.auxiliary.correction_config_artifact
                if config_role is not None:
                    config = _materialize_artifact(
                        context,
                        config_role,
                        context.workspace / "input" / "correction.cfg",
                        executable=False,
                    )
                    _extend_both(
                        stable,
                        resolved,
                        "-n",
                        f"role://artifact/{config_role}",
                        str(config),
                    )

        for flag, key in (
            ("-F", "subsample_factor"),
            ("-D", "random_seed"),
            ("-K", "long_read_break_length"),
            ("-I", "minimum_overlap_identity"),
            ("-L", "minimum_overlap_length"),
        ):
            value = str(params[key])
            _extend_both(stable, resolved, flag, value, value)
        threads = str(context.request.threads)
        _extend_both(stable, resolved, "-T", threads, threads)
        if params["keep_sequences_in_memory"]:
            stable.append("-m")
            resolved.append("-m")
        return AssemblyCommand(stable_argv=tuple(stable), resolved_argv=tuple(resolved))

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs:
        if _uses_source_reads(context):
            # Direct inputs remain user-owned. Reject a changed source before
            # publishing outputs, retaining the content boundary without a copy.
            artifact = context.input_artifacts[context.payload.long_libraries[0].reads_artifact]
            _verify_hash(artifact)
        root = context.workspace / "backend" / self.backend_id
        target = "mt" if context.request.organelle == "mitochondrion" else "pt"
        fasta_name = "PMAT_mt.fa" if target == "mt" else "PMAT_pt.fa"
        declared = [
            ("assembly_fasta", root / "gfa_result" / fasta_name, "fasta", "text/x-fasta"),
            (
                "assembly_graph",
                root / "gfa_result" / f"PMAT_{target}_main.gfa",
                "gfa",
                "text/plain",
            ),
            (
                "raw_assembly_graph",
                root / "gfa_result" / f"PMAT_{target}_raw.gfa",
                "gfa",
                "text/plain",
            ),
            (
                "pmat_all_contigs",
                root / "assembly_result" / "PMATAllContigs.fna",
                "fasta",
                "text/x-fasta",
            ),
            (
                "pmat_contig_graph",
                root / "assembly_result" / "PMATContigGraph.txt",
                "txt",
                "text/plain",
            ),
            (
                "pmat_subsample",
                root / "subsample" / "PMAT_cut_seq.fa",
                "fasta",
                "text/x-fasta",
            ),
        ]
        if target == "mt":
            declared.append(("assembly_assessment", root / "PMAT_orgAss.txt", "txt", "text/plain"))
        outputs: list[BackendOutput] = []
        for role, path, format_name, media_type in declared:
            _require_output(path, role)
            outputs.append(
                BackendOutput(
                    role=role,
                    path=path,
                    format=format_name,
                    media_type=media_type,
                )
            )
        if target == "mt":
            assessment = next(
                item.path for item in outputs if item.role == "assembly_assessment"
            ).read_text(encoding="utf-8", errors="replace")
            if "mitochond" not in assessment.lower():
                raise _output_error("PMAT assessment does not match the requested organelle")
        return RawAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="assembly_fasta",
            primary_graph_role="assembly_graph",
        )

    def normalize(
        self,
        context: AdapterContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        destinations = {
            "assembly_fasta": output_dir / "assembly.fasta",
            "assembly_graph": output_dir / "assembly.gfa",
            "raw_assembly_graph": output_dir / "assembly.raw.gfa",
            "assembly_assessment": output_dir / "assessment.txt",
            "pmat_all_contigs": output_dir / "pmat.all_contigs.fasta",
            "pmat_contig_graph": output_dir / "pmat.contig_graph.txt",
            "pmat_subsample": output_dir / "pmat.subsample.fasta",
        }
        fasta = normalize_fasta(
            raw.require("assembly_fasta").path,
            destinations["assembly_fasta"],
        )
        graph = normalize_gfa(
            raw.require("assembly_graph").path,
            destinations["assembly_graph"],
        )
        validate_fasta_against_gfa(fasta, graph)
        normalize_gfa(
            raw.require("raw_assembly_graph").path,
            destinations["raw_assembly_graph"],
        )
        normalize_fasta(
            raw.require("pmat_all_contigs").path,
            destinations["pmat_all_contigs"],
        )
        normalize_fasta(
            raw.require("pmat_subsample").path,
            destinations["pmat_subsample"],
        )
        for role in ("pmat_contig_graph",):
            shutil.copyfile(raw.require(role).path, destinations[role])
        if any(item.role == "assembly_assessment" for item in raw.outputs):
            shutil.copyfile(
                raw.require("assembly_assessment").path,
                destinations["assembly_assessment"],
            )
        outputs = [
            BackendOutput(
                role=item.role,
                path=destinations[item.role],
                format=item.format,
                media_type=item.media_type,
            )
            for item in raw.outputs
        ]
        continuation_artifacts = {
            role: ArtifactRef.from_path(
                destinations[role],
                kind="pmat_continuation",
                format=next(item.format for item in raw.outputs if item.role == role),
                media_type="application/octet-stream",
            )
            for role in ("pmat_subsample", "pmat_all_contigs", "pmat_contig_graph")
        }
        manifests = (
            DirectoryManifest(
                role="pmat_subsample",
                files=(
                    DirectoryManifestFile(
                        relative_path="PMAT_cut_seq.fa",
                        artifact_role="pmat_subsample",
                        sha256=continuation_artifacts["pmat_subsample"].sha256,
                    ),
                ),
            ),
            DirectoryManifest(
                role="pmat_assembly_result",
                files=(
                    DirectoryManifestFile(
                        relative_path="PMATAllContigs.fna",
                        artifact_role="pmat_all_contigs",
                        sha256=continuation_artifacts["pmat_all_contigs"].sha256,
                    ),
                    DirectoryManifestFile(
                        relative_path="PMATContigGraph.txt",
                        artifact_role="pmat_contig_graph",
                        sha256=continuation_artifacts["pmat_contig_graph"].sha256,
                    ),
                ),
            ),
        )
        for manifest in manifests:
            role = f"{manifest.role}_manifest"
            path = output_dir / f"{manifest.role}.manifest.json"
            path.write_bytes(manifest.canonical_bytes())
            outputs.append(
                BackendOutput(
                    role=role,
                    path=path,
                    format="json",
                    media_type="application/json",
                )
            )
        return NormalizedAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="assembly_fasta",
            primary_graph_role="assembly_graph",
            record_count=fasta.record_count,
            total_bases=fasta.total_bases,
        )


def _extend_both(
    stable: list[str],
    resolved: list[str],
    flag: str,
    stable_value: str,
    resolved_value: str,
) -> None:
    stable.extend((flag, stable_value))
    resolved.extend((flag, resolved_value))


def _append_tool(
    stable: list[str],
    resolved: list[str],
    context: AdapterContext,
    *,
    flag: str,
    name: str,
    artifact_role: str | None,
) -> None:
    if artifact_role is None:
        stable_value = f"role://environment/{name}"
        resolved_value = str(context.environment.require_executable(name))
    else:
        stable_value = f"role://artifact/{artifact_role}"
        resolved_value = str(
            _materialize_artifact(
                context,
                artifact_role,
                context.workspace / "input" / "tools" / name,
                executable=True,
            )
        )
    _extend_both(stable, resolved, flag, stable_value, resolved_value)


def _require_tool(context: AdapterContext, name: str, artifact_role: str | None) -> None:
    if artifact_role is None:
        context.environment.require_executable(name)
        return
    if artifact_role not in context.input_artifacts:
        raise _unsupported(f"PMAT {name} override artifact is missing")


def _uses_source_reads(context: AdapterContext) -> bool:
    # PMAT2 fq2fa uses kseq/gzopen for both plain and gzip input, read-only.
    # Correction and multi-library concatenation retain their staged input.
    return (
        len(context.payload.long_libraries) == 1
        and context.effective_backend_parameters["correction_task"] == "skip"
    )


def _materialize_reads(context: AdapterContext) -> Path:
    libraries = context.payload.long_libraries
    if _uses_source_reads(context):
        artifact = context.input_artifacts[libraries[0].reads_artifact]
        _verify_hash(artifact)
        return Path(artifact.uri).resolve()
    sequence_format = _sequence_format(context.input_artifacts[libraries[0].reads_artifact])
    destination = context.workspace / "input" / f"reads.{sequence_format}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as output:
            for library in libraries:
                artifact = context.input_artifacts[library.reads_artifact]
                _verify_hash(artifact)
                with _open_sequence(Path(artifact.uri)) as source:
                    shutil.copyfileobj(source, output)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _open_sequence(path: Path) -> AbstractContextManager[BinaryIO | gzip.GzipFile]:
    with path.open("rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    if is_gzip:
        return gzip.open(path, "rb")
    return nullcontext(path.open("rb"))


def _materialize_artifact(
    context: AdapterContext,
    role: str,
    destination: Path,
    *,
    executable: bool,
) -> Path:
    artifact = context.input_artifacts[role]
    _verify_hash(artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(artifact.uri), destination)
    destination.chmod(0o700 if executable else 0o600)
    return destination


def _verify_hash(artifact: ArtifactRef) -> None:
    path = Path(artifact.uri)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != artifact.sha256:
        raise _unsupported("PMAT input artifact hash no longer matches its declaration")


def _sequence_format(artifact: ArtifactRef) -> str:
    normalized = {"fa": "fasta", "fasta": "fasta", "fq": "fastq", "fastq": "fastq"}.get(
        artifact.format.lower()
    )
    if normalized is None:
        raise _unsupported("PMAT accepts only declared FASTA or FASTQ reads")
    return normalized


def _unsupported(message: str) -> OrganelleInputError:
    return OrganelleInputError(code="assembly.unsupported_data_profile", message=message)


def _require_output(path: Path, role: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise _output_error(f"PMAT required output {role!r} is missing, empty, or unsafe")


def _output_error(message: str) -> OrganelleExecutionError:
    return OrganelleExecutionError(code="assembly.output_incomplete", message=message)
