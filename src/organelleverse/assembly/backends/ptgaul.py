from __future__ import annotations

import shutil
from pathlib import Path

from organelleverse.assembly.backends.base import (
    AdapterContext,
    AssemblyCommand,
    BackendOutput,
    NormalizedAssemblyOutputs,
    RawAssemblyOutputs,
)
from organelleverse.assembly.backends.p2_cli import plan_ptgaul_argv
from organelleverse.assembly.backends.spec import AssemblyProfile
from organelleverse.assembly.normalization import normalize_fasta, normalize_gfa
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError

__all__ = ["PtgaulAdapter"]


def _input_error(message: str, **details: object) -> OrganelleInputError:
    return OrganelleInputError(
        code="assembly.unsupported_data_profile", message=message, details=details
    )


def _output_error(message: str, **details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.output_incomplete", message=message, details=details
    )


def _segment_count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.startswith("S\t"))
    except OSError as error:
        raise _output_error(
            "ptGAUL assembly graph is unavailable",
            graph_path=str(path),
            cause=str(error),
        ) from error


class PtgaulAdapter:
    """Execute the verified ptGAUL plastid ONT workflow.

    The upstream script emits a final FASTA for a one-segment graph and two
    FASTA candidates for a three-segment graph. Path 1 occupies the common
    primary slot for API compatibility; path 2 is retained as an alternate.
    Neither candidate is asserted to be biologically canonical.
    """

    backend_id = "ptgaul"

    def preflight(self, context: AdapterContext) -> None:
        if context.route.selected_backend != self.backend_id:
            raise _input_error("ptGAUL adapter selected for a non-ptGAUL route")
        if context.environment.backend_id != self.backend_id:
            raise _input_error("ptGAUL requires an existing verified environment")
        if context.request.organelle != "plastid":
            raise _input_error("ptGAUL is a plastid-only assembly pipeline")
        if context.route.profile is not AssemblyProfile.ONT_RAW:
            raise _input_error(
                "the executable ptGAUL workflow supports raw ONT reads only; "
                "its published Flye invocation uses --nano-raw"
            )
        if (
            len(context.payload.long_libraries) != 1
            or context.payload.short_libraries
            or context.payload.contig_inputs
        ):
            raise _input_error("ptGAUL requires exactly one raw ONT library")
        library = context.payload.long_libraries[0]
        if library.technology != "ont" or library.quality_state != "raw":
            raise _input_error("ptGAUL requires raw ONT reads")
        reference_role = context.payload.auxiliary.reference_fasta_artifact
        if reference_role is None or reference_role not in context.input_artifacts:
            raise _input_error(
                "ptGAUL requires a related plastome reference FASTA",
                reference_role=reference_role,
            )
        if any(
            value is not None
            for key, value in context.payload.auxiliary.model_dump().items()
            if key != "reference_fasta_artifact"
        ):
            raise _input_error("ptGAUL does not consume other auxiliary inputs")
        provided = context.request.backend_parameters
        if provided is not None and provided.backend != self.backend_id:
            raise _input_error("backend_parameters discriminator does not match ptGAUL")

    def build_command(self, context: AdapterContext) -> AssemblyCommand:
        self.preflight(context)
        params = context.effective_backend_parameters
        library = context.payload.long_libraries[0]
        source = Path(context.input_artifacts[library.reads_artifact].uri)
        reference_role = context.payload.auxiliary.reference_fasta_artifact
        assert reference_role is not None
        reference_source = Path(context.input_artifacts[reference_role].uri)
        supported_suffixes = (".fasta", ".fa", ".fastq", ".fq", ".fq.gz", ".fastq.gz")
        if not source.is_file() or not source.name.endswith(supported_suffixes):
            raise _input_error(
                "ptGAUL requires a readable FASTA/FA/FQ/FASTQ or compressed FASTQ read file",
                reads_uri=str(source),
            )
        if not reference_source.is_file() or reference_source.suffix.lower() not in {".fa", ".fasta"}:
            raise _input_error(
                "ptGAUL reference FASTA is not a readable local file",
                reference_uri=str(reference_source),
            )
        input_dir = context.workspace / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        reads = input_dir / source.name
        reference = input_dir / "reference.fasta"
        if source.resolve() != reads.resolve():
            shutil.copyfile(source, reads)
        if reference_source.resolve() != reference.resolve():
            shutil.copyfile(reference_source, reference)
        output_dir = context.workspace / "backend" / self.backend_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        executable = context.environment.require_executable("ptGAUL.sh")
        resolved = plan_ptgaul_argv(
            executable=str(executable),
            reference=reference,
            long_reads=reads,
            output_dir=output_dir,
            threads=context.request.threads,
            genome_size=int(params["genome_size"]),
            coverage=int(params["coverage"]),
            minimum_read_length=int(params["minimum_read_length"]),
        )
        stable = (
            "ptGAUL.sh",
            "-r",
            f"role://artifact/{reference_role}",
            "-l",
            f"role://artifact/{library.reads_artifact}",
            "-t",
            str(context.request.threads),
            "-g",
            str(params["genome_size"]),
            "-c",
            str(params["coverage"]),
            "-f",
            str(params["minimum_read_length"]),
            "-o",
            "role://workspace/backend/ptgaul/output",
        )
        return AssemblyCommand(stable_argv=stable, resolved_argv=resolved)

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs:
        params = context.effective_backend_parameters
        root = context.workspace / "backend" / self.backend_id / "output"
        result = root / f"result_{params['minimum_read_length']}"
        graph = result / "flye_cpONT" / "assembly_graph.gfa"
        edges = _segment_count(graph)
        if edges == 1:
            sequence = result / "ptGAUL_final_assembly" / "final_assembly.fasta"
            if not sequence.is_file() or sequence.stat().st_size == 0:
                raise _output_error(
                    "ptGAUL one-edge result FASTA is missing",
                    expected_path=str(sequence),
                    edge_count=edges,
                )
            outputs = (
                BackendOutput(role="assembly_fasta", path=sequence, format="fasta", media_type="text/x-fasta"),
                BackendOutput(role="assembly_graph", path=graph, format="gfa", media_type="text/plain"),
            )
            return RawAssemblyOutputs(
                outputs=outputs,
                primary_sequence_role="assembly_fasta",
                primary_graph_role="assembly_graph",
            )
        if edges == 3:
            folder = result / "ptGAUL_final_assembly"
            path1 = folder / "path1.fasta"
            path2 = folder / "path2.fasta"
            missing = [str(path) for path in (path1, path2) if not path.is_file() or path.stat().st_size == 0]
            if missing:
                raise _output_error(
                    "ptGAUL three-edge result must contain both documented path FASTAs",
                    missing_paths=missing,
                    edge_count=edges,
                )
            outputs = (
                BackendOutput(role="path1_candidate", path=path1, format="fasta", media_type="text/x-fasta"),
                BackendOutput(role="path2_candidate", path=path2, format="fasta", media_type="text/x-fasta"),
                BackendOutput(role="assembly_graph", path=graph, format="gfa", media_type="text/plain"),
            )
            return RawAssemblyOutputs(
                outputs=outputs,
                primary_sequence_role="path1_candidate",
                primary_graph_role="assembly_graph",
            )
        raise _output_error(
            "ptGAUL produced an unsupported graph edge count; manual graph inspection is required",
            edge_count=edges,
            graph_path=str(graph),
            supported_edge_counts=[1, 3],
        )

    def normalize(
        self,
        context: AdapterContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs:
        del context
        output_dir.mkdir(parents=True, exist_ok=True)
        fasta_stats = normalize_fasta(
            raw.require(raw.primary_sequence_role).path,
            output_dir / "assembly.fasta",
        )
        normalize_gfa(raw.require("assembly_graph").path, output_dir / "assembly.gfa")
        outputs = [
            BackendOutput(role="assembly_fasta", path=output_dir / "assembly.fasta", format="fasta", media_type="text/x-fasta"),
            BackendOutput(role="assembly_graph", path=output_dir / "assembly.gfa", format="gfa", media_type="text/plain"),
        ]
        alternate_roles: list[str] = []
        if any(item.role == "path2_candidate" for item in raw.outputs):
            role = "alternate_fasta_2"
            destination = output_dir / "alternate_2.fasta"
            normalize_fasta(raw.require("path2_candidate").path, destination)
            outputs.append(BackendOutput(role=role, path=destination, format="fasta", media_type="text/x-fasta"))
            alternate_roles.append(role)
        return NormalizedAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="assembly_fasta",
            primary_graph_role="assembly_graph",
            alternate_sequence_roles=tuple(alternate_roles),
            record_count=fasta_stats.record_count,
            total_bases=fasta_stats.total_bases,
        )
