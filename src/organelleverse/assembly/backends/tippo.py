"""TIPPo v2.4 plastid HiFi assembly adapter."""

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
from organelleverse.assembly.backends.p2_cli import plan_tippo_argv
from organelleverse.assembly.backends.spec import AssemblyProfile
from organelleverse.assembly.normalization import normalize_fasta
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError

__all__ = ["TippoAdapter"]


def _unsupported(message: str, **details: object) -> OrganelleInputError:
    return OrganelleInputError(
        code="assembly.unsupported_data_profile",
        message=message,
        details=details,
    )


def _output_incomplete(message: str, **details: object) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="assembly.output_incomplete",
        message=message,
        details=details,
    )


class TippoAdapter:
    """Run TIPPo plastid HiFi and preserve every documented FASTA path.

    TIPPo has no output-directory flag. Reads are copied into its backend
    workspace. The lexicographically first path is the primary sequence required
    by the common result contract; remaining paths are explicit alternates and
    are not ranked as biological choices.
    """

    backend_id = "tippo"

    def preflight(self, context: AdapterContext) -> None:
        if context.route.selected_backend != self.backend_id:
            raise _unsupported("TIPPo adapter selected for a non-TIPPo route")
        if context.environment.backend_id != self.backend_id:
            raise _unsupported("TIPPo requires a verified TIPPo environment")
        if context.request.organelle != "plastid":
            raise _unsupported("this TIPPo runtime supports plastid assembly only")
        if context.route.profile is not AssemblyProfile.PACBIO_HIFI:
            raise _unsupported("this TIPPo runtime supports PacBio HiFi CCS reads only")
        if len(context.payload.long_libraries) != 1 or context.payload.short_libraries or context.payload.contig_inputs:
            raise _unsupported("TIPPo requires exactly one PacBio HiFi read library")
        library = context.payload.long_libraries[0]
        if library.technology != "pacbio_hifi" or library.quality_state != "ccs":
            raise _unsupported("TIPPo assembly requires pacbio_hifi + ccs reads")
        if any(value is not None for value in context.payload.auxiliary.model_dump().values()):
            raise _unsupported("TIPPo does not consume assembly auxiliary inputs")
        provided = context.request.backend_parameters
        if provided is not None and provided.backend != self.backend_id:
            raise _unsupported("backend_parameters discriminator does not match TIPPo")

    def build_command(self, context: AdapterContext) -> AssemblyCommand:
        self.preflight(context)
        library = context.payload.long_libraries[0]
        source = Path(context.input_artifacts[library.reads_artifact].uri)
        if not source.is_file():
            raise _unsupported("TIPPo read artifact is not a readable local file", uri=str(source))
        # The upstream script has no output-directory flag. Staging reads in
        # the subprocess working directory keeps generated files isolated.
        context.workspace.mkdir(parents=True, exist_ok=True)
        reads = context.workspace / source.name
        if source.resolve() != reads.resolve():
            shutil.copyfile(source, reads)
        executable = context.environment.require_executable("TIPPo.v2.4.pl")
        resolved = plan_tippo_argv(
            executable=str(executable),
            reads=reads,
            organelle="plastid",
            platform="hifi",
            threads=context.request.threads,
        )
        stable = (
            "TIPPo.v2.4.pl",
            "-f",
            f"role://workspace/{reads.name}",
            "-g",
            "chloroplast",
            "-t",
            str(context.request.threads),
            "-p",
            "hifi",
        )
        return AssemblyCommand(stable_argv=stable, resolved_argv=resolved)

    def collect_outputs(self, context: AdapterContext) -> RawAssemblyOutputs:
        library = context.payload.long_libraries[0]
        source = Path(context.input_artifacts[library.reads_artifact].uri)
        root = context.workspace
        suffix = ".organelle.chloroplast.fasta"
        prefix = f"{source.name}.chloroplast.fasta.filter."
        candidates = tuple(sorted(
            path for path in root.glob(f"{prefix}*{suffix}")
            if path.is_file() and path.stat().st_size > 0
        ))
        if not candidates:
            raise _output_incomplete(
                "TIPPo produced no documented chloroplast path FASTA",
                output_root=str(root),
                expected_prefix=prefix,
                expected_suffix=suffix,
            )
        outputs = tuple(
            BackendOutput(
                role=f"path_fasta_{index}",
                path=path,
                format="fasta",
                media_type="text/x-fasta",
            )
            for index, path in enumerate(candidates)
        )
        return RawAssemblyOutputs(outputs=outputs, primary_sequence_role="path_fasta_0")

    def normalize(
        self,
        context: AdapterContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs:
        del context
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[BackendOutput] = []
        alternates: list[str] = []
        primary = raw.require(raw.primary_sequence_role)
        stats = normalize_fasta(primary.path, output_dir / "assembly.fasta")
        outputs.append(BackendOutput(
            role="assembly_fasta",
            path=output_dir / "assembly.fasta",
            format="fasta",
            media_type="text/x-fasta",
        ))
        for item in raw.outputs:
            if item.role == raw.primary_sequence_role:
                continue
            index = item.role.removeprefix("path_fasta_")
            role = f"alternate_fasta_{index}"
            destination = output_dir / f"alternate_{index}.fasta"
            normalize_fasta(item.path, destination)
            outputs.append(BackendOutput(
                role=role,
                path=destination,
                format="fasta",
                media_type="text/x-fasta",
            ))
            alternates.append(role)
        return NormalizedAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="assembly_fasta",
            alternate_sequence_roles=tuple(alternates),
            record_count=stats.record_count,
            total_bases=stats.total_bases,
        )
