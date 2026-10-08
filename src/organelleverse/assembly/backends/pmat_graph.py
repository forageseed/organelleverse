"""Thin PMAT2 ``graphBuild`` continuation adapter."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from organelleverse.assembly.backends.base import (
    AssemblyCommand,
    BackendOutput,
    NormalizedAssemblyOutputs,
    RawAssemblyOutputs,
)
from organelleverse.assembly.environments import PreparedEnvironment
from organelleverse.assembly.normalization import (
    normalize_fasta,
    normalize_gfa,
    validate_fasta_against_gfa,
)
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError
from organelleverse.operations.spec import StrictSpecModel

__all__ = ["PmatGraphAdapter", "PmatGraphContext"]


class PmatGraphContext(StrictSpecModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    workspace: Path
    environment: PreparedEnvironment
    subsample_dir: Path
    assembly_result_dir: Path
    organelle: Literal["mitochondrion", "plastid"]
    taxon_group: Literal["plant", "animal", "fungi"] = "plant"
    depth: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    seeds: tuple[int, ...] | None = None
    threads: int = Field(ge=1, le=256)


class PmatGraphAdapter:
    """Build the complete, path-safe PMAT2 graphBuild argv."""

    backend_id = "pmat"

    def preflight(self, context: PmatGraphContext) -> None:
        if context.environment.backend_id != self.backend_id:
            raise _unsupported("PMAT graphBuild requires a verified PMAT environment")
        if context.organelle == "plastid" and context.taxon_group != "plant":
            raise _unsupported("PMAT2 plastid graphBuild requires plant taxon mode")
        if context.seeds is not None and (
            not context.seeds
            or any(seed < 1 for seed in context.seeds)
            or len(set(context.seeds)) != len(context.seeds)
        ):
            raise _unsupported("PMAT2 graphBuild seeds must be unique positive contig IDs")

    def build_command(self, context: PmatGraphContext) -> AssemblyCommand:
        self.preflight(context)
        output_path = context.workspace / "backend" / "pmat_graph"
        output_path.mkdir(parents=True, exist_ok=True)
        target = "mt" if context.organelle == "mitochondrion" else "pt"
        taxon = {"plant": 0, "animal": 1, "fungi": 2}[context.taxon_group]
        stable = [
            "PMAT",
            "graphBuild",
            "-i",
            "role://workspace/subsample",
            "-a",
            "role://workspace/assembly_result",
            "-o",
            "role://workspace/output",
            "-G",
            target,
            "-x",
            str(taxon),
        ]
        resolved = [
            str(context.environment.require_executable("pmat")),
            "graphBuild",
            "-i",
            str(context.subsample_dir),
            "-a",
            str(context.assembly_result_dir),
            "-o",
            str(output_path),
            "-G",
            target,
            "-x",
            str(taxon),
        ]
        if context.depth is not None:
            stable.extend(("-d", str(context.depth)))
            resolved.extend(("-d", str(context.depth)))
        if context.seeds is not None:
            values = tuple(str(seed) for seed in context.seeds)
            stable.extend(("-s", *values))
            resolved.extend(("-s", *values))
        stable.extend(("-T", str(context.threads)))
        resolved.extend(("-T", str(context.threads)))
        return AssemblyCommand(stable_argv=tuple(stable), resolved_argv=tuple(resolved))

    def collect_outputs(self, context: PmatGraphContext) -> RawAssemblyOutputs:
        root = context.workspace / "backend" / "pmat_graph"
        target = "mt" if context.organelle == "mitochondrion" else "pt"
        fasta_name = f"PMAT_{target}.fa"
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
        ]
        if target == "mt":
            declared.append(("assembly_assessment", root / "PMAT_orgAss.txt", "txt", "text/plain"))
        outputs: list[BackendOutput] = []
        for role, path, format_name, media_type in declared:
            _require_output(path, role)
            outputs.append(
                BackendOutput(role=role, path=path, format=format_name, media_type=media_type)
            )
        if target == "mt":
            assessment = next(
                item.path for item in outputs if item.role == "assembly_assessment"
            ).read_text(encoding="utf-8", errors="replace")
            if "mitochond" not in assessment.lower():
                raise _output_error(
                    "PMAT graphBuild assessment does not match the requested organelle"
                )
        return RawAssemblyOutputs(
            outputs=tuple(outputs),
            primary_sequence_role="assembly_fasta",
            primary_graph_role="assembly_graph",
        )

    def normalize(
        self,
        context: PmatGraphContext,
        raw: RawAssemblyOutputs,
        output_dir: Path,
    ) -> NormalizedAssemblyOutputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        destinations = {
            "assembly_fasta": output_dir / "assembly.fasta",
            "assembly_graph": output_dir / "assembly.gfa",
            "raw_assembly_graph": output_dir / "assembly.raw.gfa",
            "assembly_assessment": output_dir / "assessment.txt",
        }
        fasta = normalize_fasta(raw.require("assembly_fasta").path, destinations["assembly_fasta"])
        graph = normalize_gfa(raw.require("assembly_graph").path, destinations["assembly_graph"])
        validate_fasta_against_gfa(fasta, graph)
        normalize_gfa(raw.require("raw_assembly_graph").path, destinations["raw_assembly_graph"])
        if any(item.role == "assembly_assessment" for item in raw.outputs):
            shutil.copyfile(
                raw.require("assembly_assessment").path,
                destinations["assembly_assessment"],
            )
        outputs = tuple(
            BackendOutput(
                role=item.role,
                path=destinations[item.role],
                format=item.format,
                media_type=item.media_type,
            )
            for item in raw.outputs
        )
        return NormalizedAssemblyOutputs(
            outputs=outputs,
            primary_sequence_role="assembly_fasta",
            primary_graph_role="assembly_graph",
            record_count=fasta.record_count,
            total_bases=fasta.total_bases,
        )


def _require_output(path: Path, role: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise _output_error(f"PMAT graphBuild output {role!r} is missing, empty, or unsafe")


def _unsupported(message: str) -> OrganelleInputError:
    return OrganelleInputError(code="assembly.unsupported_data_profile", message=message)


def _output_error(message: str) -> OrganelleExecutionError:
    return OrganelleExecutionError(code="assembly.output_incomplete", message=message)
