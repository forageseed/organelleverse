from __future__ import annotations

from pathlib import Path
from typing import Literal


class AssemblyBackendInputError(ValueError):
    """Raised when a verified upstream command contract cannot accept the inputs."""


TippoPlatform = Literal["hifi", "clr", "ont", "onthq"]


def plan_tippo_argv(
    *,
    executable: str,
    reads: Path,
    organelle: Literal["mitochondrion", "plastid"],
    platform: TippoPlatform,
    threads: int,
) -> tuple[str, ...]:
    """Build the documented TIPPo v2.4 command without invoking or installing it.

    TIPPo has no output-directory option in its published CLI, so its caller
    must run it with an isolated working directory and collect its documented
    input-prefixed output paths.
    """
    if not executable or "\x00" in executable:
        raise AssemblyBackendInputError("TIPPo executable must be a non-empty path")
    if threads < 1:
        raise AssemblyBackendInputError("threads must be at least 1")
    if not reads.is_absolute():
        raise AssemblyBackendInputError("TIPPo reads path must be absolute")
    group = "chloroplast" if organelle == "plastid" else "organelle"
    return (
        executable,
        "-f",
        str(reads),
        "-g",
        group,
        "-t",
        str(threads),
        "-p",
        platform,
    )


def plan_ptgaul_argv(
    *,
    executable: str,
    reference: Path,
    long_reads: Path,
    output_dir: Path,
    threads: int,
    genome_size: int = 160_000,
    coverage: int = 50,
    minimum_read_length: int = 3_000,
) -> tuple[str, ...]:
    """Build the documented ptGAUL 1.0.5 invocation for plastid long reads."""
    if not executable or "\x00" in executable:
        raise AssemblyBackendInputError("ptGAUL executable must be a non-empty path")
    for name, path in (
        ("reference", reference),
        ("long reads", long_reads),
        ("output directory", output_dir),
    ):
        if not path.is_absolute():
            raise AssemblyBackendInputError(f"ptGAUL {name} path must be absolute")
    if threads < 1 or genome_size < 1 or coverage < 1 or minimum_read_length < 1:
        raise AssemblyBackendInputError(
            "threads, genome_size, coverage, and minimum_read_length must be positive"
        )
    return (
        executable,
        "-r",
        str(reference),
        "-l",
        str(long_reads),
        "-t",
        str(threads),
        "-g",
        str(genome_size),
        "-c",
        str(coverage),
        "-f",
        str(minimum_read_length),
        "-o",
        str(output_dir),
    )


def require_unicycler_organism_scope(*, organism_scope: str) -> None:
    """Reject Unicycler for organelle-only assemblies.

    Upstream documents Unicycler for bacterial isolates and says it is not
    intended for eukaryotic genomes. OrganelleVerse must not imply validation
    for plant organelle assembly until a dedicated benchmark justifies it.
    """
    if organism_scope != "bacterial_isolate":
        raise AssemblyBackendInputError(
            "Unicycler is documented for bacterial isolates; "
            "direct organelle-only assembly is outside its supported scope"
        )

