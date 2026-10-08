from pathlib import Path

import pytest

from organelleverse.assembly.backends.p2_cli import (
    AssemblyBackendInputError,
    plan_ptgaul_argv,
    plan_tippo_argv,
    require_unicycler_organism_scope,
)


def test_tippo_uses_documented_v24_flags_and_organelle_mode() -> None:
    argv = plan_tippo_argv(
        executable="/tools/TIPPo.v2.4.pl",
        reads=Path("/data/reads.fastq.gz"),
        organelle="plastid",
        platform="hifi",
        threads=12,
    )

    assert argv == (
        "/tools/TIPPo.v2.4.pl",
        "-f",
        "/data/reads.fastq.gz",
        "-g",
        "chloroplast",
        "-t",
        "12",
        "-p",
        "hifi",
    )


def test_tippo_mitochondrion_uses_organelle_mode() -> None:
    argv = plan_tippo_argv(
        executable="TIPPo.v2.4.pl",
        reads=Path("/data/reads.fq"),
        organelle="mitochondrion",
        platform="ont",
        threads=1,
    )

    assert argv[3:5] == ("-g", "organelle")
    assert argv[-2:] == ("-p", "ont")


def test_tippo_rejects_invalid_threads_and_relative_reads() -> None:
    with pytest.raises(AssemblyBackendInputError, match="threads"):
        plan_tippo_argv(
            executable="TIPPo.v2.4.pl",
            reads=Path("/data/reads.fq"),
            organelle="plastid",
            platform="hifi",
            threads=0,
        )
    with pytest.raises(AssemblyBackendInputError, match="absolute"):
        plan_tippo_argv(
            executable="TIPPo.v2.4.pl",
            reads=Path("reads.fq"),
            organelle="plastid",
            platform="hifi",
            threads=1,
        )


def test_ptgaul_uses_documented_required_and_optional_flags() -> None:
    argv = plan_ptgaul_argv(
        executable="/tools/ptGAUL.sh",
        reference=Path("/data/reference.fa"),
        long_reads=Path("/data/reads.fastq.gz"),
        output_dir=Path("/runs/sample"),
        threads=8,
        genome_size=154_000,
        coverage=60,
        minimum_read_length=2_500,
    )

    assert argv == (
        "/tools/ptGAUL.sh",
        "-r",
        "/data/reference.fa",
        "-l",
        "/data/reads.fastq.gz",
        "-t",
        "8",
        "-g",
        "154000",
        "-c",
        "60",
        "-f",
        "2500",
        "-o",
        "/runs/sample",
    )


def test_ptgaul_rejects_nonpositive_parameters() -> None:
    with pytest.raises(AssemblyBackendInputError, match="positive"):
        plan_ptgaul_argv(
            executable="ptGAUL.sh",
            reference=Path("/data/reference.fa"),
            long_reads=Path("/data/reads.fq"),
            output_dir=Path("/runs/sample"),
            threads=8,
            coverage=0,
        )


def test_unicycler_scope_is_explicitly_limited_to_bacterial_isolates() -> None:
    require_unicycler_organism_scope(organism_scope="bacterial_isolate")

    with pytest.raises(AssemblyBackendInputError, match="bacterial isolates"):
        require_unicycler_organism_scope(organism_scope="organelle_only")

