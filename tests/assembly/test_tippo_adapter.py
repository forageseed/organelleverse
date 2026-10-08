from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.assembly.backends import AssemblyProfile, PreparedBackendResources
from organelleverse.assembly.backends.base import AdapterContext
from organelleverse.assembly.backends.tippo import TippoAdapter
from organelleverse.assembly.contracts import AssemblyInputPayload, AssemblyRequest
from organelleverse.assembly.environments import PreparedEnvironment, PreparedExecutable
from organelleverse.assembly.routing import AssemblyRoute
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError
from organelleverse.core.frozen import FrozenMap


def _context(tmp_path: Path, *, organelle: str = "plastid") -> AdapterContext:
    source = tmp_path / "reads.fastq.gz"
    source.write_bytes(b"raw reads")
    artifact = ArtifactRef.from_path(
        source, kind="long_read", format="fastq.gz", media_type="application/gzip"
    )
    data = OrganelleData.model_validate({
        "modality": "sequencing_reads",
        "artifacts": {"long_0_reads": artifact},
        "payload": {
            "contract_version": "organelleverse.assembly-input.v1",
            "long_libraries": [{
                "technology": "pacbio_hifi",
                "quality_state": "ccs",
                "reads_artifact": "long_0_reads",
            }],
        },
    })
    payload = AssemblyInputPayload.model_validate(dict(data.payload))
    request = AssemblyRequest(data=data, organelle=organelle, method="tippo", threads=3)
    prefix = tmp_path / "env"
    executable = prefix / "bin" / "TIPPo.v2.4.pl"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    environment = PreparedEnvironment(
        backend_id="tippo",
        carrier="conda",
        platform="linux-64",
        digest="sha256:" + "b" * 64,
        prefix=prefix,
        executables=(PreparedExecutable(name="TIPPo.v2.4.pl", path=executable),),
        version="TIPPo 2.4",
    )
    return AdapterContext(
        request=request,
        payload=payload,
        route=AssemblyRoute(
            requested_method="tippo",
            selected_backend="tippo",
            profile=AssemblyProfile.PACBIO_HIFI,
            rule_id="explicit.backend",
            compatible_candidates=("tippo",),
        ),
        environment=environment,
        profile=None,
        resources=PreparedBackendResources(),
        workspace=tmp_path / "workspace",
        input_artifacts=FrozenMap.from_items({"long_0_reads": artifact}),
        effective_backend_parameters=FrozenMap.from_json({"backend": "tippo"}),
    )


def _fasta(path: Path, name: str, sequence: str) -> Path:
    path.write_text(f">{name}\n{sequence}\n")
    return path


def test_tippo_build_command_stages_input_and_uses_verified_cli(tmp_path: Path) -> None:
    context = _context(tmp_path)
    command = TippoAdapter().build_command(context)
    assert command.stable_argv == (
        "TIPPo.v2.4.pl", "-f", "role://workspace/reads.fastq.gz",
        "-g", "chloroplast", "-t", "3", "-p", "hifi",
    )
    assert command.resolved_argv[0].endswith("TIPPo.v2.4.pl")
    assert (context.workspace / "reads.fastq.gz").read_bytes() == b"raw reads"


@pytest.mark.parametrize("organelle", ["mitochondrion"])
def test_tippo_rejects_mitochondrial_mode_until_linear_output_is_verified(
    tmp_path: Path, organelle: str
) -> None:
    context = _context(tmp_path, organelle=organelle)
    with pytest.raises(OrganelleInputError, match="plastid assembly only"):
        TippoAdapter().build_command(context)


def test_tippo_collects_all_documented_plastid_path_fastas(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.workspace.mkdir(parents=True)
    base = "reads.fastq.gz.chloroplast.fasta.filter."
    first = _fasta(context.workspace / (base + "800.round1.edge_a.edge_b.edge_c.organelle.chloroplast.fasta"), "primary", "ACGT")
    second = _fasta(context.workspace / (base + "800.round1.edge_x.edge_y.edge_z.organelle.chloroplast.fasta"), "alternate", "TTAA")
    raw = TippoAdapter().collect_outputs(context)
    assert raw.primary_sequence_role == "path_fasta_0"
    assert tuple(output.path for output in raw.outputs) == (first, second)
    normalized = TippoAdapter().normalize(context, raw, tmp_path / "normalized")
    assert normalized.primary_sequence_role == "assembly_fasta"
    assert normalized.alternate_sequence_roles == ("alternate_fasta_1",)
    assert normalized.total_bases == 4
    assert (tmp_path / "normalized" / "assembly.fasta").read_text().endswith("ACGT\n")
    assert (tmp_path / "normalized" / "alternate_1.fasta").read_text().endswith("TTAA\n")


def test_tippo_reports_missing_documented_path_fastas(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.workspace.mkdir(parents=True)
    with pytest.raises(OrganelleExecutionError) as raised:
        TippoAdapter().collect_outputs(context)
    assert raised.value.code == "assembly.output_incomplete"
