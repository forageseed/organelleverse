from __future__ import annotations

import json
from pathlib import Path

import pytest
from Bio import SeqIO
from pydantic import ValidationError

from organelleverse.assembly.backends.base import AdapterContext, PreparedBackendResources
from organelleverse.assembly.backends.novoplasty import NovoplastyAdapter
from organelleverse.assembly.contracts import (
    AssemblyAuxiliary,
    AssemblyRequest,
    NovoplastyParameters,
    ShortReadLibrary,
    effective_backend_parameters,
)
from organelleverse.assembly.data_contract import validate_novoplasty_assembly_data
from organelleverse.assembly.environments import PreparedEnvironment, PreparedExecutable
from organelleverse.assembly.routing import resolve_backend
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError
from organelleverse.core.frozen import FrozenMap
from organelleverse.io_reads import read_reads


def context(
    tmp_path: Path, *, auxiliary=None, parameters=None, organelle="plastid", layout="paired_end"
):
    reads = []
    for mate in (1, 2):
        path = tmp_path / f"reads {mate}.fq"
        path.write_text("@read\n" + "ACGT" * 38 + "\n+\n" + "I" * 152 + "\n")
        reads.append(path)
    data = read_reads(
        short_libraries=(
            ShortReadLibrary(
                technology="illumina",
                layout=layout,
                read1=reads[0],
                read2=reads[1] if layout == "paired_end" else None,
                read_length=152,
                insert_size=300,
            ),
        ),
        auxiliary=auxiliary or AssemblyAuxiliary(),
    )
    payload = validate_novoplasty_assembly_data(data)
    request = AssemblyRequest(
        data=data, organelle=organelle, method="novoplasty", backend_parameters=parameters
    )
    effective = effective_backend_parameters(
        "novoplasty", parameters, payload=payload, organelle=organelle, taxon_group="plant"
    )
    prefix = tmp_path / "env"
    executable = prefix / "bin" / "NOVOPlasty4.3.5.pl"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    env = PreparedEnvironment(
        backend_id="novoplasty",
        carrier="conda",
        platform="linux-64",
        prefix=prefix,
        digest="sha256:" + "a" * 64,
        version="4.3.5",
        executables=(PreparedExecutable(name=executable.name, path=executable),),
    )
    return AdapterContext(
        request=request,
        payload=payload,
        route=resolve_backend(data, organelle=organelle, method="novoplasty"),
        environment=env,
        resources=PreparedBackendResources(),
        workspace=tmp_path / "work",
        input_artifacts=FrozenMap.from_items(dict(data.artifacts)),
        effective_backend_parameters=FrozenMap.from_json(effective),
    )


def test_default_seed_config_and_candidate_provenance(tmp_path):
    ctx = context(tmp_path)
    adapter = NovoplastyAdapter()
    command = adapter.build_command(ctx)
    from organelleverse.assembly.manifests import AssemblyRunManifest

    assert AssemblyRunManifest.validate_stable_argv(command.stable_argv) == command.stable_argv
    config = Path(command.resolved_argv[-1])
    values = dict(line.split(" = ", 1) for line in config.read_text().splitlines())
    values = {key.strip(): value for key, value in values.items()}
    assert values["Type"] == "chloro"
    assert values["Genome Range"] == "120000-200000"
    assert values["K-mer"] == "33"
    assert values["Read Length"] == "152"
    assert values["Insert size"] == "300"
    assert values["Insert size auto"] == "yes"
    assert values["Output path"].endswith("/")
    seed = SeqIO.read(values["Seed Input"], "fasta")
    assert seed.id.endswith("_rbcL") and len(seed) > 1000
    root = config.parent
    (root / "Circularized_assembly_2_organelleverse.fasta").write_text(">circle2\nACGTACGT\n")
    (root / "Circularized_assembly_1_organelleverse.fasta").write_text(">circle1\nTTTTCCCC\n")
    (root / "Option_1_organelleverse.fasta").write_text(">contig1\nAAAA\n>contig2\nGGGG\n")
    (root / "Contigs_1_organelleverse.fasta").write_text(">fragment\nACGT\n")
    raw = adapter.collect_outputs(ctx)
    normalized = adapter.normalize(ctx, raw, tmp_path / "normalized")
    assert normalized.record_count == 1 and normalized.total_bases == 8
    assert len(normalized.alternate_sequence_roles) == 3
    report = json.loads(normalized.require("novoplasty_report").path.read_text())
    assert [x["topology"] for x in report["candidates"]] == [
        "circular",
        "circular",
        "unknown",
        "linear",
    ]
    assert report["candidates"][2]["records"] == 2
    assert report["seed"]["gene"] == "rbcL"
    assert (
        normalized.require("novoplasty_seed").path.read_text() == (root / "seed.fasta").read_text()
    )


def test_user_seed_and_plant_mito_configuration(tmp_path):
    seed = tmp_path / "seed.fa"
    seed.write_text(">user_seed\nACGTACGT\n")
    cp = tmp_path / "cp.fa"
    cp.write_text(">cp\nTTTTCCCC\n")
    ctx = context(
        tmp_path,
        organelle="mitochondrion",
        auxiliary=AssemblyAuxiliary(seed_fasta=seed, chloroplast_fasta=cp),
        parameters=NovoplastyParameters(genome_range=(300000, 600000), kmer_size=29),
    )
    config = Path(NovoplastyAdapter().build_command(ctx).resolved_argv[-1])
    from jsonschema import Draft202012Validator

    from organelleverse.assembly.data_contract import released_assembly_data_json_schema

    Draft202012Validator(released_assembly_data_json_schema()).validate(
        ctx.request.data.model_dump(mode="json")
    )
    assert "mito_plant" in config.read_text()
    assert "300000-600000" in config.read_text()
    info = json.loads((config.parent / "seed.json").read_text())
    assert info["source"] == "user" and info["records"][0]["id"] == "user_seed"


@pytest.mark.parametrize("name", [None, "Circularized_assembly_1_organelleverse.fasta"])
def test_missing_or_empty_outputs_fail(tmp_path, name):
    ctx = context(tmp_path)
    root = Path(NovoplastyAdapter().build_command(ctx).resolved_argv[-1]).parent
    if name:
        (root / name).touch()
    with pytest.raises(OrganelleExecutionError):
        NovoplastyAdapter().collect_outputs(ctx)


def test_linear_only_output_is_preserved(tmp_path):
    ctx = context(tmp_path)
    adapter = NovoplastyAdapter()
    root = Path(adapter.build_command(ctx).resolved_argv[-1]).parent
    (root / "Contigs_1_organelleverse.fasta").write_text(">one\nAAAA\n>two\nCCCC\n")
    result = adapter.normalize(ctx, adapter.collect_outputs(ctx), tmp_path / "output")
    assert result.record_count == 2
    assert (
        json.loads(result.require("novoplasty_report").path.read_text())["candidates"][0][
            "topology"
        ]
        == "linear"
    )


@pytest.mark.parametrize(
    "parameters",
    [NovoplastyParameters(genome_range=(200000, 100000)), NovoplastyParameters(kmer_size=152)],
)
def test_invalid_scientific_parameters_fail(tmp_path, parameters):
    with pytest.raises(OrganelleInputError):
        context(tmp_path, parameters=parameters)


def test_single_end_is_not_admitted(tmp_path):
    with pytest.raises(OrganelleInputError):
        context(tmp_path, layout="single_end")


def test_parameter_contract_is_closed():
    with pytest.raises(ValidationError):
        NovoplastyParameters(raw_argv="--anything")


def test_mitochondrion_requires_explicit_seed_and_range(tmp_path):
    with pytest.raises(OrganelleInputError, match="genome_range"):
        context(tmp_path, organelle="mitochondrion")
    with pytest.raises(OrganelleInputError, match="seed"):
        context(
            tmp_path,
            organelle="mitochondrion",
            parameters=NovoplastyParameters(genome_range=(100000, 900000)),
        )


@pytest.mark.parametrize("exit_code,accepted", [(2, True), (0, False), (255, False)])
def test_existing_provider_checks_declared_banner_exit(tmp_path, exit_code, accepted):
    from organelleverse.assembly.backends.runtime import RUNTIMES
    from organelleverse.assembly.environment_capabilities import runtime_capabilities
    from organelleverse.assembly.environment_contracts import EnvironmentHint
    from organelleverse.assembly.environment_resolver import EnvironmentResolver
    from organelleverse.assembly.environment_versions import resolve_backend_version
    from organelleverse.core.errors import OrganelleDependencyError

    ctx = context(tmp_path)
    prefix = ctx.environment.prefix
    for folder in (prefix, prefix / "bin"):
        folder.chmod(0o755)
    perl = prefix / "bin" / "perl"
    perl.write_text("#!/bin/sh\n")
    perl.chmod(0o755)

    class Sandbox:
        read_only = True
        network_disabled = True

        def __call__(self, executable, argv, **kwargs):
            assert argv == ("-c", "")
            return exit_code, "NOVOPlasty\nVersion 4.3.5\n", "empty configuration filename"

    resolver = EnvironmentResolver(
        _probe_sandbox=Sandbox(),
        _find_in_path=lambda name: [],
        _find_conda=lambda: [],
        _registry_lookup=lambda name: None,
    )
    request = ctx.request.model_copy(
        update={
            "environment_source": "existing",
            "environment_hint": EnvironmentHint(prefix=prefix),
        }
    )
    kwargs = dict(
        request=request,
        capability_contract=runtime_capabilities(RUNTIMES["novoplasty"], {}),
        resolved_version=resolve_backend_version("novoplasty", "tested"),
        effective_parameters={},
    )
    if accepted:
        resolution = resolver.resolve(**kwargs)
        assert resolution.selected_provider.components[0].version == "4.3.5"
    else:
        with pytest.raises(OrganelleDependencyError):
            resolver.resolve(**kwargs)
