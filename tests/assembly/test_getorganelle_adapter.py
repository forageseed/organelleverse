"""Hermetic tests for the GetOrganelle adapter and database provider."""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pytest

from organelleverse.assembly.backends.base import (
    AdapterContext,
    PreparedBackendResources,
)
from organelleverse.assembly.backends.getorganelle import (
    GetOrganelleAdapter,
    GetOrganelleResourceProvider,
)
from organelleverse.assembly.backends.spec import AssemblyProfile
from organelleverse.assembly.contracts import (
    GETORGANELLE_TARGETS,
    AssemblyInputPayload,
    AssemblyRequest,
    GetOrganelleParameters,
    effective_backend_parameters,
)
from organelleverse.assembly.environment_specs import GETORGANELLE_ENVIRONMENT
from organelleverse.assembly.environments import (
    EnvironmentManager,
    PreparedEnvironment,
    PreparedExecutable,
)
from organelleverse.assembly.routing import AssemblyRoute
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleDependencyError, OrganelleExecutionError
from organelleverse.core.frozen import FrozenMap


def _artifact(tmp_path: Path, role: str, content: bytes = b"@r\nACGT\n+\nIIII\n") -> ArtifactRef:
    path = tmp_path / "inputs" / role
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactRef.from_path(
        path, kind="sequence", format="fastq", media_type="application/x-fastq"
    )


def _data(
    tmp_path: Path,
    *,
    layout: Literal["paired_end", "single_end"] = "paired_end",
    auxiliary: Mapping[str, str | None] | None = None,
) -> OrganelleData:
    artifacts = {"read1": _artifact(tmp_path, "read1.fastq")}
    library: dict[str, object] = {
        "technology": "illumina",
        "layout": layout,
        "read1_artifact": "read1",
        "read_length": 150,
    }
    if layout == "paired_end":
        artifacts["read2"] = _artifact(tmp_path, "read2.fastq")
        library["read2_artifact"] = "read2"
    auxiliary_payload: dict[str, str] = {}
    for field, role in (auxiliary or {}).items():
        if role is not None:
            artifacts[role] = _artifact(tmp_path, f"{role}.fasta", b">seed\nACGT\n")
            auxiliary_payload[field] = role
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": artifacts,
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [library],
                "auxiliary": auxiliary_payload,
            },
        }
    )


def _environment(tmp_path: Path) -> PreparedEnvironment:
    prefix = tmp_path / "env"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    executables: list[PreparedExecutable] = []
    for name in (
        "get_organelle_from_reads.py",
        "get_organelle_config.py",
        "blastn",
        "bowtie2",
        "spades.py",
    ):
        path = bin_dir / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        executables.append(PreparedExecutable(name=name, path=path))
    return PreparedEnvironment(
        backend_id="getorganelle",
        carrier="conda",
        platform="linux-64",
        digest="sha256:" + "b" * 64,
        prefix=prefix,
        executables=tuple(executables),
        version="GetOrganelle v1.7.7.1",
    )


def _context(
    tmp_path: Path,
    *,
    data: OrganelleData | None = None,
    organelle: Literal["mitochondrion", "plastid"] = "plastid",
    taxon_group: Literal["plant", "animal", "fungi"] = "plant",
    parameters: GetOrganelleParameters | None = None,
) -> AdapterContext:
    data = data or _data(tmp_path)
    payload = AssemblyInputPayload.model_validate(dict(data.payload))
    request = AssemblyRequest(
        data=data,
        organelle=organelle,
        method="getorganelle",
        backend_parameters=parameters,
        threads=2,
        taxon_group=taxon_group,
    )
    config = tmp_path / "config"
    (config / "SeedDatabase").mkdir(parents=True, exist_ok=True)
    (config / "LabelDatabase").mkdir(parents=True, exist_ok=True)
    target = GETORGANELLE_TARGETS[(organelle, taxon_group)]
    (config / "SeedDatabase" / f"{target}.fasta").write_text(">seed\nACGT\n")
    (config / "LabelDatabase" / f"{target}.fasta").write_text(">label\nACGT\n")
    inventory = config / "organelleverse_getorganelledb.json"
    inventory.write_text("{}")
    config_artifact = ArtifactRef.from_path(
        inventory,
        kind="getorganelledb_config",
        format="json",
        media_type="application/json",
    )
    effective = effective_backend_parameters(
        "getorganelle",
        parameters,
        payload=payload,
        organelle=organelle,
        taxon_group=taxon_group,
    )
    return AdapterContext(
        request=request,
        payload=payload,
        route=AssemblyRoute(
            requested_method="getorganelle",
            selected_backend="getorganelle",
            profile=(
                AssemblyProfile.ILLUMINA_PE
                if payload.short_libraries[0].layout == "paired_end"
                else AssemblyProfile.ILLUMINA_SE
            ),
            rule_id="explicit_method",
            compatible_candidates=("getorganelle",),
        ),
        environment=_environment(tmp_path),
        resources=PreparedBackendResources(
            artifacts=FrozenMap.from_items({"getorganelledb_config": config_artifact}),
            artifact_roles=("getorganelledb_config",),
        ),
        workspace=tmp_path / "workspace",
        input_artifacts=FrozenMap.from_items(dict(data.artifacts.items())),
        effective_backend_parameters=FrozenMap(effective),
    )


def _after(argv: tuple[str, ...], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_preflight_accepts_explicit_pe_and_se(tmp_path: Path) -> None:
    adapter = GetOrganelleAdapter()
    adapter.preflight(_context(tmp_path / "pe", data=_data(tmp_path / "pe")))
    adapter.preflight(_context(tmp_path / "se", data=_data(tmp_path / "se", layout="single_end")))


def test_preflight_rejects_wrong_route_and_environment(tmp_path: Path) -> None:
    adapter = GetOrganelleAdapter()
    context = _context(tmp_path)
    wrong_route = context.model_copy(
        update={
            "route": AssemblyRoute(
                requested_method="oatk",
                selected_backend="oatk",
                profile=AssemblyProfile.PACBIO_HIFI,
                rule_id="explicit_method",
                compatible_candidates=("oatk",),
            )
        }
    )
    with pytest.raises(OrganelleExecutionError, match="non-GetOrganelle route"):
        adapter.preflight(wrong_route)
    wrong_environment = context.model_copy(
        update={"environment": context.environment.model_copy(update={"backend_id": "oatk"})}
    )
    with pytest.raises(OrganelleExecutionError, match="managed environment"):
        adapter.preflight(wrong_environment)


def test_base_argv_has_exact_roles_target_and_infrastructure(tmp_path: Path) -> None:
    command = GetOrganelleAdapter().build_command(_context(tmp_path))
    assert command.stable_argv[:5] == (
        "get_organelle_from_reads.py",
        "-1",
        "role://artifact/read1",
        "-2",
        "role://artifact/read2",
    )
    assert _after(command.stable_argv, "-o") == "role://workspace/output"
    assert _after(command.stable_argv, "-F") == "embplant_pt"
    assert _after(command.stable_argv, "--config-dir") == ("role://artifact/getorganelledb_config")
    assert _after(command.stable_argv, "-t") == "2"
    assert len(command.stable_argv) == len(command.resolved_argv)


def test_single_end_uses_u_and_no_pair_flag(tmp_path: Path) -> None:
    context = _context(tmp_path, data=_data(tmp_path, layout="single_end"))
    command = GetOrganelleAdapter().build_command(context)
    assert _after(command.stable_argv, "-u") == "role://artifact/read1"
    assert "-1" not in command.stable_argv
    assert "-2" not in command.stable_argv


@pytest.mark.parametrize("organelle,taxon_group", GETORGANELLE_TARGETS)
def test_default_databases_do_not_activate_custom_label_branch(
    tmp_path: Path,
    organelle,
    taxon_group,
) -> None:
    context = _context(tmp_path, organelle=organelle, taxon_group=taxon_group)
    command = GetOrganelleAdapter().build_command(context)
    # --genes changes upstream include-priority and graph-slimming defaults.
    # Database availability must be resolved by preparation, not custom flags.
    for argv in (command.stable_argv, command.resolved_argv):
        assert "-s" not in argv
        assert "--genes" not in argv
        assert "--ex-genes" not in argv


_VALUE_CASES: tuple[tuple[str, object, str, str], ...] = (
    ("max_reads", 100, "--max-reads", "100"),
    ("reduce_reads_for_coverage", "inf", "--reduce-reads-for-coverage", "inf"),
    ("max_ignore_percent", 0.5, "--max-ignore-percent", "0.5"),
    ("phred_offset", 33, "--phred-offset", "33"),
    ("min_quality_score", 7, "--min-quality-score", "7"),
    ("output_prefix", "sample", "--prefix", "sample"),
    ("word_size", 0.6, "-w", "0.6"),
    ("pregroup_word_size", 0.7, "--pre-w", "0.7"),
    ("max_rounds", "inf", "-R", "inf"),
    ("max_words", 1000, "--max-n-words", "1000"),
    ("jump_step", 3, "-J", "3"),
    ("mesh_size", 5, "-M", "5"),
    ("bowtie2_options", "--very-sensitive", "--bowtie2-options", "--very-sensitive"),
    ("target_genome_size", 150000, "--target-genome-size", "150000"),
    ("max_extending_length", "auto", "--max-extending-len", "auto"),
    ("spades_kmers", (21, 33, 55), "-k", "21,33,55"),
    ("spades_options", "--careful", "--spades-options", "--careful"),
    ("ignore_kmer", 21, "--ignore-k", "21"),
    ("disentangle_depth_factor", 1.5, "--disentangle-df", "1.5"),
    ("contamination_depth", 3.0, "--contamination-depth", "3.0"),
    ("contamination_similarity", 0.9, "--contamination-similarity", "0.9"),
    ("degenerate_depth", 2.0, "--degenerate-depth", "2.0"),
    ("degenerate_similarity", 0.8, "--degenerate-similarity", "0.8"),
    ("disentangle_time_limit", 60, "--disentangle-time-limit", "60"),
    ("expected_max_size", 200000, "--expected-max-size", "200000"),
    ("expected_min_size", 100000, "--expected-min-size", "100000"),
    ("max_paths", 10, "--max-paths-num", "10"),
    ("pregrouped_reads", 5000, "-P", "5000"),
    ("remove_duplicates", 2, "--remove-duplicates", "2"),
    ("flush_step", "inf", "--flush-step", "inf"),
    ("random_seed", 42, "--random-seed", "42"),
)


@pytest.mark.parametrize(("field", "value", "flag", "expected"), _VALUE_CASES)
def test_every_value_parameter_maps_to_real_cli(
    tmp_path: Path, field: str, value: object, flag: str, expected: str
) -> None:
    parameters = GetOrganelleParameters.model_validate({field: value})
    command = GetOrganelleAdapter().build_command(_context(tmp_path, parameters=parameters))
    assert _after(command.resolved_argv, flag) == expected


_FLAG_CASES = (
    ("output_per_round", "--out-per-round"),
    ("zip_files", "--zip-files"),
    ("keep_temp", "--keep-temp"),
    ("fast", "--fast"),
    ("memory_save", "--memory-save"),
    ("memory_unlimited", "--memory-unlimited"),
    ("larger_auto_word_size", "--larger-auto-ws"),
    ("no_spades", "--no-spades"),
    ("no_degenerate", "--no-degenerate"),
    ("reverse_lsc", "--reverse-lsc"),
    ("index_in_memory", "--index-in-memory"),
    ("verbose", "--verbose"),
)


@pytest.mark.parametrize(("field", "flag"), _FLAG_CASES)
def test_every_boolean_parameter_maps_to_real_cli(tmp_path: Path, field: str, flag: str) -> None:
    parameters = GetOrganelleParameters.model_validate({field: True})
    command = GetOrganelleAdapter().build_command(_context(tmp_path, parameters=parameters))
    assert flag in command.resolved_argv


def test_presets_precede_explicit_controls(tmp_path: Path) -> None:
    parameters = GetOrganelleParameters(fast=True, max_reads=100)
    argv = (
        GetOrganelleAdapter().build_command(_context(tmp_path, parameters=parameters)).resolved_argv
    )
    assert argv.index("--fast") < argv.index("--max-reads")


@pytest.mark.parametrize(
    ("organelle", "taxon_group", "target"),
    tuple((key[0], key[1], value) for key, value in GETORGANELLE_TARGETS.items()),
)
def test_every_target_mapping(
    tmp_path: Path,
    organelle: Literal["mitochondrion", "plastid"],
    taxon_group: Literal["plant", "animal", "fungi"],
    target: str,
) -> None:
    command = GetOrganelleAdapter().build_command(
        _context(tmp_path, organelle=organelle, taxon_group=taxon_group)
    )
    assert _after(command.resolved_argv, "-F") == target


def test_nested_option_metacharacters_remain_one_resolved_token(tmp_path: Path) -> None:
    value = "--very-sensitive; touch /tmp/nope"
    parameters = GetOrganelleParameters(bowtie2_options=value)
    command = GetOrganelleAdapter().build_command(_context(tmp_path, parameters=parameters))
    assert _after(command.resolved_argv, "--bowtie2-options") == value
    assert _after(command.stable_argv, "--bowtie2-options").startswith("sha256_")
    assert len(command.stable_argv) == len(command.resolved_argv)


def test_artifact_options_are_role_addressed(tmp_path: Path) -> None:
    auxiliary = {
        "seed_fasta_artifact": "seed",
        "anti_seed_artifact": "anti",
        "label_genes_artifact": "genes",
        "exclude_genes_artifact": "exclude",
    }
    command = GetOrganelleAdapter().build_command(
        _context(tmp_path, data=_data(tmp_path, auxiliary=auxiliary))
    )
    for flag, role in (
        ("-s", "seed"),
        ("-a", "anti"),
        ("--genes", "genes"),
        ("--ex-genes", "exclude"),
    ):
        assert command.stable_argv.count(flag) == 1
        assert _after(command.stable_argv, flag) == f"role://artifact/{role}"


def test_forbidden_workspace_and_dependency_controls_never_appear(tmp_path: Path) -> None:
    argv = GetOrganelleAdapter().build_command(_context(tmp_path)).resolved_argv
    for flag in (
        "--continue",
        "--overwrite",
        "--which-blast",
        "--which-bowtie2",
        "--which-spades",
        "--which-bandage",
    ):
        assert flag not in argv


def _write_raw_outputs(context: AdapterContext, *, alternate: bool = False) -> Path:
    root = context.workspace / "backend" / "getorganelle" / "output"
    root.mkdir(parents=True, exist_ok=True)
    (root / "sample.path_sequence.fasta").write_text(">seq1\nACGTACGT\n")
    if alternate:
        (root / "sample.zz.path_sequence.fasta").write_text(">seq2\nAAAACCCC\n")
    (root / "sample.selected_graph.gfa").write_text("S\tseq1\tACGTACGT\n")
    (root / "get_org.log.txt").write_text("ok\n")
    return root


def test_collects_real_root_layout_and_preserves_alternates(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_raw_outputs(context, alternate=True)
    raw = GetOrganelleAdapter().collect_outputs(context)
    assert raw.primary_sequence_role == "path_fasta_0"
    assert raw.primary_graph_role == "selected_graph"
    assert [item.role for item in raw.outputs] == [
        "selected_graph",
        "path_fasta_0",
        "path_fasta_1",
        "backend_log",
    ]


def test_collect_rejects_nested_results_fastg_and_graph_ambiguity(tmp_path: Path) -> None:
    adapter = GetOrganelleAdapter()
    context = _context(tmp_path)
    root = context.workspace / "backend" / "getorganelle" / "output"
    nested = root / "seed"
    nested.mkdir(parents=True)
    (nested / "bad.path_sequence.fasta").write_text(">bad\nACGT\n")
    (root / "bad.fastg").write_text("fastg\n")
    with pytest.raises(OrganelleExecutionError, match="path FASTA"):
        adapter.collect_outputs(context)
    _write_raw_outputs(context)
    (root / "other.path_sequence.gfa").write_text("S\tseq2\tAAAA\n")
    with pytest.raises(OrganelleExecutionError, match="exactly one"):
        adapter.collect_outputs(context)


def test_normalize_produces_valid_primary_graph_and_alternate(tmp_path: Path) -> None:
    adapter = GetOrganelleAdapter()
    context = _context(tmp_path)
    _write_raw_outputs(context, alternate=True)
    normalized = adapter.normalize(
        context, adapter.collect_outputs(context), tmp_path / "normalized"
    )
    assert normalized.record_count == 1
    assert normalized.total_bases == 8
    assert normalized.alternate_sequence_roles == ("alternate_fasta_1",)
    assert normalized.primary_sequence.path.read_text() == ">seq1\nACGTACGT\n"


class _ArchiveManager(EnvironmentManager):
    def __init__(self, cache_root: Path, archive: bytes) -> None:
        super().__init__(cache_root=cache_root)
        self.archive = archive
        self.materialize_calls = 0

    def materialize_database_files(self, database: object) -> Path:
        self.materialize_calls += 1
        root = self.cache_root / "archive"
        root.mkdir(parents=True, exist_ok=True)
        (root / GETORGANELLE_ENVIRONMENT.require_database().files[0].name).write_bytes(self.archive)
        return root


def _archive_bytes(version: str = "0.0.1") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for target in GETORGANELLE_TARGETS.values():
            content = f">{target}\nACGT\n".encode()
            info = tarfile.TarInfo(
                "GetOrganelleDB-"
                + GETORGANELLE_ENVIRONMENT.require_database().source_commit
                + f"/{version}/SeedDatabase/{target}.fasta"
            )
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def _config_environment(tmp_path: Path) -> PreparedEnvironment:
    environment = _environment(tmp_path)
    executable = environment.require_executable("get_organelle_config.py")
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "targets = sys.argv[sys.argv.index('-a') + 1].split(',')\n"
        "source = pathlib.Path(sys.argv[sys.argv.index('--use-local') + 1])\n"
        "assert all((source / 'SeedDatabase' / f'{target}.fasta').is_file() for target in targets)\n"
        "config = pathlib.Path(sys.argv[sys.argv.index('--config-dir') + 1])\n"
        "seed = config / 'SeedDatabase'\n"
        "label = config / 'LabelDatabase'\n"
        "seed.mkdir(parents=True, exist_ok=True)\n"
        "label.mkdir(parents=True, exist_ok=True)\n"
        "for target in targets:\n"
        " (seed / f'{target}.fasta').write_text('>seed\\nACGT\\n')\n"
        " (label / f'{target}.fasta').write_text('>label\\nACGT\\n')\n"
    )
    executable.chmod(0o755)
    return environment


@pytest.mark.parametrize(
    "organelle,taxon_group,targets",
    [
        ("plastid", "plant", {"embplant_pt", "embplant_mt"}),
        ("mitochondrion", "plant", {"embplant_pt", "embplant_mt"}),
        ("mitochondrion", "animal", {"animal_mt"}),
        ("mitochondrion", "fungi", {"fungus_mt"}),
    ],
)
def test_provider_prepares_upstream_required_databases(
    tmp_path: Path,
    organelle,
    taxon_group,
    targets,
) -> None:
    manager = _ArchiveManager(tmp_path / "cache", _archive_bytes())
    provider = GetOrganelleResourceProvider()
    context = _context(tmp_path, organelle=organelle, taxon_group=taxon_group)
    expected = provider.expected(
        manager, GETORGANELLE_ENVIRONMENT, context.request, context.payload
    )
    assert manager.materialize_calls == 0
    prepared = provider.prepare(
        manager,
        GETORGANELLE_ENVIRONMENT,
        context.request,
        context.payload,
        _config_environment(tmp_path),
    )
    assert prepared.database_hashes == expected.database_hashes
    artifact = prepared.artifacts["getorganelledb_config"]
    assert artifact.format == "json"
    root = Path(artifact.uri).parent
    for directory in ("SeedDatabase", "LabelDatabase"):
        assert {p.stem for p in (root / directory).glob("*.fasta")} == targets


def test_provider_rejects_tampered_prepared_database(tmp_path: Path) -> None:
    manager = _ArchiveManager(tmp_path / "cache", _archive_bytes())
    provider = GetOrganelleResourceProvider()
    context = _context(tmp_path)
    environment = _config_environment(tmp_path)
    prepared = provider.prepare(
        manager,
        GETORGANELLE_ENVIRONMENT,
        context.request,
        context.payload,
        environment,
    )
    root = Path(prepared.artifacts["getorganelledb_config"].uri).parent
    (root / "SeedDatabase" / "embplant_pt.fasta").write_text("tampered\n")
    with pytest.raises(OrganelleDependencyError, match="changed"):
        provider.prepare(
            manager,
            GETORGANELLE_ENVIRONMENT,
            context.request,
            context.payload,
            environment,
        )


def test_provider_rejects_archive_without_full_database(tmp_path):
    manager = _ArchiveManager(tmp_path / "cache", _archive_bytes("0.0.1.minima"))
    context = _context(tmp_path)
    with pytest.raises(OrganelleDependencyError, match="missing the declared full database"):
        GetOrganelleResourceProvider().prepare(
            manager,
            GETORGANELLE_ENVIRONMENT,
            context.request,
            context.payload,
            _config_environment(tmp_path),
        )
