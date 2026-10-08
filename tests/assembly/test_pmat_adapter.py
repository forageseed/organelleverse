from __future__ import annotations

import gzip
from pathlib import Path
from typing import Literal

import pytest

from organelleverse.assembly.backends.base import AdapterContext, PreparedBackendResources
from organelleverse.assembly.backends.pmat import PmatAdapter, _materialize_reads
from organelleverse.assembly.backends.pmat_graph import PmatGraphAdapter, PmatGraphContext
from organelleverse.assembly.continuation import DirectoryManifest
from organelleverse.assembly.contracts import (
    AssemblyInputPayload,
    AssemblyRequest,
    PmatParameters,
    effective_pmat_parameters,
)
from organelleverse.assembly.environments import PreparedEnvironment, PreparedExecutable
from organelleverse.assembly.profiles import AssemblyProfile
from organelleverse.assembly.routing import AssemblyRoute
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleExecutionError, OrganelleInputError
from organelleverse.core.frozen import FrozenMap

_PROFILES = {
    ("pacbio_hifi", "ccs"): AssemblyProfile.PACBIO_HIFI,
    ("pacbio_clr", "raw"): AssemblyProfile.PACBIO_CLR_RAW,
    ("pacbio_clr", "corrected"): AssemblyProfile.PACBIO_CLR_CORRECTED,
    ("ont", "raw"): AssemblyProfile.ONT_RAW,
    ("ont", "corrected"): AssemblyProfile.ONT_CORRECTED,
    ("ont", "hq"): AssemblyProfile.ONT_HQ,
    ("ont", "duplex"): AssemblyProfile.ONT_DUPLEX,
}


@pytest.mark.parametrize("compressed", [False, True])
def test_corrected_single_library_is_used_directly_and_rechecked(tmp_path, compressed):
    raw = b"@r\nACGT\n+\nIIII\n"
    artifact = _artifact(
        tmp_path / "source.fastq.gz",
        gzip.compress(raw) if compressed else raw,
        kind="long_read",
        format="fastq",
    )
    context = _context(tmp_path, technology="pacbio_hifi", quality_state="ccs", reads=(artifact,))
    assert _materialize_reads(context) == Path(artifact.uri).resolve()
    assert not (context.workspace / "input").exists()
    _write_pmat_outputs(context, target="mt")
    assert PmatAdapter().collect_outputs(context).primary_sequence_role == "assembly_fasta"
    Path(artifact.uri).write_bytes(b"@r\nTGCA\n+\nIIII\n")
    with pytest.raises(OrganelleInputError, match="hash no longer matches"):
        PmatAdapter().collect_outputs(context)


def test_multiple_libraries_still_concatenate_decompressed_records(tmp_path):
    parts = (b"@r1\nACGT\n+\nIIII\n", b"@r2\nTGCA\n+\nIIII\n")
    refs = tuple(
        _artifact(tmp_path / f"{i}.fq.gz", gzip.compress(raw), kind="long_read", format="fastq")
        for i, raw in enumerate(parts)
    )
    context = _context(tmp_path, technology="pacbio_hifi", quality_state="ccs", reads=refs)
    output = _materialize_reads(context)
    assert output.parent == context.workspace / "input"
    assert output.read_bytes() == b"".join(parts)


def _artifact(path: Path, content: bytes, *, kind: str, format: str) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactRef.from_path(
        path, kind=kind, format=format, media_type="application/octet-stream"
    )


def _context(
    tmp_path: Path,
    *,
    technology: str = "ont",
    quality_state: str = "raw",
    params: PmatParameters | None = None,
    reads: tuple[ArtifactRef, ...] | None = None,
    auxiliary: dict[str, object] | None = None,
    extra_artifacts: dict[str, ArtifactRef] | None = None,
    organelle: Literal["mitochondrion", "plastid"] = "mitochondrion",
) -> AdapterContext:
    if reads is None:
        reads = (
            _artifact(
                tmp_path / "reads" / "reads.fastq",
                b"@r\nACGT\n+\nIIII\n",
                kind="long_read",
                format="fastq",
            ),
        )
    artifacts = dict(extra_artifacts or {})
    libraries: list[dict[str, object]] = []
    for index, artifact in enumerate(reads):
        role = f"long_{index}_reads"
        artifacts[role] = artifact
        libraries.append(
            {
                "technology": technology,
                "quality_state": quality_state,
                "reads_artifact": role,
            }
        )
    payload_data: dict[str, object] = {
        "contract_version": "organelleverse.assembly-input.v1",
        "long_libraries": libraries,
    }
    if auxiliary is not None:
        payload_data["auxiliary"] = auxiliary
    data = OrganelleData.model_validate(
        {"modality": "sequencing_reads", "artifacts": artifacts, "payload": payload_data}
    )
    request = AssemblyRequest(
        data=data,
        organelle=organelle,
        method="pmat",
        backend_parameters=params,
        threads=12,
    )
    payload = AssemblyInputPayload.model_validate(dict(data.payload))
    prefix = tmp_path / "env"
    executables: list[PreparedExecutable] = []
    for name in ("PMAT", "blastn", "canu", "nextdenovo"):
        path = prefix / "bin" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
        executables.append(PreparedExecutable(name=name.lower(), path=path))
    environment = PreparedEnvironment(
        backend_id="pmat",
        carrier="conda",
        platform="linux-64",
        digest="sha256:" + "a" * 64,
        prefix=prefix,
        executables=tuple(executables),
        version="2.1.5",
    )
    effective = effective_pmat_parameters(request, payload, params)
    return AdapterContext(
        request=request,
        payload=payload,
        route=AssemblyRoute(
            requested_method="pmat",
            selected_backend="pmat",
            profile=_PROFILES[(technology, quality_state)],
            rule_id="explicit.backend",
            compatible_candidates=("pmat",),
        ),
        environment=environment,
        resources=PreparedBackendResources(),
        workspace=tmp_path / "workspace",
        input_artifacts=FrozenMap.from_items(artifacts),
        effective_backend_parameters=FrozenMap.from_json(effective),
    )


def _after(argv: tuple[str, ...], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize(
    ("technology", "quality", "seqtype", "task"),
    [
        ("pacbio_hifi", "ccs", "hifi", "0"),
        ("pacbio_clr", "raw", "clr", "1"),
        ("pacbio_clr", "corrected", "clr", "0"),
        ("ont", "raw", "ont", "1"),
        ("ont", "corrected", "ont", "0"),
        ("ont", "hq", "ont", "0"),
        ("ont", "duplex", "ont", "0"),
    ],
)
def test_profile_and_correction_mapping(
    tmp_path: Path, technology: str, quality: str, seqtype: str, task: str
) -> None:
    command = PmatAdapter().build_command(
        _context(tmp_path, technology=technology, quality_state=quality)
    )
    assert _after(command.stable_argv, "-t") == seqtype
    assert _after(command.stable_argv, "-p") == task


def test_nondefault_command_emits_every_public_automito_flag_safely(tmp_path: Path) -> None:
    canu = _artifact(
        tmp_path / "unsafe $(x)" / "canu", b"#!/bin/sh\n", kind="executable", format="binary"
    )
    cfg = _artifact(tmp_path / "unsafe $(x)" / "cfg", b"[correct]\n", kind="config", format="ini")
    params = PmatParameters(
        kmer_size=29,
        correction_task="run",
        correction_software="canu",
        subsample_factor=0.5,
        random_seed=19,
        long_read_break_length=30_000,
        minimum_overlap_identity=95,
        minimum_overlap_length=100,
        keep_sequences_in_memory=True,
    )
    context = _context(
        tmp_path,
        params=params,
        auxiliary={"genome_size": 540_000_000, "canu_executable_artifact": "canu"},
        extra_artifacts={"canu": canu},
    )
    command = PmatAdapter().build_command(context)

    assert command.stable_argv == (
        "PMAT",
        "autoMito",
        "-i",
        "role://workspace/input",
        "-o",
        "role://workspace/output",
        "-t",
        "ont",
        "-k",
        "29",
        "-g",
        "540000000",
        "-p",
        "1",
        "-G",
        "mt",
        "-x",
        "0",
        "-S",
        "canu",
        "-C",
        "role://artifact/canu",
        "-F",
        "0.5",
        "-D",
        "19",
        "-K",
        "30000",
        "-I",
        "95",
        "-L",
        "100",
        "-T",
        "12",
        "-m",
    )
    assert "$(x)" not in " ".join(command.resolved_argv)
    assert Path(_after(command.resolved_argv, "-i")).is_file()
    assert Path(_after(command.resolved_argv, "-C")).is_file()
    assert "-n" not in command.resolved_argv
    assert cfg.uri not in command.resolved_argv


def test_multiple_libraries_are_combined_in_declared_order(tmp_path: Path) -> None:
    first = _artifact(
        tmp_path / "a.fastq", b"@a\nAAAA\n+\nIIII\n", kind="long_read", format="fastq"
    )
    second = _artifact(
        tmp_path / "b.fastq", b"@b\nCCCC\n+\nIIII\n", kind="long_read", format="fastq"
    )
    command = PmatAdapter().build_command(_context(tmp_path, reads=(first, second)))
    combined = Path(_after(command.resolved_argv, "-i"))
    assert combined.read_bytes() == Path(first.uri).read_bytes() + Path(second.uri).read_bytes()
    assert first.uri not in command.resolved_argv
    assert second.uri not in command.resolved_argv


def test_nextdenovo_uses_environment_tools_and_optional_safe_config(tmp_path: Path) -> None:
    cfg = _artifact(
        tmp_path / "unsafe" / "nextdenovo.cfg", b"[correct]\n", kind="config", format="ini"
    )
    context = _context(
        tmp_path,
        auxiliary={"correction_config_artifact": "cfg"},
        extra_artifacts={"cfg": cfg},
    )
    command = PmatAdapter().build_command(context)
    assert _after(command.stable_argv, "-C") == "role://environment/canu"
    assert _after(command.stable_argv, "-N") == "role://environment/nextdenovo"
    assert _after(command.stable_argv, "-n") == "role://artifact/cfg"
    assert Path(_after(command.resolved_argv, "-n")).name == "correction.cfg"


def test_correction_config_is_rejected_when_correction_is_skipped(tmp_path: Path) -> None:
    cfg = _artifact(tmp_path / "cfg", b"[correct]\n", kind="config", format="ini")
    context = _context(
        tmp_path,
        technology="pacbio_hifi",
        quality_state="ccs",
        auxiliary={"correction_config_artifact": "cfg"},
        extra_artifacts={"cfg": cfg},
    )
    with pytest.raises(OrganelleInputError, match="correction config"):
        PmatAdapter().preflight(context)


def _write_pmat_outputs(context: AdapterContext, *, target: str) -> None:
    root = context.workspace / "backend" / "pmat"
    gfa = root / "gfa_result"
    gfa.mkdir(parents=True, exist_ok=True)
    if target == "mt":
        (gfa / "PMAT_mt.fa").write_text(">ctg\nACGTACGT\n")
        (root / "PMAT_orgAss.txt").write_text("Mitochondrial Assembly Assessment\n")
    else:
        (gfa / "PMAT_pt.fa").write_text(">ctg\nACGTACGT\n")
    (gfa / f"PMAT_{target}_main.gfa").write_text("H\tVN:Z:1.0\nS\tctg\tACGTACGT\n")
    (gfa / f"PMAT_{target}_raw.gfa").write_text("H\tVN:Z:1.0\nS\tctg\tACGTACGT\n")
    assembly = root / "assembly_result"
    assembly.mkdir()
    (assembly / "PMATAllContigs.fna").write_text(">ctg\nACGTACGT\n")
    (assembly / "PMATContigGraph.txt").write_text("1\t2\n")
    subsample = root / "subsample"
    subsample.mkdir()
    (subsample / "PMAT_cut_seq.fa").write_text(">read\nACGTACGT\n")


@pytest.mark.parametrize(
    ("organelle", "target", "raw_fasta"),
    [
        ("mitochondrion", "mt", "PMAT_mt.fa"),
        ("plastid", "pt", "PMAT_pt.fa"),
    ],
)
def test_collect_and_normalize_complete_outputs(
    tmp_path: Path,
    organelle: Literal["mitochondrion", "plastid"],
    target: str,
    raw_fasta: str,
) -> None:
    context = _context(tmp_path, organelle=organelle)
    _write_pmat_outputs(context, target=target)

    raw = PmatAdapter().collect_outputs(context)
    normalized = PmatAdapter().normalize(context, raw, tmp_path / "normalized")

    assert raw.require("assembly_fasta").path.name == raw_fasta
    assert normalized.primary_sequence.path.name == "assembly.fasta"
    assert normalized.primary_graph is not None
    assert normalized.primary_graph.path.name == "assembly.gfa"
    assert normalized.record_count == 1
    assert normalized.total_bases == 8
    expected_roles = {
        "assembly_fasta",
        "assembly_graph",
        "raw_assembly_graph",
        "pmat_all_contigs",
        "pmat_contig_graph",
        "pmat_subsample",
        "pmat_subsample_manifest",
        "pmat_assembly_result_manifest",
    }
    if organelle == "mitochondrion":
        expected_roles.add("assembly_assessment")
    assert {item.role for item in normalized.outputs} >= expected_roles
    subsample_manifest = DirectoryManifest.model_validate_json(
        normalized.require("pmat_subsample_manifest").path.read_bytes()
    )
    assert subsample_manifest.files[0].relative_path == "PMAT_cut_seq.fa"


@pytest.mark.parametrize(
    ("organelle", "target", "actual_fasta"),
    [
        ("mitochondrion", "mt", "PMAT_mt.fa"),
        ("plastid", "pt", "PMAT_pt.fa"),
    ],
)
def test_collect_accepts_pmat_2_1_5_primary_fasta_names(
    tmp_path: Path,
    organelle: Literal["mitochondrion", "plastid"],
    target: str,
    actual_fasta: str,
) -> None:
    context = _context(tmp_path, organelle=organelle)
    _write_pmat_outputs(context, target=target)

    raw = PmatAdapter().collect_outputs(context)

    assert raw.require("assembly_fasta").path.name == actual_fasta


def test_collect_rejects_missing_continuation_output(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_pmat_outputs(context, target="mt")
    (context.workspace / "backend" / "pmat" / "subsample" / "PMAT_cut_seq.fa").unlink()

    with pytest.raises(OrganelleExecutionError, match="required output"):
        PmatAdapter().collect_outputs(context)


def test_normalize_rejects_sequence_graph_mismatch(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_pmat_outputs(context, target="mt")
    graph = context.workspace / "backend" / "pmat" / "gfa_result" / "PMAT_mt_main.gfa"
    graph.write_text("H\tVN:Z:1.0\nS\tother\tTTTT\n")
    raw = PmatAdapter().collect_outputs(context)

    with pytest.raises(OrganelleExecutionError, match="not represented"):
        PmatAdapter().normalize(context, raw, tmp_path / "normalized")


@pytest.mark.parametrize("continuation", [False, True], ids=["autoMito", "graphBuild"])
def test_pmat_normalization_preserves_branches_outside_primary_path(
    tmp_path: Path, continuation: bool
) -> None:
    """A PMAT FASTA path is not the complete graph (Nipponbare diagnosis)."""
    context = _context(tmp_path)
    _write_pmat_outputs(context, target="mt")
    main = (
        "H\tVN:Z:1.0\n"
        "S\ta\tAACG\tRC:i:40\n"
        "S\tb\tTTGA\tRC:i:40\n"
        "S\tc\tCGCA\tRC:i:40\n"
        "L\ta\t+\tb\t+\t0M\n"
        "L\ta\t+\tc\t-\t0M\n"
    )
    raw_graph = main + "S\td\tAGTC\tRC:i:4\nL\tc\t-\td\t+\t0M\n"
    root = context.workspace / "backend/pmat"
    (root / "gfa_result/PMAT_mt.fa").write_text(">path_a_b\nAACGTTGA\n")
    (root / "gfa_result/PMAT_mt_main.gfa").write_text(main)
    (root / "gfa_result/PMAT_mt_raw.gfa").write_text(raw_graph)
    if continuation:
        root.rename(context.workspace / "backend/pmat_graph")
        graph_context = PmatGraphContext(
            workspace=context.workspace,
            environment=context.environment,
            subsample_dir=tmp_path / "subsample",
            assembly_result_dir=tmp_path / "contigs",
            organelle="mitochondrion",
            threads=1,
        )
        adapter = PmatGraphAdapter()
        normalized = adapter.normalize(
            graph_context, adapter.collect_outputs(graph_context), tmp_path / "normalized"
        )
    else:
        normalized = PmatAdapter().normalize(
            context, PmatAdapter().collect_outputs(context), tmp_path / "normalized"
        )
    assert normalized.require("assembly_graph").path.read_text() == main
    assert normalized.require("raw_assembly_graph").path.read_text() == raw_graph
    assert normalized.primary_sequence.path.read_text() == ">path_a_b\nAACGTTGA\n"
    assert normalized.record_count == 1
    assert normalized.total_bases == 8
