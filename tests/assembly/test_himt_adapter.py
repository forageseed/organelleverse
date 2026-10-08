from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from organelleverse.assembly.backends import AssemblyProfile, PreparedBackendResources
from organelleverse.assembly.backends.base import AdapterContext
from organelleverse.assembly.backends.himt import HimtAdapter
from organelleverse.assembly.contracts import (
    AssemblyInputPayload,
    AssemblyRequest,
    HimtParameters,
    ShortLibraryContract,
)
from organelleverse.assembly.environment_specs import HIMT_ENVIRONMENT
from organelleverse.assembly.environments import PreparedEnvironment, PreparedExecutable
from organelleverse.assembly.normalization import (
    normalize_fasta,
    normalize_gfa,
    validate_fasta_against_gfa,
)
from organelleverse.assembly.routing import AssemblyRoute
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.frozen import FrozenMap


def _read_artifact(
    tmp_path: Path,
    name: str,
    content: bytes = b"@read\nACGT\n+\nIIII\n",
    format: str = "fastq",
) -> ArtifactRef:
    path = tmp_path / "reads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactRef.from_path(
        path, kind="long_read", format=format, media_type="application/x-fastq"
    )


def _data(
    tmp_path: Path,
    *,
    technology: str,
    quality_state: str,
    reads: tuple[ArtifactRef, ...],
    auxiliary: dict[str, object] | None = None,
) -> OrganelleData:
    artifacts: dict[str, ArtifactRef] = {}
    long_libraries: list[dict[str, object]] = []
    for index, read in enumerate(reads):
        role = f"long_{index}_reads"
        artifacts[role] = read
        long_libraries.append(
            {"technology": technology, "quality_state": quality_state, "reads_artifact": role}
        )
    payload: dict[str, object] = {
        "contract_version": "organelleverse.assembly-input.v1",
        "long_libraries": long_libraries,
    }
    if auxiliary is not None:
        payload["auxiliary"] = auxiliary
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": artifacts,
            "payload": payload,
        }
    )


def _profile_for(technology: str, quality_state: str) -> AssemblyProfile:
    mapping: dict[tuple[str, str], AssemblyProfile] = {
        ("pacbio_hifi", "ccs"): AssemblyProfile.PACBIO_HIFI,
        ("pacbio_clr", "raw"): AssemblyProfile.PACBIO_CLR_RAW,
        ("pacbio_clr", "corrected"): AssemblyProfile.PACBIO_CLR_CORRECTED,
        ("ont", "raw"): AssemblyProfile.ONT_RAW,
        ("ont", "corrected"): AssemblyProfile.ONT_CORRECTED,
        ("ont", "hq"): AssemblyProfile.ONT_HQ,
        ("ont", "duplex"): AssemblyProfile.ONT_DUPLEX,
    }
    return mapping[(technology, quality_state)]


def _environment(tmp_path: Path) -> PreparedEnvironment:
    prefix = tmp_path / "env" / "himt"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    executables: list[PreparedExecutable] = []
    for name in ("himt", "flye", "makeblastdb", "tblastn", "minimap2", "miniprot"):
        path = prefix / "bin" / name
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
        executables.append(PreparedExecutable(name=name, path=path))
    return PreparedEnvironment(
        backend_id="himt",
        carrier="conda",
        platform="linux-64",
        digest="sha256:" + "a" * 64,
        prefix=prefix,
        executables=tuple(executables),
        version="HiMT 1.1.3",
    )


def _context(
    tmp_path: Path,
    *,
    technology: str = "pacbio_hifi",
    quality_state: str = "ccs",
    organelle: Literal["mitochondrion", "plastid"] = "mitochondrion",
    reads: tuple[ArtifactRef, ...] | None = None,
    backend_parameters: HimtParameters | None = None,
    threads: int = 8,
    auxiliary: dict[str, object] | None = None,
) -> AdapterContext:
    if reads is None:
        reads = (_read_artifact(tmp_path, "single.fastq"),)
    data = _data(
        tmp_path,
        technology=technology,
        quality_state=quality_state,
        reads=reads,
        auxiliary=auxiliary,
    )
    request = AssemblyRequest(
        data=data,
        organelle=organelle,
        method="himt",
        backend_parameters=backend_parameters,
        threads=threads,
    )
    payload = AssemblyInputPayload.model_validate(dict(data.payload))
    route = AssemblyRoute(
        requested_method="himt",
        selected_backend="himt",
        profile=_profile_for(technology, quality_state),
        rule_id="explicit.backend",
        compatible_candidates=("himt",),
    )
    environment = _environment(tmp_path)
    input_artifacts = {f"long_{index}_reads": read for index, read in enumerate(reads)}
    effective: dict[str, object]
    if len(payload.long_libraries) == 1:
        from organelleverse.assembly.contracts import effective_backend_parameters

        effective = effective_backend_parameters("himt", backend_parameters, payload=payload)
    else:
        effective = HimtParameters().model_dump(mode="json")
    return AdapterContext(
        request=request,
        payload=payload,
        route=route,
        environment=environment,
        profile=None,
        resources=PreparedBackendResources(),
        workspace=tmp_path / "workspace",
        input_artifacts=FrozenMap.from_items(input_artifacts),
        effective_backend_parameters=FrozenMap.from_json(effective),
    )


def _value_after(argv: tuple[str, ...], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


@pytest.mark.parametrize(
    ("technology", "quality_state", "data_type", "accuracy"),
    [
        ("pacbio_hifi", "ccs", "HiFi", "0.8"),
        ("pacbio_clr", "raw", "CLR", "0.3"),
        ("pacbio_clr", "corrected", "CLR", "0.8"),
        ("ont", "raw", "ONT", "0.3"),
        ("ont", "corrected", "ONT", "0.8"),
        ("ont", "hq", "ONT", "0.8"),
        ("ont", "duplex", "ONT", "0.8"),
    ],
)
def test_himt_profile_mapping(
    tmp_path: Path,
    technology: str,
    quality_state: str,
    data_type: str,
    accuracy: str,
) -> None:
    context = _context(tmp_path, technology=technology, quality_state=quality_state)
    command = HimtAdapter().build_command(context)
    assert _value_after(command.stable_argv, "-d") == data_type
    assert _value_after(command.stable_argv, "-c") == accuracy


def test_himt_command_emits_all_nondefault_options_in_fixed_order(tmp_path: Path) -> None:
    weird = _read_artifact(tmp_path, "weird $(name)`.fastq")
    params = HimtParameters(
        species="animal",
        kmer_length=31,
        head_number=6,
        extract_parallel=3,
        base_number=4,
        filter_depth=12,
        filter_percentage=0.2,
        proportion=0.5,
        accuracy=0.61,
        no_flye_meta=True,
        normalize_depth=-1,
    )
    context = _context(
        tmp_path,
        technology="ont",
        quality_state="raw",
        organelle="mitochondrion",
        reads=(weird,),
        backend_parameters=params,
        threads=8,
    )
    command = HimtAdapter().build_command(context)

    assert command.stable_argv == (
        "himt",
        "assemble",
        "-i",
        "role://artifact/long_0_reads",
        "-o",
        "role://workspace/assembly",
        "-s",
        "animal",
        "-d",
        "ONT",
        "-k",
        "31",
        "-n",
        "6",
        "-t",
        "8",
        "-e",
        "3",
        "-b",
        "4",
        "-fd",
        "12",
        "-fp",
        "0.2",
        "-p",
        "0.5",
        "-c",
        "0.61",
        "-x",
        "-1",
        "--no_flye_meta",
    )
    # The weird path stays as a single argv element.
    assert command.resolved_argv[3] == str(weird.uri)


def test_himt_command_keeps_shell_metacharacters_as_single_argv(tmp_path: Path) -> None:
    weird = tmp_path / "reads" / "weird $(name)`.fastq"
    weird.parent.mkdir(parents=True, exist_ok=True)
    weird.write_bytes(b"@r\nACGT\n+\nIIII\n")
    artifact = ArtifactRef.from_path(
        weird, kind="long_read", format="fastq", media_type="application/x-fastq"
    )
    context = _context(tmp_path, reads=(artifact,))
    command = HimtAdapter().build_command(context)
    assert command.resolved_argv[3] == str(weird)


def test_preflight_rejects_non_himt_route(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context = context.model_copy(
        update={"route": context.route.model_copy(update={"selected_backend": "oatk"})}
    )
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().preflight(context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_non_himt_environment(tmp_path: Path) -> None:
    context = _context(tmp_path)
    environment = context.environment.model_copy(update={"backend_id": "oatk"})
    context = context.model_copy(update={"environment": environment})
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().preflight(context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_zero_long_libraries(tmp_path: Path) -> None:
    context = _context(tmp_path)
    empty_payload = AssemblyInputPayload.model_construct(
        contract_version="organelleverse.assembly-input.v1",
        short_libraries=(),
        long_libraries=(),
        contig_inputs=(),
        auxiliary=context.payload.auxiliary,
    )
    context = AdapterContext.model_construct(
        **{
            field: getattr(context, field)
            for field in AdapterContext.model_fields
            if field != "payload"
        },
        payload=empty_payload,
    )
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().preflight(context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_two_long_libraries(tmp_path: Path) -> None:
    first = _read_artifact(tmp_path, "first.fastq")
    second = _read_artifact(tmp_path, "second.fastq")
    context = _context(tmp_path, reads=(first, second))
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().preflight(context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_short_reads(tmp_path: Path) -> None:
    context = _context(tmp_path)
    short_payload = AssemblyInputPayload.model_construct(
        contract_version="organelleverse.assembly-input.v1",
        short_libraries=(
            ShortLibraryContract.model_construct(
                technology="illumina",
                layout="paired_end",
                read1_artifact="short_0_read1",
                read2_artifact=None,
                read_length=150,
                insert_size=None,
            ),
        ),
        long_libraries=context.payload.long_libraries,
        contig_inputs=(),
        auxiliary=context.payload.auxiliary,
    )
    context = AdapterContext.model_construct(
        **{
            field: getattr(context, field)
            for field in AdapterContext.model_fields
            if field != "payload"
        },
        payload=short_payload,
    )
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().preflight(context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_contig_input(tmp_path: Path) -> None:
    read = _read_artifact(tmp_path, "single.fastq")
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"long_0_reads": read, "contig_0_fasta": read},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_0_reads",
                    }
                ],
                "contig_inputs": [{"fasta_artifact": "contig_0_fasta"}],
            },
        }
    )
    request = AssemblyRequest(
        data=data,
        organelle="mitochondrion",
        method="himt",
    )
    payload = AssemblyInputPayload.model_validate(dict(data.payload))
    context = _context(tmp_path, reads=(read,))
    context = context.model_copy(update={"request": request, "payload": payload})
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().preflight(context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_auxiliary_input(tmp_path: Path) -> None:
    read = _read_artifact(tmp_path, "single.fastq")
    context = _context(tmp_path, reads=(read,), auxiliary={"hmm_profiles_artifact": "long_0_reads"})
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().preflight(context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_unsupported_format(tmp_path: Path) -> None:
    bad = _read_artifact(tmp_path, "single.bam", format="bam")
    context = _context(tmp_path, reads=(bad,))
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().preflight(context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_filter_depth_without_proportion(tmp_path: Path) -> None:
    params = HimtParameters(filter_depth=5, proportion=0.0)
    context = _context(tmp_path, backend_parameters=params)
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().preflight(context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_animal_plastid(tmp_path: Path) -> None:
    params = HimtParameters(species="animal")
    context = _context(tmp_path, organelle="plastid", backend_parameters=params)
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().preflight(context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_accepts_fasta_and_fastq_gz(tmp_path: Path) -> None:
    for name in ("single.fa", "single.fasta", "single.fq.gz", "single.fastq.gz"):
        content = b">ctg\nACGT\n" if name.endswith((".fa", ".fasta")) else b"@r\nACGT\n+\nIIII\n"
        read = _read_artifact(tmp_path, name, content=content, format=name.split(".")[1])
        context = _context(tmp_path, reads=(read,))
        HimtAdapter().preflight(context)


def _write_fasta(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_gfa(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_collect_outputs_mitochondrion_only(tmp_path: Path) -> None:
    context = _context(tmp_path)
    backend = context.workspace / "backend" / "himt"
    _write_fasta(backend / "himt_mitochondrial_raw.fa", ">ctg1\nACGT\n")
    _write_gfa(backend / "himt_mitochondrial.gfa", "H\tVN:Z:1.0\n")

    raw = HimtAdapter().collect_outputs(context)
    assert raw.primary_sequence_role == "assembly_fasta"
    assert raw.primary_graph_role == "assembly_graph"
    assert tuple(item.role for item in raw.outputs) == ("assembly_fasta", "assembly_graph")
    assert raw.require("assembly_fasta").path.name == "himt_mitochondrial_raw.fa"


def test_collect_outputs_mitochondrion_plus_plastid(tmp_path: Path) -> None:
    context = _context(tmp_path)
    backend = context.workspace / "backend" / "himt"
    _write_fasta(backend / "himt_mitochondrial_raw.fa", ">ctg1\nACGT\n")
    _write_gfa(backend / "himt_mitochondrial.gfa", "H\tVN:Z:1.0\n")
    _write_fasta(backend / "chloroplast_path1.fa", ">pt1\nACGT\n")
    _write_gfa(backend / "himt_chloroplast.gfa", "H\tVN:Z:1.0\n")

    raw = HimtAdapter().collect_outputs(context)
    roles = tuple(item.role for item in raw.outputs)
    assert roles == (
        "assembly_fasta",
        "assembly_graph",
        "detected_plastid_graph",
        "detected_plastid_path1",
    )


def test_collect_outputs_plastid_path1_only(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="plastid")
    backend = context.workspace / "backend" / "himt"
    _write_fasta(backend / "chloroplast_path1.fa", ">pt1\nACGT\n")
    _write_gfa(backend / "himt_chloroplast.gfa", "H\tVN:Z:1.0\n")
    _write_fasta(backend / "himt_mitochondrial_raw.fa", ">mt1\nACGT\n")
    _write_gfa(backend / "himt_mitochondrial.gfa", "H\tVN:Z:1.0\n")

    raw = HimtAdapter().collect_outputs(context)
    roles = tuple(item.role for item in raw.outputs)
    assert roles == (
        "assembly_fasta",
        "assembly_graph",
        "detected_mitochondrion_fasta",
        "detected_mitochondrion_graph",
    )
    # alternate_sequence_roles is computed during normalization


def test_collect_outputs_plastid_path1_and_path2(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="plastid")
    backend = context.workspace / "backend" / "himt"
    _write_fasta(backend / "chloroplast_path1.fa", ">pt1\nACGT\n")
    _write_gfa(backend / "himt_chloroplast.gfa", "H\tVN:Z:1.0\n")
    _write_fasta(backend / "chloroplast_path2.fa", ">pt2\nACGT\n")
    _write_fasta(backend / "himt_mitochondrial_raw.fa", ">mt1\nACGT\n")
    _write_gfa(backend / "himt_mitochondrial.gfa", "H\tVN:Z:1.0\n")

    raw = HimtAdapter().collect_outputs(context)
    roles = tuple(item.role for item in raw.outputs)
    assert roles == (
        "assembly_fasta",
        "assembly_graph",
        "alternate_assembly_fasta",
        "detected_mitochondrion_fasta",
        "detected_mitochondrion_graph",
    )
    # alternate_sequence_roles is computed during normalization


def test_collect_outputs_missing_requested_target_fails(tmp_path: Path) -> None:
    context = _context(tmp_path)
    backend = context.workspace / "backend" / "himt"
    _write_gfa(backend / "himt_mitochondrial.gfa", "H\tVN:Z:1.0\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        HimtAdapter().collect_outputs(context)
    assert raised.value.code == "assembly.output_incomplete"


def test_collect_outputs_partial_detected_plastid_is_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    backend = context.workspace / "backend" / "himt"
    _write_fasta(backend / "himt_mitochondrial_raw.fa", ">ctg1\nACGT\n")
    _write_gfa(backend / "himt_mitochondrial.gfa", "H\tVN:Z:1.0\n")
    # graph present but path1 missing
    _write_gfa(backend / "himt_chloroplast.gfa", "H\tVN:Z:1.0\n")
    raw = HimtAdapter().collect_outputs(context)
    roles = tuple(item.role for item in raw.outputs)
    assert "detected_plastid_path1" not in roles
    assert "detected_plastid_graph" in roles


def test_normalize_mitochondrion_output(tmp_path: Path) -> None:
    context = _context(tmp_path)
    backend = context.workspace / "backend" / "himt"
    _write_fasta(backend / "himt_mitochondrial_raw.fa", ">ctg1\nACGTACGT\n")
    _write_gfa(backend / "himt_mitochondrial.gfa", "H\tVN:Z:1.0\nS\tctg1\tACGTACGT\n")

    raw = HimtAdapter().collect_outputs(context)
    normalized = HimtAdapter().normalize(context, raw, tmp_path / "normalized")
    assert normalized.primary_sequence.path.name == "assembly.fasta"
    assert normalized.primary_graph is not None
    assert normalized.primary_graph.path.name == "assembly.gfa"
    assert normalized.alternate_sequence_roles == ()
    assert normalized.record_count == 1
    assert normalized.total_bases == 8


def test_normalize_plastid_output(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="plastid")
    backend = context.workspace / "backend" / "himt"
    _write_fasta(backend / "chloroplast_path1.fa", ">pt1\nACGTACGT\n")
    _write_gfa(backend / "himt_chloroplast.gfa", "H\tVN:Z:1.0\nS\tpt1\tACGTACGT\n")
    _write_fasta(backend / "chloroplast_path2.fa", ">pt2\nACGTACGT\n")

    raw = HimtAdapter().collect_outputs(context)
    normalized = HimtAdapter().normalize(context, raw, tmp_path / "normalized")
    assert normalized.require("assembly_fasta").path.name == "assembly.fasta"
    assert normalized.require("assembly_graph").path.name == "assembly.gfa"
    assert (
        normalized.require("alternate_assembly_fasta").path.name == "alternate_assembly_fasta.fasta"
    )
    assert normalized.alternate_sequence_roles == ("alternate_assembly_fasta",)


def test_normalize_repairs_himt_three_segment_plastid_orientation_bug(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, organelle="plastid")
    backend = context.workspace / "backend" / "himt"
    repeat = "AACG"
    single_copy = "AGTC"
    other_single_copy = "CCTA"
    _write_gfa(
        backend / "himt_chloroplast.gfa",
        "H\tVN:Z:1.0\n"
        f"S\trepeat\t{repeat}\n"
        f"S\tsingle_a\t{single_copy}\n"
        f"S\tsingle_b\t{other_single_copy}\n"
        "L\trepeat\t+\tsingle_a\t-\t0M\n"
        "L\trepeat\t+\tsingle_a\t+\t0M\n"
        "L\trepeat\t-\tsingle_b\t-\t0M\n"
        "L\trepeat\t-\tsingle_b\t+\t0M\n",
    )
    # HiMT 1.1.3 concatenates segment strings without applying the GFA link
    # orientations. Preserve this raw evidence, but require the adapter's
    # normalized path to be a path the graph actually spells.
    _write_fasta(
        backend / "chloroplast_path1.fa",
        f">path1\n{single_copy}{repeat}{other_single_copy}CGTT\n",
    )
    _write_fasta(
        backend / "chloroplast_path2.fa",
        f">path2\n{single_copy}{repeat}TAGGCGTT\n",
    )

    raw = HimtAdapter().collect_outputs(context)
    normalized = HimtAdapter().normalize(context, raw, tmp_path / "normalized")

    graph = normalize_gfa(
        normalized.require("assembly_graph").path,
        tmp_path / "checked.gfa",
    )
    for role in ("assembly_fasta", "alternate_assembly_fasta"):
        fasta = normalize_fasta(
            normalized.require(role).path,
            tmp_path / f"checked-{role}.fasta",
        )
        validate_fasta_against_gfa(fasta, graph)
    assert (
        (backend / "chloroplast_path1.fa")
        .read_text()
        .endswith(f"{single_copy}{repeat}{other_single_copy}CGTT\n")
    )


def test_normalize_rejects_malformed_fasta(tmp_path: Path) -> None:
    context = _context(tmp_path)
    backend = context.workspace / "backend" / "himt"
    _write_fasta(backend / "himt_mitochondrial_raw.fa", "not a fasta")
    _write_gfa(backend / "himt_mitochondrial.gfa", "H\tVN:Z:1.0\n")

    raw = HimtAdapter().collect_outputs(context)
    with pytest.raises(OrganelleExecutionError):
        HimtAdapter().normalize(context, raw, tmp_path / "normalized")


def test_environment_is_database_free_and_cross_platform() -> None:
    assert HIMT_ENVIRONMENT.backend_id == "himt"
    assert HIMT_ENVIRONMENT.database is None
    assert tuple(item.platform for item in HIMT_ENVIRONMENT.platforms) == (
        "linux-64",
        "linux-aarch64",
        "osx-64",
        "osx-arm64",
    )
    assert all(item.package == "himt=1.1.3=0" for item in HIMT_ENVIRONMENT.platforms)
    assert all(
        item.executable_names == ("himt", "flye", "makeblastdb", "tblastn", "minimap2", "miniprot")
        for item in HIMT_ENVIRONMENT.platforms
    )
