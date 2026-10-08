"""ovasm as a native assembly backend: registration, opt-in routing, adapter, real run."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pytest

from organelleverse import _ovasm
from organelleverse.assembly import (
    AssemblyAuxiliary,
    LongReadLibrary,
    ShortReadLibrary,
    assemble,
    write,
)
from organelleverse.assembly.backends import BACKENDS, RUNTIMES, AssemblyProfile
from organelleverse.assembly.backends import ovasm as ovasm_backend
from organelleverse.assembly.backends.base import (
    AdapterContext,
    PreparedBackendResources,
)
from organelleverse.assembly.backends.ovasm import NativeOvasmEnvironment, OvasmAdapter
from organelleverse.assembly.contracts import (
    AssemblyInputPayload,
    AssemblyRequest,
    OvasmParameters,
    effective_backend_parameters,
)
from organelleverse.assembly.data_contract import (
    validate_ovasm_assembly_data,
    validate_released_assembly_data,
)
from organelleverse.assembly.environment_specs import OVASM_ENVIRONMENT
from organelleverse.assembly.environments import PreparedEnvironment, PreparedExecutable
from organelleverse.assembly.manifests import AssemblyRunManifest
from organelleverse.assembly.routing import (
    AUTO_RULES,
    AssemblyRoute,
    classify_profile,
    compatible_backends,
    resolve_backend,
)
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.core.frozen import FrozenMap
from organelleverse.io import read_reads


def _fastq(path: Path, text: str = "@r\nACGT\n+\nIIII\n") -> Path:
    path.write_text(text)
    return path


def _long(tmp_path: Path, technology: str = "pacbio_hifi", quality_state: str = "ccs", **aux):
    return read_reads(
        long_libraries=(
            LongReadLibrary(
                technology=technology,
                quality_state=quality_state,
                reads=_fastq(tmp_path / f"{technology}.fastq"),
            ),
        ),
        auxiliary=AssemblyAuxiliary(**aux),
    )


def _illumina(tmp_path: Path, layout: str = "single_end") -> OrganelleData:
    read2 = _fastq(tmp_path / "R2.fastq") if layout == "paired_end" else None
    return read_reads(
        short_libraries=(
            ShortReadLibrary(
                technology="illumina",
                layout=layout,
                read1=_fastq(tmp_path / "R1.fastq"),
                read2=read2,
                read_length=150,
                insert_size=350 if read2 else None,
            ),
        )
    )


def _long_plus_short(
    tmp_path: Path,
    technology: str = "ont",
    quality_state: str = "raw",
    layout: str = "paired_end",
) -> OrganelleData:
    """Long reads together with one Illumina library (second plus third generation)."""
    read2 = _fastq(tmp_path / "H2.fastq") if layout == "paired_end" else None
    return read_reads(
        long_libraries=(
            LongReadLibrary(
                technology=technology,
                quality_state=quality_state,
                reads=_fastq(tmp_path / f"{technology}_long.fastq"),
            ),
        ),
        short_libraries=(
            ShortReadLibrary(
                technology="illumina",
                layout=layout,
                read1=_fastq(tmp_path / "H1.fastq"),
                read2=read2,
                read_length=150,
                insert_size=350 if read2 else None,
            ),
        ),
    )


def _two_short(tmp_path: Path) -> OrganelleData:
    return read_reads(
        short_libraries=tuple(
            ShortReadLibrary(
                technology="illumina",
                layout="single_end",
                read1=_fastq(tmp_path / f"S{n}.fastq"),
                read_length=150,
            )
            for n in (1, 2)
        )
    )


def _payload(data: OrganelleData) -> AssemblyInputPayload:
    return AssemblyInputPayload.model_validate(dict(data.payload))


# --- registration and routing ----------------------------------------------------------------


def test_ovasm_is_a_released_native_backend() -> None:
    spec = BACKENDS["ovasm"]
    assert spec.environment.carrier.value == "native"
    assert spec.organelles == ("mitochondrion", "plastid")
    assert {p.value for p in spec.profiles} == {
        "pacbio_hifi",
        "ont_raw",
        "pacbio_clr_raw",
        "illumina_se",
        "illumina_pe",
        "ont_raw_illumina",
        "pacbio_clr_raw_illumina",
    }
    runtime = RUNTIMES.require("ovasm")
    assert runtime.environment_spec is OVASM_ENVIRONMENT
    assert runtime.environment_spec.carrier == "native"
    # the runtime brings its own environment manager: no conda prefix is prepared for it
    assert runtime.environment_manager_factory is NativeOvasmEnvironment
    assert all(
        other.environment_manager_factory is None
        for name, other in RUNTIMES.items()
        if name != "ovasm"
    )


def test_automatic_routing_never_selects_ovasm(tmp_path: Path) -> None:
    assert all(rule.backend_id != "ovasm" for rule in AUTO_RULES)
    cases = [
        (_long(tmp_path, "pacbio_hifi", "ccs"), "oatk"),
        (_illumina(tmp_path), "getorganelle"),
    ]
    for data, expected in cases:
        route = resolve_backend(data, organelle="plastid")
        assert route.selected_backend == expected
        # compatible, listed, not chosen
        assert "ovasm" in route.compatible_candidates
        assert ("ovasm", "assembly.not_selected") in {
            (item.backend_id, item.reason_code) for item in route.rejected_candidates
        }


@pytest.mark.parametrize(
    ("technology", "quality_state"),
    [("pacbio_hifi", "ccs"), ("ont", "raw"), ("pacbio_clr", "raw")],
)
def test_ovasm_is_selected_only_when_asked_for(
    tmp_path: Path, technology: str, quality_state: str
) -> None:
    data = _long(tmp_path, technology, quality_state)
    for organelle in ("mitochondrion", "plastid"):
        route = resolve_backend(data, organelle=organelle, method="ovasm")
        assert (route.selected_backend, route.rule_id) == ("ovasm", "explicit.backend")


def test_ovasm_refuses_profiles_it_was_not_run_on(tmp_path: Path) -> None:
    for data in (
        _long(tmp_path, "ont", "hq"),
        _long_plus_short(tmp_path, "ont", "hq", "single_end"),
    ):
        with pytest.raises(OrganelleInputError) as raised:
            resolve_backend(data, organelle="mitochondrion", method="ovasm")
        assert raised.value.code == "assembly.unsupported_data_profile"


# --- data contract and parameters --------------------------------------------------------------


def test_ovasm_data_contract_takes_one_library_and_at_most_a_seed(tmp_path: Path) -> None:
    seed = tmp_path / "seed.fa"
    seed.write_text(">s\nACGT\n")
    for data in (
        _long(tmp_path),
        _long(tmp_path, "ont", "raw"),
        _long(tmp_path, "pacbio_clr", "raw", seed_fasta=seed),
        _illumina(tmp_path),
    ):
        assert validate_ovasm_assembly_data(data) == _payload(data)
        validate_released_assembly_data(data)
    reasons = {
        "long_read_profile_not_supported": _long(tmp_path, "ont", "corrected"),
        "unsupported_library_combination": _two_short(tmp_path),
        "auxiliary_not_supported": _long(tmp_path, reference_fasta=seed),
    }
    for reason, data in reasons.items():
        with pytest.raises(OrganelleInputError) as raised:
            validate_ovasm_assembly_data(data)
        assert reason in {e["reason"] for e in raised.value.details["errors"]}


def test_effective_ovasm_parameters_record_the_read_type(tmp_path: Path) -> None:
    for data, read_type in (
        (_long(tmp_path), "hifi_only"),
        (_long(tmp_path, "ont", "raw"), "ont_only"),
        (_long(tmp_path, "pacbio_clr", "raw"), "clr_only"),
        (_illumina(tmp_path), "short_read"),
        (_illumina(tmp_path, "paired_end"), "short_read"),
    ):
        effective = effective_backend_parameters("ovasm", None, payload=_payload(data))
        assert effective == {
            "backend": "ovasm",
            "read_set": "whole_genome",
            "seed_source": "builtin",
            "also_discover": False,
            "sample": "sample",
            "read_type": read_type,
        }


def test_ovasm_options_that_would_not_be_used_are_refused(tmp_path: Path) -> None:
    seed = tmp_path / "seed.fa"
    seed.write_text(">s\nACGT\n")
    hifi, ont, seeded = (
        _long(tmp_path),
        _long(tmp_path, "ont", "raw"),
        _long(tmp_path, seed_fasta=seed),
    )
    cases = [
        (hifi, {"seed_source": "custom"}, "requires the seed_fasta"),
        (seeded, {}, "only with seed_source='custom'"),
        (seeded, {"seed_source": "custom", "read_set": "target_reads"}, "assembled as given"),
        (hifi, {"read_set": "target_reads", "also_discover": True}, "assembled as given"),
        (ont, {"seed_source": "discover"}, "needs PacBio HiFi"),
        (ont, {"also_discover": True}, "needs PacBio HiFi"),
        (hifi, {"seed_source": "discover", "also_discover": True}, "already uses discovery"),
        (hifi, {"sample": "a#b"}, "sample must be"),
        (hifi, {"sample": "a.b"}, "sample must be"),
    ]
    for data, options, match in cases:
        with pytest.raises(OrganelleInputError, match=match):
            effective_backend_parameters(
                "ovasm", OvasmParameters(**options), payload=_payload(data)
            )
    accepted = effective_backend_parameters(
        "ovasm", OvasmParameters(seed_source="custom", sample="Col-0"), payload=_payload(seeded)
    )
    assert accepted["seed_source"] == "custom" and accepted["sample"] == "Col-0"


# --- native environment -------------------------------------------------------------------------


def _fake_binary(path: Path, version: str = "ovasm 9.9.9") -> Path:
    path.write_text(f"#!/bin/sh\necho '{version}'\n")
    path.chmod(0o755)
    return path


@pytest.mark.skipif(sys.platform == "win32", reason="the fake binary is a shell script")
def test_native_environment_is_the_binary_and_its_identity(tmp_path: Path, monkeypatch) -> None:
    binary = _fake_binary(tmp_path / "ovasm")
    monkeypatch.setenv("ORG_VERSE_OVASM_BIN", str(binary))
    manager = NativeOvasmEnvironment()
    environment = manager.prepare(OVASM_ENVIRONMENT, policy="ensure")
    assert environment.carrier == "native" and environment.backend_id == "ovasm"
    assert environment.require_executable("ovasm") == binary.resolve()
    assert (environment.version, environment.software_version) == ("ovasm 9.9.9", "9.9.9")
    digest = manager.expected_environment_digest(OVASM_ENVIRONMENT)
    assert environment.digest == digest
    # a rebuilt binary is another environment: an earlier run is not reused for it
    _fake_binary(binary, "ovasm 9.9.10")
    assert manager.expected_environment_digest(OVASM_ENVIRONMENT) != digest
    _fake_binary(binary, "something else 1.0")
    with pytest.raises(OrganelleDependencyError, match="is not ovasm"):
        manager.prepare(OVASM_ENVIRONMENT, policy="ensure")


def test_missing_binary_is_an_unavailable_environment(monkeypatch) -> None:
    monkeypatch.setenv("ORG_VERSE_OVASM_BIN", "none")
    with pytest.raises(OrganelleDependencyError) as raised:
        NativeOvasmEnvironment().expected_environment_digest(OVASM_ENVIRONMENT)
    assert raised.value.code == "assembly.environment_unavailable"
    assert "ORG_VERSE_OVASM_BIN" in raised.value.message


# --- adapter ------------------------------------------------------------------------------------


def _context(tmp_path: Path, data: OrganelleData, **request) -> AdapterContext:
    request.setdefault("organelle", "mitochondrion")
    built = AssemblyRequest(data=data, method="ovasm", threads=3, **request)
    payload = _payload(data)
    binary = tmp_path / "bin" / "ovasm"
    binary.parent.mkdir(exist_ok=True)
    binary.write_text("#!/bin/sh\n")
    environment = PreparedEnvironment(
        backend_id="ovasm",
        carrier="native",
        platform="linux-64",
        digest="sha256:" + "c" * 64,
        prefix=binary.parent,
        executables=(PreparedExecutable(name="ovasm", path=binary),),
        version="ovasm 0.1.0",
        software_version="0.1.0",
    )
    effective = effective_backend_parameters(
        "ovasm", built.backend_parameters, payload=payload, organelle=built.organelle
    )
    return AdapterContext(
        request=built,
        payload=payload,
        route=AssemblyRoute(
            requested_method="ovasm",
            selected_backend="ovasm",
            profile=AssemblyProfile.PACBIO_HIFI,
            rule_id="explicit.backend",
            compatible_candidates=("ovasm",),
        ),
        environment=environment,
        profile=None,
        resources=PreparedBackendResources(),
        workspace=tmp_path / "workspace",
        input_artifacts=FrozenMap.from_items(dict(data.artifacts.items())),
        effective_backend_parameters=FrozenMap.from_json(effective),
    )


def test_ovasm_command_runs_the_shared_pipeline_in_a_child_python(tmp_path: Path) -> None:
    seed = tmp_path / "seed.fa"
    seed.write_text(">s\nACGT\n")
    data = _long(tmp_path, "ont", "raw", seed_fasta=seed)
    context = _context(
        tmp_path, data, backend_parameters=OvasmParameters(seed_source="custom", sample="NIP")
    )
    command = OvasmAdapter().build_command(context)
    reads_role = context.payload.long_libraries[0].reads_artifact
    seed_role = context.payload.auxiliary.seed_fasta_artifact
    assert command.stable_argv == (
        "ovasm-pipeline",
        "--reads",
        f"role://artifact/{reads_role}",
        "--organelle",
        "mitochondrion",
        "--read-type",
        "ont_only",
        "--read-set",
        "whole_genome",
        "--seed-source",
        "custom",
        "--sample",
        "NIP",
        "--threads",
        "3",
        "--seed",
        f"role://artifact/{seed_role}",
        "--out",
        "role://workspace/output",
    )
    resolved = command.resolved_argv
    assert resolved[:2] == (sys.executable, "-c")
    # the child imports the parent's own code tree, whatever PYTHONPATH the service strips
    assert Path(resolved[3]) == Path(ovasm_backend.__file__).resolve().parents[3]
    assert resolved[resolved.index("--ovasm") + 1] == str(tmp_path / "bin" / "ovasm")
    assert resolved[resolved.index("--reads") + 1] == str(tmp_path / "ont.fastq")
    assert resolved[resolved.index("--seed") + 1] == str(seed)
    assert resolved[resolved.index("--out") + 1] == str(context.workspace / "backend" / "ovasm")


def test_ovasm_preflight_refuses_what_it_cannot_honour(tmp_path: Path) -> None:
    data = _long(tmp_path)
    for request, match in (
        ({"taxon_group": "animal"}, "land plants only"),
        ({"backend_version": "1.2.3"}, "cannot select another"),
        (
            {"environment_hint": {"executable": str(tmp_path / "bin" / "ovasm")}},
            "ORG_VERSE_OVASM_BIN",
        ),
    ):
        with pytest.raises(OrganelleInputError, match=match):
            OvasmAdapter().build_command(_context(tmp_path, data, **request))


def _pipeline_outputs(root: Path, *, molecules: int = 1) -> None:
    (root / "recruit").mkdir(parents=True)
    (root / "assembly.gfa").write_text("S\tu0\tACGTACGTAC\tdp:f:30\nL\tu0\t+\tu0\t+\t4M\n")
    if molecules:
        (root / "molecules.fasta").write_text(">mol1 circular=true\nacgtac\n")
    for name in ("organelle.gfa", "organelle.json", "evidence.json", "linearization.json"):
        (root / name).write_text("{}\n")
    (root / "recruit" / "recruit.json").write_text("{}\n")
    summary = {
        "molecules": molecules,
        "k_tried": [{"k": 1001, "molecules": molecules}],
        "unitigs": 1,
        "unsolved_components": 0 if molecules else 1,
        "unbridged_anchors": [] if molecules else ["u0"],
    }
    (root / "summary.json").write_text(json.dumps(summary))


def test_ovasm_outputs_are_normalized_like_the_other_backends(tmp_path: Path) -> None:
    context = _context(tmp_path, _long(tmp_path))
    _pipeline_outputs(context.workspace / "backend" / "ovasm")
    adapter = OvasmAdapter()
    raw = adapter.collect_outputs(context)
    assert (raw.primary_sequence_role, raw.primary_graph_role) == (
        "assembly_fasta",
        "assembly_graph",
    )
    normalized = adapter.normalize(context, raw, context.workspace / "normalized")
    out = context.workspace / "normalized"
    assert (out / "assembly.fasta").read_text() == ">mol1 circular=true\nACGTAC\n"
    assert (out / "assembly.gfa").read_text().startswith("S\tu0\tACGTACGTAC")
    assert (normalized.record_count, normalized.total_bases) == (1, 6)
    # the evidence that says how far to trust the sequence is published with it
    assert {item.role: item.path.name for item in normalized.outputs} == {
        "assembly_fasta": "assembly.fasta",
        "assembly_graph": "assembly.gfa",
        "organelle_graph": "organelle.gfa",
        "graph_metadata": "organelle.json",
        "evidence": "evidence.json",
        "linearization": "linearization.json",
        "summary": "summary.json",
        "recruitment": "recruit.json",
    }


def test_a_graph_without_a_representative_sequence_fails_and_says_where_the_graph_is(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, _long(tmp_path))
    _pipeline_outputs(context.workspace / "backend" / "ovasm", molecules=0)
    with pytest.raises(OrganelleExecutionError) as raised:
        OvasmAdapter().collect_outputs(context)
    assert raised.value.code == "assembly.output_incomplete"
    assert "backend/ovasm in the run directory" in raised.value.message
    assert list(raised.value.details["unbridged_anchors"]) == ["u0"]
    assert [dict(t) for t in raised.value.details["k_tried"]] == [{"k": 1001, "molecules": 0}]


def test_backend_command_calls_the_desktop_pipeline_function(tmp_path, monkeypatch, capsys) -> None:
    from organelleverse.assembly import ovasm_runner

    seen = {}

    def run_ovasm(params, *, organelle, reads, output_dir, log):
        seen.update(params, organelle=organelle, reads=reads, output_dir=output_dir)
        log("progress")
        if params["sample"] == "bad":
            raise RuntimeError("OVASM recruited 4 reads (4021 bp)")
        return {"summary_text": "OVASM: 1 representative molecules", "output_files": {}}

    monkeypatch.setattr(ovasm_runner, "run_ovasm", run_ovasm)
    # the command points the pipeline at its binary through the environment; restored after
    monkeypatch.setenv("ORG_VERSE_OVASM_BIN", "set-by-the-test")
    argv = [
        "--ovasm", "/opt/ovasm", "--reads", "r.fastq", "--out", "out", "--organelle", "plastid",
        "--read-type", "clr_only", "--read-set", "whole_genome", "--seed-source", "custom",
        "--seed", "seed.fa", "--threads", "5",
    ]  # fmt: skip
    assert ovasm_backend.main([*argv, "--sample", "S1"]) == 0
    assert seen == {
        "strategy": "clr_only",
        "read_set": "whole_genome",
        "seed_source": "custom",
        "seed": "seed.fa",
        "also_discover": False,
        "sample": "S1",
        "threads": 5,
        "organelle": "plastid",
        "reads": Path("r.fastq"),
        "output_dir": Path("out"),
    }
    assert os.environ["ORG_VERSE_OVASM_BIN"] == "/opt/ovasm"
    captured = capsys.readouterr()
    assert captured.out.strip() == "OVASM: 1 representative molecules"
    assert "progress" in captured.err
    # a failure ends stderr with its cause and a non-zero exit, which the service records
    assert ovasm_backend.main([*argv, "--sample", "bad"]) == 1
    assert (
        capsys.readouterr()
        .err.strip()
        .endswith("ovasm pipeline failed: RuntimeError: OVASM recruited 4 reads (4021 bp)")
    )


# --- the real binary ----------------------------------------------------------------------------


@pytest.mark.skipif(
    _ovasm.resolve_ovasm() is None, reason="requires the native OVASM release binary"
)
def test_assemble_with_method_ovasm_runs_the_binary_and_publishes_a_result(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    sequence = "".join(random.Random(7).choices("ACGT", k=8000))
    reads = tmp_path / "plastid_reads.fastq"
    reads.write_text(
        "".join(
            f"@r{rep}_{start}\n{(sequence * 2)[start : start + 6000]}\n+\n{'I' * 6000}\n"
            for rep in range(5)
            for start in range(0, len(sequence), 250)
        )
    )
    data = read_reads(
        long_libraries=(
            LongReadLibrary(technology="pacbio_hifi", quality_state="ccs", reads=reads),
        )
    )
    parameters = OvasmParameters(read_set="target_reads", sample="demo")
    result = assemble(
        data, organelle="plastid", method="ovasm", backend_parameters=parameters, threads=2
    )
    assert result.status == "ok", result.model_dump_json()[:2000]
    provenance = result.provenance
    assert (provenance.requested_backend, provenance.actual_backend) == ("ovasm", "ovasm")
    assert provenance.software_versions["ovasm"]
    assert provenance.argv[0] == "ovasm-pipeline"

    # the same request again is the same published run, not a second assembly
    again = assemble(
        data, organelle="plastid", method="ovasm", backend_parameters=parameters, threads=2
    )
    assert again.provenance.run_manifest_id == provenance.run_manifest_id
    assert again.provenance.started_at == provenance.started_at

    published = write(result, output=tmp_path / "published")
    out = tmp_path / "published"
    candidate = "".join(
        line
        for line in (out / "normalized/assembly.fasta").read_text().splitlines()
        if not line.startswith(">")
    )
    reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    assert len(candidate) == 8000 and (candidate in sequence * 2 or candidate in reverse * 2)
    assert (out / "normalized/assembly.gfa").read_text().startswith(("H\t", "S\t"))
    for name in ("evidence.json", "linearization.json", "organelle.gfa", "summary.json"):
        assert (out / "normalized" / name).is_file()
    summary = json.loads((out / "normalized/summary.json").read_text())
    assert summary["backend"] == "ovasm" and summary["read_set"] == "target_reads"
    assert summary["circular"] == [True]
    assert "demo" in (out / "normalized/organelle.gfa").read_text()
    manifest = AssemblyRunManifest.model_validate_json(
        (out / "assembly_run_record.json").read_bytes()
    )
    assert manifest.selected_backend == "ovasm" and manifest.environment.carrier == "native"
    assert {item.role for item in manifest.outputs} >= {"assembly_fasta", "assembly_graph"}
    assert published.status == "ok"


# --- paired-end Illumina and Illumina + noisy long reads ----------------------------------------


def test_ovasm_takes_paired_end_and_hybrid_libraries(tmp_path: Path) -> None:
    for data in (
        _illumina(tmp_path, "paired_end"),
        _long_plus_short(tmp_path, "ont", "raw", "paired_end"),
        _long_plus_short(tmp_path, "pacbio_clr", "raw", "single_end"),
    ):
        assert validate_ovasm_assembly_data(data) == _payload(data)
        validate_released_assembly_data(data)
    reasons = {
        "hybrid_needs_noisy_long_reads": _long_plus_short(tmp_path, "pacbio_hifi", "ccs", "single_end"),
        "long_read_profile_not_supported": _long_plus_short(tmp_path, "ont", "hq", "single_end"),
        "unsupported_library_combination": _two_short(tmp_path),
    }
    for reason, data in reasons.items():
        with pytest.raises(OrganelleInputError) as raised:
            validate_ovasm_assembly_data(data)
        assert reason in {e["reason"] for e in raised.value.details["errors"]}


def test_hybrid_reads_are_a_profile_that_only_ovasm_takes(tmp_path: Path) -> None:
    ont = _long_plus_short(tmp_path, "ont", "raw")
    clr = _long_plus_short(tmp_path, "pacbio_clr", "raw", "single_end")
    assert classify_profile(ont) is AssemblyProfile.ONT_RAW_ILLUMINA
    assert classify_profile(clr) is AssemblyProfile.PACBIO_CLR_RAW_ILLUMINA
    for data in (ont, clr):
        for organelle in ("mitochondrion", "plastid"):
            assert compatible_backends(data, organelle) == ("ovasm",)
            route = resolve_backend(data, organelle=organelle, method="ovasm")
            assert (route.selected_backend, route.rule_id) == ("ovasm", "explicit.backend")
    # other noisy-read profiles still cannot be combined with short reads
    with pytest.raises(OrganelleInputError, match="cannot be combined with short reads"):
        classify_profile(_long_plus_short(tmp_path, "ont", "hq", "single_end"))
    # paired-end Illumina: ovasm joins the assemblers that already took it
    assert "ovasm" in compatible_backends(_illumina(tmp_path, "paired_end"), "mitochondrion")


def test_effective_parameters_name_the_paired_and_hybrid_read_types(tmp_path: Path) -> None:
    for data, read_type in (
        (_illumina(tmp_path, "paired_end"), "short_read"),
        (_long_plus_short(tmp_path, "ont", "raw"), "hybrid"),
        (_long_plus_short(tmp_path, "pacbio_clr", "raw", "single_end"), "hybrid"),
    ):
        effective = effective_backend_parameters("ovasm", None, payload=_payload(data))
        assert effective["read_type"] == read_type


def _resolved(command, flag: str) -> str:
    return command.resolved_argv[command.resolved_argv.index(flag) + 1]


def _stable(command, flag: str) -> str:
    return command.stable_argv[command.stable_argv.index(flag) + 1]


def test_paired_end_command_passes_both_files(tmp_path: Path) -> None:
    data = _illumina(tmp_path, "paired_end")
    context = _context(tmp_path, data)
    command = OvasmAdapter().build_command(context)
    library = context.payload.short_libraries[0]
    assert _resolved(command, "--reads") == str(tmp_path / "R1.fastq")
    assert _resolved(command, "--mate") == str(tmp_path / "R2.fastq")
    assert _stable(command, "--reads") == f"role://artifact/{library.read1_artifact}"
    assert _stable(command, "--mate") == f"role://artifact/{library.read2_artifact}"
    assert _stable(command, "--read-type") == "short_read"
    assert "--short-reads" not in command.stable_argv
    # single-end reads have no mate
    single = OvasmAdapter().build_command(_context(tmp_path, _illumina(tmp_path)))
    assert "--mate" not in single.stable_argv and "--mate" not in single.resolved_argv


def test_hybrid_command_passes_the_long_reads_and_every_illumina_file(tmp_path: Path) -> None:
    data = _long_plus_short(tmp_path, "ont", "raw", "paired_end")
    context = _context(tmp_path, data)
    command = OvasmAdapter().build_command(context)
    long_role = context.payload.long_libraries[0].reads_artifact
    short = context.payload.short_libraries[0]
    assert _stable(command, "--read-type") == "hybrid"
    assert _stable(command, "--reads") == f"role://artifact/{long_role}"
    assert _stable(command, "--short-reads") == f"role://artifact/{short.read1_artifact}"
    assert _stable(command, "--short-reads-2") == f"role://artifact/{short.read2_artifact}"
    assert _resolved(command, "--reads") == str(tmp_path / "ont_long.fastq")
    assert _resolved(command, "--short-reads") == str(tmp_path / "H1.fastq")
    assert _resolved(command, "--short-reads-2") == str(tmp_path / "H2.fastq")
    assert "--mate" not in command.stable_argv
    single = OvasmAdapter().build_command(
        _context(tmp_path, _long_plus_short(tmp_path, "ont", "raw", "single_end"))
    )
    assert "--short-reads-2" not in single.stable_argv


def test_backend_command_hands_the_extra_files_to_the_pipeline(tmp_path, monkeypatch) -> None:
    from organelleverse.assembly import ovasm_runner

    seen: dict[str, object] = {}

    def run_ovasm(params, *, organelle, reads, output_dir, log, mate=None):
        seen.update(params, reads=reads, mate=mate)
        return {"summary_text": "ok", "output_files": {}}

    monkeypatch.setattr(ovasm_runner, "run_ovasm", run_ovasm)
    monkeypatch.setenv("ORG_VERSE_OVASM_BIN", "set-by-the-test")
    base = [
        "--ovasm", "/opt/ovasm", "--out", "out", "--organelle", "plastid", "--read-set",
        "target_reads", "--seed-source", "builtin", "--sample", "S", "--threads", "2",
    ]  # fmt: skip
    argv = [*base, "--read-type", "short_read", "--reads", "a_1.fq", "--mate", "a_2.fq"]
    assert ovasm_backend.main(argv) == 0
    assert (seen["reads"], seen["mate"]) == (Path("a_1.fq"), Path("a_2.fq"))
    assert "short_reads" not in seen

    seen.clear()
    argv = [
        *base, "--read-type", "hybrid", "--reads", "long.fq", "--short-reads", "s_1.fq",
        "--short-reads-2", "s_2.fq",
    ]  # fmt: skip
    assert ovasm_backend.main(argv) == 0
    assert (seen["strategy"], seen["short_reads"], seen["short_reads_2"]) == (
        "hybrid", "s_1.fq", "s_2.fq",
    )  # fmt: skip
    assert seen["mate"] is None


class _Stop(Exception):
    """Raised by a recorder to end run_ovasm once the call under test has been seen."""


def test_runner_gives_both_files_of_a_pair_to_recruitment_and_assembly(
    tmp_path: Path, monkeypatch
) -> None:
    from organelleverse.assembly import ovasm_runner

    r1, r2 = _fastq(tmp_path / "a_1.fastq"), _fastq(tmp_path / "a_2.fastq")
    seen: dict[str, object] = {}

    def recorder(name):
        def record(reads, *args, **kwargs):
            seen[name] = [Path(p) for p in (reads if name != "evidence" else args[0])]
            raise _Stop

        return record

    monkeypatch.setattr(ovasm_runner._ovasm, "resolve_ovasm", lambda: "ovasm")
    monkeypatch.setattr(ovasm_runner._ovasm, "run_recruit", recorder("recruit"))
    monkeypatch.setattr(ovasm_runner._ovasm, "run_assemble", recorder("assemble"))

    # whole-genome reads: recruitment reads the pair
    params = {"strategy": "short_read", "read_set": "whole_genome", "seed_source": "builtin"}
    with pytest.raises(_Stop):
        ovasm_runner.run_ovasm(
            params, organelle="plastid", reads=r1, mate=r2, output_dir=tmp_path / "w", log=print
        )
    assert seen["recruit"] == [r1, r2]

    # reads that are the organelle's already: assembly reads the pair
    seen.clear()
    params = {"strategy": "short_read", "read_set": "target_reads", "seed_source": "builtin"}
    with pytest.raises(_Stop):
        ovasm_runner.run_ovasm(
            params, organelle="plastid", reads=r1, mate=r2, output_dir=tmp_path / "t", log=print
        )
    assert seen["assemble"] == [r1, r2]

    # without a mate nothing changes
    seen.clear()
    with pytest.raises(_Stop):
        ovasm_runner.run_ovasm(
            params, organelle="plastid", reads=r1, output_dir=tmp_path / "s", log=print
        )
    assert seen["assemble"] == [r1]


def test_runner_refuses_a_mate_for_other_read_types(tmp_path: Path, monkeypatch) -> None:
    from organelleverse.assembly import ovasm_runner

    reads, mate = _fastq(tmp_path / "x.fastq"), _fastq(tmp_path / "y.fastq")
    monkeypatch.setattr(ovasm_runner._ovasm, "resolve_ovasm", lambda: "ovasm")
    with pytest.raises(ValueError, match="second read file only with the Illumina read type"):
        ovasm_runner.run_ovasm(
            {"strategy": "hifi_only", "read_set": "target_reads", "seed_source": "builtin"},
            organelle="plastid",
            reads=reads,
            mate=mate,
            output_dir=tmp_path / "o",
            log=print,
        )


@pytest.mark.skipif(
    _ovasm.resolve_ovasm() is None, reason="requires the native OVASM release binary"
)
def test_assemble_with_method_ovasm_takes_a_paired_end_library(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    rng = random.Random(11)
    sequence = "".join(rng.choices("ACGT", k=8000))
    doubled = sequence + sequence[:1000]
    complement = str.maketrans("ACGT", "TGCA")
    r1, r2 = tmp_path / "pair_1.fastq", tmp_path / "pair_2.fastq"
    with r1.open("w") as first, r2.open("w") as second:
        for i in range(4000):
            start = rng.randrange(len(sequence))
            fragment = doubled[start : start + 350]
            first.write(f"@p{i}/1\n{fragment[:150]}\n+\n{'I' * 150}\n")
            mate = fragment[-150:].translate(complement)[::-1]
            second.write(f"@p{i}/2\n{mate}\n+\n{'I' * 150}\n")
    data = read_reads(
        short_libraries=(
            ShortReadLibrary(
                technology="illumina",
                layout="paired_end",
                read1=r1,
                read2=r2,
                read_length=150,
                insert_size=350,
            ),
        )
    )
    parameters = OvasmParameters(read_set="target_reads", sample="pe")
    result = assemble(
        data, organelle="plastid", method="ovasm", backend_parameters=parameters, threads=2
    )
    assert result.status == "ok", result.model_dump_json()[:2000]
    assert result.provenance.argv[0] == "ovasm-pipeline"
    assert "--mate" in result.provenance.argv
    out = tmp_path / "published"
    write(result, output=out)
    candidate = "".join(
        line
        for line in (out / "normalized/assembly.fasta").read_text().splitlines()
        if not line.startswith(">")
    )
    reverse = sequence.translate(complement)[::-1]
    assert len(candidate) == 8000 and (candidate in sequence * 2 or candidate in reverse * 2)
    summary = json.loads((out / "normalized/summary.json").read_text())
    assert summary["read_type"] == "short_read" and summary["circular"] == [True]
