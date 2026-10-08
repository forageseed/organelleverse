from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest

from organelleverse.assembly.backends.base import (
    AdapterContext,
    BackendOutput,
    NormalizedAssemblyOutputs,
    PreparedBackendResources,
    RawAssemblyOutputs,
)
from organelleverse.assembly.backends.oatk import (
    OatkAdapter,
    OatkResourceProvider,
    _profile_score,
)
from organelleverse.assembly.backends.registry import BACKENDS
from organelleverse.assembly.backends.spec import AssemblyProfile
from organelleverse.assembly.contracts import (
    AssemblyInputPayload,
    AssemblyRequest,
    OatkParameters,
)
from organelleverse.assembly.environment_specs import OATK_ENVIRONMENT, AssemblyEnvironmentSpec
from organelleverse.assembly.environments import (
    EnvironmentManager,
    PreparedEnvironment,
    PreparedExecutable,
    PreparedProfile,
)
from organelleverse.assembly.manifests import AssemblyComponentIdentity
from organelleverse.assembly.routing import AssemblyRoute
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.frozen import FrozenMap


def _read_artifact(
    tmp_path: Path, name: str, content: bytes = b"@read\nACGT\n+\nIIII\n"
) -> ArtifactRef:
    path = tmp_path / "reads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactRef.from_path(
        path, kind="long_read", format="fastq", media_type="application/x-fastq"
    )


def _hifi_data(tmp_path: Path, *, reads: tuple[ArtifactRef, ...]) -> OrganelleData:
    artifacts: dict[str, ArtifactRef] = {}
    long_libraries: list[dict[str, object]] = []
    for index, read in enumerate(reads):
        role = f"long_{index}_reads"
        artifacts[role] = read
        long_libraries.append(
            {"technology": "pacbio_hifi", "quality_state": "ccs", "reads_artifact": role}
        )
    return OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": artifacts,
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": long_libraries,
            },
        }
    )


def _profile(tmp_path: Path, organelle: str) -> ArtifactRef:
    name = "embryophyta_mito.fam" if organelle == "mitochondrion" else "embryophyta_pltd.fam"
    path = tmp_path / "profiles" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"profile-content")
    return ArtifactRef.from_path(path, kind="hmm_profile", format="fam", media_type="text/plain")


def _environment(tmp_path: Path) -> PreparedEnvironment:
    prefix = tmp_path / "env" / "oatk"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    executables: list[PreparedExecutable] = []
    for name in ("oatk", "nhmmscan", "hmmpress"):
        path = prefix / "bin" / name
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
        executables.append(PreparedExecutable(name=name, path=path))
    return PreparedEnvironment(
        backend_id="oatk",
        carrier="conda",
        platform="linux-64",
        digest="sha256:" + "a" * 64,
        prefix=prefix,
        executables=tuple(executables),
        version="oatk 1.0",
    )


def _context(
    tmp_path: Path,
    *,
    organelle: str,
    minimum_kmer_coverage: int | None = None,
    reads: tuple[ArtifactRef, ...] | None = None,
    threads: int = 4,
) -> AdapterContext:
    if reads is None:
        reads = (_read_artifact(tmp_path, "single.fastq"),)
    data = _hifi_data(tmp_path, reads=reads)
    request = AssemblyRequest(
        data=data,
        organelle=organelle,  # type: ignore[arg-type]
        threads=threads,
        backend_parameters=(
            OatkParameters(minimum_kmer_coverage=minimum_kmer_coverage)
            if minimum_kmer_coverage is not None
            else None
        ),
    )
    payload = AssemblyInputPayload.model_validate(dict(data.payload))
    profile_artifact = _profile(tmp_path, organelle)
    route = AssemblyRoute(
        requested_method="auto",
        selected_backend="oatk",
        profile=AssemblyProfile.PACBIO_HIFI,
        rule_id="auto.hifi_oatk",
        compatible_candidates=("oatk",),
    )
    environment = _environment(tmp_path)
    prepared_profile = PreparedProfile(
        source="managed",
        target=organelle,  # type: ignore[arg-type]
        path=Path(profile_artifact.uri),
        artifact=profile_artifact,
        component=AssemblyComponentIdentity(
            category="profile",
            name=profile_artifact.uri.rsplit("/", 1)[-1],
            sha256=profile_artifact.sha256,
        ),
    )
    input_artifacts = {f"long_{index}_reads": read for index, read in enumerate(reads)}
    effective = {
        "backend": "oatk",
        "kmer_size": 1001,
        "minimum_kmer_coverage": minimum_kmer_coverage if minimum_kmer_coverage else 30,
    }
    resources = PreparedBackendResources(
        artifacts=FrozenMap.from_items({"hmm_profiles": profile_artifact})
    )
    return AdapterContext(
        request=request,
        payload=payload,
        route=route,
        environment=environment,
        profile=prepared_profile,
        resources=resources,
        workspace=tmp_path / "workspace",
        input_artifacts=FrozenMap.from_items(input_artifacts),
        effective_backend_parameters=FrozenMap.from_json(effective),
    )


def test_oatk_mito_command_has_exact_stable_and_resolved_roles(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion", minimum_kmer_coverage=41)

    command = OatkAdapter().build_command(context)

    assert command.stable_argv == (
        "oatk",
        "-c",
        "41",
        "-t",
        "4",
        "--nhmmscan",
        "nhmmscan",
        "-m",
        "role://artifact/hmm_profiles",
        "-o",
        "role://workspace/assembly",
        "role://artifact/long_0_reads",
    )
    assert command.resolved_argv == (
        str(context.environment.require_executable("oatk")),
        "-c",
        "41",
        "-t",
        "4",
        "--nhmmscan",
        str(context.environment.require_executable("nhmmscan")),
        "-m",
        str(context.resources.artifacts["hmm_profiles"].uri),
        "-o",
        str(context.workspace / "backend" / "oatk" / "oatk"),
        str(context.input_artifacts["long_0_reads"].uri),
    )


def test_oatk_plastid_command_uses_p_flag(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="plastid", minimum_kmer_coverage=30)
    command = OatkAdapter().build_command(context)
    assert command.stable_argv[7] == "-p"
    assert "-m" not in command.stable_argv
    assert command.resolved_argv[7] == "-p"


def test_oatk_command_orders_multiple_hifi_libraries(tmp_path: Path) -> None:
    first = _read_artifact(tmp_path, "first.fastq")
    second = _read_artifact(tmp_path, "second.fastq")
    context = _context(tmp_path, organelle="mitochondrion", reads=(first, second))
    command = OatkAdapter().build_command(context)
    assert command.stable_argv[-2:] == (
        "role://artifact/long_0_reads",
        "role://artifact/long_1_reads",
    )
    assert command.resolved_argv[-2:] == (str(first.uri), str(second.uri))


def test_oatk_command_keeps_shell_metacharacter_paths_as_single_argv(tmp_path: Path) -> None:
    weird = tmp_path / "reads" / "weird $(name)`.fastq"
    weird.parent.mkdir(parents=True, exist_ok=True)
    weird.write_bytes(b"@r\nACGT\n+\nIIII\n")
    artifact = ArtifactRef.from_path(
        weird, kind="long_read", format="fastq", media_type="application/x-fastq"
    )
    context = _context(tmp_path, organelle="mitochondrion", reads=(artifact,))
    command = OatkAdapter().build_command(context)
    assert command.resolved_argv[-1] == str(weird)


def test_oatk_command_uses_default_coverage_when_none_provided(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    command = OatkAdapter().build_command(context)
    assert command.stable_argv[2] == "30"


def test_preflight_rejects_ont_library(tmp_path: Path) -> None:
    bad_payload = AssemblyInputPayload.model_validate(
        {
            "contract_version": "organelleverse.assembly-input.v1",
            "long_libraries": [
                {"technology": "ont", "quality_state": "raw", "reads_artifact": "long_0_reads"}
            ],
        }
    )
    context = _context(tmp_path, organelle="mitochondrion")
    bad_context = context.model_copy(update={"payload": bad_payload})
    with pytest.raises(OrganelleExecutionError) as raised:
        OatkAdapter().preflight(bad_context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_clr_library(tmp_path: Path) -> None:
    bad_payload = AssemblyInputPayload.model_validate(
        {
            "contract_version": "organelleverse.assembly-input.v1",
            "long_libraries": [
                {
                    "technology": "pacbio_clr",
                    "quality_state": "raw",
                    "reads_artifact": "long_0_reads",
                }
            ],
        }
    )
    context = _context(tmp_path, organelle="mitochondrion")
    bad_context = context.model_copy(update={"payload": bad_payload})
    with pytest.raises(OrganelleExecutionError):
        OatkAdapter().preflight(bad_context)


def test_preflight_rejects_contig_only_input(tmp_path: Path) -> None:
    bad_payload = AssemblyInputPayload.model_validate(
        {
            "contract_version": "organelleverse.assembly-input.v1",
            "contig_inputs": [{"fasta_artifact": "contigs"}],
        }
    )
    context = _context(tmp_path, organelle="mitochondrion")
    bad_context = context.model_copy(update={"payload": bad_payload})
    with pytest.raises(OrganelleExecutionError):
        OatkAdapter().preflight(bad_context)


def test_preflight_rejects_mixed_short_and_hifi_input(tmp_path: Path) -> None:
    bad_payload = AssemblyInputPayload.model_validate(
        {
            "contract_version": "organelleverse.assembly-input.v1",
            "short_libraries": [
                {
                    "technology": "illumina",
                    "layout": "single_end",
                    "read1_artifact": "short_reads",
                    "read_length": 150,
                }
            ],
            "long_libraries": [
                {
                    "technology": "pacbio_hifi",
                    "quality_state": "ccs",
                    "reads_artifact": "long_0_reads",
                }
            ],
        }
    )
    context = _context(tmp_path, organelle="mitochondrion")
    bad_context = context.model_copy(update={"payload": bad_payload})

    with pytest.raises(OrganelleExecutionError) as raised:
        OatkAdapter().preflight(bad_context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_unsupported_auxiliary_input(tmp_path: Path) -> None:
    bad_payload = AssemblyInputPayload.model_validate(
        {
            "contract_version": "organelleverse.assembly-input.v1",
            "long_libraries": [
                {
                    "technology": "pacbio_hifi",
                    "quality_state": "ccs",
                    "reads_artifact": "long_0_reads",
                }
            ],
            "auxiliary": {"reference_fasta_artifact": "reference"},
        }
    )
    context = _context(tmp_path, organelle="mitochondrion")
    bad_context = context.model_copy(update={"payload": bad_payload})

    with pytest.raises(OrganelleExecutionError) as raised:
        OatkAdapter().preflight(bad_context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_non_oatk_selected_backend(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    bad_route = context.route.model_copy(update={"selected_backend": "himt"})
    bad_context = context.model_copy(update={"route": bad_route})
    with pytest.raises(OrganelleExecutionError) as raised:
        OatkAdapter().preflight(bad_context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_mismatched_profile_target(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    assert context.profile is not None
    bad_profile = context.profile.model_copy(update={"target": "plastid"})
    bad_context = context.model_copy(update={"profile": bad_profile})
    with pytest.raises(OrganelleExecutionError) as raised:
        OatkAdapter().preflight(bad_context)
    assert raised.value.code == "assembly.unsupported_data_profile"


def test_preflight_rejects_environment_for_another_backend(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    bad_env = context.environment.model_copy(update={"backend_id": "himt"})
    bad_context = context.model_copy(update={"environment": bad_env})
    with pytest.raises(OrganelleExecutionError):
        OatkAdapter().preflight(bad_context)


def test_preflight_accepts_a_valid_hifi_request(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion", minimum_kmer_coverage=41)
    OatkAdapter().preflight(context)


def _write_oatk_outputs(context: AdapterContext, *, target: str) -> None:
    suffix = "mito" if target == "mitochondrion" else "pltd"
    prefix = context.workspace / "backend" / "oatk"
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / f"oatk.{suffix}.ctg.fasta").write_text(">ctg1\nACGT\n")
    (prefix / f"oatk.{suffix}.gfa").write_text("H\tVN:Z:1.0\nS\ts1\tACGT\n")
    (prefix / f"oatk.{suffix}.ctg.bed").write_text("ctg1\t0\t3\tgeneA\t.\t+\n")
    (prefix / f"oatk.annot_{suffix}.txt").write_text("annotation\n")
    (prefix / "oatk.utg.final.gfa").write_text("H\tVN:Z:1.0\nS\ts1\tACGT\n")


def test_mito_collection_returns_exact_official_files(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    _write_oatk_outputs(context, target="mitochondrion")
    raw = OatkAdapter().collect_outputs(context)
    prefix = context.workspace / "backend" / "oatk"
    assert raw.primary_sequence_role == "assembly_fasta"
    assert raw.primary_graph_role == "assembly_graph"
    assert tuple(item.role for item in raw.outputs) == (
        "assembly_fasta",
        "assembly_graph",
        "gene_annotations",
        "backend_annotation",
        "backend_full_graph",
    )
    assert raw.require("assembly_fasta").path == prefix / "oatk.mito.ctg.fasta"
    assert raw.require("assembly_graph").path == prefix / "oatk.mito.gfa"
    assert raw.require("gene_annotations").path == prefix / "oatk.mito.ctg.bed"
    assert raw.require("backend_annotation").path == prefix / "oatk.annot_mito.txt"
    assert raw.require("backend_full_graph").path == prefix / "oatk.utg.final.gfa"


def test_mito_collection_requires_every_target_file(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    _write_oatk_outputs(context, target="plastid")
    with pytest.raises(OrganelleExecutionError) as raised:
        OatkAdapter().collect_outputs(context)
    assert raised.value.code == "assembly.output_incomplete"


def test_plastid_collection_uses_pltd_suffix(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="plastid")
    _write_oatk_outputs(context, target="plastid")
    raw = OatkAdapter().collect_outputs(context)
    prefix = context.workspace / "backend" / "oatk"
    assert raw.require("assembly_fasta").path == prefix / "oatk.pltd.ctg.fasta"
    assert raw.require("gene_annotations").path == prefix / "oatk.pltd.ctg.bed"
    assert raw.require("backend_annotation").path == prefix / "oatk.annot_pltd.txt"


def test_collection_reports_sorted_missing_roles(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    prefix = context.workspace / "backend" / "oatk"
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "oatk.mito.ctg.fasta").write_text(">ctg1\nACGT\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        OatkAdapter().collect_outputs(context)
    details = raised.value.as_dict()["details"]
    assert details["missing_roles"] == sorted(
        {"assembly_graph", "gene_annotations", "backend_annotation", "backend_full_graph"}
    )


def test_normalize_round_trips_and_reports_metrics(tmp_path: Path) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    _write_oatk_outputs(context, target="mitochondrion")
    raw = OatkAdapter().collect_outputs(context)
    output_dir = tmp_path / "normalized"
    normalized = OatkAdapter().normalize(context, raw, output_dir)
    assert isinstance(normalized, NormalizedAssemblyOutputs)
    assert normalized.record_count == 1
    assert normalized.total_bases == 4
    assert "genea" in normalized.gene_markers
    assert normalized.require("assembly_fasta").path.name == "assembly.fasta"
    assert normalized.primary_sequence.path == normalized.require("assembly_fasta").path
    assert normalized.alternate_sequence_roles == ()
    assert (output_dir / "assembly.fasta").is_file()
    assert (output_dir / "assembly.gfa").is_file()
    assert (output_dir / "gene_annotations.bed").is_file()
    assert (output_dir / "backend_annotation.txt").is_file()
    assert (output_dir / "backend_full_graph.gfa").is_file()


def test_backend_output_models_reject_duplicate_roles(tmp_path: Path) -> None:
    path = tmp_path / "x.fasta"
    path.write_text(">ctg1\nACGT\n")
    output = BackendOutput(role="assembly_fasta", path=path, format="fasta")
    with pytest.raises(ValueError):
        RawAssemblyOutputs(
            outputs=(output, output),
            primary_sequence_role="assembly_fasta",
        )


def test_backend_output_models_require_primary_sequence_role(tmp_path: Path) -> None:
    path = tmp_path / "x.fasta"
    path.write_text(">ctg1\nACGT\n")
    output = BackendOutput(role="assembly_fasta", path=path, format="fasta")
    with pytest.raises(ValueError):
        RawAssemblyOutputs(
            outputs=(output,),
            primary_sequence_role="missing_role",
        )


def test_backend_output_models_reject_alternate_role_not_in_outputs(tmp_path: Path) -> None:
    path = tmp_path / "x.fasta"
    path.write_text(">ctg1\nACGT\n")
    output = BackendOutput(role="assembly_fasta", path=path, format="fasta")
    with pytest.raises(ValueError):
        NormalizedAssemblyOutputs(
            outputs=(output,),
            primary_sequence_role="assembly_fasta",
            alternate_sequence_roles=("missing_role",),
            record_count=1,
            total_bases=4,
        )


def test_oatk_adapter_backend_id_matches_registry() -> None:
    assert OatkAdapter().backend_id == "oatk"
    assert "oatk" in BACKENDS.ids()


def _request_with_hmm_profile(
    tmp_path: Path, *, user_override: bool = False
) -> tuple[AssemblyRequest, ArtifactRef]:
    reads = (_read_artifact(tmp_path, "reads.fastq"),)
    data = _hifi_data(tmp_path, reads=reads)
    profile = _profile(tmp_path, "mitochondrion")
    artifacts = dict(data.artifacts.items())
    payload = cast(dict[str, object], data.model_dump(mode="json")["payload"])
    if user_override:
        artifacts["hmm_profiles"] = profile
        payload["auxiliary"] = {"hmm_profiles_artifact": "hmm_profiles"}
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": artifacts,
            "payload": payload,
        }
    )
    request = AssemblyRequest(
        data=data,
        organelle="mitochondrion",  # type: ignore[arg-type]
        threads=4,
    )
    return request, profile


class _FakeManager(EnvironmentManager):
    def __init__(self, profile_artifact: ArtifactRef) -> None:
        super().__init__()
        self.artifact = profile_artifact
        self.environment_digest = "sha256:" + "e" * 64
        self.managed_component = AssemblyComponentIdentity(
            category="profile",
            name="embryophyta_mito.fam",
            sha256="0" * 64,
            locator="oatkdb:v20230921#embryophyta_mito.fam-hmm-bundle",
        )
        self.user_component = AssemblyComponentIdentity(
            category="profile",
            name="user-mito-hmm-profile",
            sha256="1" * 64,
            locator="user-supplied",
        )
        self.last_override: ArtifactRef | None = None

    def expected_environment_digest(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        platform: str | None = None,
    ) -> str:
        return self.environment_digest

    def expected_profile_identity(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        organelle: Literal["mitochondrion", "plastid"],
        override: ArtifactRef | None = None,
    ) -> AssemblyComponentIdentity:
        self.last_override = override
        return self.user_component if override is not None else self.managed_component

    def prepare_profile(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        organelle: Literal["mitochondrion", "plastid"],
        policy: Literal["ensure", "require"],
        override: ArtifactRef | None = None,
        environment: PreparedEnvironment | None = None,
    ) -> PreparedProfile:
        self.last_override = override
        component = self.user_component if override is not None else self.managed_component
        source: Literal["managed", "user_supplied"] = (
            "user_supplied" if override is not None else "managed"
        )
        return PreparedProfile(
            source=source,
            target=organelle,
            path=Path(self.artifact.uri),
            artifact=self.artifact,
            component=component,
        )


def test_oatk_resource_provider_expected_returns_managed_profile_identity(
    tmp_path: Path,
) -> None:
    request, profile = _request_with_hmm_profile(tmp_path)
    provider = OatkResourceProvider()
    manager = _FakeManager(profile)
    payload = AssemblyInputPayload.model_validate(dict(request.data.payload))

    expected = provider.expected(manager, OATK_ENVIRONMENT, request, payload)

    assert manager.last_override is None
    assert expected.components == (manager.managed_component,)
    assert expected.database_hashes == {"oatkdb": manager.managed_component.sha256}
    assert expected.model_hashes == {}


def test_oatk_resource_provider_expected_returns_user_profile_identity(
    tmp_path: Path,
) -> None:
    request, profile = _request_with_hmm_profile(tmp_path, user_override=True)
    provider = OatkResourceProvider()
    manager = _FakeManager(profile)
    payload = AssemblyInputPayload.model_validate(dict(request.data.payload))

    expected = provider.expected(manager, OATK_ENVIRONMENT, request, payload)

    assert manager.last_override == profile
    assert expected.components == (manager.user_component,)
    assert expected.model_hashes == {"hmm_profile": manager.user_component.sha256}
    assert expected.database_hashes == {}


def test_oatk_resource_provider_prepare_returns_hmm_profiles_artifact(
    tmp_path: Path,
) -> None:
    request, profile = _request_with_hmm_profile(tmp_path)
    provider = OatkResourceProvider()
    manager = _FakeManager(profile)
    payload = AssemblyInputPayload.model_validate(dict(request.data.payload))
    environment = _environment(tmp_path)

    prepared = provider.prepare(manager, OATK_ENVIRONMENT, request, payload, environment)

    assert prepared.artifacts == {"hmm_profiles": manager.artifact}
    assert prepared.components == (manager.managed_component,)
    assert prepared.database_hashes == {"oatkdb": manager.managed_component.sha256}
    assert prepared.model_hashes == {}


def _fake_scores(pltd_total: float, mito_total: float) -> dict[str, object]:
    def score(_nhmmscan: str, profile: Path, _fasta: Path) -> dict[str, object]:
        return (
            {"total": pltd_total, "best": 100.0, "hits": 5, "best_name": "rbcL"}
            if "pltd" in profile.name
            else {"total": mito_total, "best": 50.0, "hits": 3, "best_name": "cox1"}
        )

    return {"score": score}


def test_collect_outputs_rejects_plastid_contig_for_mito_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    _write_oatk_outputs(context, target="mitochondrion")
    # Sibling plastid profile exists beside the requested mito profile.
    sibling = Path(context.resources.artifacts["hmm_profiles"].uri).parent / (
        "embryophyta_pltd.fam"
    )
    sibling.write_bytes(b"sibling-profile")
    fake = _fake_scores(pltd_total=1000.0, mito_total=10.0)
    monkeypatch.setattr(
        "organelleverse.assembly.backends.oatk._profile_score",
        fake["score"],
    )
    with pytest.raises(OrganelleExecutionError) as raised:
        OatkAdapter().collect_outputs(context)
    assert raised.value.code == "assembly.organelle_mismatch"
    details = raised.value.details
    assert details["requested_organelle"] == "mitochondrion"
    assert details["detected_organelle"] == "plastid"
    assert details["sibling_summary"]["total"] == 1000.0


def test_collect_outputs_accepts_matching_organelle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, organelle="plastid")
    _write_oatk_outputs(context, target="plastid")
    sibling = Path(context.resources.artifacts["hmm_profiles"].uri).parent / (
        "embryophyta_mito.fam"
    )
    sibling.write_bytes(b"sibling-profile")
    fake = _fake_scores(pltd_total=2000.0, mito_total=100.0)
    monkeypatch.setattr(
        "organelleverse.assembly.backends.oatk._profile_score",
        fake["score"],
    )
    outputs = OatkAdapter().collect_outputs(context)
    assert outputs.primary_sequence_role == "assembly_fasta"


def test_verification_skips_when_sibling_profile_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path, organelle="mitochondrion")
    _write_oatk_outputs(context, target="mitochondrion")
    # No sibling fam next to the requested profile.

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("verification must be skipped without a sibling profile")

    monkeypatch.setattr(
        "organelleverse.assembly.backends.oatk._profile_score", explode
    )
    outputs = OatkAdapter().collect_outputs(context)
    assert outputs.primary_sequence_role == "assembly_fasta"


def test_profile_score_parses_hmmer_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout = "\n".join(
        [
            "# nhmmscan :: header",
            "Query:       ctg [L=100]",
            "Description: length=100",
            "Scores for complete hit:",
            "    E-value  score  bias  Model      start    end  Description",
            "    ------- ------ -----  --------   -----  -----  -----------",
            "          0 1647.1  50.8  rrn26      13534  16538 ",
            "   9e-215  712.6  25.2  rrn18       9826  10903 ",
            "",
            "Internal pipeline statistics summary:",
        ]
    )

    completed = SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        "organelleverse.assembly.backends.oatk.run_external",
        lambda *a, **k: completed,
    )
    summary = _profile_score("nhmmscan", tmp_path / "p.fam", tmp_path / "x.fa")
    assert summary["total"] == pytest.approx(2359.7)
    assert summary["best"] == pytest.approx(1647.1)
    assert summary["hits"] == 2
    assert summary["best_name"] == "rrn26"
