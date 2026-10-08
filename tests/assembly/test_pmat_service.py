from __future__ import annotations

import hashlib
from pathlib import Path

import organelleverse.assembly.service as assembly_service
from organelleverse.assembly.backends.himt import EmptyAssemblyResourceProvider
from organelleverse.assembly.backends.pmat import PmatAdapter
from organelleverse.assembly.backends.runtime import (
    AssemblyRuntimeRegistry,
    BackendRuntime,
)
from organelleverse.assembly.contracts import AssemblyRequest
from organelleverse.assembly.data_contract import validate_pmat_assembly_data
from organelleverse.assembly.environment_contracts import (
    EnvironmentResolution,
    ProviderComponentIdentity,
    ResolvedProvider,
)
from organelleverse.assembly.environment_resolver import EnvironmentResolver
from organelleverse.assembly.environment_specs import AssemblyEnvironmentSpec, CondaPlatformSpec
from organelleverse.assembly.environments import (
    EnvironmentManager,
    PreparedEnvironment,
    PreparedExecutable,
)
from organelleverse.assembly.manifests import AssemblyRunManifest
from organelleverse.assembly.service import execute_assembly
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.runtime import managed_run_path

_FAKE_PMAT = r"""#!/usr/bin/env python3
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--version", action="version", version="PMAT v2.1.5")
parser.add_argument("command")
parser.add_argument("-i")
parser.add_argument("-o", required=True)
parser.add_argument("-t")
parser.add_argument("-k")
parser.add_argument("-g", required=False)
parser.add_argument("-p")
parser.add_argument("-G")
parser.add_argument("-x")
parser.add_argument("-S")
parser.add_argument("-F")
parser.add_argument("-D")
parser.add_argument("-K")
parser.add_argument("-I")
parser.add_argument("-L")
parser.add_argument("-T")
parser.add_argument("-m", action="store_true")
args = parser.parse_args()
root = Path(args.o)
(root / "gfa_result").mkdir(parents=True)
(root / "assembly_result").mkdir()
(root / "subsample").mkdir()
target = args.G
fasta = "PMAT_mt.fa" if target == "mt" else "PMAT_pt.fa"
(root / "gfa_result" / fasta).write_text(">ctg\nACGTACGT\n")
(root / "gfa_result" / f"PMAT_{target}_main.gfa").write_text("H\tVN:Z:1.0\nS\tctg\tACGTACGT\n")
(root / "gfa_result" / f"PMAT_{target}_raw.gfa").write_text("H\tVN:Z:1.0\nS\tctg\tACGTACGT\n")
if target == "mt":
    (root / "PMAT_orgAss.txt").write_text("Mitochondrial Assembly Assessment\n")
(root / "assembly_result" / "PMATAllContigs.fna").write_text(">ctg\nACGTACGT\n")
(root / "assembly_result" / "PMATContigGraph.txt").write_text("1\t2\n")
(root / "subsample" / "PMAT_cut_seq.fa").write_text(">read\nACGTACGT\n")
"""


def _request(tmp_path: Path) -> AssemblyRequest:
    reads = tmp_path / "reads.fastq"
    reads.write_bytes(b"@r\nACGTACGT\n+\nIIIIIIII\n")
    artifact = ArtifactRef.from_path(
        reads,
        kind="long_read",
        format="fastq",
        media_type="application/x-fastq",
    )
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {"long_0_reads": artifact},
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "long_libraries": [
                    {
                        "technology": "pacbio_hifi",
                        "quality_state": "ccs",
                        "reads_artifact": "long_0_reads",
                    }
                ],
            },
        }
    )
    return AssemblyRequest(
        data=data,
        organelle="mitochondrion",
        method="pmat",
        threads=2,
    )


def _spec() -> AssemblyEnvironmentSpec:
    return AssemblyEnvironmentSpec(
        backend_id="pmat",
        contract_version="test.pmat.environment.v1",
        platforms=(
            CondaPlatformSpec(
                platform="linux-64",
                package="pmat=2.1.5",
                lock_resource="organelleverse.assembly.resources.environments.oatk/linux-64.explicit.txt",
                executable_names=("PMAT",),
                version_argv=("--version",),
            ),
        ),
    )


class _PmatEnvironmentManager(EnvironmentManager):
    def __init__(self, root: Path) -> None:
        super().__init__(cache_root=root)
        self.digest = "sha256:" + "b" * 64
        self.prefix = root / "pmat"
        executable = self.prefix / "bin" / "PMAT"
        executable.parent.mkdir(parents=True)
        executable.write_text(_FAKE_PMAT)
        executable.chmod(0o755)
        self.environment = PreparedEnvironment(
            backend_id="pmat",
            carrier="conda",
            platform="linux-64",
            digest=self.digest,
            prefix=self.prefix,
            executables=(PreparedExecutable(name="pmat", path=executable),),
            version="2.1.5",
        )

    def expected_environment_digest(
        self, spec: AssemblyEnvironmentSpec, *, platform: str | None = None
    ) -> str:
        return self.digest

    def prepare(
        self,
        spec: AssemblyEnvironmentSpec,
        *,
        policy: str,
        platform: str | None = None,
    ) -> PreparedEnvironment:
        return self.environment


class _ExistingPmatResolver(EnvironmentResolver):
    def __init__(self, manager: _PmatEnvironmentManager) -> None:
        super().__init__()
        executable = manager.environment.require_executable("pmat")
        self.provider = ResolvedProvider(
            requested_source="auto",
            discovery_source="agent_hint",
            carrier="conda",
            platform="linux-64",
            prefix=manager.environment.prefix,
            capability_contract_digest="sha256:" + "c" * 64,
            components=(
                ProviderComponentIdentity(
                    role="pmat",
                    kind="executable",
                    path=executable,
                    sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    version="2.1.5",
                ),
            ),
        )

    def resolve(self, **_kwargs: object) -> EnvironmentResolution:
        return EnvironmentResolution(selected_provider=self.provider)


def test_private_pmat_runtime_executes_and_publishes_evidence(tmp_path: Path) -> None:
    from organelleverse.assembly.environment_specs import PMAT_ORIENTATION_VERSION

    runtime = BackendRuntime(
        backend_id="pmat",
        environment_spec=_spec(),
        data_validator=validate_pmat_assembly_data,
        resource_provider=EmptyAssemblyResourceProvider(),
        adapter_factory=PmatAdapter,
    )
    original = assembly_service.RUNTIMES
    assembly_service.RUNTIMES = AssemblyRuntimeRegistry((runtime,))
    try:
        request = _request(tmp_path)
        manager = _PmatEnvironmentManager(tmp_path / "environment")
        manager.environment = manager.environment.model_copy(
            update={"software_version": PMAT_ORIENTATION_VERSION}
        )
        result = execute_assembly(
            request,
            environment_manager=manager,
        )
    finally:
        assembly_service.RUNTIMES = original

    assert result.status == "ok"
    assert result.provenance.software_versions["pmat"] == PMAT_ORIENTATION_VERSION
    manifest = AssemblyRunManifest.model_validate_json(
        (
            managed_run_path(
                "assembly.assemble", f"sha256-{request.resolved_semantic_hash('pmat')}"
            )
            / "assembly_run_record.json"
        ).read_bytes()
    )
    assert manifest.selected_backend == "pmat"
    assert manifest.primary_sequence_role == "assembly_fasta"
    assert set(item.role for item in manifest.outputs) >= {
        "assembly_fasta",
        "pmat_subsample_manifest",
        "pmat_assembly_result_manifest",
    }
    assert manifest.parameters.backend_parameters["correction_task"] == "skip"
    assert all(str(tmp_path) not in item for item in manifest.stable_argv)


def test_private_pmat_reuses_a_verified_existing_provider_without_install(
    tmp_path: Path,
) -> None:
    runtime = BackendRuntime(
        backend_id="pmat",
        environment_spec=_spec(),
        data_validator=validate_pmat_assembly_data,
        resource_provider=EmptyAssemblyResourceProvider(),
        adapter_factory=PmatAdapter,
    )
    manager = _PmatEnvironmentManager(tmp_path / "environment")
    resolver = _ExistingPmatResolver(manager)
    original = assembly_service.RUNTIMES
    assembly_service.RUNTIMES = AssemblyRuntimeRegistry((runtime,))
    try:
        request = _request(tmp_path)
        result = execute_assembly(
            request,
            environment_manager=manager,
            environment_resolver=resolver,
        )
    finally:
        assembly_service.RUNTIMES = original

    assert result.status == "ok"
    manifest = AssemblyRunManifest.model_validate_json(
        (
            managed_run_path(
                "assembly.assemble", f"sha256-{request.resolved_semantic_hash('pmat')}"
            )
            / "assembly_run_record.json"
        ).read_bytes()
    )
    assert manifest.environment.digest == resolver.provider.provider_digest
