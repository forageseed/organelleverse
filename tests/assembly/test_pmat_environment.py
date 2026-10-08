from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from organelleverse.assembly.backends.runtime import RUNTIMES, pmat_runtime
from organelleverse.assembly.environment_capabilities import runtime_capabilities
from organelleverse.assembly.environment_registry import InstallationRegistry
from organelleverse.assembly.environment_specs import (
    PMAT_ENVIRONMENT,
    AssemblyEnvironmentSpec,
    CondaPlatformSpec,
    ManagedBuildOutput,
    ManagedExecutableAlias,
    ManagedSourceBuildSpec,
)
from organelleverse.assembly.environments import (
    EnvironmentManager,
    _safe_extract_tar,  # pyright: ignore[reportPrivateUsage]
)
from organelleverse.core.errors import OrganelleDependencyError


class _Carrier:
    def create_prefix(self, explicit_lock: str, destination: Path) -> None:
        del explicit_lock
        bindir = destination / "bin"
        bindir.mkdir(parents=True)
        for name in ("blastn", "canu", "nextDenovo", "Rscript"):
            path = bindir / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        meta = destination / "conda-meta"
        meta.mkdir()
        (meta / "blast.json").write_text(
            json.dumps({"name": "blast", "version": "2.16.0", "build": "0"})
        )

    def probe_version(self, executable: Path, argv: tuple[str, ...]) -> str:
        del executable, argv
        return "PMAT v2.1.5"

    def probe_package_identity(self, prefix: Path) -> tuple[tuple[str, str, str], ...]:
        del prefix
        return (("blast", "2.16.0", "0"),)


class _ArchiveDownloader:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def download(self, url: str, destination: Path, expected_sha256: str) -> bytes:
        del url
        assert hashlib.sha256(self.content).hexdigest() == expected_sha256
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.content)
        return self.content


def _archive(*, malicious_name: str | None = None) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        files = {
            "PMAT2-2.1.5/PMAT": b"#!/bin/sh\necho PMAT v2.1.5\n",
            "PMAT2-2.1.5/container/runAssembly.sif": b"sif",
            "PMAT2-2.1.5/lib/genomescope.R": b"#!/usr/bin/env Rscript\n",
        }
        if malicious_name is not None:
            files[malicious_name] = b"escape"
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o755 if name.endswith(("PMAT", ".R")) else 0o444
            archive.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def _source_spec(content: bytes) -> ManagedSourceBuildSpec:
    return ManagedSourceBuildSpec(
        source_uri="https://example.org/PMAT2-v2.1.5.tar.gz",
        source_sha256=hashlib.sha256(content).hexdigest(),
        archive_root="PMAT2-2.1.5",
        build_argv=("true",),
        outputs=(
            ManagedBuildOutput(
                role="pmat",
                relative_path="PMAT",
                kind="executable",
                install_name="pmat",
            ),
            ManagedBuildOutput(
                role="pmat_container",
                relative_path="container/runAssembly.sif",
                kind="resource",
            ),
            ManagedBuildOutput(
                role="pmat_genomescope",
                relative_path="lib/genomescope.R",
                kind="resource",
            ),
        ),
        executable_aliases=(ManagedExecutableAlias(name="nextdenovo", target="nextDenovo"),),
        required_host_commands=(("apptainer", "singularity"),),
    )


def _spec(content: bytes) -> AssemblyEnvironmentSpec:
    return AssemblyEnvironmentSpec(
        backend_id="pmat",
        contract_version="organelleverse.assembly-environment.v1",
        platforms=(
            CondaPlatformSpec(
                platform="linux-64",
                package="pmat-source=2.1.5",
                lock_resource=(
                    "organelleverse.assembly.resources.environments.oatk/linux-64.explicit.txt"
                ),
                executable_names=("pmat", "blastn", "canu", "nextdenovo", "Rscript"),
                version_argv=("--version",),
                source_build=_source_spec(content),
            ),
        ),
    )


def test_pmat_environment_freezes_official_v215_source() -> None:
    platform = PMAT_ENVIRONMENT.require_platform("linux-64")
    source = platform.source_build
    assert source is not None
    assert source.source_uri.endswith("/04534a2adf0c5309cb2e7478fd2bcc98ced3818c.tar.gz")
    assert (
        source.source_sha256 == "5564becd242ab254e379758133b514ce0c38431df55c77bee74836f1914bf872"
    )
    assert source.build_argv == ("make", "CC=x86_64-conda-linux-gnu-cc")
    assert {item.relative_path for item in source.outputs} == {
        "PMAT",
        "container/runAssembly.sif",
        "lib/genomescope.R",
    }


def test_pmat_runtime_is_complete_and_released() -> None:
    runtime = pmat_runtime()
    assert runtime.backend_id == "pmat"
    assert runtime.environment_spec is PMAT_ENVIRONMENT
    assert runtime.adapter_factory().backend_id == "pmat"
    assert RUNTIMES["pmat"] is runtime


def test_managed_source_build_is_part_of_environment_identity(tmp_path: Path) -> None:
    content = _archive()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache",
        carrier=_Carrier(),
        downloader=_ArchiveDownloader(content),
        find_executable=lambda names: Path("/usr/local/bin/apptainer"),
    )
    first = _spec(content)
    changed_source = first.platforms[0].source_build.model_copy(  # type: ignore[union-attr]
        update={"build_argv": ("make", "PMAT")}
    )
    second = first.model_copy(
        update={
            "platforms": (first.platforms[0].model_copy(update={"source_build": changed_source}),)
        }
    )
    assert manager.expected_environment_digest(first, platform="linux-64") != (
        manager.expected_environment_digest(second, platform="linux-64")
    )


def test_manager_builds_source_links_aliases_and_reuses(tmp_path: Path) -> None:
    content = _archive()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache",
        carrier=_Carrier(),
        downloader=_ArchiveDownloader(content),
        find_executable=lambda names: Path("/usr/local/bin/apptainer"),
    )

    first = manager.prepare(_spec(content), policy="ensure", platform="linux-64")
    second = manager.prepare(_spec(content), policy="ensure", platform="linux-64")

    assert first == second
    assert first.require_executable("pmat").resolve().name == "PMAT"
    assert first.require_executable("nextdenovo").resolve().name == "nextDenovo"
    source = first.prefix / "share" / "pmat"
    assert (source / "container" / "runAssembly.sif").is_file()
    assert (source / "lib" / "genomescope.R").is_file()


def test_managed_build_is_registered_by_actual_component_hashes(tmp_path: Path) -> None:
    content = _archive()
    apptainer = tmp_path / "apptainer"
    apptainer.write_text("#!/bin/sh\n")
    apptainer.chmod(0o755)
    registry = InstallationRegistry(tool_root=tmp_path / "tools")
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache",
        carrier=_Carrier(),
        downloader=_ArchiveDownloader(content),
        find_executable=lambda names: apptainer,
        tool_root=tmp_path / "tools",
    )
    prepared = manager.prepare(_spec(content), policy="ensure", platform="linux-64")
    contract = runtime_capabilities(
        pmat_runtime(),
        {"correction_task": "skip", "correction_software": "nextdenovo"},
    )
    managed_path = prepared.require_executable("pmat").resolve()
    managed_path.parent.chmod(0o775)

    registered = manager.register_prepared_provider(
        prepared,
        contract,
        requested_source="auto",
        resolved_version="2.1.5",
        registry=registry,
    )

    records = registry.locate("pmat")
    assert len(records) == 1
    assert records[0].provider_digest == registered.digest
    assert managed_path.parent.stat().st_mode & 0o022 == 0
    assert {item.role for item in records[0].components} >= {
        "pmat",
        "blastn",
        "pmat_container",
        "pmat_genomescope",
        "container_runtime",
    }


def test_source_build_requires_one_declared_host_provider(tmp_path: Path) -> None:
    content = _archive()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache",
        carrier=_Carrier(),
        downloader=_ArchiveDownloader(content),
        find_executable=lambda names: None,
    )
    with pytest.raises(OrganelleDependencyError) as raised:
        manager.prepare(_spec(content), policy="ensure", platform="linux-64")
    assert raised.value.code == "assembly.environment_unavailable"
    assert not (tmp_path / "cache" / "environments" / "pmat").exists()


def test_safe_tar_extraction_rejects_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(OrganelleDependencyError):
        _safe_extract_tar(_archive(malicious_name="../escaped"), tmp_path / "extract")
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("invalid", [None, "checksum", "context"])
def test_source_patch_is_verified_and_applied_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str | None
) -> None:
    from organelleverse.assembly import environments
    from organelleverse.assembly.environment_specs import ManagedSourcePatch

    patch = (
        b"--- a/PMAT\n+++ b/PMAT\n@@ -1,2 +1,2 @@\n #!/bin/sh\n-echo PMAT v2.1.5\n+echo patched\n"
    )
    if invalid == "context":
        patch = patch.replace(b"-echo PMAT v2.1.5", b"-wrong source")
    checksum = hashlib.sha256(patch).hexdigest() if invalid != "checksum" else "0" * 64
    original_read = environments._read_resource
    monkeypatch.setattr(
        environments,
        "_read_resource",
        lambda name: patch if name == "test.patch" else original_read(name),
    )
    content = _archive()
    source = _source_spec(content).model_copy(
        update={
            "patches": (ManagedSourcePatch(resource="test.patch", sha256=checksum),),
            # This build succeeds only if the patch has already changed its input.
            "build_argv": ("sh", "-c", 'test "$(./PMAT)" = patched'),
        }
    )
    spec = _spec(content)
    spec = spec.model_copy(
        update={
            "platforms": (spec.platforms[0].model_copy(update={"source_build": source}),),
        }
    )
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache",
        carrier=_Carrier(),
        downloader=_ArchiveDownloader(content),
        find_executable=lambda _: Path("/usr/bin/true"),
    )
    if invalid:
        with pytest.raises(OrganelleDependencyError):
            manager.prepare(spec, policy="ensure", platform="linux-64")
        assert not list((tmp_path / "cache/environments/pmat").glob("sha256-*"))
    else:
        built = manager.prepare(spec, policy="ensure", platform="linux-64")
        assert built.require_executable("pmat").read_text() == "#!/bin/sh\necho patched\n"
        assert manager.expected_environment_digest(spec) != manager.expected_environment_digest(
            _spec(content)
        )


def test_packaged_orientation_patch_checksum() -> None:
    from organelleverse.assembly.environments import _read_resource

    source = PMAT_ENVIRONMENT.platforms[0].source_build
    assert source is not None
    assert len(source.patches) == 1
    patch = source.patches[0]
    assert hashlib.sha256(_read_resource(patch.resource)).hexdigest() == patch.sha256


def test_patched_version_survives_registration_and_provider_reuse(tmp_path: Path) -> None:
    from organelleverse.assembly.environment_specs import PMAT_ORIENTATION_VERSION

    class PatchedCarrier(_Carrier):
        def probe_version(self, executable: Path, argv: tuple[str, ...]) -> str:
            return "PMAT v2.1.5 + OrganelleVerse orientation patch"

    content = _archive()
    apptainer = tmp_path / "apptainer"
    apptainer.write_text("#!/bin/sh\n")
    apptainer.chmod(0o755)
    registry = InstallationRegistry(tool_root=tmp_path / "tools")
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache",
        carrier=PatchedCarrier(),
        downloader=_ArchiveDownloader(content),
        find_executable=lambda _: apptainer,
    )
    built = manager.prepare(_spec(content), policy="ensure", platform="linux-64")
    assert built.software_version == PMAT_ORIENTATION_VERSION
    contract = runtime_capabilities(pmat_runtime(), {"correction_task": "skip"})
    registered = manager.register_prepared_provider(
        built,
        contract,
        requested_source="managed",
        resolved_version="2.1.5",
        registry=registry,
    )
    assert registered.version == "2.1.5"
    assert registered.software_version == PMAT_ORIENTATION_VERSION
    from organelleverse.assembly.environment_contracts import ResolvedProvider

    record = registry.locate("pmat")[0]
    provider = ResolvedProvider(
        requested_source="auto",
        discovery_source="registry",
        carrier="conda",
        platform="linux-64",
        prefix=registered.prefix,
        capability_contract_digest=record.capability_contract_digest,
        components=tuple(item.model_dump() for item in record.components),
    )
    reused = manager.prepare_provider(provider)
    assert reused.software_version == PMAT_ORIENTATION_VERSION
