from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from organelleverse.assembly.environment_specs import (
    HIMT_ENVIRONMENT,
    OATK_ENVIRONMENT,
    AssemblyEnvironmentSpec,
    CondaPlatformSpec,
    ManagedDatabaseSpec,
    ManagedFileSpec,
    normalize_platform,
)
from organelleverse.assembly.environments import (
    EnvironmentManager,
    PreparedEnvironment,
    PreparedExecutable,
    PreparedProfile,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleDependencyError

# ---------------------------------------------------------------------------
# Recording fakes for the carrier (conda prefix materializer) and downloader.
# ---------------------------------------------------------------------------


class RecordingDownloader:
    """Deterministic downloader whose matching bytes are derived from the URL."""

    def __init__(self, *, mismatch: bool = False) -> None:
        self.calls: list[tuple[str, Path]] = []
        self._mismatch = mismatch

    def download(self, url: str, destination: Path, expected_sha256: str) -> bytes:
        self.calls.append((url, destination))
        if self._mismatch:
            destination.write_bytes(b"corrupt-bytes-not-matching-manifest")
            return b"corrupt-bytes-not-matching-manifest"
        content = (url.encode("utf-8") + b"\n") * 16
        destination.write_bytes(content)
        return content


class RecordingCarrier:
    def __init__(self, *, create_probe_files: bool = False, version: str = "oatk 1.0") -> None:
        self.calls: list[tuple[str, Path]] = []
        self.create_count = 0
        self.version_probes: list[tuple[Path, tuple[str, ...]]] = []
        self._create_probe_files = create_probe_files
        self._version = version

    def create_prefix(self, explicit_lock: str, destination: Path) -> None:
        self.calls.append((explicit_lock, destination))
        self.create_count += 1
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "bin").mkdir(parents=True, exist_ok=True)
        oatk = destination / "bin" / "oatk"
        oatk.write_text("#!/bin/sh\necho oatk\n")
        oatk.chmod(0o755)
        nhmmscan = destination / "bin" / "nhmmscan"
        nhmmscan.write_text("#!/bin/sh\necho nhmmscan\n")
        nhmmscan.chmod(0o755)
        hmmpress = destination / "bin" / "hmmpress"
        hmmpress.write_text("#!/bin/sh\necho hmmpress\n")
        hmmpress.chmod(0o755)
        if self._create_probe_files:
            (destination / "oatk.version").write_text(self._version)
            meta = destination / "conda-meta"
            meta.mkdir(parents=True, exist_ok=True)
            (meta / "oatk-1.0-h577a1d6_1.json").write_text(
                json.dumps({"name": "oatk", "version": "1.0", "build": "h577a1d6_1"})
            )

    def probe_version(self, executable: Path, argv: tuple[str, ...]) -> str:
        self.version_probes.append((executable, argv))
        return self._version

    def probe_package_identity(self, prefix: Path) -> tuple[tuple[str, str, str], ...]:
        meta = prefix / "conda-meta"
        if not meta.exists():
            return ()
        records: list[tuple[str, str, str]] = []
        for entry in sorted(meta.glob("*.json")):
            data = json.loads(entry.read_text())
            records.append((str(data["name"]), str(data["version"]), str(data["build"])))
        return tuple(records)


class CorruptingCarrier(RecordingCarrier):
    def __init__(self) -> None:
        super().__init__(create_probe_files=True, version="oatk 1.0")

    def probe_version(self, executable: Path, argv: tuple[str, ...]) -> str:
        return "unexpected-version"

    def probe_package_identity(self, prefix: Path) -> tuple[tuple[str, str, str], ...]:
        # The package record no longer matches the expected backend build.
        return (("not-oatk", "0.0", "x"),)


class FailingCarrier(RecordingCarrier):
    def __init__(self) -> None:
        super().__init__(create_probe_files=True, version="oatk 1.0")

    def create_prefix(self, explicit_lock: str, destination: Path) -> None:
        super().create_prefix(explicit_lock, destination)
        # sabotage: remove the conda-meta so package identity verification fails
        shutil.rmtree(destination / "conda-meta", ignore_errors=True)


class RecordingProfileCompiler:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    def press(self, executable: Path, profile: Path) -> None:
        self.calls.append((executable, profile))
        for suffix in ("h3f", "h3i", "h3m", "h3p"):
            profile.with_name(f"{profile.name}.{suffix}").write_bytes(
                f"compiled-{suffix}\n".encode()
            )


# ---------------------------------------------------------------------------
# A synthetic spec whose database manifest hashes match the recording downloader.
# ---------------------------------------------------------------------------


def _file_spec(name: str, url: str) -> ManagedFileSpec:
    content = (url.encode("utf-8") + b"\n") * 16
    return ManagedFileSpec(
        name=name,
        url=url,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def _prepared_environment_with_names(
    tmp_path: Path, names: tuple[str, ...], version: str = "oatk 1.0"
) -> PreparedEnvironment:
    prefix = tmp_path / "env" / "prepared"
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    executables: list[PreparedExecutable] = []
    for name in names:
        path = prefix / "bin" / name
        path.write_text("#!/bin/sh\necho ok\n")
        path.chmod(0o755)
        executables.append(PreparedExecutable(name=name, path=path))
    return PreparedEnvironment(
        backend_id="probe",
        carrier="conda",
        platform="linux-64",
        digest="sha256:" + "a" * 64,
        prefix=prefix,
        executables=tuple(executables),
        version=version,
    )


def _synthetic_spec() -> AssemblyEnvironmentSpec:
    base = "https://example.org/oatkdb/v1/"
    names = (
        "embryophyta_mito.fam",
        "embryophyta_mito.fam.h3f",
        "embryophyta_mito.fam.h3i",
        "embryophyta_mito.fam.h3m",
        "embryophyta_mito.fam.h3p",
        "embryophyta_pltd.fam",
        "embryophyta_pltd.fam.h3f",
        "embryophyta_pltd.fam.h3i",
        "embryophyta_pltd.fam.h3m",
        "embryophyta_pltd.fam.h3p",
    )
    files = tuple(_file_spec(name, base + name) for name in names)
    return AssemblyEnvironmentSpec(
        backend_id="oatk",
        contract_version="organelleverse.assembly-environment.v1",
        platforms=(
            CondaPlatformSpec(
                platform="linux-64",
                package="oatk=1.0=h577a1d6_1",
                lock_resource="organelleverse.assembly.resources.environments.oatk/linux-64.explicit.txt",
                executable_names=("oatk", "nhmmscan", "hmmpress"),
                version_argv=("--version",),
            ),
        ),
        database=ManagedDatabaseSpec(
            database_id="oatkdb",
            version="v20230921",
            source_commit="75e8db0ac4a7d508a9a518d900876003ceb70737",
            license="MIT",
            files=files,
        ),
    )


# ---------------------------------------------------------------------------
# require policy: strictly offline, no side effects.
# ---------------------------------------------------------------------------


def test_require_never_materializes_or_downloads(tmp_path: Path) -> None:
    carrier = RecordingCarrier()
    downloader = RecordingDownloader()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=carrier, downloader=downloader
    )

    with pytest.raises(OrganelleDependencyError) as raised:
        manager.prepare(OATK_ENVIRONMENT, policy="require", platform="linux-64")

    assert raised.value.code == "assembly.environment_unavailable"
    assert carrier.calls == []
    assert downloader.calls == []
    assert not (tmp_path / "cache").exists()


def test_expected_environment_digest_is_computable_without_materialization(
    tmp_path: Path,
) -> None:
    carrier = RecordingCarrier()
    downloader = RecordingDownloader()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=carrier, downloader=downloader
    )

    digest = manager.expected_environment_digest(OATK_ENVIRONMENT, platform="linux-64")

    assert digest.startswith("sha256:")
    assert carrier.calls == []
    assert downloader.calls == []
    assert not (tmp_path / "cache").exists()


def test_expected_environment_digest_is_stable_across_calls(tmp_path: Path) -> None:
    manager_a = EnvironmentManager(cache_root=tmp_path / "a", carrier=RecordingCarrier())
    manager_b = EnvironmentManager(cache_root=tmp_path / "b", carrier=RecordingCarrier())
    assert manager_a.expected_environment_digest(
        OATK_ENVIRONMENT, platform="linux-64"
    ) == manager_b.expected_environment_digest(OATK_ENVIRONMENT, platform="linux-64")
    assert manager_a.expected_environment_digest(
        OATK_ENVIRONMENT, platform="linux-64"
    ) != manager_a.expected_environment_digest(OATK_ENVIRONMENT, platform="osx-64")


def test_expected_environment_digest_rejects_unsupported_platform(tmp_path: Path) -> None:
    manager = EnvironmentManager(cache_root=tmp_path / "cache", carrier=RecordingCarrier())
    with pytest.raises(OrganelleDependencyError):
        manager.expected_environment_digest(OATK_ENVIRONMENT, platform="win-64")


def test_himt_expected_environment_digest_is_stable_and_platform_specific(
    tmp_path: Path,
) -> None:
    manager = EnvironmentManager(cache_root=tmp_path / "cache", carrier=RecordingCarrier())
    linux = manager.expected_environment_digest(HIMT_ENVIRONMENT, platform="linux-64")
    linux_aarch64 = manager.expected_environment_digest(HIMT_ENVIRONMENT, platform="linux-aarch64")
    osx = manager.expected_environment_digest(HIMT_ENVIRONMENT, platform="osx-64")
    osx_arm = manager.expected_environment_digest(HIMT_ENVIRONMENT, platform="osx-arm64")
    assert linux.startswith("sha256:")
    assert len({linux, linux_aarch64, osx, osx_arm}) == 4


def test_expected_profile_identity_is_computable_without_materialization(
    tmp_path: Path,
) -> None:
    carrier = RecordingCarrier()
    downloader = RecordingDownloader()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=carrier, downloader=downloader
    )

    identity = manager.expected_profile_identity(OATK_ENVIRONMENT, organelle="mitochondrion")

    assert identity.sha256 is not None
    assert identity.name == "embryophyta_mito.fam"
    assert carrier.calls == []
    assert downloader.calls == []
    assert not (tmp_path / "cache").exists()


def test_expected_profile_identity_selects_the_right_target(tmp_path: Path) -> None:
    manager = EnvironmentManager(cache_root=tmp_path / "cache", carrier=RecordingCarrier())
    mito = manager.expected_profile_identity(OATK_ENVIRONMENT, organelle="mitochondrion")
    pltd = manager.expected_profile_identity(OATK_ENVIRONMENT, organelle="plastid")
    assert mito.name == "embryophyta_mito.fam"
    assert pltd.name == "embryophyta_pltd.fam"
    assert mito.sha256 != pltd.sha256


# ---------------------------------------------------------------------------
# ensure policy: atomic publication, concurrency, probes.
# ---------------------------------------------------------------------------


def test_ensure_publishes_one_verified_prefix_under_concurrency(tmp_path: Path) -> None:
    carrier = RecordingCarrier(create_probe_files=True)
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=carrier, downloader=RecordingDownloader()
    )

    with ThreadPoolExecutor(max_workers=2) as pool:

        def prepare_once(_index: int) -> PreparedEnvironment:
            return manager.prepare(OATK_ENVIRONMENT, policy="ensure", platform="linux-64")

        prepared = tuple(pool.map(prepare_once, range(2)))

    assert isinstance(prepared[0], PreparedEnvironment)
    assert prepared[0].digest == prepared[1].digest
    assert prepared[0].prefix == prepared[1].prefix
    assert prepared[0].carrier == "conda"
    assert prepared[0].platform == "linux-64"
    assert prepared[0].require_executable("oatk").name == "oatk"
    assert prepared[0].require_executable("nhmmscan").name == "nhmmscan"
    assert prepared[0].require_executable("hmmpress").name == "hmmpress"
    assert carrier.create_count == 1


def test_ensure_reuses_an_existing_verified_prefix(tmp_path: Path) -> None:
    carrier = RecordingCarrier(create_probe_files=True)
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=carrier, downloader=RecordingDownloader()
    )

    first = manager.prepare(OATK_ENVIRONMENT, policy="ensure", platform="linux-64")
    second = manager.prepare(OATK_ENVIRONMENT, policy="ensure", platform="linux-64")

    assert first == second
    assert carrier.create_count == 1


def test_environment_prefix_is_safe_to_prepend_to_path(tmp_path: Path) -> None:
    carrier = RecordingCarrier(create_probe_files=True)
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache",
        carrier=carrier,
        downloader=RecordingDownloader(),
    )

    environment = manager.prepare(OATK_ENVIRONMENT, policy="ensure", platform="linux-64")

    assert os.pathsep not in environment.prefix.name
    assert environment.prefix.name == environment.digest.removeprefix("sha256:")
    assert all(os.pathsep not in destination.name for _lock, destination in carrier.calls)


def test_ensure_rejects_corrupt_cached_probes(tmp_path: Path) -> None:
    carrier = CorruptingCarrier()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=carrier, downloader=RecordingDownloader()
    )

    with pytest.raises(OrganelleDependencyError) as raised:
        manager.prepare(OATK_ENVIRONMENT, policy="ensure", platform="linux-64")
    assert raised.value.code == "assembly.environment_unavailable"


def test_ensure_cleans_up_temporary_prefix_on_failure(tmp_path: Path) -> None:
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache",
        carrier=FailingCarrier(),
        downloader=RecordingDownloader(),
    )

    with pytest.raises(OrganelleDependencyError):
        manager.prepare(OATK_ENVIRONMENT, policy="ensure", platform="linux-64")

    digest = manager.expected_environment_digest(OATK_ENVIRONMENT, platform="linux-64")
    environments_root = tmp_path / "cache" / "environments" / "oatk"
    assert not (environments_root / digest).exists()
    assert not any(p.name.startswith(".") for p in environments_root.glob("*"))


def test_prepare_rejects_unsupported_platform_under_ensure(tmp_path: Path) -> None:
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache",
        carrier=RecordingCarrier(create_probe_files=True),
        downloader=RecordingDownloader(),
    )
    with pytest.raises(OrganelleDependencyError) as raised:
        manager.prepare(OATK_ENVIRONMENT, policy="ensure", platform="win-64")
    assert raised.value.code == "assembly.environment_unavailable"


# ---------------------------------------------------------------------------
# profile preparation: managed + user override (synthetic spec).
# ---------------------------------------------------------------------------


def test_prepare_profile_managed_mitochondrion_under_ensure(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=RecordingCarrier(), downloader=RecordingDownloader()
    )
    profile = manager.prepare_profile(spec, organelle="mitochondrion", policy="ensure")
    assert isinstance(profile, PreparedProfile)
    assert profile.source == "managed"
    assert profile.target == "mitochondrion"
    assert profile.path.name == "embryophyta_mito.fam"
    assert isinstance(profile.artifact, ArtifactRef)
    assert profile.artifact.validated is True


def test_prepare_profile_managed_plastid_under_ensure(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=RecordingCarrier(), downloader=RecordingDownloader()
    )
    profile = manager.prepare_profile(spec, organelle="plastid", policy="ensure")
    assert profile.target == "plastid"
    assert profile.path.name == "embryophyta_pltd.fam"


def test_managed_profile_identity_covers_and_verifies_hmm_sidecars(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=RecordingCarrier(), downloader=RecordingDownloader()
    )
    profile = manager.prepare_profile(spec, organelle="mitochondrion", policy="ensure")
    expected = manager.expected_profile_identity(spec, organelle="mitochondrion")

    assert profile.component == expected
    assert profile.component.sha256 != profile.artifact.sha256

    sidecar = profile.path.with_name(profile.path.name + ".h3f")
    sidecar.write_bytes(b"tampered")
    with pytest.raises(OrganelleDependencyError) as raised:
        manager.prepare_profile(spec, organelle="mitochondrion", policy="require")
    assert raised.value.code == "assembly.environment_unavailable"


def test_prepare_profile_require_is_offline(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    downloader = RecordingDownloader()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=RecordingCarrier(), downloader=downloader
    )
    with pytest.raises(OrganelleDependencyError) as raised:
        manager.prepare_profile(spec, organelle="mitochondrion", policy="require")
    assert raised.value.code == "assembly.environment_unavailable"
    assert downloader.calls == []


def test_user_supplied_profile_override_is_rehashed_without_download(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    profile_file = tmp_path / "user_mito.fam"
    profile_file.write_bytes(b"user-supplied-profile-content")
    override = ArtifactRef.from_path(
        profile_file, kind="hmm_profile", format="fam", media_type="text/plain"
    )
    downloader = RecordingDownloader()
    compiler = RecordingProfileCompiler()
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache",
        carrier=RecordingCarrier(create_probe_files=True),
        downloader=downloader,
        profile_compiler=compiler,
    )

    identity = manager.expected_profile_identity(spec, organelle="mitochondrion", override=override)
    assert identity.sha256 != override.sha256
    assert downloader.calls == []

    prepared = manager.prepare_profile(
        spec, organelle="mitochondrion", policy="ensure", override=override
    )
    assert prepared.source == "user_supplied"
    assert prepared.artifact.sha256 == override.sha256
    assert downloader.calls == []
    assert len(compiler.calls) == 1
    assert prepared.path.name == "profile.fam"
    assert all(
        prepared.path.with_name(f"{prepared.path.name}.{suffix}").is_file()
        for suffix in ("h3f", "h3i", "h3m", "h3p")
    )


def test_user_profile_identity_is_independent_of_source_filename(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    first_path = tmp_path / "first-name.fam"
    second_path = tmp_path / "renamed-profile.hmm"
    first_path.write_bytes(b"identical-profile-content")
    second_path.write_bytes(first_path.read_bytes())
    first = ArtifactRef.from_path(
        first_path, kind="hmm_profile", format="fam", media_type="text/plain"
    )
    second = ArtifactRef.from_path(
        second_path, kind="hmm_profile", format="fam", media_type="text/plain"
    )
    manager = EnvironmentManager(cache_root=tmp_path / "cache", carrier=RecordingCarrier())

    assert manager.expected_profile_identity(
        spec, organelle="mitochondrion", override=first
    ) == manager.expected_profile_identity(spec, organelle="mitochondrion", override=second)


def test_changed_user_override_fails_before_reuse(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    profile_file = tmp_path / "user_mito.fam"
    profile_file.write_bytes(b"original-content")
    override = ArtifactRef.from_path(
        profile_file, kind="hmm_profile", format="fam", media_type="text/plain"
    )
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=RecordingCarrier(), downloader=RecordingDownloader()
    )
    profile_file.write_bytes(b"tampered-content")

    with pytest.raises(OrganelleDependencyError):
        manager.expected_profile_identity(spec, organelle="mitochondrion", override=override)


def test_database_download_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    spec = _synthetic_spec()
    downloader = RecordingDownloader(mismatch=True)
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=RecordingCarrier(), downloader=downloader
    )
    with pytest.raises(OrganelleDependencyError) as raised:
        manager.prepare_profile(spec, organelle="mitochondrion", policy="ensure")
    assert raised.value.code == "assembly.environment_unavailable"


def test_environment_and_profile_identities_are_independent(tmp_path: Path) -> None:
    manager = EnvironmentManager(cache_root=tmp_path / "cache", carrier=RecordingCarrier())
    env_digest = manager.expected_environment_digest(OATK_ENVIRONMENT, platform="linux-64")
    profile_identity = manager.expected_profile_identity(
        OATK_ENVIRONMENT, organelle="mitochondrion"
    )
    assert env_digest != f"sha256:{profile_identity.sha256}"


def test_prepared_environment_exposes_named_executables(tmp_path: Path) -> None:
    prepared = _prepared_environment_with_names(tmp_path, ("oatk", "nhmmscan", "hmmpress"))

    assert prepared.require_executable("oatk") == prepared.prefix / "bin" / "oatk"
    assert prepared.require_executable("nhmmscan") == prepared.prefix / "bin" / "nhmmscan"
    with pytest.raises(OrganelleDependencyError) as raised:
        prepared.require_executable("missing")
    assert raised.value.code == "assembly.environment_unavailable"


def test_carrier_probes_first_executable_with_platform_version_argv(tmp_path: Path) -> None:
    carrier = RecordingCarrier(create_probe_files=True)
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=carrier, downloader=RecordingDownloader()
    )
    manager.prepare(OATK_ENVIRONMENT, policy="ensure", platform="linux-64")

    assert len(carrier.version_probes) >= 1
    executable, argv = carrier.version_probes[0]
    assert executable.name == "oatk"
    assert argv == ("--version",)
    assert "oatk" not in str(argv)


def test_database_free_environment_cannot_prepare_profile(tmp_path: Path) -> None:
    spec = AssemblyEnvironmentSpec(
        backend_id="probe",
        contract_version="probe.environment.v1",
        platforms=(
            CondaPlatformSpec(
                platform="linux-64",
                package="probe=1.0=0",
                lock_resource="organelleverse.assembly.resources.environments.oatk/linux-64.explicit.txt",
                executable_names=("probe",),
                version_argv=("--version",),
            ),
        ),
    )
    manager = EnvironmentManager(
        cache_root=tmp_path / "cache", carrier=RecordingCarrier(), downloader=RecordingDownloader()
    )
    with pytest.raises(OrganelleDependencyError) as raised:
        manager.prepare_profile(spec, organelle="mitochondrion", policy="ensure")
    assert raised.value.code == "assembly.environment_unavailable"


def test_default_cache_root_is_user_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORGANELLEVERSE_CACHE_ROOT", raising=False)
    manager = EnvironmentManager(carrier=RecordingCarrier(), downloader=RecordingDownloader())
    assert ".cache" in str(manager.cache_root)


def test_normalize_platform_maps_known_architectures() -> None:
    assert normalize_platform(system="Linux", machine="x86_64") == "linux-64"
    assert normalize_platform(system="Darwin", machine="x86_64") == "osx-64"
    assert normalize_platform(system="Linux", machine="aarch64") == "linux-aarch64"
    assert normalize_platform(system="Darwin", machine="arm64") == "osx-arm64"
