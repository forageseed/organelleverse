"""Static environment, database, and capability contracts for GetOrganelle 1.7.7.1.

These tests pin the immutable identity of the managed GetOrganelle backend:

* the exact Bioconda package ``getorganelle=1.7.7.1=pyhdfd78af_0`` (noarch, build
  number 0) and its strict ``conda-forge,bioconda`` explicit lock;
* the declared executable set the workspace contract must supply; and
* the official GetOrganelleDB commit archive and full 0.0.1 database (URL, SHA256, size, and
  the commit the release tag points at) that Task 3 initializes into a
  content-addressed ``--config-dir``.

GetOrganelle's upstream version is the four-part ``1.7.7.1`` (verified from
``get_organelle_from_reads.py --version`` -> ``GetOrganelle v1.7.7.1``). The
shared :class:`ResolvedBackendVersion` identity and selector accept strict
canonical 3- or 4-component numeric releases, so ``1.7.7.1`` is a first-class
resolved identity: the tested resolver registry resolves ``tested`` and the exact
``1.7.7.1`` pin offline, and a verified installed provider that probes to exactly
``GetOrganelle v1.7.7.1`` is selected by the existing-first resolver before any
managed install plan. PMAT's stricter ``2.x.y``-only policy is unchanged.
"""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import cast

import pytest

from organelleverse.assembly.backends.runtime import BackendRuntime
from organelleverse.assembly.contracts import AssemblyRequest
from organelleverse.assembly.environment_capabilities import runtime_capabilities
from organelleverse.assembly.environment_contracts import (
    BackendCapabilityContract,
    CapabilityItem,
)
from organelleverse.assembly.environment_resolver import EnvironmentResolver
from organelleverse.assembly.environment_specs import (
    GETORGANELLE_ENVIRONMENT,
    GETORGANELLE_PACKAGE,
    GETORGANELLE_PLATFORM,
    GETORGANELLE_VERSION,
    GETORGANELLEDB_ARCHIVE_NAME,
    GETORGANELLEDB_ARCHIVE_SHA256,
    GETORGANELLEDB_ARCHIVE_SIZE_BYTES,
    GETORGANELLEDB_ARCHIVE_URL,
    GETORGANELLEDB_LICENSE,
    GETORGANELLEDB_SOURCE_COMMIT,
    GETORGANELLEDB_VERSION,
)
from organelleverse.assembly.environment_versions import resolve_backend_version
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.operations.parameters import OperationParameterModel

_GETORGANELLE_EXECUTABLES = (
    "get_organelle_from_reads.py",
    "get_organelle_config.py",
    "blastn",
    "bowtie2",
    "spades.py",
)


def test_getorganelle_environment_pins_the_exact_bioconda_package() -> None:
    assert GETORGANELLE_ENVIRONMENT.backend_id == "getorganelle"
    platform = GETORGANELLE_ENVIRONMENT.require_platform(GETORGANELLE_PLATFORM)
    # Real solved Bioconda build string for the noarch 1.7.7.1 package; the build
    # number is 0, surfaced as the ``pyhdfd78af_0`` build string by the solver.
    assert GETORGANELLE_PACKAGE == "getorganelle=1.7.7.1=pyhdfd78af_0"
    assert platform.package == GETORGANELLE_PACKAGE
    assert GETORGANELLE_VERSION == "1.7.7.1"


def test_getorganelle_environment_supports_the_tested_linux_platform() -> None:
    assert tuple(p.platform for p in GETORGANELLE_ENVIRONMENT.platforms) == (GETORGANELLE_PLATFORM,)
    with pytest.raises(OrganelleDependencyError) as raised:
        GETORGANELLE_ENVIRONMENT.require_platform("osx-64")
    assert raised.value.code == "assembly.environment_unavailable"


def test_getorganelle_environment_declares_the_exact_executable_set() -> None:
    platform = GETORGANELLE_ENVIRONMENT.require_platform(GETORGANELLE_PLATFORM)
    assert platform.executable_names == _GETORGANELLE_EXECUTABLES


def test_getorganelle_primary_executable_carries_a_version_probe() -> None:
    platform = GETORGANELLE_ENVIRONMENT.require_platform(GETORGANELLE_PLATFORM)
    assert platform.version_argv == ("--version",)
    assert platform.lock_resource == (
        "organelleverse.assembly.resources.environments.getorganelle/linux-64.explicit.txt"
    )


def test_getorganelle_explicit_lock_is_real_and_content_addressed() -> None:
    from organelleverse.assembly.environments import (
        _resolve_resource,  # pyright: ignore[reportPrivateUsage]
    )

    platform = GETORGANELLE_ENVIRONMENT.require_platform(GETORGANELLE_PLATFORM)
    text = _resolve_resource(platform.lock_resource).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0] == "@EXPLICIT"
    body = lines[1:]
    assert body, "explicit lock must list every transitive package"
    assert len(body) >= 50

    # The lock must be regenerated from the canonical official channels only:
    # every URL is served from conda.anaconda.org over conda-forge/bioconda, and
    # no machine-specific mirror host may appear anywhere in the lock.
    assert "tsinghua" not in text.lower(), "lock must not use the Tsinghua mirror"
    assert "mirrors." not in text.lower(), "lock must not use any mirror host"
    for line in body:
        url, _, sha = line.partition("#")
        assert url.startswith("https://"), url
        host = url.split("/", 3)[2]
        assert host == "conda.anaconda.org", (
            f"lock must use the canonical official channel host, got {host!r}"
        )
        assert "/conda-forge/" in url or "/bioconda/" in url, url
        assert len(sha) == 64, sha
        assert all(c in "0123456789abcdef" for c in sha), sha

    # The pinned GetOrganelle package itself must appear with its real hash on the
    # canonical bioconda channel.
    getorganelle_lines = [
        line for line in body if "getorganelle-1.7.7.1" in line and "noarch" in line
    ]
    assert getorganelle_lines, "lock must contain the getorganelle noarch package"
    assert getorganelle_lines[0] == (
        "https://conda.anaconda.org/bioconda/noarch/"
        "getorganelle-1.7.7.1-pyhdfd78af_0.tar.bz2#"
        "55a6134d5f82c7a3ec32db582e5d269504aeb6863e369226477e804ee1b18c0a"
    )


def test_getorganelledb_archive_identity_is_immutable_and_official() -> None:
    database = GETORGANELLE_ENVIRONMENT.require_database()
    assert database.database_id == "getorganelledb"
    assert database.version == GETORGANELLEDB_VERSION == "0.0.1"
    assert database.source_commit == GETORGANELLEDB_SOURCE_COMMIT
    assert database.source_commit == "8610b6e67d4d9269c85de91e0f58266c31f72388"
    assert database.license == GETORGANELLEDB_LICENSE == "GPL-3.0"

    assert len(database.files) == 1
    archive = database.files[0]
    assert (
        archive.name
        == GETORGANELLEDB_ARCHIVE_NAME
        == (f"GetOrganelleDB-{GETORGANELLEDB_SOURCE_COMMIT}.tar.gz")
    )
    assert (
        archive.url
        == GETORGANELLEDB_ARCHIVE_URL
        == (f"https://github.com/Kinggerm/GetOrganelleDB/archive/{database.source_commit}.tar.gz")
    )
    assert (
        archive.sha256
        == GETORGANELLEDB_ARCHIVE_SHA256
        == ("a522e42f7127d38bb2ce3eb9224213f5997fdd12aae1fa782c57b29ed3754315")
    )
    assert archive.size_bytes == GETORGANELLEDB_ARCHIVE_SIZE_BYTES == 43367830


def test_getorganelledb_production_identity_is_the_full_archive_not_minima() -> None:
    database = GETORGANELLE_ENVIRONMENT.require_database()
    archive = database.files[0]
    # The test-only ``0.0.1.minima`` database must never become the production
    # identity: the full 0.0.1 archive is the only managed database contract.
    assert "minima" not in archive.url
    assert "minima" not in archive.name
    assert archive.size_bytes > 20_000_000


def test_getorganelle_source_identity_participates_in_the_environment() -> None:
    # Task 3 consumes GETORGANELLE_ENVIRONMENT.require_database() to prepare a
    # content-addressed --config-dir; the archive URL + SHA256 are the immutable
    # identity and must round-trip through the strict spec model.
    database = GETORGANELLE_ENVIRONMENT.require_database()
    dumped = database.model_dump(mode="json")
    files = dumped["files"]
    assert files[0]["url"] == GETORGANELLEDB_ARCHIVE_URL
    assert files[0]["sha256"] == GETORGANELLEDB_ARCHIVE_SHA256
    assert dumped["source_commit"] == GETORGANELLEDB_SOURCE_COMMIT


def test_getorganelle_environment_carrier_and_contract_version() -> None:
    assert GETORGANELLE_ENVIRONMENT.carrier == "conda"
    assert GETORGANELLE_ENVIRONMENT.contract_version == ("organelleverse.assembly-environment.v1")


def test_getorganelle_environment_is_strict_and_frozen() -> None:
    database = GETORGANELLE_ENVIRONMENT.require_database()
    platform = GETORGANELLE_ENVIRONMENT.require_platform(GETORGANELLE_PLATFORM)
    for model in (
        GETORGANELLE_ENVIRONMENT,
        database,
        platform,
        database.files[0],
    ):
        assert model.model_config.get("frozen") is True
        assert model.model_config.get("extra") == "forbid"
    # environment_specs must stay a static specification module.
    from organelleverse.assembly.environment_specs import AssemblyEnvironmentSpec

    assert not issubclass(AssemblyEnvironmentSpec, OperationParameterModel)


def _getorganelle_runtime() -> BackendRuntime:
    # GetOrganelle has no released adapter/provider yet (Task 3); runtime_capabilities
    # only reads backend_id + environment_spec, so a minimal binding is sufficient to
    # prove capability derivation from the declared executables.
    return cast(
        BackendRuntime,
        types.SimpleNamespace(
            backend_id="getorganelle",
            environment_spec=GETORGANELLE_ENVIRONMENT,
        ),
    )


def test_runtime_capabilities_derive_only_the_declared_getorganelle_executables() -> None:
    contract = runtime_capabilities(_getorganelle_runtime(), {})
    assert isinstance(contract, BackendCapabilityContract)
    assert contract.backend_id == "getorganelle"
    items: list[CapabilityItem] = list(contract.items)
    assert [tuple(item.safe_names) for item in items] == [
        (name,) for name in _GETORGANELLE_EXECUTABLES
    ]
    # Only the primary backend executable (get_organelle_from_reads.py) is probed;
    # dependency executables carry no version argv.
    assert items[0].role == "getorganelle"
    assert items[0].version_argv == ("--version",)
    for item in items[1:]:
        assert item.version_argv == ()
    # Only the primary executable is probed with the strict four-part style that
    # parses ``GetOrganelle v1.7.7.1`` -> ``1.7.7.1``; unprobed dependencies keep
    # the default semver style and never change Oatk/HiMT/PMAT behavior.
    assert items[0].version_style == "strict_four_part"
    for item in items[1:]:
        assert item.version_style == "semver"


def _illumina_request(tmp_path: Path, *, environment_source: str) -> AssemblyRequest:
    """Minimal Illumina paired-end assembly request for resolver tests."""
    r1 = tmp_path / "r1.fastq"
    r2 = tmp_path / "r2.fastq"
    r1.write_bytes(b"@r\nACGT\n+\nI\n")
    r2.write_bytes(b"@r\nACGT\n+\nI\n")
    data = OrganelleData.model_validate(
        {
            "modality": "sequencing_reads",
            "artifacts": {
                "r1": ArtifactRef.from_path(
                    r1, kind="short_read", format="fastq", media_type="application/x-fastq"
                ),
                "r2": ArtifactRef.from_path(
                    r2, kind="short_read", format="fastq", media_type="application/x-fastq"
                ),
            },
            "payload": {
                "contract_version": "organelleverse.assembly-input.v1",
                "short_libraries": [
                    {
                        "technology": "illumina",
                        "layout": "paired_end",
                        "read1_artifact": "r1",
                        "read2_artifact": "r2",
                        "read_length": 150,
                    }
                ],
                "long_libraries": [],
                "contig_inputs": [],
            },
        }
    )
    return AssemblyRequest(
        data=data,
        organelle="plastid",
        method="auto",
        threads=1,
        environment_source=environment_source,  # type: ignore[arg-type]
    )


def _materialize_getorganelle_prefix(tmp_path: Path) -> Path:
    """A temporary already-installed provider prefix exposing every declared
    GetOrganelle executable under ``bin/`` with safe ownership and permissions."""
    prefix = tmp_path / "existing-getorganelle"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    os.chmod(str(prefix), 0o755)
    os.chmod(str(bin_dir), 0o755)
    for name in _GETORGANELLE_EXECUTABLES:
        exe = bin_dir / name
        exe.write_text("#!/bin/sh\necho hi\n")
        os.chmod(str(exe), 0o755)
    return prefix


class _GetOrganelleProbeSandbox:
    """Fake read-only, networkless sandbox that emits a configurable version line."""

    read_only: bool = True
    network_disabled: bool = True

    def __init__(self, output: str) -> None:
        self._output = output

    def __call__(
        self, executable: Path, argv: tuple[str, ...], *, timeout: float, max_output: int
    ) -> tuple[int, str, str]:
        return (0, self._output, "")


def _empty_str_list(_name: str) -> list[Path]:
    return []


def _no_registry_dict(_bid: str) -> dict[str, object] | None:
    return None


def _fake_getorganelle_hash(_path: Path) -> str:
    return "a" * 64


def _managed_getorganelle(bid: str) -> object | None:
    # A managed fallback exists so the existing-first selection can be proved
    # against a real managed plan, not against an empty discovery sequence.
    return GETORGANELLE_ENVIRONMENT if bid == "getorganelle" else None


def _getorganelle_platform() -> str:
    return GETORGANELLE_PLATFORM


def _getorganelle_resolver(tmp_path: Path, *, probe_output: str) -> EnvironmentResolver:
    prefix = _materialize_getorganelle_prefix(tmp_path)

    def _find_conda() -> list[Path]:
        return [prefix / "bin"]

    return EnvironmentResolver(
        _find_conda=_find_conda,
        _find_in_path=_empty_str_list,
        _registry_lookup=_no_registry_dict,
        _managed_spec=_managed_getorganelle,
        _hash_path=_fake_getorganelle_hash,
        _probe_sandbox=_GetOrganelleProbeSandbox(probe_output),
        _default_platform=_getorganelle_platform,
    )


def test_existing_getorganelle_provider_is_verified_before_managed_plan(tmp_path: Path) -> None:
    """An already-installed provider whose probe emits exactly
    ``GetOrganelle v1.7.7.1`` is verified and selected before a managed install
    plan, even though a managed fallback is available."""
    resolver = _getorganelle_resolver(tmp_path, probe_output="GetOrganelle v1.7.7.1\n")
    contract = runtime_capabilities(_getorganelle_runtime(), {})
    resolved = resolve_backend_version("getorganelle", "tested")
    result = resolver.resolve(
        request=_illumina_request(tmp_path, environment_source="auto"),
        capability_contract=contract,
        resolved_version=resolved,
        effective_parameters={},
    )
    assert result.selected_provider is not None
    assert result.managed_plan is None
    primary = result.selected_provider.components[0]
    assert primary.role == "getorganelle"
    assert primary.version == "1.7.7.1"


def test_getorganelle_provider_version_mismatch_fails_closed(tmp_path: Path) -> None:
    """A provider probing to a different four-part release (1.7.7.0) is never
    trusted against the resolved 1.7.7.1 identity and fails closed."""
    resolver = _getorganelle_resolver(tmp_path, probe_output="GetOrganelle v1.7.7.0\n")
    contract = runtime_capabilities(_getorganelle_runtime(), {})
    resolved = resolve_backend_version("getorganelle", "tested")
    with pytest.raises(OrganelleDependencyError):
        resolver.resolve(
            request=_illumina_request(tmp_path, environment_source="existing"),
            capability_contract=contract,
            resolved_version=resolved,
            effective_parameters={},
        )


@pytest.mark.parametrize(
    "probe_output",
    [
        "GetOrganelle v1.7.7.1-rc1\n",  # prerelease is not a strict four-part release
        "GetOrganelle v1.7.7\n",  # three-part output is unparseable as four-part
        "get_organelle_from_reads.py: unknown version\n",  # no version token
    ],
)
def test_getorganelle_unparseable_probe_output_fails_closed(
    tmp_path: Path, probe_output: str
) -> None:
    """Probe output that cannot be parsed as a strict canonical four-part release
    fails closed; the provider is rejected rather than trusted."""
    resolver = _getorganelle_resolver(tmp_path, probe_output=probe_output)
    contract = runtime_capabilities(_getorganelle_runtime(), {})
    resolved = resolve_backend_version("getorganelle", "tested")
    with pytest.raises(OrganelleDependencyError):
        resolver.resolve(
            request=_illumina_request(tmp_path, environment_source="existing"),
            capability_contract=contract,
            resolved_version=resolved,
            effective_parameters={},
        )
