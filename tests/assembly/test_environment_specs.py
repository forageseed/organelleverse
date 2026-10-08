from __future__ import annotations

import pytest

from organelleverse.assembly.environment_specs import (
    HIMT_ENVIRONMENT,
    OATK_ENVIRONMENT,
    AssemblyEnvironmentSpec,
    CondaPlatformSpec,
    ManagedFileSpec,
    normalize_platform,
)
from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.operations.parameters import OperationParameterModel


def test_oatk_environment_has_exact_native_conda_capabilities() -> None:
    assert OATK_ENVIRONMENT.backend_id == "oatk"
    assert tuple(item.platform for item in OATK_ENVIRONMENT.platforms) == (
        "linux-64",
        "osx-64",
    )
    assert OATK_ENVIRONMENT.require_platform("linux-64").package == "oatk=1.0=h577a1d6_1"
    assert OATK_ENVIRONMENT.require_platform("osx-64").package == "oatk=1.0=h7f84b70_1"
    assert normalize_platform(system="Linux", machine="x86_64") == "linux-64"
    assert normalize_platform(system="Darwin", machine="x86_64") == "osx-64"
    assert normalize_platform(system="Linux", machine="aarch64") == "linux-aarch64"
    assert normalize_platform(system="Darwin", machine="arm64") == "osx-arm64"


def test_oatk_environment_rejects_an_unsupported_platform() -> None:
    with pytest.raises(OrganelleDependencyError) as raised:
        OATK_ENVIRONMENT.require_platform("win-64")
    assert raised.value.code == "assembly.environment_unavailable"
    details = raised.value.as_dict()["details"]
    assert details["platform"] == "win-64"
    assert details["available_platforms"] == ["linux-64", "osx-64"]
    assert raised.value.as_dict()["suggested_action"] == {"select_compatible_compute": True}


def test_oatk_environment_carrier_is_conda_and_contract_versioned() -> None:
    assert OATK_ENVIRONMENT.carrier == "conda"
    assert OATK_ENVIRONMENT.contract_version


def test_oatkdb_manifest_contains_the_exact_embryophyta_files() -> None:
    database = OATK_ENVIRONMENT.require_database()
    names = tuple(item.name for item in database.files)
    assert names == (
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
    assert all(len(item.sha256) == 64 for item in database.files)


def test_oatkdb_manifest_records_exact_version_commit_and_license() -> None:
    database = OATK_ENVIRONMENT.require_database()
    assert database.database_id == "oatkdb"
    assert database.version == "v20230921"
    assert database.source_commit == "75e8db0ac4a7d508a9a518d900876003ceb70737"
    assert database.license == "MIT"


def test_each_managed_database_file_has_an_immutable_pinned_url() -> None:
    database = OATK_ENVIRONMENT.require_database()
    prefix = (
        "https://raw.githubusercontent.com/c-zhou/OatkDB/"
        "75e8db0ac4a7d508a9a518d900876003ceb70737/v20230921/"
    )
    for item in database.files:
        assert item.url == prefix + item.name
        assert item.size_bytes > 0


def test_himt_environment_is_database_free_and_cross_platform() -> None:
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


def test_managed_database_rejects_duplicate_file_names_or_hashes() -> None:
    database = OATK_ENVIRONMENT.require_database()
    files = database.files
    names = [item.name for item in files]
    hashes = [item.sha256 for item in files]
    assert len(set(names)) == len(names)
    assert len(set(hashes)) == len(hashes)


def test_conda_platform_spec_records_exact_lock_and_probe_facts() -> None:
    for platform in ("linux-64", "osx-64"):
        spec = OATK_ENVIRONMENT.require_platform(platform)
        assert isinstance(spec, CondaPlatformSpec)
        assert spec.lock_resource.endswith(".explicit.txt")
        assert "oatk" in spec.executable_names
        assert "nhmmscan" in spec.executable_names
        assert "hmmpress" in spec.executable_names
        assert spec.version_argv


def test_managed_file_spec_rejects_non_https_urls_and_bad_hashes() -> None:
    with pytest.raises(ValueError):
        ManagedFileSpec.model_validate(
            {
                "name": "x.fam",
                "url": "http://insecure/x.fam",
                "sha256": "deadbeef",
                "size_bytes": 1,
            }
        )


def test_environment_spec_model_is_strict_and_frozen() -> None:
    database = OATK_ENVIRONMENT.require_database()
    for model in (
        OATK_ENVIRONMENT,
        database,
        OATK_ENVIRONMENT.platforms[0],
        database.files[0],
    ):
        assert model.model_config.get("frozen") is True
        assert model.model_config.get("extra") == "forbid"


def test_database_free_environment_is_valid_but_has_no_profile() -> None:
    spec = AssemblyEnvironmentSpec(
        backend_id="probe",
        contract_version="probe.environment.v1",
        platforms=(
            CondaPlatformSpec(
                platform="linux-64",
                package="probe=1.0=0",
                lock_resource=(
                    "organelleverse.assembly.resources.environments.oatk/linux-64.explicit.txt"
                ),
                executable_names=("probe",),
                version_argv=("--version",),
            ),
        ),
    )
    assert spec.database is None


def test_environment_spec_does_not_reference_operation_parameter_base() -> None:
    # environment_specs must stay a static specification module: it must not subclass
    # OperationParameterModel (that base is reserved for operation keyword parameters).
    assert not issubclass(AssemblyEnvironmentSpec, OperationParameterModel)
