"""A provider environment finds the backend executable by role and by file name."""

import hashlib
from pathlib import Path

import pytest

from organelleverse.assembly.environment_contracts import ResolvedProvider
from organelleverse.assembly.environments import EnvironmentManager
from organelleverse.core.errors import OrganelleDependencyError


def _component(path: Path, role: str, version: str | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho {role}\n")
    path.chmod(0o755)
    return {
        "role": role,
        "kind": "executable",
        "path": path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "version": version,
    }


def _provider(prefix: Path, *components: dict) -> ResolvedProvider:
    return ResolvedProvider(
        requested_source="auto",
        discovery_source="registry",
        carrier="conda",
        platform="linux-64",
        prefix=prefix,
        capability_contract_digest="sha256:" + "1" * 64,
        components=components,
    )


def test_getorganelle_provider_answers_to_its_script_name(tmp_path):
    bin_dir = tmp_path / "env" / "bin"
    provider = _provider(
        tmp_path / "env",
        _component(bin_dir / "get_organelle_from_reads.py", "getorganelle", "1.7.7.1"),
        _component(bin_dir / "get_organelle_config.py", "get_organelle_config.py"),
        _component(bin_dir / "blastn", "blastn"),
    )
    env = EnvironmentManager(cache_root=tmp_path / "cache").prepare_provider(provider)
    main = bin_dir / "get_organelle_from_reads.py"
    assert env.require_executable("getorganelle") == main
    assert env.require_executable("get_organelle_from_reads.py") == main
    assert env.require_executable("get_organelle_config.py") == bin_dir / "get_organelle_config.py"
    assert env.version == "1.7.7.1"


def test_alias_never_shadows_or_duplicates_a_component(tmp_path):
    bin_dir = tmp_path / "env" / "bin"
    # the backend executable's file name equals its role: no second entry
    same = _provider(tmp_path / "env", _component(bin_dir / "oatk", "oatk", "1.0"))
    env = EnvironmentManager(cache_root=tmp_path / "cache").prepare_provider(same)
    assert [item.name for item in env.executables] == ["oatk"]
    # only the backend's own executable gains an alias
    other = _provider(
        tmp_path / "env2",
        _component(tmp_path / "env2/bin/run_himt", "himt", "1.0"),
        _component(tmp_path / "env2/bin/flye-2.9", "flye"),
    )
    env = EnvironmentManager(cache_root=tmp_path / "cache").prepare_provider(other)
    assert sorted(item.name for item in env.executables) == ["flye", "himt", "run_himt"]
    with pytest.raises(OrganelleDependencyError):
        env.require_executable("flye-2.9")
