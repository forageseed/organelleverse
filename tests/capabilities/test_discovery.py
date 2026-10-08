from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Never

import pytest

import organelleverse.capabilities.discovery as discovery_module
from organelleverse.capabilities.discovery import (
    discover_capabilities,
    discover_capability_candidates,
)
from organelleverse.capabilities.models import CapabilityBundle


def _bundle_text(capability_id: str, *, version: str = "1.0.0") -> str:
    return f'''\
schema = "organelleverse.capability.v1"

[capability]
id = "{capability_id}"
bundle_version = "{version}"
implementation = "native"

[contract]
contract_version = "1.0"
title = "Demo"
description = "A deterministic discovery fixture for capability tests."
keywords = ["capability", "demo", "discovery"]
execution_mode = "inline"
stage = "analyze"
input_kind = "none"
output_kind = "result"
organelle_types = ["mitochondrion"]
callable_locator = "thirdparty.impl:run"
side_effects = []
deterministic = true
idempotent = true
cacheable = true
references = []

[contract.binding]
argument_mode = "named_parameters"
result_codec = "canonical"

[contract.fallback]
allowed = false
'''


def _write_bundle(root: Path, capability_id: str, *, version: str = "1.0.0") -> Path:
    bundle_root = root / capability_id.replace(".", "-")
    bundle_root.mkdir(parents=True)
    (bundle_root / "capability.toml").write_text(
        _bundle_text(capability_id, version=version), encoding="utf-8"
    )
    package = bundle_root / "code" / "thirdparty"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "impl.py").write_text("def run():\n    return None\n", encoding="utf-8")
    return bundle_root


def test_candidate_discovery_does_not_run_default_store_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bundle(tmp_path, "thirdparty.candidate")

    def fail_admission(*args: object, **kwargs: object) -> Never:
        raise AssertionError("candidate discovery must not perform admission")

    monkeypatch.setattr(
        "organelleverse.capabilities.admission.admit_capabilities",
        fail_admission,
    )

    candidates = discover_capability_candidates(paths=[tmp_path], entry_points=())

    assert tuple(entry.capability_id for entry in candidates.entries) == ("thirdparty.candidate",)
    assert candidates.entries[0].status.value == "discovered"


@dataclass
class _FakeDistribution:
    name: str
    root: Path
    files: tuple[PurePosixPath, ...]

    def locate_file(self, path: PurePosixPath) -> Path:
        return self.root / Path(*path.parts)


class _FakeEntryPoint:
    def __init__(self, distribution: _FakeDistribution, value: str) -> None:
        self.name = "capabilities"
        self.group = "organelleverse.capabilities"
        self.value = value
        self._dist = distribution

    @property
    def dist(self) -> _FakeDistribution:
        return self._dist

    def load(self) -> Never:
        raise AssertionError("discovery must never call EntryPoint.load()")


def test_explicit_directory_discovery_never_imports_implementation_or_runs_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bundle(tmp_path, "thirdparty.demo")
    before = set(sys.modules)

    def fail_subprocess(*args: object, **kwargs: object) -> Never:
        raise AssertionError("discovery must never start a subprocess")

    def fail_network(*args: object, **kwargs: object) -> Never:
        raise AssertionError("discovery must never open a network connection")

    def fail_import(*args: object, **kwargs: object) -> Never:
        raise AssertionError("discovery must never import an implementation")

    monkeypatch.setattr(importlib, "import_module", fail_import)
    monkeypatch.setattr(subprocess, "run", fail_subprocess)
    monkeypatch.setattr(subprocess, "Popen", fail_subprocess)
    monkeypatch.setattr(socket, "socket", fail_network)
    index = discover_capabilities(paths=(tmp_path,), entry_points=())

    assert "thirdparty.impl" not in sys.modules
    assert "thirdparty.impl" not in set(sys.modules) - before
    description = index.describe("thirdparty.demo")
    assert description.capability_id == "thirdparty.demo"
    assert description.execution_identity is not None


def test_same_id_and_hash_deduplicates_and_retains_every_origin(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_bundle(first, "thirdparty.demo")
    source = first / "thirdparty-demo" / "capability.toml"
    target_root = second / "thirdparty-demo"
    target_root.mkdir(parents=True)
    (target_root / "capability.toml").write_bytes(source.read_bytes())
    target_package = target_root / "code" / "thirdparty"
    target_package.mkdir(parents=True)
    for source_file in sorted((source.parent / "code" / "thirdparty").glob("*.py")):
        (target_package / source_file.name).write_bytes(source_file.read_bytes())

    index = discover_capabilities(paths=(second, first), entry_points=())
    description = index.describe("thirdparty.demo")

    assert len(description.origins) == 2
    assert tuple(origin.source_path for origin in description.origins) == tuple(
        sorted(origin.source_path for origin in description.origins)
    )
    assert index.conflicts() == ()


def test_same_hash_core_and_local_native_selects_local_identity_regardless_of_order(
    tmp_path: Path,
) -> None:
    core_root = tmp_path / "a-core"
    local_root = tmp_path / "z-local"
    _write_bundle(core_root, "thirdparty.demo")
    local_bundle = _write_bundle(local_root, "thirdparty.demo")
    raw, diagnostics = discovery_module._scan_roots(  # pyright: ignore[reportPrivateUsage]
        (
            discovery_module.SearchRoot("core", core_root),
            discovery_module.SearchRoot("local", local_root),
        )
    )
    assert diagnostics == []

    forward, forward_conflicts, _ = discovery_module._deduplicate(  # pyright: ignore[reportPrivateUsage]
        raw
    )
    reverse, reverse_conflicts, _ = discovery_module._deduplicate(  # pyright: ignore[reportPrivateUsage]
        reversed(raw)
    )

    assert forward == reverse
    assert forward_conflicts == reverse_conflicts == ()
    assert len(forward) == 1
    entry = forward[0]
    assert tuple(origin.channel for origin in entry.origins) == ("core", "local")
    assert entry.execution_identity is not None
    assert entry.bundle_root == local_bundle.resolve()


def test_missing_native_locator_is_retained_as_a_discovery_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = _write_bundle(tmp_path, "thirdparty.demo")
    bundle = discovery_module.parse_capability_bundle(bundle_root / "capability.toml")
    contract_values = dict(vars(bundle.contract))
    contract_values["callable_locator"] = None
    contract = type(bundle.contract).model_construct(**contract_values)
    bundle_values = dict(vars(bundle))
    bundle_values["contract"] = contract
    forged_bundle = type(bundle).model_construct(**bundle_values)

    def parse_with_invariant_drift(_path: Path) -> CapabilityBundle:
        return forged_bundle

    monkeypatch.setattr(discovery_module, "parse_capability_bundle", parse_with_invariant_drift)

    index = discover_capabilities(paths=(tmp_path,), entry_points=())

    assert index.entries == ()
    diagnostic = next(
        item for item in index.diagnostics() if item.code == "capability.native_locator_missing"
    )
    assert diagnostic.capability_id == "thirdparty.demo"
    assert diagnostic.origins[0].channel == "local"


def test_missing_dedup_execution_identity_is_retained_as_a_diagnostic(
    tmp_path: Path,
) -> None:
    core_root = tmp_path / "core"
    local_root = tmp_path / "local"
    _write_bundle(core_root, "thirdparty.demo")
    _write_bundle(local_root, "thirdparty.demo")
    raw, scan_diagnostics = discovery_module._scan_roots(  # pyright: ignore[reportPrivateUsage]
        (
            discovery_module.SearchRoot("core", core_root),
            discovery_module.SearchRoot("local", local_root),
        )
    )
    assert scan_diagnostics == []
    core = next(item for item in raw if item.origins[0].channel == "core")
    local = next(item for item in raw if item.origins[0].channel == "local")
    local_values = dict(vars(local))
    local_values["execution_identity"] = None
    local_without_identity = type(local).model_construct(**local_values)

    entries, conflicts, diagnostics = discovery_module._deduplicate(  # pyright: ignore[reportPrivateUsage]
        (core, local_without_identity)
    )

    assert entries == ()
    assert conflicts == ()
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code == "capability.execution_identity_missing"
    assert diagnostic.capability_id == "thirdparty.demo"
    assert tuple(origin.channel for origin in diagnostic.origins) == ("core", "local")


def test_same_id_and_different_hash_excludes_every_variant_regardless_of_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_bundle(first, "thirdparty.demo", version="1.0.0")
    _write_bundle(second, "thirdparty.demo", version="2.0.0")

    forward = discover_capabilities(paths=(first, second), entry_points=())
    reverse = discover_capabilities(paths=(second, first), entry_points=())

    assert forward.conflicts() == reverse.conflicts()
    conflict = forward.conflicts()[0]
    assert conflict.capability_id == "thirdparty.demo"
    assert len(conflict.members) == 2
    assert len({member.content_hash for member in conflict.members}) == 2
    assert forward.is_conflicted("thirdparty.demo") is True
    assert forward.list() == ()


def test_malformed_bundle_is_a_diagnostic_and_does_not_abort_other_candidates(
    tmp_path: Path,
) -> None:
    _write_bundle(tmp_path, "thirdparty.good")
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "capability.toml").write_text("not = [valid", encoding="utf-8")

    index = discover_capabilities(paths=(tmp_path,), entry_points=())

    assert index.describe("thirdparty.good").capability_id == "thirdparty.good"
    assert any(item.code == "capability.bundle_invalid_toml" for item in index.diagnostics())


def test_invalid_bundle_code_is_retained_as_a_discovery_diagnostic(tmp_path: Path) -> None:
    bundle_root = _write_bundle(tmp_path, "thirdparty.invalid")
    for path in sorted((bundle_root / "code").rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    (bundle_root / "code").rmdir()

    index = discover_capabilities(paths=(tmp_path,), entry_points=())

    assert index.entries == ()
    diagnostic = next(
        item for item in index.diagnostics() if item.code == "capability.code_tree_missing"
    )
    assert diagnostic.capability_id == "thirdparty.invalid"


def test_entry_point_is_resolved_from_distribution_files_without_load_or_import(
    tmp_path: Path,
) -> None:
    package = tmp_path / "thirdparty_pkg"
    capability_root = package / "bundles" / "thirdparty-demo"
    capability_root.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise AssertionError('must not import')", encoding="utf-8"
    )
    (capability_root / "capability.toml").write_text(
        _bundle_text("thirdparty.demo"), encoding="utf-8"
    )
    implementation = capability_root / "code" / "thirdparty"
    implementation.mkdir(parents=True)
    (implementation / "__init__.py").write_text("", encoding="utf-8")
    (implementation / "impl.py").write_text("def run():\n    return None\n", encoding="utf-8")
    files = tuple(
        PurePosixPath(path.relative_to(tmp_path).as_posix())
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    )
    distribution = _FakeDistribution("ThirdParty-Pkg", tmp_path, files)
    entry_point = _FakeEntryPoint(distribution, "thirdparty_pkg:bundles")

    index = discover_capabilities(paths=(), entry_points=(entry_point,))

    description = index.describe("thirdparty.demo")
    assert description.origins[0].channel == "package:thirdparty-pkg"
    assert "thirdparty_pkg" not in sys.modules


def test_empty_environment_path_elements_never_scan_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_bundle(tmp_path, "thirdparty.cwd")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "ORGANELLEVERSE_CAPABILITY_PATH", f"{tmp_path / 'missing'}{__import__('os').pathsep}"
    )

    def no_entry_points(**kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr(importlib.metadata, "entry_points", no_entry_points)

    index = discover_capabilities()

    assert all(item.capability_id != "thirdparty.cwd" for item in index.entries)


def test_standard_directory_channels_are_all_discovered_with_fixed_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "installed" / "organelleverse"
    user_home = tmp_path / "home"
    project = tmp_path / "project"
    extra = tmp_path / "extra"
    _write_bundle(package / "capabilities", "channel.core")
    _write_bundle(user_home / "capabilities", "channel.user")
    _write_bundle(project / ".organelleverse" / "capabilities", "channel.project")
    _write_bundle(extra, "channel.extra")

    def packaged_files(package_name: str) -> Path:
        return package

    def no_entry_points(**kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr(importlib.resources, "files", packaged_files)
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(user_home))
    monkeypatch.setenv("ORGANELLEVERSE_CAPABILITY_PATH", str(extra))
    monkeypatch.setattr(importlib.metadata, "entry_points", no_entry_points)
    monkeypatch.chdir(project)

    index = discover_capabilities()

    channels = {entry.capability_id: entry.origins[0].channel for entry in index.entries}
    assert channels == {
        "channel.core": "core",
        "channel.extra": "local",
        "channel.project": "project",
        "channel.user": "local",
    }


def test_extra_paths_extend_the_standard_channels_instead_of_replacing_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The desktop wiring: user-installed plugin sources ride on top of the
    standard channels, and a CLI boot with zero installed plugins still sees
    the package-resident core capabilities (the production bug this locks in:
    ``paths=()`` means "scan nothing", which used to leave the whole plugin,
    capability, and conversation-tool surface empty)."""
    package = tmp_path / "installed" / "organelleverse"
    user_home = tmp_path / "home"
    project = tmp_path / "project"
    sources = tmp_path / "plugin-sources"
    _write_bundle(package / "capabilities", "channel.core")
    _write_bundle(project / ".organelleverse" / "capabilities", "channel.project")
    _write_bundle(sources, "channel.installed")

    def packaged_files(package_name: str) -> Path:
        return package

    def no_entry_points(**kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr(importlib.resources, "files", packaged_files)
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(user_home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.setattr(importlib.metadata, "entry_points", no_entry_points)
    monkeypatch.chdir(project)

    extended = discover_capabilities(extra_paths=(sources,))
    assert {entry.capability_id for entry in extended.entries} == {
        "channel.core",
        "channel.project",
        "channel.installed",
    }

    empty = discover_capabilities(extra_paths=())
    assert {entry.capability_id for entry in empty.entries} == {
        "channel.core",
        "channel.project",
    }


def test_discovery_rejects_symlinked_bundle_roots_as_diagnostics(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write_bundle(outside, "thirdparty.escape")
    scanned = tmp_path / "scanned"
    scanned.mkdir()
    (scanned / "linked").symlink_to(outside / "thirdparty-escape", target_is_directory=True)

    index = discover_capabilities(paths=(scanned,), entry_points=())

    assert index.entries == ()
    assert any(item.code == "capability.bundle_path_unsafe" for item in index.diagnostics())


def test_discovery_rejects_a_search_root_that_is_itself_a_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside-root"
    _write_bundle(outside, "thirdparty.escape")
    linked = tmp_path / "linked-root"
    linked.symlink_to(outside, target_is_directory=True)

    index = discover_capabilities(paths=(linked,), entry_points=())

    assert index.entries == ()
    assert any(item.code == "capability.bundle_path_unsafe" for item in index.diagnostics())


def test_entry_point_resource_directory_cannot_escape_its_distribution(tmp_path: Path) -> None:
    distribution = _FakeDistribution(
        "ThirdParty-Pkg",
        tmp_path,
        (PurePosixPath("thirdparty_pkg/__init__.py"),),
    )
    entry_point = _FakeEntryPoint(distribution, "thirdparty_pkg:../../outside")

    index = discover_capabilities(paths=(), entry_points=(entry_point,))

    assert index.entries == ()
    assert any(item.code == "capability.entry_point_invalid" for item in index.diagnostics())
