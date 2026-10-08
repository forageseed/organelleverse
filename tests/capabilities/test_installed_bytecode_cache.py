"""A pip-installed capability carries a ``__pycache__``; that must not break or bind it.

``pip install their-plugin`` byte-compiles on install, so every bundle that
arrives through the standard install path ships ``code/<pkg>/__pycache__/*.pyc``
next to its sources. Two properties have to hold together:

1. discovery must accept such a bundle and derive the *same* identity it would
   derive from the same sources without a cache -- install method is not
   authorship; and
2. only source that ``code_tree_hash`` actually covers may execute, even when a
   poisoned cache sits next to that source with a header CPython trusts.

The second property is what makes the first one safe, so this module proves the
poisoned cache is genuinely live for a plain import before asserting that the
real bundle-local worker still runs the source.
"""

from __future__ import annotations

import importlib.util
import marshal
import py_compile
import struct
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

import organelleverse.capabilities.worker as worker_module
from organelleverse.capabilities.admission import admit_capabilities
from organelleverse.capabilities.code_identity import inspect_bundle_code
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.hashing import hash_bundle
from organelleverse.capabilities.index import (
    CapabilityEntry,
    CapabilityOrigin,
    CapabilityStatus,
)
from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.capabilities.trust import TrustStore, trust
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.capabilities.worker import OneShotBundleWorkerExecutor
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.registry import OperationRegistry

_CAPABILITY_ID = "demo.installed_cache"
_PACKAGE_NAME = "installed_pkg"
_SOURCE_HELPER_VALUE = 10
_POISONED_HELPER_VALUE = 999

_CAPABILITY_TOML = f"""\
schema = "organelleverse.capability.v1"

[capability]
id = "{_CAPABILITY_ID}"
bundle_version = "1.0.0"
implementation = "native"

[contract]
contract_version = "1.0"
title = "Installed bytecode cache demo"
description = "A capability bundle shipped the way pip ships one, byte-compiled on install."
keywords = ["bytecode", "capability", "installed"]
execution_mode = "inline"
stage = "analyze"
input_kind = "none"
output_kind = "result"
callable_locator = "{_PACKAGE_NAME}.impl:run"

[contract.binding]
argument_mode = "named_parameters"
result_codec = "canonical_json"

[[contract.binding.parameters]]
name = "value"
codec = "json"
"""

_IMPLEMENTATION_SOURCE = f"""\
from .helper import helper_value


def run(*, value: int = 1):
    return {{
        "schema_version": "organelleverse.result.v1",
        "kind": "result",
        "operation_id": "{_CAPABILITY_ID}",
        "scope": "none",
        "status": "ok",
        "metrics": {{"value": helper_value + value}},
    }}
"""

_IMPORT_PROBE = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    f"from {_PACKAGE_NAME}.helper import helper_value\n"
    "print(helper_value)\n"
)


def _write_bundle(bundle_root: Path) -> Path:
    """Write one real, zero-fixture bundle and return its ``code/`` directory."""
    bundle_root.mkdir(parents=True)
    (bundle_root / "capability.toml").write_text(_CAPABILITY_TOML, encoding="utf-8")
    package = bundle_root / "code" / _PACKAGE_NAME
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "impl.py").write_text(_IMPLEMENTATION_SOURCE, encoding="utf-8")
    (package / "helper.py").write_text(
        f"helper_value = {_SOURCE_HELPER_VALUE}\n", encoding="utf-8"
    )
    return bundle_root / "code"


def _import_helper_value(code_root: Path, *, write_bytecode: bool) -> int:
    """Import the package in a fresh interpreter and report the value it saw."""
    flags = ["-I", "-S", "-E"] if write_bytecode else ["-I", "-S", "-E", "-B"]
    process = subprocess.run(
        [sys.executable, *flags, "-c", _IMPORT_PROBE, str(code_root)],
        capture_output=True,
        check=True,
        text=True,
    )
    return int(process.stdout.strip())


def _poison_cached_bytecode(source_path: Path, replacement: str) -> Path:
    """Replace one real cached ``.pyc`` with bytecode compiled from other source.

    CPython trusts a timestamp-based cache whose header repeats the source's
    mtime and size, so copying those two fields from the untouched ``.py`` is
    exactly what makes the interpreter execute this cache instead of the source
    the content hash covers.
    """
    cache_path = Path(importlib.util.cache_from_source(str(source_path)))
    assert cache_path.is_file(), "the cache must have been import-warmed for real first"
    status = source_path.stat()
    code = compile(replacement, str(source_path), "exec", dont_inherit=True)
    header = importlib.util.MAGIC_NUMBER + struct.pack(
        "<III", 0, int(status.st_mtime) & 0xFFFFFFFF, status.st_size & 0xFFFFFFFF
    )
    cache_path.write_bytes(header + marshal.dumps(code))
    return cache_path


def _entry(bundle_root: Path) -> CapabilityEntry:
    bundle = parse_capability_bundle(bundle_root / "capability.toml")
    content_hash = hash_bundle(bundle_root)
    return CapabilityEntry(
        capability_id=bundle.capability.id,
        content_hash=content_hash,
        bundle_root=bundle_root,
        bundle=bundle,
        origins=(
            CapabilityOrigin(
                channel="local",
                source_path=str(bundle_root / "capability.toml"),
                search_root=str(bundle_root.parent),
            ),
        ),
        execution_identity=inspect_bundle_code(
            bundle_root,
            capability_id=bundle.capability.id,
            bundle_content_hash=content_hash,
            callable_locator=cast(str, bundle.contract.callable_locator),
        ),
    )


def test_worker_runs_the_hashed_source_not_a_poisoned_bytecode_cache(tmp_path: Path) -> None:
    bundle_root = tmp_path / "demo-installed-cache"
    code_root = _write_bundle(bundle_root)
    helper = code_root / _PACKAGE_NAME / "helper.py"

    # Warm a real cache the way an install does, then poison it in place.
    assert _import_helper_value(code_root, write_bytecode=True) == _SOURCE_HELPER_VALUE
    cache_path = _poison_cached_bytecode(
        helper, f"helper_value = {_POISONED_HELPER_VALUE}\n"
    )
    assert cache_path.parent.name == "__pycache__"
    assert helper.read_text(encoding="utf-8") == f"helper_value = {_SOURCE_HELPER_VALUE}\n"

    # The poison is live: a plain import under the worker's own interpreter
    # flags executes the cache, not the source.
    assert _import_helper_value(code_root, write_bytecode=False) == _POISONED_HELPER_VALUE

    entry = _entry(bundle_root)
    executor = OneShotBundleWorkerExecutor()
    inspection = executor.inspect(entry)
    staging = tmp_path / "staging"
    staging.mkdir()
    result = executor.invoke(
        entry.model_copy(update={"worker_parameters": inspection.parameters}),
        input=None,
        parameters={"value": 2},
        run_id="installed-cache-run",
        staging_root=staging,
    )

    assert isinstance(result.value, dict)
    metrics = result.value["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["value"] == _SOURCE_HELPER_VALUE + 2
    assert helper.read_text(encoding="utf-8") == f"helper_value = {_SOURCE_HELPER_VALUE}\n"


def test_the_launcher_makes_the_worker_process_ignore_in_tree_bytecode_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real public worker launch must neutralise a poisoned in-tree cache.

    ``test_worker_runs_the_hashed_source_not_a_poisoned_bytecode_cache`` cannot
    see this, because the worker resolves the *bundle's* private root through
    its own closed source finder. This test therefore probes what the launcher
    itself configures: it replaces the worker entry point with a script that
    imports through CPython's ordinary machinery and reports what it executed,
    and drives it through the real, public :meth:`inspect` API.
    """
    bundle_root = tmp_path / "demo-installed-cache"
    code_root = _write_bundle(bundle_root)
    helper = code_root / _PACKAGE_NAME / "helper.py"
    assert _import_helper_value(code_root, write_bytecode=True) == _SOURCE_HELPER_VALUE
    _poison_cached_bytecode(helper, f"helper_value = {_POISONED_HELPER_VALUE}\n")
    assert _import_helper_value(code_root, write_bytecode=False) == _POISONED_HELPER_VALUE

    probe = tmp_path / "import_probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(code_root)!r})\n"
        f"from {_PACKAGE_NAME}.helper import helper_value\n"
        'sys.stderr.write(f"OBSERVED={helper_value}\\n")\n'
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(worker_module, "_WORKER_MAIN", probe.resolve())
    with pytest.raises(OrganelleExecutionError) as captured:
        OneShotBundleWorkerExecutor().inspect(_entry(bundle_root))

    assert captured.value.code == "capability.worker_nonzero_exit"
    payload = captured.value.as_dict()
    details = cast(dict[str, object], payload["details"])
    stderr = details.get("stderr")
    assert isinstance(stderr, str)
    assert f"OBSERVED={_SOURCE_HELPER_VALUE}" in stderr


def test_the_worker_leaves_no_bytecode_cache_behind(tmp_path: Path) -> None:
    """The private cache root is created and removed by the launcher itself."""
    bundle_root = tmp_path / "demo-installed-cache"
    _write_bundle(bundle_root)
    entry = _entry(bundle_root)
    created: list[Path] = []
    real_temporary_directory: type[tempfile.TemporaryDirectory[str]] = tempfile.TemporaryDirectory

    def tracking_temporary_directory(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
        *,
        ignore_cleanup_errors: bool = False,
    ) -> tempfile.TemporaryDirectory[str]:
        handle: tempfile.TemporaryDirectory[str] = real_temporary_directory(
            suffix=suffix,
            prefix=prefix,
            dir=dir,
            ignore_cleanup_errors=ignore_cleanup_errors,
        )
        created.append(Path(handle.name))
        return handle

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            worker_module.tempfile, "TemporaryDirectory", tracking_temporary_directory
        )
        OneShotBundleWorkerExecutor().inspect(entry)

    cache_roots = [path for path in created if "pycache" in path.name]
    assert len(cache_roots) == 1
    assert not cache_roots[0].exists()
    assert not any((bundle_root / "code").rglob("__pycache__"))


class _FakeDistribution:
    def __init__(self, name: str, root: Path, files: tuple[PurePosixPath, ...]) -> None:
        self.name = name
        self.root = root
        self.files = files

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

def test_pip_installed_bundle_with_a_bytecode_cache_reaches_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))

    site_packages = tmp_path / "site-packages"
    distribution_package = site_packages / "pear_gc_plugin"
    bundle_root = distribution_package / "bundles" / "demo-installed-cache"
    code_root = _write_bundle(bundle_root)
    (distribution_package / "__init__.py").write_text(
        "raise AssertionError('discovery must never import the plugin')", encoding="utf-8"
    )
    # Exactly what `pip install` leaves behind: byte-compiled sources.
    for source in sorted((code_root / _PACKAGE_NAME).glob("*.py")):
        py_compile.compile(str(source), doraise=True)
    assert sorted(
        path.name for path in (code_root / _PACKAGE_NAME / "__pycache__").iterdir()
    ), "pip-style byte compilation must have produced a real cache"

    files = tuple(
        PurePosixPath(path.relative_to(site_packages).as_posix())
        for path in sorted(site_packages.rglob("*"))
        if path.is_file()
    )
    entry_point = _FakeEntryPoint(
        _FakeDistribution("Pear-GC-Plugin", site_packages, files),
        "pear_gc_plugin:bundles",
    )

    index = discover_capabilities(paths=(), entry_points=(entry_point,))

    described = index.describe(_CAPABILITY_ID)
    assert described.origins[0].channel == "package:pear-gc-plugin"
    assert described.execution_identity is not None
    assert "pear_gc_plugin" not in sys.modules

    verification_store = VerificationStore(tmp_path / "verifications")
    verify_capability(
        _CAPABILITY_ID,
        store=verification_store,
        environment=LocalVerificationEnvironment(index),
    )
    trust_store = TrustStore(tmp_path / "trust.json")
    trust(_CAPABILITY_ID, described.execution_identity, store=trust_store)

    admitted = admit_capabilities(index, store=verification_store)
    admitted_entry = admitted.describe(_CAPABILITY_ID)
    assert admitted_entry.status is CapabilityStatus.ADMITTED, admitted_entry.diagnostic

    registry = OperationRegistry(
        capability_source=admitted.binding_source(
            trust_store=trust_store, executor=OneShotBundleWorkerExecutor()
        )
    )
    result = registry.invoke(_CAPABILITY_ID, input=None, parameters={"value": 5})

    assert isinstance(result, OrganelleResult)
    assert result.metrics["value"] == _SOURCE_HELPER_VALUE + 5
