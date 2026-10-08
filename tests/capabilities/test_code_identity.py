from __future__ import annotations

import py_compile
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from organelleverse.capabilities.code_identity import ExecutionIdentity, inspect_bundle_code
from organelleverse.capabilities.hashing import hash_entries
from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.spec import ResultCodec

_BUNDLE_HASH = "sha256:" + "1" * 64


def _write_package(bundle_root: Path) -> Path:
    package = bundle_root / "code" / "private_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from .impl import run\n", encoding="utf-8")
    (package / "impl.py").write_text(
        "from .helper import value\ndef run():\n    return value()\n",
        encoding="utf-8",
    )
    (package / "helper.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    return package


def _inspect(bundle_root: Path, *, locator: str = "private_pkg.impl:run"):
    return inspect_bundle_code(
        bundle_root,
        capability_id="thirdparty.demo",
        bundle_content_hash=_BUNDLE_HASH,
        callable_locator=locator,
    )


def test_hash_entries_is_deterministic_regardless_of_input_order() -> None:
    entries = (("private_pkg/z.py", b"z"), ("private_pkg/a.py", b"a"))

    assert hash_entries(entries) == hash_entries(reversed(entries))


@pytest.mark.parametrize(
    "locator",
    ("private_pkg:run", "private_pkg.impl:run"),
)
def test_locator_resolves_to_a_verified_package_or_module(tmp_path: Path, locator: str) -> None:
    _write_package(tmp_path)

    identity = _inspect(tmp_path, locator=locator)

    assert identity.callable_locator == locator
    assert identity.kind == "bundle-local-python-v1"
    assert identity.worker_protocol == "organelleverse.bundle-worker.v1"
    assert (
        identity.interpreter
        == f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"
    )


def test_changed_private_helper_invalidates_code_and_execution_identity(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    before = _inspect(tmp_path)

    (package / "helper.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    after = _inspect(tmp_path)

    assert after.code_tree_hash != before.code_tree_hash
    assert after.digest != before.digest


def test_execution_identity_rejects_a_forged_canonical_digest(tmp_path: Path) -> None:
    _write_package(tmp_path)
    payload = _inspect(tmp_path).model_dump(mode="python")
    payload["digest"] = "sha256:" + "f" * 64

    with pytest.raises(ValueError, match="digest does not match"):
        ExecutionIdentity.model_validate(payload)


def test_manifest_locator_and_interpreter_minor_each_change_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_package(tmp_path)
    baseline = _inspect(tmp_path)
    changed_manifest = inspect_bundle_code(
        tmp_path,
        capability_id="thirdparty.demo",
        bundle_content_hash="sha256:" + "2" * 64,
        callable_locator="private_pkg.impl:run",
    )
    changed_locator = _inspect(tmp_path, locator="private_pkg:run")
    monkeypatch.setattr(
        sys,
        "version_info",
        SimpleNamespace(major=sys.version_info.major, minor=sys.version_info.minor + 1),
    )
    changed_interpreter = _inspect(tmp_path)

    assert (
        len(
            {
                baseline.digest,
                changed_manifest.digest,
                changed_locator.digest,
                changed_interpreter.digest,
            }
        )
        == 4
    )
    assert baseline.code_tree_hash == changed_manifest.code_tree_hash
    assert baseline.code_tree_hash == changed_locator.code_tree_hash
    assert baseline.code_tree_hash == changed_interpreter.code_tree_hash


def test_code_tree_requires_exactly_one_root_package(tmp_path: Path) -> None:
    _write_package(tmp_path)
    second = tmp_path / "code" / "other_pkg"
    second.mkdir()
    (second / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_layout_invalid"


def test_missing_code_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_tree_missing"


def test_missing_locator_module_is_rejected(tmp_path: Path) -> None:
    _write_package(tmp_path)

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path, locator="private_pkg.missing:run")

    assert captured.value.code == "capability.code_locator_missing"


def test_symlinked_source_is_rejected(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("def surprise(): pass\n", encoding="utf-8")
    (package / "linked.py").symlink_to(outside)

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_path_unsafe"


def test_code_directory_cannot_escape_through_a_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write_package(outside)
    (tmp_path / "code").symlink_to(outside / "code", target_is_directory=True)

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_path_unsafe"


def test_namespace_package_is_rejected(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    (package / "__init__.py").unlink()

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_namespace_package_unsupported"


@pytest.mark.parametrize("suffix", (".pyc", ".pyo"))
def test_bytecode_is_rejected(tmp_path: Path, suffix: str) -> None:
    package = _write_package(tmp_path)
    (package / f"cached{suffix}").write_bytes(b"bytecode")

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_bytecode_unsupported"


def _byte_compile(package: Path) -> Path:
    """Byte-compile every module the way ``pip install`` does on install."""
    cache = package / "__pycache__"
    for source in sorted(package.glob("*.py")):
        py_compile.compile(str(source), doraise=True)
    assert sorted(path.suffix for path in cache.iterdir()) == [".pyc", ".pyc", ".pyc"]
    return cache


def test_installed_bytecode_cache_keeps_the_identity_of_the_same_sources(tmp_path: Path) -> None:
    """``pip install`` byte-compiles on install; that must not change identity."""
    sdist = tmp_path / "sdist"
    installed = tmp_path / "installed"
    _write_package(sdist)
    _byte_compile(_write_package(installed))

    from_source = _inspect(sdist)
    from_wheel = _inspect(installed)

    assert from_wheel.code_tree_hash == from_source.code_tree_hash
    assert from_wheel.digest == from_source.digest


def test_bytecode_outside_a_pycache_directory_is_still_rejected(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    cache = _byte_compile(package)
    compiled = min(cache.glob("*.pyc"), key=lambda path: path.name)
    (package / "sourceless.pyc").write_bytes(compiled.read_bytes())

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_bytecode_unsupported"
    assert str(captured.value.details["path"]).endswith("sourceless.pyc")


@pytest.mark.parametrize("suffix", (".so", ".pyd", ".dll", ".dylib"))
def test_native_extension_is_rejected(tmp_path: Path, suffix: str) -> None:
    package = _write_package(tmp_path)
    (package / f"native{suffix}").write_bytes(b"native")

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_native_extension_unsupported"


def test_non_python_package_data_is_rejected(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    (package / "data.json").write_text("{}", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_package_data_unsupported"


def test_uppercase_python_suffix_is_rejected_by_parent_discovery(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    (package / "extra.PY").write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_package_data_unsupported"


def test_canonical_json_result_codec_is_a_distinct_schema_member() -> None:
    assert ResultCodec.CANONICAL_JSON.value == "canonical_json"
    assert ResultCodec.CANONICAL.value == "canonical"


def test_source_snapshot_rejects_one_file_over_the_explicit_bound(tmp_path: Path) -> None:
    from organelleverse.capabilities.code_identity import MAX_SOURCE_FILE_BYTES

    package = _write_package(tmp_path)
    (package / "large.py").write_bytes(b"#" * (MAX_SOURCE_FILE_BYTES + 1))

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_source_too_large"
    details = captured.value.as_dict()["details"]
    assert details["max_bytes"] == MAX_SOURCE_FILE_BYTES


def test_source_snapshot_rejects_total_bytes_over_the_explicit_bound(tmp_path: Path) -> None:
    from organelleverse.capabilities.code_identity import (
        MAX_SOURCE_FILE_BYTES,
        MAX_SOURCE_TOTAL_BYTES,
    )

    package = _write_package(tmp_path)
    remaining = MAX_SOURCE_TOTAL_BYTES
    index = 0
    while remaining >= 0:
        size = min(MAX_SOURCE_FILE_BYTES, remaining + 1)
        (package / f"part_{index}.py").write_bytes(b"#" * size)
        remaining -= size
        index += 1

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_snapshot_too_large"
    assert captured.value.as_dict()["details"]["max_bytes"] == MAX_SOURCE_TOTAL_BYTES


def test_source_snapshot_rejects_ambiguous_module_fullnames(tmp_path: Path) -> None:
    package = _write_package(tmp_path)
    (package / "ambiguous.py").write_text("value = 1\n", encoding="utf-8")
    ambiguous_package = package / "ambiguous"
    ambiguous_package.mkdir()
    (ambiguous_package / "__init__.py").write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_module_ambiguous"


@pytest.mark.parametrize("root_name", ("json", "os", "importlib"))
def test_private_root_cannot_collide_with_stdlib_or_loaded_module(
    tmp_path: Path, root_name: str
) -> None:
    package = tmp_path / "code" / root_name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "impl.py").write_text("def run():\n    return 1\n", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        inspect_bundle_code(
            tmp_path,
            capability_id="thirdparty.demo",
            bundle_content_hash=_BUNDLE_HASH,
            callable_locator=f"{root_name}.impl:run",
        )

    assert captured.value.code == "capability.code_root_collision"


def test_source_snapshot_rejects_too_many_candidates(tmp_path: Path) -> None:
    from organelleverse.capabilities.code_identity import MAX_SOURCE_FILES

    package = _write_package(tmp_path)
    for index in range(MAX_SOURCE_FILES):
        (package / f"candidate_{index}.py").write_text("", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_snapshot_too_large"
    assert captured.value.as_dict()["details"]["max_files"] == MAX_SOURCE_FILES


def test_source_snapshot_rejects_a_path_over_the_explicit_depth_bound(
    tmp_path: Path,
) -> None:
    from organelleverse.capabilities.code_identity import MAX_SOURCE_DEPTH

    package = _write_package(tmp_path)
    nested = package
    for index in range(MAX_SOURCE_DEPTH + 1):
        nested /= f"level_{index}"
        nested.mkdir()
        (nested / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_path_unsafe"
    assert captured.value.as_dict()["details"]["max_depth"] == MAX_SOURCE_DEPTH


def test_source_snapshot_rejects_a_path_over_the_explicit_byte_length_bound(
    tmp_path: Path,
) -> None:
    from organelleverse.capabilities.code_identity import MAX_SOURCE_PATH_BYTES

    package = _write_package(tmp_path)
    nested = package
    for index in range(5):
        nested /= f"n{index}_" + ("a" * 205)
        nested.mkdir()
        (nested / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_path_unsafe"
    assert captured.value.as_dict()["details"]["max_path_bytes"] == (MAX_SOURCE_PATH_BYTES)


def test_source_snapshot_translates_memory_error_to_structured_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _write_package(tmp_path)
    target = package / "helper.py"
    real_open = Path.open

    class _MemoryReader:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            raise MemoryError("injected")

    def memory_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if path == target and mode == "rb":
            return _MemoryReader()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", memory_open)

    with pytest.raises(OrganelleContractError) as captured:
        _inspect(tmp_path)

    assert captured.value.code == "capability.code_snapshot_too_large"
