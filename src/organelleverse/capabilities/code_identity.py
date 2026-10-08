"""Fail-closed identity for bundle-local pure-Python implementation trees."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.spec import StrictSpecModel

from .hashing import hash_entries

_HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"
_KIND = "bundle-local-python-v1"
_WORKER_PROTOCOL = "organelleverse.bundle-worker.v1"
_BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})
_NATIVE_SUFFIXES = frozenset({".so", ".pyd", ".dll", ".dylib"})
# ``pip install`` byte-compiles on install, so a bundle that arrived the normal
# way always carries ``code/<pkg>/__pycache__/``. Its content is derived, not
# authored, and varies with interpreter version and install method, so it is
# skipped entirely: it never reaches the bytecode rule below and never enters
# ``code_tree_hash``. Only the source the hash covers is ever executed, because
# the worker runs with an isolated ``pycache_prefix`` (see ``worker.py``).
CACHE_DIRECTORY = "__pycache__"
MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_TOTAL_BYTES = 8 * 1024 * 1024
MAX_SOURCE_FILES = 256
MAX_SOURCE_CANDIDATES = 512
MAX_SOURCE_PATH_BYTES = 1024
MAX_SOURCE_DEPTH = 32
_SOURCE_READ_CHUNK = 64 * 1024


class ExecutionIdentity(StrictSpecModel):
    """Complete parent-side identity of one bundle-local Python implementation."""

    kind: Literal["bundle-local-python-v1"] = _KIND
    capability_id: str = Field(min_length=1)
    bundle_content_hash: str = Field(pattern=_HASH_PATTERN)
    code_tree_hash: str = Field(pattern=_HASH_PATTERN)
    callable_locator: str = Field(pattern=r"^[a-zA-Z_][\w.]*:[a-zA-Z_]\w*$")
    interpreter: str = Field(pattern=r"^[a-z0-9_]+-[0-9]+\.[0-9]+$")
    worker_protocol: Literal["organelleverse.bundle-worker.v1"] = _WORKER_PROTOCOL
    digest: str = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_canonical_digest(self) -> Self:
        payload = {
            "kind": self.kind,
            "capability_id": self.capability_id,
            "bundle_content_hash": self.bundle_content_hash,
            "code_tree_hash": self.code_tree_hash,
            "callable_locator": self.callable_locator,
            "interpreter": self.interpreter,
            "worker_protocol": self.worker_protocol,
        }
        if self.digest != _canonical_digest(payload):
            raise ValueError("execution identity digest does not match canonical components")
        return self


def _error(
    code: str,
    message: str,
    *,
    bundle_root: Path,
    path: Path | None = None,
    **details: object,
) -> OrganelleContractError:
    payload: dict[str, object] = {"bundle_root": str(bundle_root)}
    if path is not None:
        payload["path"] = str(path)
    payload.update(details)
    return OrganelleContractError(code=code, message=message, details=payload)


def _interpreter() -> str:
    return f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"


def _canonical_digest(payload: dict[str, str]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _safe_code_root(bundle_root: Path) -> tuple[Path, Path]:
    try:
        resolved_bundle = bundle_root.resolve(strict=True)
    except OSError as error:
        raise _error(
            "capability.code_tree_missing",
            "capability bundle root cannot be read while inspecting code",
            bundle_root=bundle_root,
            reason=str(error),
        ) from error
    if not resolved_bundle.is_dir():
        raise _error(
            "capability.code_tree_missing",
            "capability bundle root is not a directory",
            bundle_root=resolved_bundle,
        )
    code_root = resolved_bundle / "code"
    if code_root.is_symlink():
        raise _error(
            "capability.code_path_unsafe",
            "bundle code directory must not be a symlink",
            bundle_root=resolved_bundle,
            path=code_root,
        )
    try:
        resolved_code = code_root.resolve(strict=True)
    except OSError as error:
        raise _error(
            "capability.code_tree_missing",
            "capability bundle has no readable code directory",
            bundle_root=resolved_bundle,
            path=code_root,
            reason=str(error),
        ) from error
    if not resolved_code.is_dir():
        raise _error(
            "capability.code_tree_missing",
            "capability code path is not a directory",
            bundle_root=resolved_bundle,
            path=code_root,
        )
    if not resolved_code.is_relative_to(resolved_bundle):
        raise _error(
            "capability.code_path_unsafe",
            "capability code directory escapes its bundle",
            bundle_root=resolved_bundle,
            path=code_root,
        )
    return resolved_bundle, resolved_code


def _bounded_candidates(bundle_root: Path, code_root: Path) -> list[Path]:
    candidates: list[Path] = []
    try:
        for candidate in code_root.rglob("*"):
            if CACHE_DIRECTORY in candidate.relative_to(code_root).parts:
                continue
            candidates.append(candidate)
            if len(candidates) > MAX_SOURCE_CANDIDATES:
                raise _error(
                    "capability.code_snapshot_too_large",
                    "capability code tree contains too many paths to snapshot safely",
                    bundle_root=bundle_root,
                    path=code_root,
                    max_candidates=MAX_SOURCE_CANDIDATES,
                )
        candidates.sort(key=lambda item: item.as_posix())
    except OrganelleContractError:
        raise
    except (MemoryError, OSError) as error:
        raise _error(
            (
                "capability.code_snapshot_too_large"
                if isinstance(error, MemoryError)
                else "capability.code_read_failed"
            ),
            "capability code tree cannot be enumerated safely",
            bundle_root=bundle_root,
            path=code_root,
            reason=str(error),
        ) from error
    return candidates


def _read_source_bounded(
    path: Path,
    *,
    bundle_root: Path,
) -> bytes:
    content = bytearray()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_SOURCE_READ_CHUNK)
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > MAX_SOURCE_FILE_BYTES:
                    raise _error(
                        "capability.code_source_too_large",
                        "capability Python source exceeds the per-file snapshot bound",
                        bundle_root=bundle_root,
                        path=path,
                        max_bytes=MAX_SOURCE_FILE_BYTES,
                        observed_bytes=len(content),
                    )
    except OrganelleContractError:
        raise
    except MemoryError as error:
        raise _error(
            "capability.code_snapshot_too_large",
            "capability Python source could not be retained within the snapshot bound",
            bundle_root=bundle_root,
            path=path,
            max_bytes=MAX_SOURCE_TOTAL_BYTES,
            reason=str(error),
        ) from error
    except OSError as error:
        raise _error(
            "capability.code_read_failed",
            "capability Python source cannot be read",
            bundle_root=bundle_root,
            path=path,
            reason=str(error),
        ) from error
    try:
        return bytes(content)
    except MemoryError as error:
        raise _error(
            "capability.code_snapshot_too_large",
            "capability Python source could not be frozen within the snapshot bound",
            bundle_root=bundle_root,
            path=path,
            max_bytes=MAX_SOURCE_TOTAL_BYTES,
            reason=str(error),
        ) from error


def _snapshot_sources(bundle_root: Path, code_root: Path) -> tuple[tuple[str, bytes], ...]:
    candidates = _bounded_candidates(bundle_root, code_root)

    directories: list[Path] = []
    sources: list[tuple[str, bytes]] = []
    total_bytes = 0
    for candidate in candidates:
        if candidate.is_symlink():
            raise _error(
                "capability.code_path_unsafe",
                "capability code tree contains a symlink",
                bundle_root=bundle_root,
                path=candidate,
            )
        try:
            resolved = candidate.resolve(strict=True)
            stat_result = resolved.stat()
            mode = stat_result.st_mode
        except OSError as error:
            raise _error(
                "capability.code_read_failed",
                "capability code path cannot be inspected",
                bundle_root=bundle_root,
                path=candidate,
                reason=str(error),
            ) from error
        if not resolved.is_relative_to(code_root):
            raise _error(
                "capability.code_path_unsafe",
                "capability code path escapes its code directory",
                bundle_root=bundle_root,
                path=candidate,
            )
        relative = resolved.relative_to(code_root)
        relative_text = relative.as_posix()
        if (
            len(relative.parts) > MAX_SOURCE_DEPTH
            or len(relative_text.encode("utf-8")) > MAX_SOURCE_PATH_BYTES
        ):
            raise _error(
                "capability.code_path_unsafe",
                "capability code path exceeds the supported depth or length",
                bundle_root=bundle_root,
                path=candidate,
                max_depth=MAX_SOURCE_DEPTH,
                max_path_bytes=MAX_SOURCE_PATH_BYTES,
            )
        if stat.S_ISDIR(mode):
            directories.append(resolved)
            continue
        if not stat.S_ISREG(mode):
            raise _error(
                "capability.code_path_unsafe",
                "capability code tree accepts regular files only",
                bundle_root=bundle_root,
                path=candidate,
            )
        suffix = resolved.suffix
        normalized_suffix = suffix.lower()
        if normalized_suffix in _BYTECODE_SUFFIXES:
            raise _error(
                "capability.code_bytecode_unsupported",
                "capability code tree must not contain Python bytecode",
                bundle_root=bundle_root,
                path=candidate,
            )
        if normalized_suffix in _NATIVE_SUFFIXES:
            raise _error(
                "capability.code_native_extension_unsupported",
                "capability code tree must not contain native libraries",
                bundle_root=bundle_root,
                path=candidate,
            )
        if suffix != ".py":
            raise _error(
                "capability.code_package_data_unsupported",
                "capability code tree accepts Python source files only",
                bundle_root=bundle_root,
                path=candidate,
            )
        if len(sources) >= MAX_SOURCE_FILES:
            raise _error(
                "capability.code_snapshot_too_large",
                "capability code tree contains too many Python source files",
                bundle_root=bundle_root,
                path=code_root,
                max_files=MAX_SOURCE_FILES,
            )
        size = stat_result.st_size
        if size > MAX_SOURCE_FILE_BYTES:
            raise _error(
                "capability.code_source_too_large",
                "capability Python source exceeds the per-file snapshot bound",
                bundle_root=bundle_root,
                path=candidate,
                max_bytes=MAX_SOURCE_FILE_BYTES,
                observed_bytes=size,
            )
        if total_bytes + size > MAX_SOURCE_TOTAL_BYTES:
            raise _error(
                "capability.code_snapshot_too_large",
                "capability Python sources exceed the total snapshot bound",
                bundle_root=bundle_root,
                path=candidate,
                max_bytes=MAX_SOURCE_TOTAL_BYTES,
                observed_bytes=total_bytes + size,
            )
        content = _read_source_bounded(resolved, bundle_root=bundle_root)
        total_bytes += len(content)
        if total_bytes > MAX_SOURCE_TOTAL_BYTES:
            raise _error(
                "capability.code_snapshot_too_large",
                "capability Python sources exceed the total snapshot bound",
                bundle_root=bundle_root,
                path=candidate,
                max_bytes=MAX_SOURCE_TOTAL_BYTES,
                observed_bytes=total_bytes,
            )
        sources.append((relative_text, content))

    try:
        top_level = tuple(sorted(code_root.iterdir(), key=lambda item: item.name))
    except OSError as error:
        raise _error(
            "capability.code_read_failed",
            "capability code directory cannot be read",
            bundle_root=bundle_root,
            path=code_root,
            reason=str(error),
        ) from error
    if (
        len(top_level) != 1
        or top_level[0].is_symlink()
        or not top_level[0].is_dir()
        or not top_level[0].name.isidentifier()
    ):
        raise _error(
            "capability.code_layout_invalid",
            "capability code tree must contain exactly one root package",
            bundle_root=bundle_root,
            path=code_root,
        )

    source_paths = {name for name, _ in sources}
    root_name = top_level[0].name
    if root_name in sys.stdlib_module_names or root_name in sys.builtin_module_names:
        raise _error(
            "capability.code_root_collision",
            "capability private root package collides with the Python runtime",
            bundle_root=bundle_root,
            path=top_level[0],
            root_package=root_name,
        )
    package_directories = (top_level[0].resolve(strict=True), *directories)
    for directory in package_directories:
        relative = directory.relative_to(code_root)
        if any(not part.isidentifier() for part in relative.parts):
            raise _error(
                "capability.code_layout_invalid",
                "capability package directory names must be valid Python identifiers",
                bundle_root=bundle_root,
                path=directory,
            )
        init_path = (relative / "__init__.py").as_posix()
        if init_path not in source_paths:
            raise _error(
                "capability.code_namespace_package_unsupported",
                "namespace packages are not supported in bundle-local code",
                bundle_root=bundle_root,
                path=directory,
            )
    for relative_path in source_paths:
        stem = Path(relative_path).stem
        if stem != "__init__" and not stem.isidentifier():
            raise _error(
                "capability.code_layout_invalid",
                "capability module names must be valid Python identifiers",
                bundle_root=bundle_root,
                path=code_root / relative_path,
            )
    module_paths: dict[str, str] = {}
    for relative_path in sorted(source_paths):
        relative = Path(relative_path)
        if relative.name == "__init__.py":
            fullname = ".".join(relative.parent.parts)
        else:
            fullname = ".".join((*relative.parent.parts, relative.stem))
        previous = module_paths.get(fullname)
        if previous is not None:
            raise _error(
                "capability.code_module_ambiguous",
                "capability code tree maps multiple sources to one module fullname",
                bundle_root=bundle_root,
                path=code_root / relative_path,
                module=fullname,
                paths=[previous, relative_path],
            )
        module_paths[fullname] = relative_path
    return tuple(sources)


def _validate_locator(
    callable_locator: str,
    *,
    bundle_root: Path,
    code_root: Path,
    source_paths: frozenset[str],
) -> None:
    if not callable_locator.strip():
        raise _error(
            "capability.code_locator_missing",
            "bundle-local Python capability has no callable locator",
            bundle_root=bundle_root,
        )
    try:
        module_name, attribute_name = callable_locator.split(":", 1)
    except ValueError as error:
        raise _error(
            "capability.code_locator_invalid",
            "bundle-local callable locator must use module:attribute syntax",
            bundle_root=bundle_root,
            callable_locator=callable_locator,
        ) from error
    module_parts = module_name.split(".")
    if (
        not module_name
        or not attribute_name
        or any(not part.isidentifier() for part in module_parts)
        or not attribute_name.isidentifier()
    ):
        raise _error(
            "capability.code_locator_invalid",
            "bundle-local callable locator is not a valid Python locator",
            bundle_root=bundle_root,
            callable_locator=callable_locator,
        )
    root_package = next(iter(source_paths)).split("/", 1)[0]
    if module_parts[0] != root_package:
        raise _error(
            "capability.code_locator_outside_root",
            "callable locator is outside the verified root package",
            bundle_root=bundle_root,
            callable_locator=callable_locator,
            root_package=root_package,
        )
    module_path = "/".join(module_parts)
    candidates = (f"{module_path}.py", f"{module_path}/__init__.py")
    if not any(candidate in source_paths for candidate in candidates):
        raise _error(
            "capability.code_locator_missing",
            "callable locator module is absent from the verified code tree",
            bundle_root=bundle_root,
            path=code_root,
            callable_locator=callable_locator,
        )


def inspect_bundle_code(
    bundle_root: Path,
    *,
    capability_id: str,
    bundle_content_hash: str,
    callable_locator: str,
) -> ExecutionIdentity:
    """Read and identify one closed bundle-local Python source tree."""
    resolved_bundle, code_root = _safe_code_root(bundle_root)
    try:
        entries = _snapshot_sources(resolved_bundle, code_root)
        source_paths = frozenset(name for name, _ in entries)
        code_tree_hash = hash_entries(entries)
    except MemoryError as error:
        raise _error(
            "capability.code_snapshot_too_large",
            "capability code identity exceeded the bounded snapshot memory budget",
            bundle_root=resolved_bundle,
            path=code_root,
            max_bytes=MAX_SOURCE_TOTAL_BYTES,
            reason=str(error),
        ) from error
    _validate_locator(
        callable_locator,
        bundle_root=resolved_bundle,
        code_root=code_root,
        source_paths=source_paths,
    )
    interpreter = _interpreter()
    payload = {
        "kind": _KIND,
        "capability_id": capability_id,
        "bundle_content_hash": bundle_content_hash,
        "code_tree_hash": code_tree_hash,
        "callable_locator": callable_locator,
        "interpreter": interpreter,
        "worker_protocol": _WORKER_PROTOCOL,
    }
    return ExecutionIdentity(
        kind="bundle-local-python-v1",
        capability_id=capability_id,
        bundle_content_hash=bundle_content_hash,
        code_tree_hash=code_tree_hash,
        callable_locator=callable_locator,
        interpreter=interpreter,
        worker_protocol="organelleverse.bundle-worker.v1",
        digest=_canonical_digest(payload),
    )


__all__ = [
    "MAX_SOURCE_CANDIDATES",
    "MAX_SOURCE_DEPTH",
    "MAX_SOURCE_FILES",
    "MAX_SOURCE_FILE_BYTES",
    "MAX_SOURCE_PATH_BYTES",
    "MAX_SOURCE_TOTAL_BYTES",
    "ExecutionIdentity",
    "inspect_bundle_code",
]
