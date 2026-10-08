"""Translate file paths across execution contexts.

A capability declares its ``execution_context`` (host / wsl / container / remote).
When the runtime receives a path value anchored to a *different* context, it
must translate the path before the worker can access the file. Translation is
explicit and fail-closed: if the runtime cannot produce a path that is valid in
the target context, the invocation fails immediately rather than trying to
execute against a missing or wrong file.

Translation results are recorded in provenance so that a run remains auditable
and reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from organelleverse.core.errors import OrganelleInputError
from organelleverse.operations.spec import ExecutionContext


@dataclass(frozen=True)
class TranslatedPath:
    """One path together with its provenance."""

    original_value: str
    original_context: ExecutionContext
    translated_value: str
    target_context: ExecutionContext


def _without_parent_segments(path: str, *, context: str) -> str:
    normalized = path.replace("\\", "/")
    if any(part == ".." for part in normalized.split("/")):
        raise OrganelleInputError(
            code="execution_context.path_traversal",
            message=f"{context} path may not contain parent traversal segments",
            details={"path": path, "context": context},
        )
    return normalized


def _normalize_host_path(path: str) -> str:
    normalized = _without_parent_segments(path.strip(), context="host")
    if _windows_drive_letter(normalized) is not None:
        normalized = normalized.rstrip("/")
        return normalized[0].lower() + normalized[1:]
    if normalized.startswith("/"):
        return normalized.rstrip("/") or "/"
    raise OrganelleInputError(
        code="execution_context.mount_map_invalid",
        message="host mount prefixes must be absolute POSIX or Windows-drive paths",
        details={"path": path},
    )


def _normalize_container_path(path: str) -> str:
    normalized = _without_parent_segments(path.strip(), context="container")
    if not normalized.startswith("/"):
        raise OrganelleInputError(
            code="execution_context.mount_map_invalid",
            message="container mount prefixes must be absolute POSIX paths",
            details={"path": path},
        )
    return normalized.rstrip("/") or "/"


def _host_output_root(path: str) -> str:
    normalized = path.strip()
    if _windows_drive_letter(normalized) is not None:
        if normalized.replace("\\", "/").rstrip("/").endswith(":"):
            return normalized[:2] + "\\"
        return normalized.rstrip("/\\")
    return normalized.rstrip("/") or "/"


def _path_is_within(path: str, root: str) -> bool:
    return path == root or (root == "/" and path.startswith("/")) or path.startswith(f"{root}/")


def _join_posix(root: str, relative: str) -> str:
    if not relative:
        return root
    return f"/{relative}" if root == "/" else f"{root}/{relative}"


def _normalized_mounts(
    mount_map: dict[str, str] | None,
) -> tuple[tuple[str, str, str], ...]:
    if mount_map is None:
        raise OrganelleInputError(
            code="execution_context.translation_failed",
            message="container execution context requires an explicit mount_map",
            details={"target_context": "container"},
        )
    normalized: list[tuple[str, str, str]] = []
    host_roots: set[str] = set()
    container_roots: set[str] = set()
    mount_items = cast(dict[object, object], mount_map).items()
    for raw_host, raw_container in mount_items:
        if not isinstance(raw_host, str) or not isinstance(raw_container, str):
            raise OrganelleInputError(
                code="execution_context.mount_map_invalid",
                message="mount_map keys and values must be strings",
                details={
                    "host_type": type(raw_host).__name__,
                    "container_type": type(raw_container).__name__,
                },
            )
        host = _normalize_host_path(raw_host)
        container = _normalize_container_path(raw_container)
        if host in host_roots or container in container_roots:
            raise OrganelleInputError(
                code="execution_context.mount_map_ambiguous",
                message="mount_map prefixes must be unique in both directions",
                details={"host": raw_host, "container": raw_container},
            )
        host_roots.add(host)
        container_roots.add(container)
        normalized.append((host, _host_output_root(raw_host), container))
    return tuple(normalized)


def _windows_drive_letter(path: str) -> str | None:
    match = re.match(r"^([A-Za-z]):([/\\]|$)", path)
    return match.group(1).lower() if match else None


def _host_to_wsl(path: str) -> str:
    """Translate a host Windows path into a WSL path."""
    drive = _windows_drive_letter(path)
    if drive is None:
        # Unix-style host path; assume it is mountable as-is inside WSL.
        return path
    tail = path[2:].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{tail}"


def _wsl_to_host(path: str) -> str:
    """Translate a WSL path into a host Windows path."""
    match = re.match(r"^/mnt/([a-z])(?:/(.*))?$", path.replace("\\", "/"))
    if match:
        drive = match.group(1).upper()
        tail = (match.group(2) or "").replace("/", "\\")
        return f"{drive}:\\{tail}" if tail else f"{drive}:\\"
    # Pure Unix path not under /mnt: no deterministic Windows equivalent.
    raise OrganelleInputError(
        code="execution_context.translation_failed",
        message=f"WSL path '{path}' is not under /mnt/<drive> and cannot be translated to a host Windows path",
        details={"source_path": path, "source_context": "wsl", "target_context": "host"},
    )


def _host_to_container(
    path: str,
    *,
    mount_map: dict[str, str] | None = None,
) -> str:
    """Translate a host path into a container path using an explicit mount map.

    ``mount_map`` maps host prefixes to container prefixes, e.g.
    ``{"C:\\data": "/data", "/input/data": "/data"}``. The longest matching
    prefix wins.
    """
    mounts = _normalized_mounts(mount_map)
    normalized_path = _normalize_host_path(path)
    candidates = sorted(
        (
            (host, container)
            for host, _raw_host, container in mounts
            if _path_is_within(normalized_path, host)
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    if not candidates:
        raise OrganelleInputError(
            code="execution_context.translation_failed",
            message=f"host path '{path}' is not covered by any container mount_map entry",
            details={"source_path": path, "mount_map": mount_map, "target_context": "container"},
        )
    host_prefix, container_prefix = candidates[0]
    relative = normalized_path[len(host_prefix) :].lstrip("/")
    return _join_posix(container_prefix, relative)


def _container_to_host(
    path: str,
    *,
    mount_map: dict[str, str] | None = None,
) -> str:
    """Translate a container path back to a host path using an explicit mount map."""
    mounts = _normalized_mounts(mount_map)
    normalized_path = _normalize_container_path(path)
    candidates = sorted(
        (
            (container_prefix, host_prefix)
            for _normalized_host, host_prefix, container_prefix in mounts
            if _path_is_within(normalized_path, container_prefix)
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    if not candidates:
        raise OrganelleInputError(
            code="execution_context.translation_failed",
            message=f"container path '{path}' is not covered by any mount_map entry",
            details={"source_path": path, "mount_map": mount_map, "target_context": "host"},
        )
    container_prefix, host_prefix = candidates[0]
    relative = normalized_path[len(container_prefix) :].lstrip("/")
    # Restore the host separator style.
    if _windows_drive_letter(host_prefix) is not None:
        relative = relative.replace("/", "\\")
        if not relative:
            return host_prefix
        clean_host_prefix = host_prefix.rstrip("/\\")
        return f"{clean_host_prefix}\\{relative}"
    if not relative:
        return host_prefix
    return f"/{relative}" if host_prefix == "/" else f"{host_prefix}/{relative}"


def translate_path(
    path: str,
    *,
    source_context: ExecutionContext,
    target_context: ExecutionContext,
    mount_map: dict[str, str] | None = None,
) -> TranslatedPath:
    """Translate *path* from *source_context* to *target_context*.

    Raises ``OrganelleInputError`` if the translation is impossible or ambiguous.
    """
    context_name = source_context.value
    _without_parent_segments(path, context=context_name)
    if source_context == target_context:
        return TranslatedPath(
            original_value=path,
            original_context=source_context,
            translated_value=path,
            target_context=target_context,
        )

    if source_context is ExecutionContext.REMOTE or target_context is ExecutionContext.REMOTE:
        raise OrganelleInputError(
            code="execution_context.remote_not_translatable",
            message="remote execution context paths cannot be translated automatically",
            details={
                "source_path": path,
                "source_context": source_context.value,
                "target_context": target_context.value,
            },
        )

    if source_context is ExecutionContext.HOST and target_context is ExecutionContext.WSL:
        translated = _host_to_wsl(path)
    elif source_context is ExecutionContext.WSL and target_context is ExecutionContext.HOST:
        translated = _wsl_to_host(path)
    elif source_context is ExecutionContext.HOST and target_context is ExecutionContext.CONTAINER:
        translated = _host_to_container(path, mount_map=mount_map)
    elif source_context is ExecutionContext.CONTAINER and target_context is ExecutionContext.HOST:
        translated = _container_to_host(path, mount_map=mount_map)
    elif {source_context, target_context} == {ExecutionContext.WSL, ExecutionContext.CONTAINER}:
        # WSL and container both speak POSIX paths; translate via host as pivot.
        host = translate_path(path, source_context=source_context, target_context=ExecutionContext.HOST)
        return translate_path(
            host.translated_value,
            source_context=ExecutionContext.HOST,
            target_context=target_context,
            mount_map=mount_map,
        )
    else:
        raise OrganelleInputError(
            code="execution_context.translation_unsupported",
            message=f"path translation from {source_context.value} to {target_context.value} is not supported",
            details={
                "source_path": path,
                "source_context": source_context.value,
                "target_context": target_context.value,
            },
        )

    return TranslatedPath(
        original_value=path,
        original_context=source_context,
        translated_value=translated,
        target_context=target_context,
    )


def validate_path_accessible(path: str) -> Path:
    """Resolve *path* and verify it exists and is readable.

    Raises ``OrganelleInputError`` for missing or unreadable paths.
    """
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise OrganelleInputError(
            code="input.missing_artifact",
            message=f"translated path does not exist: {candidate}",
            details={"path": str(candidate)},
        ) from error
    except (OSError, RuntimeError) as error:
        raise OrganelleInputError(
            code="input.unreadable_artifact",
            message=f"translated path could not be resolved: {candidate}",
            details={"path": str(candidate), "reason": str(error)},
        ) from error
    try:
        with resolved.open("rb"):
            pass
    except OSError as error:
        raise OrganelleInputError(
            code="input.unreadable_artifact",
            message=f"translated path is not readable: {resolved}",
            details={"path": str(resolved), "reason": str(error)},
        ) from error
    return resolved


__all__ = ["TranslatedPath", "translate_path", "validate_path_accessible"]
