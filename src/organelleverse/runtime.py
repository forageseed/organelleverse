"""Managed scientific run storage and zero-duplicate artifact promotion."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import warnings
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NoReturn, TypeAlias
from uuid import uuid4

from pydantic import ValidationError

from .core.artifacts import ArtifactRef
from .core.data import OrganelleData
from .core.errors import OrganelleExecutionError, OrganelleInputError
from .core.frozen import FrozenMap
from .core.genome import OrganelleGenome
from .core.result import OrganelleResult

CoreObject: TypeAlias = OrganelleGenome | OrganelleData | OrganelleResult

_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "organelleverse"
_OPERATION_ID = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_POINTER_SUFFIX = ".pointer.json"


def cache_root() -> Path:
    """Return the configured internal cache root without creating it."""

    configured = os.environ.get("ORGANELLEVERSE_CACHE_ROOT")
    return Path(configured).expanduser().resolve() if configured else _DEFAULT_CACHE_ROOT.resolve()


def managed_runs_root() -> Path:
    """Return the root containing all managed scientific runs."""

    return cache_root() / "runs"


def managed_run_path(operation_id: str, run_id: str) -> Path:
    """Return one safe managed run path without creating it."""

    if _OPERATION_ID.fullmatch(operation_id) is None or _RUN_ID.fullmatch(run_id) is None:
        raise OrganelleInputError(
            code="runtime.invalid_run_identity",
            message="managed run operation and run identifiers must be safe path components",
            details={"operation_id": operation_id, "run_id": run_id},
        )
    return managed_runs_root() / operation_id / run_id


def staged_run_path(operation_id: str, request_run_id: str) -> Path:
    """Return the private hidden sibling used by one controlled invocation."""

    completed_shape = managed_run_path(operation_id, request_run_id)
    return completed_shape.parent / f".staging-{request_run_id}"


def create_staged_run(operation_id: str, request_run_id: str) -> Path:
    """Create one parent-owned same-filesystem staging directory."""

    staging = staged_run_path(operation_id, request_run_id)
    operation_root = staging.parent
    runs_root = managed_runs_root()
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OrganelleExecutionError(
            code="runtime.staging_create_failed",
            message="managed staging root could not be created",
            details={"path": str(runs_root), "errno": error.errno, "reason": str(error)},
        ) from error
    if runs_root.is_symlink():
        raise OrganelleInputError(
            code="runtime.unsafe_managed_path",
            message="managed staging paths may not traverse symbolic-link directories",
            details={"path": str(staging), "symlink": str(runs_root)},
        )
    if operation_root.is_symlink():
        raise OrganelleInputError(
            code="runtime.unsafe_managed_path",
            message="managed staging paths may not traverse symbolic-link directories",
            details={"path": str(staging), "symlink": str(operation_root)},
        )
    try:
        operation_root.mkdir(mode=0o700, exist_ok=True)
    except OSError as error:
        raise OrganelleExecutionError(
            code="runtime.staging_create_failed",
            message="managed operation staging root could not be created",
            details={
                "path": str(operation_root),
                "errno": error.errno,
                "reason": str(error),
            },
        ) from error
    if operation_root.is_symlink():
        raise OrganelleInputError(
            code="runtime.unsafe_managed_path",
            message="managed staging paths may not traverse symbolic-link directories",
            details={"path": str(staging), "symlink": str(operation_root)},
        )
    try:
        staging.mkdir(mode=0o700)
    except FileExistsError as error:
        raise OrganelleInputError(
            code="runtime.staging_conflict",
            message="managed staging leaf already exists",
            details={"path": str(staging)},
        ) from error
    except OSError as error:
        raise OrganelleExecutionError(
            code="runtime.staging_create_failed",
            message="managed staging leaf could not be created",
            details={"path": str(staging), "errno": error.errno, "reason": str(error)},
        ) from error
    return staging


def _core_artifacts(value: CoreObject) -> tuple[ArtifactRef, ...]:
    if isinstance(value, OrganelleResult):
        return value.artifacts
    if isinstance(value, OrganelleData):
        return tuple(value.artifacts.values())
    return tuple(
        artifact
        for artifact in (
            value.sequence,
            value.annotation,
            *value.source_manifests,
        )
        if artifact is not None
    )


def _strict_artifact_relative(uri: str) -> Path:
    parts = uri.split("/")
    if (
        not uri
        or "\x00" in uri
        or "\\" in uri
        or any(part in {"", ".", ".."} for part in parts)
        or PurePosixPath(uri).anchor
        or PureWindowsPath(uri).anchor
        or uri == "result.json"
    ):
        raise OrganelleInputError(
            code="runtime.invalid_artifact_uri",
            message="worker artifact URI must be a strict relative POSIX path",
            details={"uri": uri},
        )
    return Path(*parts)


def _validated_managed_tree_root(root: Path) -> Path:
    candidate = Path(root).expanduser().absolute()
    runs_root = managed_runs_root().absolute()
    if ".." in candidate.parts:
        raise OrganelleInputError(
            code="runtime.staging_outside_cache",
            message="managed artifact verification refuses lexical parent traversal",
            details={"path": str(candidate)},
        )
    try:
        candidate.relative_to(runs_root)
    except ValueError as error:
        raise OrganelleInputError(
            code="runtime.staging_outside_cache",
            message="managed artifact verification refuses paths outside the run store",
            details={"path": str(candidate)},
        ) from error
    _reject_symlink_components(candidate, runs_root, include_final=True)
    try:
        resolved_runs = runs_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        # The component walk above wins deterministic symlink cases. This
        # second containment check is deliberately retained for a component
        # exchanged between that walk and resolution (TOCTOU mitigation).
        resolved.relative_to(resolved_runs)
    except ValueError as error:
        raise OrganelleInputError(
            code="runtime.staging_outside_cache",
            message="managed artifact verification refuses paths outside the run store",
            details={"path": str(candidate)},
        ) from error
    except OSError as error:
        raise OrganelleInputError(
            code="runtime.invalid_temporary_run",
            message="managed artifact verification requires a resolvable run directory",
            details={"path": str(candidate), "reason": str(error)},
        ) from error
    if candidate.is_symlink() or not resolved.is_dir():
        raise OrganelleInputError(
            code="runtime.invalid_temporary_run",
            message="managed artifact verification requires a real run directory",
            details={"path": str(candidate)},
        )
    return resolved


def verify_staged_artifacts(
    value: CoreObject,
    staging_run: Path,
    *,
    trusted_input_hashes: frozenset[str] = frozenset(),
) -> tuple[Path, ...]:
    """Verify that declared L1 artifacts are exactly the safe staging tree."""

    root = _validated_managed_tree_root(staging_run)
    artifacts = _core_artifacts(value)
    relatives: list[Path] = []
    seen: set[str] = set()
    for artifact in artifacts:
        relative = _strict_artifact_relative(artifact.uri)
        normalized = relative.as_posix()
        if normalized in seen:
            raise OrganelleInputError(
                code="runtime.duplicate_artifact",
                message="worker artifact declarations must use unique paths",
                details={"uri": artifact.uri},
            )
        seen.add(normalized)
        relatives.append(relative)

    files, _ = _walk_tree(root)
    actual = {path.relative_to(root).as_posix() for path in files}
    if actual != seen:
        raise OrganelleInputError(
            code="runtime.artifact_tree_mismatch",
            message="staging files must exactly equal embedded ArtifactRef declarations",
            details={
                "missing": sorted(seen - actual),
                "undeclared": sorted(actual - seen),
            },
        )

    for artifact, relative in zip(artifacts, relatives, strict=True):
        artifact_path = root / relative
        try:
            current = ArtifactRef.from_path(
                artifact_path,
                kind=artifact.kind,
                format=artifact.format,
                media_type=artifact.media_type,
            )
        except OSError as error:
            raise OrganelleInputError(
                code="runtime.artifact_read_failed",
                message="staged artifact bytes could not be read",
                details={"path": str(artifact_path), "reason": str(error)},
            ) from error
        if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
            raise OrganelleInputError(
                code="runtime.artifact_digest_mismatch",
                message="staged artifact bytes do not match the worker declaration",
                details={"path": str(root / relative), "artifact_id": artifact.object_id},
            )
        if current.sha256 in trusted_input_hashes:
            raise OrganelleInputError(
                code="runtime.input_artifact_reemission",
                message="worker outputs may not re-emit trusted input artifact bytes",
                details={"path": str(root / relative), "sha256": current.sha256},
            )
        if artifact.validated != current.validated:
            raise OrganelleInputError(
                code="runtime.artifact_validation_mismatch",
                message="staged artifact validation claim does not match parent verification",
                details={"path": str(root / relative)},
            )
    return tuple(relatives)


def _relocated_artifact(artifact: ArtifactRef, root: Path, relative: Path) -> ArtifactRef:
    return artifact.model_copy(update={"uri": str(root / relative)})


def _relocate_core_artifacts(
    value: CoreObject,
    completed_run: Path,
    relatives: tuple[Path, ...],
) -> CoreObject:
    iterator = iter(relatives)
    if isinstance(value, OrganelleResult):
        return value.model_copy(
            update={
                "artifacts": tuple(
                    _relocated_artifact(artifact, completed_run, next(iterator))
                    for artifact in value.artifacts
                )
            }
        )
    if isinstance(value, OrganelleData):
        return value.model_copy(
            update={
                "artifacts": FrozenMap.from_items(
                    {
                        name: _relocated_artifact(artifact, completed_run, next(iterator))
                        for name, artifact in value.artifacts.items()
                    }
                )
            }
        )
    sequence = (
        _relocated_artifact(value.sequence, completed_run, next(iterator))
        if value.sequence is not None
        else None
    )
    annotation = (
        _relocated_artifact(value.annotation, completed_run, next(iterator))
        if value.annotation is not None
        else None
    )
    manifests = tuple(
        _relocated_artifact(artifact, completed_run, next(iterator))
        for artifact in value.source_manifests
    )
    return value.model_copy(
        update={
            "sequence": sequence,
            "annotation": annotation,
            "source_manifests": manifests,
        }
    )


def publish_staged_result(
    value: CoreObject,
    staging_run: Path,
    completed_run: Path,
    *,
    trusted_input_hashes: frozenset[str] = frozenset(),
) -> CoreObject:
    """Verify, durably publish, reverify, and relocate one staged L1 value."""

    source = _validated_managed_tree_root(staging_run)
    destination = Path(completed_run).expanduser().absolute()
    runs_root = managed_runs_root()
    try:
        destination_relative = destination.relative_to(runs_root)
    except ValueError as error:
        raise OrganelleInputError(
            code="runtime.publication_outside_cache",
            message="completed run must remain inside the managed run store",
            details={"path": str(destination)},
        ) from error
    if (
        source.parent != destination.parent
        or not source.name.startswith(".staging-")
        or len(destination_relative.parts) != 2
        or _RUN_ID.fullmatch(destination.name) is None
    ):
        raise OrganelleInputError(
            code="runtime.invalid_run_identity",
            message="staging and publication must be sibling managed-run paths",
            details={"staging": str(source), "completed": str(destination)},
        )

    verification_options = (
        {"trusted_input_hashes": trusted_input_hashes} if trusted_input_hashes else {}
    )
    relatives = verify_staged_artifacts(value, source, **verification_options)
    original_identity = value.object_id
    try:
        source_stat = source.stat(follow_symlinks=False)
        owned_publication_identity = (source_stat.st_dev, source_stat.st_ino)
        _fsync_tree(source)
    except OSError as error:
        raise OrganelleExecutionError(
            code="runtime.staging_sync_failed",
            message="verified staging tree could not be made durable before publication",
            details={"path": str(source), "errno": error.errno, "reason": str(error)},
        ) from error
    published_identity: tuple[int, int] | None = None
    try:
        publish_run(source, destination)
        published_identity = owned_publication_identity
        final_relatives = verify_staged_artifacts(value, destination, **verification_options)
        if final_relatives != relatives:
            raise OrganelleExecutionError(
                code="runtime.publication_verification_failed",
                message="published artifact paths differ from the verified staging tree",
                details={"path": str(destination)},
            )
        relocated = _relocate_core_artifacts(value, destination, relatives)
        model_type = type(value)
        revalidated = model_type.model_validate(relocated)
        if revalidated.object_id != original_identity:
            raise OrganelleExecutionError(
                code="runtime.relocation_identity_changed",
                message="managed ArtifactRef URI relocation changed L1 object identity",
            )
        return revalidated
    except BaseException as error:
        if published_identity is None:
            raise
        _rollback_owned_publication(destination, published_identity, error)


def _rollback_owned_publication(
    destination: Path,
    expected_identity: tuple[int, int],
    original: BaseException,
) -> NoReturn:
    """Atomically move a failed publication out of its public name for recovery.

    The rename is the ownership check: whatever occupies ``destination`` at
    that instant is transferred to an undisclosed sibling before its inode is
    inspected. A replacement is restored when possible. An owned publication
    remains at the reported quarantine path rather than being recursively
    deleted, so a later pathname substitution can never turn rollback into
    deletion of unrelated data. Controlled workers have exited before this
    runs; same-UID hostile mutation can disrupt recovery but cannot make this
    routine remove a path.
    """

    quarantine = destination.with_name(f".rollback-{destination.name}-{uuid4().hex}")
    try:
        _rename_no_replace(destination, quarantine)
    except OSError as rollback_error:
        raise _publication_rollback_error(
            destination,
            original,
            f"published path could not be transferred for rollback: {rollback_error}",
        ) from original
    try:
        _fsync_directory(destination.parent)
    except OSError as rollback_error:
        raise _publication_rollback_error(
            destination,
            original,
            f"rollback transfer could not be made durable: {rollback_error}",
            preserved_path=quarantine,
        ) from original

    try:
        quarantined_stat = quarantine.stat(follow_symlinks=False)
    except OSError as rollback_error:
        raise _publication_rollback_error(
            destination,
            original,
            f"rollback quarantine could not be identified: {rollback_error}",
            preserved_path=quarantine,
        ) from original
    quarantined_identity = (quarantined_stat.st_dev, quarantined_stat.st_ino)
    if quarantined_identity != expected_identity:
        try:
            _rename_no_replace(quarantine, destination)
        except OSError as restore_error:
            raise _publication_rollback_error(
                destination,
                original,
                f"published path was replaced before rollback and could not be restored: "
                f"{restore_error}",
                preserved_path=quarantine,
            ) from original
        try:
            _fsync_directory(destination.parent)
        except OSError as restore_error:
            raise _publication_rollback_error(
                destination,
                original,
                f"restored replacement could not be made durable: {restore_error}",
                preserved_path=destination,
            ) from original
        raise _publication_rollback_error(
            destination,
            original,
            "published path was replaced before rollback; replacement was restored",
        ) from original

    raise _publication_rollback_error(
        destination,
        original,
        "failed publication was withdrawn and preserved for explicit recovery",
        preserved_path=quarantine,
    ) from original


def _publication_rollback_error(
    destination: Path,
    original: BaseException,
    reason: str,
    *,
    preserved_path: Path | None = None,
) -> OrganelleExecutionError:
    details: dict[str, object] = {
        "destination": str(destination),
        "reason": reason,
        "original_type": type(original).__name__,
        "original_code": getattr(original, "code", None),
        "original_reason": str(original),
    }
    if preserved_path is not None:
        details["preserved_path"] = str(preserved_path)
    return OrganelleExecutionError(
        code="runtime.rollback_failed",
        message="failed staged publication could not be rolled back safely",
        details=details,
    )


def publish_run(temporary: Path, completed: Path) -> None:
    """Atomically publish a completed managed run without replacing a peer."""

    source = Path(temporary)
    destination = Path(completed)
    try:
        if source.is_symlink() or not source.is_dir():
            raise OrganelleInputError(
                code="runtime.invalid_temporary_run",
                message="managed run publication requires a real temporary directory",
                details={"path": str(source)},
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _lexists(destination):
            raise OrganelleInputError(
                code="runtime.run_conflict",
                message="managed run destination already exists",
                details={"path": str(destination)},
            )
        source_stat = source.stat(follow_symlinks=False)
    except OrganelleInputError:
        raise
    except OSError as error:
        raise OrganelleExecutionError(
            code="runtime.run_publication_failed",
            message="managed run publication preflight failed",
            details={"path": str(destination), "errno": error.errno, "reason": str(error)},
        ) from error
    source_identity = (source_stat.st_dev, source_stat.st_ino)
    renamed = False
    try:
        _rename_no_replace(source, destination)
        renamed = True
        _fsync_directory(destination.parent)
    except FileExistsError as error:
        raise OrganelleInputError(
            code="runtime.run_conflict",
            message="managed run destination appeared during publication",
            details={"path": str(destination)},
        ) from error
    except OSError as error:
        if renamed:
            _rollback_owned_publication(destination, source_identity, error)
        raise OrganelleExecutionError(
            code="runtime.run_publication_failed",
            message="managed run could not be published atomically",
            details={"path": str(destination), "errno": error.errno, "reason": str(error)},
        ) from error


def promote_result(result: OrganelleResult, destination: str | Path) -> OrganelleResult:
    """Move one artifact-backed managed Result to a user destination.

    Promotion leaves one complete artifact tree. The former managed path is a
    symbolic link to the published tree, or a small pointer record when the
    platform cannot create symbolic links.
    """

    target = Path(destination).expanduser().resolve(strict=False)
    if _lexists(target):
        try:
            return _reuse_published_result(result, target)
        except OSError as error:
            raise _destination_conflict(
                target,
                f"destination publication could not be read: {error}",
            ) from error

    try:
        source, relative_artifacts = _managed_source(result)
        _validate_destination(source, target)
        _verify_result_artifacts(result, source, relative_artifacts)

        relocated = result.model_copy(
            update={
                "artifacts": tuple(
                    artifact.model_copy(update={"uri": str(target / relative)})
                    for artifact, relative in zip(result.artifacts, relative_artifacts, strict=True)
                )
            }
        )
        if relocated.object_id != result.object_id:
            raise OrganelleExecutionError(
                code="runtime.relocation_identity_changed",
                message="artifact URI relocation changed scientific Result identity",
            )

        if any(relative == Path("result.json") for relative in relative_artifacts):
            raise OrganelleInputError(
                code="runtime.self_referential_result",
                message="result.json cannot be a declared artifact of the Result it serializes",
            )
        source_manifest = _tree_manifest(source)
        source_stat = source.stat(follow_symlinks=False)
        source_identity = (source_stat.st_dev, source_stat.st_ino)
    except (OrganelleInputError, OrganelleExecutionError):
        raise
    except OSError as error:
        raise OrganelleExecutionError(
            code="runtime.promotion_failed",
            message="managed Result promotion preflight failed",
            details={"path": str(target), "errno": error.errno, "reason": str(error)},
        ) from error

    cleanup_warning: str | None = None
    try:
        try:
            _rename_no_replace(source, target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            cleanup_warning = _promote_across_filesystems(
                source,
                target,
                result,
                relative_artifacts,
                source_manifest,
            )
        else:
            try:
                _fsync_directory(target.parent)
                _verify_promoted_tree(
                    target,
                    result,
                    relative_artifacts,
                    source_manifest,
                )
            except BaseException as error:
                _rollback_owned_publication(target, source_identity, error)
    except FileExistsError as error:
        raise OrganelleInputError(
            code="runtime.destination_conflict",
            message="publication destination appeared during promotion",
            details={"path": str(target)},
        ) from error
    except (OrganelleInputError, OrganelleExecutionError):
        raise
    except OSError as error:
        raise OrganelleExecutionError(
            code="runtime.promotion_failed",
            message="managed Result promotion failed",
            details={"path": str(target), "errno": error.errno, "reason": str(error)},
        ) from error

    postpublication_warnings: list[str] = []
    if cleanup_warning is not None:
        postpublication_warnings.append(cleanup_warning)
    try:
        _install_managed_pointer(source, target, relocated.object_id)
    except OSError as error:
        postpublication_warnings.append(
            f"publication succeeded but its managed pointer could not be installed: {error}"
        )
    try:
        _write_result(target / "result.json", relocated)
    except OSError as error:
        postpublication_warnings.append(
            f"publication succeeded but its relocated result.json could not be updated: {error}"
        )
    for warning_message in postpublication_warnings:
        warnings.warn(
            warning_message,
            RuntimeWarning,
            stacklevel=2,
        )
    return relocated


def _reuse_published_result(
    requested: OrganelleResult,
    target: Path,
) -> OrganelleResult:
    """Return an exact, verified prior publication for an idempotent retry."""

    if target.is_symlink() or not target.is_dir():
        raise _destination_conflict(target, "destination is not a real directory")
    result_path = target / "result.json"
    if result_path.is_symlink() or not result_path.is_file():
        raise _destination_conflict(target, "destination has no regular result.json")
    try:
        published = OrganelleResult.model_validate_json(result_path.read_bytes())
    except (OSError, ValidationError) as error:
        raise _destination_conflict(target, "destination result.json is invalid") from error
    if published.object_id != requested.object_id:
        raise _destination_conflict(target, "destination contains a different Result")
    if tuple(artifact.object_id for artifact in published.artifacts) != tuple(
        artifact.object_id for artifact in requested.artifacts
    ):
        raise _destination_conflict(
            target, "destination artifacts differ from the requested Result"
        )

    target_resolved = target.resolve(strict=True)
    for artifact in published.artifacts:
        path = Path(artifact.uri).expanduser().absolute()
        try:
            path.relative_to(target)
            path.resolve(strict=True).relative_to(target_resolved)
        except (FileNotFoundError, ValueError) as error:
            raise _destination_conflict(
                target,
                "destination Result references a missing or external artifact",
            ) from error
        if path.is_symlink() or not path.is_file():
            raise _destination_conflict(target, "destination contains an unsafe artifact")
        current = ArtifactRef.from_path(
            path,
            kind=artifact.kind,
            format=artifact.format,
            media_type=artifact.media_type,
        )
        if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
            raise _destination_conflict(target, "destination artifact digest does not match")
    return published


def _destination_conflict(target: Path, reason: str) -> OrganelleInputError:
    return OrganelleInputError(
        code="runtime.destination_conflict",
        message="publication destination already exists with different or invalid content",
        details={"path": str(target), "reason": reason},
    )


def remove_managed_entry(path: str | Path) -> None:
    """Remove an owned cache entry without following symbolic links."""

    candidate = Path(path).expanduser().absolute()
    try:
        candidate.relative_to(managed_runs_root())
    except ValueError as error:
        raise OrganelleInputError(
            code="runtime.cleanup_outside_cache",
            message="managed cleanup refuses paths outside the managed run store",
            details={"path": str(candidate)},
        ) from error
    _reject_symlink_components(candidate, managed_runs_root(), include_final=False)
    _remove_path_no_follow(candidate)
    pointer = candidate.with_name(f"{candidate.name}{_POINTER_SUFFIX}")
    if _lexists(pointer):
        _remove_path_no_follow(pointer)


def _remove_path_no_follow(candidate: Path) -> None:
    if candidate.is_symlink():
        candidate.unlink(missing_ok=True)
    elif candidate.is_dir():
        shutil.rmtree(candidate)
    elif _lexists(candidate):
        candidate.unlink()


def _managed_source(result: OrganelleResult) -> tuple[Path, tuple[Path, ...]]:
    if not result.artifacts:
        raise OrganelleInputError(
            code="runtime.result_has_no_artifacts",
            message="only artifact-backed Results can be promoted",
            details={"result_id": result.object_id},
        )
    runs_root = managed_runs_root()
    source: Path | None = None
    relatives: list[Path] = []
    for artifact in result.artifacts:
        lexical = Path(artifact.uri).expanduser().absolute()
        try:
            under_runs = lexical.relative_to(runs_root)
        except ValueError as error:
            raise OrganelleInputError(
                code="runtime.artifact_not_managed",
                message="Result artifact is outside the managed run store",
                details={"uri": artifact.uri},
            ) from error
        if len(under_runs.parts) < 3:
            raise OrganelleInputError(
                code="runtime.artifact_not_managed",
                message="Result artifact URI does not identify a managed run file",
                details={"uri": artifact.uri},
            )
        if under_runs.parts[0] != result.operation_id:
            raise OrganelleInputError(
                code="runtime.artifact_operation_mismatch",
                message="managed artifacts are stored under a different operation",
                details={
                    "result_operation_id": result.operation_id,
                    "managed_operation_id": under_runs.parts[0],
                },
            )
        if _RUN_ID.fullmatch(under_runs.parts[1]) is None:
            raise OrganelleInputError(
                code="runtime.invalid_run_identity",
                message="managed artifact URI contains an invalid run identifier",
                details={"run_id": under_runs.parts[1]},
            )
        candidate_source = runs_root / under_runs.parts[0] / under_runs.parts[1]
        if source is None:
            source = candidate_source
        elif source != candidate_source:
            raise OrganelleInputError(
                code="runtime.mixed_managed_runs",
                message="all promoted Result artifacts must belong to one managed run",
                details={"result_id": result.object_id},
            )
        relatives.append(Path(*under_runs.parts[2:]))
    assert source is not None
    _reject_symlink_components(source, runs_root, include_final=True)
    if source.is_symlink() or not source.is_dir():
        raise OrganelleInputError(
            code="runtime.managed_run_unavailable",
            message="managed Result source is not an unpublished run directory",
            details={"path": str(source)},
        )
    return source, tuple(relatives)


def _reject_symlink_components(
    path: Path,
    root: Path,
    *,
    include_final: bool,
) -> None:
    relative = path.relative_to(root)
    parts = relative.parts if include_final else relative.parts[:-1]
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise OrganelleInputError(
                code="runtime.unsafe_managed_path",
                message="managed run paths may not traverse symbolic-link directories",
                details={"path": str(path), "symlink": str(current)},
            )


def _validate_destination(source: Path, destination: Path) -> None:
    if _lexists(destination):
        raise OrganelleInputError(
            code="runtime.destination_conflict",
            message="publication destination already exists",
            details={"path": str(destination)},
        )
    source_absolute = source.absolute()
    if (
        destination == source_absolute
        or source_absolute in destination.parents
        or destination in source_absolute.parents
    ):
        raise OrganelleInputError(
            code="runtime.unsafe_destination",
            message="publication destination cannot be inside its managed source",
            details={"source": str(source), "destination": str(destination)},
        )
    destination.parent.mkdir(parents=True, exist_ok=True)


def _verify_result_artifacts(
    result: OrganelleResult,
    source: Path,
    relative_artifacts: tuple[Path, ...],
) -> None:
    source_resolved = source.resolve(strict=True)
    for artifact, relative in zip(result.artifacts, relative_artifacts, strict=True):
        path = source / relative
        if path.is_symlink():
            raise OrganelleInputError(
                code="runtime.unsafe_artifact",
                message="managed artifacts may not be symbolic links",
                details={"path": str(path)},
            )
        try:
            path.resolve(strict=True).relative_to(source_resolved)
        except (FileNotFoundError, ValueError) as error:
            raise OrganelleInputError(
                code="runtime.unsafe_artifact",
                message="managed artifact is missing or escapes its run directory",
                details={"path": str(path)},
            ) from error
        current = ArtifactRef.from_path(
            path,
            kind=artifact.kind,
            format=artifact.format,
            media_type=artifact.media_type,
        )
        if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
            raise OrganelleInputError(
                code="runtime.artifact_digest_mismatch",
                message="managed artifact no longer matches its declared digest",
                details={"path": str(path), "artifact_id": artifact.object_id},
            )


def _promote_across_filesystems(
    source: Path,
    destination: Path,
    result: OrganelleResult,
    relative_artifacts: tuple[Path, ...],
    source_manifest: dict[Path, tuple[int, str]],
) -> str | None:
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    committed = False
    published_identity: tuple[int, int] | None = None
    try:
        shutil.copytree(source, temporary, symlinks=False)
        if _tree_manifest(temporary) != source_manifest:
            raise OrganelleExecutionError(
                code="runtime.copy_verification_failed",
                message="cross-filesystem promotion copy differs from its managed source",
            )
        _fsync_tree(temporary)
        temporary_stat = temporary.stat(follow_symlinks=False)
        published_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        _rename_no_replace(temporary, destination)
        committed = True
        _fsync_directory(destination.parent)
        _verify_promoted_tree(
            destination,
            result,
            relative_artifacts,
            source_manifest,
        )
    except BaseException as error:
        if committed:
            assert published_identity is not None
            _rollback_owned_publication(destination, published_identity, error)
        _remove_path_no_follow(temporary)
        raise

    # Publication is committed and verified at this point. Source cleanup is a
    # post-commit maintenance action: neither its own error nor a warning filter
    # that promotes the notification to an exception may withdraw the result.
    try:
        remove_managed_entry(source)
    except (OSError, OrganelleInputError) as error:
        return (
            "verified publication succeeded but managed source cleanup was incomplete: "
            f"{source}: {error}"
        )
    return None


def _verify_promoted_tree(
    destination: Path,
    result: OrganelleResult,
    relative_artifacts: tuple[Path, ...],
    expected_manifest: dict[Path, tuple[int, str]],
) -> None:
    if _tree_manifest(destination) != expected_manifest:
        raise OrganelleExecutionError(
            code="runtime.promotion_verification_failed",
            message="published artifact tree differs from its managed source",
            details={"path": str(destination)},
        )
    _verify_result_artifacts_at(result, destination, relative_artifacts)


def _verify_result_artifacts_at(
    result: OrganelleResult,
    root: Path,
    relatives: tuple[Path, ...],
) -> None:
    for artifact, relative in zip(result.artifacts, relatives, strict=True):
        current = ArtifactRef.from_path(
            root / relative,
            kind=artifact.kind,
            format=artifact.format,
            media_type=artifact.media_type,
        )
        if current.sha256 != artifact.sha256 or current.size_bytes != artifact.size_bytes:
            raise OrganelleExecutionError(
                code="runtime.promotion_verification_failed",
                message="published artifact digest differs from its Result contract",
                details={"path": str(root / relative)},
            )


def _write_result(path: Path, result: OrganelleResult) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    temporary.write_text(result.model_dump_json(exclude={"object_id"}) + "\n")
    _fsync_file(temporary)
    os.replace(temporary, path)


def _tree_manifest(root: Path) -> dict[Path, tuple[int, str]]:
    manifest: dict[Path, tuple[int, str]] = {}
    files, _ = _walk_tree(root)
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest[path.relative_to(root)] = (path.stat().st_size, digest.hexdigest())
    return manifest


def _fsync_tree(root: Path) -> None:
    files, directories = _walk_tree(root)
    for path in files:
        _fsync_file(path)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


def _walk_tree(root: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    directories: list[Path] = []
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current / name
            if path.is_symlink():
                raise OrganelleInputError(
                    code="runtime.unsafe_managed_tree",
                    message="managed run trees may not contain symbolic links",
                    details={"path": str(path)},
                )
            directories.append(path)
        for name in file_names:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise OrganelleInputError(
                    code="runtime.unsafe_managed_tree",
                    message="managed run trees may contain only regular files and directories",
                    details={"path": str(path)},
                )
            if path.stat(follow_symlinks=False).st_nlink != 1:
                raise OrganelleInputError(
                    code="runtime.unsafe_managed_tree",
                    message="managed run artifact files must have exactly one hard link",
                    details={"path": str(path)},
                )
            files.append(path)
    return files, directories


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
    finally:
        os.close(descriptor)


def _install_managed_pointer(source: Path, destination: Path, result_id: str) -> None:
    relative_target = os.path.relpath(destination, start=source.parent)
    try:
        os.symlink(relative_target, source, target_is_directory=True)
        _fsync_directory(source.parent)
        return
    except OSError:
        pass
    pointer = source.with_name(f"{source.name}{_POINTER_SUFFIX}")
    temporary = pointer.with_name(f".{pointer.name}.tmp-{uuid4().hex}")
    payload = {
        "schema_version": "organelleverse.managed-pointer.v1",
        "destination": str(destination),
        "result_id": result_id,
    }
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    _fsync_file(temporary)
    os.replace(temporary, pointer)
    _fsync_directory(pointer.parent)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename without replacing an existing destination."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            outcome = renameat2(-100, source_bytes, -100, destination_bytes, 1)
        else:
            syscall_number = {
                "x86_64": 316,
                "amd64": 316,
                "aarch64": 276,
                "arm64": 276,
            }.get(os.uname().machine.casefold())
            if syscall_number is None:
                raise OSError(
                    errno.ENOTSUP,
                    "renameat2 syscall number is unknown on this Linux architecture",
                    str(destination),
                )
            syscall = libc.syscall
            syscall.restype = ctypes.c_long
            outcome = syscall(
                ctypes.c_long(syscall_number),
                ctypes.c_int(-100),
                ctypes.c_char_p(source_bytes),
                ctypes.c_int(-100),
                ctypes.c_char_p(destination_bytes),
                ctypes.c_uint(1),
            )
        if outcome == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, os.strerror(error_number), str(destination))
        raise OSError(error_number, os.strerror(error_number), str(destination))
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OSError(
                errno.ENOTSUP,
                "renamex_np is unavailable on this macOS runtime",
                str(destination),
            )
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        outcome = renamex_np(source_bytes, destination_bytes, 0x00000004)
        if outcome == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, os.strerror(error_number), str(destination))
        raise OSError(error_number, os.strerror(error_number), str(destination))
    if os.name == "nt":
        os.rename(source, destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace rename is unsupported on this platform",
        str(destination),
    )


__all__ = [
    "cache_root",
    "create_staged_run",
    "managed_run_path",
    "managed_runs_root",
    "promote_result",
    "publish_run",
    "publish_staged_result",
    "remove_managed_entry",
    "staged_run_path",
    "verify_staged_artifacts",
]
