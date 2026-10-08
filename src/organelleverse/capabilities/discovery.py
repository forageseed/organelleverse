"""Deterministic, side-effect-free capability bundle discovery."""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from packaging.utils import canonicalize_name
from pydantic import JsonValue

from organelleverse.core.errors import OrganelleContractError, OrganelleError

from .code_identity import inspect_bundle_code
from .hashing import hash_bundle
from .index import (
    CapabilityConflict,
    CapabilityConflictMember,
    CapabilityDiagnostic,
    CapabilityEntry,
    CapabilityIndex,
    CapabilityOrigin,
)
from .models import ImplementationKind
from .parser import parse_capability_bundle

_ENTRY_POINT_GROUP = "organelleverse.capabilities"


class DistributionLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def files(self) -> Sequence[PurePosixPath] | None: ...

    def locate_file(self, path: PurePosixPath) -> Path: ...


class EntryPointLike(Protocol):
    name: str
    value: str

    @property
    def dist(self) -> DistributionLike | None: ...


class SearchRoot:
    """One declared capability search root: its channel, path, and search-root path."""

    __slots__ = ("channel", "root", "search_root")

    def __init__(self, channel: str, root: Path, search_root: Path | None = None) -> None:
        self.channel = channel
        self.root = root
        self.search_root = search_root or root


def _diagnostic(
    *,
    code: str,
    message: str,
    origin: CapabilityOrigin | None = None,
    capability_id: str | None = None,
    details: dict[str, object] | None = None,
) -> CapabilityDiagnostic:
    return CapabilityDiagnostic.model_validate(
        {
            "code": code,
            "message": message,
            "capability_id": capability_id,
            "origins": () if origin is None else (origin,),
            "details": {} if details is None else details,
        }
    )


def _origin(channel: str, path: Path, search_root: Path) -> CapabilityOrigin:
    return CapabilityOrigin(
        channel=channel,
        source_path=str(path.resolve(strict=False)),
        search_root=str(search_root.resolve(strict=False)),
    )


def _standard_roots() -> tuple[SearchRoot, ...]:
    package_root = Path(str(importlib.resources.files("organelleverse")))
    home = Path(os.environ.get("ORGANELLEVERSE_HOME", Path.home() / ".organelleverse"))
    project = Path.cwd() / ".organelleverse" / "capabilities"
    roots = [
        SearchRoot("core", package_root / "capabilities"),
        SearchRoot("local", home / "capabilities"),
        SearchRoot("project", project),
    ]
    for text in os.environ.get("ORGANELLEVERSE_CAPABILITY_PATH", "").split(os.pathsep):
        if text.strip():
            roots.append(SearchRoot("local", Path(text)))
    return tuple(roots)


def _bundle_paths(root: SearchRoot) -> tuple[tuple[Path, CapabilityOrigin], ...]:
    if not root.root.exists() or not root.root.is_dir():
        return ()
    if root.root.is_symlink():
        return ((root.root, _origin(root.channel, root.root, root.search_root)),)
    resolved_root = root.root.resolve(strict=True)
    found: list[tuple[Path, CapabilityOrigin]] = []
    for candidate in sorted(resolved_root.rglob("*"), key=lambda path: path.as_posix()):
        if candidate.is_symlink():
            found.append((candidate, _origin(root.channel, candidate, root.search_root)))
            continue
        if candidate.name != "capability.toml" or not candidate.is_file():
            continue
        resolved = candidate.resolve(strict=True)
        if resolved.is_relative_to(resolved_root):
            found.append((resolved, _origin(root.channel, resolved, root.search_root)))
    return tuple(found)


def _default_entry_points() -> tuple[EntryPointLike, ...]:
    selected = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
    return cast(tuple[EntryPointLike, ...], tuple(selected))


def _entry_point_sort_key(entry_point: EntryPointLike) -> tuple[str, str, str]:
    distribution = entry_point.dist
    distribution_name = "" if distribution is None else canonicalize_name(distribution.name)
    return (distribution_name, entry_point.name, entry_point.value)


def _entry_point_root(
    entry_point: EntryPointLike,
) -> tuple[SearchRoot | None, CapabilityDiagnostic | None]:
    distribution = entry_point.dist
    if distribution is None:
        return None, _diagnostic(
            code="capability.entry_point_invalid",
            message="capability entry point has no owning distribution",
            details={"entry_point": entry_point.name, "value": entry_point.value},
        )
    try:
        module_name, resource_directory = entry_point.value.split(":", 1)
    except ValueError:
        return None, _diagnostic(
            code="capability.entry_point_invalid",
            message="capability entry point must be module:resource_directory",
            details={"entry_point": entry_point.name, "value": entry_point.value},
        )
    if not module_name.strip() or not resource_directory.strip():
        return None, _diagnostic(
            code="capability.entry_point_invalid",
            message="capability entry point module and resource directory must be nonblank",
            details={"entry_point": entry_point.name, "value": entry_point.value},
        )
    resource_parts = PurePosixPath(resource_directory).parts
    if (
        PurePosixPath(resource_directory).is_absolute()
        or not resource_parts
        or any(part in {"", ".", ".."} for part in resource_parts)
    ):
        return None, _diagnostic(
            code="capability.entry_point_invalid",
            message="capability entry point resource directory must be a contained relative path",
            details={"entry_point": entry_point.name, "value": entry_point.value},
        )
    module_path = PurePosixPath(*module_name.split("."))
    resource_path = module_path / PurePosixPath(resource_directory)
    files = tuple(distribution.files or ())
    prefix = resource_path.as_posix().rstrip("/") + "/"
    if not any(item.as_posix().startswith(prefix) for item in files):
        return None, _diagnostic(
            code="capability.entry_point_resource_missing",
            message="capability entry point resource directory is absent from distribution files",
            details={
                "distribution": distribution.name,
                "entry_point": entry_point.name,
                "value": entry_point.value,
            },
        )
    root = Path(distribution.locate_file(resource_path))
    channel = f"package:{canonicalize_name(distribution.name)}"
    return SearchRoot(channel, root), None


def _scan_roots(
    roots: Iterable[SearchRoot],
) -> tuple[list[CapabilityEntry], list[CapabilityDiagnostic]]:
    entries: list[CapabilityEntry] = []
    diagnostics: list[CapabilityDiagnostic] = []
    for root in roots:
        for path, origin in _bundle_paths(root):
            if path.is_symlink():
                diagnostics.append(
                    _diagnostic(
                        code="capability.bundle_path_unsafe",
                        message="capability discovery refuses symlinked bundle content",
                        origin=origin,
                        details={"path": str(path)},
                    )
                )
                continue
            capability_id: str | None = None
            try:
                bundle = parse_capability_bundle(path)
                capability_id = bundle.capability.id
                content_hash = hash_bundle(path.parent)
                execution_identity = None
                if (
                    bundle.capability.implementation is ImplementationKind.NATIVE
                    and origin.channel != "core"
                ):
                    locator = bundle.contract.callable_locator
                    if locator is None:
                        raise OrganelleContractError(
                            code="capability.native_locator_missing",
                            message=(
                                "native capability has no callable locator after validation: "
                                f"{capability_id}"
                            ),
                            details={
                                "capability_id": capability_id,
                                "bundle_root": str(path.parent),
                            },
                        )
                    execution_identity = inspect_bundle_code(
                        path.parent,
                        capability_id=capability_id,
                        bundle_content_hash=content_hash,
                        callable_locator=locator,
                    )
            except OrganelleError as error:
                payload = error.as_dict()
                diagnostics.append(
                    _diagnostic(
                        code=error.code,
                        message=error.message,
                        origin=origin,
                        capability_id=capability_id,
                        details=cast(dict[str, object], payload["details"]),
                    )
                )
                continue
            entries.append(
                CapabilityEntry(
                    capability_id=bundle.capability.id,
                    content_hash=content_hash,
                    bundle_root=path.parent.resolve(),
                    bundle=bundle,
                    origins=(origin,),
                    execution_identity=execution_identity,
                )
            )
    return entries, diagnostics


def _deduplicate(
    raw_entries: Iterable[CapabilityEntry],
) -> tuple[
    tuple[CapabilityEntry, ...], tuple[CapabilityConflict, ...], tuple[CapabilityDiagnostic, ...]
]:
    by_id: dict[str, list[CapabilityEntry]] = defaultdict(list)
    for entry in raw_entries:
        by_id[entry.capability_id].append(entry)
    entries: list[CapabilityEntry] = []
    conflicts: list[CapabilityConflict] = []
    diagnostics: list[CapabilityDiagnostic] = []
    for capability_id in sorted(by_id):
        candidates = by_id[capability_id]
        by_hash: dict[str, list[CapabilityEntry]] = defaultdict(list)
        for candidate in candidates:
            by_hash[candidate.content_hash].append(candidate)
        if len(by_hash) > 1:
            members = tuple(
                CapabilityConflictMember(
                    content_hash=candidate.content_hash,
                    origin_channel=origin.channel,
                    source_path=origin.source_path,
                )
                for candidate in candidates
                for origin in candidate.origins
            )
            conflict = CapabilityConflict(capability_id=capability_id, members=members)
            conflicts.append(conflict)
            diagnostics.append(
                _diagnostic(
                    code="capability.conflict",
                    message=f"different bundle contents declare {capability_id}",
                    capability_id=capability_id,
                    details={
                        "members": [item.model_dump(mode="json") for item in conflict.members]
                    },
                )
            )
            continue
        same_hash = next(iter(by_hash.values()))
        origins = tuple(origin for candidate in same_hash for origin in candidate.origins)
        contains_non_core = any(origin.channel != "core" for origin in origins)
        if (
            same_hash[0].bundle.capability.implementation is ImplementationKind.NATIVE
            and contains_non_core
        ):
            identity_candidates = tuple(
                candidate for candidate in same_hash if candidate.execution_identity is not None
            )
            if not identity_candidates:
                diagnostics.append(
                    CapabilityDiagnostic(
                        code="capability.execution_identity_missing",
                        message=(
                            "non-core native capability has no bundle-local execution "
                            f"identity: {capability_id}"
                        ),
                        capability_id=capability_id,
                        origins=origins,
                        details=cast(
                            dict[str, JsonValue],
                            {
                                "content_hash": same_hash[0].content_hash,
                                "bundle_roots": sorted(
                                    str(candidate.bundle_root) for candidate in same_hash
                                ),
                            },
                        ),
                    )
                )
                continue
            selected = min(identity_candidates, key=lambda item: item.bundle_root.as_posix())
        else:
            selected = min(same_hash, key=lambda item: item.bundle_root.as_posix())
        entries.append(
            selected.model_copy(
                update={
                    "origins": origins,
                    "execution_identity": selected.execution_identity,
                }
            )
        )
    return tuple(entries), tuple(conflicts), tuple(diagnostics)


def select_search_roots(
    *,
    paths: Iterable[Path] | None = None,
    extra_paths: Iterable[Path] | None = None,
    entry_points: Iterable[EntryPointLike] | None = None,
) -> tuple[tuple[SearchRoot, ...], tuple[CapabilityDiagnostic, ...]]:
    """Resolve the exact search roots a discovery pass would scan.

    ``None`` selects a standard channel; an explicit empty iterable selects none.
    ``extra_paths`` adds local roots *on top of* whatever ``paths`` selected —
    the desktop's shape, where user-installed plugin sources extend the
    standard channels instead of replacing them. Returned alongside the roots
    are the entry-point selection diagnostics, which belong to the scan as a
    whole rather than to any one root. Hot-plug (:mod:`.hotplug`) watches and
    incrementally rescans exactly these declared roots — never anything wider.
    """
    if paths is None:
        roots = list(_standard_roots())
    else:
        roots = [SearchRoot("local", Path(path)) for path in paths]
    if extra_paths is not None:
        roots.extend(SearchRoot("local", Path(path)) for path in extra_paths)
    selected_entry_points = _default_entry_points() if entry_points is None else tuple(entry_points)
    package_roots: list[SearchRoot] = []
    diagnostics: list[CapabilityDiagnostic] = []
    for entry_point in sorted(selected_entry_points, key=_entry_point_sort_key):
        root, diagnostic = _entry_point_root(entry_point)
        if root is not None:
            package_roots.append(root)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return tuple((*roots, *package_roots)), tuple(diagnostics)


def scan_search_roots(
    roots: Iterable[SearchRoot],
) -> tuple[list[CapabilityEntry], list[CapabilityDiagnostic]]:
    """Scan the given roots without importing implementations or running code."""
    return _scan_roots(roots)


def deduplicate_entries(
    raw_entries: Iterable[CapabilityEntry],
) -> tuple[
    tuple[CapabilityEntry, ...], tuple[CapabilityConflict, ...], tuple[CapabilityDiagnostic, ...]
]:
    """Apply the same-ID/same-content merge and conflict rules to raw scan entries."""
    return _deduplicate(raw_entries)


def discover_capability_candidates(
    *,
    paths: Iterable[Path] | None = None,
    extra_paths: Iterable[Path] | None = None,
    entry_points: Iterable[EntryPointLike] | None = None,
) -> CapabilityIndex:
    """Discover candidate bundles without performing admission.

    See :func:`select_search_roots` for root-selection semantics.
    The returned entries retain ``status="discovered"``. This is the public
    read-side boundary for callers that need an explicit verification store.
    """
    roots, diagnostics = select_search_roots(
        paths=paths, extra_paths=extra_paths, entry_points=entry_points
    )
    raw, scan_diagnostics = scan_search_roots(roots)
    entries, conflicts, conflict_diagnostics = deduplicate_entries(raw)
    return CapabilityIndex(
        entries=entries,
        diagnostic_items=tuple((*diagnostics, *scan_diagnostics, *conflict_diagnostics)),
        conflict_items=conflicts,
    )


def discover_capabilities(
    *,
    paths: Iterable[Path] | None = None,
    extra_paths: Iterable[Path] | None = None,
    entry_points: Iterable[EntryPointLike] | None = None,
) -> CapabilityIndex:
    """Discover candidates and admit them against the default verification store."""
    discovered = discover_capability_candidates(
        paths=paths,
        extra_paths=extra_paths,
        entry_points=entry_points,
    )
    from .admission import admit_capabilities

    return admit_capabilities(discovered)


__all__ = [
    "EntryPointLike",
    "SearchRoot",
    "deduplicate_entries",
    "discover_capabilities",
    "discover_capability_candidates",
    "scan_search_roots",
    "select_search_roots",
]
