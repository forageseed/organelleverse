"""Read-only inspection of declared operation dependencies."""

from __future__ import annotations

import keyword
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from shutil import which

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion

from .spec import DependencyKind, DependencySpec, OperationSpec, StrictSpecModel


class DependencyState(StrEnum):
    """The observed state of one declared dependency."""

    PRESENT = "present"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    UNVERIFIED = "unverified"


class DependencyCheck(StrictSpecModel):
    """The read-only inspection result for one declared dependency."""

    dependency: DependencySpec
    state: DependencyState
    observed_version: str = ""
    resolved_locator: str = ""
    details: str = ""


class DependencyReport(StrictSpecModel):
    """A complete, declaration-ordered dependency inspection result."""

    operation_id: str
    checks: tuple[DependencyCheck, ...]
    ready: bool


def check_dependencies(spec: OperationSpec) -> DependencyReport:
    """Inspect declared dependencies without importing, executing, or installing anything."""
    checks = tuple(_check_one(dependency) for dependency in spec.dependencies)
    ready = all(
        check.state is DependencyState.PRESENT or check.dependency.optional for check in checks
    )
    return DependencyReport(operation_id=spec.operation_id, checks=checks, ready=ready)


def _check_one(dependency: DependencySpec) -> DependencyCheck:
    version_spec = _parse_version_spec(dependency)
    if isinstance(version_spec, str):
        return _unverified(dependency, version_spec)
    if dependency.kind is DependencyKind.PYTHON:
        return _check_python(dependency, version_spec)
    if dependency.kind is DependencyKind.EXECUTABLE:
        return _check_executable(dependency, version_spec)
    if dependency.kind is DependencyKind.MODEL:
        return _check_path(dependency, version_spec, requires_directory=False)
    if dependency.kind is DependencyKind.DATABASE:
        return _check_path(dependency, version_spec, requires_directory=True)
    return _unverified(dependency, "invalid_dependency_kind")


def _parse_version_spec(dependency: DependencySpec) -> SpecifierSet | str:
    if not dependency.version_spec:
        return SpecifierSet()
    try:
        return SpecifierSet(dependency.version_spec)
    except (InvalidSpecifier, TypeError, ValueError):
        return "invalid_version_spec"


def _check_python(dependency: DependencySpec, version_spec: SpecifierSet) -> DependencyCheck:
    if not _is_top_level_import_name(dependency.name):
        return _unverified(dependency, "invalid_python_import_name")
    try:
        found = find_spec(dependency.name)
    except (ImportError, AttributeError, ValueError):
        return _unverified(dependency, "python_locator_unavailable")
    if found is None:
        return DependencyCheck(dependency=dependency, state=DependencyState.MISSING)
    locator = dependency.locator or dependency.name
    if not dependency.version_spec:
        return DependencyCheck(
            dependency=dependency,
            state=DependencyState.PRESENT,
            resolved_locator=locator,
        )
    try:
        observed_version = version(locator)
    except PackageNotFoundError:
        return _unverified(dependency, "version_not_found", resolved_locator=locator)
    except (ImportError, OSError, ValueError):
        return _unverified(dependency, "version_unavailable", resolved_locator=locator)
    try:
        state = (
            DependencyState.PRESENT
            if version_spec.contains(observed_version)
            else DependencyState.VERSION_MISMATCH
        )
    except InvalidVersion:
        return _unverified(
            dependency,
            "invalid_observed_version",
            observed_version=observed_version,
            resolved_locator=locator,
        )
    return DependencyCheck(
        dependency=dependency,
        state=state,
        observed_version=observed_version,
        resolved_locator=locator,
    )


def _check_executable(dependency: DependencySpec, version_spec: SpecifierSet) -> DependencyCheck:
    try:
        locator = which(dependency.locator or dependency.name)
    except (OSError, ValueError):
        return _unverified(dependency, "executable_locator_unavailable")
    if locator is None:
        return DependencyCheck(dependency=dependency, state=DependencyState.MISSING)
    if dependency.version_spec:
        return _unverified(
            dependency,
            "version_requires_execution",
            resolved_locator=locator,
        )
    return DependencyCheck(
        dependency=dependency,
        state=DependencyState.PRESENT,
        resolved_locator=locator,
    )


def _check_path(
    dependency: DependencySpec,
    version_spec: SpecifierSet,
    *,
    requires_directory: bool,
) -> DependencyCheck:
    if not dependency.locator:
        return _unverified(dependency, "missing_locator")
    path = Path(dependency.locator)
    try:
        present = path.is_dir() if requires_directory else path.is_file()
    except OSError:
        return _unverified(dependency, "path_unavailable", resolved_locator=dependency.locator)
    if not present:
        return DependencyCheck(
            dependency=dependency,
            state=DependencyState.MISSING,
            resolved_locator=dependency.locator,
        )
    if dependency.version_spec:
        return _unverified(
            dependency,
            "version_unavailable",
            resolved_locator=dependency.locator,
        )
    return DependencyCheck(
        dependency=dependency,
        state=DependencyState.PRESENT,
        resolved_locator=dependency.locator,
    )


def _is_top_level_import_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def _unverified(
    dependency: DependencySpec,
    details: str,
    *,
    observed_version: str = "",
    resolved_locator: str = "",
) -> DependencyCheck:
    return DependencyCheck(
        dependency=dependency,
        state=DependencyState.UNVERIFIED,
        observed_version=observed_version,
        resolved_locator=resolved_locator,
        details=details,
    )
