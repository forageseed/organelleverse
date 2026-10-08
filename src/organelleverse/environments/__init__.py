"""Public read-only facade over the persistent installation registry.

``ov.environments.list()`` and ``ov.environments.locate(backend)`` expose which
backend providers have been installed and verified. The facade is lazy and
read-only: it materializes no directories, writes no files, and only reads the
registry under the default tool root. Provider records are published by the
managed-install path (Task 3); until then the facade reports whatever the
registry already holds.

This package also provides execution-context path translation helpers used by
the plugin worker to route file parameters between host, WSL and container
contexts.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from organelleverse.assembly.environment_registry import (
    InstallationRegistry,
    RegisteredProvider,
)
from organelleverse.operations.spec import StrictSpecModel

from .path_translation import (
    TranslatedPath,
    translate_path,
    validate_path_accessible,
)

__all__ = [
    "EnvironmentRecord",
    "TranslatedPath",
    "list",
    "locate",
    "locate_executable",
    "translate_path",
    "validate_path_accessible",
]


class EnvironmentRecord(StrictSpecModel):
    """A single registered environment, projected for human and tool use."""

    backend_id: str
    prefix: Path | None
    executables: tuple[Path, ...]
    version: str | None
    provider_digest: str
    source: str
    activation: str


def _registry() -> InstallationRegistry:
    return InstallationRegistry()


def _to_record(provider: RegisteredProvider) -> EnvironmentRecord:
    executables = tuple(
        component.path
        for component in provider.components
        if component.kind in ("executable", "host_provider")
    )
    activation = f"conda activate {provider.prefix}" if provider.prefix is not None else ""
    return EnvironmentRecord(
        backend_id=provider.backend_id,
        prefix=provider.prefix,
        executables=executables,
        version=provider.version,
        provider_digest=provider.provider_digest,
        source=provider.source,
        activation=activation,
    )


def list() -> tuple[EnvironmentRecord, ...]:
    """Return every registered environment. Lazy and read-only."""
    return tuple(_to_record(provider) for provider in _registry().list())


def locate(backend: str) -> tuple[EnvironmentRecord, ...]:
    """Return registered environments for *backend*. Lazy and read-only."""
    return tuple(_to_record(provider) for provider in _registry().locate(backend))


def locate_executable(name: str) -> Path | None:
    """Locate one executable on PATH without running it or probing a version."""
    located = shutil.which(name)
    return Path(located).resolve() if located else None
