"""Resolve exact executable identities for assembly-QC evidence collection.

Resolution is deliberately small and ordered: a complete trusted installed
toolset, then a hash-bound entry in the shared installation registry, then an
explicit managed resolver supplied by the caller.  This module does not invent
an environment lock when the package has no pinned QC environment resource.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import Field, field_serializer

from organelleverse.assembly.environment_registry import InstallationRegistry
from organelleverse.core.base import StrictFrozenModel
from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError
from organelleverse.core.external import MESSAGE_TAIL_LINES, failure_details, tail_lines

from .contracts import ToolIdentity

QcEnvironmentSource = Literal["installed", "registered", "managed"]
QcToolName = Literal["minimap2", "samtools", "meryl"]


class QcExecutable(StrictFrozenModel[Literal["qc_executable"]]):
    """One executable accepted at the QC trust boundary."""

    kind: Literal["qc_executable"] = "qc_executable"
    name: QcToolName
    path: Path
    version: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: QcEnvironmentSource

    @field_serializer("path")
    def _serialize_path(self, value: Path) -> str:
        return str(value)


class QcEnvironment(StrictFrozenModel[Literal["qc_environment"]]):
    """A complete minimap2/samtools QC execution environment."""

    kind: Literal["qc_environment"] = "qc_environment"
    environment_id: str
    source: QcEnvironmentSource
    minimap2: QcExecutable
    samtools: QcExecutable
    meryl: QcExecutable | None = None

    @property
    def tool_identities(self) -> tuple[ToolIdentity, ...]:
        components = (self.minimap2, self.samtools) + (
            (self.meryl,) if self.meryl is not None else ()
        )
        return tuple(
            ToolIdentity(
                name=component.name,
                version=component.version,
                executable_sha256=component.sha256,
                environment_id=self.environment_id,
            )
            for component in components
        )


ManagedQcResolver = Callable[[], QcEnvironment]


def mapper_preset(technology: str, quality_state: str | None) -> str:
    """Return the minimap2 preset for declared sequencing metadata.

    File names and FASTQ contents are intentionally ignored.  Incoherent or
    unknown metadata fails closed before a command is launched.
    """

    normalized_state = quality_state or ""
    presets = {
        ("illumina", ""): "sr",
        ("pacbio_hifi", "ccs"): "map-hifi",
        ("pacbio_clr", "raw"): "map-pb",
        ("pacbio_clr", "corrected"): "map-pb",
        ("ont", "raw"): "map-ont",
        ("ont", "corrected"): "map-ont",
        ("ont", "hq"): "lr:hq",
        ("ont", "duplex"): "lr:hq",
    }
    preset = presets.get((technology, normalized_state))
    if preset is None:
        raise OrganelleInputError(
            code="qc.unsupported_technology",
            message="sequencing technology and quality state do not select a mapper preset",
            details={"technology": technology, "quality_state": quality_state},
        )
    return preset


def resolve_qc_environment(
    *,
    installed: Mapping[str, str | Path] | None = None,
    registry_root: Path | None = None,
    managed_resolver: ManagedQcResolver | None = None,
) -> QcEnvironment:
    """Resolve a complete QC tool environment in the documented order.

    ``installed=None`` performs PATH discovery.  Passing an explicit mapping,
    including an empty mapping, makes discovery deterministic for tests and
    Agent-managed callers.
    """

    discovered = _discover_installed(installed)
    if {"minimap2", "samtools"} <= discovered.keys():
        return _environment_from_paths(discovered, source="installed")

    registry = InstallationRegistry(tool_root=registry_root)
    for provider in registry.list():
        components = {
            component.role: component
            for component in provider.components
            if component.kind == "executable"
        }
        if not {"minimap2", "samtools"} <= components.keys():
            continue
        paths: dict[str, Path] = {}
        valid = True
        for name in ("minimap2", "samtools", "meryl"):
            component = components.get(name)
            if component is None:
                continue
            path = component.path.resolve()
            if (
                not _is_executable(path)
                or _sha256(path) != component.sha256
                or (component.version and _probe_version(name, path) != component.version)
            ):
                valid = False
                break
            paths[name] = path
        if valid:
            return _environment_from_paths(paths, source="registered")

    if managed_resolver is not None:
        managed = managed_resolver()
        if managed.source != "managed":
            raise OrganelleDependencyError(
                code="qc.dependency_unavailable",
                message="managed QC resolver returned a non-managed environment",
                details={"source": managed.source},
            )
        _verify_environment(managed)
        return managed

    raise OrganelleDependencyError(
        code="qc.dependency_unavailable",
        message="minimap2 and samtools are required for assembly QC",
        details={
            "required": ["minimap2", "samtools"],
            "resolution_order": ["installed", "registered", "managed"],
            "managed_environment_bundled": False,
        },
        retryable=True,
        suggested_action={
            "install": ["minimap2", "samtools"],
            "register_verified_environment": True,
        },
    )


def _discover_installed(
    supplied: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    if supplied is None:
        found: dict[str, Path] = {}
        for name in ("minimap2", "samtools", "meryl"):
            candidate = shutil.which(name)
            if candidate is not None:
                found[name] = Path(candidate)
        return found
    return {
        name: Path(path)
        for name, path in supplied.items()
        if name in {"minimap2", "samtools", "meryl"}
    }


def _environment_from_paths(
    paths: Mapping[str, Path], *, source: QcEnvironmentSource
) -> QcEnvironment:
    components: dict[str, QcExecutable] = {}
    for name in ("minimap2", "samtools", "meryl"):
        candidate = paths.get(name)
        if candidate is None:
            continue
        resolved = candidate.expanduser().resolve()
        if not _is_executable(resolved):
            if name == "meryl":
                continue
            raise OrganelleDependencyError(
                code="qc.dependency_unavailable",
                message=f"QC executable is missing or not executable: {resolved}",
                details={"tool": name, "path": str(resolved)},
            )
        components[name] = QcExecutable(
            name=name,  # type: ignore[arg-type]
            path=resolved,
            version=_probe_version(name, resolved),
            sha256=_sha256(resolved),
            source=source,
        )

    if "minimap2" not in components or "samtools" not in components:
        raise OrganelleDependencyError(
            code="qc.dependency_unavailable",
            message="a complete QC environment requires minimap2 and samtools",
            details={"resolved": sorted(components)},
        )

    identity_payload = [
        {
            "name": component.name,
            "version": component.version,
            "sha256": component.sha256,
        }
        for component in sorted(components.values(), key=lambda item: item.name)
    ]
    digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return QcEnvironment(
        environment_id=f"qc-environment:sha256:{digest}",
        source=source,
        minimap2=components["minimap2"],
        samtools=components["samtools"],
        meryl=components.get("meryl"),
    )


def _verify_environment(environment: QcEnvironment) -> None:
    for component in (environment.minimap2, environment.samtools) + (
        (environment.meryl,) if environment.meryl is not None else ()
    ):
        path = component.path.resolve()
        if (
            not _is_executable(path)
            or _sha256(path) != component.sha256
            or _probe_version(component.name, path) != component.version
        ):
            raise OrganelleDependencyError(
                code="qc.dependency_unavailable",
                message="managed QC component does not match its recorded identity",
                details={"tool": component.name, "path": str(path)},
            )


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_version(name: str, path: Path) -> str:
    argv = (str(path), "--version")
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        message = f"unable to probe {name}"
        tail = tail_lines(error.stderr, MESSAGE_TAIL_LINES)
        if tail:
            message = f"{message}: {tail}"
        raise OrganelleDependencyError(
            code="qc.dependency_unavailable",
            message=message,
            details=failure_details(
                argv,
                None,
                stdout=error.stdout,
                stderr=error.stderr,
                extra={"tool": name, "path": str(path), "reason": str(error)},
            ),
        ) from error
    except (OSError, subprocess.SubprocessError) as error:
        raise OrganelleDependencyError(
            code="qc.dependency_unavailable",
            message=f"unable to probe {name}",
            details={"tool": name, "path": str(path), "reason": str(error)},
        ) from error
    output = completed.stdout.strip() or completed.stderr.strip()
    if completed.returncode != 0 or not output:
        message = f"{name} version probe failed"
        tail = tail_lines(completed.stderr, MESSAGE_TAIL_LINES)
        if tail:
            message = f"{message}: {tail}"
        raise OrganelleDependencyError(
            code="qc.dependency_unavailable",
            message=message,
            details=failure_details(
                argv,
                completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                extra={"tool": name, "path": str(path)},
            ),
        )
    first_line = output.splitlines()[0].strip()
    if name == "samtools" and first_line.lower().startswith("samtools "):
        return first_line.split(maxsplit=1)[1]
    return first_line


__all__ = [
    "ManagedQcResolver",
    "QcEnvironment",
    "QcEnvironmentSource",
    "QcExecutable",
    "mapper_preset",
    "resolve_qc_environment",
]
