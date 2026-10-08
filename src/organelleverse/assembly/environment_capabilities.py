"""Small task-scoped capability contracts derived from backend runtime facts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from organelleverse.assembly.environment_contracts import (
    BackendCapabilityContract,
    CapabilityItem,
    ResolvedBackendVersion,
)
from organelleverse.assembly.environment_specs import (
    PMAT_ORIENTATION_BANNER,
    AssemblyEnvironmentSpec,
)

__all__ = ["runtime_capabilities"]

_VersionStyle = Literal["semver", "major_minor", "strict_four_part"]


class CapabilityRuntime(Protocol):
    """Minimal structural view of a runtime needed to derive capabilities.

    The members are read-only so a runtime whose ``backend_id`` is a narrower
    ``Literal`` (e.g. :data:`AssemblyMethod`) still satisfies the protocol.
    """

    @property
    def backend_id(self) -> str: ...

    @property
    def environment_spec(self) -> AssemblyEnvironmentSpec: ...


def runtime_capabilities(
    runtime: CapabilityRuntime,
    effective_parameters: Mapping[str, object],
    resolved_version: ResolvedBackendVersion | None = None,
) -> BackendCapabilityContract:
    if runtime.backend_id == "pmat":
        items = _pmat_items(effective_parameters)
        if (
            resolved_version is not None
            and resolved_version.commit == "04534a2adf0c5309cb2e7478fd2bcc98ced3818c"
        ):
            items = tuple(
                item.model_copy(update={"required_version_text": PMAT_ORIENTATION_BANNER})
                if item.role == "pmat"
                else item
                for item in items
            )
    else:
        platform = runtime.environment_spec.platforms[0]
        items = tuple(
            CapabilityItem(
                role=runtime.backend_id if index == 0 else name,
                kind="executable",
                safe_names=(name,),
                version_argv=platform.version_argv if index == 0 else (),
                version_specifier=">=2.4,<2.5"
                if runtime.backend_id == "tippo" and index == 0
                else None,
                version_exit_codes=platform.version_exit_codes if index == 0 else (0,),
                version_style=_primary_version_style(runtime.backend_id, index),
            )
            for index, name in enumerate(platform.executable_names)
        )
    return BackendCapabilityContract(
        schema_version="organelleverse.backend-capabilities.v1",
        backend_id=runtime.backend_id,
        items=items,
    )


def _primary_version_style(backend_id: str, index: int) -> _VersionStyle:
    """Probe style for a derived capability item.

    Only the primary executable (``index == 0``) is ever probed. GetOrganelle
    uses the strict four-part style that parses ``GetOrganelle v1.7.7.1`` ->
    ``1.7.7.1``. Oatk keeps its major.minor style; every other backend and every
    dependency executable keeps the default three-part semver behavior.
    """
    if index != 0:
        return "semver"
    if backend_id == "getorganelle":
        return "strict_four_part"
    if backend_id == "oatk":
        return "major_minor"
    return "semver"


def _pmat_items(effective: Mapping[str, object]) -> tuple[CapabilityItem, ...]:
    items = [
        CapabilityItem(
            role="pmat",
            kind="executable",
            safe_names=("PMAT", "pmat"),
            version_argv=("--version",),
            version_specifier=">=2,<3",
        ),
        CapabilityItem(role="blastn", kind="executable", safe_names=("blastn",)),
        CapabilityItem(
            role="pmat_container",
            kind="resource",
            safe_names=("pmat/container/runAssembly.sif",),
        ),
        CapabilityItem(
            role="pmat_genomescope",
            kind="resource",
            safe_names=("pmat/lib/genomescope.R",),
        ),
        CapabilityItem(
            role="container_runtime",
            kind="host_provider",
            safe_names=("apptainer", "singularity"),
            version_argv=("--version",),
            host_scope="path",
        ),
    ]
    if effective.get("correction_task") == "run":
        items.append(CapabilityItem(role="canu", kind="executable", safe_names=("canu",)))
        if effective.get("correction_software") == "nextdenovo":
            items.append(
                CapabilityItem(
                    role="nextdenovo",
                    kind="executable",
                    safe_names=("nextDenovo", "nextdenovo"),
                )
            )
    return tuple(items)
