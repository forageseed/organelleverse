# pyright: reportUnsupportedDunderAll=false
# Registry and spec initialize before base because AdapterContext resolves forward
# references through routing, which relies on BACKENDS being defined. Base and runtime
# stay lazy so importing lightweight contracts cannot re-enter this package through
# routing while it is only partially initialized. Static imports under TYPE_CHECKING
# preserve exact exported types without affecting that runtime boundary.
from typing import TYPE_CHECKING

from .registry import (
    BACKENDS,
    AssemblyBackendRegistry,
    backend_ids,
    get_backend,
    install_info,
)
from .spec import (
    AssemblyBackendSpec,
    AssemblyEnvironmentRef,
    AssemblyProfile,
    AuxiliaryRole,
    BackendInstallSpec,
    EnvironmentCarrier,
    OrganelleType,
)

if TYPE_CHECKING:
    from .base import (
        AdapterContext,
        AssemblyAdapter,
        AssemblyCommand,
        BackendOutput,
        ExpectedBackendResources,
        NormalizedAssemblyOutputs,
        PreparedBackendResources,
        RawAssemblyOutputs,
    )
    from .runtime import (
        RUNTIMES,
        AssemblyResourceProvider,
        AssemblyRuntimeRegistry,
        BackendRuntime,
    )

# ---------------------------------------------------------------------------
# Lazy imports for base and runtime (avoid circular import through _rebuild)
# ---------------------------------------------------------------------------

_base_loaded = False
_runtime_loaded = False

_BASE_NAMES = frozenset(
    {
        "AdapterContext",
        "AssemblyAdapter",
        "AssemblyCommand",
        "BackendOutput",
        "ExpectedBackendResources",
        "NormalizedAssemblyOutputs",
        "PreparedBackendResources",
        "RawAssemblyOutputs",
    }
)

_RUNTIME_NAMES = frozenset(
    {
        "AssemblyResourceProvider",
        "AssemblyRuntimeRegistry",
        "BackendRuntime",
        "RUNTIMES",
    }
)


def __getattr__(name: str) -> object:
    if name in _BASE_NAMES:
        global _base_loaded
        if not _base_loaded:
            from . import base as _base_mod

            for _n in _BASE_NAMES:
                globals()[_n] = getattr(_base_mod, _n)
            _base_loaded = True
        return globals()[name]
    if name in _RUNTIME_NAMES:
        global _runtime_loaded
        if not _runtime_loaded:
            from . import runtime as _runtime_mod

            for _n in _RUNTIME_NAMES:
                globals()[_n] = getattr(_runtime_mod, _n)
            _runtime_loaded = True
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BACKENDS",
    "RUNTIMES",
    "AdapterContext",
    "AssemblyAdapter",
    "AssemblyBackendRegistry",
    "AssemblyBackendSpec",
    "AssemblyCommand",
    "AssemblyEnvironmentRef",
    "AssemblyProfile",
    "AssemblyResourceProvider",
    "AssemblyRuntimeRegistry",
    "AuxiliaryRole",
    "BackendInstallSpec",
    "BackendOutput",
    "BackendRuntime",
    "EnvironmentCarrier",
    "ExpectedBackendResources",
    "NormalizedAssemblyOutputs",
    "OrganelleType",
    "PreparedBackendResources",
    "RawAssemblyOutputs",
    "backend_ids",
    "get_backend",
    "install_info",
]
