"""Lazy public facade for the four released assembly backends."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import (
        AssemblyEnvironmentHint,
        assemble,
        pmat_continue,
        pmat_graph_build,
        write,
    )
    from .contracts import (
        AssemblyAuxiliary,
        AssemblyInputPayload,
        ContigInput,
        GenomeSizeEvidence,
        LongReadLibrary,
        ShortReadLibrary,
    )
    from .install import check_all_backends, check_backend, install_backend
    from .routing import AssemblyRoute, classify_profile, compatible_backends, resolve_backend

__all__ = [
    "AssemblyAuxiliary",
    "AssemblyEnvironmentHint",
    "AssemblyInputPayload",
    "AssemblyRoute",
    "ContigInput",
    "GenomeSizeEvidence",
    "LongReadLibrary",
    "ShortReadLibrary",
    "assemble",
    "check_all_backends",
    "check_backend",
    "classify_profile",
    "compatible_backends",
    "install_backend",
    "pmat_continue",
    "pmat_graph_build",
    "resolve_backend",
    "write",
]

_LAZY_ATTRIBUTES = {
    "AssemblyAuxiliary": (".contracts", "AssemblyAuxiliary"),
    "AssemblyEnvironmentHint": (".api", "AssemblyEnvironmentHint"),
    "AssemblyInputPayload": (".contracts", "AssemblyInputPayload"),
    "AssemblyRoute": (".routing", "AssemblyRoute"),
    "ContigInput": (".contracts", "ContigInput"),
    "GenomeSizeEvidence": (".contracts", "GenomeSizeEvidence"),
    "LongReadLibrary": (".contracts", "LongReadLibrary"),
    "ShortReadLibrary": (".contracts", "ShortReadLibrary"),
    "assemble": (".api", "assemble"),
    "check_all_backends": (".install", "check_all_backends"),
    "check_backend": (".install", "check_backend"),
    "classify_profile": (".routing", "classify_profile"),
    "compatible_backends": (".routing", "compatible_backends"),
    "install_backend": (".install", "install_backend"),
    "pmat_continue": (".api", "pmat_continue"),
    "pmat_graph_build": (".api", "pmat_graph_build"),
    "resolve_backend": (".routing", "resolve_backend"),
    "write": (".api", "write"),
}


def __getattr__(name: str) -> object:
    lazy_attribute = _LAZY_ATTRIBUTES.get(name)
    if lazy_attribute is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = lazy_attribute
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_ATTRIBUTES})
