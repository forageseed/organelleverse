"""Plant organelle assembly, annotation, fetch, and QC."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .core import (
    ArtifactRef,
    ErrorDetail,
    Finding,
    LineageRecord,
    OperationSuggestion,
    OrganelleData,
    OrganelleGenome,
    OrganelleMetadata,
    OrganelleResult,
    ResultProvenance,
)
from .writer import write

if TYPE_CHECKING:
    from . import (
        annotation,
        assembly,
        assets,
        environments,
        fetch,
        io,
        operations,
        qc,
        report,
    )
    from .io import read

__version__ = "1.0.0"

#: Short human-facing names, in the spirit of OmicVerse's ``pp`` / ``pl``.
#:
#: Each is a lazy route to the full module, resolved to the *same object*:
#: ``ov.viz is ov.visualization``. There is no second implementation, no second
#: registry entry, and no second contract — operation ids stay full-length
#: (``visualization.synteny_matrix``), so agents and bundles are unaffected.
#: People get the short spelling; machines get the stable one.
_SHORT_MODULE_ALIASES = {
    "anno": "annotation",
    "asm": "assembly",
    "cmp": "comparative",
    "codon": "codon_composition",
    "div": "diversity",
    "editing": "rna_editing",
    "erc": "coevolution",
    "fmt": "format_conversion",
    "gc": "composition",
    "graph": "structure",
    "ir": "ir_boundary",
    "loc": "localization",
    "morph": "morphology",
    "pan": "pangenome",
    "pheno": "phenotype",
    "phylo": "phylogeny",
    "pop": "population",
    "splicing": "trans_splicing",
    "var": "variation",
    "viz": "visualization",
}

_PUBLIC_MODULES = frozenset(
    {
        "annotation",
        "assembly",
        "assets",
        "barcode",
        "capabilities",
        "codon_composition",
        "coevolution",
        "comparative",
        "composition",
        "diversity",
        "environments",
        "fetch",
        "format_conversion",
        "hgt",
        "io",
        "ir_boundary",
        "localization",
        "morphology",
        "operations",
        "pangenome",
        "phenotype",
        "phylogeny",
        "population",
        "qc",
        "report",
        "rna_editing",
        "selection",
        "structure",
        "trans_splicing",
        "transfer",
        "variation",
        "visualization",
    }
)

__all__ = [
    "ArtifactRef",
    "ErrorDetail",
    "Finding",
    "LineageRecord",
    "OperationSuggestion",
    "OrganelleData",
    "OrganelleGenome",
    "OrganelleMetadata",
    "OrganelleResult",
    "ResultProvenance",
    "__version__",
    "anno",
    "annotation",
    "asm",
    "assembly",
    "assets",
    "barcode",
    "cmp",
    "codon",
    "codon_composition",
    "coevolution",
    "comparative",
    "composition",
    "div",
    "diversity",
    "editing",
    "environments",
    "erc",
    "fetch",
    "fmt",
    "format_conversion",
    "gc",
    "hgt",
    "io",
    "ir",
    "ir_boundary",
    "loc",
    "localization",
    "morph",
    "morphology",
    "operations",
    "pan",
    "pangenome",
    "pheno",
    "phenotype",
    "phylo",
    "phylogeny",
    "pop",
    "population",
    "qc",
    "read",
    "report",
    "rna_editing",
    "selection",
    "splicing",
    "structure",
    "trans_splicing",
    "transfer",
    "var",
    "variation",
    "visualization",
    "viz",
    "write",
]

_PUBLIC_ATTRIBUTES = {"read": (".io", "read")}


def __getattr__(name: str) -> object:
    if name in _SHORT_MODULE_ALIASES:
        value = import_module(f".{_SHORT_MODULE_ALIASES[name]}", __name__)
    elif name in _PUBLIC_MODULES:
        value = import_module(f".{name}", __name__)
    elif name in _PUBLIC_ATTRIBUTES:
        module_name, attribute_name = _PUBLIC_ATTRIBUTES[name]
        value = getattr(import_module(module_name, __name__), attribute_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_PUBLIC_MODULES, *_SHORT_MODULE_ALIASES, *_PUBLIC_ATTRIBUTES})
