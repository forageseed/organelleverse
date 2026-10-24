"""``ov.read``: one entry that hands the file to a module's own ``read``.

Every module that accepts files exposes ``ov.<module>.read(path, ...)`` with its own
signature and returns the object that module's functions take. ``ov.read(path,
module="phylogeny")`` is the same call spelled at the top level; the remaining
keyword arguments go to the module unchanged. Without ``module`` the call is routed
by what it states: genome options mean ``annotation``, reads options mean
``assembly``.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from .core.errors import OrganelleInputError
from .io_genome import ReadFormat
from .inputs import read_contract
from .io_reads import _FASTQ_SUFFIXES

#: Public modules that publish ``read``, with the module that implements it.
_READERS: dict[str, str] = {
    "annotation": "organelleverse.annotation",
    "assembly": "organelleverse.assembly",
    "phylogeny": "organelleverse.phylogeny",
    "diversity": "organelleverse.diversity",
    "variation": "organelleverse.variation",
    "barcode": "organelleverse.barcode",
    "selection": "organelleverse.selection",
    "codon_composition": "organelleverse.codon_composition",
    "composition": "organelleverse.composition",
    "localization": "organelleverse.localization",
    "hgt": "organelleverse.hgt",
    "phenotype": "organelleverse.phenotype",
    "transfer": "organelleverse.transfer",
    "population": "organelleverse.population",
    "structure": "organelleverse.structure",
    "comparative": "organelleverse.comparative",
    "rna_editing": "organelleverse.rna_editing",
    "pangenome": "organelleverse.pangenome",
    "ir_boundary": "organelleverse.ir_boundary",
    "trans_splicing": "organelleverse.trans_splicing",
    "format_conversion": "organelleverse.format_conversion",
    "morphology": "organelleverse.morphology",
    "visualization": "organelleverse.visualization",
}

#: Options that only a genome read takes / only a reads read takes.
_GENOME_OPTIONS = (
    "format",
    "organelle",
    "species",
    "accession",
    "genetic_code",
    "assembly_type",
    "plastid_type",
    "source",
)
_READS_OPTIONS = (
    "technology",
    "quality_state",
    "read_length",
    "short_libraries",
    "long_libraries",
    "contig_inputs",
    "auxiliary",
)


def readable_modules() -> tuple[str, ...]:
    """The modules whose ``read`` ``ov.read(..., module=...)`` can reach."""
    return tuple(_READERS)


def _resolve_module(name: str) -> str:
    from . import _SHORT_MODULE_ALIASES

    resolved = _SHORT_MODULE_ALIASES.get(name, name)
    if resolved not in _READERS:
        raise OrganelleInputError(
            code="input.unknown_read_module",
            message=f"module {name!r} has no read; choose one of {', '.join(_READERS)}",
            details={"module": name, "readable_modules": list(_READERS)},
        )
    return resolved


def _given(options: dict[str, Any], names: tuple[str, ...]) -> bool:
    """True when one of ``names`` carries a value (empty strings and tuples do not)."""
    return any(options.get(name) not in (None, "", ()) for name in names)


def _infer_module(path: str | Path | None, options: dict[str, Any]) -> str:
    genome = _given(options, _GENOME_OPTIONS)
    reads = _given(options, _READS_OPTIONS) or "auxiliary" in options
    if genome and reads:
        raise OrganelleInputError(
            code=(
                "input.reads_shorthand_conflict"
                if options.get("technology") is not None
                else "input.ambiguous_read"
            ),
            message="genome options and sequencing-read options cannot be combined in one read",
        )
    if reads or path is None and not genome:
        return "assembly"
    if path is None:
        raise OrganelleInputError(
            code="input.sequence_parameters_without_path",
            message="Sequence format, organelle, and genome metadata require a sequence path",
        )
    if genome:
        return "annotation"
    if isinstance(path, (str, Path)) and Path(path).name.lower().endswith(_FASTQ_SUFFIXES):
        return "assembly"
    raise OrganelleInputError(
        code="input.module_required",
        message=(
            "ov.read needs module= for this file: "
            + ", ".join(f"module={name!r}" for name in _READERS)
            + " (a genome is read with organelle=, reads with technology=)"
        ),
        details={"readable_modules": list(_READERS)},
    )


def read(
    path: str | Path | None = None,
    *,
    module: str | None = None,
    **options: Any,
) -> Any:
    """Read ``path`` with the ``read`` of ``module``.

    ``ov.read("genes.fasta", module="phylogeny")`` is ``ov.phylogeny.read("genes.fasta")``.
    ``ov.read("mt.fasta", organelle="mito", species="...")`` reads a genome and
    ``ov.read("reads.fq.gz", technology="hifi")`` reads sequencing reads; both name their
    module by their options. A ``.json`` manifest written by ``ov.write`` loads the saved
    result (or genome, or data) without a module. ``options`` are the module's own arguments, see
    ``ov.<module>.read``. :func:`readable_modules` lists the modules.
    """
    if module is None and not options and path is not None and Path(path).suffix.lower() == ".json":
        return read_contract(path)
    name = _resolve_module(module) if module is not None else _infer_module(path, options)
    reader = import_module(_READERS[name]).read
    return reader(**options) if path is None else reader(path, **options)


__all__ = ["ReadFormat", "read", "readable_modules"]
