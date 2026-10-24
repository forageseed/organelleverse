"""File-backed inputs other than sequences, and the ``read`` each module publishes.

A read names the file's format and checks that the file is one; it never decides what
the data are for. ``make_reader`` builds a module's ``read`` from the formats that
module's functions take. A module whose functions also take a genome adds
``genome=True``: ``organelle=`` then reads the file as an ``OrganelleGenome``.
"""

from __future__ import annotations

import gzip
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core.errors import OrganelleInputError
from .sequences import Sequences
from .sequences import read as read_sequences

_IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
_HEAD_BYTES = 8192


@dataclass(frozen=True)
class Variants:
    """A VCF file and the samples named in its header."""

    path: Path
    samples: tuple[str, ...]


@dataclass(frozen=True)
class ReadAlignments:
    """A BAM or CRAM file of reads aligned to a reference."""

    path: Path
    format: str


@dataclass(frozen=True)
class Images:
    """Image or label-map files, in the order given."""

    paths: tuple[Path, ...]


@dataclass(frozen=True)
class Tree:
    """A phylogenetic tree file in Newick or NEXUS format."""

    path: Path
    format: str


@dataclass(frozen=True)
class Graph:
    """A GFA assembly or pangenome graph file."""

    path: Path


def _missing(path: Path) -> OrganelleInputError:
    return OrganelleInputError(
        code="input.missing_artifact",
        message=f"Artifact does not exist: {path.name}",
        details={"basename": path.name},
    )


def _is_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def _head(path: Path) -> bytes:
    """The first bytes of the file, decompressed when it is gzip or BGZF (as BAM is)."""
    opener = gzip.open if _is_gzip(path) else open
    with opener(path, "rb") as handle:  # type: ignore[operator]
        return handle.read(_HEAD_BYTES)


def detect_kind(path: str | Path) -> str:
    """``fasta``, ``vcf``, ``bam``, ``cram``, ``tree``, ``gfa`` or ``image``, from the file itself."""
    candidate = Path(path)
    if not candidate.is_file():
        raise _missing(candidate)
    if candidate.name.lower().endswith(_IMAGE_SUFFIXES):
        return "image"
    head = _head(candidate)
    if head[:4] == b"BAM\x01":
        return "bam"
    if head[:4] == b"CRAM":
        return "cram"
    text = head.decode("utf-8", errors="ignore").lstrip()
    if text.startswith(">"):
        return "fasta"
    if text.startswith("##fileformat=VCF"):
        return "vcf"
    if text.startswith("(") or text[:6].upper() == "#NEXUS":
        return "tree"
    if text.startswith(("H\t", "S\t")):
        return "gfa"
    raise OrganelleInputError(
        code="input.unknown_file_format",
        message="Input is not a recognizable FASTA, VCF, BAM, CRAM, tree, GFA or image file",
        details={"basename": candidate.name},
    )


def read_contract(path: str | Path) -> Any:
    """Load a saved ``OrganelleResult``, ``OrganelleGenome`` or ``OrganelleData`` manifest."""
    import json

    from .core.serialization import load_data, load_genome, load_result

    candidate = Path(path)
    if not candidate.is_file():
        raise _missing(candidate)
    try:
        kind = json.loads(candidate.read_text(encoding="utf-8")).get("kind")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        kind = None
    loaders = {"result": load_result, "genome": load_genome, "data": load_data}
    if kind not in loaders:
        raise OrganelleInputError(
            code="input.not_a_contract_manifest",
            message=f"Not a saved result, genome or data manifest: {candidate.name}",
            details={"basename": candidate.name},
        )
    loaded = loaders[kind](candidate)
    if kind == "result":
        base = candidate.resolve().parent
        loaded = loaded.model_copy(
            update={
                "artifacts": tuple(
                    a.model_copy(update={"uri": str(a.resolve(base))}) for a in loaded.artifacts
                )
            }
        )
    return loaded


def read_variants(path: str | Path) -> Variants:
    """Read a VCF (plain or gzip) and collect its sample names."""
    candidate = Path(path)
    opener = gzip.open if _is_gzip(candidate) else open
    with opener(candidate, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            if line.startswith("#CHROM"):
                return Variants(candidate, tuple(line.rstrip("\r\n").split("\t")[9:]))
    raise OrganelleInputError(
        code="input.invalid_variants",
        message=f"VCF has no #CHROM header line: {candidate.name}",
        details={"basename": candidate.name},
    )


def _read_images(paths: Sequence[str | Path]) -> Images:
    files = tuple(Path(p) for p in paths)
    for file in files:
        if not file.is_file():
            raise _missing(file)
        if not file.name.lower().endswith(_IMAGE_SUFFIXES):
            raise OrganelleInputError(
                code="input.unknown_file_format",
                message=f"Not an image file ({', '.join(_IMAGE_SUFFIXES)}): {file.name}",
                details={"basename": file.name},
            )
    return Images(files)


_BUILD: dict[str, Callable[[Path], Any]] = {
    "fasta": read_sequences,
    "vcf": read_variants,
    "bam": lambda path: ReadAlignments(path, "bam"),
    "cram": lambda path: ReadAlignments(path, "cram"),
    "tree": lambda path: Tree(path, "nexus" if _head(path)[:6].upper() == b"#NEXUS" else "newick"),
    "gfa": Graph,
}

#: What each accepted kind is called when it is offered by a module.
_KIND_GROUPS = {"bam": ("bam", "cram")}


def make_reader(module: str, *kinds: str, genome: bool = False) -> Callable[..., Any]:
    """Build ``ov.<module>.read`` for the formats ``kinds`` (and a genome when ``genome``)."""
    accepted = tuple(k for kind in kinds for k in _KIND_GROUPS.get(kind, (kind,)))

    def read_file(path: Any) -> Any:
        if "image" in accepted and not isinstance(path, (str, Path)):
            return _read_images(path)
        kind = detect_kind(path)
        if kind not in accepted:
            wanted = ", ".join(kinds + (("a genome with organelle=",) if genome else ()))
            raise OrganelleInputError(
                code="input.unsupported_read_format",
                message=f"ov.{module}.read takes {wanted}; this file is {kind}",
                details={"module": module, "found": kind},
            )
        if kind == "image":
            return _read_images([path])
        return _BUILD[kind](Path(path))

    offered = ", ".join(kinds)
    if not genome:
        reader = read_file
        reader.__doc__ = f"Read a file for ``ov.{module}``: {offered}. The format comes from the file."
    else:

        def reader(path: Any, *, organelle: str | None = None, **genome_options: Any) -> Any:
            if organelle is not None:
                from .io_genome import read_genome

                return read_genome(path, organelle=organelle, **genome_options)
            if genome_options:
                raise OrganelleInputError(
                    code="input.genome_options_without_organelle",
                    message="species, format and the other genome options need organelle=",
                )
            return read_file(path)

        reader.__doc__ = (
            f"Read a file for ``ov.{module}``: "
            + (f"{offered}, or, with " if offered else "with ")
            + "``organelle=``, a genome (``species=`` and the other genome options)."
        )
    reader.__name__ = "read"
    reader.__qualname__ = "read"
    reader.__module__ = f"organelleverse.{module}"
    return reader
