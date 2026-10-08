"""Low-level format readers that no released operation owns yet.

This module is deliberately minimal. Its predecessor was a 922-line grab bag
that imported the pre-v1 dataclass contracts and reached into the short-alias
facades (``erc``, ``anno``, ``pan``); both are gone and must stay gone. Only the
readers a restored suite actually calls live here, ported to import nothing but
the standard library and their own parser.

New format support belongs in ``organelleverse.io`` behind a registered
operation. This module exists so restored science is not blocked on that work,
not as a second I/O surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast

from organelleverse.core.errors import OrganelleDependencyError, OrganelleInputError

__all__ = ["read_vcf"]


class _Substitution(Protocol):
    """The one alternate-allele field this reader inspects."""

    @property
    def value(self) -> object: ...


class _Call(Protocol):
    """One sample's genotype call on a record."""

    @property
    def sample(self) -> object: ...
    @property
    def data(self) -> dict[str, object]: ...


class _Record(Protocol):
    """The subset of a VCF record this reader depends on."""

    @property
    def REF(self) -> object: ...
    @property
    def ALT(self) -> list[_Substitution]: ...
    @property
    def POS(self) -> int: ...
    @property
    def calls(self) -> list[_Call]: ...


class _Reader(Protocol):
    """The subset of ``vcfpy.Reader`` this reader depends on."""

    def __iter__(self) -> Iterator[_Record]: ...
    def close(self) -> None: ...


def read_vcf(path: str | Path) -> dict[int, dict[str, str]]:
    """Read biallelic SNP genotypes from a VCF file.

    Indels and multi-allelic records are skipped: the population statistics that
    consume this reader are defined over biallelic sites only, so admitting the
    rest would silently change what is being measured.

    Arguments:
        path: The VCF file to read. Compression is handled by the parser.

    Returns:
        A mapping of position to ``{sample: genotype}``, where genotype is the
        raw ``GT`` string (``"./."`` when the call carries none).

    Raises:
        OrganelleDependencyError: ``vcfpy`` is not installed.
        OrganelleInputError: The file is missing or cannot be parsed as VCF.
    """
    try:
        import vcfpy
    except ImportError as error:  # pragma: no cover - exercised only without vcfpy
        raise OrganelleDependencyError(
            code="io.dependency_unavailable",
            message="Reading VCF requires the 'vcfpy' package",
            details={"dependency": "vcfpy", "path": str(path)},
        ) from error

    source = Path(path)
    if not source.is_file():
        raise OrganelleInputError(
            code="input.missing_artifact",
            message=f"VCF file does not exist: {source}",
            details={"path": str(source)},
        )

    snps: dict[int, dict[str, str]] = {}
    try:
        reader = cast("_Reader", vcfpy.Reader.from_path(str(source)))
    except Exception as error:
        raise OrganelleInputError(
            code="input.malformed_vcf",
            message=f"Unreadable VCF file: {source}",
            details={"path": str(source), "error": str(error)},
        ) from error

    try:
        for record in reader:
            if len(str(record.REF)) != 1:
                continue
            alt = record.ALT
            if len(alt) != 1 or len(str(alt[0].value)) != 1:
                continue
            snps[int(record.POS)] = {
                str(call.sample): str(call.data.get("GT", "./.")) for call in record.calls
            }
    except Exception as error:
        raise OrganelleInputError(
            code="input.malformed_vcf",
            message=f"Malformed record in VCF file: {source}",
            details={"path": str(source), "error": str(error)},
        ) from error
    finally:
        reader.close()

    return snps
