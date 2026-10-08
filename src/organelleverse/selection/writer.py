"""Canonical materialization boundary for selection results.

This is the selection half of the single publication boundary: the package
level ``organelleverse.writer.write`` dispatches a canonical
:class:`~organelleverse.core.result.OrganelleResult` to the suite that
produced it, exactly as it already does for ``assembly.``, ``annotation.``
and ``qc.`` results.

The per-operation materializers themselves (Ka/Ks table, aligned protein
FASTA, codon alignment, PAML alignment) are unchanged; this module only maps
``operation_id`` to a materializer and wraps the written file in a canonical
result.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult
from . import _contract

__all__ = ["materialize_result"]


def _materializers() -> dict[str, Callable[[OrganelleResult, str | Path], Path]]:
    """Import the suite materializers lazily, one per materializing operation."""

    from .codeml import _materialize_aligned_protein, _materialize_pal2nal, _materialize_paml
    from .kaks import write_kaks

    return {
        "selection.kaks": write_kaks,
        "selection.align_protein": _materialize_aligned_protein,
        "selection.pal2nal": _materialize_pal2nal,
        "selection.to_paml": _materialize_paml,
    }


def materialize_result(result: OrganelleResult, output: str | Path) -> OrganelleResult:
    """Materialize a non-failed selection result and return its written state."""

    if result.status == "failed":
        raise OrganelleInputError(
            code="output.failed_result",
            message="selection.write requires a non-failed selection result",
            details={"operation_id": result.operation_id},
        )
    materializer = _materializers().get(result.operation_id)
    if materializer is None:
        raise OrganelleInputError(
            code="output.unsupported_operation",
            message="selection.write accepts Results from materializing selection operations",
            details={"operation_id": result.operation_id},
        )
    path = Path(materializer(result, output))
    return OrganelleResult(
        operation_id="selection.write",
        operation_version=_contract.OPERATION_VERSION,
        scope=result.scope,
        status=result.status,
        summary_text=f"Materialized {result.operation_id} to {path.name}.",
        metrics={
            "file_count": 1,
            "source_result_object_id": result.object_id,
            "source_result_status": result.status,
        },
        flags=tuple(dict.fromkeys(("selection_materialized", *result.flags))),
        artifacts=(_contract.artifact(path),),
        provenance=result.provenance,
    )
