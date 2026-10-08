"""Canonical publication boundary for coevolution results.

This mirrors ``annotation/writer.py`` and ``quality_control/writer.py``: the
single ``materialize_result`` seam that ``organelleverse.writer.write`` routes
``coevolution.*`` results to. Callers should use :func:`write`, never the
legacy ``module_writer`` factory.
"""

from __future__ import annotations

from pathlib import Path

from ..core.errors import OrganelleInputError
from ..core.result import OrganelleResult

__all__ = ["materialize_result", "write"]


def materialize_result(result: OrganelleResult, output: str | Path) -> Path:
    """Materialize a non-failed canonical coevolution result."""

    if not isinstance(result, OrganelleResult):
        raise OrganelleInputError(
            code="output.unsupported_value",
            message="coevolution.write accepts a canonical OrganelleResult",
            details={"type": type(result).__name__},
        )
    if result.status == "failed":
        raise OrganelleInputError(
            code="output.failed_result",
            message="coevolution.write requires a non-failed result",
            details={"operation_id": result.operation_id, "status": result.status},
        )
    if result.operation_id == "coevolution.reconcile_trees":
        from .coevolution import _materialize_reconcile_trees

        return _materialize_reconcile_trees(result, output)
    raise OrganelleInputError(
        code="output.unsupported_operation",
        message="coevolution.write accepts results from materializing coevolution operations",
        details={"operation_id": result.operation_id},
    )


def write(result: OrganelleResult, *, output: str | Path) -> Path:
    """Atomically materialize a non-failed canonical coevolution result."""

    return materialize_result(result, output)
