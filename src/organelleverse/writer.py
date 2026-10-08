"""The single publication boundary for canonical v1 result values.

Scientific operations compute into a managed run store and return an immutable
:class:`~organelleverse.core.result.OrganelleResult`. Choosing a user-facing
destination is a separate, explicit action, and this module is the only place
that action happens.

Which suites can materialize is **resolved**, not hardcoded. A suite becomes
writable by providing ``materialize_result(result, output) -> OrganelleResult``
in ``<suite>/writer.py`` (or ``<suite>/<suite>.py``); nothing here changes. That
keeps the cost of adding a capability at the edge instead of in the core, which
is the whole point of the operation contract.

Resolution is not a fallback: a result whose suite provides no materializer is
refused with a typed error naming the suites that do. Nothing is ever written
through a generic path that would silently produce a file with the wrong shape.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast

from .core.errors import OrganelleInputError
from .core.result import OrganelleResult

__all__ = ["materializing_suites", "write"]

# Operation-id prefixes whose implementation package is named differently.
_SUITE_PACKAGES = {"qc": "quality_control"}

# Where a suite may keep its materializer, in search order.
_MATERIALIZER_MODULES = ("{package}.writer", "{package}.{package}")

_KNOWN_SUITES = (
    "annotation",
    "assembly",
    "barcode",
    "coevolution",
    "comparative",
    "hgt",
    "ir_boundary",
    "phenotype",
    "phylogeny",
    "qc",
    "rna_editing",
    "selection",
    "structure",
    "transfer",
    "visualization",
)


class _Materializer(Protocol):
    def __call__(self, result: OrganelleResult, output: Path) -> OrganelleResult: ...


def _package_for(suite: str) -> str:
    return _SUITE_PACKAGES.get(suite, suite)


def _resolve_materializer(suite: str) -> _Materializer | None:
    """Find a suite's materializer, or None when it does not publish one."""
    package = _package_for(suite)
    for template in _MATERIALIZER_MODULES:
        try:
            module = import_module(f".{template.format(package=package)}", __package__)
        except ImportError:
            continue
        candidate = getattr(module, "materialize_result", None)
        if callable(candidate):
            return cast("_Materializer", candidate)
    return None


def materializing_suites() -> tuple[str, ...]:
    """Return the suites that can currently materialize a result.

    Resolution is dynamic, so this is the auditable answer to "what does
    ``ov.write`` accept right now" without reading the dispatch code.
    """
    return tuple(suite for suite in _KNOWN_SUITES if _resolve_materializer(suite) is not None)


def write(obj: Any, output: str | Path) -> Any:
    """Materialize a released Result or a constructed HTML report.

    Arguments:
        obj: An :class:`OrganelleResult` from a materializing operation, or an
            ``OrganelleReport``.
        output: The user-facing destination. Its meaning is the materializer's
            to define — a directory bundle for some suites, a single file for
            others.

    Returns:
        For a result, the same result with its published artifacts attached.

    Raises:
        OrganelleInputError: The value is not writable, or its suite publishes
            no materializer.
    """
    path = Path(output)
    if isinstance(obj, OrganelleResult):
        suite = obj.operation_id.split(".", 1)[0]
        materialize = _resolve_materializer(suite)
        if materialize is not None:
            return materialize(obj, path)
        raise OrganelleInputError(
            code="output.unsupported_operation",
            message=(
                f"No materializer is published for suite {suite!r}. A suite becomes "
                "writable by providing materialize_result(result, output)."
            ),
            details={
                "operation_id": obj.operation_id,
                "suite": suite,
                "materializing_suites": list(materializing_suites()),
            },
        )

    from .reporting import OrganelleReport
    from .reporting import write as write_report

    if isinstance(obj, OrganelleReport):
        return write_report(obj, path)
    raise OrganelleInputError(
        code="output.unsupported_value",
        message="ov.write accepts a released Result or OrganelleReport",
        details={"type": type(obj).__name__},
    )
