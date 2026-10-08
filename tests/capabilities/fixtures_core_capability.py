"""A trivial, genuinely importable callable used to exercise a *core*-origin
capability bundle end to end.

Unlike ``fixtures_for_binding.py`` (a plain dataclass used for parameter
codec tests), this module stands in for "the installed organelleverse
distribution's own implementation": a core capability's ``callable_locator``
points at something like this, not at bundle-local ``code/``.
"""

from __future__ import annotations

from organelleverse.core.frozen import FrozenMap
from organelleverse.core.result import OrganelleResult


def run(*, value: int = 1) -> OrganelleResult:
    return OrganelleResult(
        operation_id="capability03.core_admission_demo",
        scope="none",
        status="ok",
        summary_text="core capability executed in-process",
        metrics=FrozenMap({"value": value}),
    )
