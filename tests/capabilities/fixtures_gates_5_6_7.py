"""Importable implementations exercising admission gates 5/6/7.

Each callable stands in for an installed organelleverse implementation, the
way ``fixtures_core_capability.py`` does for the core admission pipeline:

* ``run_echo`` / ``run_echo_original`` / ``run_echo_drifted`` - gate 7
  (rewrite equivalence): the current implementation returns ``value``, the
  "original" it rewrites returns the same thing, and a drifted original
  returns something else.
* ``run_flaky`` - gate 6 (determinism): its output changes between calls
  exactly when the test queues different offsets, so a test controls which
  verification reruns agree and which disagree.
* ``run_validated`` - gate 5 (adversarial): raises a stable, structured
  error code for one parameter value the way a fail-closed capability must.
"""

from __future__ import annotations

from organelleverse.core.errors import OrganelleParameterError
from organelleverse.core.frozen import FrozenMap
from organelleverse.core.result import OrganelleResult

#: Offsets ``run_flaky`` consumes one per call (and clears as consumed), so a
#: test scripts exactly which invocations diverge instead of relying on
#: randomness a failing assertion could not reproduce.
_FLAKY_OFFSETS: list[int] = []


def queue_flaky_offsets(*offsets: int) -> None:
    """Script the next ``run_flaky`` outputs: one offset per future call."""
    _FLAKY_OFFSETS.clear()
    _FLAKY_OFFSETS.extend(offsets)


def _result(operation_id: str, value: int) -> OrganelleResult:
    return OrganelleResult(
        operation_id=operation_id,
        scope="none",
        status="ok",
        summary_text="gate fixture capability executed in-process",
        metrics=FrozenMap({"value": value}),
    )


def run_echo(*, value: int = 1) -> OrganelleResult:
    """The current implementation: a stable, pure echo."""
    return _result("demo.gates_echo", value)


def run_echo_original(*, value: int = 1) -> OrganelleResult:
    """The original implementation ``run_echo`` claims to rewrite, unchanged."""
    return _result("demo.gates_echo", value)


def run_echo_drifted(*, value: int = 1) -> OrganelleResult:
    """An 'original' the rewrite silently changed behavior relative to."""
    return _result("demo.gates_echo", value + 1)


def run_flaky(*, value: int = 1) -> OrganelleResult:
    """Non-deterministic by test script: consumes one queued offset per call."""
    offset = _FLAKY_OFFSETS.pop(0) if _FLAKY_OFFSETS else 0
    return _result("demo.gates_flaky", value + offset)


def run_validated(*, mode: str = "ok") -> OrganelleResult:
    """Fail-closed validator: one parameter value must raise a stable code."""
    if mode == "bad":
        raise OrganelleParameterError(
            code="capability.parameter_invalid",
            message="mode 'bad' is rejected by the declared parameter contract",
            details={"parameter": "mode", "mode": mode},
        )
    return _result("demo.gates_validated", len(mode))
