"""Deterministic Slurm state → provider lifecycle mapping (spec §13).

``squeue``/``sacct`` output is parsed into the stable provider lifecycle. The
mapping is total and deterministic: every observed scheduler state maps to
exactly one lifecycle verdict, and absent jobs within the bounded accounting
window stay nonterminal (never resubmit). Scheduler/provider failures never
fabricate a scientific ``OrganelleResult(status="failed")``.

Parsing operates on bounded ``squeue``/``sacct`` stdout captured by an
injectable runner, so it is fully testable with fixture output and no real
Slurm cluster.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from organelleverse.core.errors import OrganelleContractError
from organelleverse.operations.spec import StrictSpecModel

__all__ = [
    "SlurmLifecycleVerdict",
    "SlurmObservation",
    "map_slurm_state",
    "parse_sacct_state",
    "parse_squeue_state",
]

LifecycleStatus = Literal["queued", "running", "completed", "cancelled", "failed", "accounting_pending"]


class SlurmLifecycleVerdict(StrictSpecModel):
    """The provider-lifecycle verdict for one Slurm observation."""

    status: LifecycleStatus
    normalized_state: str = Field(min_length=1)
    is_terminal: bool
    is_scientific_candidate: bool = False
    exit_code: int | None = None


# Spec §13 mapping tables. Each raw Slurm state → (lifecycle, terminal, scientific?).
# Nonterminal states stay nonterminal until a terminal accounting record appears.
_NONTERMINAL_QUEUED = frozenset(
    {"PENDING", "CONFIGURING", "RESV_DEL_HOLD", "REQUEUE_FED", "REQUEUE_HOLD", "REQUEUED"}
)
_NONTERMINAL_RUNNING = frozenset(
    {"RUNNING", "COMPLETING", "STAGE_OUT", "STOPPED", "SUSPENDED"}
)
_TERMINAL_CANCELLED = frozenset({"CANCELLED"})
# Cancelled has variants like "CANCELLED by 1000" — handled by prefix match below.
_TERMINAL_FAILED = frozenset(
    {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}
)


def _normalize(raw_state: str) -> str:
    # sacct states can carry " by <user>" suffixes or be composite; take the
    # leading token (the canonical state name).
    return raw_state.strip().split()[0].upper() if raw_state.strip() else ""


def map_slurm_state(raw_state: str, *, exit_code: int | None = None) -> SlurmLifecycleVerdict:
    """Map one normalized Slurm state to the provider lifecycle (spec §13).

    ``exit_code`` is the job's exit code from accounting (None while queued/
    running). A ``COMPLETED`` state with a nonzero exit is a provider
    infrastructure failure, not a scientific candidate.
    """
    state = _normalize(raw_state)
    if not state:
        raise OrganelleContractError(
            code="compute.slurm_empty_state",
            message="Slurm reported an empty state; the scheduler output is malformed",
        )

    if state in _NONTERMINAL_QUEUED:
        return SlurmLifecycleVerdict(
            status="queued", normalized_state=state, is_terminal=False
        )
    if state in _NONTERMINAL_RUNNING:
        return SlurmLifecycleVerdict(
            status="running", normalized_state=state, is_terminal=False
        )
    if state == "COMPLETED":
        # COMPLETED + zero exit is a scientific candidate (still blocked on the
        # six host checks). A nonzero exit is a provider infrastructure failure.
        if exit_code is not None and exit_code != 0:
            return SlurmLifecycleVerdict(
                status="failed", normalized_state=state, is_terminal=True, exit_code=exit_code
            )
        return SlurmLifecycleVerdict(
            status="completed",
            normalized_state=state,
            is_terminal=True,
            is_scientific_candidate=True,
            exit_code=exit_code,
        )
    if state.startswith("CANCELLED") or state in _TERMINAL_CANCELLED:
        return SlurmLifecycleVerdict(
            status="cancelled", normalized_state="CANCELLED", is_terminal=True
        )
    if state in _TERMINAL_FAILED:
        return SlurmLifecycleVerdict(
            status="failed", normalized_state=state, is_terminal=True, exit_code=exit_code
        )

    # Unknown state: never fabricate a result. Treat as nonterminal working so
    # the caller keeps polling; an unknown terminal state surfaces once sacct
    # reports a known one.
    return SlurmLifecycleVerdict(
        status="running", normalized_state=state, is_terminal=False
    )


class SlurmObservation(StrictSpecModel):
    """One bounded observation of a job's scheduler state."""

    raw_state: str
    exit_code: int | None = None
    seen_in: Literal["squeue", "sacct"] = "squeue"


def parse_squeue_state(stdout: bytes) -> str | None:
    """Parse ``squeue --noheader --jobs <id> --format %T`` output.

    Returns the single state token, or ``None`` if the job is absent from the
    queue (caller then consults accounting).
    """
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    # exactly one line, one token expected; take the first token defensively
    return text.splitlines()[0].strip().split()[0]


def parse_sacct_state(stdout: bytes) -> SlurmObservation:
    """Parse ``sacct --parsable2 --format JobIDRaw,State,ExitCode`` output.

    Selects the exact allocation row (JobIDRaw with no step suffix) and returns
    its state + exit code. Raises if no allocation row is present.
    """
    text = stdout.decode("utf-8", errors="replace")
    allocation = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id_raw, state, exit_field = parts[0], parts[1], parts[2]
        # allocation row has a bare numeric id; step/array rows carry suffixes
        if "." in job_id_raw or "_" in job_id_raw:
            continue
        try:
            exit_code = int(str(exit_field).split(":")[0])
        except ValueError:
            exit_code = None
        allocation = SlurmObservation(
            raw_state=state, exit_code=exit_code, seen_in="sacct"
        )
        break
    if allocation is None:
        raise OrganelleContractError(
            code="compute.slurm_accounting_missing",
            message="sacct returned no allocation row for the job within the accounting window",
        )
    return allocation
