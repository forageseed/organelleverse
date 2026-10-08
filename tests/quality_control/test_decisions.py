from __future__ import annotations

import pytest

from organelleverse.quality_control.contracts import QcCheck
from organelleverse.quality_control.service import aggregate_decision, result_status_for


def _checks(statuses: tuple[str, ...]) -> tuple[QcCheck, ...]:
    return tuple(
        QcCheck(
            check_id=f"test.check_{index}",
            category="test",
            status=status,  # type: ignore[arg-type]
            message=status,
        )
        for index, status in enumerate(statuses)
    )


@pytest.mark.parametrize(
    ("statuses", "missing_required", "decision", "result_status"),
    [
        (("fail",), False, "not_ready", "warning"),
        (("warn",), False, "needs_review", "warning"),
        (("pass",), True, "insufficient_evidence", "warning"),
        (("pass",), False, "ready", "ok"),
    ],
)
def test_decision_precedence(
    statuses: tuple[str, ...],
    missing_required: bool,
    decision: str,
    result_status: str,
) -> None:
    actual = aggregate_decision(
        _checks(statuses),
        required_evidence_missing=missing_required,
    )
    assert actual == decision
    assert result_status_for(actual) == result_status
