"""Closed experiment model contracts (Plugin-04, Task 2)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from organelleverse.plugin_experiments.models import (
    ExperimentRecord,
    ExperimentRequest,
    ExperimentTrial,
)


def _request() -> ExperimentRequest:
    return ExperimentRequest(
        capability_id="demo.experiment",
        inputs={"images": "/data/images"},
        fixed_parameters={"mode": "fast"},
        candidates=({"threshold": 0.25}, {"threshold": 0.5}),
    )


def test_request_requires_at_least_one_candidate() -> None:
    with pytest.raises(ValidationError):
        ExperimentRequest(capability_id="demo.experiment", inputs={}, candidates=())


def test_models_are_closed_to_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ExperimentRequest(
            capability_id="demo.experiment",
            inputs={},
            candidates=({"threshold": 0.5},),
            extra="nope",  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        ExperimentTrial(index=0, parameters={}, status="queued", hacked=True)  # type: ignore[call-arg]


def test_trial_index_and_score_constraints() -> None:
    with pytest.raises(ValidationError):
        ExperimentTrial(index=-1, parameters={}, status="queued")
    with pytest.raises(ValidationError):
        ExperimentTrial(index=0, parameters={}, status="succeeded", score=math.nan)
    trial = ExperimentTrial(index=0, parameters={}, status="succeeded", score=0.75)
    assert trial.score == 0.75


def test_record_round_trips_through_canonical_json() -> None:
    record = ExperimentRecord(
        experiment_id="exp-1",
        capability_id="demo.experiment",
        status="queued",
        submitted_at=datetime.now(UTC),
        request=_request(),
        trials=(
            ExperimentTrial(index=0, parameters={"threshold": 0.25}, status="queued"),
            ExperimentTrial(index=1, parameters={"threshold": 0.5}, status="queued"),
        ),
    )
    restored = ExperimentRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert restored.best_trial_index is None


def test_best_trial_index_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        ExperimentRecord(
            experiment_id="exp-1",
            capability_id="demo.experiment",
            status="succeeded",
            submitted_at=datetime.now(UTC),
            request=_request(),
            trials=(),
            best_trial_index=-1,
        )


def test_terminal_record_rejects_a_nonterminal_trial_or_invalid_winner() -> None:
    with pytest.raises(ValidationError):
        ExperimentRecord(
            experiment_id="exp-1",
            capability_id="demo.experiment",
            status="succeeded",
            submitted_at=datetime.now(UTC),
            request=_request(),
            trials=(ExperimentTrial(index=0, parameters={}, status="running"),),
            best_trial_index=0,
        )
    with pytest.raises(ValidationError):
        ExperimentRecord(
            experiment_id="exp-2",
            capability_id="demo.experiment",
            status="succeeded",
            submitted_at=datetime.now(UTC),
            request=_request(),
            trials=(ExperimentTrial(index=0, parameters={}, status="failed"),),
            best_trial_index=0,
        )
