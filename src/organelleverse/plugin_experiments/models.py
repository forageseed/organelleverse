"""Closed durable models for governed plugin optimization experiments."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, JsonValue, field_serializer, field_validator, model_validator

from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.operations.spec import StrictSpecModel

__all__ = [
    "ExperimentRecord",
    "ExperimentRequest",
    "ExperimentTrial",
]


class ExperimentRequest(StrictSpecModel):
    """One governed submission and its exact materialized candidate sequence."""

    capability_id: str = Field(min_length=1)
    inputs: dict[str, str]
    fixed_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    strategy: Literal["explicit", "grid", "random", "agent"] = "explicit"
    seed: int = 0
    rationale: str | None = Field(default=None, min_length=1, max_length=2000)
    candidates: tuple[dict[str, JsonValue], ...] = ()
    contract_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def _validate_strategy_payload(self) -> Self:
        if self.rationale is not None and not self.rationale.strip():
            raise ValueError("experiment rationale must not be blank")
        if self.strategy in {"explicit", "agent"} and not self.candidates:
            raise ValueError(f"{self.strategy} strategy requires candidates")
        if self.strategy == "agent" and self.rationale is None:
            raise ValueError("agent strategy requires a rationale")
        if self.strategy != "agent" and self.rationale is not None:
            raise ValueError("only agent strategy accepts a rationale")
        encoded = json.dumps(
            self.candidates,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > 64 * 1024:
            raise ValueError("candidate JSON must not exceed 64 KiB")
        return self


class ExperimentTrial(StrictSpecModel):
    """One candidate's durable outcome."""

    index: int = Field(ge=0)
    parameters: dict[str, JsonValue]
    status: Literal["queued", "running", "succeeded", "failed"]
    result: OrganelleResult | None = None
    score: float | None = None
    error: ErrorDetail | None = None

    @field_validator("score")
    @classmethod
    def _score_must_be_finite(cls, score: float | None) -> float | None:
        if score is not None and not math.isfinite(score):
            raise ValueError("trial score must be a finite number")
        return score

    @field_serializer("result")
    def _serialize_result(self, result: OrganelleResult | None) -> object:
        # OrganelleResult.object_id is a computed identity; serializing it
        # would fail nested validation (its retain/verify context only exists
        # for top-level OrganelleResult.model_validate). It is recomputed on
        # load, so excluding it loses nothing.
        if result is None:
            return None
        return result.model_dump(mode="json", exclude={"object_id"})

    @field_serializer("error")
    def _serialize_error(self, error: ErrorDetail | None) -> object:
        # Same computed-identity rule as ``result`` above.
        if error is None:
            return None
        return error.model_dump(mode="json", exclude={"object_id"})


class ExperimentRecord(StrictSpecModel):
    """The durable, restart-safe record of one experiment."""

    experiment_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    status: Literal["queued", "running", "succeeded", "failed"]
    submitted_at: datetime
    completed_at: datetime | None = None
    request: ExperimentRequest
    trials: tuple[ExperimentTrial, ...]
    best_trial_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _terminal_state_is_complete_and_well_scored(self) -> Self:
        terminal = {"succeeded", "failed"}
        if self.status in terminal and any(trial.status not in terminal for trial in self.trials):
            raise ValueError("terminal experiment records require terminal trials")
        if self.best_trial_index is None:
            if self.status == "succeeded":
                raise ValueError("a succeeded experiment record requires a best trial")
            return self
        if self.status != "succeeded" or self.best_trial_index >= len(self.trials):
            raise ValueError("best trial is invalid for this experiment record")
        winner = self.trials[self.best_trial_index]
        if winner.status != "succeeded" or winner.score is None:
            raise ValueError("best trial must be a successfully scored trial")
        return self
