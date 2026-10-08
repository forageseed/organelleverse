"""Immutable identities and independent evaluation gates for optimization v3."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, StrictFloat, StrictInt, StrictStr, model_validator

from organelleverse.operations.spec import StrictSpecModel

StrictNumber: TypeAlias = StrictInt | StrictFloat
_CAPABILITY_ID = r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$"
_HASH = r"^sha256:[0-9a-f]{64}$"
_SEMVER = r"^[0-9]+\.[0-9]+\.[0-9]+$"


class CapabilityIdentityV3(StrictSpecModel):
    """Content-addressed identity projected from one admitted capability."""

    capability_id: Annotated[StrictStr, Field(pattern=_CAPABILITY_ID)]
    bundle_version: Annotated[StrictStr, Field(pattern=_SEMVER)]
    contract_version: Annotated[StrictStr, Field(pattern=r"^[0-9]+\.[0-9]+$")]
    bundle_content_hash: Annotated[StrictStr, Field(pattern=_HASH)]
    execution_identity: Annotated[StrictStr, Field(pattern=_HASH)] | None = None
    implementation: Literal["native", "external", "composite"]
    surface: Literal["native", "plugin", "composite"]

    @model_validator(mode="after")
    def validate_surface(self) -> Self:
        if self.surface == "composite" and self.implementation != "composite":
            raise ValueError("composite surface requires composite implementation")
        if self.surface != "composite" and self.implementation == "composite":
            raise ValueError("composite implementation requires composite surface")
        return self


class BenchmarkSplitV3(StrictSpecModel):
    """One immutable benchmark partition."""

    split_id: Annotated[StrictStr, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]
    content_hash: Annotated[StrictStr, Field(pattern=_HASH)]
    case_count: Annotated[StrictInt, Field(ge=1)]
    artifact_ref: Annotated[StrictStr, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]


class BenchmarkContractV3(StrictSpecModel):
    """Versioned search and held-out validation identities."""

    benchmark_id: Annotated[StrictStr, Field(pattern=_CAPABILITY_ID)]
    benchmark_version: Annotated[StrictStr, Field(pattern=_SEMVER)]
    search: BenchmarkSplitV3
    validation: BenchmarkSplitV3

    @model_validator(mode="after")
    def validate_independent_splits(self) -> Self:
        if self.search.split_id == self.validation.split_id:
            raise ValueError("search and validation split IDs must differ")
        if self.search.content_hash == self.validation.content_hash:
            raise ValueError("search and validation content hashes must differ")
        return self


class EvaluatorContractV3(StrictSpecModel):
    """An admitted evaluator identity and its fixed output protocol."""

    identity: CapabilityIdentityV3
    protocol_version: Literal["organelleverse.optimization.evaluator.v1"] = (
        "organelleverse.optimization.evaluator.v1"
    )
    metrics_pointer: Literal["/metrics/optimization_evaluation"] = (
        "/metrics/optimization_evaluation"
    )


class AdoptionPolicyV3(StrictSpecModel):
    """Hard evidence policy that a later engine must apply without scalarization."""

    decision_rule: Literal["all_objectives", "pareto_non_dominated"]
    minimum_success_rate: StrictNumber
    maximum_failure_rate: StrictNumber
    maximum_wall_time_ratio: StrictNumber
    maximum_peak_memory_ratio: StrictNumber
    approval: Literal["automatic", "human_required"]

    @model_validator(mode="after")
    def validate_rates_and_ratios(self) -> Self:
        values = (
            self.minimum_success_rate,
            self.maximum_failure_rate,
            self.maximum_wall_time_ratio,
            self.maximum_peak_memory_ratio,
        )
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("adoption policy numbers must be finite")
        if not 0 < self.minimum_success_rate <= 1:
            raise ValueError("minimum_success_rate must be in (0, 1]")
        if not 0 <= self.maximum_failure_rate < 1:
            raise ValueError("maximum_failure_rate must be in [0, 1)")
        if self.minimum_success_rate + self.maximum_failure_rate > 1:
            raise ValueError("success and failure rates must sum to at most one")
        if self.maximum_wall_time_ratio <= 0 or self.maximum_peak_memory_ratio <= 0:
            raise ValueError("resource ratios must be positive")
        return self


__all__ = [
    "AdoptionPolicyV3",
    "BenchmarkContractV3",
    "BenchmarkSplitV3",
    "CapabilityIdentityV3",
    "EvaluatorContractV3",
]
