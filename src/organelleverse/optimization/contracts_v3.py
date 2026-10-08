"""Strict implementation-independent optimization contract v3."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Hashable, Iterable
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, JsonValue, StrictFloat, StrictInt, StrictStr, model_validator

from organelleverse.operations.spec import StrictSpecModel

from .evaluation import (
    AdoptionPolicyV3,
    BenchmarkContractV3,
    CapabilityIdentityV3,
    EvaluatorContractV3,
)
from .models import ParameterDomain, _canonical_json  # pyright: ignore[reportPrivateUsage]

StrictNumber: TypeAlias = StrictInt | StrictFloat
_POINTER = r"^/metrics/optimization_evaluation(?:/(?:[^~/]|~[01])+)+$"


class BaselineCandidateV3(StrictSpecModel):
    """Explicit parameters retained whenever evidence does not justify adoption."""

    parameters: dict[StrictStr, JsonValue] = Field(min_length=1)


class PriorCandidateV3(StrictSpecModel):
    """An evidenced initial candidate with no score or adoption semantics."""

    prior_id: Annotated[StrictStr, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]
    source: Literal["declared", "historical", "minegraph"]
    parameters: dict[StrictStr, JsonValue] = Field(min_length=1)
    evidence_refs: tuple[
        Annotated[StrictStr, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")], ...
    ] = Field(min_length=1)
    role: Literal["initial_candidate"] = "initial_candidate"


class ObjectiveV3(StrictSpecModel):
    """One independently compared evaluator metric."""

    name: Annotated[StrictStr, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    metric_pointer: Annotated[StrictStr, Field(pattern=_POINTER)]
    direction: Literal["maximize", "minimize"]
    minimum_improvement: StrictNumber
    improvement_mode: Literal["absolute", "relative"]
    aggregation: Literal["mean", "median"]

    @model_validator(mode="after")
    def validate_improvement(self) -> Self:
        value = self.minimum_improvement
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("minimum_improvement must be finite and non-negative")
        return self


class MetricConstraintV3(StrictSpecModel):
    """One non-compensable evaluator metric threshold."""

    name: Annotated[StrictStr, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    metric_pointer: Annotated[StrictStr, Field(pattern=_POINTER)]
    operator: Literal["le", "ge"]
    threshold: StrictNumber

    @model_validator(mode="after")
    def validate_threshold(self) -> Self:
        if isinstance(self.threshold, bool) or not math.isfinite(self.threshold):
            raise ValueError("constraint threshold must be finite")
        return self


class OptimizationBudgetV3(StrictSpecModel):
    """Hard search and resource ceilings."""

    max_trials: Annotated[StrictInt, Field(ge=1, le=256)]
    parallelism: Annotated[StrictInt, Field(ge=1, le=32)]
    max_wall_time_seconds: Annotated[StrictInt, Field(ge=1)]
    max_cpu_time_seconds: Annotated[StrictInt, Field(ge=1)]
    max_peak_memory_bytes: Annotated[StrictInt, Field(ge=1)]

    @model_validator(mode="after")
    def validate_parallelism(self) -> Self:
        if self.parallelism > self.max_trials:
            raise ValueError("parallelism must not exceed max_trials")
        return self


class RepeatPolicyV3(StrictSpecModel):
    """Reproducible search and held-out validation repetition policy."""

    search_repeats: Annotated[StrictInt, Field(ge=1)]
    validation_repeats: Annotated[StrictInt, Field(ge=2)]
    seed: StrictInt


class StrategyReferenceV3(StrictSpecModel):
    """One search kind bound to an admitted, content-addressed capability."""

    kind: Literal["explicit", "grid", "random", "agent", "space_filling", "successive_halving"]
    identity: CapabilityIdentityV3


class OptimizationDeclarationV3(StrictSpecModel):
    """Scientific optimization declaration before evaluator admission is attached."""

    parameters: tuple[ParameterDomain, ...] = Field(min_length=1)
    baseline: BaselineCandidateV3
    priors: tuple[PriorCandidateV3, ...] = ()
    objectives: tuple[ObjectiveV3, ...] = Field(min_length=1)
    constraints: tuple[MetricConstraintV3, ...] = ()
    strategies: tuple[StrategyReferenceV3, ...] = Field(min_length=1)
    budget: OptimizationBudgetV3
    repeats: RepeatPolicyV3
    adoption: AdoptionPolicyV3
    benchmark: BenchmarkContractV3 | None = None

    @model_validator(mode="after")
    def validate_declaration(self) -> Self:
        _require_unique((item.name for item in self.parameters), "parameter names")
        _require_unique((item.prior_id for item in self.priors), "prior IDs")
        _require_unique((item.name for item in self.objectives), "objective names")
        _require_unique((item.name for item in self.constraints), "constraint names")
        _require_unique((item.metric_pointer for item in self.objectives), "objective pointers")
        _require_unique((item.metric_pointer for item in self.constraints), "constraint pointers")
        _require_unique(
            (
                *(item.metric_pointer for item in self.objectives),
                *(item.metric_pointer for item in self.constraints),
            ),
            "metric pointers",
        )
        _require_unique((item.kind for item in self.strategies), "strategy kinds")
        _require_unique(
            (item.identity.capability_id for item in self.strategies),
            "strategy capability IDs",
        )
        _validate_candidate(self.parameters, self.baseline.parameters, "baseline")
        for prior in self.priors:
            _validate_candidate(self.parameters, prior.parameters, f"prior {prior.prior_id!r}")
        return self


class OptimizationContractV3(OptimizationDeclarationV3):
    """Enabled, independently evaluable optimization contract."""

    schema_version: Literal["organelleverse.optimization.contract.v3"] = (
        "organelleverse.optimization.contract.v3"
    )
    target: CapabilityIdentityV3
    benchmark: BenchmarkContractV3  # pyright: ignore[reportIncompatibleVariableOverride,reportGeneralTypeIssues]
    evaluator: EvaluatorContractV3

    @model_validator(mode="after")
    def validate_independent_evaluator(self) -> Self:
        if self.target.capability_id == self.evaluator.identity.capability_id:
            raise ValueError("target and evaluator capability IDs must differ")
        return self

    @property
    def digest(self) -> str:
        canonical = json.dumps(
            _canonical_json(self.model_dump(mode="json")),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


class OptimizationProfileV3(StrictSpecModel):
    """Closed classification of one admitted capability."""

    status: Literal["enabled", "eligible", "not_applicable"]
    target: CapabilityIdentityV3
    contract: OptimizationContractV3 | None = None
    reason_code: Annotated[StrictStr, Field(min_length=1, pattern=r"^\S+$")] | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> Self:
        if self.status == "enabled":
            if self.contract is None or self.reason_code is not None:
                raise ValueError("enabled profiles require a contract and no reason")
            if self.contract.target != self.target:
                raise ValueError("profile target must match contract target")
        elif self.contract is not None or self.reason_code is None:
            raise ValueError("non-enabled profiles require a reason and no contract")
        return self


def _require_unique(values: Iterable[Hashable], label: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{label} must be unique")


def _validate_candidate(
    domains: tuple[ParameterDomain, ...], parameters: dict[str, JsonValue], label: str
) -> None:
    expected = {domain.name for domain in domains}
    if set(parameters) != expected:
        raise ValueError(f"{label} parameter names must exactly match declared domains")
    for domain in domains:
        value = parameters[domain.name]
        if not _value_in_domain(domain, value):
            raise ValueError(f"{label} parameter {domain.name!r} is outside its domain")


def _value_in_domain(domain: ParameterDomain, value: JsonValue) -> bool:
    if domain.kind == "boolean":
        return isinstance(value, bool)
    if domain.kind == "integer":
        if type(value) is not int:
            return False
        assert type(domain.minimum) is int
        assert type(domain.maximum) is int
        assert type(domain.step) is int
        return (
            domain.minimum <= value <= domain.maximum
            and (value - domain.minimum) % domain.step == 0
        )
    if domain.kind == "number":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return False
        assert domain.minimum is not None and domain.maximum is not None
        if not domain.minimum <= value <= domain.maximum:
            return False
        return not domain.values or _canonical_member(value, domain.values)
    return _canonical_member(value, domain.values)


def _canonical_member(value: JsonValue, choices: tuple[object, ...]) -> bool:
    encoded = json.dumps(_canonical_json(value), sort_keys=True, separators=(",", ":"))
    return any(
        encoded == json.dumps(_canonical_json(choice), sort_keys=True, separators=(",", ":"))
        for choice in choices
    )


__all__ = [
    "BaselineCandidateV3",
    "MetricConstraintV3",
    "ObjectiveV3",
    "OptimizationBudgetV3",
    "OptimizationContractV3",
    "OptimizationDeclarationV3",
    "OptimizationProfileV3",
    "PriorCandidateV3",
    "RepeatPolicyV3",
    "StrategyReferenceV3",
]
