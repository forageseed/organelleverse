"""Closed immutable inputs and outputs for adaptive search strategies."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, JsonValue, StrictInt, StrictStr, model_validator

from organelleverse.operations.spec import StrictSpecModel

from .contracts_v3 import (
    OptimizationContractV3,
    _value_in_domain,  # pyright: ignore[reportPrivateUsage]
)
from .evaluation import CapabilityIdentityV3
from .models import ParameterDomain


class StrategyBinding(StrictSpecModel):
    kind: Literal["grid", "random", "space_filling", "successive_halving"]
    identity: CapabilityIdentityV3


class SuccessiveHalvingConfig(StrictSpecModel):
    resource_parameter: Annotated[StrictStr, Field(min_length=1, pattern=r"^\S+$")]
    rungs: tuple[StrictInt, ...] = Field(min_length=2)
    eta: Annotated[StrictInt, Field(ge=2)]
    base_strategy: StrategyBinding

    @model_validator(mode="after")
    def validate_rungs(self) -> Self:
        if any(left >= right for left, right in zip(self.rungs, self.rungs[1:], strict=False)):
            raise ValueError("successive-halving resource rungs must strictly increase")
        if self.base_strategy.kind == "successive_halving":
            raise ValueError("successive-halving base strategy cannot be recursive")
        return self


class StrategySpec(StrictSpecModel):
    parameters: tuple[ParameterDomain, ...] = Field(min_length=1)
    seed: StrictInt
    total_design_size: Annotated[StrictInt, Field(ge=1, le=256)]
    successive_halving: SuccessiveHalvingConfig | None = None
    objective_order: tuple[tuple[StrictStr, Literal["maximize", "minimize"]], ...] = ()


class EvaluatorBinding(StrictSpecModel):
    identity: CapabilityIdentityV3
    evaluator_digest: Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class OpaqueArtifactBinding(StrictSpecModel):
    """Content-bound resolver handle; never a caller-controlled filesystem path."""

    schema_version: Literal["organelleverse.opaque-artifact-binding.v1"] = (
        "organelleverse.opaque-artifact-binding.v1"
    )
    artifact_ref: Annotated[
        StrictStr,
        Field(pattern=r"^artifact:[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$"),
    ]
    content_digest: Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class AdaptiveStudyRequest(StrictSpecModel):
    study_id: Annotated[StrictStr, Field(pattern=r"^study-[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    contract: OptimizationContractV3
    strategy_binding: StrategyBinding
    evaluator_binding: EvaluatorBinding
    objective_order: tuple[Annotated[StrictStr, Field(min_length=1)], ...] = Field(min_length=1)
    finalist_limit: Annotated[StrictInt, Field(ge=1, le=256)]
    input_digest: Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    fixed_parameters_digest: Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    input_artifact: OpaqueArtifactBinding
    fixed_parameters_artifact: OpaqueArtifactBinding
    environment_digest: Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    successive_halving: SuccessiveHalvingConfig | None = None

    @model_validator(mode="after")
    def validate_bindings_and_order(self) -> Self:
        if (
            self.input_artifact.content_digest != self.input_digest
            or self.fixed_parameters_artifact.content_digest != self.fixed_parameters_digest
        ):
            raise ValueError("artifact content bindings must match their declared digests")
        references = {reference.kind: reference.identity for reference in self.contract.strategies}
        if references.get(self.strategy_binding.kind) != self.strategy_binding.identity:
            raise ValueError("strategy binding must exactly match the enabled contract")
        if self.evaluator_binding.identity != self.contract.evaluator.identity:
            raise ValueError("evaluator binding must exactly match the enabled contract")
        expected = tuple(objective.name for objective in self.contract.objectives)
        if len(self.objective_order) != len(set(self.objective_order)) or set(
            self.objective_order
        ) != set(expected):
            raise ValueError("objective_order must contain every objective exactly once")
        if (self.strategy_binding.kind == "successive_halving") != (
            self.successive_halving is not None
        ):
            raise ValueError("successive-halving binding and configuration must appear together")
        if self.successive_halving is not None:
            base = self.successive_halving.base_strategy
            if references.get(base.kind) != base.identity:
                raise ValueError(
                    "successive-halving base strategy must exactly match an enabled contract strategy"
                )
            domains = {domain.name: domain for domain in self.contract.parameters}
            resource_domain = domains.get(self.successive_halving.resource_parameter)
            if resource_domain is None or any(
                not _value_in_domain(resource_domain, rung)
                for rung in self.successive_halving.rungs
            ):
                raise ValueError("successive-halving rungs must belong to the resource domain")
        return self


class CandidateProposal(StrictSpecModel):
    parameters: dict[StrictStr, JsonValue]
    candidate_digest: Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    base_candidate_digest: Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None = (
        None
    )
    resource_rung: StrictInt | None = None


class SearchObservation(StrictSpecModel):
    candidate_digest: Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    base_candidate_digest: Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None = (
        None
    )
    resource_rung: StrictInt | None = None
    metrics: dict[StrictStr, float] = Field(default_factory=dict)
    feasible: bool
    failed: bool = False


class StrategyState(StrictSpecModel):
    kind: Literal["grid", "random", "space_filling"]
    binding: StrategyBinding
    spec: StrategySpec
    step: Annotated[StrictInt, Field(ge=0)] = 0
    emitted: tuple[Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")], ...] = ()

    @model_validator(mode="after")
    def validate_core_state(self) -> Self:
        if self.binding.kind != self.kind:
            raise ValueError("core strategy state kind and binding must match")
        if self.spec.successive_halving is not None or self.spec.objective_order:
            raise ValueError("core strategy state cannot carry successive-halving configuration")
        if len(self.emitted) != len(set(self.emitted)):
            raise ValueError("core strategy emitted identities must be unique")
        return self


class SuccessiveHalvingState(StrictSpecModel):
    kind: Literal["successive_halving"] = "successive_halving"
    binding: StrategyBinding
    spec: StrategySpec
    config: SuccessiveHalvingConfig
    phase: Literal["ready", "awaiting", "complete"] = "ready"
    rung_index: Annotated[StrictInt, Field(ge=0)] = 0
    cohort: tuple[CandidateProposal, ...]
    current_assignments: tuple[CandidateProposal, ...] = ()
    completed_assignment_digests: tuple[
        Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")], ...
    ] = ()
    promoted: tuple[Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")], ...] = ()

    @model_validator(mode="after")
    def validate_halving_state(self) -> Self:
        if self.binding.kind != "successive_halving":
            raise ValueError("successive-halving state requires its matching binding")
        if self.spec.successive_halving != self.config or not self.spec.objective_order:
            raise ValueError("successive-halving state requires its exact config and objectives")
        if self.rung_index >= len(self.config.rungs):
            raise ValueError("successive-halving rung index is outside its configuration")
        if self.phase == "awaiting":
            if not self.cohort or len(self.current_assignments) != len(self.cohort):
                raise ValueError("awaiting successive-halving state requires a complete rung")
        elif self.current_assignments:
            raise ValueError("only awaiting successive-halving state carries assignments")
        if self.phase == "ready" and not self.cohort:
            raise ValueError("ready successive-halving state requires a non-empty cohort")
        if any(
            item.base_candidate_digest is not None or item.resource_rung is not None
            for item in self.cohort
        ):
            raise ValueError("successive-halving cohort must contain base candidates")
        if len(self.completed_assignment_digests) != len(set(self.completed_assignment_digests)):
            raise ValueError("completed successive-halving assignments must be unique")
        if len(self.promoted) != len(set(self.promoted)):
            raise ValueError("promoted successive-halving identities must be unique")
        return self


class ProposalBatch(StrictSpecModel):
    proposals: tuple[CandidateProposal, ...]
    state: StrategyState | SuccessiveHalvingState
    exhausted: bool = False


class SearchStrategy(Protocol):
    def initialize(
        self, spec: StrategySpec, binding: StrategyBinding
    ) -> StrategyState | SuccessiveHalvingState: ...

    def propose(
        self,
        state: StrategyState | SuccessiveHalvingState,
        observations: tuple[SearchObservation, ...],
        remaining_budget: int,
    ) -> ProposalBatch: ...

    def observe(
        self,
        state: StrategyState | SuccessiveHalvingState,
        observations: tuple[SearchObservation, ...],
    ) -> StrategyState | SuccessiveHalvingState: ...


__all__ = [
    "AdaptiveStudyRequest",
    "CandidateProposal",
    "EvaluatorBinding",
    "OpaqueArtifactBinding",
    "ProposalBatch",
    "SearchObservation",
    "SearchStrategy",
    "StrategyBinding",
    "StrategySpec",
    "StrategyState",
    "SuccessiveHalvingConfig",
    "SuccessiveHalvingState",
]
