"""Closed public identities and evaluator port for adaptive studies."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from organelleverse.operations.spec import StrictSpecModel
from organelleverse.plugin_experiments.adaptive_models import AttemptIdentity, RepeatEvidence

from .evaluation import BenchmarkSplitV3, CapabilityIdentityV3
from .models import _canonical_json  # pyright: ignore[reportPrivateUsage]
from .strategies import candidate_digest
from .strategy_models import (
    AdaptiveStudyRequest,
    EvaluatorBinding,
    OpaqueArtifactBinding,
    StrategyBinding,
)

Digest = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
SafeText = Annotated[StrictStr, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]


class EvaluationRequest(StrictSpecModel):
    schema_version: Literal["organelleverse.optimization.evaluation-request.v1"] = (
        "organelleverse.optimization.evaluation-request.v1"
    )
    study_id: Annotated[StrictStr, Field(pattern=r"^study-[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    target: CapabilityIdentityV3
    contract_digest: Digest
    evaluator_binding: EvaluatorBinding
    identity: AttemptIdentity
    benchmark_split: BenchmarkSplitV3
    parameters: dict[StrictStr, JsonValue]
    input_digest: Digest
    fixed_parameters_digest: Digest
    input_artifact: OpaqueArtifactBinding
    fixed_parameters_artifact: OpaqueArtifactBinding
    environment_digest: Digest
    seed: StrictInt
    run_id: Annotated[StrictStr, Field(pattern=r"^run-[0-9a-f]{64}$")]
    required_metric_pointers: tuple[SafeText, ...] = Field(min_length=1)
    remaining_wall_time_seconds: Annotated[StrictFloat, Field(gt=0)]
    remaining_cpu_time_seconds: Annotated[StrictFloat, Field(gt=0)]
    max_peak_memory_bytes: Annotated[StrictInt, Field(ge=0)]

    @field_validator("remaining_wall_time_seconds", "remaining_cpu_time_seconds")
    @classmethod
    def _finite_grant(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("evaluation resource grants must be finite")
        return value

    @model_validator(mode="after")
    def validate_dispatch(self) -> Self:
        identity = self.identity
        if (
            identity.evaluator_digest != self.evaluator_binding.evaluator_digest
            or identity.evaluator_identity != self.evaluator_binding.identity
            or identity.split_id != self.benchmark_split.split_id
            or identity.split_content_hash != self.benchmark_split.content_hash
            or identity.parameter_digest != candidate_digest(self.parameters)
            or self.input_artifact.content_digest != self.input_digest
            or self.fixed_parameters_artifact.content_digest != self.fixed_parameters_digest
        ):
            raise ValueError("evaluation request identity does not match its closed dispatch")
        if len(self.required_metric_pointers) != len(set(self.required_metric_pointers)):
            raise ValueError("required metric pointers must be unique")
        return self

    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


class EvaluatorRunner(Protocol):
    @property
    def binding(self) -> EvaluatorBinding: ...

    def evaluate(self, request: EvaluationRequest) -> RepeatEvidence: ...


class AdoptedStudyIdentity(StrictSpecModel):
    schema_version: Literal["organelleverse.adopted-study-identity.v1"] = (
        "organelleverse.adopted-study-identity.v1"
    )
    target: CapabilityIdentityV3
    contract_digest: Digest
    strategy_binding: StrategyBinding
    evaluator_binding: EvaluatorBinding
    benchmark_digest: Digest
    budget_digest: Digest
    input_digest: Digest
    fixed_parameters_digest: Digest
    environment_digest: Digest
    objective_order: tuple[SafeText, ...] = Field(min_length=1)
    finalist_limit: Annotated[StrictInt, Field(ge=1, le=256)]
    successive_halving_digest: Digest | None = None

    @classmethod
    def from_request(cls, request: AdaptiveStudyRequest) -> AdoptedStudyIdentity:
        return cls(
            target=request.contract.target,
            contract_digest=request.contract.digest,
            strategy_binding=request.strategy_binding,
            evaluator_binding=request.evaluator_binding,
            benchmark_digest=canonical_digest(request.contract.benchmark.model_dump(mode="json")),
            budget_digest=canonical_digest(request.contract.budget.model_dump(mode="json")),
            input_digest=request.input_digest,
            fixed_parameters_digest=request.fixed_parameters_digest,
            environment_digest=request.environment_digest,
            objective_order=request.objective_order,
            finalist_limit=request.finalist_limit,
            successive_halving_digest=(
                None
                if request.successive_halving is None
                else canonical_digest(request.successive_halving.model_dump(mode="json"))
            ),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        _canonical_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "AdoptedStudyIdentity",
    "EvaluationRequest",
    "EvaluatorRunner",
    "canonical_digest",
]
