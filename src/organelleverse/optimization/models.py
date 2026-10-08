"""Strict, frozen control-plane contracts for governed optimization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Annotated, Literal, Self, TypeAlias, cast

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from organelleverse.operations.spec import StrictSpecModel

StrictNumber: TypeAlias = StrictInt | StrictFloat
JsonScalar: TypeAlias = StrictStr | StrictInt | StrictFloat | StrictBool | None
OptimizationStrategy: TypeAlias = Literal["explicit", "grid", "random", "agent"]
_HASH_PATTERN = r"^sha256:[0-9a-f]{64}$"


class ParameterDomain(StrictSpecModel):
    """One closed parameter search domain."""

    name: Annotated[StrictStr, Field(min_length=1)]
    kind: Literal["integer", "number", "categorical", "boolean"]
    minimum: StrictNumber | None = None
    maximum: StrictNumber | None = None
    step: StrictNumber | None = None
    logarithmic: StrictBool = False
    values: tuple[JsonScalar, ...] = ()

    @model_validator(mode="after")
    def validate_closed_domain(self) -> Self:
        if not self.name.strip():
            raise ValueError("parameter domain name must not be blank")
        if self.kind in {"integer", "number"}:
            self._validate_numeric_domain()
        elif self.kind == "categorical":
            self._validate_categorical_domain()
        else:
            self._reject_numeric_declarations()
            if self.values:
                raise ValueError("boolean domains do not declare explicit values")
        return self

    def _validate_numeric_domain(self) -> None:
        if self.minimum is None or self.maximum is None:
            raise ValueError("numeric domains require closed minimum and maximum")
        bounds = (self.minimum, self.maximum)
        if any(isinstance(value, bool) for value in bounds) or any(
            not math.isfinite(value) for value in bounds
        ):
            raise ValueError("numeric domain bounds must be finite numbers")
        if self.minimum >= self.maximum:
            raise ValueError("numeric domain minimum must be less than maximum")
        if self.kind == "integer":
            if not all(type(value) is int for value in (*bounds, self.step)):
                raise ValueError("integer domains require integer bounds and step")
            if self.step is None or self.step <= 0:
                raise ValueError("integer domains require a positive step")
            if self.values:
                raise ValueError("integer domains are represented by bounds and step")
        elif self.step is not None and (
            isinstance(self.step, bool) or not math.isfinite(self.step) or self.step <= 0
        ):
            raise ValueError("numeric domain step must be finite and positive")
        if self.logarithmic and self.minimum <= 0:
            raise ValueError("logarithmic domains require a positive minimum")
        if self.kind == "number" and self.values:
            self._validate_explicit_numeric_values()

    def _validate_explicit_numeric_values(self) -> None:
        assert self.minimum is not None and self.maximum is not None
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < self.minimum
            or value > self.maximum
            for value in self.values
        ):
            raise ValueError("explicit numeric values must be finite and within bounds")
        self._require_unique_values()

    def _validate_categorical_domain(self) -> None:
        self._reject_numeric_declarations()
        if len(self.values) < 2:
            raise ValueError("categorical domains require at least two values")
        if any(isinstance(value, float) and not math.isfinite(value) for value in self.values):
            raise ValueError("categorical values must be finite JSON scalars")
        self._require_unique_values()

    def _reject_numeric_declarations(self) -> None:
        if any(value is not None for value in (self.minimum, self.maximum, self.step)):
            raise ValueError(f"{self.kind} domains do not declare numeric bounds")
        if self.logarithmic:
            raise ValueError(f"{self.kind} domains cannot be logarithmic")

    def _require_unique_values(self) -> None:
        encoded = [
            json.dumps(
                _canonical_json(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for value in self.values
        ]
        if len(encoded) != len(set(encoded)):
            raise ValueError("domain values must be unique")


class ObjectiveSpec(StrictSpecModel):
    """One named optimization objective and its preferred direction."""

    name: Literal["plugin_optimization_score"] = "plugin_optimization_score"
    direction: Literal["maximize", "minimize"]


class OptimizationBudget(StrictSpecModel):
    """Hard trial and concurrency limits."""

    max_trials: Annotated[StrictInt, Field(ge=1, le=256)]
    parallelism: Annotated[StrictInt, Field(ge=1, le=32)]

    @model_validator(mode="after")
    def validate_parallelism(self) -> Self:
        if self.parallelism > self.max_trials:
            raise ValueError("parallelism must not exceed max_trials")
        return self


class OptimizationContract(StrictSpecModel):
    """Versioned target identity and complete governed search declaration."""

    schema_version: Literal["organelleverse.optimization.contract.v2"] = (
        "organelleverse.optimization.contract.v2"
    )
    target_capability_id: Annotated[StrictStr, Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")]
    target_bundle_version: Annotated[StrictStr, Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")]
    target_contract_version: Annotated[StrictStr, Field(pattern=r"^[0-9]+\.[0-9]+$")]
    bundle_content_hash: Annotated[StrictStr, Field(pattern=_HASH_PATTERN)]
    execution_identity: Annotated[StrictStr, Field(pattern=_HASH_PATTERN)] | None = None
    parameters: tuple[ParameterDomain, ...] = Field(min_length=1)
    objective: ObjectiveSpec
    strategies: tuple[OptimizationStrategy, ...] = Field(min_length=1)
    seed: StrictInt
    budget: OptimizationBudget

    @model_validator(mode="after")
    def validate_unique_and_supported_definitions(self) -> Self:
        parameter_names = [domain.name for domain in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("parameter domain names must be unique")
        if len(self.strategies) != len(set(self.strategies)):
            raise ValueError("optimization strategies must be unique")
        if "grid" in self.strategies and any(
            domain.kind == "number" and not domain.values for domain in self.parameters
        ):
            raise ValueError("grid strategy requires explicit values for continuous number domains")
        return self

    @property
    def digest(self) -> str:
        """Canonical identity of every executable contract decision."""

        canonical = json.dumps(
            _canonical_json(self.model_dump(mode="json")),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _canonical_json(value: object) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("canonical JSON forbids NaN and Infinity")
        return {"$number": _canonical_number_text(value)}
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _canonical_json(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(Sequence[object], value)
        return [_canonical_json(item) for item in sequence]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def _canonical_number_text(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if value == 0.0:
        return "0"
    sign, decimal_digits, decimal_exponent = Decimal(str(value)).as_tuple()
    digits = list(decimal_digits)
    exponent = cast(int, decimal_exponent)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    text = "".join(str(digit) for digit in digits)
    point = len(text) + exponent
    if point <= 0:
        text = f"0.{('0' * -point)}{text}"
    elif point < len(text):
        text = f"{text[:point]}.{text[point:]}"
    elif point > len(text):
        text = f"{text}{'0' * (point - len(text))}"
    return f"-{text}" if sign else text


__all__ = [
    "ObjectiveSpec",
    "OptimizationBudget",
    "OptimizationContract",
    "ParameterDomain",
]
