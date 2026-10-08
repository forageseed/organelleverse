"""Decorator for registering canonical operation callables."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

from .registry import CoreObject, OperationRegistry
from .registry import registry as default_registry
from .spec import OperationSpec

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result", bound=CoreObject)


def operation(
    *,
    spec: OperationSpec,
    registry: OperationRegistry = default_registry,
) -> Callable[[Callable[Parameters, Result]], Callable[Parameters, Result]]:
    """Register a function and return the checked, signature-preserving wrapper."""

    def decorate(function: Callable[Parameters, Result]) -> Callable[Parameters, Result]:
        return cast(Callable[Parameters, Result], registry.register(spec, function).function)

    return decorate
