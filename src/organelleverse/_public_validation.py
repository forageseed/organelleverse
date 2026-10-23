"""Lightweight declaration of the shared Python/operation validation boundary."""

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, Protocol, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")


class _PublicMetadata(Protocol):
    __wrapped__: Callable[..., object]


def unchecked_public(function: Callable[P, R]) -> Callable[P, R]:
    """Remove only our boundary, preserving any scientific decorators."""
    if hasattr(function, "__ov_operation_id__"):
        return cast(Callable[P, R], cast(_PublicMetadata, function).__wrapped__)
    return function


def validated_public(
    operation_id: str, *, python_only: tuple[str, ...] = ()
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Preserve the Python signature and injected objects; validate scientific values.

    The operations layer is imported on invocation, so importing a scientific
    module neither enumerates the catalog nor creates an import cycle.
    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def checked(*args: P.args, **kwargs: P.kwargs) -> R:
            from organelleverse.operations.public_validation import invoke_python

            return cast(R, invoke_python(operation_id, function, args, kwargs, python_only))

        # Runtime metadata does not widen the public Callable[P, R] type.
        setattr(checked, "__ov_operation_id__", operation_id)  # noqa: B010
        setattr(checked, "__ov_python_only__", python_only)  # noqa: B010
        checked.__doc__ = (function.__doc__ or "") + (
            "\n\nValidation\n----------\n"
            "Scientific arguments use the shared operation validator; invalid "
            "arguments raise OrganelleParameterError before execution. "
            "The native Python signature and return value are preserved.\n"
            "A single path for a path-sequence parameter is normalized to a "
            "one-element list, using the same path-codec rule as JSON.\n"
            f"JSON fields: ov.operations.parameter_schema({operation_id!r}). "
            "Rich Python objects and injection interfaces are described in "
            "the installed public Python contract guide.\n"
            + (
                "Python-only injection arguments (passed unchanged): "
                + ", ".join(python_only)
                + ".\n"
                if python_only
                else ""
            )
        )
        return checked

    return decorate
