"""Structured errors shared by Python and Agent adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .input_requests import NeedsInputRequest


class OrganelleError(Exception):
    """Base class for stable, serializable OrganelleVerse errors."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        retryable: bool = False,
        suggested_action: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        from .frozen import freeze_json

        self.details = freeze_json(details if details is not None else {})
        self.retryable = retryable
        self.suggested_action = freeze_json(
            suggested_action if suggested_action is not None else {}
        )

    def as_dict(self) -> dict[str, Any]:
        from .frozen import thaw_json

        return {
            "error_code": self.code,
            "message": self.message,
            "details": thaw_json(self.details),
            "retryable": self.retryable,
            "suggested_action": thaw_json(self.suggested_action),
        }


class OrganelleInputError(OrganelleError):
    """Input file or scientific input contract is invalid."""


class OrganelleConflictError(OrganelleError):
    """A valid mutation conflicts with newer or still-referenced state."""


class OrganelleParameterError(OrganelleError):
    """An operation parameter is invalid."""


class OrganelleDependencyError(OrganelleError):
    """A declared package, executable, model, or database is unavailable."""


class OrganellePermissionError(OrganelleError):
    """An operation lacks permission for its declared side effect."""


class OrganelleExecutionError(OrganelleError):
    """An operation started but did not produce a usable result."""


class OrganelleContractError(OrganelleError):
    """Runtime behavior violates the registered public contract."""


class OrganelleInternalError(OrganelleError):
    """An internal invariant failed; callers should not silently recover."""


class OrganelleNeedsInput(OrganelleError):
    """Missing information that a person or Agent can legitimately supply.

    This is not a scientific failure. Direct Python and Registry invocations
    raise this exception; the Agent JSON transport converts it into a typed
    ``needs_input`` envelope that precedes the generic error payload and carries
    a deterministic :class:`~organelleverse.core.input_requests.NeedsInputRequest`.
    """

    def __init__(
        self,
        *,
        needs_input: NeedsInputRequest,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        retryable: bool = False,
        suggested_action: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            details=details,
            retryable=retryable,
            suggested_action=suggested_action,
        )
        self.needs_input = needs_input
