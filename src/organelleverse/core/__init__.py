"""Canonical immutable contracts shared by Python and Agent operations."""

from .artifacts import ArtifactRef
from .data import LineageRecord, OrganelleData
from .errors import (
    OrganelleContractError,
    OrganelleDependencyError,
    OrganelleError,
    OrganelleExecutionError,
    OrganelleInputError,
    OrganelleInternalError,
    OrganelleNeedsInput,
    OrganelleParameterError,
    OrganellePermissionError,
)
from .genome import OrganelleGenome, OrganelleMetadata, OrganelleType
from .input_requests import (
    NEEDS_INPUT_SCHEMA_VERSION,
    NeedsInputRequest,
    make_needs_input_request,
)
from .provenance import ResultProvenance
from .result import (
    ErrorDetail,
    Finding,
    OperationSuggestion,
    OrganelleResult,
    ResultScope,
    ResultStatus,
)

__all__ = [
    "NEEDS_INPUT_SCHEMA_VERSION",
    "ArtifactRef",
    "ErrorDetail",
    "Finding",
    "LineageRecord",
    "NeedsInputRequest",
    "OperationSuggestion",
    "OrganelleContractError",
    "OrganelleData",
    "OrganelleDependencyError",
    "OrganelleError",
    "OrganelleExecutionError",
    "OrganelleGenome",
    "OrganelleInputError",
    "OrganelleInternalError",
    "OrganelleMetadata",
    "OrganelleNeedsInput",
    "OrganelleParameterError",
    "OrganellePermissionError",
    "OrganelleResult",
    "OrganelleType",
    "ResultProvenance",
    "ResultScope",
    "ResultStatus",
    "make_needs_input_request",
]
