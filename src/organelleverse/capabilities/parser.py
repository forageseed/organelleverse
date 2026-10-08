"""Read-only ``capability.toml`` -> :class:`CapabilityBundle` parser.

Discovery-safe by construction: parsing only ever reads bytes from *path* and
hands a plain ``dict`` to Pydantic. It never imports ``contract.callable_locator``,
never starts a subprocess, and never touches the network - there is simply no
code path here that could do any of those things.

Every declared path field (``fixture.input.path``, ``fixture.expect``) is kept
exactly as written in the TOML. This parser never calls ``.resolve()`` or
joins them against *path* - a relative fixture path in the source document
stays a relative string in the returned contract.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from organelleverse.core.errors import OrganelleContractError

from .models import (
    SUPPORTED_BUNDLE_SCHEMA_VERSIONS,
    BundleDocument,
    CapabilityBundle,
    PluginCapabilityBundle,
)

_BUNDLE_MODELS_BY_SCHEMA: dict[str, type[BundleDocument]] = {
    "organelleverse.capability.v1": CapabilityBundle,
    "organelleverse.capability.v2": PluginCapabilityBundle,
}


def parse_capability_bundle(path: Path) -> BundleDocument:
    """Parse one ``capability.toml`` into a validated v1 or v2 bundle document.

    Dispatch happens on the top-level ``schema`` field *before* any
    field-level validation: an unsupported version fails with
    ``capability.schema_unsupported`` naming what it found and what this
    installation supports, never with a confusing field error and never by
    silently parsing a newer document as an older model.
    """
    document = _read_toml(path)
    document = _inject_default_operation_id(document)
    schema = document.get("schema")
    model = _BUNDLE_MODELS_BY_SCHEMA.get(schema) if isinstance(schema, str) else None
    if model is None:
        raise _parser_error(
            path,
            code="capability.schema_unsupported",
            message="capability bundle schema is not supported by this installation",
            details={
                "schema": document.get("schema"),
                "supported": sorted(SUPPORTED_BUNDLE_SCHEMA_VERSIONS),
            },
        )
    try:
        return model.model_validate(document)
    except ValidationError as error:
        raise _parser_error(
            path,
            code="capability.bundle_invalid_contract",
            message="capability bundle failed contract validation",
            details={"validation_errors": _summarize(error)},
        ) from error


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise _parser_error(
            path,
            code="capability.bundle_read_failed",
            message="capability bundle file could not be read",
            details={"reason": str(error)},
        ) from error
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _parser_error(
            path,
            code="capability.bundle_invalid_encoding",
            message="capability bundle is not valid UTF-8",
            details={"reason": str(error)},
        ) from error
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise _parser_error(
            path,
            code="capability.bundle_invalid_toml",
            message="capability bundle is not valid TOML",
            details={"reason": str(error)},
        ) from error


def _inject_default_operation_id(document: dict[str, Any]) -> dict[str, Any]:
    """Default ``contract.operation_id`` to ``capability.id`` when omitted.

    Never overrides an explicitly declared ``operation_id``: a mismatch
    between an explicit value and ``capability.id`` is left for
    ``CapabilityBundle``'s own validator to reject.
    """
    capability_section = document.get("capability")
    if not isinstance(capability_section, dict) or "id" not in capability_section:
        return document
    contract_section = document.get("contract")
    if not isinstance(contract_section, dict) or "operation_id" in contract_section:
        return document
    typed_contract_section = cast("dict[str, Any]", contract_section)
    updated_contract: dict[str, Any] = dict(typed_contract_section)
    updated_contract["operation_id"] = capability_section["id"]
    updated_document = dict(document)
    updated_document["contract"] = updated_contract
    return updated_document


def _summarize(error: ValidationError) -> list[dict[str, object]]:
    return [
        {
            "location": [str(part) for part in detail["loc"]],
            "message": detail["msg"],
            "type": detail["type"],
        }
        for detail in error.errors()
    ]


def _parser_error(
    path: Path,
    *,
    code: str,
    message: str,
    details: dict[str, object],
) -> OrganelleContractError:
    return OrganelleContractError(
        code=code,
        message=message,
        details={"bundle_path": str(path), **details},
    )


__all__ = ["parse_capability_bundle"]
