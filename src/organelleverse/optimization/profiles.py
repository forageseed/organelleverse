"""Pure projections from admitted capabilities to optimization contracts."""

from __future__ import annotations

import math

from pydantic import JsonValue, ValidationError

from organelleverse.capabilities.index import CapabilityEntry, CapabilityStatus
from organelleverse.capabilities.models import PluginCapabilityBundle
from organelleverse.capabilities.plugin_descriptor import PluginField, describe_plugin
from organelleverse.core.errors import OrganelleContractError

from .models import (
    ObjectiveSpec,
    OptimizationBudget,
    OptimizationContract,
    OptimizationStrategy,
    ParameterDomain,
)


def contract_from_capability_entry(
    entry: CapabilityEntry,
    *,
    seed: int = 0,
) -> OptimizationContract:
    """Normalize one admitted legacy plugin declaration without losing identity."""

    if entry.status is not CapabilityStatus.ADMITTED:
        raise OrganelleContractError(
            code="capability.not_admitted",
            message="optimization profiles require an admitted capability",
            details={"capability_id": entry.capability_id},
        )
    bundle = entry.bundle
    if not isinstance(bundle, PluginCapabilityBundle):
        raise OrganelleContractError(
            code="capability.plugin_required",
            message="legacy optimization profiles require a plugin capability",
            details={"capability_id": entry.capability_id},
        )
    optimization = bundle.contract.optimization
    if optimization is None:
        raise OrganelleContractError(
            code="capability.optimization_not_declared",
            message="the capability declares no optimization profile",
            details={"capability_id": entry.capability_id},
        )
    fields = {field.name: field for field in describe_plugin(bundle).parameters}
    parameters = tuple(
        _declared_domain(fields, name, capability_id=entry.capability_id)
        for name in optimization.parameters
    )
    strategies: tuple[OptimizationStrategy, ...] = (
        ("explicit", "grid", "random", "agent")
        if all(_is_grid_searchable(domain) for domain in parameters)
        else ("explicit", "random", "agent")
    )
    return OptimizationContract(
        target_capability_id=entry.capability_id,
        target_bundle_version=bundle.capability.bundle_version,
        target_contract_version=bundle.contract.contract_version,
        bundle_content_hash=entry.content_hash,
        execution_identity=(
            None if entry.execution_identity is None else entry.execution_identity.digest
        ),
        parameters=parameters,
        objective=ObjectiveSpec(direction=optimization.direction),
        strategies=strategies,
        seed=seed,
        budget=OptimizationBudget(
            max_trials=optimization.max_trials,
            parallelism=optimization.parallelism,
        ),
    )


def _domain_from_plugin_field(
    field: PluginField,
    *,
    capability_id: str,
) -> ParameterDomain:
    schema = field.json_schema
    try:
        enum = schema.get("enum")
        if isinstance(enum, list):
            _reject_unsupported_constraints(
                schema,
                allowed={"type", "enum"},
                capability_id=capability_id,
                parameter=field.name,
            )
            schema_type = schema.get("type")
            if not isinstance(schema_type, str) or not all(
                _matches_json_schema_type(value, schema_type) for value in enum
            ):
                raise _domain_not_closed(capability_id, field.name)
            return ParameterDomain(
                name=field.name,
                kind="categorical",
                values=tuple(enum),  # pyright: ignore[reportArgumentType]
            )
        if schema.get("type") == "boolean":
            _reject_unsupported_constraints(
                schema,
                allowed={"type"},
                capability_id=capability_id,
                parameter=field.name,
            )
            return ParameterDomain(name=field.name, kind="boolean")
        if schema.get("type") == "integer":
            _reject_unsupported_constraints(
                schema,
                allowed={"type", "minimum", "maximum", "multipleOf"},
                capability_id=capability_id,
                parameter=field.name,
            )
            domain = ParameterDomain(
                name=field.name,
                kind="integer",
                minimum=_integer(schema, "minimum"),
                maximum=_integer(schema, "maximum"),
                step=_integer(schema, "multipleOf"),
            )
            assert type(domain.minimum) is int
            assert type(domain.step) is int
            if domain.minimum % domain.step != 0:
                raise _domain_not_closed(capability_id, field.name)
            return domain
        if schema.get("type") == "number":
            _reject_unsupported_constraints(
                schema,
                allowed={"type", "minimum", "maximum"},
                capability_id=capability_id,
                parameter=field.name,
            )
            return ParameterDomain(
                name=field.name,
                kind="number",
                minimum=_number(schema, "minimum"),
                maximum=_number(schema, "maximum"),
            )
    except ValidationError as error:
        raise _domain_not_closed(capability_id, field.name) from error
    raise _domain_not_closed(capability_id, field.name)


def _declared_domain(
    fields: dict[str, PluginField],
    name: str,
    *,
    capability_id: str,
) -> ParameterDomain:
    field = fields.get(name)
    if field is None:
        raise _domain_not_closed(capability_id, name)
    return _domain_from_plugin_field(field, capability_id=capability_id)


def _number(schema: dict[str, JsonValue], key: str) -> int | float | None:
    value = schema.get(key)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _integer(schema: dict[str, JsonValue], key: str) -> int | float | None:
    value = _number(schema, key)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _matches_json_schema_type(value: JsonValue, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "number":
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    if schema_type == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and math.isfinite(value) and value.is_integer()
        )
    return False


_ANNOTATION_KEYS = frozenset(
    {
        "$comment",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "ui:fileExt",
        "ui:placeholder",
        "ui_file_ext",
        "ui_placeholder",
        "writeOnly",
    }
)


def _reject_unsupported_constraints(
    schema: dict[str, JsonValue],
    *,
    allowed: set[str],
    capability_id: str,
    parameter: str,
) -> None:
    unsupported = set(schema) - allowed - _ANNOTATION_KEYS
    if unsupported:
        raise _domain_not_closed(capability_id, parameter)


def _is_grid_searchable(domain: ParameterDomain) -> bool:
    return domain.kind != "number" or bool(domain.values)


def _domain_not_closed(capability_id: str, parameter: str) -> OrganelleContractError:
    return OrganelleContractError(
        code="optimization.domain_not_closed",
        message="optimized parameters require a closed JSON Schema domain",
        details={"capability_id": capability_id, "parameter": parameter},
    )


__all__ = ["contract_from_capability_entry"]
