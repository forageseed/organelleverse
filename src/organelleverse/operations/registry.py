"""Operation registration and checked invocation."""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import wraps
from typing import Never, ParamSpec, Protocol, TypeAlias, TypeVar, cast

from pydantic import BaseModel, JsonValue, ValidationError

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleInputError,
    OrganelleParameterError,
)
from organelleverse.core.frozen import thaw_json
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult

from ._validation import validation_summaries, validation_summary
from .data_contracts import BUILTIN_DATA_CONTRACTS, DataContract, DataContractRegistry
from .dependencies import DependencyReport, check_dependencies
from .schemas import invocation_schema as compose_invocation_schema
from .schemas import invocation_schema_from_frozen, parameter_schema_from_frozen
from .schemas import parameter_schema as compose_parameter_schema
from .signature import OperationSignature, derive_operation_signature
from .spec import CoreKind, OperationSpec, OperationStage

CoreObject: TypeAlias = OrganelleGenome | OrganelleData | OrganelleResult
ParameterDecoder: TypeAlias = Callable[[type[BaseModel], Mapping[str, object]], BaseModel]
SuggestionValidator: TypeAlias = Callable[[OrganelleResult], None]

_logger = logging.getLogger(__name__)

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result", bound=CoreObject)


class SignatureAwareCallable(Protocol):
    """Callable metadata needed for an inspectable wrapped operation."""

    __signature__: inspect.Signature


class InvocationStrategy(Protocol):
    """Replace only the implementation call inside the shared validation path."""

    def invoke(
        self,
        input: CoreObject | None,
        parameters: Mapping[str, object],
    ) -> CoreObject: ...


class CapabilityBindingSource(Protocol):
    """Lazy, read-mostly capability source without a registry import cycle."""

    def list_specs(self) -> tuple[OperationSpec, ...]: ...

    def known_ids(self) -> tuple[str, ...]: ...

    def describe_spec(self, operation_id: str) -> OperationSpec | None: ...

    def parameter_schema(self, operation_id: str) -> dict[str, object] | None: ...

    def is_conflicted(self, operation_id: str) -> bool: ...

    def conflict_details(self, operation_id: str) -> dict[str, object]: ...

    def rejection(self, operation_id: str) -> tuple[str, str, dict[str, object]] | None: ...

    def resolve(self, operation_id: str) -> BoundOperation | None: ...


@dataclass(frozen=True)
class BoundOperation:
    """A registered specification and its checked public callable."""

    spec: OperationSpec
    implementation: Callable[..., CoreObject]
    function: Callable[..., CoreObject]
    signature: OperationSignature
    data_contracts: tuple[DataContract, ...]
    output_data_contracts: tuple[DataContract, ...] = ()
    suggestion_validator: SuggestionValidator | None = None
    invocation_strategy: InvocationStrategy | None = None

    def invoke(
        self,
        input: CoreObject | None,
        parameters: Mapping[str, object],
    ) -> CoreObject:
        """Route raw mapping values through the checked public callable."""
        try:
            raw_kwargs = cast(dict[object, object], dict(parameters))
        except (TypeError, ValueError) as error:
            raise _parameter_error(error) from error
        return self.invoke_with_parameter_decoder(
            input,
            cast(Mapping[str, object], raw_kwargs),
        )

    def invoke_with_parameter_decoder(
        self,
        input: CoreObject | None,
        parameters: Mapping[str, object],
        *,
        parameter_decoder: ParameterDecoder | None = None,
    ) -> CoreObject:
        """Run core and modality checks before decoding one parameter mapping."""
        if self.spec.stage is OperationStage.READ and input is not None:
            raise OrganelleInputError(
                code="input.unexpected_operation_core_input",
                message=f"{self.spec.operation_id} does not accept a core input",
                details={
                    "operation_id": self.spec.operation_id,
                    "provided_input_type": type(input).__name__,
                },
            )
        values = dict(parameters)
        if self.signature.input_name is not None:
            values[self.signature.input_name] = input
        return _invoke_checked_python(
            self.spec,
            self.signature,
            self.implementation,
            values,
            self.data_contracts,
            output_data_contracts=self.output_data_contracts,
            suggestion_validator=self.suggestion_validator,
            parameter_decoder=parameter_decoder,
            invocation_strategy=self.invocation_strategy,
        )

    def invoke_validated(self, input: CoreObject | None, parameters: BaseModel) -> CoreObject:
        """Invoke with the exact generated parameter model instance."""

        def use_validated(
            parameter_model: type[BaseModel],
            raw_parameters: Mapping[str, object],
        ) -> BaseModel:
            return parameters

        return self.invoke_with_parameter_decoder(
            input,
            {},
            parameter_decoder=use_validated,
        )


class OperationRegistry:
    """A deterministic registry of canonical operation callables."""

    def __init__(
        self,
        *,
        data_contracts: tuple[DataContract, ...] = (),
        capability_source: CapabilityBindingSource | None = None,
    ) -> None:
        self._bindings: dict[str, BoundOperation] = {}
        self._data_contracts = DataContractRegistry((*BUILTIN_DATA_CONTRACTS, *data_contracts))
        self._capability_source: CapabilityBindingSource | None = None
        self._capability_lock = threading.RLock()
        self._authoritative_operation_ids: set[str] = set()
        if capability_source is not None:
            self.attach_capability_source(capability_source)

    def attach_capability_source(self, source: CapabilityBindingSource) -> None:
        """Attach an already-discovered source without triggering discovery or imports."""
        with self._capability_lock:
            if self._capability_source is not None and self._capability_source is not source:
                raise OrganelleContractError(
                    code="contract.capability_source_already_attached",
                    message="this operation registry already has a capability source",
                )
            known_ids = set(source.known_ids())
            collisions = (known_ids & set(self._bindings)) - self._authoritative_operation_ids
            if collisions:
                operation_id = sorted(collisions)[0]
                raise OrganelleContractError(
                    code="contract.duplicate_operation_id",
                    message=f"duplicate operation ID: {operation_id}",
                    details={"operation_ids": sorted(collisions)},
                )
            self._capability_source = source
            # A capability may legitimately declare an id one of the released
            # operations already authoritatively serves - that is a staged
            # migration in progress (build and admit the capability first,
            # retire the release binding second), not an error, and
            # `require`/`list` already resolve the release binding first
            # unconditionally (see below) so behavior never changes here.
            # But a capability that is discovered, verified, and admitted and
            # then simply invisible with no trace is a debugging trap, so the
            # shadow is logged and stays queryable via
            # `shadowed_capability_ids()` instead of staying silent.
            shadowed = sorted(known_ids & self._authoritative_operation_ids)
            if shadowed:
                _logger.warning(
                    "capability source declares %d id(s) already served by a released "
                    "operation; the released binding wins and the capability stays "
                    "unreachable through require()/list(): %s",
                    len(shadowed),
                    ", ".join(shadowed),
                )

    def shadowed_capability_ids(self) -> tuple[str, ...]:
        """IDs an attached capability source declares that a released binding wins over.

        These ids stay fully admitted and fully real in the capability source
        - `require`/`list` simply never reach them, because the released
        binding always answers first (see `require()`/`list()` below). This
        is the queryable half of that decision: whatever it does not resolve
        can still be found here instead of vanishing without a trace.
        """
        if self._capability_source is None:
            return ()
        return tuple(
            sorted(set(self._capability_source.known_ids()) & self._authoritative_operation_ids)
        )

    def _ensure_catalog(self) -> None:
        """Hook used only by the lazy default release registry."""

    def _resolve_data_contracts(
        self,
        spec: OperationSpec,
        additional: tuple[DataContract, ...] = (),
    ) -> tuple[tuple[DataContract, ...], tuple[DataContract, ...]]:
        if spec.input_kind is CoreKind.DATA and len(set(spec.input_modalities)) != len(
            spec.input_modalities
        ):
            raise OrganelleContractError(
                code="contract.duplicate_operation_modality",
                message=f"{spec.operation_id} declares duplicate DATA input modalities",
                details={
                    "operation_id": spec.operation_id,
                    "modalities": list(spec.input_modalities),
                },
            )
        available = DataContractRegistry((*self._data_contracts.list(), *additional))
        inputs = (
            tuple(available.require(modality) for modality in spec.input_modalities)
            if spec.input_kind is CoreKind.DATA
            else ()
        )
        outputs = tuple(
            contract
            for modality in spec.output_modalities
            if (contract := available.find(modality)) is not None
        )
        return inputs, outputs

    def _suggestion_validator(self, source_spec: OperationSpec) -> SuggestionValidator:
        def validate_suggestions(result: OrganelleResult) -> None:
            """Enforce that every suggestion on *result* is followable."""
            for suggestion in result.suggested_operations:
                binding = self._bindings.get(suggestion.operation_id)
                target_spec = binding.spec if binding is not None else None
                frozen_schema: dict[str, object] | None = None
                if target_spec is None and self._capability_source is not None:
                    source = self._capability_source
                    if (
                        not source.is_conflicted(suggestion.operation_id)
                        and source.rejection(suggestion.operation_id) is None
                    ):
                        target_spec = source.describe_spec(suggestion.operation_id)
                        frozen_schema = source.parameter_schema(suggestion.operation_id)
                if target_spec is None:
                    raise OrganelleContractError(
                        code="contract.unknown_suggested_operation",
                        message=(
                            f"{source_spec.operation_id} suggested unknown operation "
                            f"{suggestion.operation_id!r}"
                        ),
                        details={
                            "operation_id": source_spec.operation_id,
                            "suggested_operation_id": suggestion.operation_id,
                        },
                    )
                target_kind = target_spec.input_kind
                if target_kind not in {CoreKind.NONE, source_spec.output_kind}:
                    raise OrganelleContractError(
                        code="contract.incompatible_suggested_operation",
                        message=(
                            f"{source_spec.operation_id} suggested "
                            f"{suggestion.operation_id!r}, which consumes "
                            f"{target_kind.value}"
                        ),
                        details={
                            "operation_id": source_spec.operation_id,
                            "suggested_operation_id": suggestion.operation_id,
                            "suggested_input_kind": target_kind.value,
                            "emitting_output_kind": source_spec.output_kind.value,
                        },
                    )
                raw_changes = thaw_json(suggestion.parameter_changes)
                validation_errors: list[dict[str, object]] = []
                if binding is not None:
                    try:
                        binding.signature.parameter_model.model_validate(raw_changes)
                    except ValidationError as error:
                        validation_errors = validation_summaries(error)
                elif frozen_schema is not None:
                    from jsonschema import Draft202012Validator

                    validation_errors = [
                        {
                            "message": error.message,
                            "path": list(error.absolute_path),
                        }
                        for error in Draft202012Validator(frozen_schema).iter_errors(  # pyright: ignore[reportUnknownMemberType]
                            cast(JsonValue, raw_changes)
                        )
                    ]
                else:
                    validation_errors = [
                        {"message": "target capability has no frozen parameter schema"}
                    ]
                if validation_errors:
                    raise OrganelleContractError(
                        code="contract.unsatisfiable_suggested_operation",
                        message=(
                            f"{source_spec.operation_id} suggested "
                            f"{suggestion.operation_id!r} with parameter changes its "
                            "contract rejects"
                        ),
                        details={
                            "operation_id": source_spec.operation_id,
                            "suggested_operation_id": suggestion.operation_id,
                            "validation_errors": validation_errors,
                        },
                    )

        return validate_suggestions

    def _adopt_capability_binding(self, binding: BoundOperation) -> BoundOperation:
        """Apply the same registry-owned data and suggestion checks as register()."""
        # Capability sources may expose a validated OperationSpec subclass.
        # Revalidating it as the base model would discard its declared contract
        # surface (notably v2 plugin ports and optimization metadata) before
        # the registry ever invokes it.
        spec = type(binding.spec).model_validate(binding.spec)
        inputs, outputs = self._resolve_data_contracts(spec, binding.data_contracts)
        return replace(
            binding,
            spec=spec,
            data_contracts=inputs,
            output_data_contracts=outputs,
            suggestion_validator=self._suggestion_validator(spec),
        )

    def register(
        self,
        spec: OperationSpec,
        function: Callable[Parameters, Result],
        *,
        data_contracts: tuple[DataContract, ...] = (),
    ) -> BoundOperation:
        """Register a typed implementation and return its checked binding."""
        validated_spec = OperationSpec.model_validate(spec)
        source = self._capability_source
        if validated_spec.operation_id in self._bindings or (
            source is not None and validated_spec.operation_id in source.known_ids()
        ):
            raise OrganelleContractError(
                code="contract.duplicate_operation_id",
                message=f"duplicate operation ID: {validated_spec.operation_id}",
            )
        resolved_contracts, resolved_output_contracts = self._resolve_data_contracts(
            validated_spec,
            data_contracts,
        )
        signature = derive_operation_signature(function, validated_spec)
        implementation = cast(Callable[..., CoreObject], function)
        validate_suggestions = self._suggestion_validator(validated_spec)

        @wraps(function)
        def checked(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
            known_names = signature.python_signature.parameters
            known_kwargs = {name: value for name, value in kwargs.items() if name in known_names}
            unknown_kwargs = {
                name: value for name, value in kwargs.items() if name not in known_names
            }
            try:
                bound = signature.python_signature.bind_partial(*args, **known_kwargs)
            except TypeError as error:
                raise _parameter_error(error) from error
            bound.apply_defaults()
            bound.arguments.update(unknown_kwargs)
            return cast(
                Result,
                _invoke_checked_python(
                    validated_spec,
                    signature,
                    implementation,
                    bound.arguments,
                    resolved_contracts,
                    output_data_contracts=resolved_output_contracts,
                    suggestion_validator=validate_suggestions,
                ),
            )

        cast(SignatureAwareCallable, checked).__signature__ = signature.python_signature
        binding = BoundOperation(
            spec=validated_spec,
            implementation=implementation,
            function=cast(Callable[..., CoreObject], checked),
            signature=signature,
            data_contracts=resolved_contracts,
            output_data_contracts=resolved_output_contracts,
            suggestion_validator=validate_suggestions,
        )
        self._bindings[validated_spec.operation_id] = binding
        return binding

    def register_authoritative(
        self,
        spec: OperationSpec,
        function: Callable[Parameters, Result],
        *,
        data_contracts: tuple[DataContract, ...] = (),
    ) -> BoundOperation:
        """Register a release binding that wins over a later capability source."""
        binding = self.register(spec, function, data_contracts=data_contracts)
        self._authoritative_operation_ids.add(binding.spec.operation_id)
        return binding

    def list(
        self,
        *,
        stage: OperationStage | None = None,
        input_kind: CoreKind | None = None,
        organelle: str | None = None,
        input_modality: str | None = None,
        output_modality: str | None = None,
    ) -> tuple[OperationSpec, ...]:
        """List registered specs in operation-ID order with exact filters."""
        self._ensure_catalog()
        source = self._capability_source
        bound_specs = tuple(binding.spec for binding in self._bindings.values())
        bound_ids = {spec.operation_id for spec in bound_specs}
        capability_specs = (
            tuple(spec for spec in source.list_specs() if spec.operation_id not in bound_ids)
            if source is not None
            else ()
        )
        specs = (*bound_specs, *capability_specs)
        return tuple(
            sorted(
                (
                    spec
                    for spec in specs
                    if (stage is None or spec.stage is stage)
                    and (input_kind is None or spec.input_kind is input_kind)
                    and (organelle is None or organelle in spec.organelle_types)
                    and (input_modality is None or input_modality in spec.input_modalities)
                    and (output_modality is None or output_modality in spec.output_modalities)
                ),
                key=lambda spec: spec.operation_id,
            )
        )

    def describe(self, operation_id: str) -> OperationSpec:
        """Return the registered specification for an operation ID."""
        self._ensure_catalog()
        binding = self._bindings.get(operation_id)
        if binding is not None:
            return binding.spec
        source = self._capability_source
        if source is not None:
            self._raise_capability_conflict(source, operation_id)
            spec = source.describe_spec(operation_id)
            if spec is not None:
                return spec
        self._raise_unknown_operation(operation_id)

    def parameter_schema(self, operation_id: str) -> dict[str, object]:
        """Return the strict parameter JSON Schema for an operation."""
        self._ensure_catalog()
        binding = self._bindings.get(operation_id)
        if binding is not None:
            return compose_parameter_schema(binding)
        source = self._capability_source
        if source is not None:
            self._raise_capability_conflict(source, operation_id)
            schema = source.parameter_schema(operation_id)
            if schema is not None:
                return parameter_schema_from_frozen(schema)
            self._raise_capability_rejection(source, operation_id)
        self._raise_unknown_operation(operation_id)

    def invocation_schema(self, operation_id: str) -> dict[str, object]:
        """Return the complete strict Agent invocation JSON Schema."""
        self._ensure_catalog()
        binding = self._bindings.get(operation_id)
        if binding is not None:
            return compose_invocation_schema(binding)
        source = self._capability_source
        if source is not None:
            self._raise_capability_conflict(source, operation_id)
            spec = source.describe_spec(operation_id)
            schema = source.parameter_schema(operation_id)
            if spec is not None and schema is not None:
                inputs, _ = self._resolve_data_contracts(spec)
                return invocation_schema_from_frozen(spec, schema, data_contracts=inputs)
            self._raise_capability_rejection(source, operation_id)
        self._raise_unknown_operation(operation_id)

    def check_dependencies(self, operation_id: str) -> DependencyReport:
        """Inspect the registered operation's declared dependencies without executing it."""
        return check_dependencies(self.describe(operation_id))

    def invoke(
        self,
        operation_id: str,
        *,
        input: CoreObject | None,
        parameters: Mapping[str, object],
    ) -> CoreObject:
        """Invoke an operation by ID with mapping parameters."""
        return self.require(operation_id).invoke(input, parameters)

    def require(self, operation_id: str) -> BoundOperation:
        """Return a binding or raise a stable error for unknown IDs."""
        self._ensure_catalog()
        existing = self._bindings.get(operation_id)
        if existing is not None:
            return existing
        source = self._capability_source
        if source is not None:
            self._raise_capability_conflict(source, operation_id)
            self._raise_capability_rejection(source, operation_id)
            with self._capability_lock:
                existing = self._bindings.get(operation_id)
                if existing is not None:
                    return existing
                resolved = source.resolve(operation_id)
                if resolved is not None:
                    resolved = self._adopt_capability_binding(resolved)
                    self._bindings[operation_id] = resolved
                    return resolved
        self._raise_unknown_operation(operation_id)

    @staticmethod
    def _raise_capability_conflict(
        source: CapabilityBindingSource,
        operation_id: str,
    ) -> None:
        if source.is_conflicted(operation_id):
            raise OrganelleContractError(
                code="capability.conflict",
                message=f"conflicting capability bundles declare {operation_id}",
                details=source.conflict_details(operation_id),
            )

    @staticmethod
    def _raise_capability_rejection(
        source: CapabilityBindingSource,
        operation_id: str,
    ) -> None:
        diagnostic = source.rejection(operation_id)
        if diagnostic is not None:
            code, message, details = diagnostic
            raise OrganelleContractError(
                code=code,
                message=message,
                details=details,
            )

    @staticmethod
    def _raise_unknown_operation(operation_id: str) -> Never:
        raise OrganelleInputError(
            code="input.unknown_operation",
            message=f"unknown operation: {operation_id}",
        )


class _ReleaseOperationRegistry(OperationRegistry):
    """Default registry that loads its release catalog on first discovery/use."""

    def __init__(self) -> None:
        super().__init__()
        self._catalog_loaded = False
        self._catalog_lock = threading.RLock()

    def _ensure_catalog(self) -> None:
        if self._catalog_loaded:
            return
        with self._catalog_lock:
            if self._catalog_loaded:
                return
            from .catalog import load_release_catalog

            staging = OperationRegistry()
            load_release_catalog(staging)
            duplicate_ids = sorted(self._bindings.keys() & staging._bindings.keys())
            if duplicate_ids:
                raise OrganelleContractError(
                    code="contract.duplicate_operation_id",
                    message=f"duplicate operation ID: {duplicate_ids[0]}",
                    details={"operation_ids": duplicate_ids},
                )
            self._bindings.update(staging._bindings)
            self._authoritative_operation_ids.update(staging._bindings)
            self._attach_default_capability_source()
            self._catalog_loaded = True

    def _attach_default_capability_source(self) -> None:
        """Give the shared default registry a real, safe path to admitted capabilities.

        This is opt-out, not opt-in, and deliberately so: `default_verification_store()`
        (`capabilities/verification.py`) and `TrustStore()` (`capabilities/trust.py`)
        already default to `~/.organelleverse/...` without a caller passing a path, and
        `discover_capabilities()` (`capabilities/discovery.py`) with no arguments already
        scans the standard core/local/project/package roots by default - "discovery
        defaults on" is a choice this codebase already made, one layer down. Requiring an
        explicit call here on top of that would not add a safety check; it would just
        relocate the same silent default one layer up, one call an application would have
        to remember to make - which is exactly the gap this task exists to close
        (`attach_capability_source` had no production caller at all).

        It is safe to default on because reachability here is not the same as
        trust or execution:

        - Only ADMITTED capabilities become visible at all
          (`CapabilityIndex.list()`/`IndexBindingSource.resolve()` in
          `capabilities/index.py`), and admission demands a byte-for-byte matching real
          `VerificationRecord` produced by the real verify -> admit pipeline
          (`capabilities/admission.py`). Auto-attaching cannot make an unverified or
          fabricated bundle appear.
        - For a `core`-channel capability, "reachable" already means "executes in the
          same in-process distribution the 20 released operations already run in" - no
          new trust boundary is crossed.
        - For every other channel, reachability through `list()`/`describe()`/`resolve()`
          is not executability: `WorkerInvocationStrategy.invoke()` (`capabilities/worker.py`)
          re-checks `TrustStore.is_trusted()` on every dispatch and refuses to run an
          identity the owner has not explicitly `trust()`-ed. Auto-attach can, at most,
          make an admitted capability *visible* in a listing or schema; it can never make
          one *execute* without a separate, explicit trust decision.

        This call is scoped to the shared default registry class only: an application
        that builds its own `OperationRegistry()` still gets no capability source unless
        it asks for one, exactly as before.

        Also deliberately unguarded: at the point this runs, `self._bindings` is exactly
        the 20 just-loaded release bindings, every one of which was just added to
        `self._authoritative_operation_ids` above, so `attach_capability_source`'s own
        collision check is structurally empty here - nothing external has had a chance to
        `register()` yet, because `register()` calls `_ensure_catalog()` first. A real
        `discover_capabilities()` failure (a broken environment, not a collision) is left
        to surface rather than being swallowed, matching this codebase's existing
        no-silent-fallback stance (e.g. `TrustStore` refusing an unsafe file mode instead
        of ignoring it).
        """
        from organelleverse.capabilities.discovery import discover_capabilities

        index = discover_capabilities()
        self.attach_capability_source(index.binding_source())

    def register(
        self,
        spec: OperationSpec,
        function: Callable[Parameters, Result],
        *,
        data_contracts: tuple[DataContract, ...] = (),
    ) -> BoundOperation:
        """Reserve the canonical release catalog before external registration."""
        self._ensure_catalog()
        with self._catalog_lock:
            return super().register(spec, function, data_contracts=data_contracts)


def _invoke_checked_python(
    spec: OperationSpec,
    signature: OperationSignature,
    implementation: Callable[..., CoreObject],
    bound_arguments: Mapping[str, object],
    data_contracts: tuple[DataContract, ...],
    *,
    output_data_contracts: tuple[DataContract, ...] = (),
    suggestion_validator: SuggestionValidator | None = None,
    parameter_decoder: ParameterDecoder | None = None,
    invocation_strategy: InvocationStrategy | None = None,
) -> CoreObject:
    values = dict(bound_arguments)
    core_input: CoreObject | None = None
    if signature.input_name is not None:
        if signature.input_name not in values:
            raise OrganelleInputError(
                code="input.operation_core_type_mismatch",
                message=f"{spec.operation_id} requires {spec.input_kind.value}",
            )
        candidate_input = values.pop(signature.input_name)
        if signature.input_sequence:
            # Ruling 3: a sequence of exactly the declared core type; every
            # element gets the same type check and frozen-model revalidation
            # a single core input gets. The sequence rides as the positional
            # core input - parameters never see it, exactly as single inputs.
            if not isinstance(candidate_input, (list, tuple)) or not all(
                type(element) is signature.input_type for element in candidate_input
            ):
                raise OrganelleInputError(
                    code="input.operation_core_type_mismatch",
                    message=(
                        f"{spec.operation_id} requires a sequence of "
                        f"{spec.input_kind.value} objects"
                    ),
                )
            element_type = cast("type[BaseModel]", signature.input_type)
            core_input = [
                _revalidate_core_input(
                    spec, element_type, cast(CoreObject, element)
                )
                for element in candidate_input
            ]
            for element in core_input:
                _validate_data_modality(spec, element, data_contracts)
        else:
            if type(candidate_input) is not signature.input_type:
                raise OrganelleInputError(
                    code="input.operation_core_type_mismatch",
                    message=f"{spec.operation_id} requires {spec.input_kind.value}",
                )
            core_input = _revalidate_core_input(
                spec,
                cast(type[BaseModel], signature.input_type),
                cast(CoreObject, candidate_input),
            )
            _validate_data_modality(spec, core_input, data_contracts)
    decoded = (
        _validate_parameters(signature, values)
        if parameter_decoder is None
        else parameter_decoder(signature.parameter_model, values)
    )
    validated = _revalidate_parameter_instance(spec, signature, decoded)
    parameters = _typed_parameter_values(validated)
    if invocation_strategy is None:
        result = (
            implementation(**parameters)
            if signature.input_name is None
            else implementation(core_input, **parameters)
        )
    else:
        result = invocation_strategy.invoke(core_input, parameters)
    if type(result) is not signature.output_type:
        raise OrganelleContractError(
            code="contract.operation_output_type_mismatch",
            message=f"{spec.operation_id} returned {type(result).__name__}",
        )
    return _revalidate_core_output(
        spec,
        cast(type[BaseModel], signature.output_type),
        result,
        output_data_contracts=output_data_contracts,
        suggestion_validator=suggestion_validator,
    )


def _validate_data_modality(
    spec: OperationSpec,
    core_input: CoreObject,
    data_contracts: tuple[DataContract, ...],
) -> None:
    if not data_contracts:
        return
    data = cast(OrganelleData, core_input)
    contract = next(
        (candidate for candidate in data_contracts if candidate.modality == data.modality),
        None,
    )
    if contract is None:
        raise OrganelleInputError(
            code="input.unsupported_operation_modality",
            message=f"{spec.operation_id} does not accept data modality {data.modality!r}",
            details={
                "operation_id": spec.operation_id,
                "provided_modality": data.modality,
                "accepted_modalities": [candidate.modality for candidate in data_contracts],
            },
        )
    contract.validate(data)


def _revalidate_core_input(
    spec: OperationSpec,
    model_type: type[BaseModel],
    value: CoreObject,
) -> CoreObject:
    try:
        return cast(CoreObject, model_type.model_validate(value))
    except ValidationError as error:
        raise OrganelleInputError(
            code="input.invalid_core_object",
            message="core input is invalid",
            details={
                "operation_id": spec.operation_id,
                "expected_core_kind": spec.input_kind.value,
                "validation_errors": validation_summaries(error),
            },
        ) from error


def _revalidate_core_output(
    spec: OperationSpec,
    model_type: type[BaseModel],
    value: CoreObject,
    *,
    output_data_contracts: tuple[DataContract, ...] = (),
    suggestion_validator: SuggestionValidator | None = None,
) -> CoreObject:
    """Re-validate a returned core object against its declared contract.

    For DATA output the returned modality must be declared (I1b) and, when a
    resolvable contract is supplied, its payload is validated (I1c). For RESULT
    output the optional ``suggestion_validator`` enforces that every suggestion
    on the returned :class:`OrganelleResult` is followable by an agent: it must
    exist in the invoking registry, its ``input_kind`` must be a member of
    ``{NONE, emitting output_kind}`` (NONE is admitted because the agent calls a
    root independently rather than chaining the Result into it), and its
    ``parameter_changes`` must validate against the target's parameter model.

    Two limitations of the suggestion check, stated verbatim because they are
    counterintuitive:

    - It fires only when the code path executes. The assembly success path that
      would carry a suggestion needs a real backend, so CI does not exercise it;
      a suggestion can therefore ship unchallenged until a real invocation runs.
    - It resolves against the invoking registry, so a local registry or a mock
      registration masks the defect — the suggestion resolves there even if the
      released catalog could not satisfy it.

    The static AST scan is therefore the STRONGER mechanism for this hole class,
    which is the opposite of what an implementer would assume.
    """
    try:
        validated = cast(CoreObject, model_type.model_validate(value))
    except ValidationError as error:
        raise OrganelleContractError(
            code="contract.invalid_operation_output",
            message=f"{spec.operation_id} returned an invalid {spec.output_kind.value}",
            details={
                "operation_id": spec.operation_id,
                "expected_core_kind": spec.output_kind.value,
                "validation_errors": validation_summaries(error),
            },
        ) from error
    if spec.output_kind is CoreKind.DATA:
        data = cast(OrganelleData, validated)
        if data.modality not in spec.output_modalities:
            raise OrganelleContractError(
                code="contract.operation_output_modality_mismatch",
                message=(
                    f"{spec.operation_id} returned data modality {data.modality!r}, "
                    "which it does not declare"
                ),
                details={
                    "operation_id": spec.operation_id,
                    "returned_modality": data.modality,
                    "declared_modalities": list(spec.output_modalities),
                },
            )
        for contract in output_data_contracts:
            if contract.modality == data.modality:
                # The modality validators were written for the input side and raise
                # OrganelleInputError. Applied to an output they are diagnosing the
                # package violating its own declaration, not caller-supplied data, so
                # the error is re-raised in the contract class. Leaving it as input.*
                # would blame the caller for a fault the operation committed.
                try:
                    contract.validate(data)
                except OrganelleInputError as error:
                    raise OrganelleContractError(
                        code="contract.operation_output_payload_invalid",
                        message=(
                            f"{spec.operation_id} returned a {data.modality!r} payload "
                            "its declared data contract rejects"
                        ),
                        details={
                            "operation_id": spec.operation_id,
                            "returned_modality": data.modality,
                            "contract_error_code": error.code,
                            "contract_error_message": error.message,
                        },
                    ) from error
    if spec.output_kind is CoreKind.RESULT and suggestion_validator is not None:
        suggestion_validator(cast(OrganelleResult, validated))
    return validated


def _validate_parameters(
    signature: OperationSignature,
    parameters: Mapping[str, object],
) -> BaseModel:
    try:
        return signature.parameter_model.model_validate(dict(parameters))
    except (TypeError, ValidationError, ValueError) as error:
        raise _parameter_error(error) from error


def _revalidate_parameter_instance(
    spec: OperationSpec,
    signature: OperationSignature,
    parameters: BaseModel,
) -> BaseModel:
    if type(parameters) is not signature.parameter_model:
        raise OrganelleContractError(
            code="contract.parameter_model_mismatch",
            message=f"{spec.operation_id} received the wrong parameter model",
        )
    try:
        declared_fields = set(signature.parameter_model.model_fields)
        unexpected_fields = _public_parameter_state_keys(parameters) - declared_fields
        if unexpected_fields:
            rendered = ", ".join(sorted(repr(name) for name in unexpected_fields))
            raise ValueError(f"unexpected parameter fields: {rendered}")
        values = {
            name: getattr(parameters, name) for name in signature.parameter_model.model_fields
        }
    except AttributeError as error:
        raise _parameter_error(ValueError(str(error))) from error
    except (TypeError, ValueError) as error:
        raise _parameter_error(error) from error
    return _validate_parameters(signature, values)


def _public_parameter_state_keys(parameters: BaseModel) -> set[object]:
    raw_state = cast(object, parameters.__dict__)
    extra_state = cast(object, parameters.__pydantic_extra__)
    fields_set = cast(object, parameters.__pydantic_fields_set__)
    if not isinstance(raw_state, Mapping):
        raise TypeError("parameter model state must be a mapping")
    if extra_state is not None and not isinstance(extra_state, Mapping):
        raise TypeError("parameter model extra state must be a mapping")
    if not isinstance(fields_set, (set, frozenset)):
        raise TypeError("parameter model fields set must be a set")

    keys: set[object] = set(cast(Mapping[object, object], raw_state))
    if extra_state is not None:
        keys.update(cast(Mapping[object, object], extra_state))
    keys.update(cast(set[object] | frozenset[object], fields_set))
    return {key for key in keys if not isinstance(key, str) or not key.startswith("_")}


def _typed_parameter_values(parameters: BaseModel) -> dict[str, object]:
    """Extract validated fields without serializing their Python values."""
    return {name: getattr(parameters, name) for name in type(parameters).model_fields}


def _parameter_error(error: TypeError | ValidationError | ValueError) -> OrganelleParameterError:
    if isinstance(error, ValidationError):
        summaries = validation_summaries(error)
    else:
        summaries = [validation_summary((), str(error), "signature_binding")]
    return OrganelleParameterError(
        code="parameter.invalid_operation_parameters",
        message="operation parameters are invalid",
        details={"validation_errors": summaries},
    )


registry = _ReleaseOperationRegistry()
