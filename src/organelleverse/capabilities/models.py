"""The capability bundle envelope around one OperationSpec contract.

``OperationSpec`` remains the only executable contract: ``CapabilityBundle``
does not introduce a parallel schema for it. It only adds the
``[capability]`` envelope (a stable capability id, a bundle version, and an
implementation discriminator) that a ``capability.toml`` needs on top of a
field-for-field ``[contract]`` = ``OperationSpec`` serialization.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, cast

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)

from organelleverse.operations.spec import (
    AgentSurface,
    DependencyKind,
    OperationSpec,
    PluginOperationSpec,
    StrictSpecModel,
)

_IDENTIFIER_PATTERN = r"^[a-z][a-z0-9_]*$"
_CAPABILITY_ID_PATTERN = r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$"
_DATA_CONTRACT_FACTORY_LOCATOR_PATTERN = r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"


def _reject_blank_string(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank or whitespace-only")
    return value


NonBlankStr = Annotated[str, AfterValidator(_reject_blank_string)]
"""A string identifier/reference that rejects both "" and whitespace-only.

For fields already constrained by a charset-anchored ``pattern=`` (e.g.
``_IDENTIFIER_PATTERN``, ``_CAPABILITY_ID_PATTERN``), blank is already
unreachable and this adds nothing; it exists for the fields that are
otherwise free-form text (a step id, a cross-reference, a fixture's
recorded expectation).
"""

StepId = Annotated[str, Field(pattern=_IDENTIFIER_PATTERN)]
"""A composite step identifier: exactly ``_IDENTIFIER_PATTERN`` - no
whitespace (leading, trailing, or internal), no path separators, no dots,
no punctuation beyond ``_``. The one type for ``CompositeStep.id``,
``StepOutputReference.step_id``, and every ``depends_on`` element, so a
step can never be *declared* under looser rules than it can be
*referenced* under.
"""


_JSON_POINTER_PATTERN = r"^(/(?:[^~/]|~[01])*)+$"
"""One or more RFC 6901 ``"/" reference-token`` segments: each segment is
any run of characters that are neither ``/`` nor ``~``, or a properly
escaped ``~0`` (literal ``~``)/``~1`` (literal ``/``). The outer ``+``
(not ``*``) requires at least one segment, so this never matches the empty
string or a string lacking a leading ``/`` - dot-notation, ``..``, bare
relative paths, and unescaped/mis-escaped ``~`` are all excluded by
construction, not by a separate check.

A native ``Field(pattern=...)`` constraint rather than an ``AfterValidator``
on purpose: pydantic enforces a ``pattern`` at runtime *and* reflects it
verbatim into ``model_json_schema()`` as the field's own JSON Schema
``"pattern"`` - an ``AfterValidator``'s Python function is invisible to
schema generation, so a schema consumer that never runs Python (an
admission tool validating a ``capability.toml`` against the frozen schema
alone) would see no constraint at all.
"""

JsonPointer = Annotated[str, Field(pattern=_JSON_POINTER_PATTERN)]
"""An RFC 6901 JSON Pointer string, e.g. ``/metrics/gc_content``."""


def _assert_finite_json(value: object, *, field_name: str) -> None:
    """Reject NaN/Infinity/-Infinity anywhere within a JSON-like value tree.

    A per-field ``Field(allow_inf_nan=False)`` only constrains a field whose
    own schema is a bare ``float`` - applied to a ``JsonValue`` field (a
    recursive scalar/list/dict union), pydantic-core's finite-float check
    still runs against the *whole* value and crashes with a bare
    ``TypeError`` the moment that value is actually a container, not a
    number. Non-finite floats nested inside a composite JsonValue tree must
    therefore be caught by hand, recursively, here instead.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must not contain NaN or Infinity")
    if isinstance(value, dict):
        mapping_value = cast("dict[object, object]", value)
        for item in mapping_value.values():
            _assert_finite_json(item, field_name=field_name)
    elif isinstance(value, list):
        sequence_value = cast("list[object]", value)
        for item in sequence_value:
            _assert_finite_json(item, field_name=field_name)


class ImplementationKind(StrEnum):
    """How a capability's callable is provided."""

    NATIVE = "native"
    EXTERNAL = "external"
    COMPOSITE = "composite"


class EquivalenceMethod(StrEnum):
    """How a fixture's produced result is compared against its recorded expectation."""

    EXACT = "exact"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    SET_MEMBERSHIP = "set_membership"


class CapabilityDeclaration(StrictSpecModel):
    """The ``[capability]`` section: identity, version, and implementation kind."""

    id: str = Field(min_length=1)
    bundle_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    implementation: ImplementationKind


class ProbeSpec(StrictSpecModel):
    """One ``[[probe]]`` entry: an executable dependency's interface surface.

    Probes never gate on a version string (see spec §5.2): ``version_argv``/
    ``version_capture`` exist only to record human-readable evidence in
    provenance. Admission and capability both key off ``requires``, the
    declared CLI interface surface, never off a parsed version number.
    """

    dependency: str = Field(min_length=1)
    help_argv: tuple[str, ...] = Field(min_length=1)
    requires: tuple[str, ...] = ()
    version_argv: tuple[str, ...] = ()
    version_capture: str = ""

    @field_validator("help_argv", "requires", "version_argv")
    @classmethod
    def validate_no_blank_elements(cls, values: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError(f"{info.field_name} entries must not be blank")
        return values


class DataContractReference(StrictSpecModel):
    """One core-provided DATA modality validator available to a Bundle binding.

    The reference deliberately belongs to the capability envelope rather than
    ``OperationSpec``: a release operation's serialized scientific contract
    stays byte-identical while its Bundle can name the provider required to
    validate a suite-specific modality. The locator is resolved only by the
    core binding/verification path, never by discovery or admission.
    """

    modality: str = Field(pattern=_IDENTIFIER_PATTERN)
    factory_locator: str = Field(pattern=_DATA_CONTRACT_FACTORY_LOCATOR_PATTERN)


class FixtureSpec(StrictSpecModel):
    """One ``[[fixture]]`` entry: a frozen-regression, reference-equivalence,
    or adversarial fail-closed case.

    A normal fixture requires exactly one of ``expect`` (frozen regression)
    or ``reference`` (live reference capability).  An adversarial fixture
    expects the capability to fail with a specific error code and must not
    declare an expectation.
    """

    case: str = Field(min_length=1)
    input: dict[str, JsonValue] = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    expect: NonBlankStr | None = None
    reference: NonBlankStr | None = None
    equivalence: EquivalenceMethod = EquivalenceMethod.EXACT
    tolerance: float | None = None
    adversarial: bool = False
    expect_failure_code: str | None = None

    @field_validator("input", "parameters")
    @classmethod
    def validate_finite(cls, value: dict[str, JsonValue], info: ValidationInfo) -> dict[str, JsonValue]:
        _assert_finite_json(value, field_name=info.field_name or "value")
        return value

    @model_validator(mode="after")
    def validate_fixture(self) -> FixtureSpec:
        if self.adversarial:
            if not self.expect_failure_code or not self.expect_failure_code.strip():
                raise ValueError("adversarial fixture requires expect_failure_code")
            if self.expect is not None or self.reference is not None:
                raise ValueError("adversarial fixture must not declare expect or reference")
            return self
        if (self.expect is None) == (self.reference is None):
            raise ValueError("fixture requires exactly one of 'expect' or 'reference'")
        if self.equivalence is EquivalenceMethod.NUMERIC_TOLERANCE:
            if (
                self.tolerance is None
                or not math.isfinite(self.tolerance)
                or self.tolerance <= 0
            ):
                raise ValueError(
                    "numeric_tolerance equivalence requires a finite, positive tolerance"
                )
        elif self.tolerance is not None:
            raise ValueError(
                f"tolerance is only meaningful for numeric_tolerance equivalence, not "
                f"{self.equivalence.value!r}"
            )
        return self


class ConditionOperator(StrEnum):
    """A numeric comparison a step condition applies to a referenced value.

    Matches exactly the operator set the pre-existing declarative
    orchestrator (``organelleverse_orchestrate.py``) already evaluates - no
    ``exists``: v1 cannot give "a value exists" precise, implementation-free
    semantics (does a missing step output mean "false" or "unevaluable"?)
    without building the execution engine Spec 2 owns, so it is dropped
    rather than kept underspecified.
    """

    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    EQ = "eq"


class StepOutputReference(StrictSpecModel):
    """A reference to a value produced by an upstream step's declared output slot.

    ``slot`` must equal the managed output slot the referenced step actually
    declares (see ``CompositeSpec.validate_composite``'s slot-matching
    check) - a reference naming a step with no output, or the wrong slot
    name, is rejected at parse time rather than left to fail only when an
    executor tries to resolve it. ``value_path`` is an RFC 6901 JSON Pointer
    into that slot's recorded result (e.g. ``/metrics/gc_content``);
    omitted (``None``), the reference means "the whole slot" - never the
    RFC's own empty-string spelling of that, and never dot-notation.
    """

    kind: Literal["step_output"] = "step_output"
    step_id: StepId
    slot: str = Field(min_length=1, pattern=_IDENTIFIER_PATTERN)
    value_path: JsonPointer | None = None


class ConditionReference(StepOutputReference):
    """A condition's own reference: like ``StepOutputReference``, but
    ``value_path`` is mandatory.

    A condition compares one addressed numeric value, never an entire
    managed-slot result - "the whole slot" (``value_path`` omitted) is
    meaningful for an ordinary data-flow input mapping but not here. This
    is a genuine field-presence difference between two distinct schema
    types (each with its own ``$defs`` entry and its own ``required``
    list), not a runtime-only rule bolted onto the shared base - so it
    stays visible in ``CapabilityBundle.model_json_schema()``, not just in
    ``model_validate()``.
    """

    # Narrowing an inherited field from Optional to required is a supported,
    # common Pydantic subclassing pattern; pyright's general soundness rule
    # for mutable-attribute overrides doesn't know that - and both models
    # are frozen=True, so the unsound-mutation scenario it's guarding
    # against can't actually happen.
    value_path: JsonPointer  # pyright: ignore[reportIncompatibleVariableOverride, reportGeneralTypeIssues]


class CompositeInputReference(StrictSpecModel):
    """A reference to one of the composite bundle's own declared inputs."""

    kind: Literal["composite_input"] = "composite_input"
    input_name: NonBlankStr


class ConstantValue(StrictSpecModel):
    """A literal, inline JSON value for one mapped input."""

    kind: Literal["constant"] = "constant"
    value: JsonValue

    @field_validator("value")
    @classmethod
    def validate_value_is_finite(cls, value: JsonValue) -> JsonValue:
        _assert_finite_json(value, field_name="constant value")
        return value


MappedInput = Annotated[
    ConstantValue | CompositeInputReference | StepOutputReference,
    Field(discriminator="kind"),
]
"""One step invocation input: a constant, a composite input, or an upstream
step's output - never an arbitrary Python value or expression."""


class ManagedOutputBinding(StrictSpecModel):
    """A step's output, bound to a managed slot name - never a filesystem path.

    ``slot`` is a stable, logical identifier a later step can reference via
    ``StepOutputReference``; the actual managed-run path is resolved by L6 at
    execution time (Spec 2), never authored here. There is deliberately no
    ``output_dir``/``path``/``write_path`` field of any kind on this model -
    ``extra="forbid"`` rejects one outright rather than needing an explicit
    denylist.
    """

    slot: str = Field(min_length=1, pattern=_IDENTIFIER_PATTERN)


class CapabilityInvocation(StrictSpecModel):
    """One call to a real capability: its id, its mapped inputs, its output slot.

    Reused for a step's own invocation, its condition's ``if_true``/
    ``if_false`` branches, and its ``fallback`` - all four are "call this
    capability with these inputs", never a bespoke shape per position.
    """

    capability_id: str = Field(pattern=_CAPABILITY_ID_PATTERN)
    inputs: dict[str, MappedInput] = Field(default_factory=dict)
    output: ManagedOutputBinding | None = None


class StepConditionBranch(StrictSpecModel):
    """A guard that picks exactly one of two invocations based on a comparison.

    v1 only declares and validates this *syntax* (per admission design
    §5.3): comparing ``reference`` against ``value`` with ``operator`` and
    dispatching to ``if_true``/``if_false`` is Spec 2's execution semantics,
    not implemented here. ``reference`` is a ``ConditionReference`` - its
    ``value_path`` is structurally required (a missing one is a schema
    validation error, not a secondary runtime check), since a condition
    compares one numeric value, never an entire managed-slot result (an
    ``OrganelleResult``/``OrganelleData``/``OrganelleGenome``).
    """

    reference: ConditionReference
    operator: ConditionOperator
    value: float = Field(allow_inf_nan=False)
    if_true: CapabilityInvocation
    if_false: CapabilityInvocation


class CompositeStep(StrictSpecModel):
    """One step of a composite bundle's declarative step graph.

    A step is exactly one of two shapes: a direct ``invocation`` (optionally
    with a ``fallback`` invocation tried if the primary one fails), or a
    ``condition`` branch-point with no invocation of its own. ``depends_on``
    declares pure ordering with no data dependency; real data dependencies
    (an ``invocation``/``fallback`` input referencing another step's output,
    or a ``condition``'s reference) are automatically real graph edges too -
    restating them in ``depends_on`` is never required.
    """

    id: StepId
    depends_on: tuple[StepId, ...] = ()
    invocation: CapabilityInvocation | None = None
    condition: StepConditionBranch | None = None
    fallback: CapabilityInvocation | None = None

    @model_validator(mode="after")
    def validate_step(self) -> CompositeStep:
        if (self.invocation is None) == (self.condition is None):
            raise ValueError(
                f"step {self.id!r} requires exactly one of 'invocation' or 'condition'"
            )
        if self.fallback is not None and self.invocation is None:
            raise ValueError(
                f"step {self.id!r} 'fallback' is only meaningful alongside 'invocation', "
                "not a 'condition' branch"
            )
        return self


class CompositeInputDeclaration(StrictSpecModel):
    """One named input the whole composite bundle accepts."""

    name: str = Field(min_length=1, pattern=_IDENTIFIER_PATTERN)


class CompositeSpec(StrictSpecModel):
    """A composite bundle's full declarative step graph.

    Per admission design §5.3: v1 may parse and statically validate this
    (step id uniqueness, every reference resolves *and* names a slot the
    target step actually declares, the union of depends_on/data-flow/
    condition edges forms a DAG whose *declared step order* is already a
    valid topological order) but never executes it. A composite bundle with
    zero steps is rejected outright - it declares nothing to describe.
    """

    inputs: tuple[CompositeInputDeclaration, ...] = ()
    steps: tuple[CompositeStep, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_composite(self) -> CompositeSpec:
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            duplicates = sorted({step_id for step_id in step_ids if step_ids.count(step_id) > 1})
            raise ValueError(f"composite step ids must be unique: {duplicates}")

        input_names = [declared.name for declared in self.inputs]
        if len(input_names) != len(set(input_names)):
            duplicates = sorted({name for name in input_names if input_names.count(name) > 1})
            raise ValueError(f"composite input declaration names must be unique: {duplicates}")
        declared_input_names = set(input_names)

        steps_by_id = {step.id: step for step in self.steps}
        for step in self.steps:
            _assert_step_output_slots_are_compatible(step)

        edges_by_step: dict[str, set[str]] = {step.id: set(step.depends_on) for step in self.steps}
        for step in self.steps:
            for dependency in step.depends_on:
                if dependency == step.id:
                    raise ValueError(f"step {step.id!r} cannot depend_on itself")
                if dependency not in steps_by_id:
                    raise ValueError(f"step {step.id!r} depends on unknown step id {dependency!r}")
            if step.invocation is not None:
                _collect_invocation_edges(
                    step.id, "invocation", step.invocation, steps_by_id, declared_input_names, edges_by_step[step.id]
                )
            if step.fallback is not None:
                _collect_invocation_edges(
                    step.id, "fallback", step.fallback, steps_by_id, declared_input_names, edges_by_step[step.id]
                )
            if step.condition is not None:
                _validate_step_output_reference(
                    step.id, "condition reference", step.condition.reference, steps_by_id, edges_by_step[step.id]
                )
                _collect_invocation_edges(
                    step.id,
                    "condition if_true",
                    step.condition.if_true,
                    steps_by_id,
                    declared_input_names,
                    edges_by_step[step.id],
                )
                _collect_invocation_edges(
                    step.id,
                    "condition if_false",
                    step.condition.if_false,
                    steps_by_id,
                    declared_input_names,
                    edges_by_step[step.id],
                )

        _assert_composite_is_acyclic(edges_by_step)
        _assert_declared_order_is_topological(step_ids, edges_by_step)
        return self


def _assert_step_output_slots_are_compatible(step: CompositeStep) -> None:
    """A step's two possible outcomes must agree on where their output lands.

    Only one of a step's declared invocations actually runs - the primary
    invocation or its fallback; the condition's if_true or if_false - so a
    downstream ``StepOutputReference`` naming this step needs ONE stable
    slot regardless of which one that turns out to be. Both sides must
    therefore declare the exact same slot, or both must declare no output
    at all; this is checked unconditionally, whether or not anything
    currently references this step downstream, since a mismatch is already
    a genuine authoring error either way.
    """
    if step.invocation is not None and step.fallback is not None:
        _assert_two_outputs_are_compatible(
            step.id, "invocation", step.invocation.output, "fallback", step.fallback.output
        )
    if step.condition is not None:
        _assert_two_outputs_are_compatible(
            step.id,
            "condition if_true",
            step.condition.if_true.output,
            "condition if_false",
            step.condition.if_false.output,
        )


def _assert_two_outputs_are_compatible(
    step_id: str,
    first_label: str,
    first: ManagedOutputBinding | None,
    second_label: str,
    second: ManagedOutputBinding | None,
) -> None:
    if (first is None) != (second is None):
        raise ValueError(
            f"step {step_id!r} {first_label} and {second_label} must either both declare "
            "an output slot or both omit one"
        )
    if first is not None and second is not None and first.slot != second.slot:
        raise ValueError(
            f"step {step_id!r} {first_label} declares output slot {first.slot!r} but "
            f"{second_label} declares {second.slot!r} - both must agree, since only one "
            "of them actually runs"
        )


def _effective_output_slot(step: CompositeStep) -> str | None:
    """The slot a downstream StepOutputReference can name for *step*, if any.

    Compatibility between a step's two possible outcomes is asserted by
    ``_assert_step_output_slots_are_compatible`` before this ever runs, so
    reading either side of an invocation/fallback or if_true/if_false pair
    is equivalent - this reads whichever side is structurally present.
    """
    if step.invocation is not None:
        return step.invocation.output.slot if step.invocation.output is not None else None
    assert step.condition is not None, "CompositeStep.validate_step guarantees invocation xor condition"
    if_true_output = step.condition.if_true.output
    return if_true_output.slot if if_true_output is not None else None


def _validate_step_output_reference(
    referencing_step_id: str,
    role: str,
    reference: StepOutputReference,
    steps_by_id: dict[str, CompositeStep],
    edges: set[str],
) -> None:
    """Validate one StepOutputReference and, if valid, add its data-flow edge.

    Shared by every place a ``StepOutputReference`` can appear - a mapped
    input and a condition's own ``reference`` - so unknown-step, self-
    reference, and slot-matching are enforced identically everywhere.
    """
    if reference.step_id == referencing_step_id:
        raise ValueError(f"step {referencing_step_id!r} {role} cannot reference its own output")
    target_step = steps_by_id.get(reference.step_id)
    if target_step is None:
        raise ValueError(
            f"step {referencing_step_id!r} {role} references unknown step id "
            f"{reference.step_id!r}"
        )
    effective_slot = _effective_output_slot(target_step)
    if effective_slot is None:
        raise ValueError(
            f"step {referencing_step_id!r} {role} references step {reference.step_id!r}, "
            "which declares no output slot"
        )
    if effective_slot != reference.slot:
        raise ValueError(
            f"step {referencing_step_id!r} {role} references slot {reference.slot!r} on "
            f"step {reference.step_id!r}, but that step's declared output slot is "
            f"{effective_slot!r}"
        )
    edges.add(reference.step_id)


def _collect_invocation_edges(
    step_id: str,
    role: str,
    invocation: CapabilityInvocation,
    steps_by_id: dict[str, CompositeStep],
    declared_input_names: set[str],
    edges: set[str],
) -> None:
    """Validate one invocation's mapped inputs and add any real data-flow edges."""
    for input_name, mapped in invocation.inputs.items():
        if isinstance(mapped, CompositeInputReference):
            if mapped.input_name not in declared_input_names:
                raise ValueError(
                    f"step {step_id!r} {role} input {input_name!r} references undeclared "
                    f"composite input {mapped.input_name!r}"
                )
        elif isinstance(mapped, StepOutputReference):
            _validate_step_output_reference(
                step_id, f"{role} input {input_name!r}", mapped, steps_by_id, edges
            )


def _assert_composite_is_acyclic(edges_by_step: dict[str, set[str]]) -> None:
    unvisited, in_progress, done = 0, 1, 2
    status = dict.fromkeys(edges_by_step, unvisited)

    def visit(step_id: str, path: tuple[str, ...]) -> None:
        status[step_id] = in_progress
        for dependency in edges_by_step[step_id]:
            if status[dependency] == in_progress:
                cycle = (*path[path.index(dependency) :], dependency)
                raise ValueError(f"composite steps form a cycle: {' -> '.join(cycle)}")
            if status[dependency] == unvisited:
                visit(dependency, (*path, dependency))
        status[step_id] = done

    for step_id in edges_by_step:
        if status[step_id] == unvisited:
            visit(step_id, (step_id,))


def _assert_declared_order_is_topological(
    step_ids: list[str], edges_by_step: dict[str, set[str]]
) -> None:
    """Every edge must point to a step declared earlier in ``step_ids``.

    Distinct from acyclicity: a graph can be perfectly acyclic yet still be
    *listed* in an order that forward-references a not-yet-declared step
    (e.g. step B, listed first, referencing step A, listed second). v1
    requires the declared order to already be a valid topological sort, so
    no reader or future executor needs a separate reordering pass.
    """
    position = {step_id: index for index, step_id in enumerate(step_ids)}
    for step_id, dependencies in edges_by_step.items():
        for dependency in dependencies:
            if position[dependency] >= position[step_id]:
                raise ValueError(
                    f"step {step_id!r} references {dependency!r}, which is not declared "
                    "earlier in the composite's step list"
                )


#: Schema versions this installation parses. The parser dispatches on the
#: top-level ``schema`` field before any field-level validation; anything
#: outside this set fails with ``capability.schema_unsupported``.
SUPPORTED_BUNDLE_SCHEMA_VERSIONS = frozenset(
    {
        "organelleverse.capability.v1",
        "organelleverse.capability.v2",
    }
)


class PluginMetadata(StrictSpecModel):
    """Authorship and discovery presentation for one v2 plugin bundle."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    author: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    icon: str = ""
    category: Literal["mito", "chloro", "general", "image"] = "general"
    badge: Literal["job", "read"] = "read"

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: tuple[str, ...]) -> tuple[str, ...]:
        if any(not tag.strip() for tag in tags):
            raise ValueError("plugin tags must not be blank")
        return tags


class AgentMetadata(StrictSpecModel):
    """Agent-facing task description for one v2 plugin bundle."""

    task_description: str = Field(min_length=1)
    examples: tuple[str, ...] = ()
    prompt: str = ""
    agent_surface: AgentSurface = AgentSurface.GRANULAR

    @field_validator("examples")
    @classmethod
    def validate_examples(cls, examples: tuple[str, ...]) -> tuple[str, ...]:
        if any(not example.strip() for example in examples):
            raise ValueError("agent examples must not be blank")
        return examples


class GuiMetadata(StrictSpecModel):
    """Optional custom desktop page entry for one v2 plugin bundle."""

    page: str | None = None


class CapabilityBundle(StrictSpecModel):
    """One capability.toml, fully parsed: envelope plus its OperationSpec contract."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        populate_by_name=True,
    )

    schema_version: str = Field(alias="schema", pattern=r"^organelleverse\.capability\.v[0-9]+$")
    capability: CapabilityDeclaration
    contract: OperationSpec
    probes: tuple[ProbeSpec, ...] = Field(default=(), alias="probe")
    data_contracts: tuple[DataContractReference, ...] = Field(default=(), alias="data_contract")
    fixtures: tuple[FixtureSpec, ...] = Field(default=(), alias="fixture")
    composite: CompositeSpec | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> CapabilityBundle:
        if self.capability.id != self.contract.operation_id:
            raise ValueError(
                f"capability.id {self.capability.id!r} must equal "
                f"contract.operation_id {self.contract.operation_id!r}"
            )
        modalities = tuple(reference.modality for reference in self.data_contracts)
        if len(set(modalities)) != len(modalities):
            raise ValueError("duplicate data contract modality")
        requires_locator = self.capability.implementation in {
            ImplementationKind.NATIVE,
            ImplementationKind.EXTERNAL,
        }
        has_locator = self.contract.callable_locator is not None
        if requires_locator and not has_locator:
            raise ValueError(
                f"{self.capability.implementation.value} implementation requires a callable_locator"
            )
        if not requires_locator and has_locator:
            raise ValueError("composite implementation forbids a callable_locator")
        if self.capability.implementation is ImplementationKind.COMPOSITE:
            if self.composite is None:
                raise ValueError("composite implementation requires a [composite] step declaration")
        elif self.composite is not None:
            raise ValueError("a [composite] step declaration is only meaningful for composite")
        self._validate_probes()
        self._validate_fixture_cases_are_unique()
        return self

    def _validate_probes(self) -> None:
        implementation = self.capability.implementation
        if implementation is ImplementationKind.NATIVE and self.probes:
            raise ValueError("native implementation forbids any probe")
        if implementation is ImplementationKind.COMPOSITE and self.probes:
            raise ValueError("composite implementation forbids any probe")
        if implementation is ImplementationKind.EXTERNAL and not self.probes:
            raise ValueError("external implementation requires at least one probe")
        probe_dependency_names = [probe.dependency for probe in self.probes]
        if len(probe_dependency_names) != len(set(probe_dependency_names)):
            duplicates = sorted(
                {name for name in probe_dependency_names if probe_dependency_names.count(name) > 1}
            )
            raise ValueError(f"duplicate probed dependencies: {duplicates}")
        probed_dependencies = set(probe_dependency_names)
        executable_names = {
            dependency.name
            for dependency in self.contract.dependencies
            if dependency.kind is DependencyKind.EXECUTABLE
        }
        missing = sorted(executable_names - probed_dependencies)
        if missing:
            raise ValueError(f"executable dependencies without a probe: {missing}")
        undeclared = sorted(probed_dependencies - executable_names)
        if undeclared:
            raise ValueError(f"probes reference dependencies not declared as executable: {undeclared}")

    def _validate_fixture_cases_are_unique(self) -> None:
        cases = [fixture.case for fixture in self.fixtures]
        if len(cases) != len(set(cases)):
            duplicates = sorted({case for case in cases if cases.count(case) > 1})
            raise ValueError(f"fixture cases must be unique within a bundle: {duplicates}")


class PluginCapabilityBundle(CapabilityBundle):
    """One v2 plugin ``capability.toml``: the v1 bundle plus plugin metadata.

    Inherits every v1 rule (id equality, probe/implementation pairing,
    fixture case uniqueness, composite pairing). The v2 additions are the
    ``plugin``/``agent``/``gui`` presentation sections and a contract that
    permits named outputs, an optimization declaration, presentation
    schemas, and the ``plugin_protocol`` argument mode.
    """

    # Narrowing the parent's str field to the single v2 literal is the whole
    # point of the subclass: v2 documents parse only as this exact version.
    schema_version: Literal["organelleverse.capability.v2"] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        alias="schema"
    )
    contract: PluginOperationSpec  # pyright: ignore[reportIncompatibleVariableOverride] - v2 narrows the contract to the plugin operation model
    plugin: PluginMetadata
    agent: AgentMetadata
    gui: GuiMetadata = GuiMetadata()


#: What ``parse_capability_bundle`` returns after schema dispatch: the exact
#: v1 model for v1 documents, the plugin model for v2. Lifecycle code uses
#: the shared v1 surface and never branches on the concrete type.
BundleDocument = CapabilityBundle | PluginCapabilityBundle


__all__ = [
    "SUPPORTED_BUNDLE_SCHEMA_VERSIONS",
    "AgentMetadata",
    "BundleDocument",
    "CapabilityBundle",
    "CapabilityDeclaration",
    "CapabilityInvocation",
    "CompositeInputDeclaration",
    "CompositeInputReference",
    "CompositeSpec",
    "CompositeStep",
    "ConditionOperator",
    "ConditionReference",
    "ConstantValue",
    "DataContractReference",
    "EquivalenceMethod",
    "FixtureSpec",
    "GuiMetadata",
    "ImplementationKind",
    "ManagedOutputBinding",
    "MappedInput",
    "PluginCapabilityBundle",
    "PluginMetadata",
    "ProbeSpec",
    "StepConditionBranch",
    "StepOutputReference",
]
