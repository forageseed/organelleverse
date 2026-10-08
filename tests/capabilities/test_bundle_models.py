"""CapabilityBundle: the discriminated native/external/composite bundle schema.

``OperationSpec`` remains the only executable contract (``[contract]`` in a
real ``capability.toml`` is a field-for-field serialization of it, not a
parallel structure); ``CapabilityBundle`` only adds the ``[capability]``
envelope (id, bundle_version, implementation) around one.
"""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from organelleverse.capabilities.models import (
    CapabilityBundle,
    CapabilityInvocation,
    CompositeStep,
    ConstantValue,
    ImplementationKind,
    StepOutputReference,
)
from organelleverse.operations import ExecutionMode
from organelleverse.operations.spec import DependencyKind


def _contract(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "operation_id": "demo.probe",
        "contract_version": "1.0",
        "title": "Demo probe capability",
        "description": "Test fixture contract used only to probe bundle model validation.",
        "keywords": ("demo", "fixture", "test"),
        "execution_mode": ExecutionMode.INLINE,
        "stage": "analyze",
        "input_kind": "genome",
        "output_kind": "result",
        "callable_locator": "demo.api:probe",
    }
    base.update(overrides)
    return base


_DEFAULT_EXTERNAL_PROBE: dict[str, object] = {
    "dependency": "demo_tool",
    "help_argv": ["--help"],
}

_DEFAULT_COMPOSITE_STEP: dict[str, object] = {
    "id": "step_one",
    "invocation": {"capability_id": "demo.step_one"},
}


_UNSET = object()  # sentinel: "no override given" vs. an explicit None (omit composite)


def minimal_bundle(
    *,
    implementation: str = "native",
    callable_locator: str | None = "demo.api:probe",
    capability_id: str = "demo.probe",
    probes: tuple[dict[str, object], ...] | None = None,
    fixtures: tuple[dict[str, object], ...] = (),
    composite: object = _UNSET,
) -> dict[str, Any]:
    using_default_probe = probes is None and implementation == "external"
    if probes is None:
        probes = (_DEFAULT_EXTERNAL_PROBE,) if implementation == "external" else ()
    if composite is _UNSET:
        composite = {"steps": [_DEFAULT_COMPOSITE_STEP]} if implementation == "composite" else None
    # The probed-dependency set must equal the executable-dependency set:
    # when the default external probe (covering "demo_tool") is in play,
    # the contract must declare that same dependency as executable.
    dependencies = [{"kind": "executable", "name": "demo_tool"}] if using_default_probe else []
    payload: dict[str, Any] = {
        "schema": "organelleverse.capability.v1",
        "capability": {
            "id": capability_id,
            "bundle_version": "1.0.0",
            "implementation": implementation,
        },
        "contract": _contract(
            operation_id=capability_id, callable_locator=callable_locator, dependencies=dependencies
        ),
        "probe": list(probes),
        "fixture": list(fixtures),
    }
    if composite is not None:
        payload["composite"] = composite
    return payload


def test_native_requires_locator_and_composite_forbids_it() -> None:
    native = minimal_bundle(implementation="native", callable_locator=None)
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(native)

    composite = minimal_bundle(
        implementation="composite",
        callable_locator="pkg.module:run",
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(composite)


def test_external_also_requires_a_locator() -> None:
    external = minimal_bundle(implementation="external", callable_locator=None)
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(external)


def test_composite_with_no_locator_is_valid() -> None:
    composite = minimal_bundle(implementation="composite", callable_locator=None)
    bundle = CapabilityBundle.model_validate(composite)
    assert bundle.capability.implementation is ImplementationKind.COMPOSITE
    assert bundle.contract.callable_locator is None


def test_bundle_id_is_operation_id() -> None:
    bundle = CapabilityBundle.model_validate(minimal_bundle())
    assert bundle.contract.operation_id == bundle.capability.id


def test_mismatched_capability_id_and_operation_id_is_rejected() -> None:
    payload = minimal_bundle()
    payload["contract"] = _contract(operation_id="other.op", callable_locator="demo.api:probe")
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_native_and_external_are_otherwise_identical_shapes() -> None:
    native = CapabilityBundle.model_validate(minimal_bundle(implementation="native"))
    external = CapabilityBundle.model_validate(minimal_bundle(implementation="external"))
    # "dependencies" legitimately differs: external's probed executable
    # dependency has no native equivalent (native declares no executables at
    # all, since every executable dependency now requires a matching probe).
    excluded = {"binding", "dependencies"}
    assert native.contract.model_dump(exclude=excluded) == external.contract.model_dump(
        exclude=excluded
    )


def test_bundle_is_frozen() -> None:
    bundle = CapabilityBundle.model_validate(minimal_bundle())
    with pytest.raises(ValidationError):
        bundle.schema_version = "organelleverse.capability.v2"  # pyright: ignore[reportAttributeAccessIssue]


def test_bundle_forbids_unknown_top_level_fields() -> None:
    payload = minimal_bundle()
    payload["unexpected"] = "value"
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_bundle_accepts_core_data_contract_references_and_exposes_them_in_schema() -> None:
    payload = minimal_bundle()
    payload["data_contract"] = [
        {
            "modality": "sequencing_reads",
            "factory_locator": (
                "organelleverse.assembly.data_contract:"
                "released_assembly_sequencing_reads_data_contract"
            ),
        }
    ]

    bundle = CapabilityBundle.model_validate(payload)

    assert bundle.data_contracts[0].modality == "sequencing_reads"
    schema = CapabilityBundle.model_json_schema()
    assert "DataContractReference" in schema["$defs"]


def test_bundle_rejects_duplicate_data_contract_modalities() -> None:
    payload = minimal_bundle()
    payload["data_contract"] = [
        {
            "modality": "sequencing_reads",
            "factory_locator": "demo.contracts:reads",
        },
        {
            "modality": "sequencing_reads",
            "factory_locator": "demo.contracts:other_reads",
        },
    ]

    with pytest.raises(ValidationError, match="duplicate data contract modality"):
        CapabilityBundle.model_validate(payload)


# --- probes --------------------------------------------------------------------


def test_native_forbids_any_probe() -> None:
    payload = minimal_bundle(implementation="native", probes=(_DEFAULT_EXTERNAL_PROBE,))
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_external_requires_at_least_one_probe() -> None:
    payload = minimal_bundle(implementation="external", probes=())
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_forbids_probe() -> None:
    payload = minimal_bundle(
        implementation="composite", callable_locator=None, probes=(_DEFAULT_EXTERNAL_PROBE,)
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_external_with_one_probe_is_valid() -> None:
    bundle = CapabilityBundle.model_validate(minimal_bundle(implementation="external"))
    assert len(bundle.probes) == 1
    assert bundle.probes[0].dependency == "demo_tool"
    assert bundle.probes[0].help_argv == ("--help",)


def test_every_executable_dependency_requires_a_corresponding_probe() -> None:
    payload = minimal_bundle(implementation="external")
    payload["contract"] = _contract(
        operation_id="demo.probe",
        callable_locator="demo.api:probe",
        dependencies=[
            {"kind": "executable", "name": "blastn"},
        ],
    )
    # The declared probe covers "demo_tool", not "blastn".
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)

    payload["probe"] = [
        {"dependency": "blastn", "help_argv": ["-help"]},
    ]
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.contract.dependencies[0].kind is DependencyKind.EXECUTABLE


def test_probe_help_argv_must_be_non_empty() -> None:
    payload = minimal_bundle(
        implementation="external",
        probes=({"dependency": "demo_tool", "help_argv": []},),
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_probe_referencing_an_undeclared_executable_dependency_is_rejected() -> None:
    payload = minimal_bundle(implementation="external")
    payload["contract"] = _contract(
        operation_id="demo.probe",
        callable_locator="demo.api:probe",
        dependencies=[{"kind": "executable", "name": "blastn"}],
    )
    payload["probe"] = [
        {"dependency": "blastn", "help_argv": ["-help"]},
        {"dependency": "some_tool_never_declared", "help_argv": ["--help"]},
    ]
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_duplicate_probe_dependency_is_rejected() -> None:
    payload = minimal_bundle(implementation="external")
    payload["contract"] = _contract(
        operation_id="demo.probe",
        callable_locator="demo.api:probe",
        dependencies=[{"kind": "executable", "name": "blastn"}],
    )
    payload["probe"] = [
        {"dependency": "blastn", "help_argv": ["-help"]},
        {"dependency": "blastn", "help_argv": ["--version"]},
    ]
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


@pytest.mark.parametrize("field", ["help_argv", "version_argv", "requires"])
def test_probe_argv_like_fields_reject_a_blank_element(field: str) -> None:
    probe: dict[str, object] = {"dependency": "demo_tool", "help_argv": ["--help"]}
    probe[field] = ["--help", ""] if field == "help_argv" else ["", "x"]
    payload = minimal_bundle(implementation="external", probes=(probe,))
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_probed_dependency_set_must_equal_executable_dependency_set() -> None:
    """Not just 'every executable has a probe' - the reverse too."""
    payload = minimal_bundle(implementation="external")
    payload["contract"] = _contract(
        operation_id="demo.probe",
        callable_locator="demo.api:probe",
        dependencies=[
            {"kind": "executable", "name": "blastn"},
            {"kind": "executable", "name": "makeblastdb"},
        ],
    )
    payload["probe"] = [
        {"dependency": "blastn", "help_argv": ["-help"]},
        {"dependency": "makeblastdb", "help_argv": ["-help"]},
    ]
    bundle = CapabilityBundle.model_validate(payload)
    assert len(bundle.probes) == 2


# --- fixtures --------------------------------------------------------------------


_MIN_FIXTURE_INPUT: dict[str, object] = {"kind": "genome", "path": "fixtures/case/input/mito.fasta"}


def test_fixture_expect_and_reference_are_mutually_exclusive() -> None:
    neither = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "equivalence": "exact",
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(neither)

    both = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "expect": "fixtures/case/expected.json",
                "reference": "cap:phylogeny.iqtree_ref",
                "equivalence": "exact",
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(both)


def test_fixture_exact_forbids_tolerance() -> None:
    payload = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "expect": "fixtures/case/expected.json",
                "equivalence": "exact",
                "tolerance": 1e-6,
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_fixture_numeric_tolerance_requires_a_positive_tolerance() -> None:
    missing = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "reference": "cap:phylogeny.iqtree_ref",
                "equivalence": "numeric_tolerance",
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(missing)

    negative = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "reference": "cap:phylogeny.iqtree_ref",
                "equivalence": "numeric_tolerance",
                "tolerance": -1e-6,
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(negative)

    valid = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "reference": "cap:phylogeny.iqtree_ref",
                "equivalence": "numeric_tolerance",
                "tolerance": 1e-7,
            },
        )
    )
    bundle = CapabilityBundle.model_validate(valid)
    assert bundle.fixtures[0].tolerance == 1e-7


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), float("-inf")])
def test_fixture_numeric_tolerance_rejects_non_finite_values(tolerance: float) -> None:
    payload = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "reference": "cap:phylogeny.iqtree_ref",
                "equivalence": "numeric_tolerance",
                "tolerance": tolerance,
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_fixture_expect_rejects_a_blank_string() -> None:
    payload = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "expect": "",
                "equivalence": "exact",
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_fixture_reference_rejects_a_blank_string() -> None:
    payload = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "reference": "",
                "equivalence": "numeric_tolerance",
                "tolerance": 1e-7,
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_fixture_case_must_be_unique_within_a_bundle() -> None:
    payload = minimal_bundle(
        fixtures=(
            {
                "case": "duplicate",
                "input": _MIN_FIXTURE_INPUT,
                "expect": "fixtures/case/expected.json",
                "equivalence": "exact",
            },
            {
                "case": "duplicate",
                "input": _MIN_FIXTURE_INPUT,
                "expect": "fixtures/case/expected.json",
                "equivalence": "exact",
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_fixture_set_membership_is_accepted_by_schema_but_forbids_tolerance() -> None:
    payload = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "expect": "fixtures/case/expected.json",
                "equivalence": "set_membership",
            },
        )
    )
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.fixtures[0].equivalence == "set_membership"


# --- composite: declarative step graph, per admission design §5.3 ---------------
#
# capability_id (never suite_op) lives on a CapabilityInvocation, reused for a
# step's own invocation, its condition's if_true/if_false, and its fallback.
# Every mapped input is one of three kinds (constant/composite_input/
# step_output); a step's output is a managed slot name, never a filesystem
# path. Data-flow references (a mapped input naming a step's output) and
# condition references both count as real dependency edges, exactly like
# depends_on - together they must form a DAG whose declared order is already
# topological (no forward references).


def _invocation(capability_id: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {"capability_id": capability_id}
    payload.update(overrides)
    return payload


def test_composite_implementation_requires_a_step_declaration() -> None:
    payload = minimal_bundle(implementation="composite", callable_locator=None, composite=None)
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_native_and_external_forbid_a_composite_step_declaration() -> None:
    native = minimal_bundle(composite={"steps": [_DEFAULT_COMPOSITE_STEP]})
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(native)

    external = minimal_bundle(
        implementation="external", composite={"steps": [_DEFAULT_COMPOSITE_STEP]}
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(external)


def test_composite_with_zero_steps_is_rejected() -> None:
    payload = minimal_bundle(
        implementation="composite", callable_locator=None, composite={"steps": []}
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_step_ids_must_be_unique() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "step_one", "invocation": _invocation("demo.step_one")},
                {"id": "step_one", "invocation": _invocation("demo.step_two")},
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_step_id_rejects_a_blank_string() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={"steps": [{"id": "", "invocation": _invocation("demo.step_one")}]},
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_invocation_capability_id_must_match_operation_id_pattern() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={"steps": [{"id": "step_one", "invocation": _invocation("Not A Valid Id")}]},
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_invocation_capability_id_rejects_a_blank_string() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={"steps": [{"id": "step_one", "invocation": _invocation("")}]},
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_step_requires_exactly_one_of_invocation_or_condition() -> None:
    neither = minimal_bundle(
        implementation="composite", callable_locator=None, composite={"steps": [{"id": "a"}]}
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(neither)

    both = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation("demo.step_a"),
                    "condition": {
                        "reference": {
                            "step_id": "a",
                            "slot": "result",
                            "value_path": "/metrics/gc",
                        },
                        "operator": "eq",
                        "value": 1.0,
                        "if_true": _invocation("demo.true_branch"),
                        "if_false": _invocation("demo.false_branch"),
                    },
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(both)


def test_composite_fallback_is_forbidden_on_a_condition_step() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "source", "invocation": _invocation("demo.source")},
                {
                    "id": "branch",
                    "depends_on": ["source"],
                    "condition": {
                        "reference": {
                            "step_id": "source",
                            "slot": "result",
                            "value_path": "/metrics/gc",
                        },
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.true_branch"),
                        "if_false": _invocation("demo.false_branch"),
                    },
                    "fallback": _invocation("demo.fallback"),
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_step_depends_on_an_unknown_step_is_rejected() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "step_one",
                    "invocation": _invocation("demo.step_one"),
                    "depends_on": ["step_that_does_not_exist"],
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_depends_on_rejects_a_self_reference() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [{"id": "a", "invocation": _invocation("demo.step_a"), "depends_on": ["a"]}],
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_rejects_a_two_step_depends_on_cycle() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "a", "invocation": _invocation("demo.step_a"), "depends_on": ["b"]},
                {"id": "b", "invocation": _invocation("demo.step_b"), "depends_on": ["a"]},
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


# --- input mapping: constant / composite_input / step_output --------------------


def test_composite_input_mapping_constant_is_valid() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation(
                        "demo.step_a",
                        inputs={"threshold": {"kind": "constant", "value": 0.5}},
                    ),
                }
            ]
        },
    )
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.composite is not None
    mapped = bundle.composite.steps[0].invocation
    assert mapped is not None
    threshold = mapped.inputs["threshold"]
    assert isinstance(threshold, ConstantValue)
    assert threshold.value == 0.5


def test_composite_input_mapping_composite_input_reference_requires_a_declared_input() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation(
                        "demo.step_a",
                        inputs={
                            "fasta": {"kind": "composite_input", "input_name": "fasta_path"}
                        },
                    ),
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)

    composite_section = cast(dict[str, object], payload["composite"])
    composite_section["inputs"] = [{"name": "fasta_path"}]
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.composite is not None
    assert bundle.composite.inputs[0].name == "fasta_path"


def test_composite_input_mapping_step_output_reference_to_an_unknown_step_is_rejected() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation(
                        "demo.step_a",
                        inputs={
                            "genome": {
                                "kind": "step_output",
                                "step_id": "step_that_does_not_exist",
                                "slot": "genome",
                            }
                        },
                    ),
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_input_mapping_step_output_reference_rejects_self_reference() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation(
                        "demo.step_a",
                        inputs={
                            "genome": {"kind": "step_output", "step_id": "a", "slot": "g"}
                        },
                    ),
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_step_output_reference_slot_rejects_a_blank_string() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "a", "invocation": _invocation("demo.step_a", output={"slot": "g"})},
                {
                    "id": "b",
                    "invocation": _invocation(
                        "demo.step_b",
                        inputs={
                            "genome": {"kind": "step_output", "step_id": "a", "slot": ""}
                        },
                    ),
                    "depends_on": ["a"],
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_step_output_reference_value_path_rejects_a_blank_string() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "a", "invocation": _invocation("demo.step_a", output={"slot": "g"})},
                {
                    "id": "b",
                    "invocation": _invocation(
                        "demo.step_b",
                        inputs={
                            "genome": {
                                "kind": "step_output",
                                "step_id": "a",
                                "slot": "g",
                                "value_path": "   ",
                            }
                        },
                    ),
                    "depends_on": ["a"],
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_data_reference_without_explicit_depends_on_is_a_valid_implicit_edge() -> None:
    """A step-output reference is itself a real dependency edge - restating
    it in depends_on is not required."""
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "fetch",
                    "invocation": _invocation("demo.fetch", output={"slot": "genome"}),
                },
                {
                    "id": "annotate",
                    "invocation": _invocation(
                        "demo.annotate",
                        inputs={
                            "genome": {
                                "kind": "step_output",
                                "step_id": "fetch",
                                "slot": "genome",
                            }
                        },
                    ),
                },
            ]
        },
    )
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.composite is not None
    assert len(bundle.composite.steps) == 2


def test_composite_accepts_a_valid_multi_step_data_flow() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "fetch",
                    "invocation": _invocation("demo.fetch", output={"slot": "genome"}),
                },
                {
                    "id": "annotate",
                    "invocation": _invocation(
                        "demo.annotate",
                        inputs={
                            "genome": {
                                "kind": "step_output",
                                "step_id": "fetch",
                                "slot": "genome",
                            }
                        },
                        output={"slot": "annotated_genome"},
                    ),
                },
                {
                    "id": "report",
                    "invocation": _invocation(
                        "demo.report",
                        inputs={
                            "annotation": {
                                "kind": "step_output",
                                "step_id": "annotate",
                                "slot": "annotated_genome",
                                "value_path": "/result",
                            }
                        },
                    ),
                },
            ]
        },
    )
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.composite is not None
    assert len(bundle.composite.steps) == 3
    annotate_invocation = bundle.composite.steps[1].invocation
    assert annotate_invocation is not None
    assert annotate_invocation.output is not None
    assert annotate_invocation.output.slot == "annotated_genome"
    report_invocation = bundle.composite.steps[2].invocation
    assert report_invocation is not None
    annotation_input = report_invocation.inputs["annotation"]
    assert isinstance(annotation_input, StepOutputReference)
    assert annotation_input.value_path == "/result"


def test_composite_data_reference_to_a_later_declared_step_is_rejected() -> None:
    """The declared step order must already be a valid topological order -
    a step cannot reference a step declared after it, even if the overall
    graph would otherwise be acyclic. 'fetch' declares a matching output
    slot so the only violation exercised here is the ordering rule, not an
    unrelated missing-slot rejection."""
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "annotate",
                    "invocation": _invocation(
                        "demo.annotate",
                        inputs={
                            "genome": {
                                "kind": "step_output",
                                "step_id": "fetch",
                                "slot": "genome",
                            }
                        },
                    ),
                },
                {
                    "id": "fetch",
                    "invocation": _invocation("demo.fetch", output={"slot": "genome"}),
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


# --- managed output binding: a slot name, never an arbitrary path ---------------


def test_composite_output_binding_rejects_an_arbitrary_output_dir() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation(
                        "demo.step_a", output={"output_dir": "/tmp/whatever"}
                    ),
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_output_binding_rejects_an_arbitrary_write_path() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation("demo.step_a", output={"path": "/tmp/x.json"}),
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_output_binding_slot_is_valid() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "a", "invocation": _invocation("demo.step_a", output={"slot": "result"})}
            ]
        },
    )
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.composite is not None
    invocation = bundle.composite.steps[0].invocation
    assert invocation is not None
    assert invocation.output is not None
    assert invocation.output.slot == "result"


# --- conditions: lt/lte/gt/gte/eq, if_true/if_false/fallback --------------------


@pytest.mark.parametrize("operator", ["lt", "lte", "gt", "gte", "eq"])
def test_composite_condition_accepts_each_operator(operator: str) -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "source",
                    "invocation": _invocation("demo.source", output={"slot": "source_result"}),
                },
                {
                    "id": "branch",
                    "depends_on": ["source"],
                    "condition": {
                        "reference": {
                            "step_id": "source",
                            "slot": "source_result",
                            "value_path": "/metrics/gc",
                        },
                        "operator": operator,
                        "value": 0.5,
                        "if_true": _invocation("demo.true_branch"),
                        "if_false": _invocation("demo.false_branch"),
                    },
                },
            ]
        },
    )
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.composite is not None
    condition = bundle.composite.steps[1].condition
    assert condition is not None
    assert condition.operator == operator


def test_composite_condition_reference_to_an_unknown_step_is_rejected() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "branch",
                    "condition": {
                        "reference": {
                            "step_id": "step_that_does_not_exist",
                            "slot": "result",
                            "value_path": "/metrics/gc",
                        },
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.true_branch"),
                        "if_false": _invocation("demo.false_branch"),
                    },
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_condition_reference_rejects_a_self_reference() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "branch",
                    "condition": {
                        "reference": {
                            "step_id": "branch",
                            "slot": "result",
                            "value_path": "/metrics/gc",
                        },
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.true_branch"),
                        "if_false": _invocation("demo.false_branch"),
                    },
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_condition_reference_to_a_later_declared_step_is_rejected() -> None:
    """'source' declares a matching output slot so the only violation
    exercised here is the declared-order rule, not an unrelated
    missing-slot rejection."""
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "branch",
                    "condition": {
                        "reference": {
                            "step_id": "source",
                            "slot": "source_result",
                            "value_path": "/metrics/gc",
                        },
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.true_branch"),
                        "if_false": _invocation("demo.false_branch"),
                    },
                },
                {
                    "id": "source",
                    "invocation": _invocation("demo.source", output={"slot": "source_result"}),
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_bidirectional_condition_cycle_is_rejected() -> None:
    """step 'a's condition references 'b'; 'b's condition references 'a' -
    a genuine cycle entirely through condition edges, no depends_on at all.
    Both conditions declare compatible, matching output slots on their
    if_true/if_false branches so the cycle detector is what actually fires,
    not an unrelated missing-slot rejection."""
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "condition": {
                        "reference": {
                            "step_id": "b",
                            "slot": "b_result",
                            "value_path": "/metrics/gc",
                        },
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.true_branch", output={"slot": "a_result"}),
                        "if_false": _invocation("demo.false_branch", output={"slot": "a_result"}),
                    },
                },
                {
                    "id": "b",
                    "condition": {
                        "reference": {
                            "step_id": "a",
                            "slot": "a_result",
                            "value_path": "/metrics/gc",
                        },
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.true_branch", output={"slot": "b_result"}),
                        "if_false": _invocation("demo.false_branch", output={"slot": "b_result"}),
                    },
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_full_condition_and_fallback_shape_is_valid() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "inputs": [{"name": "fasta_path"}],
            "steps": [
                {
                    "id": "qc",
                    "invocation": _invocation(
                        "demo.qc",
                        inputs={
                            "fasta": {"kind": "composite_input", "input_name": "fasta_path"}
                        },
                        output={"slot": "qc_result"},
                    ),
                    "fallback": _invocation("demo.qc_fallback", output={"slot": "qc_result"}),
                },
                {
                    "id": "branch",
                    "depends_on": ["qc"],
                    "condition": {
                        "reference": {
                            "step_id": "qc",
                            "slot": "qc_result",
                            "value_path": "/metrics/gc_content",
                        },
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.multiconf"),
                        "if_false": _invocation("demo.detect_contamination"),
                    },
                },
            ],
        },
    )
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.composite is not None
    qc_step, branch_step = bundle.composite.steps
    assert qc_step.fallback is not None
    assert qc_step.fallback.capability_id == "demo.qc_fallback"
    assert qc_step.fallback.output is not None
    assert qc_step.fallback.output.slot == "qc_result"
    assert branch_step.condition is not None
    assert branch_step.condition.if_true.capability_id == "demo.multiconf"
    assert branch_step.condition.if_false.capability_id == "demo.detect_contamination"


# --- managed output binding: referenced steps must declare a matching slot ------


def test_composite_step_output_is_optional_when_not_referenced_downstream() -> None:
    """A step nobody references downstream may omit its output entirely -
    only a step that IS referenced must declare a matching slot."""
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={"steps": [{"id": "a", "invocation": _invocation("demo.step_a")}]},
    )
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.composite is not None
    assert bundle.composite.steps[0].invocation is not None
    assert bundle.composite.steps[0].invocation.output is None


def test_composite_step_output_reference_to_a_step_with_no_output_is_rejected() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "a", "invocation": _invocation("demo.step_a")},
                {
                    "id": "b",
                    "invocation": _invocation(
                        "demo.step_b",
                        inputs={
                            "genome": {"kind": "step_output", "step_id": "a", "slot": "genome"}
                        },
                    ),
                    "depends_on": ["a"],
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_step_output_reference_to_an_unknown_slot_is_rejected() -> None:
    """'a' declares output slot 'genome', but 'b' references slot 'fasta' -
    the step exists and has an output, but the slot name does not match."""
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "a", "invocation": _invocation("demo.step_a", output={"slot": "genome"})},
                {
                    "id": "b",
                    "invocation": _invocation(
                        "demo.step_b",
                        inputs={
                            "genome": {"kind": "step_output", "step_id": "a", "slot": "fasta"}
                        },
                    ),
                    "depends_on": ["a"],
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_fallback_output_absent_while_invocation_has_one_is_rejected() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation("demo.step_a", output={"slot": "result"}),
                    "fallback": _invocation("demo.step_a_fallback"),
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_fallback_output_slot_mismatched_with_invocation_is_rejected() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation("demo.step_a", output={"slot": "result"}),
                    "fallback": _invocation(
                        "demo.step_a_fallback", output={"slot": "fallback_result"}
                    ),
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_condition_if_true_output_absent_while_if_false_has_one_is_rejected() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "source", "invocation": _invocation("demo.source", output={"slot": "s"})},
                {
                    "id": "branch",
                    "depends_on": ["source"],
                    "condition": {
                        "reference": {
                            "step_id": "source",
                            "slot": "s",
                            "value_path": "/metrics/gc",
                        },
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.true_branch"),
                        "if_false": _invocation("demo.false_branch", output={"slot": "result"}),
                    },
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_condition_if_true_if_false_output_slot_mismatch_is_rejected() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "source", "invocation": _invocation("demo.source", output={"slot": "s"})},
                {
                    "id": "branch",
                    "depends_on": ["source"],
                    "condition": {
                        "reference": {
                            "step_id": "source",
                            "slot": "s",
                            "value_path": "/metrics/gc",
                        },
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.true_branch", output={"slot": "true_result"}),
                        "if_false": _invocation(
                            "demo.false_branch", output={"slot": "false_result"}
                        ),
                    },
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


# --- composite input declarations: unique names ----------------------------------


def test_composite_input_declaration_names_must_be_unique() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "inputs": [{"name": "fasta_path"}, {"name": "fasta_path"}],
            "steps": [_DEFAULT_COMPOSITE_STEP],
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


# --- whitespace-only (not just empty) identifiers and references ----------------


def test_composite_step_id_rejects_a_whitespace_only_string() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={"steps": [{"id": "   ", "invocation": _invocation("demo.step_one")}]},
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_composite_input_reference_input_name_rejects_a_whitespace_only_string() -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "inputs": [{"name": "fasta_path"}],
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation(
                        "demo.step_a",
                        inputs={"fasta": {"kind": "composite_input", "input_name": "   "}},
                    ),
                }
            ],
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_fixture_expect_rejects_a_whitespace_only_string() -> None:
    payload = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "expect": "   ",
                "equivalence": "exact",
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


def test_fixture_reference_rejects_a_whitespace_only_string() -> None:
    payload = minimal_bundle(
        fixtures=(
            {
                "case": "case_a",
                "input": _MIN_FIXTURE_INPUT,
                "reference": "   ",
                "equivalence": "numeric_tolerance",
                "tolerance": 1e-7,
            },
        )
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


# --- all non-finite-number entry points: condition value, constant, fixture -----


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_step_condition_branch_value_rejects_non_finite_numbers(value: float) -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "source", "invocation": _invocation("demo.source", output={"slot": "s"})},
                {
                    "id": "branch",
                    "depends_on": ["source"],
                    "condition": {
                        "reference": {
                            "step_id": "source",
                            "slot": "s",
                            "value_path": "/metrics/gc",
                        },
                        "operator": "lt",
                        "value": value,
                        "if_true": _invocation("demo.true_branch"),
                        "if_false": _invocation("demo.false_branch"),
                    },
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), {"nested": [1, float("-inf")]}],
    ids=["top_level_nan", "top_level_inf", "nested_-inf"],
)
def test_composite_constant_input_rejects_non_finite_numbers(value: object) -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "a",
                    "invocation": _invocation(
                        "demo.step_a", inputs={"threshold": {"kind": "constant", "value": value}}
                    ),
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


@pytest.mark.parametrize("field", ["input", "parameters"])
@pytest.mark.parametrize(
    "payload_value",
    [{"x": float("nan")}, {"x": {"y": [float("inf")]}}],
    ids=["top_level", "nested"],
)
def test_fixture_input_and_parameters_reject_non_finite_numbers(
    field: str, payload_value: dict[str, object]
) -> None:
    fixture: dict[str, object] = {
        "case": "case_a",
        "input": _MIN_FIXTURE_INPUT,
        "expect": "fixtures/case/expect.json",
        "equivalence": "exact",
    }
    fixture[field] = payload_value
    payload = minimal_bundle(fixtures=(fixture,))
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


# --- StepId: one identifier pattern for step id, step_id references, depends_on ---


@pytest.mark.parametrize(
    "bad_id",
    [
        "step one",
        "Step_One",
        "step-one",
        "step.one",
        "step/one",
        "../secret",
        "1step",
        " step",
        "step ",
    ],
)
def test_composite_step_id_rejects_non_identifier_strings(bad_id: str) -> None:
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={"steps": [{"id": bad_id, "invocation": _invocation("demo.step_a")}]},
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


@pytest.mark.parametrize(
    "bad_id", ["step one", "step.one", "step/one", "../secret", "Step_One"]
)
def test_composite_depends_on_element_rejects_non_identifier_strings(bad_id: str) -> None:
    """Constructs CompositeStep directly, bypassing CompositeSpec's
    cross-step "unknown step id" check entirely - every bad_id here also
    fails to name any real step, so going through the full bundle would let
    that unrelated check mask whether the StepId pattern itself ever fired."""
    with pytest.raises(ValidationError):
        CompositeStep(
            id="b",
            invocation=CapabilityInvocation(capability_id="demo.step_b"),
            depends_on=(bad_id,),
        )


def test_step_output_reference_step_id_rejects_a_non_identifier_string() -> None:
    """Constructs StepOutputReference directly, for the same reason: "step
    one" also fails to name any real step, so validating through a full
    bundle could pass via "unknown step id" instead of the StepId pattern."""
    with pytest.raises(ValidationError):
        StepOutputReference(step_id="step one", slot="g")


# --- value_path: RFC 6901 JSON Pointer, required on a condition reference -------


def test_step_condition_branch_reference_requires_a_value_path() -> None:
    """A condition compares one numeric value addressed within a step's
    output, never the whole managed-slot result - omitting value_path (the
    "whole slot" spelling, meaningful for an ordinary data-flow mapping) is
    rejected here specifically."""
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "source", "invocation": _invocation("demo.source", output={"slot": "s"})},
                {
                    "id": "branch",
                    "depends_on": ["source"],
                    "condition": {
                        "reference": {"step_id": "source", "slot": "s"},
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.true_branch"),
                        "if_false": _invocation("demo.false_branch"),
                    },
                },
            ]
        },
    )
    with pytest.raises(ValidationError):
        CapabilityBundle.model_validate(payload)


@pytest.mark.parametrize(
    "pointer",
    ["/metrics/gc_content", "/a/~0/~1", "/", "/0", "/a/b/c"],
)
def test_step_output_reference_value_path_accepts_valid_json_pointers(pointer: str) -> None:
    ref = StepOutputReference(step_id="a", slot="s", value_path=pointer)
    assert ref.value_path == pointer


@pytest.mark.parametrize(
    "pointer",
    ["metrics.gc", "metrics..gc", "../secret", "/a/~2bad", "/trailing~", ""],
    ids=[
        "dot_path",
        "double_dot_path",
        "relative_dot_slash",
        "invalid_escape_digit",
        "trailing_unescaped_tilde",
        "empty_string",
    ],
)
def test_step_output_reference_value_path_rejects_invalid_forms(pointer: str) -> None:
    with pytest.raises(ValidationError):
        StepOutputReference(step_id="a", slot="s", value_path=pointer)


def test_step_output_reference_value_path_none_means_the_whole_slot() -> None:
    ref = StepOutputReference(step_id="a", slot="s")
    assert ref.value_path is None


def test_composite_plain_data_mapping_may_omit_value_path() -> None:
    """An ordinary MappedInput StepOutputReference (not a condition
    reference) may still omit value_path entirely, meaning "pass the whole
    upstream managed slot to the downstream capability"."""
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {
                    "id": "fetch",
                    "invocation": _invocation("demo.fetch", output={"slot": "genome"}),
                },
                {
                    "id": "annotate",
                    "invocation": _invocation(
                        "demo.annotate",
                        inputs={
                            "genome": {
                                "kind": "step_output",
                                "step_id": "fetch",
                                "slot": "genome",
                            }
                        },
                    ),
                },
            ]
        },
    )
    bundle = CapabilityBundle.model_validate(payload)
    assert bundle.composite is not None
    annotate_invocation = bundle.composite.steps[1].invocation
    assert annotate_invocation is not None
    genome_input = annotate_invocation.inputs["genome"]
    assert isinstance(genome_input, StepOutputReference)
    assert genome_input.value_path is None


# --- value_path constraints reflected in CapabilityBundle.model_json_schema() ---
# --- itself, verified with Draft202012Validator - not just model_validate() ----


def _capability_bundle_defs() -> dict[str, Any]:
    schema = CapabilityBundle.model_json_schema()
    return cast("dict[str, Any]", schema["$defs"])


@pytest.mark.parametrize(
    "pointer",
    ["metrics.gc", "metrics..gc", "../secret", "/a/~2bad", "/trailing~", ""],
    ids=[
        "dot_path",
        "double_dot_path",
        "relative_dot_slash",
        "invalid_escape_digit",
        "trailing_unescaped_tilde",
        "empty_string",
    ],
)
def test_json_schema_step_output_reference_value_path_rejects_invalid_forms(pointer: str) -> None:
    """The generated JSON Schema itself - not just Pydantic's runtime
    AfterValidator/model_validator, which are invisible to schema
    generation - must reject these: a schema consumer that never runs
    Python (an admission tool checking a capability.toml against the
    frozen schema alone) needs the same guarantee model_validate() gives."""
    value_path_schema = _capability_bundle_defs()["StepOutputReference"]["properties"]["value_path"]
    validator = Draft202012Validator(value_path_schema)
    errors = list(validator.iter_errors(pointer))  # pyright: ignore[reportUnknownMemberType]
    assert errors, f"JSON Schema should reject value_path={pointer!r}"


@pytest.mark.parametrize("pointer", ["/metrics/gc_content", "/a/~0/~1", "/", "/0"])
def test_json_schema_step_output_reference_value_path_accepts_valid_pointers(pointer: str) -> None:
    value_path_schema = _capability_bundle_defs()["StepOutputReference"]["properties"]["value_path"]
    validator = Draft202012Validator(value_path_schema)
    errors = list(validator.iter_errors(pointer))  # pyright: ignore[reportUnknownMemberType]
    assert not errors, f"JSON Schema should accept value_path={pointer!r}, got {errors}"


def test_json_schema_step_output_reference_value_path_is_not_required() -> None:
    """An ordinary MappedInput StepOutputReference's schema must still
    allow omitting value_path entirely - meaning "the whole slot"."""
    assert "value_path" not in _capability_bundle_defs()["StepOutputReference"]["required"]


def test_json_schema_condition_reference_requires_value_path() -> None:
    """ConditionReference's schema must list value_path as required - the
    field-presence distinction between a plain data-flow reference and a
    condition's own reference must survive into the generated schema, not
    live only in a Pydantic model_validator invisible to schema consumers."""
    assert "value_path" in _capability_bundle_defs()["ConditionReference"]["required"]


def test_json_schema_condition_reference_rejects_a_payload_missing_value_path() -> None:
    validator = Draft202012Validator(_capability_bundle_defs()["ConditionReference"])
    errors = list(validator.iter_errors({"step_id": "source", "slot": "s"}))  # pyright: ignore[reportUnknownMemberType]
    assert errors


@pytest.mark.parametrize("pointer", ["metrics.gc", "../secret", "/a/~2bad"])
def test_json_schema_condition_reference_rejects_invalid_value_path_forms(pointer: str) -> None:
    validator = Draft202012Validator(_capability_bundle_defs()["ConditionReference"])
    payload = {"step_id": "source", "slot": "s", "value_path": pointer}
    errors = list(validator.iter_errors(payload))  # pyright: ignore[reportUnknownMemberType]
    assert errors, f"JSON Schema should reject ConditionReference.value_path={pointer!r}"


@pytest.mark.parametrize("pointer", ["/metrics/gc_content", "/a/~0/~1"])
def test_json_schema_condition_reference_accepts_valid_value_path_forms(pointer: str) -> None:
    validator = Draft202012Validator(_capability_bundle_defs()["ConditionReference"])
    payload = {"step_id": "source", "slot": "s", "value_path": pointer}
    errors = list(validator.iter_errors(payload))  # pyright: ignore[reportUnknownMemberType]
    assert not errors, f"JSON Schema should accept ConditionReference.value_path={pointer!r}, got {errors}"


def test_json_schema_step_condition_branch_reference_refs_condition_reference() -> None:
    """StepConditionBranch.reference must $ref ConditionReference (the
    required-value_path variant), not the shared StepOutputReference used
    by ordinary MappedInput entries - otherwise the required constraint
    above would never actually be reachable from a real composite bundle's
    condition."""
    reference_schema = _capability_bundle_defs()["StepConditionBranch"]["properties"]["reference"]
    assert reference_schema == {"$ref": "#/$defs/ConditionReference"}


def test_json_schema_full_bundle_validates_a_complete_composite_condition_payload() -> None:
    """End-to-end: validate a full CapabilityBundle payload against
    CapabilityBundle.model_json_schema() itself via Draft202012Validator -
    not an isolated sub-schema - proving the $ref graph composes correctly
    for exactly what a non-Python schema consumer would see."""
    payload = minimal_bundle(
        implementation="composite",
        callable_locator=None,
        composite={
            "steps": [
                {"id": "source", "invocation": _invocation("demo.source", output={"slot": "s"})},
                {
                    "id": "branch",
                    "depends_on": ["source"],
                    "condition": {
                        "reference": {
                            "step_id": "source",
                            "slot": "s",
                            "value_path": "/metrics/gc_content",
                        },
                        "operator": "lt",
                        "value": 0.6,
                        "if_true": _invocation("demo.true_branch"),
                        "if_false": _invocation("demo.false_branch"),
                    },
                },
            ]
        },
    )
    # A schema consumer only ever sees genuine JSON (lists, not the tuples
    # this Python test helper builds for convenience) - round-trip through
    # json so Draft202012Validator sees exactly that, not Python-only shapes
    # jsonschema's "array"/"string" type checks don't recognize.
    json_payload = json.loads(json.dumps(payload))
    validator = Draft202012Validator(CapabilityBundle.model_json_schema())
    errors = list(validator.iter_errors(json_payload))  # pyright: ignore[reportUnknownMemberType]
    assert not errors, f"a valid bundle should validate cleanly against its own schema: {errors}"

    # Corrupt the condition reference's value_path to a dot-path: the full
    # bundle schema - not just an isolated sub-schema - must reject it.
    bad_payload = copy.deepcopy(json_payload)
    bad_payload["composite"]["steps"][1]["condition"]["reference"]["value_path"] = "metrics.gc"
    bad_errors = list(validator.iter_errors(bad_payload))  # pyright: ignore[reportUnknownMemberType]
    assert bad_errors, "the full bundle schema should reject a dot-path value_path"

    # Remove value_path entirely: the full bundle schema must reject a
    # condition reference missing it.
    missing_payload = copy.deepcopy(json_payload)
    del missing_payload["composite"]["steps"][1]["condition"]["reference"]["value_path"]
    missing_errors = list(validator.iter_errors(missing_payload))  # pyright: ignore[reportUnknownMemberType]
    assert missing_errors, "the full bundle schema should reject a condition reference with no value_path"
