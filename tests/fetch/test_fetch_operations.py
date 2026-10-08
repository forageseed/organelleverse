"""The fetch OperationSpecs, and the violations the contract must refuse.

The refusal tests are the first concrete instance of contract-violation
injection: a spec that lies about its own trustworthiness must fail to
construct. A validator with no teeth is worse than no validator.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from organelleverse.fetch._agent_ops import op_refseq_snapshot
from organelleverse.fetch.operations import (
    ENTREZ_QUERY_SPEC,
    GENOME_SIZE_CANDIDATES_SPEC,
    REFSEQ_SNAPSHOT_SPEC,
)
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationSpec,
    OperationStage,
    SideEffect,
)

# ─────────────────────────────────────────────────────────────
# The declared specs
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec", [REFSEQ_SNAPSHOT_SPEC, ENTREZ_QUERY_SPEC, GENOME_SIZE_CANDIDATES_SPEC]
)
def test_fetch_specs_are_read_stage_producing_data(spec: OperationSpec) -> None:
    assert spec.stage is OperationStage.READ
    assert spec.input_kind is CoreKind.NONE
    assert spec.output_kind is CoreKind.DATA


@pytest.mark.parametrize(
    "spec", [REFSEQ_SNAPSHOT_SPEC, ENTREZ_QUERY_SPEC, GENOME_SIZE_CANDIDATES_SPEC]
)
def test_fetch_specs_declare_network_and_writes(spec: OperationSpec) -> None:
    """Downloading touches the network and the disk. Say so."""
    assert SideEffect.NETWORK in spec.side_effects
    assert SideEffect.WRITE_FILES in spec.side_effects


@pytest.mark.parametrize(
    "spec", [REFSEQ_SNAPSHOT_SPEC, ENTREZ_QUERY_SPEC, GENOME_SIZE_CANDIDATES_SPEC]
)
def test_fetch_specs_are_not_cacheable(spec: OperationSpec) -> None:
    """Forced by the contract: network/write side effects forbid caching."""
    assert spec.cacheable is False


@pytest.mark.parametrize(
    "spec", [REFSEQ_SNAPSHOT_SPEC, ENTREZ_QUERY_SPEC, GENOME_SIZE_CANDIDATES_SPEC]
)
def test_fetch_specs_cite_nothing_because_nothing_is_benchmarked_yet(
    spec: OperationSpec,
) -> None:
    """An empty citation list is the honest way to say 'not yet validated'."""
    assert spec.references == ()


@pytest.mark.parametrize(
    "spec", [REFSEQ_SNAPSHOT_SPEC, ENTREZ_QUERY_SPEC, GENOME_SIZE_CANDIDATES_SPEC]
)
def test_fetch_specs_are_retryable_on_transient_network_failure(
    spec: OperationSpec,
) -> None:
    assert spec.retry.max_attempts > 1
    assert spec.retry.retryable_error_codes


def test_refseq_snapshot_is_deterministic_because_the_release_is_pinned() -> None:
    assert REFSEQ_SNAPSHOT_SPEC.deterministic is True
    assert inspect.signature(op_refseq_snapshot).parameters["release"].default is (
        inspect.Parameter.empty
    )


def test_entrez_query_is_not_deterministic_because_the_database_is_live() -> None:
    """nuccore changes daily. Claiming reproducibility here would be a lie."""
    assert ENTREZ_QUERY_SPEC.deterministic is False


def test_genome_size_candidates_is_not_deterministic_because_assemblies_land() -> None:
    """NCBI Datasets is live: new assemblies land and the ranking can shift."""
    assert GENOME_SIZE_CANDIDATES_SPEC.deterministic is False
    assert GENOME_SIZE_CANDIDATES_SPEC.idempotent is False


def test_the_two_channels_are_separate_operations() -> None:
    """Different determinism => different operation. Merging them forces a lie."""
    assert REFSEQ_SNAPSHOT_SPEC.operation_id != ENTREZ_QUERY_SPEC.operation_id
    assert REFSEQ_SNAPSHOT_SPEC.deterministic != ENTREZ_QUERY_SPEC.deterministic


# ─────────────────────────────────────────────────────────────
# Contract-violation injection — the validator must refuse these
# ─────────────────────────────────────────────────────────────


def _fetch_spec(**overrides: object) -> OperationSpec:
    base: dict[str, object] = {
        "operation_id": "fetch.probe",
        "contract_version": "1.0",
        "title": "Demo fetch probe",
        "description": "Test fixture spec used only to probe fetch contract violations.",
        "keywords": ("demo", "fetch", "test"),
        "execution_mode": ExecutionMode.DURABLE,
        "stage": OperationStage.READ,
        "input_kind": CoreKind.NONE,
        "output_kind": CoreKind.DATA,
        # I1a/I1d: a DATA output must declare exactly one output modality.
        "output_modalities": ("organelle_records",),
        "callable_locator": "organelleverse.fetch.ncbi:entrez_query",
        "side_effects": (SideEffect.NETWORK, SideEffect.WRITE_FILES),
    }
    base.update(overrides)
    return OperationSpec(**base)  # type: ignore[arg-type]


def test_the_baseline_probe_spec_is_valid() -> None:
    """Guard the guard: the violations below must fail for the stated reason."""
    assert _fetch_spec().operation_id == "fetch.probe"


def test_contract_refuses_a_cacheable_network_operation() -> None:
    """A cached network read is not reproducible. The type system says no."""
    with pytest.raises(ValidationError):
        _fetch_spec(cacheable=True, deterministic=True)


def test_contract_refuses_a_blank_reference() -> None:
    """A citation that is whitespace is not a citation."""
    with pytest.raises(ValidationError):
        _fetch_spec(references=("   ",))


def test_contract_refuses_a_read_stage_that_consumes_a_core_object() -> None:
    with pytest.raises(ValidationError):
        _fetch_spec(input_kind=CoreKind.GENOME)
