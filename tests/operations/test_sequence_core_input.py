"""Tests for sequence-of-core inputs (Ruling 3, Decision 004 approval 2026-08-14).

``OperationSpec.input_sequence`` makes the canonical-core first parameter a
``list[T]``/``Sequence[T]`` of the declared core type; the Agent JSON input is
an array of core-object payloads, each validated against T's L1 schema before
the callable runs. ``pangenome.build_graph`` is the documented honest refusal
(a Callable-injection parameter no binding mode accepts).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from organelleverse.capabilities.parser import parse_capability_bundle
from organelleverse.core.errors import OrganelleInputError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.adapters.json import _decode_input
from organelleverse.operations.spec import CoreKind, OperationSpec


def _genome_payload(accession: str) -> dict[str, object]:
    from organelleverse.core.artifacts import ArtifactRef
    from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata

    genome = OrganelleGenome.model_validate(
        {
            "organelle": "mitochondrion",
            "sequence": ArtifactRef(
                kind="sequence",
                uri=f"{accession}.fa",
                format="fasta",
                sha256="a" * 64,
                size_bytes=8,
            ),
            "metadata": OrganelleMetadata(
                species=f"Species {accession}", source="test"
            ),
        }
    )
    return genome.model_dump(mode="json")


def test_bundle_declares_input_sequence() -> None:
    from pathlib import Path

    bundle = parse_capability_bundle(
        Path(__file__).resolve().parents[2]
        / "src" / "organelleverse" / "capabilities"
        / "comparative-compare-genes"
        / "capability.toml"
    )
    assert bundle.contract.input_sequence is True
    assert bundle.contract.input_kind is CoreKind.GENOME


def test_decode_input_sequence_validates_every_element() -> None:
    decoded = _decode_input(
        [_genome_payload("a"), _genome_payload("b")],
        CoreKind.GENOME,
        input_sequence=True,
    )
    assert isinstance(decoded, list) and len(decoded) == 2
    assert all(isinstance(item, OrganelleGenome) for item in decoded)


def test_decode_input_sequence_rejects_non_array() -> None:
    with pytest.raises(OrganelleInputError):
        _decode_input(_genome_payload("a"), CoreKind.GENOME, input_sequence=True)


def test_decode_input_sequence_rejects_bad_element() -> None:
    with pytest.raises(OrganelleInputError):
        _decode_input(
            [_genome_payload("a"), {"not": "a genome"}],
            CoreKind.GENOME,
            input_sequence=True,
        )


def test_spec_rejects_sequence_without_core_kind() -> None:
    base = dict(
        operation_id="t.seq",
        contract_version="1.0",
        title="T",
        description="a test operation spec for the ruling",
        keywords=("a", "b", "c"),
        execution_mode="inline",
        stage="analyze",
        input_kind="none",
        output_kind="result",
    )
    with pytest.raises(ValidationError):
        OperationSpec(**base, input_sequence=True)


def test_invoke_json_runs_a_sequence_operation_end_to_end(
    tmp_path, monkeypatch
) -> None:
    from organelleverse.capabilities.discovery import discover_capabilities
    from organelleverse.capabilities.verification import (
        LocalVerificationEnvironment,
        VerificationStore,
        verify_capability,
    )
    from organelleverse.operations.adapters.json import invoke_json
    from organelleverse.operations.registry import registry

    # a private registry + this test's own verification records: immune to
    # whatever home/store the shared session registry happened to attach
    from organelleverse.capabilities.index import IndexBindingSource
    from organelleverse.capabilities.verification import default_verification_store
    from organelleverse.operations.registry import OperationRegistry

    index = discover_capabilities()
    environment = LocalVerificationEnvironment(index)
    verify_capability(
        "comparative.compare_genes",
        store=VerificationStore(default_verification_store().root),
        environment=environment,
    )
    private_registry = OperationRegistry()
    private_registry.attach_capability_source(
        IndexBindingSource(discover_capabilities())
    )

    response = invoke_json(
        {
            "operation_id": "comparative.compare_genes",
            "input": [_genome_payload("a"), _genome_payload("b")],
            "parameters": {},
        },
        registry=private_registry,
        granted_side_effects=["read_files"],
    )
    assert response["ok"] is True, response.get("error")
    # the JSON transport returns the L1 result as its serialized form
    result = response["result"]
    metrics = result.metrics if isinstance(result, OrganelleResult) else result["metrics"]
    assert metrics["genome_count"] == 2
