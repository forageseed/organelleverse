"""The fetch suite is bound into the operation registry. Offline.

Defining a spec is not registering it. The release catalog calls
``register_fetch_operations`` for the default registry; callers use it directly
for private registries. These tests keep both paths from silently regressing.

They also pin down the reason there is a separate ``_agent_ops`` layer: the
contract *rejected* the Python API. A READ operation must take its source
positionally and every parameter must be JSON-safe, and ``entrez_query`` is
keyword-only with a ``Transport`` protocol and a ``FetchFilters`` dataclass in
its signature. That rejection is correct — an agent sends JSON, and it cannot
send a Protocol.
"""

from __future__ import annotations

import pytest

from organelleverse.core.data import OrganelleData
from organelleverse.core.errors import (
    OrganelleContractError,
    OrganelleInputError,
    OrganelleParameterError,
)
from organelleverse.operations import OperationRegistry
from tests._paths import child_env


@pytest.fixture
def registry() -> OperationRegistry:
    """A private registry — never mutate global process state in a test."""
    from organelleverse.fetch.operations import register_fetch_operations

    local = OperationRegistry()
    register_fetch_operations(local)
    return local


EXPECTED = {
    "fetch.refseq_snapshot",
    "fetch.entrez_query",
    "fetch.accession_list",
    "fetch.gene_records",
    "fetch.nuclear_assembly",
    "fetch.genome_size_candidates",
    "fetch.ngdc_gwh",
    "fetch.dedup_genomes",
}


# ─────────────────────────────────────────────────────────────
# 1. list — the catalog is no longer empty
# ─────────────────────────────────────────────────────────────


def test_every_fetch_spec_is_bound(registry: OperationRegistry) -> None:
    assert {spec.operation_id for spec in registry.list()} == EXPECTED


def test_the_release_catalog_registers_fetch_in_the_default_registry() -> None:
    """Agents discover released fetch operations without a second loader."""
    from organelleverse import operations

    assert {spec.operation_id for spec in operations.list()} >= EXPECTED


def test_registration_is_idempotent(registry: OperationRegistry) -> None:
    """Importing twice must not explode."""
    from organelleverse.fetch.operations import register_fetch_operations

    register_fetch_operations(registry)
    assert {spec.operation_id for spec in registry.list()} == EXPECTED


# ─────────────────────────────────────────────────────────────
# 2. schema — the agent tool-calling surface, derived not hand-written
# ─────────────────────────────────────────────────────────────


def test_a_json_schema_is_derived_for_every_operation(registry: OperationRegistry) -> None:
    for operation_id in EXPECTED:
        schema = registry.parameter_schema(operation_id)
        assert schema["type"] == "object"
        assert "properties" in schema


def test_the_accession_list_schema_names_its_required_parameters(
    registry: OperationRegistry,
) -> None:
    schema = registry.parameter_schema("fetch.accession_list")
    assert set(schema["required"]) == {"organelle", "dest", "accessions"}
    assert "genbank_metadata" in schema["properties"]  # optional, has a default


def test_a_transform_schema_excludes_the_core_input(registry: OperationRegistry) -> None:
    """``data`` is the core object, not a JSON parameter."""
    schema = registry.parameter_schema("fetch.dedup_genomes")
    assert "data" not in schema.get("properties", {})


def test_dedup_invocation_schema_uses_the_builtin_organelle_records_contract(
    registry: OperationRegistry,
) -> None:
    schema = registry.invocation_schema("fetch.dedup_genomes")
    input_schema = schema["properties"]["input"]

    assert input_schema["properties"]["modality"]["const"] == "organelle_records"
    payload_schema = input_schema["properties"]["payload"]
    assert payload_schema["additionalProperties"] is True
    assert "records" in payload_schema["required"]


# ─────────────────────────────────────────────────────────────
# 3. invoke — JSON in, core object out
# ─────────────────────────────────────────────────────────────


def test_a_transform_invoked_with_json_returns_a_core_object(
    registry: OperationRegistry,
) -> None:
    """The full agent shape, without touching the network: a core object in, a
    core object out, and the duplicate collapsed."""
    data = OrganelleData(
        modality="organelle_records",
        payload={
            "manifest": {"n_records": 2},
            "records": [
                {
                    "accession": "BA000029.3",
                    "organism": "Oryza sativa",
                    "length": 490_520,
                    "is_refseq": False,
                    "completeness": "",
                    "derived_from": "",
                    "comment": "",
                },
                {
                    "accession": "NC_011033.1",
                    "organism": "Oryza sativa",
                    "length": 490_520,
                    "is_refseq": True,
                    "completeness": "full",
                    "derived_from": "BA000029",
                    "comment": "The reference sequence was derived from BA000029.",
                },
            ],
        },
    )

    result = registry.invoke("fetch.dedup_genomes", input=data, parameters={})

    assert isinstance(result, OrganelleData)
    manifest = result.payload["manifest"]
    assert manifest["n_genomes_deduped"] == 1
    assert manifest["n_duplicates_removed"] == 1
    (cluster,) = result.payload["clusters"]
    assert cluster["representative"] == "NC_011033.1"
    assert cluster["reason"] == "refseq_mirror"


def test_dedup_keeps_the_original_records(registry: OperationRegistry) -> None:
    """Dedup marks duplicates; it never deletes evidence."""
    data = OrganelleData(
        modality="organelle_records",
        payload={
            "records": [
                {"accession": "A.1", "is_refseq": False, "derived_from": "", "comment": ""},
                {"accession": "NC_1.1", "is_refseq": True, "derived_from": "A", "comment": "x"},
            ]
        },
    )
    result = registry.invoke("fetch.dedup_genomes", input=data, parameters={})
    assert len(result.payload["records"]) == 2


def test_an_unparsed_refseq_link_reaches_the_manifest(registry: OperationRegistry) -> None:
    """If NCBI changes its COMMENT wording, the mirror goes unrecognised and a
    genome is counted twice. That must never be silent."""
    data = OrganelleData(
        modality="organelle_records",
        payload={
            "records": [
                {
                    "accession": "NC_9.1",
                    "is_refseq": True,
                    "comment": "Some wording we have never seen.",
                    "derived_from": "",
                }
            ]
        },
    )
    result = registry.invoke("fetch.dedup_genomes", input=data, parameters={})
    # OrganelleData freezes its payload, so the list arrives as a tuple.
    assert list(result.payload["manifest"]["unlinked_refseq_records"]) == ["NC_9.1"]


# ─────────────────────────────────────────────────────────────
# 4. the contract rejects bad calls before they run
# ─────────────────────────────────────────────────────────────


def test_an_unknown_operation_id_is_rejected(registry: OperationRegistry) -> None:
    with pytest.raises(OrganelleInputError):
        registry.invoke("fetch.does_not_exist", input=None, parameters={})


def test_missing_required_parameters_are_rejected(registry: OperationRegistry) -> None:
    """No network call is made: the contract stops it first."""
    with pytest.raises(OrganelleParameterError):
        registry.invoke("fetch.accession_list", input=None, parameters={"organelle": "mito"})


def test_the_python_api_itself_cannot_be_registered() -> None:
    """The reason ``_agent_ops`` exists. ``entrez_query`` is keyword-only and
    takes a ``Transport`` protocol — neither is expressible in an agent's JSON,
    and the contract says so rather than letting it through.
    """
    from organelleverse.fetch.ncbi import entrez_query
    from organelleverse.fetch.operations import ENTREZ_QUERY_SPEC

    local = OperationRegistry()
    with pytest.raises(OrganelleContractError):
        local.register(ENTREZ_QUERY_SPEC, entrez_query)


# ─────────────────────────────────────────────────────────────
# 5. Cold-start discovery
#
# An agent enumerates the catalog before it knows what to call. `operations`
# loads the release catalog on import, so a cold interpreter sees fetch without
# requiring the caller to import `ov.fetch` first. Discovery still performs no
# I/O and loads no heavy dependency. There is exactly one discovery entry point.
# ─────────────────────────────────────────────────────────────


def test_a_cold_interpreter_sees_fetch_without_an_explicit_suite_import(tmp_path) -> None:
    import json
    import subprocess
    import sys

    script = """
import json
from organelleverse import operations

ids = sorted(spec.operation_id for spec in operations.list())
print(json.dumps({
    "fetch_ids": [i for i in ids if i.startswith("fetch.")],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=child_env(),
    )
    result = json.loads(completed.stdout)

    assert set(result["fetch_ids"]) == EXPECTED


def test_a_cold_interpreter_can_load_the_human_facade_before_the_catalog(tmp_path) -> None:
    """Human and agent entry points must work in either import order."""
    import json
    import subprocess
    import sys

    script = """
import json
import organelleverse as ov

facade_function = ov.fetch.fetch_accessions.__name__
from organelleverse import operations

print(json.dumps({
    "facade_function": facade_function,
    "fetch_ids": sorted(
        spec.operation_id
        for spec in operations.list()
        if spec.operation_id.startswith("fetch.")
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=child_env(),
    )
    result = json.loads(completed.stdout)

    assert result == {
        "facade_function": "fetch_accessions",
        "fetch_ids": sorted(EXPECTED),
    }


def test_the_catalog_is_the_only_discovery_entry_point() -> None:
    """A second loader would be a second thing to keep in sync. There is one."""
    import organelleverse as ov

    assert not hasattr(ov, "load_operations")


# ─────────────────────────────────────────────────────────────
# 6. Contract conformance — the spec must describe what actually comes back
#
# `fetch.gene_records` declared modality `gene_records` while `fetch_genes()`
# returned `organelle_records`. Every test passed: none of them exercised the
# real return path. A contract nothing checks is decoration.
# ─────────────────────────────────────────────────────────────


def test_every_operation_returns_the_modality_its_spec_declares(
    registry: OperationRegistry, tmp_path
) -> None:
    """Offline, through the real return path, for each modality-declaring op."""
    from organelleverse.fetch.genes import fetch_genes
    from organelleverse.fetch.ncbi import entrez_query
    from tests.fetch.test_fetch_ncbi import FakeTransport

    produced = {
        "fetch.entrez_query": entrez_query(
            organelle="plastid", dest=tmp_path / "a", transport=FakeTransport()
        ),
        "fetch.gene_records": fetch_genes(
            gene="matK", dest=tmp_path / "b", transport=FakeTransport()
        ),
    }

    for operation_id, data in produced.items():
        declared = registry.describe(operation_id).output_modalities
        assert data.modality in declared, (
            f"{operation_id} declares {declared} but returned {data.modality!r}"
        )


def test_gene_records_are_not_organelle_records() -> None:
    """The specific deviation, pinned. A matK fragment is not an organelle
    genome, and the object must not claim to be one."""
    import tempfile

    from organelleverse.fetch.genes import fetch_genes
    from tests.fetch.test_fetch_ncbi import FakeTransport

    with tempfile.TemporaryDirectory() as tmp:
        data = fetch_genes(gene="matK", dest=tmp, transport=FakeTransport())
    assert data.modality == "gene_records"


# ─────────────────────────────────────────────────────────────
# 7. Module-level exports
# ─────────────────────────────────────────────────────────────


def test_the_four_alternate_nuclear_sources_are_exported() -> None:
    """Alternate nuclear sources use their full canonical function names."""
    import organelleverse.fetch as fetch_module

    for name in ("fetch_gir", "fetch_imp", "fetch_pgd", "fetch_tair"):
        assert callable(getattr(fetch_module, name))
        assert name in fetch_module.__all__
