"""A registered operation and the public Python function that runs it take the same parameters.

``ov.operations.public_function`` names the ``ov.<suite>.<function>`` for an
operation id (see ``organelleverse.operations.public_api``). A caller that builds
a form from ``parameter_schema`` and then calls the Python function must be able
to pass every field, which is what ``fetch.entrez_query`` once broke.
"""

from __future__ import annotations

import inspect

import pytest

import organelleverse as ov
from organelleverse.operations import PUBLIC_FUNCTIONS, public_function, resolve_public_function

# The registered operations a consumer (phytoorga-web, 2026-10-16) found without a
# public function. Each now resolves; see docs/operations/public-python-functions.md.
FORMERLY_REGISTRY_ONLY = (
    "fetch.accession_list",
    "fetch.gene_records",
    "fetch.ngdc_gwh",
    "heteroplasmy.estimate_structural_fractions",
    "heteroplasmy.quantify_variant_fractions",
    "localization.score_nls",
    "localization.score_pts",
    "localization.score_signal_peptide",
    "localization.score_transit_peptides",
    "localization.score_transmembrane",
    "morphology.batch_measure",
    "phylogeny.run_astral",
    "ppr.predict_binding_sites",
    "rna_editing.quantify_efficiency",
    "rna_seq.quantify_exon_expression",
    "rna_seq.quantify_splicing_efficiency",
)


def _accepts(function: object, names: set[str]) -> set[str]:
    parameters = inspect.signature(function).parameters  # type: ignore[arg-type]
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return set()
    return names - set(parameters)


def test_every_resolved_public_function_accepts_its_operation_parameters() -> None:
    lacking = {}
    for spec in ov.operations.list():
        function = resolve_public_function(spec.operation_id)
        if function is None:
            continue
        properties = set(ov.operations.parameter_schema(spec.operation_id).get("properties", {}))
        missing = _accepts(function, properties)
        if missing:
            lacking[spec.operation_id] = sorted(missing)
    assert lacking == {}


@pytest.mark.parametrize("operation_id", FORMERLY_REGISTRY_ONLY)
def test_operations_the_site_uses_have_a_public_function(operation_id: str) -> None:
    path = public_function(operation_id)
    assert path is not None
    suite, _, name = path.partition(".")
    assert getattr(getattr(ov, suite), name) is resolve_public_function(operation_id)


def test_the_name_table_only_covers_registered_ids_without_a_same_name_function() -> None:
    registered = {spec.operation_id for spec in ov.operations.list()}
    for operation_id, path in PUBLIC_FUNCTIONS.items():
        assert operation_id in registered
        suite, _, name = operation_id.partition(".")
        # A same-name function would make the id ambiguous: one name per function.
        assert not hasattr(getattr(ov, suite), name), operation_id
        assert public_function(operation_id) == path


def test_an_operation_without_a_public_function_resolves_to_none() -> None:
    assert public_function("rna_editing.encode_matrix") is None
    assert public_function("no_such_suite.no_such_operation") is None
