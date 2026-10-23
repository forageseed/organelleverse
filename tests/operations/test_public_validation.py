"""Behavioral parity at native Python and registered operation boundaries."""

from __future__ import annotations

import importlib
import inspect
import pydoc
from unittest.mock import patch

import pytest
from organelleverse.core.errors import OrganelleParameterError
from organelleverse.operations.public_validation import python_signature
from organelleverse.operations.python_only import PYTHON_ONLY_PARAMETERS

import organelleverse as ov


@pytest.fixture
def genome(tmp_path):
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">test\nACGTACGT\n", encoding="utf-8")
    return ov.read(fasta, organelle="mitochondrion", species="Testus auditii")


def test_every_mapped_public_entry_declares_validation_and_keeps_its_signature():
    entries = 0
    for spec in ov.operations.list():
        function = ov.operations.resolve_public_function(spec.operation_id)
        if function is None:
            continue
        entries += 1
        assert hasattr(function, "__ov_operation_id__"), spec.operation_id
        assert inspect.signature(function) == inspect.signature(function.__wrapped__)
        assert "shared operation validator" in pydoc.render_doc(function)
        binding = ov.operations.registry.require(spec.operation_id)
        core_name = binding.signature.input_name
        if core_name not in inspect.signature(function).parameters:
            core_name = None
        _, _, model = python_signature(function.__wrapped__, function.__ov_python_only__, core_name)
        # Required values are absent here; only invalid optional defaults are defects.
        try:
            model.model_validate({})
        except Exception as error:
            assert hasattr(error, "errors"), (spec.operation_id, error)
            assert all(e["type"] == "missing" for e in error.errors()), (spec.operation_id, error)
    assert entries == 250


def test_every_scientific_argument_is_registered_or_has_an_explicit_native_reason():
    for spec in ov.operations.list():
        function = ov.operations.resolve_public_function(spec.operation_id)
        if function is None:
            continue
        binding = ov.operations.registry.require(spec.operation_id)
        represented = set(binding.signature.parameter_model.model_fields)
        represented.add(binding.signature.input_name)
        represented |= {p.name for p in spec.binding.parameters if p.source.value == "environment"}
        missing = set(inspect.signature(function).parameters) - represented
        exceptions = PYTHON_ONLY_PARAMETERS.get(spec.operation_id, {})
        assert missing == set(exceptions), (spec.operation_id, missing, exceptions)
        assert all(exceptions.values())


def test_native_and_registered_gc_share_window_parameter_and_validation(tmp_path):
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">test\nACGTACGT\n", encoding="utf-8")
    registry_module = importlib.import_module("organelleverse.operations.registry")

    with patch.object(
        registry_module, "_invoke_checked_python", wraps=registry_module._invoke_checked_python
    ) as checked:
        native = ov.composition.gc_content(fasta, window_size=4)
        assert checked.call_count == 1
    registered = ov.operations.invoke(
        "composition.gc_content",
        input=None,
        parameters={"genome_fasta": str(fasta), "window_size": 4},
    )
    assert native.metrics == registered.metrics
    assert native.metrics["windows"] == 2


@pytest.mark.parametrize("window", [0, -1, True, "4"])
def test_gc_rejects_invalid_windows_before_file_io(window):
    for function in (ov.composition.gc_content, ov.composition.compute_gc_content):
        with pytest.raises(OrganelleParameterError):
            function("does-not-exist.fasta", window_size=window)
    with pytest.raises(OrganelleParameterError):
        ov.operations.invoke(
            "composition.gc_content",
            input=None,
            parameters={"genome_fasta": "missing.fasta", "window_size": window},
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"threads": 300},
        {"threads": True},
        {"call_trna": "false"},
    ],
)
def test_annotation_rejects_bad_values_before_backend(genome, parameters):
    from organelleverse.annotation import api

    with patch.object(api, "run_annotation") as backend:
        for call in (
            lambda: ov.annotation.annotate(genome, **parameters),
            lambda: ov.operations.invoke(
                "annotation.annotate", input=genome, parameters=parameters
            ),
        ):
            with pytest.raises(OrganelleParameterError):
                call()
        backend.assert_not_called()


@pytest.mark.parametrize(
    "parameters",
    [
        {"threshold": -0.1},
        {"threshold": 2.0},
        {"backend": "not-a-backend"},
    ],
)
def test_rna_editing_constraints_match_before_model_execution(genome, parameters):
    with pytest.raises(OrganelleParameterError):
        ov.rna_editing.predict_edits(genome, **parameters)
    with pytest.raises(OrganelleParameterError):
        ov.operations.invoke("rna_editing.predict_edits", input=genome, parameters=parameters)


@pytest.mark.parametrize(
    "operation,field,value",
    [
        ("fetch.gene_records", "scope", "not-a-scope"),
        ("fetch.ngdc_gwh", "organelle", "not-an-organelle"),
        ("fetch.nuclear_assembly", "assembly_level", "not-a-level"),
    ],
)
def test_fetch_enum_models_reject_invalid_values(operation, field, value, tmp_path):
    parameters = {"gene": "audit", "taxon": "audit", "dest": str(tmp_path), field: value}
    binding = ov.operations.registry.require(operation)
    parameters = {
        k: v for k, v in parameters.items() if k in binding.signature.parameter_model.model_fields
    }
    with pytest.raises(OrganelleParameterError):
        binding.invoke(None, parameters)
    function = ov.operations.resolve_public_function(operation)
    with pytest.raises(OrganelleParameterError):
        function(**parameters)


def test_executor_is_passed_unchanged_and_not_exposed_as_json(tmp_path, monkeypatch):
    from organelleverse.phylogeny import phylo

    def executor(argv):
        return None

    class BoundaryReached(Exception):
        pass

    def boundary(*args, **kwargs):
        assert kwargs["executor"] is executor
        assert kwargs["bootstrap"] == 25
        raise BoundaryReached

    monkeypatch.setattr(phylo, "run_build_tree", boundary)
    with pytest.raises(BoundaryReached):
        ov.phylogeny.run_build_tree(tmp_path / "tiny.fa", bootstrap=25, executor=executor)
    schema = ov.operations.parameter_schema("phylogeny.run_build_tree")["properties"]
    assert {"bootstrap", "seed", "method", "threads", "model"} <= schema.keys()
    assert "executor" not in schema


def test_transport_is_passed_unchanged_without_network(tmp_path, monkeypatch):
    from organelleverse.fetch import ncbi

    transport = object()

    class BoundaryReached(Exception):
        pass

    def boundary(http, term):
        assert http is transport
        raise BoundaryReached

    monkeypatch.setattr(ncbi, "_esearch_history", boundary)
    with pytest.raises(BoundaryReached):
        ov.fetch.entrez_query(organelle="plastid", dest=tmp_path, transport=transport)
    assert "transport" not in ov.operations.parameter_schema("fetch.entrez_query")["properties"]


def test_fetch_defaults_and_nullable_release_match():
    accessions = ov.operations.parameter_schema("fetch.accession_list")
    assert "organelle" not in accessions.get("required", [])
    assert accessions["properties"]["organelle"]["default"] == "plastid"
    model = ov.operations.registry.require("fetch.refseq_snapshot").signature.parameter_model
    assert model.model_validate({"organelle": "plastid", "dest": "unused"}).release is None
    assert not ov.operations.describe("fetch.refseq_snapshot").deterministic


@pytest.mark.parametrize("as_string", [False, True])
def test_native_genome_map_keeps_documented_single_path_input(tmp_path, as_string):
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqFeature import SeqFeature, SimpleLocation
    from Bio.SeqRecord import SeqRecord

    path = tmp_path / "plastid.gb"
    record = SeqRecord(Seq("ACGT" * 20), id="test", name="test")
    record.annotations["molecule_type"] = "DNA"
    record.features.append(
        SeqFeature(
            SimpleLocation(0, 80), type="source", qualifiers={"organelle": ["plastid:chloroplast"]}
        )
    )
    SeqIO.write([record], path, "genbank")
    native_path = str(path) if as_string else path
    plot = ov.visualization.plot_genome_map(native_path)
    from organelleverse.visualization.plot_object import OrganellePlot

    assert isinstance(plot, OrganellePlot)


def test_non_callable_executor_is_rejected_before_execution(tmp_path):
    with pytest.raises(OrganelleParameterError):
        ov.phylogeny.run_build_tree(tmp_path / "missing.fa", executor=object())


def test_native_filter_object_type_is_checked_before_network(tmp_path):
    from organelleverse.fetch import ncbi

    with patch.object(ncbi, "_esearch_history") as network:
        with pytest.raises(OrganelleParameterError):
            ov.fetch.entrez_query(organelle="plastid", dest=tmp_path, filters="invalid")
        network.assert_not_called()
