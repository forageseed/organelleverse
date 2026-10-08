from __future__ import annotations

import ast
import builtins
import inspect
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Never

import pytest
from pytest import MonkeyPatch

import organelleverse.operations as operations
from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
    operation,
)




def test_operations_import_is_lightweight(tmp_path: Path) -> None:
    script = """
import builtins
import json
import pathlib
import socket
import subprocess
import sys

def fail_io(*args, **kwargs):
    raise AssertionError("release catalog discovery must not perform I/O")

subprocess.run = fail_io
subprocess.Popen = fail_io
socket.create_connection = fail_io
builtins.open = fail_io
pathlib.Path.mkdir = fail_io
pathlib.Path.write_text = fail_io
heavy = ["Bio", "tensorflow", "matplotlib", "skimage", "torch", "gradio"]
before_operations = set(sys.modules)
import organelleverse.operations as operations
listed = operations.list()
described = operations.describe("annotation.annotate")
schema = operations.parameter_schema("annotation.annotate")
invocation_schema = operations.invocation_schema("annotation.annotate")
print(json.dumps({
    "loaded": [name for name in heavy if name in sys.modules and name not in before_operations],
    "operation_ids": [spec.operation_id for spec in listed],
    "described": described.operation_id,
    "schema_backend": schema["properties"]["backend"]["enum"],
    "invocation_operation_id": invocation_schema["properties"]["operation_id"]["const"],
    "research_loaded": any(name.startswith("organelleverse.annotation.research") for name in sys.modules),
    "assembly_service_loaded": "organelleverse.assembly.service" in sys.modules,
    "qc_service_loaded": "organelleverse.quality_control.service" in sys.modules,
    "qc_annotation_service_loaded": "organelleverse.quality_control.annotation_service" in sys.modules,
    "qc_writer_loaded": "organelleverse.quality_control.writer" in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )

    assert json.loads(completed.stdout) == {
        "described": "annotation.annotate",
        "loaded": [],
        "assembly_service_loaded": False,
        "qc_service_loaded": False,
        "qc_annotation_service_loaded": False,
        "qc_writer_loaded": False,
        "invocation_operation_id": "annotation.annotate",
        "operation_ids": [
            "annotation.annotate",
            "annotation.extract",
            "annotation.write",
            "assembly.assemble",
            "assembly.pmat_graph_build",
            "assembly.write",
            "fetch.accession_list",
            "fetch.dedup_genomes",
            "fetch.entrez_query",
            "fetch.gene_records",
            "fetch.genome_size_candidates",
            "fetch.ngdc_gwh",
            "fetch.nuclear_assembly",
            "fetch.refseq_snapshot",
            "io.read_fasta_genome",
            "io.read_genbank_genome",
            "io.read_long_reads",
            "qc.annotation",
            "qc.assembly",
            "qc.write",
        ],
        "research_loaded": False,
        "schema_backend": ["auto", "mitochondrion", "plastome"],
    }


def test_registry_has_no_module_scope_capabilities_import() -> None:
    operations_root = Path(__file__).resolve().parents[2] / "src" / "organelleverse" / "operations"
    path = operations_root / "registry.py"
    offenders: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names = (node.module or "",)
        else:
            continue
        if any(name.startswith("organelleverse.capabilities") for name in names):
            offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_assembly_data_contract_import_is_lazy_and_public_facade_is_cached(
    tmp_path: Path,
) -> None:
    script = """
import importlib
import json
import sys

import organelleverse.assembly.data_contract as data_contract
import organelleverse.assembly as assembly

blocked = [
    "organelleverse.assembly.install",
    "organelleverse.assembly.routing",
    "organelleverse.assembly.backends.registry",
]
loaded_after_contract_import = [name for name in blocked if name in sys.modules]
sources = {
    "AssemblyAuxiliary": "organelleverse.assembly.contracts",
    "AssemblyEnvironmentHint": "organelleverse.assembly.api",
    "AssemblyInputPayload": "organelleverse.assembly.contracts",
    "AssemblyRoute": "organelleverse.assembly.routing",
    "ContigInput": "organelleverse.assembly.contracts",
    "GenomeSizeEvidence": "organelleverse.assembly.contracts",
    "LongReadLibrary": "organelleverse.assembly.contracts",
    "ShortReadLibrary": "organelleverse.assembly.contracts",
    "assemble": "organelleverse.assembly.api",
    "check_all_backends": "organelleverse.assembly.install",
    "check_backend": "organelleverse.assembly.install",
    "classify_profile": "organelleverse.assembly.routing",
    "compatible_backends": "organelleverse.assembly.routing",
    "install_backend": "organelleverse.assembly.install",
    "pmat_continue": "organelleverse.assembly.api",
    "pmat_graph_build": "organelleverse.assembly.api",
    "resolve_backend": "organelleverse.assembly.routing",
    "write": "organelleverse.assembly.api",
}
identities = {}
cached = {}
for name, module_name in sources.items():
    value = getattr(assembly, name)
    identities[name] = value is getattr(importlib.import_module(module_name), name)
    cached[name] = vars(assembly).get(name) is value and getattr(assembly, name) is value

print(json.dumps({
    "contract_modality": data_contract.SEQUENCING_READS_DATA_CONTRACT.modality,
    "loaded_after_contract_import": loaded_after_contract_import,
    "public_symbols": assembly.__all__,
    "identities": identities,
    "cached": cached,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    result = json.loads(completed.stdout)

    assert result["contract_modality"] == "sequencing_reads"
    assert result["loaded_after_contract_import"] == []
    assert result["public_symbols"] == [
        "AssemblyAuxiliary",
        "AssemblyEnvironmentHint",
        "AssemblyInputPayload",
        "AssemblyRoute",
        "ContigInput",
        "GenomeSizeEvidence",
        "LongReadLibrary",
        "ShortReadLibrary",
        "assemble",
        "check_all_backends",
        "check_backend",
        "classify_profile",
        "compatible_backends",
        "install_backend",
        "pmat_continue",
        "pmat_graph_build",
        "resolve_backend",
        "write",
    ]
    assert all(result["identities"].values())
    assert all(result["cached"].values())


def test_root_canonical_modules_are_lazy_and_cached(tmp_path: Path) -> None:
    script = """
import importlib
import json
import sys

import organelleverse as ov

module_names = ["annotation", "assembly", "environments", "fetch", "io", "operations", "qc", "report"]
initially_loaded = [name for name in module_names if f"organelleverse.{name}" in sys.modules]
module_identities = {}
for name in module_names:
    value = getattr(ov, name)
    module_identities[name] = (
        value is importlib.import_module(f"organelleverse.{name}")
        and vars(ov)[name] is value
    )

print(json.dumps({
    "initially_loaded": initially_loaded,
    "modules": module_identities,
    "read": ov.read is importlib.import_module("organelleverse.io").read,
    "write": ov.write is importlib.import_module("organelleverse.writer").write,
    "save_forbidden": not hasattr(ov, "save"),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    result = json.loads(completed.stdout)

    assert result["initially_loaded"] == []
    assert all(result["modules"].values())
    assert result["read"] is True
    assert result["write"] is True
    assert result["save_forbidden"] is True


def test_short_aliases_are_lazy_cached_and_route_to_the_same_module(
    tmp_path: Path,
) -> None:
    """The 20 short human-facing names (``ov.viz``, ``ov.erc``, ...) are a
    routing layer over ``_PUBLIC_MODULES``, not a second implementation.

    Each alias must: stay unimported until first touched (laziness), resolve
    to the exact same module object as its full name (no second module, no
    second contract), be cached on the second access (no repeated
    ``import_module`` call), and leave the Registry's operation count and IDs
    completely unaffected (no second ``OperationSpec``, no second Registry
    entry created merely by resolving a short name).
    """
    script = """
import importlib
import json
import sys

import organelleverse as ov

aliases = dict(ov._SHORT_MODULE_ALIASES)

# Captured before anything else touches the registry or any alias: building
# the core catalog's signatures (operations.list()) legitimately imports
# annotation/assembly as a side effect of the existing (pre-bundle) catalog,
# which is unrelated to alias laziness and must not contaminate this check.
initially_loaded = sorted(
    short for short, full in aliases.items() if f"organelleverse.{full}" in sys.modules
)

import organelleverse.operations as operations

before_operation_ids = sorted(spec.operation_id for spec in operations.list())

identities = {}
cached = {}
for short, full in aliases.items():
    first = getattr(ov, short)
    second = getattr(ov, short)
    identities[short] = (
        first is getattr(ov, full)
        and first is importlib.import_module(f"organelleverse.{full}")
    )
    cached[short] = second is first and vars(ov)[short] is first

after_operation_ids = sorted(spec.operation_id for spec in operations.list())

print(json.dumps({
    "alias_count": len(aliases),
    "initially_loaded": initially_loaded,
    "identities": identities,
    "cached": cached,
    "before_operation_ids": before_operation_ids,
    "after_operation_ids": after_operation_ids,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    result = json.loads(completed.stdout)

    assert result["alias_count"] == 20
    assert result["initially_loaded"] == []
    assert all(result["identities"].values())
    assert all(result["cached"].values())
    # Resolving every short alias must not register a second OperationSpec
    # or a second Registry ID: the catalog is exactly what it was before.
    assert result["after_operation_ids"] == result["before_operation_ids"]
    assert len(result["after_operation_ids"]) == 20


def test_root_facade_lazily_exposes_operations_in_a_clean_process(
    tmp_path: Path,
) -> None:
    script = """
import json
import sys

import organelleverse as ov

initially_loaded = "organelleverse.operations" in sys.modules
first = ov.operations
second = ov.operations
print(json.dumps({
    "initially_loaded": initially_loaded,
    "cached": first is second and vars(ov).get("operations") is first,
    "listed": [spec.operation_id for spec in first.list()],
    "public": "operations" in ov.__all__,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    result = json.loads(completed.stdout)

    assert result["initially_loaded"] is False
    assert result["cached"] is True
    assert "annotation.annotate" in result["listed"]
    assert result["public"] is True


def test_package_exports_registry_api_with_explicit_signatures() -> None:
    expected = {
        "list": ("stage", "input_kind", "organelle", "input_modality", "output_modality"),
        "describe": ("operation_id",),
        "parameter_schema": ("operation_id",),
        "invocation_schema": ("operation_id",),
        "check_dependencies": ("operation_id",),
        "invoke": ("operation_id", "input", "parameters"),
    }

    for name, parameters in expected.items():
        api = getattr(operations, name)
        signature = inspect.signature(api)
        assert callable(api)
        assert tuple(signature.parameters) == parameters
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )


def test_maturity_is_not_a_public_operation_symbol() -> None:
    assert "Maturity" not in operations.__all__
    assert not hasattr(operations, "Maturity")


def test_discovery_returns_sorted_json_safe_metadata_without_io(
    monkeypatch: MonkeyPatch,
) -> None:
    local_registry = OperationRegistry()
    _register_read_operation(local_registry, operation_id="io.zeta")
    _register_read_operation(local_registry, operation_id="io.alpha")
    monkeypatch.setattr(operations, "registry", local_registry)

    def fail_io(*args: object, **kwargs: object) -> Never:
        raise AssertionError("discovery must not perform I/O")

    monkeypatch.setattr(builtins, "open", fail_io)
    monkeypatch.setattr(socket, "socket", fail_io)
    monkeypatch.setattr(subprocess, "Popen", fail_io)
    monkeypatch.setattr(subprocess, "run", fail_io)

    listed = operations.list()
    metadata = [spec.model_dump(mode="json") for spec in listed]

    assert [spec.operation_id for spec in listed] == ["io.alpha", "io.zeta"]
    assert json.loads(json.dumps(metadata)) == metadata
    assert operations.describe("io.alpha") is listed[0]
    assert operations.parameter_schema("io.alpha")["additionalProperties"] is False


def test_lazy_release_catalog_failure_is_atomic_and_retryable(
    monkeypatch: MonkeyPatch,
    tmp_path,
) -> None:
    from organelleverse.operations import catalog
    from organelleverse.operations.registry import (
        _ReleaseOperationRegistry,  # pyright: ignore[reportPrivateUsage]
    )

    # Isolate the capability source: the 16 bundle-migrated release operations
    # are admitted only with verification records, which an empty home denies.
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "empty-home"))
    attempts = 0

    def load(target: OperationRegistry) -> None:
        nonlocal attempts
        attempts += 1
        _register_catalog_stub(target, operation_id="io.release")
        if attempts == 1:
            raise RuntimeError("synthetic catalog failure")

    monkeypatch.setattr(catalog, "load_release_catalog", load)
    registry = _ReleaseOperationRegistry()

    with pytest.raises(RuntimeError, match="synthetic catalog failure"):
        registry.list()

    assert [spec.operation_id for spec in registry.list()] == ["io.release"]
    assert attempts == 2


def test_default_registry_reserves_release_ids_before_external_registration(
    monkeypatch: MonkeyPatch,
    tmp_path,
) -> None:
    from organelleverse.operations import catalog
    from organelleverse.operations.registry import (
        _ReleaseOperationRegistry,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "empty-home"))

    def load(target: OperationRegistry) -> None:
        _register_catalog_stub(target, operation_id="io.release")

    monkeypatch.setattr(catalog, "load_release_catalog", load)
    registry = _ReleaseOperationRegistry()

    with pytest.raises(OrganelleContractError) as duplicate:
        _register_catalog_stub(registry, operation_id="io.release")

    assert duplicate.value.code == "contract.duplicate_operation_id"
    assert [spec.operation_id for spec in registry.list()] == ["io.release"]


def _register_read_operation(registry: OperationRegistry, *, operation_id: str) -> None:
    spec = OperationSpec(
        operation_id=operation_id,
        contract_version="1.0",
        title="Demo read operation",
        description="Test fixture spec used only to probe discovery safety.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.READ,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.GENOME,
        callable_locator="organelleverse.io.api:read_genome",
    )

    @operation(spec=spec, registry=registry)
    def read_genome(path: str) -> OrganelleGenome:
        raise AssertionError("discovery must not invoke operations")

    assert registry.require(operation_id).function is read_genome


def _register_catalog_stub(registry: OperationRegistry, *, operation_id: str) -> None:
    spec = OperationSpec(
        operation_id=operation_id,
        contract_version="1.0",
        title="Demo catalog-stub read operation",
        description="Test fixture spec used only to probe discovery safety.",
        keywords=("demo", "fixture", "test"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.READ,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.GENOME,
        callable_locator="organelleverse.io:read_genome",
    )

    def read_genome(source: Path) -> OrganelleGenome:
        raise AssertionError("catalog loading must not invoke operations")

    registry.register(spec, read_genome)
