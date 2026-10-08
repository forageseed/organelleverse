"""G5.1: the zero-click default workflow's dependency closure.

The default workflow (docs/TASK-G-desktop-packaging.md):

    ov.read / fetch -> annotation.annotate -> qc.annotation -> ov.write
    morphology.segment (orgseg) -> measure

must run with zero user action, so every dependency it declares must live in
the bundled layer (layer 1). These tests pin that closure so a future
dependency addition to a default-chain operation fails loudly here instead
of silently breaking the zero-click promise on a user's machine.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.operations.registry import registry

#: The default workflow's operation ids (bundle ids for the suite ops).
DEFAULT_CHAIN_BUNDLES = (
    "io.read_fasta_genome",
    "fetch.accession_list",
    "qc.annotation",
    "annotation.write",
    "morphology.segment",
    "morphology.measure",
)
DEFAULT_CHAIN_CORE = ("annotation.annotate",)

#: Layer-1 (bundled) python dependencies — everything the default chain may
#: declare without any user action. CPU-only torch + the micro-sam minimal
#: closure join this set after the T-G5 measurement lands.
BUNDLED_PYTHON_DEPS = frozenset(
    {
        "numpy",
        "scipy",
        "skimage",
        "Bio",
        "pyhmmer",
        "PIL",
        "pandas",
        "seaborn",
        "allel",
        "matplotlib",
    }
)

#: The only non-python dependency the default chain may declare today:
#: BLAST+ executables (redistribution terms being verified — see the task
#: doc's layer table). Anything else fails this test.
ALLOWED_EXECUTABLE_DEPS = frozenset({"blastn", "makeblastdb", "tblastn"})


@pytest.fixture()
def admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".organelleverse" / "capabilities").mkdir(parents=True)
    home.mkdir()
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kw: ())
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)
    discovered = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for capability_id in DEFAULT_CHAIN_BUNDLES:
        verify_capability(
            capability_id,
            store=store,
            environment=LocalVerificationEnvironment(discovered),
        )
    return discover_capabilities()


def test_default_chain_operations_admit(admitted) -> None:
    for capability_id in DEFAULT_CHAIN_BUNDLES:
        entry = admitted.describe(capability_id)
        assert str(entry.status) == "admitted", f"{capability_id}: {entry.diagnostic}"


def test_default_chain_python_deps_stay_inside_the_bundled_layer(admitted) -> None:
    for capability_id in DEFAULT_CHAIN_BUNDLES:
        spec = admitted.binding_source().describe_spec(capability_id)
        assert spec is not None
        for dependency in spec.dependencies:
            if dependency.kind.value == "python":
                assert dependency.name in BUNDLED_PYTHON_DEPS, (
                    f"{capability_id} declares python dep {dependency.name!r} "
                    f"outside the bundled layer — the zero-click default chain "
                    f"may only use bundled dependencies"
                )
            elif dependency.kind.value == "executable":
                assert dependency.name in ALLOWED_EXECUTABLE_DEPS, (
                    f"{capability_id} declares executable dep {dependency.name!r}; "
                    f"only BLAST+ is allowed on the default chain"
                )

    for operation_id in DEFAULT_CHAIN_CORE:
        spec = registry.describe(operation_id)
        for dependency in spec.dependencies:
            if dependency.kind.value == "python":
                assert dependency.name in BUNDLED_PYTHON_DEPS, operation_id
            elif dependency.kind.value == "executable":
                assert dependency.name in ALLOWED_EXECUTABLE_DEPS, operation_id


def test_qc_annotation_does_not_need_pysam(admitted) -> None:
    """pysam is optional (no Windows wheel); the default chain must not need it."""
    spec = admitted.binding_source().describe_spec("qc.annotation")
    assert spec is not None
    assert all(d.name != "pysam" for d in spec.dependencies)


def test_default_segmentation_backend_remains_orgseg() -> None:
    """micro-SAM gives instance masks without semantic classes; the default
    stays orgseg (the four-class semantic model). Pin the auto route."""
    from organelleverse.morphology.orgseg import _route_backend

    assert _route_backend("auto", is_3d_stack=False) == "orgseg"
