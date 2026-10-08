"""Three representative direct/Agent equivalence fixtures, one per result codec.

These are hand-authored test fixtures, not capability bundles admitted to any
registry or discovery directory: Capability Plan 02 Task 2 (discovery/
admission) has not started, so ``Bundles discovered`` must stay ``0/253`` and
``Registry operations`` must stay ``20`` regardless of this file's existence.

Each fixture proves, for one of the four ResultCodec values:

* the direct Python call is completely unchanged (same function, same
  signature, same return type, same value) — the Agent path never replaces
  it with a wrapper;
* ``bound.invoke`` reaches the *same* underlying callable (``bound.function
  is real_callable``) and produces the same scientific conclusion, projected
  through the declared codec into a canonical ``OrganelleResult``.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

from organelleverse.capabilities.models import CapabilityBundle
from organelleverse.composition import gc_core
from organelleverse.core.result import OrganelleResult
from organelleverse.operations import ExecutionMode
from organelleverse.operations.python_binding import bind_python_capability
from organelleverse.operations.spec import (
    ArgumentMode,
    ParameterBindingSpec,
    ParameterCodec,
    ParameterSource,
    PythonBindingSpec,
    ResultCodec,
)
from organelleverse.phenotype.cms.pipeline import cms
from organelleverse.visualization.ogdraw import plot_ogdraw_map
from organelleverse.visualization.plot_object import OrganellePlot

FIXTURE_GENBANK = Path(__file__).resolve().parents[1] / "data" / "mito.gbk"

# compute_gc_content's own signature (composition/gc_core.py, out of this
# task's scope to change) returns a bare, unparameterized `dict`; naming its
# real type explicitly here, once, keeps that inherent gap from silently
# propagating "Unknown" through every downstream use in this file.
compute_gc_content: Callable[..., dict[str, object]] = gc_core.compute_gc_content  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle(*, operation_id: str, callable_locator: str, binding: PythonBindingSpec) -> CapabilityBundle:
    return CapabilityBundle.model_validate(
        {
            "schema": "organelleverse.capability.v1",
            "capability": {
                "id": operation_id,
                "bundle_version": "1.0.0",
                "implementation": "native",
            },
            "contract": {
                "operation_id": operation_id,
                "contract_version": "1.0",
                "title": "Test-only representative equivalence fixture",
                "description": (
                    "Hand-authored fixture bundle for a direct/Agent equivalence "
                    "test; never admitted to any registry or discovery directory."
                ),
                "keywords": ("demo", "equivalence", "fixture"),
                "execution_mode": ExecutionMode.INLINE,
                "stage": "analyze",
                "input_kind": "none",
                "output_kind": "result",
                "callable_locator": callable_locator,
                "binding": binding,
            },
        }
    )


# --- json_metric: composition.compute_gc_content -----------------------------


def _write_fasta(path: Path) -> None:
    sequence = "ACGTGGCCAATTGGCCTTAAGGCCTTAAGGCCAATTGGCCTTAAGGCCAATTGGCCTTAA" * 10
    path.write_text(f">demo_contig\n{sequence}\n", encoding="utf-8")


def test_compute_gc_content_equivalence(tmp_path: Path) -> None:
    fasta_path = tmp_path / "genome.fasta"
    _write_fasta(fasta_path)

    bundle = _bundle(
        operation_id="composition.compute_gc_content",
        callable_locator="organelleverse.composition.gc_core:compute_gc_content",
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="fasta_path", codec=ParameterCodec.PATH, source=ParameterSource.AGENT
                ),
                ParameterBindingSpec(
                    name="window_size", codec=ParameterCodec.JSON, source=ParameterSource.AGENT
                ),
            ),
            result_codec=ResultCodec.JSON_METRIC,
            result_key="gc_content_report",
        ),
    )
    bound = bind_python_capability(bundle, compute_gc_content, frozen_schema=None)

    # The Python call is completely unchanged: same function, same signature.
    assert bound.function is compute_gc_content
    assert inspect.signature(bound.function) == inspect.signature(compute_gc_content)

    direct = compute_gc_content(fasta_path, window_size=100)
    assert type(direct) is dict

    agent = bound.invoke(None, {"fasta_path": str(fasta_path), "window_size": 100})
    assert type(agent) is OrganelleResult

    from organelleverse.core.frozen import thaw_json

    assert thaw_json(agent.metrics["gc_content_report"]) == direct


# --- legacy_result: phenotype.cms.pipeline.cms --------------------------------


def test_cms_pipeline_equivalence(tmp_path: Path) -> None:
    fasta_path = tmp_path / "mito.fasta"
    _write_fasta(fasta_path)

    bundle = _bundle(
        # operation_id must be exactly two dot-separated segments; "phenotype.cms"
        # is what cms() itself already stamps into every OrganelleResult it
        # returns (see phenotype/cms/pipeline.py). The module path
        # phenotype.cms.pipeline lives in callable_locator, not operation_id.
        operation_id="phenotype.cms",
        callable_locator="organelleverse.phenotype.cms.pipeline:cms",
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="genome_fasta", codec=ParameterCodec.PATH, source=ParameterSource.AGENT
                ),
            ),
            result_codec=ResultCodec.LEGACY_RESULT,
        ),
    )
    bound = bind_python_capability(bundle, cms, frozen_schema=None)

    assert bound.function is cms
    assert inspect.signature(bound.function) == inspect.signature(cms)

    direct = cms(fasta_path)
    assert type(direct) is OrganelleResult  # cms() already returns canonical OrganelleResult

    agent = bound.invoke(None, {"genome_fasta": str(fasta_path)})
    assert type(agent) is OrganelleResult

    # Same scientific conclusion: legacy_result revalidates rather than
    # reinterpreting, so every field survives unchanged except operation_id
    # (stamped from the bundle's contract, not the direct call's own choice).
    assert agent.status == direct.status
    assert agent.summary_text == direct.summary_text
    assert agent.metrics == direct.metrics
    assert agent.findings == direct.findings
    assert agent.flags == direct.flags


# --- artifact: visualization.plot_ogdraw_map ----------------------------------


def test_plot_ogdraw_map_equivalence() -> None:
    if not FIXTURE_GENBANK.is_file():  # pragma: no cover - repository fixture guard
        pytest.skip(f"missing GenBank fixture: {FIXTURE_GENBANK}")

    bundle = _bundle(
        operation_id="visualization.plot_ogdraw_map",
        callable_locator="organelleverse.visualization.ogdraw:plot_ogdraw_map",
        binding=PythonBindingSpec(
            argument_mode=ArgumentMode.NAMED_PARAMETERS,
            parameters=(
                ParameterBindingSpec(
                    name="genbank_path", codec=ParameterCodec.PATH, source=ParameterSource.AGENT
                ),
            ),
            result_codec=ResultCodec.ARTIFACT,
        ),
    )
    bound = bind_python_capability(bundle, plot_ogdraw_map, frozen_schema=None)

    assert bound.function is plot_ogdraw_map
    assert inspect.signature(bound.function) == inspect.signature(plot_ogdraw_map)

    direct = plot_ogdraw_map(FIXTURE_GENBANK)
    assert type(direct) is OrganellePlot  # compute-only: no file written yet

    agent = bound.invoke(None, {"genbank_path": str(FIXTURE_GENBANK)})
    assert type(agent) is OrganelleResult
    assert agent.status == "ok"
    assert len(agent.artifacts) >= 1

    # Artifact digest is recomputable from the actually-written file content.
    primary = agent.artifacts[0]
    assert _sha256(Path(primary.uri)) == primary.sha256


def test_none_of_the_three_fixtures_are_registered() -> None:
    """Bundles discovered stays 0/253 and Registry operations stays 20."""
    import organelleverse as ov

    registered_ids = {spec.operation_id for spec in ov.operations.list()}
    assert len(registered_ids) == 20
    assert "composition.compute_gc_content" not in registered_ids
    assert "phenotype.cms" not in registered_ids
    assert "visualization.plot_ogdraw_map" not in registered_ids
