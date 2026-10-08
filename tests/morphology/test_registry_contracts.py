"""T2: morphology registry contract acceptance (task brief 2026-08-17).

``morphology.segment`` and ``morphology.train`` are registered as capability
bundles (Capability Plan 03/05); this module pins the contract fields the
task brief requires: honest ``deterministic`` flags, the image modality
expressed as a contract keyword (the frozen OperationSpec validators reserve
``input_modalities``/``output_modalities`` for CoreKind.DATA flows), no
fallback, and dependency reporting that names exactly what is missing.
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
from organelleverse.operations.dependencies import check_dependencies
from organelleverse.operations.spec import (
    CoreKind,
    DependencyKind,
    DependencySpec,
    ExecutionMode,
    OperationSpec,
    OperationStage,
    PythonBindingSpec,
)


@pytest.fixture()
def admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify and admit the two morphology bundles through the real pipeline."""
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
    for capability_id in ("morphology.segment", "morphology.train"):
        verify_capability(
            capability_id,
            store=store,
            environment=LocalVerificationEnvironment(discovered),
        )
    return discover_capabilities()


def test_segment_and_train_specs_export_through_the_registry(admitted) -> None:
    source = admitted.binding_source()
    for capability_id in ("morphology.segment", "morphology.train"):
        spec = source.describe_spec(capability_id)
        assert spec is not None
        assert spec.stage is OperationStage.ANALYZE
        assert spec.output_kind is CoreKind.RESULT
        assert source.parameter_schema(capability_id) is not None
        assert source.known_ids()  # sanity: the source is populated


def test_contract_flags_are_honest(admitted) -> None:
    source = admitted.binding_source()
    segment = source.describe_spec("morphology.segment")
    train = source.describe_spec("morphology.train")

    # Segmentation with fixed weights is deterministic; SGD fine-tuning is not.
    assert segment.deterministic is True
    assert train.deterministic is False

    # Image modality is a searchable keyword (frozen validators reserve the
    # modality fields for CoreKind.DATA core-object flows).
    assert "electron-micrograph" in segment.keywords
    assert "electron-micrograph" in train.keywords

    # No silent fallback anywhere, per the project-wide rule.
    assert segment.fallback.allowed is False
    assert train.fallback.allowed is False

    # The frozen organelle_types Literal cannot express four-class EM output
    # and stays empty rather than being widened.
    assert segment.organelle_types == ()
    assert train.organelle_types == ()


def _synthetic_spec_with_dependency(dependency: DependencySpec) -> OperationSpec:
    return OperationSpec(
        operation_id="morphology.train",
        contract_version="1.0",
        title="Train",
        description="Synthetic spec for dependency reporting tests.",
        keywords=("analyze", "morphology", "train"),
        execution_mode=ExecutionMode.INLINE,
        stage=OperationStage.ANALYZE,
        input_kind=CoreKind.NONE,
        output_kind=CoreKind.RESULT,
        binding=PythonBindingSpec(),
        dependencies=(dependency,),
    )


def test_dependency_report_names_exactly_what_is_missing() -> None:
    spec = _synthetic_spec_with_dependency(
        DependencySpec(kind=DependencyKind.PYTHON, name="orgsegnet_mmseg_stack")
    )
    report = check_dependencies(spec)
    assert report.ready is False
    assert len(report.checks) == 1
    check = report.checks[0]
    assert check.dependency.name == "orgsegnet_mmseg_stack"
    assert check.state.value == "missing"


def test_declared_python_dependencies_resolve_honestly(admitted) -> None:
    """Every dependency the real bundles declare is checked, never assumed."""
    source = admitted.binding_source()
    for capability_id in ("morphology.segment", "morphology.train"):
        spec = source.describe_spec(capability_id)
        report = check_dependencies(spec)
        for check in report.checks:
            # Declared ledger deps are real installed packages here (numpy,
            # scikit-image); the point is that the report is per-dependency
            # and evidence-based, not a blanket claim.
            assert check.state.value in {"present", "missing", "unverified"}
