"""Capability Plan 03, Task 3: the restored ``rna_editing`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 9 bundles under
``src/organelleverse/capabilities/rna-editing-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.rna_editing``'s module docstring for
the full reasoning, including a new blocker class: two ledger ids
(``rna_editing._deepred.score_cytidines`` / ``._plantc2u.score_cytidines``)
embed an underscore-prefixed submodule segment and so are not valid
``operation_id`` values at all under ``CapabilityBundle``'s schema pattern
(exactly one dot, every segment starting with a lowercase letter) -
confirmed these are the *only* two such ids across the entire 253-record
ledger.

Mirrors ``tests/capabilities/test_restored_trans_splicing_bundles.py``
(canonical_core + adapter mixed in one domain).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from tests._paths import PROJECT_ROOT
from tests.capabilities._bundle_regeneration_check import (
    assert_regenerated_domain_matches_committed,
)

_CHANGE_N_ID = "rna_editing.change_n"
_ENCODE_MATRIX_ID = "rna_editing.encode_matrix"
_EXTRACT_WINDOW_ID = "rna_editing.extract_window"
_EXTRACT_WINDOWS_ID = "rna_editing.extract_windows"
_PLANTC2U_MODEL_PATH_ID = "rna_editing.plantc2u_model_path"
_PREDICT_EDITS_ID = "rna_editing.predict_edits"
_PREDICT_EDITS_DEEPRED_ID = "rna_editing.predict_edits_deepred"
_PREDICT_EDITS_PLANTC2U_ID = "rna_editing.predict_edits_plantc2u"
_PREDICT_EDITS_PREP_ID = "rna_editing.predict_edits_prep"
_RESTORED_IDS = (
    _CHANGE_N_ID,
    _ENCODE_MATRIX_ID,
    _EXTRACT_WINDOW_ID,
    _EXTRACT_WINDOWS_ID,
    _PLANTC2U_MODEL_PATH_ID,
    _PREDICT_EDITS_ID,
    _PREDICT_EDITS_DEEPRED_ID,
    _PREDICT_EDITS_PLANTC2U_ID,
    _PREDICT_EDITS_PREP_ID,
    "rna_editing.validate_edits",
)
_EXCLUDED_IDS = (
    "rna_editing._deepred.score_cytidines",
    "rna_editing._plantc2u.score_cytidines",
    "rna_editing.deepredmt_model_path",
    "rna_editing.write_sites",
)

_MITO_GBK = PROJECT_ROOT / "tests" / "data" / "mito.gbk"

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "rna-editing-change-n",
    "rna-editing-encode-matrix",
    "rna-editing-extract-window",
    "rna-editing-extract-windows",
    "rna-editing-plantc2u-model-path",
    "rna-editing-predict-edits",
    "rna-editing-predict-edits-deepred",
    "rna-editing-predict-edits-plantc2u",
    "rna-editing-predict-edits-prep",
    "rna-editing-validate-edits"
}


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_restored_bundles", _GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _isolate_non_core_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".organelleverse" / "capabilities").mkdir(parents=True)
    home.mkdir()

    def no_entry_points(**kwargs: object) -> tuple[()]:
        return ()

    monkeypatch.setattr(importlib.metadata, "entry_points", no_entry_points)
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)
    return home


def _admit_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _isolate_non_core_roots(tmp_path, monkeypatch)
    discovered = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        verify_capability(
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
    return discover_capabilities()


def test_the_nine_bindable_capabilities_are_discovered_from_the_real_core_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)

    index = discover_capabilities()

    discovered_ids = {entry.capability_id for entry in index.entries}
    for capability_id in _RESTORED_IDS:
        entry = index.describe(capability_id)
        assert entry.origins[0].channel == "core"
        assert entry.execution_identity is None
    for excluded_id in _EXCLUDED_IDS:
        assert excluded_id not in discovered_ids


def test_the_generator_produces_exactly_these_ids_and_skips_the_rest(
    tmp_path: Path,
) -> None:
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="rna_editing",
        ledger_path=_LEDGER_PATH,
        current_root=_CURRENT_ROOT,
        output_root=isolated_output_root,
    )

    generated_ids = {item.capability_id for item in report.generated}
    assert generated_ids == set(_RESTORED_IDS)

    skip_codes = {item.capability_id: item.code for item in report.skipped}
    assert skip_codes == {
        excluded_id: "capability.no_adapter_override" for excluded_id in _EXCLUDED_IDS
    }

    generated_dirs = {item.path.parent.name for item in report.generated}
    assert generated_dirs == _EXPECTED_BUNDLE_DIRS

    assert_regenerated_domain_matches_committed(
        isolated_output_root, _BUNDLE_ROOT, _EXPECTED_BUNDLE_DIRS
    )


def test_every_generated_bundle_reaches_admitted_through_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _isolate_non_core_roots(tmp_path, monkeypatch)

    discovered = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        assert discovered.describe(capability_id).status is CapabilityStatus.REJECTED

    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        record = verify_capability(
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
        assert record.execution_identity is None
        assert record.worker_parameters == ()

    admitted = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        entry = admitted.describe(capability_id)
        assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic
    admitted_ids = {item.capability_id for item in admitted.list()}
    assert admitted_ids == set(_RESTORED_IDS)


def test_a_capability_the_generator_refused_to_produce_is_simply_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)
    index = discover_capabilities()

    for excluded_id in _EXCLUDED_IDS:
        with pytest.raises(Exception) as excinfo:
            index.describe(excluded_id)
        assert "unknown_capability" in str(excinfo.value) or "unknown capability" in str(
            excinfo.value
        )


def test_predict_edits_prep_resolves_and_invokes_the_canonical_core_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PREDICT_EDITS_PREP_ID)
    assert binding is not None

    genome = OrganelleGenome(
        organelle="mitochondrion",
        annotation=ArtifactRef.from_path(_MITO_GBK, kind="annotation", format="genbank"),
    )
    result = binding.invoke(genome, {})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _PREDICT_EDITS_PREP_ID
    assert result.status == "ok"


def test_change_n_and_extract_window_and_extract_windows_resolve_and_invoke_real_json_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    change_binding = admitted.binding_source().resolve(_CHANGE_N_ID)
    assert change_binding is not None
    change_result = change_binding.invoke(None, {"seq": "ACGTXQZ"})
    assert isinstance(change_result, OrganelleResult)
    assert change_result.status == "ok"
    assert change_result.metrics["sequence"] == "ACGTNNN"

    window_binding = admitted.binding_source().resolve(_EXTRACT_WINDOW_ID)
    assert window_binding is not None
    window_result = window_binding.invoke(
        None, {"sequence": "ACGTACGTACGTACGT", "position_1based": 5, "strand": 1}
    )
    assert isinstance(window_result, OrganelleResult)
    assert window_result.status == "ok"
    assert isinstance(window_result.metrics["window"], str)

    fin_path = tmp_path / "cds.fasta"
    fin_path.write_text(">gene1\nACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n")
    windows_binding = admitted.binding_source().resolve(_EXTRACT_WINDOWS_ID)
    assert windows_binding is not None
    windows_result = windows_binding.invoke(None, {"fin": str(fin_path)})
    assert isinstance(windows_result, OrganelleResult)
    assert windows_result.status == "ok"
    assert len(windows_result.metrics["windows"]) > 0


def test_encode_matrix_resolves_and_invokes_producing_a_real_numpy_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_ENCODE_MATRIX_ID)
    assert binding is not None

    result = binding.invoke(None, {"seqs": ["ACGTACGT", "TTTTGGGG"]})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.format == "npy"
    loaded = np.load(artifact.resolve())
    assert loaded.shape == (2, 180, 5)  # seq_len keeps its own default (PLANTC2U_WINDOW)


def test_plantc2u_model_path_resolves_and_invokes_a_real_packaged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves a zero-parameter, real-packaged-file artifact result works end to end."""
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PLANTC2U_MODEL_PATH_ID)
    assert binding is not None

    result = binding.invoke(None, {})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert len(result.artifacts) == 1
