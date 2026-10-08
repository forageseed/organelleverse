"""Capability Plan 03, Task 3: the restored ``localization`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 9 bundles under
``src/organelleverse/capabilities/localization-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.localization``'s module docstring
for the full per-capability reasoning.

Mirrors ``tests/capabilities/test_restored_format_conversion_bundles.py``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult
from tests._paths import PROJECT_ROOT
from tests.capabilities._bundle_regeneration_check import (
    assert_regenerated_domain_matches_committed,
)

_CHECK_ALL_BACKENDS_ID = "localization.check_all_backends"
_CHECK_BACKEND_ID = "localization.check_backend"
_INSTALL_HINT_ID = "localization.install_hint"
_PREDICT_ID = "localization.predict"
_SCORE_NLS_ID = "localization.score_nls"
_SCORE_PTS_ID = "localization.score_pts"
_SCORE_SIGNAL_PEPTIDE_ID = "localization.score_signal_peptide"
_SCORE_TRANSIT_PEPTIDES_ID = "localization.score_transit_peptides"
_SCORE_TRANSMEMBRANE_ID = "localization.score_transmembrane"
_RESTORED_IDS = (
    _CHECK_ALL_BACKENDS_ID,
    _CHECK_BACKEND_ID,
    _INSTALL_HINT_ID,
    _PREDICT_ID,
    _SCORE_NLS_ID,
    _SCORE_PTS_ID,
    _SCORE_SIGNAL_PEPTIDE_ID,
    _SCORE_TRANSIT_PEPTIDES_ID,
    _SCORE_TRANSMEMBRANE_ID,
)
_EXCLUDED_IDS = ("localization.predict_heuristic",)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "localization-check-all-backends",
    "localization-check-backend",
    "localization-install-hint",
    "localization-predict",
    "localization-score-nls",
    "localization-score-pts",
    "localization-score-signal-peptide",
    "localization-score-transit-peptides",
    "localization-score-transmembrane",
}

# A plant chloroplast transit-peptide-like N-terminus for a real, non-trivial score.
_TP_LIKE_SEQ = "MASTAVSAASSAFAGKAVKLSPSASELASRSSRRLVVRA" + "AKPTVLDSSSVASLSA" * 3


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


def test_the_generator_produces_exactly_these_ids_and_skips_the_rest(tmp_path: Path) -> None:
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="localization",
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


def test_check_backend_and_install_hint_resolve_and_invoke_real_json_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)

    check_binding = admitted.binding_source().resolve(_CHECK_BACKEND_ID)
    assert check_binding is not None
    check_result = check_binding.invoke(None, {"name": "deeploc"})
    assert isinstance(check_result, OrganelleResult)
    assert check_result.status == "ok"
    assert check_result.metrics["backend_status"]["name"] == "deeploc"

    hint_binding = admitted.binding_source().resolve(_INSTALL_HINT_ID)
    assert hint_binding is not None
    hint_result = hint_binding.invoke(None, {"name": "deeploc"})
    assert isinstance(hint_result, OrganelleResult)
    assert hint_result.status == "ok"
    assert isinstance(hint_result.metrics["hint_text"], str)


def test_check_all_backends_resolves_and_invokes_with_zero_agent_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_CHECK_ALL_BACKENDS_ID)
    assert binding is not None

    result = binding.invoke(None, {})

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert "deeploc" in result.metrics["backend_status"]


def test_predict_resolves_and_invokes_the_heuristic_backend_on_a_real_fasta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_PREDICT_ID)
    assert binding is not None

    fasta_path = tmp_path / "proteins.fasta"
    fasta_path.write_text(f">protein1\n{_TP_LIKE_SEQ}\n")

    result = binding.invoke(None, {"fasta": str(fasta_path)})

    assert isinstance(result, OrganelleResult)
    assert result.status in {"ok", "warning"}
    assert result.metrics["n_proteins"] == 1


def test_all_five_score_functions_resolve_and_invoke_real_json_scalar_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    cases = (
        (_SCORE_NLS_ID, "nls"),
        (_SCORE_PTS_ID, "pts"),
        (_SCORE_SIGNAL_PEPTIDE_ID, "signal_peptide"),
        (_SCORE_TRANSIT_PEPTIDES_ID, "transit_peptides"),
        (_SCORE_TRANSMEMBRANE_ID, "transmembrane"),
    )
    for capability_id, result_key in cases:
        binding = admitted.binding_source().resolve(capability_id)
        assert binding is not None
        result = binding.invoke(None, {"seq": _TP_LIKE_SEQ})
        assert isinstance(result, OrganelleResult)
        assert result.operation_id == capability_id
        assert result.status == "ok"
        assert result_key in result.metrics
