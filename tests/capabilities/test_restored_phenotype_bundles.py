"""Capability Plan 03, Task 3: the restored ``phenotype`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 5 bundles under
``src/organelleverse/capabilities/phenotype-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.phenotype``'s module docstring for
the full reasoning, including a newly discovered gap: a directory-valued
*result* (``cms_data_dir``/``cms_model_dir``) is just as unbindable by
``ResultCodec.ARTIFACT`` as a directory-valued *parameter* is by
``ParameterCodec.PATH`` - both hit the same ``is_file()`` preflight.

``phenotype.download_cms_sequences`` is generated and admitted here, but
deliberately never invoked for real: doing so would perform live NCBI
network I/O and overwrite the repository-committed
``phenotype/cms/data/cms_proteins.fasta`` fixture that
``cms_protein_fasta`` (also in this domain) reads.

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

_CMS_ID = "phenotype.cms"
_CMS_PROTEIN_FASTA_ID = "phenotype.cms_protein_fasta"
_CMS_REFERENCE_JSON_ID = "phenotype.cms_reference_json"
_COMPUTE_CMS_CANDIDATES_ID = "phenotype.compute_cms_candidates"
_DOWNLOAD_CMS_SEQUENCES_ID = "phenotype.download_cms_sequences"
_RESTORED_IDS = (
    _CMS_ID,
    _CMS_PROTEIN_FASTA_ID,
    _CMS_REFERENCE_JSON_ID,
    _COMPUTE_CMS_CANDIDATES_ID,
    _DOWNLOAD_CMS_SEQUENCES_ID,
)
_EXCLUDED_IDS = ("phenotype.cms_data_dir", "phenotype.cms_model_dir")

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "phenotype-cms",
    "phenotype-cms-protein-fasta",
    "phenotype-cms-reference-json",
    "phenotype-compute-cms-candidates",
    "phenotype-download-cms-sequences",
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


def test_the_five_bindable_capabilities_are_discovered_from_the_real_core_channel(
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
        domain="phenotype",
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


def test_cms_and_compute_cms_candidates_resolve_and_invoke_on_a_real_fasta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    genome_path = tmp_path / "mito.fasta"
    genome_path.write_text(">contig1\n" + "ATGGCTGCTGCTGCTGCTGCTGCTGCTGCTGCTTAA" * 4 + "\n")

    cms_binding = admitted.binding_source().resolve(_CMS_ID)
    assert cms_binding is not None
    cms_result = cms_binding.invoke(None, {"genome_fasta": str(genome_path)})
    assert isinstance(cms_result, OrganelleResult)
    assert cms_result.operation_id == _CMS_ID
    assert cms_result.status == "ok"

    compute_binding = admitted.binding_source().resolve(_COMPUTE_CMS_CANDIDATES_ID)
    assert compute_binding is not None
    compute_result = compute_binding.invoke(None, {"fasta_path": str(genome_path)})
    assert isinstance(compute_result, OrganelleResult)
    assert compute_result.status == "ok"
    assert "cms_candidates" in compute_result.metrics


def test_cms_protein_fasta_and_cms_reference_json_resolve_and_invoke_real_packaged_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves a zero-parameter, real-packaged-file artifact result works end to end."""
    admitted = _admit_all(tmp_path, monkeypatch)

    fasta_binding = admitted.binding_source().resolve(_CMS_PROTEIN_FASTA_ID)
    assert fasta_binding is not None
    fasta_result = fasta_binding.invoke(None, {})
    assert isinstance(fasta_result, OrganelleResult)
    assert fasta_result.status == "ok"
    assert len(fasta_result.artifacts) == 1
    assert fasta_result.artifacts[0].format == "fasta"

    json_binding = admitted.binding_source().resolve(_CMS_REFERENCE_JSON_ID)
    assert json_binding is not None
    json_result = json_binding.invoke(None, {})
    assert isinstance(json_result, OrganelleResult)
    assert json_result.status == "ok"
    assert len(json_result.artifacts) == 1
    assert json_result.artifacts[0].format == "json"
