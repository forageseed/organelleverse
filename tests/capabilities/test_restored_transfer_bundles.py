"""Capability Plan 03, Task 3: the restored ``transfer`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated 4 bundles under
``src/organelleverse/capabilities/transfer-*`` from
``docs/operations/restored-capabilities.toml`` - see
``organelleverse.capabilities.adapters.transfer``'s module docstring for the
full per-capability reasoning.

``transfer.detect_transfers_blast`` is a second genuinely ungated real
external-tool call, beyond ``phylogeny.align``: it has no ``executor``
parameter and spawns ``makeblastdb``/``blastn`` directly. This environment
has both installed, so the invoke test below runs the real subprocess
pipeline.

Mirrors ``tests/capabilities/test_restored_format_conversion_bundles.py``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
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

_COMPUTE_TRANSFER_ID = "transfer.compute_transfer"
_DETECT_MTPT_ID = "transfer.detect_mtpt"
_DETECT_TRANSFERS_BLAST_ID = "transfer.detect_transfers_blast"
_DETECT_TRANSFERS_EVIDENCE_ID = "transfer.detect_transfers_evidence"
_RESTORED_IDS = (
    _COMPUTE_TRANSFER_ID,
    _DETECT_MTPT_ID,
    _DETECT_TRANSFERS_BLAST_ID,
    _DETECT_TRANSFERS_EVIDENCE_ID,
    "transfer.validate_transfers_depth",
    "transfer.validate_transfers_longread",
)
_EXCLUDED_IDS = (
    "transfer.annotate_nuclear_locus",
    "transfer.annotate_organelle_genes",
    "transfer.detect",
    "transfer.write_fragments",
)

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "transfer-compute-transfer",
    "transfer-detect-mtpt",
    "transfer-detect-transfers-blast",
    "transfer-detect-transfers-evidence",
    "transfer-validate-transfers-depth",
    "transfer-validate-transfers-longread"
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


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in records))


def test_the_four_bindable_capabilities_are_discovered_from_the_real_core_channel(
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
        domain="transfer",
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


def test_detect_transfers_blast_resolves_and_really_invokes_blastn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The domain's ungated real external-tool call: makeblastdb + blastn.

    Real BLASTN/makeblastdb are installed in this environment, so this
    asserts on a real subprocess pipeline outcome, not a degraded plan-only
    path - either a real hit set or a real empty-hit result, whichever
    BLASTN actually reports, not a fabricated one.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_DETECT_TRANSFERS_BLAST_ID)
    assert binding is not None

    nuclear_path = tmp_path / "nuclear.fasta"
    organelle_path = tmp_path / "organelle.fasta"
    shared = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" * 3
    _write_fasta(nuclear_path, [("nuc1", f"GGGG{shared}TTTT")])
    _write_fasta(organelle_path, [("mito1", shared)])

    result = binding.invoke(
        None, {"nuclear_fasta": str(nuclear_path), "organelle_fasta": str(organelle_path)}
    )

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == "transfer.detect_transfers_blast"
    from organelleverse._losat import resolve_losat

    # LOSAT-first: the vendored LOSAT alone is enough; NCBI needs both tools.
    ncbi = shutil.which("blastn") is not None and shutil.which("makeblastdb") is not None
    if resolve_losat() is not None or ncbi:
        assert result.status == "ok"
    else:  # pragma: no cover - LOSAT is vendored in this checkout
        assert result.status == "failed"


def test_detect_mtpt_and_compute_transfer_resolve_and_invoke_a_real_kmer_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    mito_path = tmp_path / "mito.fasta"
    cp_path = tmp_path / "cp.fasta"
    shared = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    _write_fasta(mito_path, [("mito1", f"GGGG{shared}TTTT")])
    _write_fasta(cp_path, [("cp1", shared)])

    mtpt_binding = admitted.binding_source().resolve(_DETECT_MTPT_ID)
    assert mtpt_binding is not None
    mtpt_result = mtpt_binding.invoke(
        None, {"mito_fasta": str(mito_path), "cp_fasta": str(cp_path)}
    )
    assert isinstance(mtpt_result, OrganelleResult)
    assert mtpt_result.operation_id == _DETECT_MTPT_ID
    assert mtpt_result.status == "ok"

    compute_binding = admitted.binding_source().resolve(_COMPUTE_TRANSFER_ID)
    assert compute_binding is not None
    compute_result = compute_binding.invoke(
        None, {"target_fasta": str(mito_path), "source_fasta": str(cp_path)}
    )
    assert isinstance(compute_result, OrganelleResult)
    assert compute_result.status == "ok"
    assert "transfer_fragments" in compute_result.metrics


def test_detect_transfers_evidence_resolves_and_invokes_with_only_the_required_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the single-arm ``str | Path | OrganelleData`` PATH codec works.

    Every other parameter (including ``organelle_fasta``) has a default and
    is left uncovered; a real call with only ``nuclear_fasta`` degrades
    gracefully rather than raising.
    """
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_DETECT_TRANSFERS_EVIDENCE_ID)
    assert binding is not None

    nuclear_path = tmp_path / "nuclear.fasta"
    _write_fasta(nuclear_path, [("nuc1", "ACGTACGTACGTACGTACGTACGT")])

    result = binding.invoke(None, {"nuclear_fasta": str(nuclear_path)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == "transfer.detect_transfers_evidence"
