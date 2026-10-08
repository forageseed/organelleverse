"""Capability Plan 03, Task 3: the restored ``trans_splicing`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated one bundle
under ``src/organelleverse/capabilities/trans-splicing-detect-trans-splicing``
from ``docs/operations/restored-capabilities.toml``.
``trans_splicing.detect_trans_splicing`` matches the generator's generic
``canonical_core`` shape on its own (``genome: OrganelleGenome`` first
parameter, ``-> OrganelleResult``) - **this domain needs no adapter
module at all**, so none was created, per the charter's instruction not to
author an empty one for symmetry.

``trans_splicing.compute_trans_splicing`` (``trans_splicing_core.py``) takes
an untyped ``genome`` parameter (no annotation at all in its AST) - no codec
can honestly be chosen for an unannotated parameter, so it fails closed with
``capability.no_adapter_override`` exactly like every other capability with
no adapter override, whether or not an adapter module exists for the domain.

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
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from tests._paths import PROJECT_ROOT
from tests.capabilities._bundle_regeneration_check import (
    assert_regenerated_domain_matches_committed,
)

_DETECT_ID = "trans_splicing.detect_trans_splicing"
_RESTORED_IDS = (_DETECT_ID,)
_EXCLUDED_ID = "trans_splicing.compute_trans_splicing"

_CP_GBK = PROJECT_ROOT / "tests" / "data" / "cp.gbk"

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {"trans-splicing-detect-trans-splicing"}


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


def test_detect_trans_splicing_is_discovered_from_the_real_core_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)

    index = discover_capabilities()

    discovered_ids = {entry.capability_id for entry in index.entries}
    entry = index.describe(_DETECT_ID)
    assert entry.origins[0].channel == "core"
    assert entry.execution_identity is None
    assert _EXCLUDED_ID not in discovered_ids


def test_the_generator_produces_exactly_this_id_and_skips_compute_trans_splicing(
    tmp_path: Path,
) -> None:
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="trans_splicing",
        ledger_path=_LEDGER_PATH,
        current_root=_CURRENT_ROOT,
        output_root=isolated_output_root,
    )

    generated_ids = {item.capability_id for item in report.generated}
    assert generated_ids == set(_RESTORED_IDS)

    skip_codes = {item.capability_id: item.code for item in report.skipped}
    assert skip_codes == {
        _EXCLUDED_ID: "capability.excluded_by_domain_adapter"  # zero-out ruling
    }

    generated_dirs = {item.path.parent.name for item in report.generated}
    assert generated_dirs == _EXPECTED_BUNDLE_DIRS

    assert_regenerated_domain_matches_committed(
        isolated_output_root, _BUNDLE_ROOT, _EXPECTED_BUNDLE_DIRS
    )


def test_detect_trans_splicing_reaches_admitted_through_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _isolate_non_core_roots(tmp_path, monkeypatch)

    discovered = discover_capabilities()
    assert discovered.describe(_DETECT_ID).status is CapabilityStatus.REJECTED

    store = VerificationStore(home / "verifications")
    record = verify_capability(
        _DETECT_ID, store=store, environment=LocalVerificationEnvironment(discovered)
    )
    assert record.execution_identity is None
    assert record.worker_parameters == ()

    admitted = discover_capabilities()
    entry = admitted.describe(_DETECT_ID)
    assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic


def test_detect_trans_splicing_resolves_and_invokes_the_canonical_core_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _isolate_non_core_roots(tmp_path, monkeypatch)
    discovered = discover_capabilities()
    store = VerificationStore(home / "verifications")
    verify_capability(_DETECT_ID, store=store, environment=LocalVerificationEnvironment(discovered))
    admitted = discover_capabilities()

    binding = admitted.binding_source().resolve(_DETECT_ID)
    assert binding is not None

    genome = OrganelleGenome(
        organelle="plastid",
        annotation=ArtifactRef.from_path(_CP_GBK, kind="annotation", format="genbank"),
    )
    result = binding.invoke(genome, {})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _DETECT_ID
    assert result.status == "ok"


def test_compute_trans_splicing_the_generator_refused_to_produce_is_simply_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)
    index = discover_capabilities()

    with pytest.raises(Exception) as excinfo:
        index.describe(_EXCLUDED_ID)
    assert "unknown_capability" in str(excinfo.value) or "unknown capability" in str(excinfo.value)
