"""Capability Plan 03, Task 3: the restored ``format_conversion`` bundles, for real.

``scripts/capabilities/build_restored_bundles.py`` generated the three
bundles under ``src/organelleverse/capabilities/format-conversion-*`` from
``docs/operations/restored-capabilities.toml`` - see the generator's and
``organelleverse.capabilities.adapters.format_conversion``'s module
docstrings for why ``format_conversion`` was chosen (smallest domain that
both declares a ``custom`` result_shape and actually admits a bundle under
it) and why ``format_conversion.write_conversion`` was deliberately not
generated (no codec can decode its ``OrganelleResult | OrganelleGenome``
parameter honestly).

Every test below drives the real ``discover_capabilities`` /
``verify_capability`` / ``admit_capabilities`` / ``binding_source().resolve``
/ ``invoke`` pipeline against those three *committed* bundles - the same
shape of proof ``tests/capabilities/test_core_admission.py`` established for
Task 1's synthetic fixture, now against real restored capabilities. None of
these tests construct a ``CapabilityEntry`` with ``status="admitted"``, a
``VerificationRecord``, or an ``execution_identity`` by hand.

Unlike ``test_core_admission.py``, discovery here is **not** routed to an
isolated ``tmp_path`` package root: ``importlib.resources.files("organelleverse")``
is left untouched, so the ``"core"`` channel resolves to the real, installed
``src/organelleverse/`` tree - exactly where the generator wrote these
bundles. Only the *other* standard roots (``ORGANELLEVERSE_HOME``, the
project-local ``.organelleverse/capabilities``, and package entry points)
are isolated, so a real verification store write and a real project
directory elsewhere on the machine can never leak into these tests.
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
from organelleverse.capabilities.parser import parse_capability_bundle
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

_CONVERT_ID = "format_conversion.convert"
_TO_FASTA_ID = "format_conversion.convert_genbank_to_fasta"
_TO_GFF3_ID = "format_conversion.convert_genbank_to_gff3"
_RESTORED_IDS = (_CONVERT_ID, _TO_FASTA_ID, _TO_GFF3_ID)
_EXCLUDED_ID = "format_conversion.write_conversion"

_CP_GBK = PROJECT_ROOT / "tests" / "data" / "cp.gbk"

_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "capabilities" / "build_restored_bundles.py"
_LEDGER_PATH = PROJECT_ROOT / "docs" / "operations" / "restored-capabilities.toml"
_CURRENT_ROOT = PROJECT_ROOT / "src" / "organelleverse"
_BUNDLE_ROOT = _CURRENT_ROOT / "capabilities"
_EXPECTED_BUNDLE_DIRS = {
    "format-conversion-convert",
    "format-conversion-convert-genbank-to-fasta",
    "format-conversion-convert-genbank-to-gff3",
}


def _load_generator() -> ModuleType:
    """Load ``build_restored_bundles.py`` as a standalone module, without a package.

    Mirrors ``tests/contracts/test_restored_inventory.py``'s ``_load_generator``
    for the sibling ledger-inventory generator: ``scripts/`` is a CLI-script
    directory, not an importable package, so this loads the file directly
    rather than importing ``scripts.capabilities.build_restored_bundles``.
    """
    spec = importlib.util.spec_from_file_location("build_restored_bundles", _GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _isolate_non_core_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every standard root except ``"core"``, which stays the real package.

    Mirrors ``test_core_admission.py``'s ``_isolate_standard_roots`` for the
    ``ORGANELLEVERSE_HOME``/project/entry-point isolation, but deliberately
    does **not** monkeypatch ``importlib.resources.files``: the whole point
    of this module is to prove the bundles actually committed under
    ``src/organelleverse/capabilities/`` are discoverable and admittable
    through the real "core" channel, not a synthetic stand-in for it.
    """
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


def test_the_three_bindable_capabilities_are_discovered_from_the_real_core_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate_non_core_roots(tmp_path, monkeypatch)

    index = discover_capabilities()

    discovered_ids = {entry.capability_id for entry in index.entries}
    for capability_id in _RESTORED_IDS:
        entry = index.describe(capability_id)
        assert entry.origins[0].channel == "core"
        assert entry.execution_identity is None
    # format_conversion.write_conversion was deliberately not generated: its
    # first parameter (OrganelleResult | OrganelleGenome) has no honest
    # codec, and no adapter override was authored to invent one.
    assert _EXCLUDED_ID not in discovered_ids


def test_the_generator_produces_exactly_these_ids_and_skips_write_conversion(
    tmp_path: Path,
) -> None:
    """Pin the generator's build report by id, not just by count.

    A count-only assertion ("3 generated") would stay green even if a future
    signature change made ``convert`` itself unbindable while some other,
    previously-skipped capability became bindable - the domain would silently
    become a *different* set of 3. This asserts the exact ids on both sides
    of the generator's fail-closed boundary, the exact skip reason code for
    the one capability it refuses to bind, and that the generated set is
    exactly what is committed on disk under
    ``src/organelleverse/capabilities/format-conversion-*`` - so a bundle
    deleted from the tree without regenerating, or regenerated with a
    different id, also fails here.
    """
    generator = _load_generator()
    isolated_output_root = tmp_path / "capabilities"

    report = generator.build_domain(
        domain="format_conversion",
        ledger_path=_LEDGER_PATH,
        current_root=_CURRENT_ROOT,
        output_root=isolated_output_root,
    )

    generated_ids = {item.capability_id for item in report.generated}
    assert generated_ids == set(_RESTORED_IDS)

    skip_codes = {item.capability_id: item.code for item in report.skipped}
    assert skip_codes == {_EXCLUDED_ID: "capability.no_adapter_override"}

    generated_dirs = {item.path.parent.name for item in report.generated}
    assert generated_dirs == _EXPECTED_BUNDLE_DIRS

    assert_regenerated_domain_matches_committed(
        isolated_output_root, _BUNDLE_ROOT, _EXPECTED_BUNDLE_DIRS
    )

    # Capability Plan 04 Task 1: the two genbank_path capabilities carry a
    # real, generator-captured fixture; ``convert`` is canonical_core
    # (genome-input), out of the fixture evaluator's supported
    # input_kind=none scope, so it carries none.
    fixture_cases_by_id = {
        item.capability_id: {
            fixture.case for fixture in parse_capability_bundle(item.path).fixtures
        }
        for item in report.generated
    }
    assert fixture_cases_by_id == {
        _CONVERT_ID: set(),
        _TO_FASTA_ID: {"basic"},
        _TO_GFF3_ID: {"basic"},
    }


def test_every_generated_bundle_reaches_admitted_through_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discover -> verify -> re-admit, for all three real restored bundles."""
    home = _isolate_non_core_roots(tmp_path, monkeypatch)

    discovered = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        assert discovered.describe(capability_id).status is CapabilityStatus.REJECTED

    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        record = verify_capability(
            capability_id,
            store=store,
            environment=LocalVerificationEnvironment(discovered),
        )
        # A pure-core native capability has no bundle-local code/ tree to
        # content-hash: its integrity is the installed distribution itself
        # (verification.py's _is_pure_core_native).
        assert record.execution_identity is None
        assert record.worker_parameters == ()

    admitted = discover_capabilities()
    for capability_id in _RESTORED_IDS:
        entry = admitted.describe(capability_id)
        assert entry.status is CapabilityStatus.ADMITTED, entry.diagnostic
    admitted_ids = {item.capability_id for item in admitted.list()}
    assert admitted_ids == set(_RESTORED_IDS)


def _admit_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = _isolate_non_core_roots(tmp_path, monkeypatch)
    discovered = discover_capabilities()
    store = VerificationStore(home / "verifications")
    for capability_id in _RESTORED_IDS:
        verify_capability(
            capability_id, store=store, environment=LocalVerificationEnvironment(discovered)
        )
    return discover_capabilities()


def test_convert_genbank_to_gff3_resolves_and_invokes_against_a_real_genbank_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_TO_GFF3_ID)
    assert binding is not None

    result = binding.invoke(None, {"genbank_path": str(_CP_GBK)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _TO_GFF3_ID
    assert result.status == "ok"
    # OrganelleResult.metrics freezes JSON to immutable form (FrozenMap): a
    # list return value round-trips as a tuple, not a list.
    gff3_lines = result.metrics["gff3_lines"]
    assert gff3_lines[0] == "##gff-version 3"
    assert any("CDS" in line for line in gff3_lines)


def test_convert_genbank_to_fasta_resolves_and_invokes_against_a_real_genbank_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_TO_FASTA_ID)
    assert binding is not None

    result = binding.invoke(None, {"genbank_path": str(_CP_GBK)})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _TO_FASTA_ID
    assert result.status == "ok"
    fasta_records = result.metrics["fasta_records"]
    assert len(fasta_records) == 1
    seqid, sequence = fasta_records[0]
    assert seqid == "CPTEST"
    assert sequence.startswith("ACGTACGTAC")


def test_convert_resolves_and_invokes_the_canonical_core_binding_on_a_real_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = _admit_all(tmp_path, monkeypatch)
    binding = admitted.binding_source().resolve(_CONVERT_ID)
    assert binding is not None

    genome = OrganelleGenome(
        organelle="plastid",
        annotation=ArtifactRef.from_path(_CP_GBK, kind="annotation", format="genbank"),
    )
    result = binding.invoke(genome, {})

    assert isinstance(result, OrganelleResult)
    assert result.operation_id == _CONVERT_ID
    assert result.status == "ok"
    assert result.metrics["records"] == 1
    files = result.metrics["files"]
    assert set(files) == {"gff3", "fasta"}


def test_a_capability_the_generator_refused_to_produce_is_simply_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``write_conversion`` never became a bundle, so it is unknown, not rejected.

    Distinguishes "the generator declined to emit a bundle at all" (no entry
    anywhere in the index) from "a bundle exists but was rejected" - the
    fail-closed behavior asked for is the former.
    """
    _isolate_non_core_roots(tmp_path, monkeypatch)
    index = discover_capabilities()

    with pytest.raises(Exception) as excinfo:
        index.describe(_EXCLUDED_ID)
    assert "unknown_capability" in str(excinfo.value) or "unknown capability" in str(excinfo.value)
