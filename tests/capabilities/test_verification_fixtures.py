"""``LocalVerificationEnvironment.evaluate_fixture``: the real, generic evaluator.

Decision 004 (``.superpowers/sdd/004-the-real-admission-ceiling.md``) measured
132 admitted capability bundles shipping zero fixtures, because
``evaluate_fixture`` failed closed for any fixture-bearing bundle
unconditionally. This file proves the real evaluator: a fixture is actually
run through the real implementation, its produced value is actually hashed
and compared against the fixture's recorded expectation, and a genuine
``EquivalenceEvidence`` records the real verdict - pass or fail.

Capability picked: ``composition.gc_content`` (one of the 132 admitted
bundles - see ``src/organelleverse/capabilities/composition-gc-content/``).
Deterministic, pure-Python (no external tool, no network, no subprocess),
and cheap: exactly the "stable and cheap to compare" shape the task asks
for.

Why the fixture bundle is built in ``tmp_path`` rather than by editing the
real, on-disk ``composition-gc-content/capability.toml``:

1. ``tests/capabilities/test_restored_composition_bundles.py`` re-runs the
   real bundle generator (``scripts/capabilities/build_restored_bundles.py``)
   against this exact directory and unconditionally overwrites
   ``capability.toml`` from the ledger. Hand-editing that file to add a
   ``[[fixture]]`` block would survive only until that test next runs, then
   silently disappear - a landmine, not a fixture.
2. A bundle-local (non-core) capability's worker sandbox refuses to import
   any module outside its own bundle-local ``code/`` tree (see
   ``_worker_main.py``'s ``_load_callable``), so a fixture cannot reuse
   ``organelleverse.composition.gc:gc_content`` - the real, installed,
   already-admitted callable - from a *non-core* bundle at all. Only a
   *core*-channel bundle (ships inside the ``organelleverse`` distribution
   itself, no bundle-local code tree, direct in-process import - see
   ``verification.py``'s ``_is_pure_core_native``) can run this real
   callable, and the only real core-channel root is the actual installed
   package directory ``discover_capabilities()`` already scans.

So this file calls the real ``discover_capabilities()`` to fetch the real,
current, admitted ``composition.gc_content`` entry (proving discovery is
genuine - not hand-forged), then augments only its ``bundle.fixtures`` and
``bundle_root`` in memory to point at a fixture directory this test itself
writes under ``tmp_path``. Every field this augmented entry could tell a lie
about - ``status``, ``execution_identity``, the real ``VerificationRecord``,
and every ``EquivalenceEvidence`` - is still produced by the real
``verify_capability`` / ``admit_capabilities`` pipeline, never fabricated.
"""

from __future__ import annotations

import importlib.metadata
import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from organelleverse.capabilities.admission import admit_capabilities
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.hashing import hash_bundle, hash_fixture_dataset
from organelleverse.capabilities.index import CapabilityEntry, CapabilityIndex, CapabilityStatus
from organelleverse.capabilities.models import FixtureSpec
from organelleverse.capabilities.verification import (
    EnvironmentSnapshot,
    LocalVerificationEnvironment,
    VerificationStore,
    compute_environment_key,
    current_platform_identity,
    strip_run_provenance,
    verify_capability,
)
from organelleverse.composition.gc import gc_content
from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.registry import OperationRegistry

_CAPABILITY_ID = "composition.gc_content"
# 20 bp: 6 G + 6 C + 8 A -> gc_content == 0.6, well under the default
# window_size (500), so gc_content() always produces exactly one window.
_SEQUENCE = "GGGGGGCCCCCCAAAAAAAA"


def test_fixture_comparison_strips_execution_run_bookkeeping() -> None:
    first = {
        "operation_id": "demo.worker",
        "provenance": {"run_id": "run-a", "parameters_hash": "a" * 64},
    }
    second = {
        "operation_id": "demo.worker",
        "provenance": {"run_id": "run-b", "parameters_hash": "a" * 64},
    }

    assert strip_run_provenance(first) == strip_run_provenance(second)


def test_fixture_comparison_strips_machine_specific_transport_paths() -> None:
    common = {
        "operation_id": "demo.worker",
        "provenance": {
            "path_translations": [
                {
                    "parameter": "source",
                    "source_context": "host",
                    "target_context": "container",
                    "original_value": "/machine-a/input.fa",
                    "translated_value": "/work-a/input.fa",
                }
            ],
            "inline_materializations": [
                {
                    "parameter": "source",
                    "format": "fasta",
                    "sha256": "a" * 64,
                    "size_bytes": 12,
                    "materialized_path": "/tmp/run-a/source.fa",
                }
            ],
        },
    }
    other = deepcopy(common)
    provenance = cast(dict[str, object], cast(dict[str, object], other)["provenance"])
    cast(list[dict[str, object]], provenance["path_translations"])[0].update(
        original_value="/machine-b/input.fa", translated_value="/work-b/input.fa"
    )
    cast(list[dict[str, object]], provenance["inline_materializations"])[0][
        "materialized_path"
    ] = "/tmp/run-b/source.fa"

    assert strip_run_provenance(common) == strip_run_provenance(other)


def _isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate discover_capabilities() from this real machine's home/cwd/entry points.

    Mirrors test_restored_composition_bundles.py's own
    ``_isolate_non_core_roots`` exactly - only the real "core" search root
    (the installed package directory) stays live; "local"/"project"/entry
    point channels are all neutralized so nothing on the developer's own
    machine can leak into this test.
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


def _real_gc_content_entry() -> CapabilityEntry:
    entry = discover_capabilities().describe(_CAPABILITY_ID)
    assert entry.origins[0].channel == "core"
    assert entry.execution_identity is None
    return entry


def _write_genome_fasta(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "genome.fasta"
    path.write_text(f">demo\n{_SEQUENCE}\n", encoding="utf-8")
    return path


def _augment(
    real_entry: CapabilityEntry, bundle_root: Path, fixtures: tuple[FixtureSpec, ...]
) -> CapabilityEntry:
    """Attach *fixtures* to a copy of *real_entry*, rooted at *bundle_root*.

    Only ``bundle.fixtures``, ``bundle_root``, and ``content_hash`` change;
    ``status``/``diagnostic``/``parameter_schema``/etc. reset to their
    freshly-discovered (unverified) shape rather than carrying over whatever
    the real entry happened to have, since this is a different bundle
    content now.
    """
    return real_entry.model_copy(
        update={
            "bundle": real_entry.bundle.model_copy(update={"fixtures": fixtures}),
            "bundle_root": bundle_root,
            "content_hash": hash_bundle(bundle_root),
            "status": CapabilityStatus.DISCOVERED,
            "diagnostic": None,
            "parameter_schema": None,
            "verification_environment_key": None,
            "worker_parameters": None,
        }
    )


def _seed_passing_fixture(bundle_root: Path, *, case: str = "basic") -> FixtureSpec:
    """Write real fixture data by actually running gc_content once, for real.

    This is exactly how a frozen-regression fixture is meant to be authored:
    run the real implementation once, freeze its real output, and future
    verification runs must keep matching it byte-for-byte.
    """
    case_dir = bundle_root / "fixtures" / case
    fasta_path = _write_genome_fasta(case_dir / "input")

    real_result = gc_content(fasta_path)
    assert real_result.status == "ok"
    assert real_result.metrics["gc_content"] == 0.6

    (case_dir / "expected.json").write_text(
        json.dumps(real_result.model_dump(mode="json"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FixtureSpec.model_validate(
        {
            "case": case,
            "input": {"kind": "none"},
            "parameters": {"genome_fasta": f"fixtures/{case}/input/genome.fasta"},
            "expect": f"fixtures/{case}/expected.json",
            "equivalence": "exact",
        }
    )


def _seed_failing_fixture(bundle_root: Path, *, case: str = "wrong") -> FixtureSpec:
    """Write a fixture whose recorded expectation a real run cannot match."""
    case_dir = bundle_root / "fixtures" / case
    _write_genome_fasta(case_dir / "input")
    (case_dir / "expected.json").write_text(
        json.dumps({"status": "ok", "metrics": {"gc_content": 0.999}}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FixtureSpec.model_validate(
        {
            "case": case,
            "input": {"kind": "none"},
            "parameters": {"genome_fasta": f"fixtures/{case}/input/genome.fasta"},
            "expect": f"fixtures/{case}/expected.json",
            "equivalence": "exact",
        }
    )


def _seed_numeric_tolerance_fixture(
    bundle_root: Path, *, case: str, delta: float, tolerance: float
) -> FixtureSpec:
    """A real run's gc_content, perturbed by *delta*, checked within *tolerance*."""
    case_dir = bundle_root / "fixtures" / case
    fasta_path = _write_genome_fasta(case_dir / "input")

    real_result = gc_content(fasta_path)
    perturbed = cast(dict[str, object], real_result.model_dump(mode="json"))
    metrics = cast(dict[str, object], perturbed["metrics"])
    metrics["gc_content"] = cast(float, metrics["gc_content"]) + delta

    (case_dir / "expected.json").write_text(
        json.dumps(perturbed, sort_keys=True) + "\n", encoding="utf-8"
    )
    return FixtureSpec.model_validate(
        {
            "case": case,
            "input": {"kind": "none"},
            "parameters": {"genome_fasta": f"fixtures/{case}/input/genome.fasta"},
            "expect": f"fixtures/{case}/expected.json",
            "equivalence": "numeric_tolerance",
            "tolerance": tolerance,
        }
    )


def _seed_fixture_with_foreign_provenance(
    bundle_root: Path, *, case: str = "portable"
) -> FixtureSpec:
    """A real fixture whose recorded ``expect`` carries *someone else's*
    run/machine identity - a different checkout's ``object_id``,
    ``parameters_hash``, ``package_version``, and ``git_commit`` - but the
    exact same real science. Simulates exactly the defect this fixture
    proves is fixed: a fixture captured on one machine/checkout, verified
    on another, where only the absolute path (or install/environment)
    differed, never the result.
    """
    case_dir = bundle_root / "fixtures" / case
    fasta_path = _write_genome_fasta(case_dir / "input")

    real_result = gc_content(fasta_path)
    assert real_result.status == "ok"
    assert real_result.metrics["gc_content"] == 0.6

    foreign = cast(dict[str, object], real_result.model_dump(mode="json"))
    foreign["object_id"] = "result:sha256:" + "ab" * 32
    provenance = cast(dict[str, object], foreign["provenance"])
    provenance["object_id"] = "provenance:sha256:" + "cd" * 32
    provenance["parameters_hash"] = "ef" * 32
    provenance["package_version"] = "9.9.9-from-a-different-checkout"
    provenance["git_commit"] = "0000000000000000000000000000000000dead"

    (case_dir / "expected.json").write_text(
        json.dumps(foreign, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FixtureSpec.model_validate(
        {
            "case": case,
            "input": {"kind": "none"},
            "parameters": {"genome_fasta": f"fixtures/{case}/input/genome.fasta"},
            "expect": f"fixtures/{case}/expected.json",
            "equivalence": "exact",
        }
    )


def test_a_fixture_with_a_different_checkouts_provenance_still_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact defect this closes: Decision 004's fixture generator
    captured ``expected.json`` files whose ``object_id`` and
    ``provenance.parameters_hash`` embedded the *capturing* checkout's
    absolute path (for a PATH-codec parameter, ``operations/python_binding
    .py``'s ``_hash_agent_parameters`` hashes the raw, unresolved
    Agent-submitted path string) - so a byte-identical scientific result,
    reproduced on a different checkout, produced a different hash and the
    fixture failed to verify anywhere except the machine that captured it.

    This seeds an ``expected.json`` with real science but a *foreign*
    ``object_id``/``parameters_hash``/``package_version``/``git_commit`` -
    exactly what a different checkout's own real capture would have
    produced - and asserts verification still passes: only
    ``strip_run_provenance``'s excluded fields differ, so equivalence must
    hold.
    """
    _isolated_roots(tmp_path, monkeypatch)
    real_entry = _real_gc_content_entry()

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    fixture = _seed_fixture_with_foreign_provenance(bundle_root)
    entry = _augment(real_entry, bundle_root, (fixture,))
    index = CapabilityIndex(entries=(entry,))
    store = VerificationStore(tmp_path / "verifications")

    record = verify_capability(
        _CAPABILITY_ID, store=store, environment=LocalVerificationEnvironment(index)
    )

    evidence = record.equivalence[0]
    assert evidence.verdict == "pass", "same science, foreign run-provenance must still pass"
    # The evaluator's own recorded provenance for *this* run is untouched -
    # only the comparison ignored the foreign fixture's fields, nothing was
    # silently coerced to match it.
    assert evidence.evaluator_version == "organelleverse.fixture-evaluator.v2"


def test_a_fixture_with_a_real_scientific_difference_still_fails_despite_foreign_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same foreign-provenance shape as above, but this time the actual
    science (``metrics.gc_content``) is also wrong - proving
    ``strip_run_provenance`` narrows what is ignored to exactly the
    run-identity fields, not to "provenance in general", and a real
    scientific mismatch is still caught.
    """
    _isolated_roots(tmp_path, monkeypatch)
    real_entry = _real_gc_content_entry()

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    fixture = _seed_fixture_with_foreign_provenance(bundle_root)
    expected_path = bundle_root / "fixtures" / "portable" / "expected.json"
    corrupted = json.loads(expected_path.read_text(encoding="utf-8"))
    cast(dict[str, object], corrupted["metrics"])["gc_content"] = 0.999
    expected_path.write_text(json.dumps(corrupted, sort_keys=True) + "\n", encoding="utf-8")

    entry = _augment(real_entry, bundle_root, (fixture,))
    index = CapabilityIndex(entries=(entry,))
    store = VerificationStore(tmp_path / "verifications")

    record = verify_capability(
        _CAPABILITY_ID, store=store, environment=LocalVerificationEnvironment(index)
    )

    assert record.equivalence[0].verdict == "fail"


# --- deliverable 4: one real capability, one real fixture, driven end to end ----


def test_real_capability_with_a_real_passing_fixture_goes_discover_verify_admit_invoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _isolated_roots(tmp_path, monkeypatch)
    real_entry = _real_gc_content_entry()

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    fixture = _seed_passing_fixture(bundle_root)
    entry = _augment(real_entry, bundle_root, (fixture,))
    index = CapabilityIndex(entries=(entry,))
    store = VerificationStore(home / "verifications")

    # verify -> the fixture is actually evaluated: a real EquivalenceEvidence,
    # not an exception and not a fabricated record.
    record = verify_capability(
        _CAPABILITY_ID, store=store, environment=LocalVerificationEnvironment(index)
    )
    assert len(record.equivalence) == 1
    evidence = record.equivalence[0]
    assert evidence.case == "basic"
    assert evidence.method == "exact"
    assert evidence.verdict == "pass"
    assert evidence.candidate.bundle_content_hash == entry.content_hash
    assert evidence.candidate.environment_key == record.environment_key
    assert store.read(entry.capability_id, entry.content_hash, record.environment_key) == record

    # admit -> a passing fixture verdict is one more real admission rule
    # already satisfied, not bypassed.
    admitted_index = admit_capabilities(index, store=store)
    admitted_entry = admitted_index.describe(_CAPABILITY_ID)
    assert admitted_entry.status is CapabilityStatus.ADMITTED, admitted_entry.diagnostic

    # invoke -> the real capability still works for a value the fixture
    # never saw.
    binding_source = admitted_index.binding_source()
    registry = OperationRegistry(capability_source=binding_source)
    invoke_fasta = _write_genome_fasta(tmp_path / "invoke")
    result = registry.invoke(
        _CAPABILITY_ID, input=None, parameters={"genome_fasta": str(invoke_fasta)}
    )

    assert isinstance(result, OrganelleResult)
    assert result.status == "ok"
    assert result.metrics["gc_content"] == 0.6


# --- deliverable 5: a fixture that does not match rejects admission -------------


def test_mismatched_fixture_expectation_fails_the_verdict_and_admission_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _isolated_roots(tmp_path, monkeypatch)
    real_entry = _real_gc_content_entry()

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    fixture = _seed_failing_fixture(bundle_root)
    entry = _augment(real_entry, bundle_root, (fixture,))
    index = CapabilityIndex(entries=(entry,))
    store = VerificationStore(home / "verifications")

    # A scientific mismatch is a recorded verdict, not an exception.
    record = verify_capability(
        _CAPABILITY_ID, store=store, environment=LocalVerificationEnvironment(index)
    )
    assert record.equivalence[0].verdict == "fail"
    assert store.read(entry.capability_id, entry.content_hash, record.environment_key) == record

    admitted_index = admit_capabilities(index, store=store)
    admitted_entry = admitted_index.describe(_CAPABILITY_ID)
    assert admitted_entry.status is CapabilityStatus.REJECTED
    assert admitted_entry.diagnostic is not None
    assert admitted_entry.diagnostic.code == "capability.equivalence_failed"


# --- deliverable 2: NUMERIC_TOLERANCE, both directions ---------------------------


def test_numeric_tolerance_fixture_passes_within_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _isolated_roots(tmp_path, monkeypatch)
    real_entry = _real_gc_content_entry()

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    fixture = _seed_numeric_tolerance_fixture(
        bundle_root, case="close", delta=0.0005, tolerance=0.001
    )
    entry = _augment(real_entry, bundle_root, (fixture,))
    index = CapabilityIndex(entries=(entry,))
    store = VerificationStore(home / "verifications")

    record = verify_capability(
        _CAPABILITY_ID, store=store, environment=LocalVerificationEnvironment(index)
    )

    evidence = record.equivalence[0]
    assert evidence.method == "numeric_tolerance"
    assert evidence.tolerance == 0.001
    assert evidence.verdict == "pass"


def test_numeric_tolerance_fixture_fails_outside_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _isolated_roots(tmp_path, monkeypatch)
    real_entry = _real_gc_content_entry()

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    fixture = _seed_numeric_tolerance_fixture(bundle_root, case="far", delta=0.05, tolerance=0.001)
    entry = _augment(real_entry, bundle_root, (fixture,))
    index = CapabilityIndex(entries=(entry,))
    store = VerificationStore(home / "verifications")

    record = verify_capability(
        _CAPABILITY_ID, store=store, environment=LocalVerificationEnvironment(index)
    )

    evidence = record.equivalence[0]
    assert evidence.method == "numeric_tolerance"
    assert evidence.verdict == "fail"


# --- scope boundaries: unimplementable cases fail closed, loudly, by name -------


def _direct_environment(real_entry: CapabilityEntry) -> LocalVerificationEnvironment:
    return LocalVerificationEnvironment(CapabilityIndex(entries=(real_entry,)))


def _guard_call_arguments(real_entry: CapabilityEntry) -> dict[str, object]:
    snapshot = EnvironmentSnapshot(platform=current_platform_identity(), observed_environment={})
    return {
        "environment_key": compute_environment_key(snapshot),
        "fixture_dataset_hash": hash_fixture_dataset(real_entry.bundle_root),
    }


def test_evaluate_fixture_fails_closed_for_reference_equivalence_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference-equivalence fixtures (comparing against a second, live
    capability) are not implemented yet - a real gap, not a fabricated one.
    Once both sides are computed, comparing their hashes is still a
    data-level operation; resolving and safely executing the *second*
    capability is the actual missing piece, so this fails closed with its
    own code rather than silently skipping or passing.
    """
    _isolated_roots(tmp_path, monkeypatch)
    real_entry = _real_gc_content_entry()
    fixture = FixtureSpec.model_validate(
        {
            "case": "cross_check",
            "input": {"kind": "none"},
            "parameters": {"genome_fasta": "fixtures/cross_check/input/genome.fasta"},
            "reference": "cap:composition.compute_gc_content",
            "equivalence": "numeric_tolerance",
            "tolerance": 1e-6,
        }
    )
    environment = _direct_environment(real_entry)

    with pytest.raises(OrganelleContractError) as captured:
        environment.evaluate_fixture(
            real_entry,
            fixture,
            executor=environment.executor,
            **_guard_call_arguments(real_entry),
        )

    assert captured.value.code == "capability.fixture_reference_unsupported"


def test_evaluate_fixture_fails_closed_for_non_none_input_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructing a CoreObject from a fixture's JSON ``input`` is a
    separate adapter this evaluator does not build yet (7/132 admitted
    capabilities declare a non-``none`` input_kind - see Decision 004). This
    fails closed with its own code instead of guessing at a CoreObject
    shape.
    """
    _isolated_roots(tmp_path, monkeypatch)
    real_entry = _real_gc_content_entry()
    genome_contract = real_entry.bundle.contract.model_copy(update={"input_kind": "genome"})
    genome_bundle = real_entry.bundle.model_copy(update={"contract": genome_contract})
    entry = real_entry.model_copy(update={"bundle": genome_bundle})
    fixture = FixtureSpec.model_validate(
        {
            "case": "genome_case",
            "input": {"kind": "genome", "path": "fixtures/genome_case/input/genome.fasta"},
            "expect": "fixtures/genome_case/expected.json",
            "equivalence": "exact",
        }
    )
    environment = _direct_environment(entry)

    with pytest.raises(OrganelleContractError) as captured:
        environment.evaluate_fixture(
            entry,
            fixture,
            executor=environment.executor,
            **_guard_call_arguments(entry),
        )

    assert captured.value.code == "capability.fixture_input_kind_unsupported"


def test_evaluate_fixture_fails_closed_for_set_membership_defensively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``verify_capability`` already rejects any bundle declaring a
    ``set_membership`` fixture before ``evaluate_fixture`` is ever called
    (see ``verification.py``'s ``unsupported`` check, which this fixture's
    own ``equivalence`` value would trip). Calling ``evaluate_fixture``
    directly - bypassing that upstream guard - proves the evaluator itself
    also refuses this case, with the *same* error code, rather than
    silently mis-comparing it as if it were ``exact``.
    """
    _isolated_roots(tmp_path, monkeypatch)
    real_entry = _real_gc_content_entry()
    fixture = FixtureSpec.model_validate(
        {
            "case": "membership_case",
            "input": {"kind": "none"},
            "expect": "fixtures/membership_case/expected.json",
            "equivalence": "set_membership",
        }
    )
    environment = _direct_environment(real_entry)

    with pytest.raises(OrganelleContractError) as captured:
        environment.evaluate_fixture(
            real_entry,
            fixture,
            executor=environment.executor,
            **_guard_call_arguments(real_entry),
        )

    assert captured.value.code == "capability.equivalence_method_unsupported"
