"""Admission gates 5/6/7: adversarial fail-closed, determinism, rewrite.

Every test here drives the real ``discover_capabilities`` → ``verify_capability``
→ ``admit_capabilities`` pipeline against genuinely importable core-channel
native capabilities (``fixtures_gates_5_6_7.py``), the same discipline
``test_core_admission.py`` follows. Nothing constructs an ``EquivalenceEvidence``
or a ``VerificationRecord`` by hand.

* Gate 5 - an adversarial fixture fails closed: it passes verification only
  when the real implementation raises exactly the declared
  ``expect_failure_code`` on the fixture's inputs; an unexpected success or a
  different code is a recorded ``fail`` verdict, and admission rejects.
* Gate 6 - a contract may declare ``deterministic = false`` only with an
  honest ``deterministic_reason``, and verification empirically reruns such
  an operation a second time: reruns that disagree fail admission.
* Gate 7 - a contract that declares a ``rewrite`` must still produce output
  equivalent to the original implementation it names, compared with the
  declared method and tolerance.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from organelleverse.capabilities.admission import admit_capabilities
from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.index import CapabilityStatus
from organelleverse.capabilities.models import FixtureSpec
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.spec import OperationSpec, RewriteSpec

from . import fixtures_gates_5_6_7 as gate_impls

_LOCATOR_PREFIX = "tests.capabilities.fixtures_gates_5_6_7:"

_ECHO_ID = "demo.gates_echo"
_FLAKY_ID = "demo.gates_flaky"
_VALIDATED_ID = "demo.gates_validated"


def _bundle_toml(
    capability_id: str,
    locator_attribute: str,
    *,
    description: str,
    parameter_name: str,
    contract_extra: str = "",
    fixture_blocks: str = "",
) -> str:
    return (
        'schema = "organelleverse.capability.v1"\n'
        "\n"
        "[capability]\n"
        f'id = "{capability_id}"\n'
        'bundle_version = "1.0.0"\n'
        'implementation = "native"\n'
        "\n"
        "[contract]\n"
        'contract_version = "1.0"\n'
        f'title = "Gate fixture {capability_id}"\n'
        f"description = {json.dumps(description)}\n"
        'keywords = ["admission", "capability", "gates"]\n'
        'execution_mode = "inline"\n'
        'stage = "analyze"\n'
        'input_kind = "none"\n'
        'output_kind = "result"\n'
        f'callable_locator = "{_LOCATOR_PREFIX}{locator_attribute}"\n'
        f"{contract_extra}"
        "\n"
        "[contract.binding]\n"
        'argument_mode = "named_parameters"\n'
        'result_codec = "canonical"\n'
        "\n"
        "[[contract.binding.parameters]]\n"
        f'name = "{parameter_name}"\n'
        'codec = "json"\n'
        f"{fixture_blocks}"
    )


def _seed_expect(bundle_root: Path, case: str, produced: OrganelleResult) -> None:
    """Freeze a real implementation output as the fixture's expectation."""
    case_dir = bundle_root / "fixtures" / case
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "expected.json").write_text(
        json.dumps(produced.model_dump(mode="json"), sort_keys=True) + "\n", encoding="utf-8"
    )


def _isolate_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tomls: dict[str, str]
) -> tuple[Path, dict[str, Path]]:
    """Route every standard discovery root at isolated tmp directories.

    Returns the isolated ``ORGANELLEVERSE_HOME`` plus each bundle's root (the
    "core" package capabilities directory this test populates), so callers can
    seed fixture files next to the ``capability.toml`` they just installed.
    """
    package = tmp_path / "installed-package"
    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / ".organelleverse" / "capabilities").mkdir(parents=True)
    home.mkdir()
    core_root = package / "capabilities"
    core_root.mkdir(parents=True)
    bundle_roots: dict[str, Path] = {}
    for slug, text in tomls.items():
        bundle_root = core_root / slug
        bundle_root.mkdir()
        (bundle_root / "capability.toml").write_text(text, encoding="utf-8")
        bundle_roots[slug] = bundle_root

    monkeypatch.setattr(importlib.resources, "files", lambda package_name: package)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: ())
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(home))
    monkeypatch.delenv("ORGANELLEVERSE_CAPABILITY_PATH", raising=False)
    monkeypatch.chdir(project)
    return home, bundle_roots


def _verify_and_admit(capability_id: str, home: Path) -> tuple[object, CapabilityStatus]:
    index = discover_capabilities()
    store = VerificationStore(home / "verifications")
    record = verify_capability(
        capability_id, store=store, environment=LocalVerificationEnvironment(index)
    )
    admitted = admit_capabilities(index, store=store)
    return record, admitted.describe(capability_id).status


# --- gate 5: adversarial fixtures fail closed -----------------------------------


def _validated_bundle(*, adversarial_parameters: str, expect_failure_code: str) -> str:
    return _bundle_toml(
        _VALIDATED_ID,
        "run_validated",
        description=(
            "An importable validator whose 'bad' mode raises a stable structured "
            "error code, exercising the adversarial admission gate."
        ),
        parameter_name="mode",
        fixture_blocks=(
            "[[fixture]]\n"
            'case = "rejects_bad_mode"\n'
            'input = {kind = "none"}\n'
            f"parameters = {adversarial_parameters}\n"
            "adversarial = true\n"
            f'expect_failure_code = "{expect_failure_code}"\n'
        ),
    )


def test_adversarial_fixture_passes_when_the_declared_code_is_actually_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml = _validated_bundle(
        adversarial_parameters='{mode = "bad"}',
        expect_failure_code="capability.parameter_invalid",
    )
    home, _ = _isolate_roots(tmp_path, monkeypatch, {"demo-gates-validated": toml})

    record, status = _verify_and_admit(_VALIDATED_ID, home)

    evidence = record.equivalence[0]
    assert evidence.verdict == "pass"
    assert evidence.determinism_verdict == "not_checked"
    assert evidence.rewrite_verdict == "not_checked"
    assert status is CapabilityStatus.ADMITTED


def test_adversarial_fixture_fails_when_the_capability_succeeds_instead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fail-closed half of gate 5: an adversarial input the capability
    happily accepts is a recorded failure, not a skipped check."""
    toml = _validated_bundle(
        adversarial_parameters='{mode = "ok"}',
        expect_failure_code="capability.parameter_invalid",
    )
    home, _ = _isolate_roots(tmp_path, monkeypatch, {"demo-gates-validated": toml})

    index = discover_capabilities()
    store = VerificationStore(home / "verifications")
    record = verify_capability(
        _VALIDATED_ID, store=store, environment=LocalVerificationEnvironment(index)
    )
    admitted = admit_capabilities(index, store=store)

    assert record.equivalence[0].verdict == "fail"
    entry = admitted.describe(_VALIDATED_ID)
    assert entry.status is CapabilityStatus.REJECTED
    assert entry.diagnostic is not None
    assert entry.diagnostic.code == "capability.equivalence_failed"


def test_adversarial_fixture_fails_when_a_different_code_is_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared code the implementation never raises is just as much a
    failure as no error at all - the gate pins the exact code."""
    toml = _validated_bundle(
        adversarial_parameters='{mode = "bad"}',
        expect_failure_code="capability.some_other_code",
    )
    home, _ = _isolate_roots(tmp_path, monkeypatch, {"demo-gates-validated": toml})

    record, _ = _verify_and_admit(_VALIDATED_ID, home)

    assert record.equivalence[0].verdict == "fail"


def test_adversarial_fixture_model_requires_a_code_and_forbids_expect() -> None:
    with pytest.raises(ValidationError):
        FixtureSpec.model_validate({"case": "x", "input": {"kind": "none"}, "adversarial": True})
    with pytest.raises(ValidationError):
        FixtureSpec.model_validate(
            {
                "case": "x",
                "input": {"kind": "none"},
                "adversarial": True,
                "expect_failure_code": "capability.parameter_invalid",
                "expect": "fixtures/x/expected.json",
            }
        )
    fixture = FixtureSpec.model_validate(
        {
            "case": "x",
            "input": {"kind": "none"},
            "adversarial": True,
            "expect_failure_code": "capability.parameter_invalid",
        }
    )
    assert fixture.expect is None
    assert fixture.reference is None


# --- gate 6: deterministic declaration + empirical rerun ------------------------


_FLAKY_REASON = "scripted offsets make consecutive calls diverge in the determinism gate"


def _flaky_bundle() -> str:
    return _bundle_toml(
        _FLAKY_ID,
        "run_flaky",
        description=(
            "An importable non-deterministic echo whose outputs are scripted per "
            "call, exercising the determinism admission gate."
        ),
        parameter_name="value",
        contract_extra=(f'deterministic = false\ndeterministic_reason = "{_FLAKY_REASON}"\n'),
        fixture_blocks=(
            "[[fixture]]\n"
            'case = "basic"\n'
            'input = {kind = "none"}\n'
            "parameters = {value = 5}\n"
            'expect = "fixtures/basic/expected.json"\n'
            'equivalence = "exact"\n'
        ),
    )


def test_non_deterministic_operation_admits_when_verification_reruns_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml = _flaky_bundle()
    home, roots = _isolate_roots(tmp_path, monkeypatch, {"demo-gates-flaky": toml})
    # One offset per future run_flaky call: the seed capture, then the two
    # verification invokes. All agree, so the empirical rerun passes.
    gate_impls.queue_flaky_offsets(5, 5, 5)
    _seed_expect(roots["demo-gates-flaky"], "basic", gate_impls.run_flaky(value=5))

    record, status = _verify_and_admit(_FLAKY_ID, home)

    evidence = record.equivalence[0]
    assert evidence.verdict == "pass"
    assert evidence.determinism_verdict == "pass"
    assert evidence.determinism_produced_hash == evidence.candidate.produced_hash
    assert status is CapabilityStatus.ADMITTED


def test_non_deterministic_operation_rejects_when_verification_reruns_disagree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded expectation still matches, but the two verification runs
    disagree with each other - only the determinism verdict catches it."""
    toml = _flaky_bundle()
    home, roots = _isolate_roots(tmp_path, monkeypatch, {"demo-gates-flaky": toml})
    gate_impls.queue_flaky_offsets(5, 5, 9)
    _seed_expect(roots["demo-gates-flaky"], "basic", gate_impls.run_flaky(value=5))

    index = discover_capabilities()
    store = VerificationStore(home / "verifications")
    record = verify_capability(
        _FLAKY_ID, store=store, environment=LocalVerificationEnvironment(index)
    )
    admitted = admit_capabilities(index, store=store)

    evidence = record.equivalence[0]
    assert evidence.determinism_verdict == "fail"
    assert evidence.determinism_produced_hash is not None
    assert evidence.determinism_produced_hash != evidence.candidate.produced_hash
    assert evidence.verdict == "fail"
    entry = admitted.describe(_FLAKY_ID)
    assert entry.status is CapabilityStatus.REJECTED
    assert entry.diagnostic is not None
    assert entry.diagnostic.code == "capability.equivalence_failed"


def test_determinism_model_requires_a_reason_exactly_when_non_deterministic() -> None:
    def kwargs(**overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "operation_id": "demo.gates_model",
            "contract_version": "1.0",
            "title": "Determinism model gate",
            "description": "A minimal contract exercising deterministic_reason rules.",
            "keywords": ("admission", "capability", "gates"),
            "execution_mode": "inline",
            "stage": "analyze",
            "input_kind": "none",
            "output_kind": "result",
            "deterministic": True,
        }
        base.update(overrides)
        return base

    with pytest.raises(ValidationError):
        OperationSpec.model_validate(kwargs(deterministic=False))
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(kwargs(deterministic=True, deterministic_reason="anything"))
    spec = OperationSpec.model_validate(
        kwargs(deterministic=False, deterministic_reason="a live database changes")
    )
    assert spec.deterministic_reason == "a live database changes"


# --- gate 7: rewrite equivalence against the named original --------------------


def _echo_bundle(rewrite_original: str, *, method: str = "exact", tolerance: str = "") -> str:
    rewrite_table = (
        "[contract.rewrite]\n"
        f'original = "{rewrite_original}"\n'
        f'method = "{method}"\n' + (f"tolerance = {tolerance}\n" if tolerance else "")
    )
    return _bundle_toml(
        _ECHO_ID,
        "run_echo",
        description=(
            "An importable echo that declares itself a rewrite of an original "
            "implementation, exercising the rewrite admission gate."
        ),
        parameter_name="value",
        contract_extra=rewrite_table,
        fixture_blocks=(
            "[[fixture]]\n"
            'case = "basic"\n'
            'input = {kind = "none"}\n'
            "parameters = {value = 7}\n"
            'expect = "fixtures/basic/expected.json"\n'
            'equivalence = "exact"\n'
        ),
    )


def test_rewrite_equivalence_passes_against_an_identical_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml = _echo_bundle(f"{_LOCATOR_PREFIX}run_echo_original")
    home, roots = _isolate_roots(tmp_path, monkeypatch, {"demo-gates-echo": toml})
    _seed_expect(roots["demo-gates-echo"], "basic", gate_impls.run_echo(value=7))

    record, status = _verify_and_admit(_ECHO_ID, home)

    evidence = record.equivalence[0]
    assert evidence.verdict == "pass"
    # A deterministic operation is not rerun by the determinism gate.
    assert evidence.determinism_verdict == "not_checked"
    assert evidence.rewrite_verdict == "pass"
    assert evidence.rewrite_original_callable == f"{_LOCATOR_PREFIX}run_echo_original"
    assert evidence.rewrite_reference_hash == evidence.candidate.produced_hash
    assert status is CapabilityStatus.ADMITTED


def test_rewrite_equivalence_fails_when_the_original_output_drifted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml = _echo_bundle(f"{_LOCATOR_PREFIX}run_echo_drifted")
    home, roots = _isolate_roots(tmp_path, monkeypatch, {"demo-gates-echo": toml})
    _seed_expect(roots["demo-gates-echo"], "basic", gate_impls.run_echo(value=7))

    index = discover_capabilities()
    store = VerificationStore(home / "verifications")
    record = verify_capability(
        _ECHO_ID, store=store, environment=LocalVerificationEnvironment(index)
    )
    admitted = admit_capabilities(index, store=store)

    evidence = record.equivalence[0]
    assert evidence.rewrite_verdict == "fail"
    assert evidence.rewrite_reference_hash is not None
    assert evidence.rewrite_reference_hash != evidence.candidate.produced_hash
    assert evidence.verdict == "fail"
    entry = admitted.describe(_ECHO_ID)
    assert entry.status is CapabilityStatus.REJECTED
    assert entry.diagnostic is not None
    assert entry.diagnostic.code == "capability.equivalence_failed"


def test_rewrite_model_gates_tolerance_on_method() -> None:
    with pytest.raises(ValidationError):
        RewriteSpec.model_validate(
            {"original": "demo.module:original", "method": "numeric_tolerance"}
        )
    with pytest.raises(ValidationError):
        RewriteSpec.model_validate(
            {"original": "demo.module:original", "method": "exact", "tolerance": 0.5}
        )
    spec = RewriteSpec.model_validate(
        {"original": "demo.module:original", "method": "numeric_tolerance", "tolerance": 1e-6}
    )
    assert spec.tolerance == 1e-6
