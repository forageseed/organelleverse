"""Implementation-independent v3 capability projections."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.capabilities.index import CapabilityEntry, CapabilityOrigin, CapabilityStatus
from organelleverse.capabilities.models import CapabilityBundle, PluginCapabilityBundle
from organelleverse.core.errors import OrganelleContractError
from organelleverse.optimization import contract_from_capability_entry
from organelleverse.optimization.contracts_v3 import (
    OptimizationBudgetV3,
    OptimizationDeclarationV3,
    StrategyReferenceV3,
)
from organelleverse.optimization.models import OptimizationContract
from organelleverse.optimization.profiles_v3 import (
    classify_optimization_profile,
    contract_v3_from_v2,
    identity_from_capability_entry,
)
from tests.capabilities.test_bundle_models import minimal_bundle
from tests.capabilities.test_plugin_v2_models import v2_payload
from tests.optimization.test_contracts_v3 import declaration_payload, identity
from tests.plugin_experiments.conftest import admit, build_plugin_entry

_HASH = "sha256:" + "d" * 64


def entry(kind: str, tmp_path: Path, capability_id: str | None = None) -> CapabilityEntry:
    if kind == "plugin":
        bundle = PluginCapabilityBundle.model_validate(v2_payload())
    else:
        payload = minimal_bundle(
            capability_id=capability_id or f"demo.{kind}",
            implementation="composite" if kind == "composite" else "native",
            callable_locator=None if kind == "composite" else "demo.api:probe",
        )
        bundle = CapabilityBundle.model_validate(payload)
    return CapabilityEntry(
        capability_id=bundle.capability.id,
        content_hash=_HASH,
        bundle_root=tmp_path,
        bundle=bundle,
        origins=(CapabilityOrigin(channel="core", source_path=f"fixture/{kind}"),),
        status=CapabilityStatus.ADMITTED,
    )


def strategy_entries(tmp_path: Path) -> dict[str, CapabilityEntry]:
    return {
        kind: entry("native", tmp_path, f"strategy.{kind}")
        for kind in ("explicit", "random", "agent")
    }


def bound_declaration(
    tmp_path: Path, *, benchmark_enabled: bool = True
) -> tuple[OptimizationDeclarationV3, dict[str, CapabilityEntry]]:
    strategy = entry("native", tmp_path, "strategy.grid")
    payload = declaration_payload()
    payload["strategies"] = (
        StrategyReferenceV3(kind="grid", identity=identity_from_capability_entry(strategy)),
    )
    if not benchmark_enabled:
        payload["benchmark"] = None
    return OptimizationDeclarationV3(**payload), {"grid": strategy}


@pytest.mark.parametrize(
    ("kind", "surface", "implementation"),
    [
        ("native", "native", "native"),
        ("plugin", "plugin", "native"),
        ("composite", "composite", "composite"),
    ],
)
def test_all_capability_implementations_project_to_the_same_identity_and_profile_shape(
    tmp_path: Path, kind: str, surface: str, implementation: str
) -> None:
    target = entry(kind, tmp_path)
    declaration, strategies = bound_declaration(tmp_path, benchmark_enabled=False)

    identity = identity_from_capability_entry(target)
    profile = classify_optimization_profile(
        target, declaration=declaration, strategy_entries=strategies
    )

    assert identity.capability_id == target.capability_id
    assert identity.bundle_content_hash == target.content_hash
    assert identity.surface == surface
    assert identity.implementation == implementation
    assert profile.target == identity
    assert profile.status == "eligible"
    assert profile.contract is None
    assert profile.reason_code == "optimization.benchmark_required"


def test_evaluator_gate_requires_exact_admitted_identity(tmp_path: Path) -> None:
    target = entry("native", tmp_path)
    evaluator = entry("plugin", tmp_path).model_copy(update={"capability_id": "demo.spoofed"})
    declaration, strategies = bound_declaration(tmp_path)

    with pytest.raises(OrganelleContractError) as raised:
        classify_optimization_profile(
            target,
            declaration=declaration,
            evaluator=evaluator,
            strategy_entries=strategies,
        )

    assert raised.value.code == "optimization.identity_mismatch"


def test_complete_independent_evaluator_enables_profile(tmp_path: Path) -> None:
    target = entry("native", tmp_path)
    evaluator = entry("plugin", tmp_path)
    declaration, strategies = bound_declaration(tmp_path)

    profile = classify_optimization_profile(
        target,
        declaration=declaration,
        evaluator=evaluator,
        strategy_entries=strategies,
    )

    assert profile.status == "enabled"
    assert profile.reason_code is None
    assert profile.contract is not None
    assert profile.contract.evaluator.identity.capability_id == evaluator.capability_id


@pytest.mark.parametrize(("max_trials", "parallelism"), [(7, 2), (6, 1)])
def test_explicit_v2_upgrade_preserves_identity_domains_seed_and_budget_ceiling(
    tmp_path: Path, max_trials: int, parallelism: int
) -> None:
    target = admit(build_plugin_entry(tmp_path / "legacy", max_trials=7, parallelism=2))
    evaluator = entry("native", tmp_path)
    legacy = contract_from_capability_entry(target, seed=41)
    original_digest = legacy.digest
    payload = declaration_payload()
    budget = payload["budget"]
    budget = budget.model_copy(  # type: ignore[union-attr]
        update={"max_trials": max_trials, "parallelism": parallelism}
    )
    repeats = payload["repeats"].model_copy(update={"seed": 41})  # type: ignore[union-attr]

    upgraded = contract_v3_from_v2(
        target,
        legacy,
        baseline={"parameters": {"threshold": 0.5}},
        benchmark=payload["benchmark"],
        evaluator=evaluator,
        objectives=payload["objectives"],
        strategy_entries=strategy_entries(tmp_path),
        constraints=payload["constraints"],
        adoption=payload["adoption"],
        budget=budget,
        repeats=repeats,
    )

    assert upgraded.target.capability_id == legacy.target_capability_id
    assert upgraded.target.bundle_content_hash == legacy.bundle_content_hash
    assert upgraded.parameters == legacy.parameters
    assert upgraded.repeats.seed == legacy.seed == 41
    assert tuple(item.kind for item in upgraded.strategies) == legacy.strategies
    assert tuple(item.identity.capability_id for item in upgraded.strategies) == (
        "strategy.explicit",
        "strategy.random",
        "strategy.agent",
    )
    assert upgraded.budget.max_trials == max_trials <= legacy.budget.max_trials
    assert upgraded.budget.parallelism == parallelism <= legacy.budget.parallelism
    assert legacy.digest == original_digest
    assert OptimizationContract.model_validate(legacy.model_dump()) == legacy


def test_invalid_evaluator_never_downgrades_to_eligible_when_benchmark_is_missing(
    tmp_path: Path,
) -> None:
    target = entry("native", tmp_path)
    evaluator = entry("plugin", tmp_path).model_copy(update={"status": CapabilityStatus.REJECTED})
    declaration, strategies = bound_declaration(tmp_path, benchmark_enabled=False)

    with pytest.raises(OrganelleContractError) as raised:
        classify_optimization_profile(
            target,
            declaration=declaration,
            evaluator=evaluator,
            strategy_entries=strategies,
        )

    assert raised.value.code == "optimization.evaluator_not_admitted"


def test_classification_requires_explicit_declaration_or_not_applicable_reason(
    tmp_path: Path,
) -> None:
    target = entry("native", tmp_path)

    with pytest.raises(OrganelleContractError) as missing:
        classify_optimization_profile(target, declaration=None)
    assert missing.value.code == "optimization.classification_required"

    profile = classify_optimization_profile(
        target, declaration=None, not_applicable_reason="optimization.no_scientific_objective"
    )
    assert profile.status == "not_applicable"
    assert profile.reason_code == "optimization.no_scientific_objective"

    with pytest.raises(OrganelleContractError) as conflict:
        classify_optimization_profile(
            target,
            declaration=bound_declaration(tmp_path)[0],
            not_applicable_reason="optimization.no_scientific_objective",
        )
    assert conflict.value.code == "optimization.classification_conflict"


def test_identity_projection_itself_rejects_non_admitted_entry(tmp_path: Path) -> None:
    rejected = entry("native", tmp_path).model_copy(update={"status": CapabilityStatus.REJECTED})

    with pytest.raises(OrganelleContractError) as raised:
        identity_from_capability_entry(rejected)

    assert raised.value.code == "capability.not_admitted"


def test_same_target_evaluator_is_rejected(tmp_path: Path) -> None:
    target = entry("native", tmp_path)

    declaration, strategies = bound_declaration(tmp_path)
    with pytest.raises(OrganelleContractError) as raised:
        classify_optimization_profile(
            target,
            declaration=declaration,
            evaluator=target,
            strategy_entries=strategies,
        )

    assert raised.value.code == "optimization.evaluator_not_independent"


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("missing", "optimization.strategy_binding_mismatch"),
        ("extra", "optimization.strategy_binding_mismatch"),
        ("non_admitted", "optimization.strategy_not_admitted"),
        ("identity_mismatch", "optimization.strategy_binding_mismatch"),
        ("forged_identity", "optimization.strategy_binding_mismatch"),
    ],
)
def test_classify_requires_exact_admitted_strategy_bindings_before_any_status(
    tmp_path: Path, case: str, expected_code: str
) -> None:
    target = entry("native", tmp_path)
    declaration, mapping = bound_declaration(tmp_path, benchmark_enabled=False)
    if case == "missing":
        supplied = None
    elif case == "extra":
        supplied = {
            **mapping,
            "random": entry("native", tmp_path, "strategy.random"),
        }
    elif case == "non_admitted":
        supplied = {
            "grid": mapping["grid"].model_copy(update={"status": CapabilityStatus.REJECTED})
        }
    elif case == "identity_mismatch":
        supplied = {
            "grid": mapping["grid"].model_copy(update={"content_hash": "sha256:" + "e" * 64})
        }
    else:
        payload = declaration_payload()
        payload["benchmark"] = None
        payload["strategies"] = (
            StrategyReferenceV3(kind="grid", identity=identity("fake.strategy")),
        )
        declaration = OptimizationDeclarationV3(**payload)
        supplied = mapping

    with pytest.raises(OrganelleContractError) as raised:
        classify_optimization_profile(
            target,
            declaration=declaration,
            strategy_entries=supplied,
        )

    assert raised.value.code == expected_code


@pytest.mark.parametrize("case", ["missing", "extra", "non_admitted"])
def test_v2_upgrade_requires_exact_admitted_strategy_mapping(tmp_path: Path, case: str) -> None:
    target = admit(build_plugin_entry(tmp_path / "legacy-map", max_trials=7, parallelism=2))
    legacy = contract_from_capability_entry(target, seed=41)
    mapping = strategy_entries(tmp_path)
    if case == "missing":
        mapping.pop("agent")
    elif case == "extra":
        mapping["grid"] = entry("native", tmp_path, "strategy.grid")
    else:
        mapping["random"] = mapping["random"].model_copy(
            update={"status": CapabilityStatus.REJECTED}
        )
    payload = declaration_payload()

    with pytest.raises(OrganelleContractError) as raised:
        contract_v3_from_v2(
            target,
            legacy,
            baseline={"parameters": {"threshold": 0.5}},
            benchmark=payload["benchmark"],
            evaluator=entry("native", tmp_path),
            objectives=payload["objectives"],
            strategy_entries=mapping,
            adoption=payload["adoption"],
            budget=OptimizationBudgetV3(
                max_trials=7,
                parallelism=2,
                max_wall_time_seconds=60,
                max_cpu_time_seconds=120,
                max_peak_memory_bytes=1024,
            ),
            repeats=payload["repeats"].model_copy(update={"seed": 41}),  # type: ignore[union-attr]
        )

    assert raised.value.code == "optimization.v2_upgrade_incomplete"


@pytest.mark.parametrize(
    ("max_trials", "parallelism", "seed"),
    [(8, 2, 41), (7, 3, 41), (7, 2, 42)],
)
def test_v2_upgrade_rejects_budget_expansion_and_seed_mismatch(
    tmp_path: Path, max_trials: int, parallelism: int, seed: int
) -> None:
    target = admit(build_plugin_entry(tmp_path / "legacy-budget", max_trials=7, parallelism=2))
    legacy = contract_from_capability_entry(target, seed=41)
    payload = declaration_payload()

    with pytest.raises(OrganelleContractError) as raised:
        contract_v3_from_v2(
            target,
            legacy,
            baseline={"parameters": {"threshold": 0.5}},
            benchmark=payload["benchmark"],
            evaluator=entry("native", tmp_path),
            objectives=payload["objectives"],
            strategy_entries=strategy_entries(tmp_path),
            adoption=payload["adoption"],
            budget=OptimizationBudgetV3(
                max_trials=max_trials,
                parallelism=parallelism,
                max_wall_time_seconds=60,
                max_cpu_time_seconds=120,
                max_peak_memory_bytes=1024,
            ),
            repeats=payload["repeats"].model_copy(update={"seed": seed}),  # type: ignore[union-attr]
        )

    assert raised.value.code == "optimization.v2_upgrade_incomplete"
