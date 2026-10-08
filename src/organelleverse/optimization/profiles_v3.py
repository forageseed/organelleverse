"""Pure projections from admitted capabilities to optimization contract v3."""

from __future__ import annotations

from collections.abc import Mapping

from organelleverse.capabilities.index import CapabilityEntry, CapabilityStatus
from organelleverse.capabilities.models import ImplementationKind, PluginCapabilityBundle
from organelleverse.core.errors import OrganelleContractError

from .contracts_v3 import (
    BaselineCandidateV3,
    MetricConstraintV3,
    ObjectiveV3,
    OptimizationBudgetV3,
    OptimizationContractV3,
    OptimizationDeclarationV3,
    OptimizationProfileV3,
    PriorCandidateV3,
    RepeatPolicyV3,
    StrategyReferenceV3,
)
from .evaluation import (
    AdoptionPolicyV3,
    BenchmarkContractV3,
    CapabilityIdentityV3,
    EvaluatorContractV3,
)
from .models import OptimizationContract, OptimizationStrategy


def identity_from_capability_entry(entry: CapabilityEntry) -> CapabilityIdentityV3:
    """Project exact immutable identity without importing capability code."""

    _require_admitted(entry)
    if entry.capability_id != entry.bundle.capability.id:
        raise _error(
            "optimization.identity_mismatch",
            "capability entry ID does not match its bundle identity",
            entry.capability_id,
            bundle_capability_id=entry.bundle.capability.id,
        )
    implementation = entry.bundle.capability.implementation
    surface = (
        "plugin"
        if isinstance(entry.bundle, PluginCapabilityBundle)
        else "composite"
        if implementation is ImplementationKind.COMPOSITE
        else "native"
    )
    return CapabilityIdentityV3(
        capability_id=entry.capability_id,
        bundle_version=entry.bundle.capability.bundle_version,
        contract_version=entry.bundle.contract.contract_version,
        bundle_content_hash=entry.content_hash,
        execution_identity=(
            None if entry.execution_identity is None else entry.execution_identity.digest
        ),
        implementation=implementation.value,
        surface=surface,
    )


def classify_optimization_profile(
    target: CapabilityEntry,
    *,
    declaration: OptimizationDeclarationV3 | None,
    strategy_entries: Mapping[str, CapabilityEntry] | None = None,
    evaluator: CapabilityEntry | None = None,
    not_applicable_reason: str | None = None,
) -> OptimizationProfileV3:
    """Classify one capability without executing or inventing scientific evidence."""

    _require_admitted(target)
    target_identity = identity_from_capability_entry(target)
    if not_applicable_reason is not None:
        if declaration is not None or strategy_entries is not None or evaluator is not None:
            raise _error(
                "optimization.classification_conflict",
                "not-applicable classification cannot include optimization inputs",
                target.capability_id,
            )
        return OptimizationProfileV3(
            status="not_applicable",
            target=target_identity,
            reason_code=not_applicable_reason,
        )
    if declaration is None:
        raise _error(
            "optimization.classification_required",
            "capability requires an optimization declaration or not-applicable reason",
            target.capability_id,
        )
    _validate_strategy_bindings(target, declaration.strategies, strategy_entries)
    if evaluator is not None:
        _require_admitted(evaluator, evaluator=True)
        if evaluator.capability_id == target.capability_id:
            raise _error(
                "optimization.evaluator_not_independent",
                "optimization evaluator must differ from its target",
                target.capability_id,
                evaluator_id=evaluator.capability_id,
            )
    if declaration.benchmark is None:
        return OptimizationProfileV3(
            status="eligible",
            target=target_identity,
            reason_code="optimization.benchmark_required",
        )
    if evaluator is None:
        return OptimizationProfileV3(
            status="eligible",
            target=target_identity,
            reason_code="optimization.evaluator_required",
        )
    contract = OptimizationContractV3(
        **declaration.model_dump(),
        target=target_identity,
        evaluator=EvaluatorContractV3(identity=identity_from_capability_entry(evaluator)),
    )
    return OptimizationProfileV3(status="enabled", target=target_identity, contract=contract)


def _validate_strategy_bindings(
    target: CapabilityEntry,
    references: tuple[StrategyReferenceV3, ...],
    entries: Mapping[str, CapabilityEntry] | None,
) -> None:
    expected = {reference.kind for reference in references}
    if entries is None or set(entries) != expected:
        raise _error(
            "optimization.strategy_binding_mismatch",
            "strategy entries must exactly match the declared strategy kinds",
            target.capability_id,
        )
    by_kind = {reference.kind: reference for reference in references}
    for kind in sorted(expected):
        entry = entries[kind]
        if entry.status is not CapabilityStatus.ADMITTED:
            raise _error(
                "optimization.strategy_not_admitted",
                "optimization strategies must be admitted capabilities",
                target.capability_id,
                strategy_kind=kind,
                strategy_capability_id=entry.capability_id,
            )
        try:
            actual_identity = identity_from_capability_entry(entry)
        except OrganelleContractError as error:
            raise _error(
                "optimization.strategy_binding_mismatch",
                "strategy entry identity is internally inconsistent",
                target.capability_id,
                strategy_kind=kind,
                strategy_capability_id=entry.capability_id,
            ) from error
        if actual_identity != by_kind[kind].identity:
            raise _error(
                "optimization.strategy_binding_mismatch",
                "strategy entry identity does not match its declaration",
                target.capability_id,
                strategy_kind=kind,
                strategy_capability_id=entry.capability_id,
            )


def contract_v3_from_v2(
    target: CapabilityEntry,
    legacy: OptimizationContract,
    *,
    baseline: BaselineCandidateV3,
    benchmark: BenchmarkContractV3,
    evaluator: CapabilityEntry,
    objectives: tuple[ObjectiveV3, ...],
    strategy_entries: Mapping[OptimizationStrategy, CapabilityEntry],
    constraints: tuple[MetricConstraintV3, ...] = (),
    priors: tuple[PriorCandidateV3, ...] = (),
    adoption: AdoptionPolicyV3,
    budget: OptimizationBudgetV3,
    repeats: RepeatPolicyV3,
) -> OptimizationContractV3:
    """Explicitly supplement a v2 contract without mutating or guessing its science."""

    _require_admitted(target)
    target_identity = identity_from_capability_entry(target)
    legacy_identity = (
        legacy.target_capability_id,
        legacy.target_bundle_version,
        legacy.target_contract_version,
        legacy.bundle_content_hash,
        legacy.execution_identity,
    )
    current_identity = (
        target_identity.capability_id,
        target_identity.bundle_version,
        target_identity.contract_version,
        target_identity.bundle_content_hash,
        target_identity.execution_identity,
    )
    if legacy_identity != current_identity:
        raise _error(
            "optimization.identity_mismatch",
            "v2 contract identity must match the admitted target",
            target.capability_id,
        )
    expected_strategies = set(legacy.strategies)
    if (
        set(strategy_entries) != expected_strategies
        or repeats.seed != legacy.seed
        or budget.max_trials > legacy.budget.max_trials
        or budget.parallelism > legacy.budget.parallelism
    ):
        raise _error(
            "optimization.v2_upgrade_incomplete",
            "v2 upgrade requires exact strategies, seed, and non-expanding hard budget",
            target.capability_id,
        )
    strategies: list[StrategyReferenceV3] = []
    for kind in legacy.strategies:
        strategy_entry = strategy_entries[kind]
        if strategy_entry.status is not CapabilityStatus.ADMITTED:
            raise _error(
                "optimization.v2_upgrade_incomplete",
                "v2 strategy mappings require admitted capability entries",
                target.capability_id,
                strategy_kind=kind,
                strategy_capability_id=strategy_entry.capability_id,
            )
        strategies.append(
            StrategyReferenceV3(
                kind=kind,
                identity=identity_from_capability_entry(strategy_entry),
            )
        )
    _require_admitted(evaluator, evaluator=True)
    if evaluator.capability_id == target.capability_id:
        raise _error(
            "optimization.evaluator_not_independent",
            "optimization evaluator must differ from its target",
            target.capability_id,
            evaluator_id=evaluator.capability_id,
        )
    return OptimizationContractV3(
        target=target_identity,
        parameters=legacy.parameters,
        baseline=baseline,
        priors=priors,
        objectives=objectives,
        constraints=constraints,
        strategies=tuple(strategies),
        budget=budget,
        repeats=repeats,
        adoption=adoption,
        benchmark=benchmark,
        evaluator=EvaluatorContractV3(identity=identity_from_capability_entry(evaluator)),
    )


def _require_admitted(entry: CapabilityEntry, *, evaluator: bool = False) -> None:
    if entry.status is CapabilityStatus.ADMITTED:
        return
    code = "optimization.evaluator_not_admitted" if evaluator else "capability.not_admitted"
    raise _error(
        code,
        "optimization profiles require admitted capabilities",
        entry.capability_id,
        **({"evaluator_id": entry.capability_id} if evaluator else {}),
    )


def _error(code: str, message: str, capability_id: str, **details: str) -> OrganelleContractError:
    return OrganelleContractError(
        code=code,
        message=message,
        details={"capability_id": capability_id, **details},
    )


__all__ = [
    "classify_optimization_profile",
    "contract_v3_from_v2",
    "identity_from_capability_entry",
]
