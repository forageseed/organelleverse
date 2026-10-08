"""Public governed optimization contracts."""

from typing import TYPE_CHECKING

from .auto import (
    AutoOptimizationContext,
    AutoOptimizationDecision,
    AutoOptimizationRequest,
    OperationInvocation,
    PluginInvocation,
    decide_auto_optimization,
)
from .candidates import (
    grid_candidates,
    optimization_request_schema,
    random_candidates,
)
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
    BenchmarkSplitV3,
    CapabilityIdentityV3,
    EvaluatorContractV3,
)
from .models import (
    ObjectiveSpec,
    OptimizationBudget,
    OptimizationContract,
    ParameterDomain,
)
from .profiles import contract_from_capability_entry
from .profiles_v3 import (
    classify_optimization_profile,
    contract_v3_from_v2,
    identity_from_capability_entry,
)
from .strategy_models import OpaqueArtifactBinding

if TYPE_CHECKING:
    from .strategy_models import (
        AdaptiveStudyRequest,
        EvaluatorBinding,
        StrategyBinding,
        SuccessiveHalvingConfig,
    )
    from .strategy_registry import StrategyRegistry
    from .study_engine import AdaptiveStudyEngine
    from .study_models import AdoptedStudyIdentity, EvaluationRequest, EvaluatorRunner


def __getattr__(name: str) -> object:
    if name == "AdaptiveStudyEngine":
        from .study_engine import AdaptiveStudyEngine

        return AdaptiveStudyEngine
    if name in {"AdoptedStudyIdentity", "EvaluationRequest", "EvaluatorRunner"}:
        from .study_models import AdoptedStudyIdentity, EvaluationRequest, EvaluatorRunner

        return {
            "AdoptedStudyIdentity": AdoptedStudyIdentity,
            "EvaluationRequest": EvaluationRequest,
            "EvaluatorRunner": EvaluatorRunner,
        }[name]
    if name in {
        "AdaptiveStudyRequest",
        "EvaluatorBinding",
        "StrategyBinding",
        "SuccessiveHalvingConfig",
    }:
        from .strategy_models import (
            AdaptiveStudyRequest,
            EvaluatorBinding,
            StrategyBinding,
            SuccessiveHalvingConfig,
        )

        return {
            "AdaptiveStudyRequest": AdaptiveStudyRequest,
            "EvaluatorBinding": EvaluatorBinding,
            "StrategyBinding": StrategyBinding,
            "SuccessiveHalvingConfig": SuccessiveHalvingConfig,
        }[name]
    if name == "StrategyRegistry":
        from .strategy_registry import StrategyRegistry

        return StrategyRegistry
    raise AttributeError(name)


__all__ = [
    "AdaptiveStudyEngine",
    "AdaptiveStudyRequest",
    "AdoptedStudyIdentity",
    "AdoptionPolicyV3",
    "AutoOptimizationContext",
    "AutoOptimizationDecision",
    "AutoOptimizationRequest",
    "BaselineCandidateV3",
    "BenchmarkContractV3",
    "BenchmarkSplitV3",
    "CapabilityIdentityV3",
    "EvaluationRequest",
    "EvaluatorBinding",
    "EvaluatorContractV3",
    "EvaluatorRunner",
    "MetricConstraintV3",
    "ObjectiveSpec",
    "ObjectiveV3",
    "OpaqueArtifactBinding",
    "OperationInvocation",
    "OptimizationBudget",
    "OptimizationBudgetV3",
    "OptimizationContract",
    "OptimizationContractV3",
    "OptimizationDeclarationV3",
    "OptimizationProfileV3",
    "ParameterDomain",
    "PluginInvocation",
    "PriorCandidateV3",
    "RepeatPolicyV3",
    "StrategyBinding",
    "StrategyReferenceV3",
    "StrategyRegistry",
    "SuccessiveHalvingConfig",
    "classify_optimization_profile",
    "contract_from_capability_entry",
    "contract_v3_from_v2",
    "decide_auto_optimization",
    "grid_candidates",
    "identity_from_capability_entry",
    "optimization_request_schema",
    "random_candidates",
]
