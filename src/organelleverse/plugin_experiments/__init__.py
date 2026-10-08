"""Plugin-04: declared-score experiments and framework-neutral Agent tools.

This package adds no new registry, discovery channel, trust state, execution
provider, run callable, or scientific result type. Experiments invoke the same
admitted Registry binding once per explicit candidate; scores come only from
the plugin's declared ``score_locator`` hook via the reserved L6 metric.
Normal GUI/Agent plugin runs live in :mod:`organelleverse.plugin_runs`.
"""

from organelleverse.plugin_experiments.adaptive_models import (
    AdaptiveStudyRecord,
    AdoptionDecision,
    AttemptIdentity,
    BudgetAccount,
    CandidateAggregate,
    ExecutionSelection,
    FinalExecutionClaim,
    FinalExecutionLink,
    MetricAggregate,
    RepeatEvidence,
)
from organelleverse.plugin_experiments.agent_tools import (
    CapabilityAgentTool,
    CapabilityAgentToolCatalog,
)
from organelleverse.plugin_experiments.models import (
    ExperimentRecord,
    ExperimentRequest,
    ExperimentTrial,
)
from organelleverse.plugin_experiments.service import ExperimentService
from organelleverse.plugin_experiments.store import ExperimentStore

__all__ = [
    "AdaptiveStudyRecord",
    "AdoptionDecision",
    "AttemptIdentity",
    "BudgetAccount",
    "CandidateAggregate",
    "CapabilityAgentTool",
    "CapabilityAgentToolCatalog",
    "ExecutionSelection",
    "ExperimentRecord",
    "ExperimentRequest",
    "ExperimentService",
    "ExperimentStore",
    "ExperimentTrial",
    "FinalExecutionClaim",
    "FinalExecutionLink",
    "MetricAggregate",
    "RepeatEvidence",
]
