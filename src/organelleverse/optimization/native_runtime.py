"""Configured native optimization through the shared execute-best service.

Profiles are explicit local scientific configurations, never automatically
derived from a tool's self-reported score. Unconfigured capabilities remain
eligible or not applicable in the complete built-in coverage inventory.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, JsonValue, field_validator

from organelleverse.capabilities.index import CapabilityIndex
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleContractError, OrganelleInputError
from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from organelleverse.operations.registry import OperationRegistry
from organelleverse.operations.spec import StrictSpecModel
from organelleverse.plugin_experiments.auto_service import (
    AutoOptimizationService,
    AutoOptimizationSnapshot,
    FinalExecutionReceipt,
    PreparedAdaptiveStudy,
)

from .auto import AutoOptimizationRequest, OperationInvocation, PluginInvocation
from .contracts_v3 import OptimizationContractV3, OptimizationProfileV3
from .coverage import builtin_profiles
from .profiles_v3 import identity_from_capability_entry
from .strategies import candidate_digest


class GraphStudyConfiguration(StrictSpecModel):
    schema_version: Literal["organelleverse.pangenome.optimization-configuration.v1"] = (
        "organelleverse.pangenome.optimization-configuration.v1"
    )
    contract: OptimizationContractV3
    search_truth: ArtifactRef
    validation_truth: ArtifactRef
    environment_artifacts: tuple[ArtifactRef, ...] = Field(min_length=1)
    allowed_input_object_ids: tuple[str, ...] = Field(min_length=2)
    applicability_note: str = Field(min_length=1)
    scope: Literal["mitochondrion", "plastid"]
    strategy_kind: Literal["grid", "random", "space_filling"] = "grid"

    @field_validator("search_truth", "validation_truth", mode="before")
    @classmethod
    def restore_truth(cls, value):
        return ArtifactRef.model_validate(value)

    @field_validator("environment_artifacts", mode="before")
    @classmethod
    def restore_environment(cls, values):
        return tuple(ArtifactRef.model_validate(value) for value in values)


class NativeOptimizationRuntime:
    def __init__(self, *, index_provider: Callable[[], CapabilityIndex], root: Path):
        self.index_provider = index_provider
        self.root = Path(root)
        self._configurations = self._read_configurations()

    def configurations(self) -> dict[str, GraphStudyConfiguration]:
        """Profiles are immutable for this service lifetime; reload by restarting it."""
        return dict(self._configurations)

    def _read_configurations(self) -> dict[str, GraphStudyConfiguration]:
        configured = {}
        for path in sorted((self.root / "profiles").glob("*.json")):
            config = GraphStudyConfiguration.model_validate_json(path.read_text())
            target = config.contract.target.capability_id
            if target != "pangenome.build_graph" or target in configured:
                raise ValueError("Configure exactly one native PGGB profile per target")
            configured[target] = config
        return configured

    def profiles(self) -> Mapping[str, OptimizationProfileV3]:
        index = self.index_provider()
        profiles = builtin_profiles(index)
        for target, config in self.configurations().items():
            identities = (
                config.contract.target,
                config.contract.evaluator.identity,
                *(item.identity for item in config.contract.strategies),
            )
            entries = {entry.capability_id: entry for entry in index.list()}
            if all(
                identity.capability_id in entries
                and identity_from_capability_entry(entries[identity.capability_id]) == identity
                for identity in identities
            ):
                profiles[target] = OptimizationProfileV3(
                    status="enabled", target=config.contract.target, contract=config.contract
                )
        return profiles

    def _inputs(self, capability_id: str, invocation: OperationInvocation | PluginInvocation):
        config = self.configurations().get(capability_id)
        if config is None or not isinstance(invocation, OperationInvocation):
            raise ValueError(
                "This native optimization runtime requires a configured PGGB operation"
            )
        if not isinstance(invocation.input, list):
            raise ValueError("PGGB invocation requires a serialized sequence of Genome objects")
        genomes = [OrganelleGenome.model_validate(value) for value in invocation.input]
        if tuple(genome.object_id for genome in genomes) != config.allowed_input_object_ids:
            raise OrganelleContractError(
                code="optimization.input_outside_benchmark_scope",
                message="This profile is bound to a different declared input cohort; supply an applicable scientific profile",
            )
        if any(genome.organelle != config.scope for genome in genomes):
            raise ValueError("Configured benchmark scope and final input organelle must match")
        return config, genomes

    def prepare(
        self, request: AutoOptimizationRequest, snapshot: AutoOptimizationSnapshot
    ) -> PreparedAdaptiveStudy:
        from organelleverse.pangenome.optimization_runtime import prepare_graph_study

        config, _ = self._inputs(request.capability_id, request.invocation)
        contract = config.contract
        if snapshot.profile.contract != contract:
            raise ValueError("The configured profile changed after the decision snapshot")
        if request.budget is not None:
            contract = contract.model_copy(update={"budget": request.budget})
        tuned = {domain.name for domain in contract.parameters}
        fixed = {
            name: value
            for name, value in request.invocation.parameters.items()
            if name not in tuned
        }
        return prepare_graph_study(
            index=snapshot.index,
            contract=contract,
            search_truth=config.search_truth,
            validation_truth=config.validation_truth,
            input_payload=request.invocation.input,
            fixed_parameters=fixed,
            scope=config.scope,
            evidence_root=self.root / "studies",
            environment_artifacts=config.environment_artifacts,
            strategy_kind=config.strategy_kind,
        )

    def execute_final(
        self,
        capability_id: str,
        invocation: OperationInvocation | PluginInvocation,
        parameters: dict[str, JsonValue],
    ) -> FinalExecutionReceipt:
        config, genomes = self._inputs(capability_id, invocation)
        index = self.index_provider()
        if identity_from_capability_entry(index.describe(capability_id)) != config.contract.target:
            raise ValueError("Final constructor no longer matches the admitted target identity")
        if parameters.get("method") != "pggb":
            raise ValueError("A PGGB optimization profile cannot execute another constructor")
        operation = OperationRegistry(capability_source=index.binding_source()).require(
            capability_id
        )
        result = operation.invoke(genomes, parameters)
        if not isinstance(result, OrganelleResult):
            raise ValueError("Final graph execution did not return a native Result")
        run_id = "run-" + uuid4().hex
        destination = self.root / "final-executions" / run_id
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
        (destination / "selection.json").write_text(
            json.dumps(
                {
                    "capability_id": capability_id,
                    "parameters": parameters,
                    "contract_digest": config.contract.digest,
                    "input_object_ids": [genome.object_id for genome in genomes],
                    "applicability_note": config.applicability_note,
                    "result_id": result.object_id,
                },
                indent=2,
            )
            + "\n"
        )
        return FinalExecutionReceipt(
            run_id=run_id, status="succeeded" if result.status == "ok" else "failed"
        )

    def validate_final_link(self, link) -> None:
        from organelleverse.pangenome.graph_selection import verified_artifact

        directory = self.root / "final-executions" / link.run_receipt_id
        try:
            result = OrganelleResult.model_validate_json((directory / "result.json").read_text())
            selected = json.loads((directory / "selection.json").read_text())
            config = self.configurations()[selected["capability_id"]]
            tuned = {domain.name: selected["parameters"][domain.name] for domain in config.contract.parameters}
            if (selected["result_id"] != result.object_id
                    or selected["contract_digest"] != config.contract.digest
                    or candidate_digest(tuned) != link.executed_parameter_digest
                    or (result.status == "ok") != (link.execution_status == "succeeded")):
                raise ValueError("Final native receipt no longer matches its adopted selection")
            for artifact in result.artifacts:
                verified_artifact(artifact)
        except (ValueError, OSError, KeyError, OrganelleInputError) as error:
            raise OrganelleContractError(code="optimization.final_evidence_changed",
                message="Final execution evidence changed or is unavailable; the result cannot be reused") from error

    def service(self, *, profile_provider=None) -> AutoOptimizationService:
        return AutoOptimizationService(
            index_provider=self.index_provider,
            profile_provider=profile_provider or self.profiles,
            prepared_study_provider=self.prepare,
            final_executor=self.execute_final,
            final_link_validator=self.validate_final_link,
        )


def configured_native_service(
    *, index_provider: Callable[[], CapabilityIndex], root: Path, profile_provider=None
) -> AutoOptimizationService:
    return NativeOptimizationRuntime(index_provider=index_provider, root=root).service(
        profile_provider=profile_provider
    )
