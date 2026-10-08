"""Framework-neutral Agent projection tests (Plugin-04, Task 3)."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from organelleverse.capabilities.index import CapabilityIndex
from organelleverse.capabilities.trust import TrustStore
from organelleverse.core.errors import OrganelleInputError
from organelleverse.operations import OperationRegistry, SideEffect
from organelleverse.operations.catalog import load_release_catalog
from organelleverse.operations.spec import OperationSpec
from organelleverse.optimization.auto import (
    AutoOptimizationDecision,
    AutoOptimizationRequest,
)
from organelleverse.optimization.contracts_v3 import OptimizationProfileV3
from organelleverse.optimization.profiles_v3 import identity_from_capability_entry
from organelleverse.plugin_experiments import ExperimentService, ExperimentStore
from organelleverse.plugin_experiments.agent_tools import CapabilityAgentToolCatalog
from organelleverse.plugin_experiments.auto_service import ExecuteBestResult
from organelleverse.plugin_runs import PluginRunService, PluginRunStore
from tests.plugin_experiments.conftest import (
    TRIAL_PLUGIN,
    PluginEnvironment,
    wait_for_experiment,
)

_HASH_A = "sha256:" + "a" * 64


def _target_only(index: CapabilityIndex) -> CapabilityIndex:
    return CapabilityIndex(
        entries=tuple(
            entry.model_copy(
                update={
                    "parameter_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    }
                }
            )
            for entry in index.entries
            if entry.capability_id == "demo.target"
        )
    )


class _AutoService:
    def __init__(self) -> None:
        self.calls: list[tuple[AutoOptimizationRequest, bool]] = []

    def execute(
        self, request: AutoOptimizationRequest, *, risk_gate_satisfied: bool = False
    ) -> ExecuteBestResult:
        self.calls.append((request, risk_gate_satisfied))
        return ExecuteBestResult(
            decision=AutoOptimizationDecision(
                decision="ask",
                reason_code="optimization.profile_eligible",
                capability_id=request.capability_id,
                contract_digest=None,
                benchmark_digest=None,
                invocation_digest=_HASH_A,
                fixed_parameters_digest=_HASH_A,
                approval_required=True,
            )
        )


def _catalog(
    env: PluginEnvironment,
    tmp_path: Path,
    *,
    profile_provider: Callable[[], Mapping[str, OptimizationProfileV3]] | None = None,
    auto_service: object | None = None,
    risk_gate_satisfied: bool = False,
) -> tuple[CapabilityAgentToolCatalog, ExperimentService, PluginRunService]:
    service = ExperimentService(
        index_provider=lambda: env.index,
        trust_store=env.trust_store,
        store=ExperimentStore(tmp_path / "experiments"),
    )
    run_service = PluginRunService(
        index_provider=lambda: env.index,
        trust_store=env.trust_store,
        store=PluginRunStore(tmp_path / "runs"),
    )
    return (
        CapabilityAgentToolCatalog(
            index_provider=lambda: env.index,
            trust_store=env.trust_store,
            experiment_service=service,
            run_service=run_service,
            granted_side_effects=frozenset(),
            profile_provider=profile_provider,  # type: ignore[arg-type]
            auto_service=auto_service,  # type: ignore[arg-type]
            risk_gate_satisfied=risk_gate_satisfied,
        ),
        service,
        run_service,
    )


def test_execute_best_projection_is_exact_profile_bound_and_additive(tmp_path: Path) -> None:
    from tests.plugin_experiments.test_auto_service import _fixture

    index, enabled, _ = _fixture(tmp_path)
    environment = PluginEnvironment(tmp_path / "plugin")
    environment.index = _target_only(index)
    service = _AutoService()
    catalog, _, _ = _catalog(
        environment,
        tmp_path,
        profile_provider=lambda: {"demo.target": enabled},
        auto_service=service,
    )

    tools = {tool.name: tool for tool in catalog.list()}
    execute_best = tools["execute_best__demo__target"]
    assert execute_best.action == "execute_best"
    assert execute_best.invocation_style == "operation"

    eligible = OptimizationProfileV3(
        status="eligible", target=enabled.target, reason_code="optimization.evaluator_missing"
    )
    not_applicable = OptimizationProfileV3(
        status="not_applicable",
        target=enabled.target,
        reason_code="optimization.not_scientifically_tunable",
    )
    for profile, visible in ((eligible, True), (not_applicable, False)):
        candidate, _, _ = _catalog(
            environment,
            tmp_path / profile.status,
            profile_provider=lambda profile=profile: {"demo.target": profile},
            auto_service=service,
        )
        assert ("execute_best__demo__target" in {tool.name for tool in candidate.list()}) is visible

    drifted = eligible.model_copy(
        update={
            "target": eligible.target.model_copy(
                update={"bundle_content_hash": "sha256:" + "f" * 64}
            )
        }
    )
    drifted_catalog, _, _ = _catalog(
        environment,
        tmp_path / "drifted",
        profile_provider=lambda: {"demo.target": drifted},
        auto_service=service,
    )
    assert "execute_best__demo__target" not in {tool.name for tool in drifted_catalog.list()}


def test_execute_best_schema_is_closed_surface_specific_and_keeps_legacy_tools(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    from tests.plugin_experiments.test_auto_service import _fixture

    baseline, _, _ = _catalog(plugin_env, tmp_path / "baseline")
    baseline_tools = tuple(tool.model_dump(mode="json") for tool in baseline.list())
    empty, _, _ = _catalog(
        plugin_env,
        tmp_path / "empty",
        profile_provider=lambda: {},
        auto_service=_AutoService(),
    )
    assert tuple(tool.model_dump(mode="json") for tool in empty.list()) == baseline_tools

    index, profile, _ = _fixture(tmp_path / "native")
    native = PluginEnvironment(tmp_path / "native-env")
    native.index = _target_only(index)
    catalog, _, _ = _catalog(
        native,
        tmp_path / "native-catalog",
        profile_provider=lambda: {"demo.target": profile},
        auto_service=_AutoService(),
    )
    schema = {tool.name: tool for tool in catalog.list()}["execute_best__demo__target"].input_schema
    properties = cast(dict[str, object], schema["properties"])
    assert "capability_id" not in properties
    assert schema["additionalProperties"] is False
    for forbidden in (
        "approved",
        "approval",
        "risk",
        "score",
        "adopted_parameters",
        "contract_digest",
    ):
        assert forbidden not in properties
    validator = Draft202012Validator(schema)
    validator.validate(
        {
            "invocation": {
                "kind": "operation",
                "input": {"sample": "A"},
                "parameters": {},
            }
        }
    )
    with pytest.raises(JsonSchemaValidationError):
        validator.validate({"invocation": {"kind": "plugin", "inputs": {}, "parameters": {}}})

    plugin_profile = OptimizationProfileV3(
        status="eligible",
        target=identity_from_capability_entry(plugin_env.entry),
        reason_code="optimization.benchmark_required",
    )
    plugin_catalog, _, _ = _catalog(
        plugin_env,
        tmp_path / "plugin-catalog",
        profile_provider=lambda: {plugin_env.entry.capability_id: plugin_profile},
        auto_service=_AutoService(),
    )
    plugin_tool = {tool.name: tool for tool in plugin_catalog.list()}[
        "execute_best__demo__experiment"
    ]
    assert plugin_tool.invocation_style == "plugin"
    plugin_validator = Draft202012Validator(plugin_tool.input_schema)
    plugin_validator.validate({"invocation": {"kind": "plugin", "inputs": {}, "parameters": {}}})
    with pytest.raises(JsonSchemaValidationError):
        plugin_validator.validate(
            {"invocation": {"kind": "operation", "input": None, "parameters": {}}}
        )


def test_execute_best_invoke_revalidates_current_identity_and_delegates_browser_safe(
    tmp_path: Path,
) -> None:
    from tests.plugin_experiments.test_auto_service import _fixture

    index, profile, _ = _fixture(tmp_path)
    environment = PluginEnvironment(tmp_path / "environment")
    environment.index = _target_only(index)
    profiles = {"demo.target": profile}
    service = _AutoService()
    catalog, _, _ = _catalog(
        environment,
        tmp_path / "catalog",
        profile_provider=lambda: profiles,
        auto_service=service,
        risk_gate_satisfied=True,
    )
    name = "execute_best__demo__target"
    catalog.list()
    result = catalog.invoke(
        name,
        {
            "invocation": {
                "kind": "operation",
                "input": {"sample": "A"},
                "parameters": {},
            }
        },
    )
    assert isinstance(result, dict)
    assert result["decision"]["decision"] == "ask"  # type: ignore[index]
    assert result["run_id"] is None
    assert service.calls[0][0].capability_id == "demo.target"
    assert service.calls[0][1] is True

    profiles["demo.target"] = OptimizationProfileV3(
        status="eligible",
        target=profile.target.model_copy(update={"bundle_content_hash": "sha256:" + "f" * 64}),
        reason_code="optimization.identity_changed",
    )
    with pytest.raises(OrganelleInputError) as failure:
        catalog.invoke(name, {"invocation": {"kind": "operation", "input": None, "parameters": {}}})
    assert failure.value.code == "input.unknown_agent_tool"
    assert len(service.calls) == 1


def test_plugin_run_projection_exposes_only_declared_inputs_and_parameters(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    catalog, _, _ = _catalog(plugin_env, tmp_path)

    tools = {tool.name: tool for tool in catalog.list()}

    run = tools["run__demo__experiment"]
    assert run.invocation_style == "plugin"
    assert run.description.startswith("Segment organelles")
    assert {key: value for key, value in run.input_schema.items() if key != "examples"} == {
        "type": "object",
        "properties": {
            "inputs": {
                "type": "object",
                "properties": {"images": {"type": "string"}},
                "required": ["images"],
                "additionalProperties": False,
            },
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
                    "probe_log": {"type": "string", "default": ""},
                    "fail_at": {"type": "number", "default": -1.0},
                },
                "additionalProperties": False,
            },
        },
        "required": ["inputs"],
        "additionalProperties": False,
    }
    assert run.input_schema["examples"] == ["Segment mitochondria in this image directory."]
    rendered = str(run.input_schema)
    for forbidden in (
        "masks_dir",
        "output",
        "trust",
        "bundle_root",
        "managed",
        "environment",
        "destination",
    ):
        assert forbidden not in rendered


def test_optimization_projection_is_conditional_and_bounded(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    catalog, _, _ = _catalog(plugin_env, tmp_path)
    tools = {tool.name: tool for tool in catalog.list()}

    optimize = tools["optimize__demo__experiment"]
    assert optimize.action == "optimize"
    schema = optimize.input_schema
    properties = cast(dict[str, object], schema["properties"])
    assert schema["required"] == ["inputs"]
    assert schema["additionalProperties"] is False
    assert properties["strategy"] == {
        "type": "string",
        "enum": ["explicit", "random", "agent"],
        "default": "explicit",
    }
    assert properties["seed"] == {"type": "integer", "default": 0}
    assert properties["rationale"] == {
        "type": "string",
        "minLength": 1,
        "maxLength": 2000,
        "pattern": r"\S",
    }
    assert cast(dict[str, object], properties["candidates"])["maxItems"] == 3
    branches = cast(list[dict[str, object]], schema["oneOf"])
    assert [
        cast(dict[str, object], branch.get("properties", {})).get("strategy") for branch in branches
    ] == [
        {"const": "explicit"},
        {"const": "random"},
        {"const": "agent"},
    ]
    validator = Draft202012Validator(schema)
    invalid = (
        {
            "inputs": {"images": "/data/images"},
            "strategy": "explicit",
            "candidates": [{"threshold": 0.5}],
            "rationale": "not permitted",
        },
        {
            "inputs": {"images": "/data/images"},
            "strategy": "random",
            "candidates": [{"threshold": 0.5}],
        },
        {
            "inputs": {"images": "/data/images"},
            "strategy": "agent",
            "candidates": [{"threshold": 0.5}],
        },
    )
    for payload in invalid:
        with pytest.raises(JsonSchemaValidationError):
            validator.validate(payload)


def test_plugin_without_optimization_has_no_optimization_tool(tmp_path: Path) -> None:
    environment = PluginEnvironment(tmp_path, optimization=False)

    catalog, _, _ = _catalog(environment, tmp_path)
    names = {tool.name for tool in catalog.list()}

    assert "run__demo__experiment" in names
    assert "optimize__demo__experiment" not in names


def test_invoke_matches_direct_plugin_and_experiment_service(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    catalog, service, run_service = _catalog(plugin_env, tmp_path)
    arguments = {"inputs": {"images": str(plugin_env.images)}, "parameters": {"threshold": 0.25}}

    agent_result = catalog.invoke("run__demo__experiment", arguments)
    assert isinstance(agent_result, dict)
    direct = run_service.get(cast(str, agent_result["run_id"]))
    assert direct.result is not None
    result = cast(dict[str, object], agent_result["result"])
    assert agent_result["ok"] is True
    assert agent_result["operation_id"] == "demo.experiment"
    assert agent_result["status"] == "succeeded"
    assert agent_result["l6_run_id"] == direct.l6_run_id
    assert run_service.list() == (direct,)
    expected = direct.result.model_dump(mode="json")
    expected["artifacts"] = [
        {
            "artifact_id": artifact.object_id,
            "schema_version": artifact.schema_version,
            "kind": artifact.kind,
            "format": artifact.format,
            "media_type": artifact.media_type,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
            "validated": artifact.validated,
        }
        for artifact in direct.result.artifacts
    ]
    cast(dict[str, object], expected["provenance"]).pop("argv")
    assert result == expected
    assert str(plugin_env.images) not in str(agent_result)
    artifacts = cast(list[dict[str, object]], result["artifacts"])
    assert all("uri" not in artifact for artifact in artifacts)

    submitted = catalog.invoke(
        "optimize__demo__experiment",
        {"inputs": {"images": str(plugin_env.images)}, "candidates": [{"threshold": 0.75}]},
    )
    assert not isinstance(submitted, dict)
    completed = wait_for_experiment(service, submitted.experiment_id)
    assert completed.trials[0].parameters == {"threshold": 0.75}

    automatic = catalog.invoke(
        "optimize__demo__experiment",
        {
            "inputs": {"images": str(plugin_env.images)},
            "strategy": "random",
            "seed": 23,
        },
    )
    assert not isinstance(automatic, dict)
    assert automatic.request.strategy == "random"
    assert automatic.request.seed == 23
    assert len(automatic.request.candidates) == 3


def test_released_operation_wins_over_a_same_id_admitted_capability(tmp_path: Path) -> None:
    environment = PluginEnvironment(tmp_path, capability_id="io.read_long_reads")
    service = ExperimentService(
        index_provider=lambda: environment.index,
        trust_store=environment.trust_store,
        store=ExperimentStore(tmp_path / "experiments"),
    )
    catalog = CapabilityAgentToolCatalog(
        index_provider=lambda: environment.index,
        trust_store=environment.trust_store,
        experiment_service=service,
        run_service=PluginRunService(
            index_provider=lambda: environment.index,
            trust_store=environment.trust_store,
            store=PluginRunStore(tmp_path / "runs"),
        ),
        granted_side_effects={SideEffect.READ_FILES},
    )

    matching = [tool for tool in catalog.list() if tool.capability_id == "io.read_long_reads"]

    assert len(matching) == 1
    assert matching[0].name == "run__io__read_long_reads"
    assert matching[0].invocation_style == "operation"
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\n!!!!\n", encoding="utf-8")
    response = catalog.invoke(
        matching[0].name,
        {
            "input": None,
            "parameters": {
                "reads": str(reads),
                "technology": "ont",
                "quality_state": "raw",
            },
        },
    )
    assert isinstance(response, dict)
    assert response["ok"] is True


def test_agent_projection_removes_recursive_absolute_path_values(tmp_path: Path) -> None:
    source = TRIAL_PLUGIN.replace(
        'summary={"threshold": threshold},',
        "summary={\n"
        '            "threshold": threshold,\n'
        '            "work_dir": str(context.work_dir),\n'
        '            "managed_reference": "managed://runs/demo/output",\n'
        '            "file_reference": "file:///tmp/worker-payload.json",\n'
        '            "nested": {"staging": str(Path(outputs["masks"]).parent), "sample": "chloroplast"},\n'
        '            "paths": [str(context.log_path), "safe scientific value"],\n'
        '            "worker_payload": {"path": str(context.work_dir)},\n'
        '            "argv": ["python", str(context.log_path)],\n'
        '            "staging_root": "private",\n'
        '            "plugin_root": "private",\n'
        "        },",
    ).replace(
        'raise RuntimeError("synthetic trial failure")',
        'raise RuntimeError("worker failed at " + str(context.work_dir) + " after staging")',
    )
    environment = PluginEnvironment(tmp_path, source=source)
    catalog, _, _ = _catalog(environment, tmp_path)

    response = catalog.invoke(
        "run__demo__experiment",
        {"inputs": {"images": str(environment.images)}, "parameters": {"threshold": 0.25}},
    )

    assert isinstance(response, dict)
    result = cast(dict[str, object], response["result"])
    assert isinstance(result["object_id"], str)
    assert result["metrics"] == {
        "nested": {"sample": "chloroplast"},
        "paths": ["safe scientific value"],
        "plugin_optimization_score": 0.25,
        "threshold": 0.25,
    }
    assert str(environment.images) not in str(response)
    assert "/tmp" not in str(response)
    assert "managed://" not in str(response)
    assert "file://" not in str(response)
    assert "worker_payload" not in str(response)
    assert "argv" not in str(response)
    assert "staging_root" not in str(response)
    assert "plugin_root" not in str(response)

    failed = catalog.invoke(
        "run__demo__experiment",
        {
            "inputs": {"images": str(environment.images)},
            "parameters": {"threshold": 0.25, "fail_at": 0.25},
        },
    )

    assert isinstance(failed, dict)
    failed_result = cast(dict[str, object], failed["result"])
    assert "/tmp" not in str(failed_result)
    errors = cast(list[dict[str, object]], failed_result["errors"])
    assert errors[0]["message"] == "plugin callable raised an exception"
    assert errors[0]["details"] == {"exception_type": "RuntimeError"}

    error_projection = CapabilityAgentToolCatalog._agent_result_payload(
        {
            "object_id": "result:sha256:safe",
            "errors": [
                {
                    "message": str(tmp_path / "managed" / "worker.log"),
                    "safe_detail": "reconstruct the chloroplast genome",
                }
            ],
        }
    )
    assert error_projection == {
        "object_id": "result:sha256:safe",
        "errors": [{"safe_detail": "reconstruct the chloroplast genome"}],
    }


def test_untrusted_plugin_is_not_projected_or_executed(tmp_path: Path) -> None:
    environment = PluginEnvironment(tmp_path)
    untrusted = environment.entry.model_copy(
        update={"execution_identity": environment.entry.execution_identity}
    )
    environment.trust_store = type(environment.trust_store)(tmp_path / "other-trust.json")
    environment.index = CapabilityIndex(entries=(untrusted,))
    catalog, _, _ = _catalog(environment, tmp_path)

    assert {tool.capability_id for tool in catalog.list()} == {
        operation.operation_id for operation in _release_operations()
    }
    with pytest.raises(OrganelleInputError) as captured:
        catalog.invoke("run__demo__experiment", {"inputs": {"images": str(environment.images)}})
    assert getattr(captured.value, "code", "") == "input.unknown_agent_tool"


def test_previously_projected_plugin_tool_records_revoked_trust(
    plugin_env: PluginEnvironment, tmp_path: Path
) -> None:
    catalog, _, _ = _catalog(plugin_env, tmp_path)
    assert catalog.list()
    plugin_env.trust_store.path.unlink()

    response = catalog.invoke(
        "run__demo__experiment", {"inputs": {"images": str(plugin_env.images)}}
    )

    assert isinstance(response, dict)
    assert response["ok"] is False
    assert response["status"] == "failed"
    assert response["result"] is None
    assert isinstance(response["run_id"], str)


def test_catalog_projects_and_invokes_every_residual_release_operation(tmp_path: Path) -> None:
    index = CapabilityIndex()
    trust_store = TrustStore(tmp_path / "trust.json")
    service = ExperimentService(
        index_provider=lambda: index,
        trust_store=trust_store,
        store=ExperimentStore(tmp_path / "experiments"),
    )
    catalog = CapabilityAgentToolCatalog(
        index_provider=lambda: index,
        trust_store=trust_store,
        experiment_service=service,
        run_service=PluginRunService(
            index_provider=lambda: index,
            trust_store=trust_store,
            store=PluginRunStore(tmp_path / "runs"),
        ),
        granted_side_effects={SideEffect.READ_FILES},
    )
    authoritative = _release_operations()

    tools = catalog.list()
    assert {tool.capability_id for tool in tools} == {
        operation.operation_id for operation in authoritative
    }

    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\n!!!!\n", encoding="utf-8")
    response = catalog.invoke(
        "run__io__read_long_reads",
        {
            "input": None,
            "parameters": {
                "reads": str(reads),
                "technology": "ont",
                "quality_state": "raw",
            },
        },
    )

    assert isinstance(response, dict)
    assert response["ok"] is True
    assert str(reads) not in str(response)


def _release_operations() -> tuple[OperationSpec, ...]:
    registry = OperationRegistry()
    load_release_catalog(registry)
    return registry.list()


def test_projection_cold_import_has_no_optional_agent_or_desktop_kernels() -> None:
    for module in tuple(sys.modules):
        if module.startswith(
            ("organelleverse.plugin_experiments", "deepagents", "langchain", "mcp", "fastapi")
        ):
            sys.modules.pop(module)

    __import__("organelleverse.plugin_experiments")

    assert not any(
        module.startswith(("deepagents", "langchain", "mcp", "fastapi")) for module in sys.modules
    )
