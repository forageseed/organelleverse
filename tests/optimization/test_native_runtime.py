import json

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import OrganelleContractError
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.optimization.auto import OperationInvocation
from organelleverse.optimization.contracts_v3 import OptimizationContractV3
from organelleverse.optimization.native_runtime import (
    GraphStudyConfiguration,
    NativeOptimizationRuntime,
)
from organelleverse.optimization.profiles_v3 import identity_from_capability_entry
from organelleverse.optimization.study_models import canonical_digest
from organelleverse.pangenome.optimization_runtime import _capture_binding


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    index = discover_capabilities()
    store = VerificationStore(tmp_path / "home/verifications")
    names = ("pangenome.build_graph", "pangenome.evaluate_graph_benchmark", "optimization.grid")
    for name in names:
        verify_capability(name, store=store, environment=LocalVerificationEnvironment(index))
    index = discover_capabilities()
    identities = {
        name: identity_from_capability_entry(index.describe(name)).model_dump(mode="json")
        for name in names
    }
    refs = {}
    for name in ("search", "validation", "executable"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"split": name}))
        refs[name] = ArtifactRef.from_path(path, kind="benchmark_truth", format="json")
    genomes = []
    for name in ("a", "b"):
        path = tmp_path / (name + ".fasta")
        path.write_text(">chr\nACGT\n")
        genomes.append(
            OrganelleGenome(
                organelle="mitochondrion",
                sequence=ArtifactRef.from_path(path, kind="sequence", format="fasta"),
                metadata=OrganelleMetadata(accession=name),
            )
        )
    contract = OptimizationContractV3.model_validate(
        {
            "target": identities[names[0]],
            "parameters": [
                {"name": "identity", "kind": "categorical", "values": [90, 95]},
                {"name": "segment_length", "kind": "categorical", "values": [500, 5000]},
            ],
            "baseline": {"parameters": {"identity": 90, "segment_length": 5000}},
            "objectives": [
                {
                    "name": "homology_f1",
                    "metric_pointer": "/metrics/optimization_evaluation/homology_f1",
                    "direction": "maximize",
                    "minimum_improvement": 0.01,
                    "improvement_mode": "absolute",
                    "aggregation": "mean",
                }
            ],
            "strategies": [{"kind": "grid", "identity": identities[names[2]]}],
            "budget": {
                "max_trials": 16,
                "parallelism": 1,
                "max_wall_time_seconds": 60,
                "max_cpu_time_seconds": 120,
                "max_peak_memory_bytes": 1024**3,
            },
            "repeats": {"search_repeats": 1, "validation_repeats": 2, "seed": 1},
            "adoption": {
                "decision_rule": "all_objectives",
                "minimum_success_rate": 1.0,
                "maximum_failure_rate": 0.0,
                "maximum_wall_time_ratio": 2.0,
                "maximum_peak_memory_ratio": 2.0,
                "approval": "automatic",
            },
            "benchmark": {
                "benchmark_id": "pangenome.fixture",
                "benchmark_version": "1.0.0",
                **{
                    name: {
                        "split_id": name,
                        "content_hash": "sha256:" + refs[name].sha256,
                        "case_count": 1,
                        "artifact_ref": refs[name].object_id,
                    }
                    for name in ("search", "validation")
                },
            },
            "evaluator": {"identity": identities[names[1]]},
        }
    )
    config = GraphStudyConfiguration(
        contract=contract,
        search_truth=refs["search"],
        validation_truth=refs["validation"],
        environment_artifacts=(refs["executable"],),
        allowed_input_object_ids=tuple(g.object_id for g in genomes),
        scope="mitochondrion",
        applicability_note="Unit test configuration, no benchmark execution enabled by this test.",
    )
    profiles = tmp_path / "runtime/profiles"
    profiles.mkdir(parents=True)
    (profiles / "pggb.json").write_text(config.model_dump_json())
    return index, config, genomes, tmp_path / "runtime"


def test_native_configuration_roundtrip_keeps_and_checks_existing_artifact_ids(configuration):
    _, config, _, _ = configuration
    restored = GraphStudyConfiguration.model_validate_json(config.model_dump_json())
    assert restored == config
    raw = config.model_dump(mode="json")
    raw["search_truth"]["object_id"] = raw["validation_truth"]["object_id"]
    with pytest.raises(ValueError, match="does not match computed object_id"):
        GraphStudyConfiguration.model_validate(raw)


def test_native_profile_is_explicit_and_cannot_apply_to_another_cohort(configuration):
    index, config, genomes, root = configuration
    runtime = NativeOptimizationRuntime(index_provider=lambda: index, root=root)
    assert runtime.profiles()["pangenome.build_graph"].status == "enabled"
    invocation = OperationInvocation(
        input=[g.model_dump(mode="json") for g in genomes],
        parameters={"method": "pggb", "threads": 1},
    )
    assert runtime._inputs("pangenome.build_graph", invocation)[1] == genomes
    changed = genomes[0].evolve(metadata=OrganelleMetadata(accession="different-cohort"))
    invalid = invocation.model_copy(
        update={"input": [changed.model_dump(mode="json"), genomes[1].model_dump(mode="json")]}
    )
    with pytest.raises(OrganelleContractError, match="different declared input cohort"):
        runtime._inputs("pangenome.build_graph", invalid)
    # Immutable service configuration prevents a profile changing between
    # decision and final execution. Restart explicitly to load a new profile.
    (root / "profiles/pggb.json").write_text("{}")
    assert runtime.configurations()["pangenome.build_graph"] == config


def test_unconfigured_native_modules_remain_eligible(configuration, tmp_path):
    index, _, _, _ = configuration
    runtime = NativeOptimizationRuntime(index_provider=lambda: index, root=tmp_path / "empty")
    assert runtime.profiles()["pangenome.build_graph"].status == "eligible"


def test_study_capture_reuses_canonical_numeric_identity_and_rejects_changes(tmp_path):
    payload = {"threads": 2, "method": "pggb"}
    source = tmp_path / "fixed.json"
    binding = _capture_binding(payload, source)
    assert binding.content_digest == canonical_digest(payload)
    assert _capture_binding(payload, source) == binding
    with pytest.raises(ValueError, match="cannot be changed"):
        _capture_binding({"threads": 3, "method": "pggb"}, source)



def test_adopted_native_reuse_rechecks_final_artifact_bytes(configuration):
    from datetime import UTC, datetime

    from organelleverse.core.result import OrganelleResult
    from organelleverse.optimization.strategies import candidate_digest
    from organelleverse.plugin_experiments.adaptive_models import FinalExecutionLink

    index, config, _, root = configuration
    runtime = NativeOptimizationRuntime(index_provider=lambda: index, root=root)
    directory = root / "final-executions/run-fixture"
    directory.mkdir(parents=True)
    graph = directory / "graph.gfa"
    graph.write_text("S\t1\tACGT\n")
    artifact = ArtifactRef.from_path(graph, kind="pangenome_graph", format="gfa")
    result = OrganelleResult(operation_id="pangenome.build_graph", status="ok", scope="mitochondrion", artifacts=(artifact,))
    (directory / "result.json").write_text(result.model_dump_json())
    parameters = dict(config.contract.baseline.parameters)
    (directory / "selection.json").write_text(json.dumps({"capability_id": "pangenome.build_graph", "parameters": parameters,
        "contract_digest": config.contract.digest, "result_id": result.object_id}))
    link = FinalExecutionLink(run_receipt_id="run-fixture", executed_parameter_digest=candidate_digest(parameters),
                             linked_at=datetime.now(UTC), execution_status="succeeded")
    runtime.validate_final_link(link)
    graph.write_text("S\t1\tTGCA\n")
    with pytest.raises(OrganelleContractError, match="cannot be reused"):
        runtime.validate_final_link(link)
