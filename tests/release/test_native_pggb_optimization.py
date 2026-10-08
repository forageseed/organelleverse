"""Real, admitted PGGB optimization and reuse gate on synthetic ancestry truth.

This is a software regression benchmark, not validation on biological samples.
"""

import json
import shutil

import pytest

from organelleverse.capabilities.discovery import discover_capabilities
from organelleverse.capabilities.verification import (
    LocalVerificationEnvironment,
    VerificationStore,
    verify_capability,
)
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.optimization.auto import AutoOptimizationRequest, OperationInvocation
from organelleverse.optimization.contracts_v3 import OptimizationContractV3
from organelleverse.optimization.native_runtime import (
    GraphStudyConfiguration,
    NativeOptimizationRuntime,
)
from organelleverse.optimization.profiles_v3 import identity_from_capability_entry
from organelleverse.pangenome.benchmark_evaluator import GraphBenchmarkTruth
from organelleverse.pangenome.optimization_runtime import (
    _benchmark_genomes,
    capture_graph_environment,
)
from organelleverse.plugin_experiments.store import ExperimentStore
from tests.pangenome.benchmark_fixture import case

pytestmark = pytest.mark.integration


def benchmark_contract(index, truth):
    ids = {
        name: identity_from_capability_entry(index.describe(name)).model_dump(mode="json")
        for name in (
            "pangenome.build_graph",
            "pangenome.evaluate_graph_benchmark",
            "optimization.grid",
        )
    }
    contract = OptimizationContractV3.model_validate(
        {
            "target": ids["pangenome.build_graph"],
            "parameters": [
                {"name": "identity", "kind": "categorical", "values": [90, 95]},
                {"name": "segment_length", "kind": "categorical", "values": [5000, 500]},
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
            "constraints": [
                {
                    "name": "exact_input_paths",
                    "metric_pointer": "/metrics/optimization_evaluation/exact_input_paths",
                    "operator": "ge",
                    "threshold": 1.0,
                }
            ],
            "strategies": [{"kind": "grid", "identity": ids["optimization.grid"]}],
            "budget": {
                "max_trials": 16,
                "parallelism": 1,
                "max_wall_time_seconds": 900,
                "max_cpu_time_seconds": 1800,
                "max_peak_memory_bytes": 4 * 1024**3,
            },
            "repeats": {"search_repeats": 2, "validation_repeats": 2, "seed": 41829},
            "adoption": {
                "decision_rule": "all_objectives",
                "minimum_success_rate": 1.0,
                "maximum_failure_rate": 0.0,
                "maximum_wall_time_ratio": 3.0,
                "maximum_peak_memory_ratio": 3.0,
                "approval": "automatic",
            },
            "benchmark": {
                "benchmark_id": "pangenome.synthetic_conserved_homology",
                "benchmark_version": "1.0.0",
                **{
                    name: {
                        "split_id": "synthetic-v1-" + name,
                        "content_hash": "sha256:" + artifact.sha256,
                        "case_count": 1,
                        "artifact_ref": artifact.object_id,
                    }
                    for name, artifact in truth.items()
                },
            },
            "evaluator": {"identity": ids["pangenome.evaluate_graph_benchmark"]},
        }
    )
    return contract


def test_real_admitted_search_validation_final_execution_and_reuse(tmp_path, monkeypatch):
    for name in ("pggb", "wfmash", "seqwish", "smoothxg", "odgi"):
        assert shutil.which(name), f"Real optimization gate requires installed {name}"
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    index = discover_capabilities()
    store = VerificationStore(tmp_path / "home/verifications")
    for name in (
        "pangenome.build_graph",
        "pangenome.evaluate_graph_benchmark",
        "optimization.grid",
    ):
        verify_capability(name, store=store, environment=LocalVerificationEnvironment(index))
    index = discover_capabilities()
    truth = {}
    for split, seed in (("search", 11927), ("validation", 91331)):
        source = tmp_path / (split + ".json")
        source.write_text(json.dumps(case(seed, split + "-sv-v1"), separators=(",", ":")) + "\n")
        truth[split] = ArtifactRef.from_path(source, kind="benchmark_truth", format="json")
    contract = benchmark_contract(index, truth)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    genomes = _benchmark_genomes(
        GraphBenchmarkTruth.model_validate_json(truth["search"].resolve().read_text()),
        inputs,
        "mitochondrion",
    )
    config = GraphStudyConfiguration(
        contract=contract,
        search_truth=truth["search"],
        validation_truth=truth["validation"],
        environment_artifacts=capture_graph_environment(),
        allowed_input_object_ids=tuple(g.object_id for g in genomes),
        applicability_note="Synthetic ancestry regression only; not applicable to biological input cohorts.",
        scope="mitochondrion",
    )
    profiles = tmp_path / "runtime/profiles"
    profiles.mkdir(parents=True)
    (profiles / "pggb.json").write_text(config.model_dump_json())
    service = NativeOptimizationRuntime(
        index_provider=discover_capabilities, root=tmp_path / "runtime"
    ).service()
    request = AutoOptimizationRequest(
        capability_id="pangenome.build_graph",
        mode="always",
        invocation=OperationInvocation(
            input=[g.model_dump(mode="json") for g in genomes],
            parameters={"method": "pggb", "threads": 2},
        ),
    )
    result = service.execute(request, risk_gate_satisfied=True)
    assert result.execution_status == "succeeded", result
    record = ExperimentStore(tmp_path / "runtime/studies/store").get_study(result.study_id)
    assert record.status == "adopted", record.decision
    assert len(record.attempts) == 12
    assert all(attempt.status == "succeeded" for attempt in record.attempts)
    assert all(
        attempt.metrics["/metrics/optimization_evaluation/exact_input_paths"] == 1.0
        for attempt in record.attempts
    )
    assert {attempt.identity.split for attempt in record.attempts} == {"search", "validation"}
    assert all(
        attempt.wall_time_seconds > 0
        and attempt.cpu_time_seconds > 0
        and attempt.peak_memory_bytes > 0
        for attempt in record.attempts
    )
    again = service.execute(request, risk_gate_satisfied=True)
    assert again.decision.decision == "reuse"
    assert again.run_id == result.run_id and again.study_id == result.study_id
    assert len(list((tmp_path / "runtime/studies").glob("study-*/run-*/build-result.json"))) == 12
    assert len(list((tmp_path / "runtime/final-executions").glob("*/result.json"))) == 1
