"""Real PGGB benchmark execution inside the governed adaptive-study boundary.

No default biological profile is invented here. A caller supplies an admitted
v3 contract with independently captured search/validation truth, declared
candidate domains, resource limits and adoption thresholds.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from organelleverse.capabilities.index import CapabilityIndex
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.operations.registry import OperationRegistry
from organelleverse.optimization.contracts_v3 import OptimizationContractV3
from organelleverse.optimization.models import _canonical_json
from organelleverse.optimization.profiles_v3 import identity_from_capability_entry
from organelleverse.optimization.strategies import strategy_for
from organelleverse.optimization.strategy_models import (
    AdaptiveStudyRequest,
    EvaluatorBinding,
    OpaqueArtifactBinding,
    StrategyBinding,
)
from organelleverse.optimization.strategy_registry import StrategyRegistry
from organelleverse.optimization.study_models import EvaluationRequest, canonical_digest
from organelleverse.optimization.evidence_models import RepeatEvidence

from .benchmark_evaluator import GraphBenchmarkTruth
from .conformation_benchmark import ConformationTruth
from .taxonomy_benchmark import TaxonomyTruth
from .graph_selection import verified_artifact


def _exact_entry(index, identity):
    entry = index.describe(identity.capability_id)
    if identity_from_capability_entry(entry) != identity:
        raise ValueError("Optimization capability no longer matches its admitted identity")
    return entry


def _benchmark_genomes(truth: GraphBenchmarkTruth, directory: Path, scope: str):
    """Preserve biological samples and independently named input molecules."""
    samples = {}
    for path, sequence in truth.sequences.items():
        parts = path.split("#")
        if len(parts) != 3 or parts[1] != "1" or not parts[0] or not parts[2].isdigit():
            raise ValueError("PGGB benchmark paths must use sample#1#positive-molecule-index")
        samples.setdefault(parts[0], {})[int(parts[2])] = sequence
    genomes = []
    for number, (sample, molecules) in enumerate(samples.items()):
        if set(molecules) != set(range(1, len(molecules) + 1)):
            raise ValueError("Benchmark molecule indices must be contiguous starting at one")
        source = directory / f"benchmark-sample-{number}.fasta"
        source.write_text("".join(f">molecule-{i}\n{molecules[i]}\n" for i in sorted(molecules)))
        genomes.append(
            OrganelleGenome(
                organelle=scope,
                sequence=ArtifactRef.from_path(source, kind="sequence", format="fasta"),
                metadata=OrganelleMetadata(accession=sample),
            )
        )
    return genomes


@dataclass(frozen=True)
class GraphBenchmarkRunner:
    """Pickleable worker; its process-local fresh cache prevents timing cache hits."""

    binding: EvaluatorBinding
    index: CapabilityIndex
    contract: OptimizationContractV3
    search_truth: ArtifactRef
    validation_truth: ArtifactRef
    fixed_parameters: dict
    scope: str
    evidence_root: Path
    environment_artifacts: tuple[ArtifactRef, ...]
    environment_digest: str

    def evaluate(self, request: EvaluationRequest) -> RepeatEvidence:
        import resource

        if (
            request.target != self.contract.target
            or request.evaluator_binding != self.binding
            or request.contract_digest != self.contract.digest
            or request.environment_digest != self.environment_digest
            or request.fixed_parameters_digest != canonical_digest(self.fixed_parameters)
        ):
            raise ValueError("Benchmark dispatch does not match its captured runtime bindings")
        _exact_entry(self.index, request.target)
        _exact_entry(self.index, self.binding.identity)
        _verify_environment(self.environment_artifacts)
        split_name = request.identity.split
        split = getattr(self.contract.benchmark, split_name)
        source = self.search_truth if split_name == "search" else self.validation_truth
        if (
            request.benchmark_split != split
            or split.artifact_ref != source.object_id
            or split.content_hash != "sha256:" + source.sha256
        ):
            raise ValueError("Benchmark dispatch must resolve the exact captured split")
        truth_model = _truth_model(self.binding.identity.capability_id)
        truth = truth_model.model_validate_json(verified_artifact(source).read_text())
        if (
            truth.benchmark_id != self.contract.benchmark.benchmark_id
            or truth.benchmark_version != self.contract.benchmark.benchmark_version
            or split.case_count != 1
        ):
            raise ValueError("Benchmark truth metadata must match its declared one-cohort split")
        directory = self.evidence_root / request.study_id / request.run_id
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "dispatch.json").write_text(request.model_dump_json(indent=2) + "\n")
        started = time.monotonic()
        cpu_started = time.process_time()
        children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        previous_cache = os.environ.get("ORGANELLEVERSE_CACHE_ROOT")
        os.environ["ORGANELLEVERSE_CACHE_ROOT"] = str(directory / "cache")
        metrics = {}
        error = None
        try:
            genomes = _benchmark_genomes(truth, directory, self.scope)
            registry = OperationRegistry(capability_source=self.index.binding_source())
            built = registry.require(request.target.capability_id).invoke(
                genomes,
                {
                    **self.fixed_parameters,
                    **request.parameters,
                },
            )
            if not isinstance(built, OrganelleResult):
                raise ValueError("Graph constructor did not return a native Result")
            (directory / "build-result.json").write_text(built.model_dump_json(indent=2) + "\n")
            if built.status != "ok":
                raise ValueError("Graph constructor failed; no scientific score was assigned")
            bound = built.model_copy(update={"artifacts": (*built.artifacts, source)})
            evaluated = registry.require(self.binding.identity.capability_id).invoke(
                bound, {"truth_artifact_id": source.object_id}
            )
            if not isinstance(evaluated, OrganelleResult) or evaluated.status != "ok":
                raise ValueError("Independent evaluator did not return a successful Result")
            (directory / "evaluation-result.json").write_text(
                evaluated.model_dump_json(indent=2) + "\n"
            )
            measured = evaluated.model_dump(mode="json")["metrics"]["optimization_evaluation"]
            metrics = {
                "/metrics/optimization_evaluation/" + key: value for key, value in measured.items()
            }
            if not set(request.required_metric_pointers).issubset(metrics):
                raise ValueError("Evaluator did not produce every declared metric")
        except Exception as failure:
            # Failure evidence is retained locally; no fabricated numerical score.
            (directory / "failure.json").write_text(
                json.dumps({"error_type": type(failure).__name__, "message": str(failure)}) + "\n"
            )
            error = ErrorDetail(
                code="optimization.graph_benchmark_failed",
                message="Real graph construction or independent evaluation failed",
            )
            metrics = {}
        finally:
            if previous_cache is None:
                os.environ.pop("ORGANELLEVERSE_CACHE_ROOT", None)
            else:
                os.environ["ORGANELLEVERSE_CACHE_ROOT"] = previous_cache
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = RepeatEvidence(
            identity=request.identity,
            status="succeeded" if error is None else "failed",
            run_id=request.run_id,
            seed=request.seed,
            dispatch_digest=request.digest,
            granted_wall_time_seconds=request.remaining_wall_time_seconds,
            granted_cpu_time_seconds=request.remaining_cpu_time_seconds,
            granted_peak_memory_bytes=request.max_peak_memory_bytes,
            metrics=metrics,
            wall_time_seconds=time.monotonic() - started,
            cpu_time_seconds=time.process_time()
            - cpu_started
            + children.ru_utime
            + children.ru_stime
            - children_before.ru_utime
            - children_before.ru_stime,
            peak_memory_bytes=max(
                children.ru_maxrss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            * 1024,
            error=error,
        )
        (directory / "repeat-evidence.json").write_text(result.model_dump_json(indent=2) + "\n")
        return result


_PGGB_TOOLS = ("pggb", "wfmash", "seqwish", "smoothxg", "odgi")


def _truth_model(evaluator_id):
    models = {
        "pangenome.evaluate_graph_benchmark": GraphBenchmarkTruth,
        "pangenome.evaluate_conformation_benchmark": ConformationTruth,
        "pangenome.evaluate_taxonomy_benchmark": TaxonomyTruth,
    }
    if evaluator_id not in models:
        raise ValueError("Graph study requires an admitted independent graph evaluator")
    return models[evaluator_id]


def capture_graph_environment() -> tuple[ArtifactRef, ...]:
    """Capture the actual executables selected by the current managed PATH."""
    artifacts = []
    for name in _PGGB_TOOLS:
        executable = shutil.which(name)
        if executable is None:
            raise ValueError(f"PGGB benchmark requires installed {name}")
        artifacts.append(ArtifactRef.from_path(executable, kind="tool_executable", format="binary"))
    return tuple(artifacts)


def _verify_environment(artifacts: tuple[ArtifactRef, ...]) -> None:
    current = capture_graph_environment()
    if {artifact.object_id for artifact in current} != {
        artifact.object_id for artifact in artifacts
    }:
        raise ValueError("The captured PGGB toolchain differs from the executables selected now")
    for artifact in artifacts:
        verified_artifact(artifact)


def _capture_binding(payload: object, path: Path) -> OpaqueArtifactBinding:
    encoded = json.dumps(
        _canonical_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError("An existing study binding cannot be changed; start a new study")
    else:
        path.write_text(encoded)
    artifact = ArtifactRef.from_path(path, kind="artifact", format="json")
    digest = canonical_digest(payload)
    if digest != "sha256:" + artifact.sha256:
        raise ValueError("Captured optimization payload must use canonical JSON")
    return OpaqueArtifactBinding(artifact_ref=artifact.object_id, content_digest=digest)
