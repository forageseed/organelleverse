"""Managed, lossless OVASM graph union with explicit source membership.

Exact k-mer sharing is not evidence of orthology across distant species. This
entry point deliberately does not turn unobserved sequence into a PAV zero.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from .._ovasm import run_pan
from ..core.artifacts import ArtifactRef
from ..core.result import OrganelleResult
from ..runtime import (create_staged_run, managed_run_path, publish_staged_result,
                       remove_managed_entry, verify_staged_artifacts)
from ._contract import make_provenance, parameters_hash
from ._gfa import load_gfa
from .assembly_cohort import AssemblyCohort, verify_source

_OP = "pangenome.build_exact_pan"
_RC = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")


def _membership(graph: Path, mapping: Path, sources: dict[str, Path], output: Path) -> int:
    """Verify every source segment reconstruction, including segments off paths."""
    pan = load_gfa(graph)
    originals = {sample: load_gfa(path) for sample, path in sources.items()}
    expected = {(sample, name) for sample, g in originals.items() for name in g.segments}
    observed = set()
    holders: dict[str, set[str]] = {}
    with mapping.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["sample"], row["segment"])
            if key not in expected or key in observed:
                raise ValueError(f"duplicate or unknown source segment: {key}")
            observed.add(key)
            pieces = []
            for step in row["pieces"].split(","):
                name, orientation = step[:-1], step[-1:]
                if orientation not in {"+", "-"} or name not in pan.segments:
                    raise ValueError(f"invalid mapped piece: {step}")
                sequence = pan.segments[name].sequence
                if sequence is None:
                    raise ValueError("exact pan graph requires stored sequences")
                pieces.append(sequence if orientation == "+" else sequence.translate(_RC)[::-1])
                holders.setdefault(name, set()).add(row["sample"])
            original = originals[key[0]].segments[key[1]].sequence
            if "".join(pieces).upper() != original.upper() or int(row["length"]) != len(original):
                raise ValueError(f"source segment does not reconstruct: {key}")
    if observed != expected or set(holders) != set(pan.segments):
        raise ValueError("source mapping is incomplete")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["node", "sample", "observed_source_membership"])
        for node, samples in holders.items():
            writer.writerows((node, sample, 1) for sample in sorted(samples))
    return len(observed)


def build_exact_pan(cohort: AssemblyCohort, *, group: str, k: int = 63) -> OrganelleResult:
    """Build one explicitly selected group; retain every cohort member in evidence.

    All alternatives and unplaced graph segments are retained. Source membership
    reports observed assembly content, not read-supported absence or homology.
    """
    if not 15 <= k <= 63 or k % 2 == 0:
        raise ValueError("k must be odd and between 15 and 63")
    members = [m for m in cohort.members if m.group == group]
    included = [m for m in members if m.graph is not None and m.state in {"representative", "graph_only"}]
    if len(included) < 2:
        raise ValueError("a pan group requires at least two eligible biological samples")
    for m in included:
        verify_source(m.graph)
    parameters = {"group": group, "k": k, "contract": 1, "organelle": cohort.organelle,
                  "members": [{"sample": m.sample, "group": m.group, "species": m.species, "result": m.result_id,
                               "state": m.state, "limitations": m.limitations,
                               "graph": m.graph.object_id if m.graph else None} for m in cohort.members]}
    # Reuse the suite's existing parameter/artifact identity and run publication.
    digest = parameters_hash(parameters)
    completed = managed_run_path(_OP, f"sha256-{digest}")
    if completed.exists():
        record = json.loads((completed / "run-record.json").read_text(encoding="utf-8"))
        if record["parameters_hash"] != digest:
            raise ValueError("existing exact-pan record differs from requested cohort")
        base = OrganelleResult.model_validate(record["result"])
        artifact = ArtifactRef.from_path(completed / "run-record.json", kind="pangenome_run_record", format="json")
        value = base.model_copy(update={"artifacts": (*base.artifacts, artifact.model_copy(update={"uri": "run-record.json"}))})
        verify_staged_artifacts(value, completed)
        return value.model_copy(update={"artifacts": tuple(a.model_copy(update={"uri": str(completed / a.uri)}) for a in value.artifacts)})
    staging = create_staged_run(_OP, f"sha256-{digest}")
    try:
        inputs = staging / "inputs"
        inputs.mkdir()
        sources = {}
        for m in included:
            destination = inputs / f"{m.sample}.gfa"
            shutil.copyfile(m.graph.resolve(), destination)
            verify_source(m.graph.model_copy(update={"uri": str(destination)}))
            sources[m.sample] = destination
        graph = staging / "pan.gfa"
        report = run_pan(list(sources.values()), out_gfa=graph, out_json=staging / "pan.json", k=k)
        count = _membership(graph, staging / "pan.segments.tsv", sources, staging / "source_membership.tsv")
        if report["verified_segments"] != count:
            raise ValueError("native pan validation does not cover every source segment")
        # The graph and mapping reconstruct the inputs; no redundant sequence snapshot.
        shutil.rmtree(inputs)
        (staging / "pan.inputs.txt").unlink()
        (staging / "cohort.json").write_text(cohort.model_dump_json(indent=2) + "\n", encoding="utf-8")
        kinds = {"pan.gfa": ("pangenome_graph", "gfa"), "pan.json": ("pangenome_validation", "json"),
                 "pan.segments.tsv": ("pangenome_source_segments", "tsv"),
                 "pan.links.tsv": ("pangenome_source_links", "tsv"),
                 "source_membership.tsv": ("pangenome_source_membership", "tsv"),
                 "cohort.json": ("pangenome_cohort", "json")}
        artifacts = tuple(ArtifactRef.from_path(staging / name, kind=kind, format=fmt).model_copy(update={"uri": name})
                          for name, (kind, fmt) in kinds.items())
        result = OrganelleResult(
            operation_id=_OP, operation_version="0.1", scope=cohort.organelle, status="warning",
            summary_text=f"Exact graph union of {len(included)}/{len(members)} declared samples in {group}; biological absence and cross-species homology are not assessed.",
            metrics={"declared_samples": len(members), "included_samples": len(included),
                     "graph_only_samples": sum(m.state == "graph_only" for m in included),
                     "verified_source_segments": count},
            flags=("graph_built", "exact_sequence_sharing", "absence_not_assessed", "homology_not_assessed"),
            artifacts=artifacts,
            provenance=make_provenance(operation_id=_OP, parameters=parameters,
                                       input_object_ids=tuple(m.result_id for m in members),
                                       input_artifact_hashes=tuple(m.graph.sha256 for m in included),
                                       requested_backend="ovasm", actual_backend="ovasm", attempted_backends=("ovasm",)),
        )
        record = staging / "run-record.json"
        record.write_text(json.dumps({"parameters_hash": digest, "result": result.model_dump(mode="json")}, indent=2) + "\n", encoding="utf-8")
        record_ref = ArtifactRef.from_path(record, kind="pangenome_run_record", format="json").model_copy(update={"uri": record.name})
        return publish_staged_result(result.model_copy(update={"artifacts": (*artifacts, record_ref)}), staging, completed)
    finally:
        if staging.exists():
            remove_managed_entry(staging)
