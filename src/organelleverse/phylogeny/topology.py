"""IQ-TREE topology tests for competing Newick trees."""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from Bio import Phylo

from ..core.external import run_external
from ..core.result import OrganelleResult
from ._results import artifact_for, failed_result, ok_result, provenance


def _parse_user_tree_table(report: Path) -> list[dict[str, object]]:
    """Read the USER TREES test table from an IQ-TREE .iqtree report."""
    lines = report.read_text(encoding="utf-8", errors="replace").splitlines()
    in_user_trees = False
    header: list[str] | None = None
    rows: list[dict[str, object]] = []
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if upper == "USER TREES":
            in_user_trees = True
            continue
        if not in_user_trees:
            continue
        fields = stripped.split()
        if header is None:
            if "logL" in fields and "p-AU" in fields:
                header = fields
            continue
        if not fields:
            if rows:
                break
            continue
        if not re.fullmatch(r"\d+", fields[0]):
            if rows:
                break
            continue
        fields = [fields[0], *(field for field in fields[1:] if field not in {"+", "-"})]
        if len(fields) < len(header):
            continue
        values: dict[str, object] = {"tree": int(fields[0])}
        for key, raw in zip(header[1:], fields[1:], strict=False):
            try:
                values[key] = float(raw)
            except ValueError:
                values[key] = raw
        rows.append(values)
    if not rows:
        raise ValueError(f"No USER TREES topology-test table found in {report}")
    return rows


def test_topologies(
    alignment_fasta: str | Path,
    candidate_trees: str | Path,
    *,
    model: str = "MFP",
    replicates: int = 10000,
    seed: int = 42,
    threads: int = 1,
) -> OrganelleResult:
    """Run IQ-TREE AU/SH/KH topology tests on a set of candidate trees.

    Candidate trees must be Newick records separated by semicolons. IQ-TREE
    estimates model parameters from its initial parsimony tree (``-n 0``),
    computes candidate-tree likelihoods, then runs RELL tests. AU, SH, and KH
    columns are p-values; BP-RELL and c-ELW columns are weights, as reported
    by IQ-TREE. ``replicates`` defaults to 10000, the count IQ-TREE 3
    recommends for reliable AU p-values.
    """
    alignment = Path(alignment_fasta)
    trees = Path(candidate_trees)
    params = {
        "alignment_fasta": str(alignment),
        "candidate_trees": str(trees),
        "model": model,
        "replicates": replicates,
        "seed": seed,
        "threads": threads,
    }
    if not alignment.is_file():
        return failed_result(
            "test_topologies",
            summary_text=f"Alignment FASTA does not exist: {alignment}",
            code="phylogeny.test_topologies.alignment_missing",
            anomalies=["missing_alignment"],
            result_provenance=provenance("test_topologies", method="iqtree2", parameters=params),
        )
    if not trees.is_file():
        return failed_result(
            "test_topologies",
            summary_text=f"Candidate tree file does not exist: {trees}",
            code="phylogeny.test_topologies.trees_missing",
            anomalies=["missing_candidate_trees"],
            result_provenance=provenance("test_topologies", method="iqtree2", parameters=params),
        )
    if replicates < 1000 or threads < 1 or seed < 0:
        return failed_result(
            "test_topologies",
            summary_text="replicates must be >=1000, threads >=1, and seed >=0.",
            code="phylogeny.test_topologies.invalid_parameters",
            anomalies=["invalid_parameters"],
            result_provenance=provenance("test_topologies", method="iqtree2", parameters=params),
        )
    try:
        tree_count = sum(1 for _ in Phylo.parse(str(trees), "newick"))
    except Exception as exc:
        return failed_result(
            "test_topologies",
            summary_text=f"Candidate tree file is not valid Newick: {exc}",
            code="phylogeny.test_topologies.invalid_tree_file",
            anomalies=["invalid_candidate_trees"],
            result_provenance=provenance("test_topologies", method="iqtree2", parameters=params),
        )
    if tree_count < 2:
        return failed_result(
            "test_topologies",
            summary_text="At least two semicolon-terminated Newick trees are required.",
            code="phylogeny.test_topologies.too_few_trees",
            anomalies=["fewer_than_two_candidate_trees"],
            details={"candidate_tree_count": tree_count},
            result_provenance=provenance("test_topologies", method="iqtree2", parameters=params),
        )

    binary_v3 = shutil.which("iqtree3")
    binary = binary_v3 or shutil.which("iqtree2") or shutil.which("iqtree")
    if binary is None:
        return failed_result(
            "test_topologies",
            summary_text="IQ-TREE 2 executable not found (expected iqtree2 or iqtree).",
            code="phylogeny.test_topologies.backend_missing",
            anomalies=["iqtree_missing"],
            result_provenance=provenance("test_topologies", method="iqtree2", parameters=params),
        )

    from ..runtime import managed_run_path

    # Destination-free (output-boundary contract): runs live in managed storage.
    out = managed_run_path("phylogeny.test_topologies", uuid4().hex)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / "topology_test"
    if binary_v3:
        argv = [
            binary,
            "-s",
            str(alignment),
            "-n",
            "0",
            "-m",
            model,
            "--trees",
            str(trees),
            "--test",
            str(replicates),
            "--test-au",
            "--seed",
            str(seed),
            "-T",
            str(threads),
            "--prefix",
            str(prefix),
        ]
    else:
        argv = [
            binary,
            "-s",
            str(alignment),
            "-n",
            "0",
            "-m",
            model,
            "-z",
            str(trees),
            "-zb",
            str(replicates),
            "-au",
            "-seed",
            str(seed),
            "-nt",
            str(threads),
            "-pre",
            str(prefix),
        ]
    try:
        run_external(argv, tool="IQ-TREE", code="phylogeny.test_topologies.external_failed")
        report = Path(f"{prefix}.iqtree")
        results = _parse_user_tree_table(report)
    except Exception as exc:
        details = getattr(exc, "details", None)
        details = dict(details) if isinstance(details, Mapping) else {}
        return failed_result(
            "test_topologies",
            summary_text=f"IQ-TREE topology test failed: {exc}",
            code=str(getattr(exc, "code", "phylogeny.test_topologies.report_invalid")),
            anomalies=["iqtree_topology_test_failed"],
            details=details,
            result_provenance=provenance(
                "test_topologies", method="iqtree2", argv=argv, parameters=params, random_seed=seed
            ),
        )

    artifact = artifact_for(report, kind="topology_test_report", format="text")
    return ok_result(
        "test_topologies",
        summary_text=f"IQ-TREE compared {len(results)} candidate topologies with AU/SH/KH tests.",
        metrics={
            "candidate_tree_count": tree_count,
            "tested_tree_count": len(results),
            "replicates": replicates,
            "seed": seed,
            "threads": threads,
            "results": results,
            "report_path": str(report),
            "argv": argv,
        },
        flags=("topology_tests_complete",),
        artifacts=(artifact,) if artifact is not None else (),
        result_provenance=provenance(
            "test_topologies", method="iqtree2", argv=argv, parameters=params, random_seed=seed
        ),
    )
