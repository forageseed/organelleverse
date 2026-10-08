from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.errors import OrganelleExecutionError
from ..core.external import run_external
from ._results import artifact_for, failed_result, findings, ok_result, provenance


def run_astral(
    gene_trees: str | Path,
    *,
    astral_jar: str | Path | None = None,
    java: str = "java",
) -> Any:
    """Infer an ASTRAL species tree and local quartet support from gene trees.

    ASTRAL-III is invoked as ``java -jar astral.jar -i trees -o tree -t 2``.
    The JAR can be passed explicitly or located via ``ASTRAL_JAR``. This
    operation does not install Java or download ASTRAL.
    """
    trees_path = Path(gene_trees)
    jar_path = Path(astral_jar or os.environ.get("ASTRAL_JAR", ""))
    if not trees_path.is_file():
        return failed_result(
            "run_astral",
            summary_text=f"Gene-tree file does not exist: {trees_path}",
            code="phylogeny.astral.input_missing",
            anomalies=["input_missing"],
            result_provenance=provenance(
                "run_astral", parameters={"gene_trees": str(trees_path)}
            ),
        )
    if not trees_path.read_text(encoding="utf-8").strip():
        return failed_result(
            "run_astral",
            summary_text="Gene-tree file is empty.",
            code="phylogeny.astral.input_empty",
            anomalies=["input_empty"],
            result_provenance=provenance(
                "run_astral", parameters={"gene_trees": str(trees_path)}
            ),
        )
    if not jar_path.is_file():
        return failed_result(
            "run_astral",
            summary_text=(
                "ASTRAL JAR not found. Pass astral_jar or set ASTRAL_JAR; "
                "Java and ASTRAL must be installed separately."
            ),
            code="phylogeny.astral.jar_missing",
            anomalies=["dependency_missing"],
            details={"astral_jar": str(jar_path)},
            result_provenance=provenance(
                "run_astral",
                method="ASTRAL-III",
                parameters={"gene_trees": str(trees_path), "astral_jar": str(jar_path)},
            ),
        )

    from ..runtime import managed_run_path

    # Destination-free (output-boundary contract): runs live in managed storage.
    out_dir = managed_run_path("phylogeny.run_astral", uuid4().hex)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_path = out_dir / "astral_species_tree.tre"
    argv = [java, "-jar", str(jar_path), "-i", str(trees_path), "-o", str(tree_path), "-t", "2"]
    try:
        proc = run_external(argv, code="phylogeny.astral.execution_failed", tool="ASTRAL")
    except OrganelleExecutionError as exc:
        tail = str((exc.details or {}).get("stderr_tail", ""))
        message = f"ASTRAL failed: {exc}"
        if tail:
            message += f"\n{tail}"
        return failed_result(
            "run_astral",
            summary_text=message,
            code="phylogeny.astral.execution_failed",
            anomalies=["astral_failed"],
            details={"stderr_tail": tail, "argv": argv},
            result_provenance=provenance(
                "run_astral",
                method="ASTRAL-III",
                argv=argv,
                parameters={"gene_trees": str(trees_path), "astral_jar": str(jar_path)},
            ),
        )

    if not tree_path.is_file():
        return failed_result(
            "run_astral",
            summary_text="ASTRAL completed without writing its species-tree output.",
            code="phylogeny.astral.output_missing",
            anomalies=["output_missing"],
            details={"stdout_tail": "\n".join(proc.stdout.splitlines()[-40:])},
            result_provenance=provenance("run_astral", method="ASTRAL-III", argv=argv),
        )

    return ok_result(
        "run_astral",
        summary_text=f"ASTRAL inferred a species tree from gene trees in {trees_path}.",
        metrics={
            "gene_trees": str(trees_path),
            "n_gene_trees": sum(1 for line in trees_path.read_text().splitlines() if line.strip()),
            "species_tree_path": str(tree_path),
            "astral_stdout": proc.stdout,
        },
        result_findings=findings(("method", "ASTRAL-III")),
        flags=("gene_tree_discordance_estimated",),
        artifacts=(artifact_for(tree_path, kind="species_tree", format="newick"),),
        result_provenance=provenance(
            "run_astral",
            method="ASTRAL-III",
            argv=argv,
            parameters={"gene_trees": str(trees_path), "astral_jar": str(jar_path)},
        ),
    )
