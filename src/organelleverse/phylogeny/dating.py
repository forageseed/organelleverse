"""Calibrated LSD2 or MCMCTree dating of an extant-taxon tree.

Ages are Ma before present; LSD2 dates have the opposite sign. LSD2 calibration
intervals are hard constraints, not Bayesian priors. IQ-TREE estimates branch
lengths on the supplied topology, then LSD2 fits dates on the specified root
split. CI resampling is conditional on that topology and the calibrations.
"""

from __future__ import annotations

import io
import math
import re
from pathlib import Path

from Bio import Phylo
from Bio.Phylo.BaseTree import Tree
from pydantic import Field, ValidationError

from .._bio import read_fasta
from ..core.errors import OrganelleDependencyError, OrganelleExecutionError, OrganelleInputError
from ..core.external import run_external
from ..core.result import OrganelleResult
from ..operations.parameters import OperationParameterModel
from ._results import artifact_for, ok_result, provenance
from .partition import resolve_iqtree


class Calibration(OperationParameterModel):
    """One MRCA or root constraint, in Ma (at least one bound required).

    Use exactly one of ``mrca=[leaf_a, leaf_b]`` and ``root=True``. Equal
    bounds fix the LSD2 age (unsupported for MCMCTree). ``distribution=None``
    or ``"bounds"`` selects age bounds: hard for LSD2, soft B/L/U for MCMCTree
    with default 2.5% tails. Explicit probability densities are unsupported.
    ``source`` records the fossil/reference or secondary-calibration origin.
    """

    mrca: list[str] | None = None
    root: bool = False
    min_age_ma: float | None = None
    max_age_ma: float | None = None
    distribution: str | None = None
    source: str | None = None


class MCMCTreeOptions(OperationParameterModel):
    """Priors use 100 Ma time units; Gamma parameters are shape and rate.

    Multiple partitions use PAML's gamma-Dirichlet prior with concentration=1
    for both locus rates and rate variances (prior type=0).

    Each of two chains discards burnin iterations then saves nsample samples
    every sampfreq iterations. Convergence thresholds are fixed: ESS >= 200
    per chain/parameter and split R-hat <= 1.01. No automatic chain extension.
    """

    clock: int = Field(default=2, ge=2, le=3)
    burnin: int = Field(default=50000, ge=0)
    sampfreq: int = Field(default=50, ge=1)
    nsample: int = Field(default=20000, ge=4)
    rgene_gamma: list[float] = Field(default=[2.0, 20.0])
    sigma2_gamma: list[float] = Field(default=[2.0, 10.0])


def _invalid(message: str) -> OrganelleInputError:
    return OrganelleInputError(code="phylogeny.dating.invalid_input", message=message)


def _output_error(message: str) -> OrganelleExecutionError:
    return OrganelleExecutionError(code="phylogeny.dating.invalid_output", message=message)


def _read_input(
    tree_newick: str | Path, alignment_fasta: str | Path, outgroup: list[str] | None
) -> tuple[Tree, list[str], int]:
    try:
        tree = Phylo.read(tree_newick, "newick")
        sequences = list(read_fasta(alignment_fasta))
    except (OSError, ValueError) as error:
        raise _invalid(f"Cannot read tree/alignment: {error}") from error
    taxa = [n.name for n in tree.get_terminals()]
    if len(taxa) < 4 or len(set(taxa)) != len(taxa):
        raise _invalid("Tree must contain at least four uniquely named tips.")
    if any(not t or re.search(r"[\s,():;\[\]'\"]", t) for t in taxa):
        raise _invalid("Tip names must not contain whitespace or Newick/date-file delimiters.")
    names = [name for name, _ in sequences]
    lengths = {len(seq) for _, seq in sequences}
    if len(names) != len(set(names)) or set(names) != set(taxa):
        raise _invalid("Tree and alignment must have the same unique tip names.")
    if len(lengths) != 1 or not next(iter(lengths), 0):
        raise _invalid("FASTA must be a nonempty equal-length alignment.")
    if outgroup is not None:
        if not outgroup or len(set(outgroup)) != len(outgroup) or not set(outgroup) < set(taxa):
            raise _invalid("Outgroup must be a nonempty proper subset of unique tree tips.")
        target = set(outgroup)
        edge = next(
            (
                node
                for node in tree.find_clades()
                if {t.name for t in node.get_terminals()} in (target, set(taxa) - target)
            ),
            None,
        )
        if edge is None:
            raise _invalid("Outgroup must define a monophyletic side of a tree split.")
        tree.root_with_outgroup(edge, outgroup_branch_length=0.0)
    elif len(tree.root.clades) != 2:
        raise _invalid("Unrooted tree requires an explicit outgroup.")
    # IQ-TREE unroots reversible-model trees; -o retains the intended root edge.
    rooting = [t.name for t in tree.root.clades[0].get_terminals()]
    return tree, rooting, next(iter(lengths))


def _calibrations(tree: Tree, calibrations: list[Calibration]) -> list[Calibration]:
    if not calibrations:
        raise _invalid("At least one ancestral age calibration is required.")
    taxa = {t.name for t in tree.get_terminals()}
    resolved = []
    rows = []
    for value in calibrations:
        try:
            c = Calibration.model_validate(value)
        except ValidationError as error:
            raise _invalid(f"Invalid calibration: {error}") from error
        if c.root == (c.mrca is not None):
            raise _invalid("Specify exactly one of root=True and mrca=[leaf_a, leaf_b].")
        if c.mrca is not None and (
            len(c.mrca) != 2 or len(set(c.mrca)) != 2 or not set(c.mrca) <= taxa
        ):
            raise _invalid(f"MRCA must name two different existing leaves: {c.mrca}")
        if c.distribution not in (None, "bounds"):
            raise _invalid(
                "Calibration supports bounds only; explicit normal/lognormal/uniform priors are not supported."
            )
        low, high = c.min_age_ma, c.max_age_ma
        if low is None and high is None:
            raise _invalid("Calibration requires min_age_ma or max_age_ma.")
        if any(x is not None and (not math.isfinite(x) or x < 0) for x in (low, high)):
            raise _invalid("Calibration ages must be finite and nonnegative Ma.")
        if low is not None and high is not None and low > high:
            raise _invalid("min_age_ma must not exceed max_age_ma.")
        node = tree.root if c.root else tree.common_ancestor(c.mrca)
        descendants = {n.name for n in node.get_terminals()}
        if any(descendants == old for old, _ in resolved):
            raise _invalid(
                "Multiple calibrations resolve to the same node; provide their intersection once."
            )
        resolved.append((descendants, c))
        rows.append(c)
    for ancestor, a in resolved:
        for descendant, d in resolved:
            if (
                descendant < ancestor
                and a.max_age_ma is not None
                and d.min_age_ma is not None
                and a.max_age_ma < d.min_age_ma
            ):
                raise _invalid(
                    "An ancestor's maximum age is younger than a descendant's minimum age."
                )
    if not any((c.min_age_ma or 0) > 0 for c in rows):
        raise _invalid("An absolute time scale requires at least one positive minimum/fixed age.")
    return rows


def _date_bound(c: Calibration, *, date_file: bool = False) -> str:
    """Convert age bounds to increasing-time LSD2 syntax, reversing endpoints."""
    low, high = c.min_age_ma, c.max_age_ma
    if low == high:
        return f"{-low:.17g}"
    if date_file:
        earliest = f"{-high:.17g}" if high is not None else "NA"
        latest = f"{-low:.17g}" if low is not None else "NA"
        return f"{earliest}:{latest}"
    if high is None:
        return f"u({-low:.17g})"
    if low is None:
        return f"l({-high:.17g})"
    return f"b({-high:.17g},{-low:.17g})"


def date_tree(
    tree_newick: str | Path,
    alignment_fasta: str | Path,
    calibrations: list[Calibration],
    *,
    output_dir: str | Path,
    backend: str = "iqtree_lsd2",
    mcmctree_options: MCMCTreeOptions | None = None,
    outgroup: list[str] | None = None,
    model: str = "GTR+G4",
    partition_nexus: str | Path | None = None,
    ci_replicates: int = 100,
    clock_sd: float = 0.2,
    seed: int = 42,
    threads: int = 1,
    iqtree_bin: str | None = None,
    mcmctree_bin: str | None = None,
    dry_run: bool = False,
) -> OrganelleResult:
    """Date an extant-taxon topology with LSD2 or two-chain MCMCTree.

    ``backend="mcmctree"`` uses soft bounds, GTR+G4 and 100 Ma time units;
    configure chains/priors with ``mcmctree_options``. See MCMCTreeOptions.
    NEXUS charsets define its DNA partitions (codon positions or genes), with
    GTR+G4 fitted independently in each; they must cover every site once.

    An existing IQ-TREE ``.treefile`` is accepted with its original alignment;
    ``partition_nexus`` can reuse its ``.best_scheme.nex``. Branch lengths are
    re-estimated, topology is fixed. A bifurcating Newick root is trusted;
    otherwise an explicit monophyletic outgroup is required. All tips are
    extant (age zero). LSD2 may optimize root position along the chosen edge.
    ``ci_replicates=0`` disables intervals (reported as null). No tool fallback.
    """
    tree, rooting, n_sites = _read_input(tree_newick, alignment_fasta, outgroup)
    rows = _calibrations(tree, calibrations)
    if backend == "mcmctree":
        from .mcmctree import run_mcmctree

        return run_mcmctree(
            tree,
            alignment_fasta,
            rows,
            output_dir=output_dir,
            options=mcmctree_options,
            model=model,
            partition_nexus=partition_nexus,
            seed=seed,
            binary=mcmctree_bin,
            dry_run=dry_run,
        )
    if backend != "iqtree_lsd2":
        raise _invalid("backend must be 'iqtree_lsd2' or 'mcmctree'.")
    if mcmctree_options is not None:
        raise _invalid("mcmctree_options requires backend='mcmctree'.")
    if isinstance(ci_replicates, bool) or not isinstance(ci_replicates, int) or ci_replicates < 0:
        raise _invalid("ci_replicates must be a nonnegative integer.")
    if not math.isfinite(clock_sd) or clock_sd < 0 or threads < 1 or seed < 1:
        raise _invalid(
            "clock_sd must be finite and nonnegative; threads and seed must be positive."
        )
    if partition_nexus is not None and not Path(partition_nexus).is_file():
        raise _invalid(f"Partition file not found: {partition_nexus}")
    binary = resolve_iqtree(iqtree_bin)
    if binary is None and not dry_run:
        raise OrganelleDependencyError(
            code="phylogeny.dating.missing_iqtree",
            message="IQ-TREE with LSD2 is required. Install: micromamba create -n ov-dating -c conda-forge -c bioconda iqtree=2.4.0; set ORGANELLEVERSE_IQTREE_BIN.",
        )
    output = Path(output_dir).resolve()
    prefix = output / "dating"
    rooted = output / "rooted.nwk"
    date_file = output / "calibrations.txt"
    argv = [
        binary or "iqtree",
        "-s",
        str(Path(alignment_fasta).resolve()),
        "-te",
        str(rooted),
        "-m",
        model,
        "-T",
        str(threads),
        "--prefix",
        str(prefix),
        "-seed",
        str(seed),
        "-o",
        ",".join(rooting),
        "--dating",
        "LSD",
        "--date-tip",
        "0",
        "--clock-sd",
        str(clock_sd),
        "--date-options",
        "-r k",
    ]
    if ci_replicates:
        argv += ["--date-ci", str(ci_replicates)]
    if partition_nexus is not None:
        argv += ["-p", str(Path(partition_nexus).resolve())]
    date_lines = []
    for c in rows:
        if c.root:
            argv += ["--date-root", _date_bound(c)]
        else:
            date_lines.append(f"{','.join(c.mrca)} {_date_bound(c, date_file=True)}")
    if date_lines:
        argv += ["--date", str(date_file)]
    metrics = {
        "backend": "iqtree_lsd2",
        "age_unit": "Ma",
        "tip_age_ma": 0,
        "calibrations": [c.model_dump(mode="json") for c in rows],
        "outgroup": rooting,
        "n_taxa": len(tree.get_terminals()),
        "alignment_length": n_sites,
        "ci_replicates": ci_replicates,
        "clock_sd": clock_sd,
        "argv": argv,
        "model": model,
        "partition_nexus": str(partition_nexus) if partition_nexus else None,
        "planned": dry_run,
    }
    prov = provenance(
        "date_tree", method="iqtree_lsd2", argv=argv, parameters=metrics, random_seed=seed
    )
    if dry_run:
        return ok_result(
            "date_tree",
            status="warning",
            summary_text="IQ-TREE/LSD2 dating plan; no ages estimated.",
            metrics=metrics,
            flags=("planned_only",),
            result_provenance=prov,
        )
    # Exclusive run directory prevents stale successful output from masking a failed run.
    output.mkdir(parents=True, exist_ok=False)
    Phylo.write(tree, rooted, "newick", format_branch_length="%1.12g")
    if date_lines:
        date_file.write_text("\n".join(date_lines) + "\n")
    completed = run_external(
        argv, cwd=output, tool="IQ-TREE/LSD2", code="phylogeny.dating.execution_failed"
    )
    (output / "stdout.txt").write_text(completed.stdout)
    (output / "stderr.txt").write_text(completed.stderr)
    nex = Path(str(prefix) + ".timetree.nex")
    nwk = Path(str(prefix) + ".timetree.nwk")
    if not nex.is_file() and not nwk.is_file():
        raise _output_error(
            "IQ-TREE produced no time tree (reproduced with 3.0.1). Use an LSD2-enabled build, e.g. IQ-TREE 2.4.0, via ORGANELLEVERSE_IQTREE_BIN."
        )
    report = Path(str(prefix) + ".timetree.lsd")
    report_text = report.read_text() if report.is_file() else ""
    rate_match = re.search(rf"\brate\s+({_NUMBER})", report_text)
    rate = float(rate_match.group(1)) if rate_match else None
    parsed = parse_timetree(
        nex if nex.is_file() else nwk,
        require_ci=ci_replicates > 0,
        substitution_rate=rate,
        calibrations=rows,
    )
    time_newick = output / "dated_tree.nwk"
    time_newick.write_text(parsed["timetree_newick"] + "\n")
    metrics["substitution_rate_per_site_per_ma"] = rate
    metrics["time_newick_path"] = str(time_newick)
    if set(parsed["taxa"]) != {n.name for n in tree.get_terminals()}:
        raise _output_error("Time tree taxa differ from input taxa.")
    metrics.update(parsed)
    metrics["timetree_path"] = str(nex if nex.is_file() else nwk)
    version = re.search(r"IQ-TREE[^\n]*version[^\n]*", completed.stdout)
    if version is None:
        raise _output_error("IQ-TREE version missing from output.")
    metrics["software_version"] = version.group(0)
    metrics["lsd_report"] = report_text
    metrics["lsd2_version"] = report_text.strip().splitlines()[0] if report_text.strip() else None
    artifacts = tuple(
        a
        for p, fmt in [(nex, "nexus"), (time_newick, "newick")]
        if (a := artifact_for(p, kind="tree", format=fmt)) is not None
    )
    return ok_result(
        "date_tree",
        status="warning" if parsed["dating_warnings"] else "ok",
        flags=tuple(parsed["dating_warnings"]),
        summary_text=f"LSD2 dated {len(parsed['node_ages'])} internal nodes (Ma)."
        + (
            " WARNING: zero/near-zero time branches or calibration boundary ages; inspect "
            "degeneracy affected_nodes. Uncalibrated nodes collapsed onto calibrated nodes "
            "do not provide separately resolved age estimates."
            if parsed["dating_warnings"]
            else ""
        ),
        metrics=metrics,
        artifacts=artifacts,
        result_provenance=prov,
    )


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_DATE = re.compile(rf'\bdate\s*=\s*"?({_NUMBER})(?=[",\]\s]|$)')
_CI_DATE = re.compile(rf'\bCI_date\s*=\s*"?\{{\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\}}')


def _degeneracy(tree, rows, calibrations):
    """Diagnose the fitted tree without changing any dates or branches.

    LSD2 prints six significant digits. Half a last-place unit at the oldest
    reported age is the absolute age resolution used for all comparisons.
    """
    oldest = max(r["age_ma"] for r in rows)
    tolerance = 0.5 * 10 ** (math.floor(math.log10(oldest)) - 5) if oldest > 0 else 0.0
    nodes = tree.get_nonterminals(order="preorder")
    records = dict(zip(nodes, rows, strict=True))
    ids = {node: row["node_id"] for node, row in records.items()}
    ids.update({tip: f"tip:{tip.name}" for tip in tree.get_terminals()})
    try:
        calibrated = {
            tree.root if c.root else tree.common_ancestor(c.mrca): c for c in calibrations
        }
    except ValueError as error:
        raise _output_error("LSD2 time tree is missing calibration MRCA taxa.") from error
    boundaries = []
    for node, c in calibrated.items():
        for side, bound in (("min", c.min_age_ma), ("max", c.max_age_ma)):
            if bound is not None and abs(records[node]["age_ma"] - bound) <= tolerance:
                boundaries.append(
                    {
                        "node_id": ids[node],
                        "boundary": side,
                        "bound_ma": bound,
                        "age_ma": records[node]["age_ma"],
                    }
                )
    edges, neighbors = [], {node: [] for node in tree.find_clades()}
    for parent in nodes:
        for child in parent.clades:
            if child.branch_length <= tolerance:
                edges.append(
                    {
                        "parent_id": ids[parent],
                        "child_id": ids[child],
                        "child_is_tip": child.is_terminal(),
                        "length_ma": child.branch_length,
                    }
                )
                neighbors[parent].append(child)
                neighbors[child].append(parent)
    collapsed = {}
    visited = set()
    for node in neighbors:
        if node in visited or not neighbors[node]:
            continue
        component, pending = [], [node]
        visited.add(node)
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        for current in component:
            if current.is_terminal() or current in calibrated:
                continue
            peers = [
                ids[c]
                for c in component
                if c in calibrated
                and abs(records[c]["age_ma"] - records[current]["age_ma"]) <= tolerance
            ]
            if peers:
                collapsed[ids[current]] = sorted(peers)
    affected = {e[k] for e in edges for k in ("parent_id", "child_id")}
    affected.update(b["node_id"] for b in boundaries)
    warnings = []
    if edges:
        warnings.append("lsd2_zero_time_branches")
    if boundaries:
        warnings.append("lsd2_calibration_boundary")
    if collapsed:
        warnings.append("lsd2_uncalibrated_node_collapsed")
    return {
        "dating_warnings": warnings,
        "degeneracy": {
            "tolerance_ma": tolerance,
            "tolerance_basis": "half a last-place unit at oldest age, LSD2 six significant digits",
            "zero_time_branches": edges,
            "calibration_boundary_hits": boundaries,
            "affected_nodes": [
                dict(
                    row,
                    calibrated=node in calibrated,
                    collapsed_with_calibrated_node_ids=collapsed.get(ids[node], []),
                )
                for node, row in records.items()
                if ids[node] in affected
            ],
            "collapsed_uncalibrated_node_ids": sorted(collapsed),
        },
    }


def parse_timetree(
    path: str | Path,
    *,
    require_ci: bool = False,
    substitution_rate: float | None = None,
    calibrations: list[Calibration] | None = None,
) -> dict[str, object]:
    """Read LSD2 output with explicit units; never treat substitutions as Ma.

    IQ-TREE 2.4's ``.timetree.nex`` has time branches and date/CI annotations.
    Its ``.timetree.nwk`` instead has fitted substitution lengths (!), so the
    latter requires the LSD report's rate (substitutions/site/Ma), and cannot
    supply CIs. This is an output-format distinction, not a backend fallback.
    ``calibrations`` enables boundary and calibrated-node collapse diagnostics;
    zero/near-zero branches are reported even without calibrations.
    """
    try:
        text = Path(path).read_text()
        nexus = text.lstrip().upper().startswith("#NEXUS")
        tree = Phylo.read(io.StringIO(text), "nexus" if nexus else "newick")
    except (OSError, ValueError) as error:
        raise _output_error(f"Cannot parse LSD2 tree: {error}") from error
    if not nexus:
        if (
            substitution_rate is None
            or not math.isfinite(substitution_rate)
            or substitution_rate <= 0
        ):
            raise _output_error(
                "LSD2 Newick substitution lengths require the positive rate from its report."
            )
        for node in tree.find_clades():
            if node.branch_length is not None:
                node.branch_length /= substitution_rate
    rows = []
    for node in tree.find_clades(order="preorder"):
        if node is not tree.root and (
            node.branch_length is None
            or not math.isfinite(node.branch_length)
            or node.branch_length < 0
        ):
            raise _output_error("LSD2 tree has missing, negative or nonfinite branch lengths.")
        comment = node.comment or ""
        match = _DATE.search(comment)
        if nexus:
            if match is None:
                raise _output_error("LSD2 NEXUS node is missing its numeric date annotation.")
            age = -float(match.group(1))
        else:
            age = tree.distance(node, node.get_terminals()[0])
        if not math.isfinite(age) or age < 0:
            raise _output_error("LSD2 returned a negative or nonfinite age.")
        if node.is_terminal():
            if age != 0:
                raise _output_error("LSD2 tip date is not zero; this API requires extant taxa.")
            continue
        ci = _CI_DATE.search(comment)
        if require_ci and ci is None:
            raise _output_error(
                "Requested LSD2 confidence intervals are missing from the time tree."
            )
        lower, upper = (-float(ci.group(2)), -float(ci.group(1))) if ci else (None, None)
        if ci and (not all(math.isfinite(v) for v in (lower, upper)) or not 0 <= lower <= upper):
            raise _output_error("Invalid LSD2 age confidence interval.")
        rows.append(
            {
                "node_id": f"node_{len(rows)}",
                "is_root": node is tree.root,
                "descendant_taxa": sorted(t.name for t in node.get_terminals()),
                "age_ma": age,
                "ci_lower_ma": lower,
                "ci_upper_ma": upper,
            }
        )
    if not rows:
        raise _output_error("LSD2 output has no internal nodes.")
    diagnostics = _degeneracy(tree, rows, calibrations or [])
    taxa = sorted(t.name for t in tree.get_terminals())
    for node in tree.find_clades():
        node.comment = None
    stream = io.StringIO()
    Phylo.write(tree, stream, "newick", format_branch_length="%1.12g")
    return {
        **diagnostics,
        "timetree_newick": stream.getvalue().strip(),
        "node_ages": rows,
        "taxa": taxa,
        "age_source": "lsd2_date_annotation" if nexus else "substitution_length_divided_by_rate",
        "ci_method": "LSD2 Poisson/lognormal branch-length resampling"
        if any(r["ci_lower_ma"] is not None for r in rows)
        else None,
        "ci_level": 0.95 if any(r["ci_lower_ma"] is not None for r in rows) else None,
    }
