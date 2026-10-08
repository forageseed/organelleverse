"""Independent, auditable evidence axes for CMS candidate proteins."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from ..._bio import read_fasta, write_fasta
from ...core.artifacts import ArtifactRef
from ...core.errors import OrganelleDependencyError, OrganelleInputError, OrganelleParameterError
from ...core.external import run_external
from ...core.frozen import thaw_json
from ...core.result import OrganelleResult
from .pipeline import _has_tm
from .resources import cms_protein_fasta, cms_reference_json


def _invalid(message: str) -> OrganelleInputError:
    return OrganelleInputError(code="cms.invalid_evidence_input", message=message)


def _proteins(path: Path) -> dict[str, str]:
    try:
        rows = read_fasta(path)
    except (OSError, ValueError) as error:
        raise _invalid(str(error)) from error
    proteins = {name: sequence.upper().removesuffix("*") for name, sequence in rows}
    for name, sequence in proteins.items():
        if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWYXBZJUO*"):
            raise _invalid(f"{name}: expected a nonempty amino-acid sequence")
    return proteins


def assess_cms_candidates(
    protein_fasta: Path,
    *,
    conserved_proteins_fasta: Path,
    cms_proteins_fasta: Path | None = None,
    expression_tsv: Path | None = None,
    topology_predictions: Path | None = None,
    min_identity: float = 80.0,
    min_homolog_coverage: float = 0.8,
    min_alignment_aa: int = 20,
    evalue: float = 1e-5,
    tm_window: int = 21,
    tm_threshold: float = 0.5,
    threads: int = 1,
    timeout: int = 120,
    losat_path: str = "LOSAT",
) -> OrganelleResult:
    """Assess proteins using Rust LOSAT BLASTP and optional TPMs.

    Default CMS references are a small, accession-checked ORF79/ORF138 seed.
    Users must supply ordinary mitochondrial protein references separately.
    Identity/alignment thresholds select individual HSPs; a near-full homolog
    requires BOTH query and subject coverage >= min_homolog_coverage in one HSP.
    Partial conserved matches and unmatched query intervals are reported, not
    promoted to confirmed chimeras. Hydrophobicity is only a KD-window hint.
    TPM contrasts are descriptive means/ratios, not differential-expression tests.
    Homology never establishes fertility phenotype or CMS causality.
    """
    for name, value in {
        "min_alignment_aa": min_alignment_aa,
        "tm_window": tm_window,
        "threads": threads,
        "timeout": timeout,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise OrganelleParameterError(
                code="cms.invalid_threshold", message=f"{name} must be a positive integer"
            )
    if not (
        math.isfinite(min_identity)
        and 0 < min_identity <= 100
        and math.isfinite(min_homolog_coverage)
        and 0 < min_homolog_coverage <= 1
        and math.isfinite(evalue)
        and evalue > 0
        and math.isfinite(tm_threshold)
    ):
        raise OrganelleParameterError(
            code="cms.invalid_threshold",
            message="Invalid identity, coverage, E-value or hydrophobicity threshold",
        )
    executable = shutil.which(losat_path)
    if executable is None:
        raise OrganelleDependencyError(
            code="cms.losat_missing", message=f"Rust LOSAT not found: {losat_path}"
        )
    queries = _proteins(protein_fasta)
    core = _proteins(conserved_proteins_fasta)
    cms_path = cms_proteins_fasta if cms_proteins_fasta is not None else cms_protein_fasta()
    cms = _proteins(cms_path)
    expressions = _expression(expression_tsv, queries) if expression_tsv is not None else {}
    from .topology import load_tmbed_predictions

    topology = (
        load_tmbed_predictions(topology_predictions, queries)
        if topology_predictions is not None
        else {}
    )
    metadata = (
        {row["name"]: row for row in json.loads(cms_reference_json().read_text())}
        if cms_proteins_fasta is None
        else {}
    )
    version_output = run_external(
        [executable, "--version"], timeout=timeout, code="cms.losat_failed", tool="LOSAT"
    ).stdout.strip()
    if not version_output.lower().startswith("losat "):
        raise OrganelleDependencyError(
            code="cms.losat_wrong_executable",
            message=f"Expected Rust LOSAT, received version output: {version_output!r}",
        )
    tool_version = version_output.splitlines()[0]
    with TemporaryDirectory(prefix="organelleverse-cms-") as scratch:
        work = Path(scratch)
        query = write_fasta(work / "queries.fasta", list(queries.items()))
        hits = {}
        for label, subjects in (("cms", cms), ("conserved", core)):
            subject = write_fasta(work / f"{label}.fasta", list(subjects.items()))
            run = run_external(
                [
                    executable,
                    "blastp",
                    "--task",
                    "blastp",
                    "--query",
                    str(query),
                    "--subject",
                    str(subject),
                    "--outfmt",
                    "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen slen nident",
                    "--seg",
                    "no",
                    "--comp-based-stats",
                    "2",
                    "--evalue",
                    str(evalue),
                    "--num-threads",
                    str(threads),
                    "--max-target-seqs",
                    str(len(subjects)),
                ],
                timeout=timeout,
                code="cms.losat_failed",
                tool="LOSAT blastp",
            )
            hits[label] = _parse_hits(run.stdout, queries, subjects, min_identity, min_alignment_aa)
    rows = []
    for name, protein in queries.items():
        cms_hits = hits["cms"][name]
        core_hits = hits["conserved"][name]
        near_cms = [h for h in cms_hits if _near_full(h, min_homolog_coverage)]
        near_core = [h for h in core_hits if _near_full(h, min_homolog_coverage)]
        unmatched = _unmatched(
            len(protein), [(h["query_start"], h["query_end"]) for h in core_hits]
        )
        labels = []
        if near_cms:
            labels.append("known_cms_reference_homolog")
        if near_core:
            labels.append("conserved_gene_homolog")
        if core_hits and unmatched:
            labels.append("partial_conserved_homology_with_unmatched_sequence")
        if not labels:
            labels.append("uncharacterized_orf")
        rows.append(
            {
                "candidate_id": name,
                "length_aa": len(protein),
                "protein_sequence": protein,
                "evidence_labels": labels,
                "cms_reference_hits": cms_hits,
                "cms_reference_metadata": [
                    metadata[subject_id]
                    for subject_id in sorted({h["subject_id"] for h in near_cms})
                    if subject_id in metadata
                ],
                "conserved_gene_hits": core_hits,
                "unmatched_regions": unmatched,
                "hydrophobicity_hint": _has_tm(protein, tm_window, tm_threshold),
                "hydrophobicity_method": "kyte_doolittle_window_not_topology_prediction",
                "internal_stop_positions": [i + 1 for i, aa in enumerate(protein) if aa == "*"],
                "expression": expressions.get(name),
                "transmembrane_topology": topology.get(name),
                "evidence_availability": {
                    "sequence_homology": True,
                    "hydrophobicity_hint": True,
                    "topology": name in topology,
                    "condition_expression": name in expressions,
                    "genomic_context": False,
                    "rna_editing": False,
                },
                "interpretation": "Sequence/expression evidence only; CMS phenotype and causality untested",
            }
        )
    return OrganelleResult(
        operation_id="phenotype.assess_cms_candidates",
        operation_version="1.0",
        scope="mitochondrion",
        status="ok",
        summary_text=f"Assessed {len(rows)} proteins across separate CMS evidence axes.",
        metrics={
            "candidate_count": len(rows),
            "candidate_table": rows,
            "cms_reference_count": len(cms),
            "conserved_reference_count": len(core),
            "cms_reference_kind": "accession_checked_seed"
            if cms_proteins_fasta is None
            else "caller_supplied",
            "alignment_backend": "losat_blastp",
            "losat_version": tool_version,
            "parameters": {
                "min_identity": min_identity,
                "min_homolog_coverage": min_homolog_coverage,
                "min_alignment_aa": min_alignment_aa,
                "evalue": evalue,
                "tm_window": tm_window,
                "tm_threshold": tm_threshold,
                "threads": threads,
                "timeout": timeout,
                "seg": "no",
                "comp_based_stats": 2,
            },
            "coordinate_system": "protein/query/subject intervals: 1-based inclusive",
        },
        flags=(
            "cms_causality_not_established",
            "fragment_homology_not_confirmed_chimerism",
            "hydrophobicity_hint_not_tm_topology",
            "limited_cms_reference_coverage",
            "topology_is_model_prediction"
            if topology_predictions is not None
            else "topology_not_measured",
            "expression_descriptive_only"
            if expression_tsv is not None
            else "expression_not_measured",
        ),
    )


def _near_full(hit, coverage):
    return hit["query_coverage"] >= coverage and hit["subject_coverage"] >= coverage


def _parse_hits(text, queries, subjects, identity, min_aa):
    rows = {name: [] for name in queries}
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 13:
            raise _invalid("Unexpected BLASTP tabular output: expected 13 columns")
        query, subject = fields[:2]
        if query not in queries or subject not in subjects:
            raise _invalid("BLASTP output identifiers do not match input proteins")
        try:
            pident, length = float(fields[2]), int(fields[3])
            qstart, qend, sstart, send = map(int, fields[4:8])
            evalue, bitscore = map(float, fields[8:10])
            qlen, slen, nident = map(int, fields[10:])
        except ValueError as error:
            raise _invalid("Invalid numeric BLASTP output") from error
        if (
            qlen != len(queries[query])
            or slen != len(subjects[subject])
            or not (1 <= qstart <= qend <= qlen and 1 <= sstart <= send <= slen)
        ):
            raise _invalid("BLASTP output length or interval does not match input")
        if pident < identity or length < min_aa:
            continue
        rows[query].append(
            {
                "subject_id": subject,
                "identity_percent": pident,
                "alignment_length": length,
                "identical_residues": nident,
                "query_start": qstart,
                "query_end": qend,
                "subject_start": sstart,
                "subject_end": send,
                "query_coverage": (qend - qstart + 1) / qlen,
                "subject_coverage": (send - sstart + 1) / slen,
                "evalue": evalue,
                "bitscore": bitscore,
            }
        )
    for hits in rows.values():
        hits.sort(
            key=lambda h: (-h["bitscore"], h["subject_id"], h["query_start"], h["subject_start"])
        )
    return rows


def _unmatched(length, intervals):
    """Complement the union of passing query HSPs; overlapping hits count once."""
    gaps = []
    cursor = 1
    for start, end in sorted(intervals):
        if start > cursor:
            gaps.append({"start": cursor, "end": start - 1})
        cursor = max(cursor, end + 1)
    if cursor <= length:
        gaps.append({"start": cursor, "end": length})
    return gaps


def _expression(path, queries):
    groups = {}
    seen = set()
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"candidate_id", "condition", "replicate", "tpm"}
            if not required <= set(reader.fieldnames or ()):
                raise ValueError("Expression TSV requires candidate_id, condition, replicate, tpm")
            for row in reader:
                name, condition, replicate = (
                    row[key].strip() for key in ("candidate_id", "condition", "replicate")
                )
                value = float(row["tpm"])
                if (
                    name not in queries
                    or condition not in {"sterile", "maintainer", "restored"}
                    or not replicate
                ):
                    raise ValueError(
                        "Expression IDs/conditions/replicates do not match the contract"
                    )
                if not math.isfinite(value) or value < 0:
                    raise ValueError("TPM must be finite and nonnegative")
                key = (name, condition, replicate)
                if key in seen:
                    raise ValueError("Duplicate candidate/condition/replicate expression row")
                seen.add(key)
                groups.setdefault(name, {}).setdefault(condition, []).append(
                    {"replicate": replicate, "tpm": value}
                )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise _invalid(str(error)) from error
    result = {}
    for name, conditions in groups.items():
        means = {label: mean(row["tpm"] for row in values) for label, values in conditions.items()}
        ratios = {}
        for control in ("maintainer", "restored"):
            if "sterile" in means and control in means:
                ratios[f"sterile_over_{control}"] = (
                    means["sterile"] / means[control] if means[control] > 0 else None
                )
        result[name] = {
            "observations": conditions,
            "mean_tpm": means,
            "ratios": ratios,
            "interpretation": "Descriptive TPM ratios; no statistical or protein-level inference",
        }
    return result


def write_cms_evidence(result: OrganelleResult, *, output: str | Path) -> OrganelleResult:
    """Export full evidence JSON and a TSV retaining separate evidence axes."""
    if result.operation_id not in {
        "phenotype.assess_cms_candidates",
        "phenotype.assess_cms_annotation",
    } or result.status not in {"ok", "warning"}:
        raise _invalid("Expected successful assess_cms_candidates result")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = thaw_json(result.metrics)
    json_path = directory / "cms_evidence.json"
    json_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    table_path = directory / "cms_evidence.tsv"
    fields = [
        "candidate_id",
        "length_aa",
        "evidence_labels",
        "hydrophobicity_hint",
        "cms_reference_hits",
        "conserved_gene_hits",
        "unmatched_regions",
        "expression",
        "transmembrane_topology",
        "genomic_context",
        "rna_editing",
        "evidence_availability",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in metrics["candidate_table"]:
            writer.writerow(
                {
                    key: json.dumps(row.get(key))
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fields
                }
            )
    artifacts = (
        ArtifactRef.from_path(
            json_path, kind="result", format="json", media_type="application/json"
        ),
        ArtifactRef.from_path(
            table_path, kind="table", format="tsv", media_type="text/tab-separated-values"
        ),
    )
    return result.model_copy(
        update={"operation_id": "phenotype.write_cms_evidence", "artifacts": artifacts}
    )
