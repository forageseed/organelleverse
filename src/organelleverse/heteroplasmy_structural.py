"""Estimate structural-configuration read fractions from GraphAligner GAF.

Only caller-specified competing, oriented graph walks are counted. A molecule
is eligible at a branch when a full-query GAF alignment spans exactly
one configured path signature after MAPQ filtering. This is a read fraction
conditional on unique alignment and informative span, not a molecule-copy
fraction or a structural variant caller.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import ErrorDetail, Finding, OrganelleResult

_OPERATION = "estimate_structural_fractions"
_VERSION = "1.0"
_TOKEN = re.compile(r"([><])([^><]+)")


def estimate_structural_fractions(
    gaf_path: str | Path,
    configurations_tsv: str | Path,
    *,
    scope: Literal["mitochondrion", "plastid"] = "mitochondrion",
    min_mapping_quality: int = 20,
) -> OrganelleResult:
    """Count long reads uniquely supporting competing oriented graph walks.

    `configurations_tsv` has columns `branch_id,configuration_id,path_signature`.
    A signature is a GAF oriented walk such as `>left>choiceA>right`; all
    configurations within a branch must share an oriented first token
    (alternative outgoing junctions) or an oriented last token (alternative
    incoming junctions). GAF query names identify molecules. Only records with
    a full query interval (qstart=0, qend=query_length), non-star graph path, and
    `min_mapping_quality <= MAPQ < 255` are eligible. MAPQ 255 means unknown
    and is excluded. Candidate signatures are searched as contiguous oriented
    token walks in either direction in the alignment's graph path. Reverse
    traversal reverses node order and flips every node orientation.

    For each branch, unique informative denominator = distinct query names
    matching exactly one configuration over all eligible GAF records. Reads
    matching multiple configurations are reported as ambiguous and excluded;
    reads matching none do not enter the denominator. Repeated records for the
    same query/configuration count once. Wilson 95% binomial intervals are
    conditional on this denominator.

    This requires GraphAligner GAF from long reads mapped to the same graph
    node identifiers used in the TSV. It does not infer candidate configurations,
    correct mapping bias, resolve NUMTs, or estimate DNA molecule copy number.
    """
    parameters = {
        "gaf_path": str(gaf_path),
        "configurations_tsv": str(configurations_tsv),
        "scope": scope,
        "min_mapping_quality": min_mapping_quality,
    }
    if scope not in ("mitochondrion", "plastid"):
        return _failed(
            "scope must be mitochondrion or plastid.",
            scope="mitochondrion",
            code="heteroplasmy.invalid_scope",
            parameters=parameters,
        )
    if not 0 <= min_mapping_quality < 255:
        return _failed(
            "min_mapping_quality must be in [0, 254].",
            scope=scope,
            code="heteroplasmy.invalid_mapq",
            parameters=parameters,
        )
    try:
        configs = _read_configurations(configurations_tsv)
    except (OSError, ValueError) as exc:
        return _failed(
            f"Invalid configuration TSV: {exc}",
            scope=scope,
            code="heteroplasmy.invalid_configurations",
            parameters=parameters,
        )
    try:
        evidence = _read_gaf(gaf_path, configs, min_mapping_quality)
    except (OSError, ValueError) as exc:
        return _failed(
            f"Could not read GAF evidence: {exc}",
            scope=scope,
            code="heteroplasmy.invalid_gaf",
            parameters=parameters,
        )

    branch_rows: list[dict[str, object]] = []
    for branch_id, alternatives in configs.items():
        assigned = evidence[branch_id]
        counts = {config_id: 0 for config_id, _ in alternatives}
        ambiguous_reads = 0
        observed_reads = 0
        for matched in assigned.values():
            if not matched:
                continue
            observed_reads += 1
            if len(matched) == 1:
                counts[next(iter(matched))] += 1
            else:
                ambiguous_reads += 1
        denominator = sum(counts.values())
        configurations = []
        for config_id, signature in alternatives:
            count = counts[config_id]
            fraction = count / denominator if denominator else None
            low, high = _wilson(count, denominator) if denominator else (None, None)
            configurations.append(
                {
                    "configuration_id": config_id,
                    "path_signature": signature,
                    "supporting_reads": count,
                    "fraction": round(fraction, 6) if fraction is not None else None,
                    "ci95_lower": round(low, 6) if low is not None else None,
                    "ci95_upper": round(high, 6) if high is not None else None,
                }
            )
        branch_rows.append(
            {
                "branch_id": branch_id,
                "denominator_unique_informative_reads": denominator,
                "observed_matching_reads": observed_reads,
                "ambiguous_reads_excluded": ambiguous_reads,
                "configurations": configurations,
                "status": "assessed" if denominator else "no_informative_reads",
            }
        )

    assessed = sum(row["status"] == "assessed" for row in branch_rows)
    return OrganelleResult(
        operation_id=f"heteroplasmy.{_OPERATION}",
        operation_version=_VERSION,
        scope=scope,
        status="ok",
        summary_text=f"Estimated structural read fractions for {assessed}/{len(branch_rows)} branches.",
        metrics={
            "branch_count": len(branch_rows),
            "assessed_branches": assessed,
            "branches": branch_rows,
            "denominator_definition": "Distinct long-read query names uniquely matching one supplied oriented path signature at the branch.",
        },
        findings=(
            Finding(
                code="heteroplasmy.assessed_structural_branches",
                metric="assessed_branches",
                value=assessed,
            ),
        ),
        flags=(
            "caller_defined_configurations_only",
            "graphaligner_gaf_long_reads",
            "read_fraction_not_molecule_copy_fraction",
            "numt_mapping_not_resolved",
        ),
        provenance=_provenance(parameters),
    )


def _read_configurations(path: str | Path) -> dict[str, tuple[tuple[str, str], ...]]:
    required = {"branch_id", "configuration_id", "path_signature"}
    grouped: dict[str, list[tuple[str, str, tuple[str, ...]]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("header must include branch_id,configuration_id,path_signature")
        for line, row in enumerate(reader, start=2):
            branch = (row.get("branch_id") or "").strip()
            config = (row.get("configuration_id") or "").strip()
            signature = (row.get("path_signature") or "").strip()
            tokens = _parse_walk(signature)
            if len(tokens) < 2:
                raise ValueError(f"line {line}: configuration needs two flanking anchors")
            if not branch or not config:
                raise ValueError(f"line {line}: branch_id and configuration_id are required")
            if (branch, config) in seen:
                raise ValueError(f"line {line}: duplicate configuration {branch}/{config}")
            seen.add((branch, config))
            grouped[branch].append((config, signature, tokens))
    if not grouped:
        raise ValueError("at least one branch configuration is required")
    result: dict[str, tuple[tuple[str, str], ...]] = {}
    for branch, rows in grouped.items():
        if len(rows) < 2:
            raise ValueError(f"branch {branch!r} needs at least two competing configurations")
        starts = {tokens[0] for _, _, tokens in rows}
        ends = {tokens[-1] for _, _, tokens in rows}
        if len(starts) != 1 and len(ends) != 1:
            raise ValueError(
                f"branch {branch!r} configurations must share an oriented start or end anchor"
            )
        if len({tokens for _, _, tokens in rows}) != len(rows):
            raise ValueError(f"branch {branch!r} has duplicate path signatures")
        result[branch] = tuple((config, signature) for config, signature, _ in rows)
    return result


def _parse_walk(path: str) -> tuple[str, ...]:
    tokens = tuple(f"{orientation}{name}" for orientation, name in _TOKEN.findall(path))
    if not tokens or "".join(tokens) != path:
        raise ValueError(f"invalid oriented GAF path signature {path!r}")
    return tokens


def _read_gaf(
    path: str | Path,
    configs: dict[str, tuple[tuple[str, str], ...]],
    min_mapq: int,
) -> dict[str, dict[str, set[str]]]:
    token_configs = {
        branch: tuple((config, _parse_walk(signature)) for config, signature in alternatives)
        for branch, alternatives in configs.items()
    }
    assigned: dict[str, dict[str, set[str]]] = {branch: defaultdict(set) for branch in configs}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#") or line.startswith("@"):
                continue
            fields = line.rstrip(chr(10)).split(chr(9))
            if len(fields) < 12:
                raise ValueError(f"line {line_number}: GAF record has fewer than 12 fields")
            try:
                query = fields[0]
                query_length, query_start, query_end = map(int, fields[1:4])
                mapq = int(fields[11])
            except ValueError as exc:
                raise ValueError(f"line {line_number}: invalid numeric GAF field") from exc
            if not query or query_length < 1 or not (0 <= query_start <= query_end <= query_length):
                raise ValueError(f"line {line_number}: invalid query name/coordinates")
            if query_start != 0 or query_end != query_length or not min_mapq <= mapq < 255:
                continue
            path_field = fields[5]
            if path_field == "*":
                continue
            try:
                walk = _parse_walk(path_field)
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
            for branch, alternatives in token_configs.items():
                matches = {
                    config for config, signature in alternatives if _contains_walk(walk, signature)
                }
                if matches:
                    assigned[branch][query].update(matches)
    return assigned


def _contains_walk(walk: tuple[str, ...], signature: tuple[str, ...]) -> bool:
    size = len(signature)
    reverse = tuple(
        ("<" if token[0] == ">" else ">") + token[1:] for token in reversed(signature)
    )
    return any(
        walk[start : start + size] in (signature, reverse)
        for start in range(len(walk) - size + 1)
    )


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return center - half_width, center + half_width


def _provenance(parameters: dict[str, object]) -> ResultProvenance:
    payload = json.dumps(
        parameters, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    try:
        package_version = version("organelleverse")
    except PackageNotFoundError:
        package_version = "0.0.1"
    return ResultProvenance(
        operation_id=f"heteroplasmy.{_OPERATION}",
        operation_version=_VERSION,
        package_version=package_version,
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=hashlib.sha256(payload).hexdigest(),
        requested_backend="GraphAligner GAF",
        actual_backend="GraphAligner GAF",
        attempted_backends=("GraphAligner GAF",),
    )


def _failed(
    message: str, *, scope: str, code: str, parameters: dict[str, object]
) -> OrganelleResult:
    return OrganelleResult(
        operation_id=f"heteroplasmy.{_OPERATION}",
        operation_version=_VERSION,
        scope=scope,
        status="failed",
        summary_text=message,
        errors=(ErrorDetail(code=code, message=message),),
        provenance=_provenance(parameters),
    )
