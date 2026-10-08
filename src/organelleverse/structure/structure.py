"""Genome structure: multi-configuration, introns, repeats. Self-contained.

- multiconf(): detect repeat-mediated sub-genomic configurations. Direct and
  inverted repeats are found via a canonical k-mer index (both strands), and
  configurations are counted per the plant-mito recombination model (direct ->
  2 configs; inverted -> 2 isomers; Gualberto 2014, Wang 2024).
- introns(): parse intron features from GenBank; report count + phase distribution.
- repeats(): detect exact SSRs using MISA copy thresholds by motif length.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from .._bio import read_fasta
from .._sequtil import reverse_complement
from ..annotation.genbank import parse_genbank
from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError, OrganelleParameterError
from ..core.frozen import thaw_json
from ..core.genome import OrganelleGenome
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult, ResultScope

_OPERATION_VERSION = "1.0"

# Legacy short organelle names accepted by ``resolve_configs`` mapped onto the
# canonical result scope vocabulary.
_SCOPE_ALIASES: dict[str, ResultScope] = {
    "mito": "mitochondrion",
    "mitochondrion": "mitochondrion",
    "chloro": "plastid",
    "plastid": "plastid",
}


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scope(organelle: str) -> ResultScope:
    try:
        return _SCOPE_ALIASES[organelle]
    except KeyError:
        raise OrganelleParameterError(
            code="structure.unknown_organelle",
            message="organelle must be 'mitochondrion' or 'plastid'",
            details={"organelle": organelle, "accepted": sorted(_SCOPE_ALIASES)},
        ) from None


def _provenance(
    operation_id: str,
    *,
    parameters: dict[str, object],
    started_at: datetime,
    finished_at: datetime,
    input_object_ids: tuple[str, ...] = (),
    input_artifact_hashes: tuple[str, ...] = (),
) -> ResultProvenance:
    return ResultProvenance(
        operation_id=operation_id,
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=input_object_ids,
        input_artifact_hashes=input_artifact_hashes,
        parameters_hash=_sha256_json(parameters),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=max(0.0, (finished_at - started_at).total_seconds()),
    )


def multiconf(
    genome_fasta: str | Path,
    *,
    min_repeat_len: int = 50,
    max_mismatches: int = 0,
) -> OrganelleResult:
    """Report repeat-pair recombination candidates separately for each record.

    Scans for direct and inverted repeats >= ``min_repeat_len``; each repeat
    pair is a candidate recombination junction producing sub-genomic
    configurations. Per the plant-mitochondrial recombination model
    (Gualberto et al. 2014; Wang et al. 2024):
      - each **direct-repeat** pair yields **2** configurations (master + sub-circle)
      - each **inverted-repeat** pair yields **2** isomers (flip-out)
    The legacy ``predicted_configs`` and ``candidate_configs`` fields count
    two potential outcomes per repeat pair, not distinct validated genomic
    configurations. Repeat positions are 1-based within ``sequence_id``;
    separate FASTA records are never joined or treated as one molecule.
    ``max_mismatches`` permits that many substitutions between exact k-mer
    seeds on one repeat-pair diagonal; the default reports exact repeats.
    """
    started_at = datetime.now(UTC)
    fasta_path = Path(genome_fasta)
    seqs = read_fasta(fasta_path)
    repeats = _find_record_repeats(seqs, min_repeat_len, max_mismatches=max_mismatches)
    direct = [r for r in repeats if r["type"] == "direct"]
    inverted = [r for r in repeats if r["type"] == "inverted"]
    direct_pairs = len(direct)
    inverted_pairs = len(inverted)
    predicted_configs = 2 * direct_pairs + 2 * inverted_pairs
    return OrganelleResult(
        operation_id="structure.multiconf",
        operation_version=_OPERATION_VERSION,
        scope="mitochondrion",
        status="ok",
        summary_text=(
            f"{direct_pairs} direct + {inverted_pairs} inverted repeat "
            f"pairs -> {predicted_configs} candidate recombination outcomes (not validated configurations)."
        ),
        metrics={
            "direct_repeat_pairs": direct_pairs,
            "inverted_repeat_pairs": inverted_pairs,
            "candidate_configs": predicted_configs,
            "predicted_configs": predicted_configs,
            "repeats": repeats,
            "sequence_count": len(seqs),
        },
        findings=(
            Finding(
                code="structure.direct_repeat_pairs",
                metric="direct_repeat_pairs",
                value=direct_pairs,
            ),
            Finding(
                code="structure.inverted_repeat_pairs",
                metric="inverted_repeat_pairs",
                value=inverted_pairs,
            ),
            Finding(
                code="structure.predicted_configs",
                metric="predicted_configs",
                value=predicted_configs,
            ),
        ),
        flags=("repeat_pair_candidates_only",),
        provenance=_provenance(
            "structure.multiconf",
            parameters={"min_repeat_len": min_repeat_len, "max_mismatches": max_mismatches},
            input_artifact_hashes=(_sha256_file(fasta_path),),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
    )


def introns(genome: OrganelleGenome) -> OrganelleResult:
    """Analyze explicitly annotated intron features (GenBank ``intron`` only).

    Exon gaps implied by multi-part CDS features are NOT inferred: an
    annotation that records no ``intron`` features honestly yields 0 even
    when its CDS are spliced (observed on plastome transfer annotations).
    """
    started_at = datetime.now(UTC)
    g = genome
    if g.annotation is None:
        return OrganelleResult(
            operation_id="structure.introns",
            operation_version=_OPERATION_VERSION,
            scope=g.organelle,
            status="failed",
            summary_text="introns() needs an annotation artifact.",
            provenance=_provenance(
                "structure.introns",
                parameters={},
                input_object_ids=(g.object_id,),
                started_at=started_at,
                finished_at=datetime.now(UTC),
            ),
            errors=(
                ErrorDetail(
                    code="structure.missing_annotation",
                    message="introns() requires an OrganelleGenome carrying an annotation.",
                    details={"organelle": g.organelle},
                ),
            ),
        )
    document = parse_genbank(g.annotation.resolve())
    intron_length_mod3 = {0: 0, 1: 0, 2: 0}
    total = 0
    for record in document.records:
        for feature in record.features:
            if feature.type.casefold() == "intron":
                total += 1
                length = sum(part.end - part.start for part in feature.parts)
                mod3 = length % 3
                intron_length_mod3[mod3] = intron_length_mod3.get(mod3, 0) + 1
    return OrganelleResult(
        operation_id="structure.introns",
        operation_version=_OPERATION_VERSION,
        scope=g.organelle,
        status="ok",
        summary_text=f"{total} introns (length%3 = 0/1/2: {intron_length_mod3[0]}/{intron_length_mod3[1]}/{intron_length_mod3[2]}).",
        metrics={
            "intron_count": total,
            "length_mod3_0": intron_length_mod3[0],
            "length_mod3_1": intron_length_mod3[1],
            "length_mod3_2": intron_length_mod3[2],
        },
        findings=(Finding(code="structure.intron_count", metric="intron_count", value=total),),
        flags=(),
        provenance=_provenance(
            "structure.introns",
            parameters={},
            input_object_ids=(g.object_id,),
            input_artifact_hashes=(g.annotation.sha256,),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
    )


def repeats(
    genome_fasta: str | Path,
    *,
    min_ssr_unit: int = 1,
    max_ssr_unit: int = 6,
    min_copy: int = 3,
) -> OrganelleResult:
    """Detect SSR (simple sequence repeat) motifs in a FASTA.

    Uses MISA copy thresholds (10/5/4/3/3/3 for unit lengths 1-6).
    ``min_copy`` can raise, but not lower, the threshold for any unit.
    """
    started_at = datetime.now(UTC)
    fasta_path = Path(genome_fasta)
    seqs = read_fasta(fasta_path)
    ssrs = [
        {"sequence_id": seq_id, **ssr}
        for seq_id, seq in seqs
        for ssr in _find_ssrs(seq.upper(), min_ssr_unit, max_ssr_unit, min_copy)
    ]
    return OrganelleResult(
        operation_id="structure.repeats",
        operation_version=_OPERATION_VERSION,
        scope="mitochondrion",
        status="ok",
        summary_text=f"{len(ssrs)} SSR motifs detected.",
        metrics={"ssr_count": len(ssrs), "ssrs": ssrs},
        findings=(Finding(code="structure.ssr_count", metric="ssr_count", value=len(ssrs)),),
        flags=(),
        provenance=_provenance(
            "structure.repeats",
            parameters={
                "min_ssr_unit": min_ssr_unit,
                "max_ssr_unit": max_ssr_unit,
                "min_copy": min_copy,
            },
            input_artifact_hashes=(_sha256_file(fasta_path),),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
    )


def _find_record_repeats(
    records: list[tuple[str, str]],
    min_len: int,
    *,
    max_mismatches: int = 0,
) -> list[dict]:
    """Find intra-record repeats with coordinates local to each FASTA record."""
    return [
        {"sequence_id": sequence_id, **repeat}
        for sequence_id, sequence in records
        for repeat in _find_repeats(sequence.upper(), min_len, max_mismatches=max_mismatches)
    ]


def _find_repeats(seq: str, min_len: int, *, max_mismatches: int = 0) -> list[dict]:
    """Find direct and inverted repeats of length >= ``min_len``.

    Definitions (standard repeat-mediated recombination terminology):
    - **direct repeat**: two non-overlapping substrings that are identical,
      ``seq[i:i+L] == seq[j:j+L]`` for ``i != j``.
    - **inverted repeat**: a substring whose reverse complement appears
      elsewhere, ``seq[i:i+L] == reverse_complement(seq[j:j+L])``.

    Index canonical k-mers, then merge seed pairs along their alignment
    diagonal into maximal repeat pairs. Direct seeds share ``j - i``;
    inverted seeds share ``i + j``. Optional mismatches bridge short gaps
    between exact seed runs on the same diagonal.

    ``positions`` contains the two 1-based starts on the forward sequence.
    """
    if min_len < 1 or max_mismatches < 0:
        raise ValueError("min_len must be positive and max_mismatches nonnegative")
    if len(seq) < min_len:
        return []
    k = min_len
    by_canonical: dict[str, list[tuple[int, str]]] = {}
    for i in range(len(seq) - k + 1):
        sub = seq[i : i + k]
        if "N" in sub:
            continue
        rc = reverse_complement(sub)
        canonical = sub if sub <= rc else rc
        by_canonical.setdefault(canonical, []).append((i, sub))

    hits: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for occurrences in by_canonical.values():
        if len(occurrences) < 2:
            continue
        for idx_a in range(len(occurrences)):
            pos_a, fwd_a = occurrences[idx_a]
            for idx_b in range(idx_a + 1, len(occurrences)):
                pos_b, fwd_b = occurrences[idx_b]
                if pos_b - pos_a < k:
                    continue
                if fwd_a == fwd_b:
                    hits[("direct", pos_b - pos_a)].append((pos_a, pos_b))
                else:
                    hits[("inverted", pos_a + pos_b)].append((pos_a, pos_b))

    out: list[dict] = []
    for (kind, _diagonal), pairs in hits.items():
        pairs.sort()
        start_i, start_j = last_i, last_j = pairs[0]
        mismatches = 0
        for pair in [*pairs[1:], None]:
            added_mismatches = 0
            can_merge = pair is not None
            if pair is not None:
                i, j = pair
                gap = i - (last_i + k)
                if gap > 0:
                    if gap > max_mismatches - mismatches:
                        can_merge = False
                    else:
                        first_gap = seq[last_i + k : i]
                        if kind == "direct":
                            second_gap = seq[last_j + k : j]
                        else:
                            second_gap = reverse_complement(seq[j + k : last_j])
                        added_mismatches = sum(
                            a != b or a not in "ACGT"
                            for a, b in zip(first_gap, second_gap, strict=True)
                        )
                        can_merge = added_mismatches <= max_mismatches - mismatches
            if can_merge and pair is not None:
                last_i, last_j = pair
                mismatches += added_mismatches
                continue

            length = k + last_i - start_i
            second_start = start_j if kind == "direct" else last_j
            # Recombination pairs represent two distinct intervals.
            if length > second_start - start_i:
                trimmed = length - (second_start - start_i)
                length -= trimmed
                if kind == "inverted":
                    second_start += trimmed
            if length >= k:
                repeat = {
                    "type": kind,
                    "length": length,
                    "positions": [start_i + 1, second_start + 1],
                }
                if max_mismatches:
                    repeat["mismatches"] = mismatches
                out.append(repeat)
            if pair is not None:
                start_i, start_j = last_i, last_j = pair
                mismatches = 0
    out.sort(key=lambda repeat: (*repeat["positions"], repeat["type"]))
    return out


def _find_ssrs(seq: str, min_u: int, max_u: int, min_copy: int) -> list[dict]:
    """Find exact tandem SSRs with MISA's per-unit copy thresholds.

    Scan each unit length independently, as MISA does. Regex matches are
    maximal and non-overlapping within a unit length; loci may be reported for
    more than one unit length when both motifs meet their own thresholds.
    """
    thresholds = (10, 5, 4, 3, 3, 3)
    out: list[dict] = []
    for unit_len in range(max(1, min_u), min(6, max_u) + 1):
        copies_required = max(thresholds[unit_len - 1], min_copy)
        pattern = re.compile(rf"([ACGT]{{{unit_len}}})\1{{{copies_required - 1},}}")
        for match in pattern.finditer(seq):
            motif = match.group(1)
            out.append(
                {
                    "unit": motif,
                    "length": match.end() - match.start(),
                    "copies": (match.end() - match.start()) // unit_len,
                    "start": match.start() + 1,
                    "end": match.end(),
                }
            )
    out.sort(key=lambda r: (r["start"], len(r["unit"]), r["unit"]))
    return out


# ---------------------------------------------------------------------------
# B1 enhancement: resolve_configs — GFA-based multipartite structure
# ---------------------------------------------------------------------------


def resolve_configs(
    genome_fasta: str | Path,
    *,
    gfa_path: str | Path | None = None,
    min_repeat_len: int = 50,
    max_mismatches: int = 0,
    organelle: str = "mito",
) -> OrganelleResult:
    """Resolve sub-genomic configurations from repeat-mediated recombination.

    If a GFA (minigraph/pggb output) is provided, parse the assembly graph to
    enumerate branched paths (alternative configurations). Otherwise, infer
    candidate configurations from direct/inverted repeat pairs in the FASTA,
    using the standard plant-mito recombination model (direct -> 2 configs,
    inverted -> 2 isomers; Gualberto 2014, Wang 2024).
    ``max_mismatches`` applies only to repeat inference, not GFA parsing.
    """
    started_at = datetime.now(UTC)
    scope = _scope(organelle)
    fasta_path = Path(genome_fasta)
    seqs = read_fasta(fasta_path)
    input_hashes = [_sha256_file(fasta_path)]
    if gfa_path is not None:
        if not Path(gfa_path).is_file():
            raise OrganelleInputError(
                code="structure.missing_graph",
                message=f"Requested GFA is not a readable file: {gfa_path}",
            )
        configs = _parse_gfa_configs(Path(gfa_path))
        method = "gfa_graph"
        input_hashes.append(_sha256_file(Path(gfa_path)))
    else:
        repeats = _find_record_repeats(seqs, min_repeat_len, max_mismatches=max_mismatches)
        configs = _infer_configs_from_repeats(repeats)
        method = "repeat_inference"
    master_len = sum(len(seq) for _, seq in seqs)
    n_configs = len(configs)
    return OrganelleResult(
        operation_id="structure.resolve_configs",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="ok",
        summary_text=f"Resolved {n_configs} sub-genomic configuration(s) via {method}.",
        metrics={
            "sub_configs": n_configs,
            "master_length": master_len,
            "method": method,
            "configs": configs,
        },
        findings=(
            Finding(
                code="structure.sub_genomic_configs",
                metric="sub_genomic_configs",
                value=n_configs,
            ),
            Finding(code="structure.method", metric="method", value=method),
        ),
        flags=("multipartite_resolved",) if n_configs > 1 else (),
        provenance=_provenance(
            "structure.resolve_configs",
            parameters={
                "gfa_path": str(gfa_path) if gfa_path else None,
                "min_repeat_len": min_repeat_len,
                "max_mismatches": max_mismatches,
                "organelle": scope,
                "method": method,
            },
            input_artifact_hashes=tuple(input_hashes),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
    )


def write_multiconf(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write repeat-pair candidates from ``multiconf()``."""
    metrics = _result_metrics(result)
    repeats = list(metrics.get("repeats", ()))
    path = _resolve_output_path(output, "repeat_pairs.tsv")
    lines = ["type\tlength\tpositions\tsequence_id"]
    for repeat in repeats:
        positions = ",".join(str(p) for p in repeat.get("positions", ()))
        lines.append(
            f"{repeat.get('type', '')}\t{repeat.get('length', 0)}\t{positions}\t{repeat.get('sequence_id', '')}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_introns(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write intron summary metrics from ``introns()``."""
    metrics = _result_metrics(result)
    path = _resolve_output_path(output, "introns.tsv")
    lines = ["metric\tvalue"]
    for key in ("intron_count", "length_mod3_0", "length_mod3_1", "length_mod3_2"):
        lines.append(f"{key}\t{metrics.get(key, 0)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_repeats(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write SSR calls from ``repeats()``."""
    metrics = _result_metrics(result)
    ssrs = list(metrics.get("ssrs", ()))
    path = _resolve_output_path(output, "ssr_repeats.tsv")
    lines = ["start\tend\tunit\tlength\tcopies"]
    for ssr in ssrs:
        lines.append(
            f"{ssr.get('start', '')}\t{ssr.get('end', '')}\t{ssr.get('unit', '')}"
            f"\t{ssr.get('length', '')}\t{ssr.get('copies', '')}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_resolve_configs(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write resolved multipartite configuration rows from ``resolve_configs()``."""
    metrics = _result_metrics(result)
    configs = list(metrics.get("configs", ()))
    path = _resolve_output_path(output, "resolved_configs.tsv")
    keys = sorted({str(key) for row in configs if isinstance(row, Mapping) for key in row})
    lines = ["\t".join(keys) if keys else "config"]
    for row in configs:
        if isinstance(row, Mapping):
            lines.append("\t".join(str(row.get(key, "")) for key in keys))
        else:
            lines.append(str(row))
    path.write_text("\n".join(lines) + "\n")
    return path


_MATERIALIZERS = {
    "structure.multiconf": (write_multiconf, "structure_repeat_pairs"),
    "structure.introns": (write_introns, "structure_introns"),
    "structure.repeats": (write_repeats, "structure_ssr_repeats"),
    "structure.resolve_configs": (write_resolve_configs, "structure_resolved_configs"),
}


def materialize_result(result: OrganelleResult, output: str | Path) -> OrganelleResult:
    """Canonical write boundary for ``structure.*`` results.

    Mirrors ``annotation.writer.materialize_result`` / ``quality_control.writer.
    materialize_result``: it never recomputes, it only writes the already
    computed metrics of a successful Result and returns the same Result with the
    written artifact attached.
    """

    if not isinstance(result, OrganelleResult):
        raise OrganelleInputError(
            code="structure.unsupported_value",
            message="structure.write accepts a canonical OrganelleResult",
            details={"type": type(result).__name__},
        )
    if result.status == "failed":
        raise OrganelleInputError(
            code="structure.failed_result",
            message="structure.write requires a successful structure result",
            details={"operation_id": result.operation_id},
        )
    entry = _MATERIALIZERS.get(result.operation_id)
    if entry is None:
        raise OrganelleInputError(
            code="structure.unsupported_operation",
            message="structure.write accepts Results from structure operations",
            details={"operation_id": result.operation_id, "accepted": sorted(_MATERIALIZERS)},
        )
    writer, artifact_kind = entry
    path = writer(result, output)
    artifact = ArtifactRef.from_path(
        path,
        kind=artifact_kind,
        format="tsv",
        media_type="text/tab-separated-values",
    )
    return result.evolve(artifacts=(*result.artifacts, artifact))


def _result_metrics(result: OrganelleResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, OrganelleResult):
        return cast(dict[str, Any], thaw_json(result.metrics))
    observed = result.get("metrics") if isinstance(result, Mapping) else None
    if isinstance(observed, Mapping):
        return dict(observed)
    if isinstance(result, Mapping):
        return dict(result)
    raise TypeError("result must be an OrganelleResult or metrics mapping")


def _resolve_output_path(output: str | Path, default_name: str) -> Path:
    path = Path(output)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name


def _parse_gfa_configs(gfa: Path) -> list[dict]:
    """Parse a GFA file to enumerate alternative paths (configurations).

    GFA 'S' (segment) lines define nodes; 'L' (link) lines define edges.
    Branch points (segments with >2 links) indicate alternative configurations.
    """
    segments: dict[str, int] = {}
    links: dict[str, list[str]] = {}
    for line in gfa.read_text().splitlines():
        if line.startswith("S\t"):
            parts = line.split("\t")
            if len(parts) >= 3:
                # segment length from optional tags or sequence
                seq = parts[2]
                segments[parts[1]] = len(seq) if seq != "*" else 0
        elif line.startswith("L\t"):
            parts = line.split("\t")
            if len(parts) >= 5:
                links.setdefault(parts[1], []).append(parts[3])
                links.setdefault(parts[3], []).append(parts[1])
    # find branch points (nodes with degree > 2)
    branches = [
        {"segment": s, "degree": len(neigh)} for s, neigh in links.items() if len(neigh) > 2
    ]
    return [
        {
            "branch_segment": b["segment"],
            "degree": b["degree"],
            "alternative_paths": b["degree"] - 1,
        }
        for b in branches
    ]


def _infer_configs_from_repeats(repeats: list[dict]) -> list[dict]:
    """Infer candidate sub-genomic configs from repeat pairs.

    Each direct-repeat pair yields 2 configurations (master chromosome + a
    sub-genomic circle); each inverted-repeat pair yields 2 isomers (the
    intervening segment flips orientation). This is the standard
    repeat-mediated recombination model for plant mitochondrial genomes
    (Gualberto et al. 2014; Wang et al. 2024).
    """
    configs = []
    for r in repeats:
        # Both direct and inverted repeats produce 2 predicted configurations.
        n = 2
        configs.append(
            {
                "repeat_type": r["type"],
                "length": r["length"],
                "predicted_configs": n,
                "sequence_id": r.get("sequence_id"),
                "positions": r["positions"],
            }
        )
    return configs
