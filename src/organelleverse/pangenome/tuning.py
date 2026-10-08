"""Pure, evidence-preserving pangenome parameter estimators."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError
from ..core.genome import OrganelleGenome

RepeatCategory = Literal["interspersed", "simple", "satellite", "rna"]
RecommendationValue = str | int | float | bool | list[str]
Recommendation = dict[str, RecommendationValue]

# Poales 192-sample comparison (2026-09-24): p84/s5000 retained all six
# tested families as monophyletic; the repeat-derived s500 did not.
PLASTID_SEGMENT_MINIMUM = 5000
SEGMENT_POLICY_SOURCE = "Poales 192 plastomes benchmark, 2026-09-24; p84+s5000"


def protected_segment_length(genomes: list[OrganelleGenome], proposed: int) -> int:
    """Apply the empirical plastid PGGB floor; retain mitochondrial estimates."""
    return (
        max(proposed, PLASTID_SEGMENT_MINIMUM)
        if any(genome.organelle == "plastid" for genome in genomes)
        else proposed
    )


def recommendation_segment_estimate(recommendation: Recommendation) -> int:
    """Recover the pre-floor estimate from a validated recommendation's evidence."""
    span = cast(int, recommendation["longest_repeat_span"])
    if span == 0:
        return cast(int, recommendation["no_repeat_fallback"])
    multiplier = cast(float, recommendation["repeat_multiplier"])
    round_to = cast(int, recommendation["round_to"])
    return math.ceil(span * multiplier / round_to) * round_to


_RECOMMENDATION_KEYS = frozenset(
    (  # noqa: SIM905 - compact closed schema keeps this module bounded
        "schema_version policy_version input_hashes threads run_repeatmasker species "
        "identity_margin repeat_multiplier round_to no_repeat_fallback include_rna "
        "mash_executable mash_version repeatmasker_executable repeatmasker_version "
        "repeatmasker_library_identity repeatmasker_database_label identity "
        "segment_length max_mash_distance longest_repeat_span repeat_fallback_used "
        "mash_sketch_argv mash_distance_argv repeatmasker_argv recommendation_digest"
    ).split()
)


@dataclass(frozen=True)
class RepeatInterval:
    query: str
    start: int
    end: int
    category: RepeatCategory


@dataclass(frozen=True)
class SegmentLengthEstimate:
    segment_length: int
    longest_span: int
    category_spans: dict[str, int]
    fallback_used: bool


def estimate_identity(
    distances: tuple[float, ...] | list[float],
    *,
    margin: float = 2,
    minimum: float = 50,
    maximum: float = 100,
) -> float:
    """Apply MineGraph's pinned maximum-Mash-distance identity policy."""
    if (
        not distances
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in distances)
        or not math.isfinite(margin)
        or margin < 0
        or not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or not 0 < minimum <= maximum <= 100
    ):
        _reject("pangenome.invalid_mash_distance", "Mash estimator inputs are invalid")
    proposed = math.ceil(100 - max(distances) * 100) - margin
    return max(minimum, min(maximum, proposed))


def estimate_segment_length(
    intervals: tuple[RepeatInterval, ...] | list[RepeatInterval],
    *,
    multiplier: float = 1.2,
    round_to: int = 10,
    no_repeat_fallback: int = 5000,
    include_rna: bool = True,
) -> SegmentLengthEstimate:
    """Merge inclusive intervals per query/category and round the longest span."""
    if not math.isfinite(multiplier) or multiplier <= 0 or round_to <= 0 or no_repeat_fallback <= 0:
        _reject("pangenome.invalid_repeat_policy", "RepeatMasker estimator policy is invalid")
    enabled = {"interspersed", "simple", "satellite"}
    if include_rna:
        enabled.add("rna")
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for interval in intervals:
        if not interval.query or interval.start <= 0 or interval.end <= 0:
            _reject("pangenome.invalid_repeat_row", "RepeatMasker interval is invalid")
        if interval.category not in enabled:
            continue
        start, end = sorted((interval.start, interval.end))
        grouped.setdefault((interval.query, interval.category), []).append((start, end))

    category_spans: dict[str, int] = {}
    for (_, category), values in grouped.items():
        current_start = current_end = 0
        for start, end in sorted(values):
            if current_start and start <= current_end + 1:
                current_end = max(current_end, end)
                continue
            if current_start:
                category_spans[category] = max(
                    category_spans.get(category, 0), current_end - current_start + 1
                )
            current_start, current_end = start, end
        if current_start:
            category_spans[category] = max(
                category_spans.get(category, 0), current_end - current_start + 1
            )

    longest = max(category_spans.values(), default=0)
    if longest == 0:
        return SegmentLengthEstimate(
            segment_length=no_repeat_fallback,
            longest_span=0,
            category_spans={},
            fallback_used=True,
        )
    rounded = math.ceil(longest * multiplier / round_to) * round_to
    return SegmentLengthEstimate(
        segment_length=rounded,
        longest_span=longest,
        category_spans=category_spans,
        fallback_used=False,
    )


def parse_mash_distances(output: str) -> tuple[float, ...]:
    """Parse strict five-column ``mash dist`` stdout."""
    distances: list[float] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        columns = line.split("\t") if "\t" in line else line.split()
        if len(columns) != 5:
            _reject(
                "pangenome.malformed_mash_output",
                "Mash distance row must have five columns",
                line=line_number,
            )
        try:
            distance = float(columns[2])
            p_value = float(columns[3])
            shared_text, total_text = columns[4].split("/", maxsplit=1)
            if not shared_text.isdigit() or not total_text.isdigit():
                raise ValueError("shared hashes must be integers")
            shared, total = int(shared_text), int(total_text)
        except ValueError:
            _reject(
                "pangenome.malformed_mash_output",
                "Mash distance row contains malformed numeric evidence",
                line=line_number,
            )
        if not math.isfinite(distance) or not 0 <= distance <= 1:
            _reject(
                "pangenome.malformed_mash_output",
                "Mash distance must be finite and between zero and one",
                line=line_number,
            )
        if (
            not math.isfinite(p_value)
            or not 0 <= p_value <= 1
            or total <= 0
            or not 0 <= shared <= total
        ):
            _reject(
                "pangenome.malformed_mash_output",
                "Mash p-value or shared hash count is outside valid bounds",
                line=line_number,
            )
        distances.append(distance)
    if not distances:
        _reject("pangenome.empty_mash_output", "Mash produced no distance rows")
    return tuple(distances)


def parse_repeatmasker_output(output: str) -> tuple[RepeatInterval, ...]:
    """Parse RepeatMasker ``.out`` rows into normalized typed intervals."""
    # ProcessRepeats emits this standalone sentence instead of table headers
    # when no repeat survives annotation (RepeatMasker 4.2.2, real release gate).
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    no_hits_prefix = "There were no repetitive sequences detected in "
    if (
        len(lines) == 1
        and lines[0].startswith(no_hits_prefix)
        and lines[0][len(no_hits_prefix) :].strip()
    ):
        return ()
    intervals: list[RepeatInterval] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if _repeatmasker_header(line):
            continue
        columns = line.split()
        if not columns[0].isdigit():
            _reject(
                "pangenome.malformed_repeatmasker_output",
                "RepeatMasker emitted an unrecognized non-data row",
                line=line_number,
            )
        if len(columns) < 14:
            _reject(
                "pangenome.malformed_repeatmasker_output",
                "RepeatMasker data row has too few columns",
                line=line_number,
            )
        try:
            start, end = int(columns[5]), int(columns[6])
        except ValueError:
            _reject(
                "pangenome.malformed_repeatmasker_output",
                "RepeatMasker query coordinates must be integers",
                line=line_number,
            )
        query = columns[4]
        class_family = columns[10].casefold()
        category: RepeatCategory | None
        if "satellite" in class_family:
            category = "satellite"
        elif "rna" in class_family:
            category = "rna"
        elif "simple" in class_family or "low_complexity" in class_family:
            category = "simple"
        elif "/" in class_family:
            category = "interspersed"
        else:
            category = None
        if not query or start <= 0 or end <= 0:
            _reject(
                "pangenome.malformed_repeatmasker_output",
                "RepeatMasker query and coordinates must be positive",
                line=line_number,
            )
        if category is not None:
            normalized_start, normalized_end = sorted((start, end))
            intervals.append(RepeatInterval(query, normalized_start, normalized_end, category))
    return tuple(intervals)


def sequence_input_hashes(genomes: list[OrganelleGenome]) -> tuple[str, ...]:
    """Return the ordered hashes of sequence artifacts consumed by graph tuning/building."""
    return tuple(genome.sequence.sha256 for genome in genomes if genome.sequence is not None)


def _repeatmasker_header(line: str) -> bool:
    lowered = line.casefold()
    return lowered.startswith(("sw ", "score ")) or set(line) <= {"-", " "}


def validate_recommendation(
    value: Mapping[str, object],
    *,
    input_hashes: tuple[str, ...],
    identity: float,
    segment_length: int,
) -> Recommendation:
    """Strictly verify one unchanged browser-safe recommendation payload."""
    if frozenset(value) != _RECOMMENDATION_KEYS:
        _reject(
            "pangenome.invalid_recommendation",
            "Recommendation fields do not match the closed schema",
        )
    raw = cast(dict[str, Any], dict(value))
    try:
        string_fields = (
            "schema_version",
            "policy_version",
            "species",
            "mash_executable",
            "mash_version",
            "repeatmasker_executable",
            "repeatmasker_version",
            "repeatmasker_library_identity",
            "repeatmasker_database_label",
            "recommendation_digest",
        )
        integer_fields = (
            "threads",
            "round_to",
            "no_repeat_fallback",
            "segment_length",
            "longest_repeat_span",
        )
        boolean_fields = ("run_repeatmasker", "include_rna", "repeat_fallback_used")
        numeric_fields = (
            "identity_margin",
            "repeat_multiplier",
            "identity",
            "max_mash_distance",
        )
        if any(not isinstance(raw[name], str) for name in string_fields):
            raise ValueError("string field type")
        if any(type(raw[name]) is not int for name in integer_fields):
            raise ValueError("integer field type")
        if any(type(raw[name]) is not bool for name in boolean_fields):
            raise ValueError("boolean field type")
        if any(
            isinstance(raw[name], bool)
            or not isinstance(raw[name], (int, float))
            or not math.isfinite(raw[name])
            for name in numeric_fields
        ):
            raise ValueError("numeric field type")
        if raw["schema_version"] != "organelleverse.pangenome.recommendation.v1":
            raise ValueError("schema version")
        if raw["policy_version"] not in {
            "minegraph-compatible.v1",
            "sample-mash-minegraph-prior.v2",
            "sample-mash-plastid-floor.v3",
            "sample-mash-mitochondrial.v3",
        }:
            raise ValueError("policy version")
        recorded_hashes = raw["input_hashes"]
        if not isinstance(recorded_hashes, list):
            raise ValueError("input hashes")
        recorded_hash_items = cast(list[object], recorded_hashes)
        if not all(
            isinstance(item, str)
            and len(item) == 64
            and all(character in "0123456789abcdef" for character in item)
            for item in recorded_hash_items
        ):
            raise ValueError("input hashes")
        recommended_identity = raw["identity"]
        recommended_segment = raw["segment_length"]
        if (
            isinstance(recommended_identity, bool)
            or not isinstance(recommended_identity, (int, float))
            or not math.isfinite(recommended_identity)
            or not 50 <= recommended_identity <= 100
            or isinstance(recommended_segment, bool)
            or not isinstance(recommended_segment, int)
            or recommended_segment <= 0
            or not 1 <= raw["threads"] <= 64
            or not 0 <= raw["identity_margin"] <= 10
            or not 0.1 <= raw["repeat_multiplier"] <= 10
            or not 1 <= raw["round_to"] <= 100_000
            or not 1 <= raw["no_repeat_fallback"] <= 10_000_000
            or not 0 <= raw["max_mash_distance"] <= 1
            or raw["longest_repeat_span"] < 0
        ):
            raise ValueError("recommended values")
        for name in ("mash_sketch_argv", "mash_distance_argv", "repeatmasker_argv"):
            argv = raw[name]
            if not isinstance(argv, list) or not all(
                isinstance(item, str) for item in cast(list[object], argv)
            ):
                raise ValueError(name)
        digest = raw["recommendation_digest"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("digest")
        unsigned = dict(raw)
        unsigned.pop("recommendation_digest")
        expected_digest = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        if digest != expected_digest:
            raise ValueError("digest")
    except (KeyError, TypeError, ValueError) as error:
        raise OrganelleInputError(
            code="pangenome.invalid_recommendation",
            message="Pangenome recommendation is malformed or has been changed",
            details={"reason": str(error)},
        ) from error
    if (
        recorded_hashes != list(input_hashes)
        or recommended_identity != identity
        or recommended_segment != segment_length
    ):
        raise OrganelleInputError(
            code="pangenome.recommendation_mismatch",
            message="Pangenome inputs or adopted values differ from the recommendation",
        )
    return cast(Recommendation, dict(raw))


def write_adoption_record(
    path: str | Path,
    *,
    recommendation: Recommendation,
    graph: ArtifactRef,
) -> ArtifactRef:
    """Write the graph-owned proof of exact parameter adoption."""
    adoption = {
        "schema_version": "organelleverse.pangenome.adoption.v1",
        "recommendation_digest": recommendation["recommendation_digest"],
        "input_hashes": recommendation["input_hashes"],
        "identity": recommendation["identity"],
        "segment_length": recommendation["segment_length"],
        "graph_sha256": graph.sha256,
    }
    target = Path(path)
    target.write_text(_canonical_json(adoption) + "\n", encoding="utf-8")
    return ArtifactRef.from_path(
        target,
        kind="pangenome_parameter_adoption",
        format="json",
        media_type="application/json",
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def _reject(code: str, message: str, **details: int) -> NoReturn:
    raise OrganelleInputError(code=code, message=message, details=details)
