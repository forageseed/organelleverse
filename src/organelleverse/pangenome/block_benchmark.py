"""Published alignment-block consistency; not base-resolution breakpoint truth.

An HSP supplies a directed query interval and a directed reference interval.
Without its alignment CIGAR, interior base-to-base offsets are not asserted.
Each query base is classified as unmapped, ambiguous, or uniquely mapped to a
reference graph locus; unique placements are checked against the HSP's interval
and strand. Repeated/overlapping HSP bases are reported as block observations.
"""

from bisect import bisect_right
from collections import Counter, defaultdict
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from organelleverse.core.result import OrganelleResult

from .graph import path_sequences, path_steps


class AlignmentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str
    query_start: StrictInt = Field(ge=0)
    query_end: StrictInt = Field(gt=0)
    reference_start: StrictInt = Field(ge=0)
    reference_end: StrictInt = Field(gt=0)
    strand: Literal["+", "-"]
    source_row: StrictInt = Field(ge=1)


class BlockBenchmarkTruth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["organelleverse.pangenome.block-truth.v1"]
    benchmark_id: str
    benchmark_version: str
    case_id: str
    source_description: str
    reference: str
    sequences: dict[str, str] = Field(min_length=2)
    blocks: tuple[AlignmentBlock, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_blocks(self) -> Self:
        if self.reference not in self.sequences:
            raise ValueError("Reference is missing from the declared input molecules")
        if any(not s or set(s.upper()) - set("ACGT") for s in self.sequences.values()):
            raise ValueError("Block benchmark requires complete A/C/G/T molecules")
        if len({b.source_row for b in self.blocks}) != len(self.blocks):
            raise ValueError("Published rows must be unique")
        for b in self.blocks:
            if b.query == self.reference or b.query not in self.sequences:
                raise ValueError("Block query must name a non-reference input molecule")
            if not 0 <= b.query_start < b.query_end <= len(self.sequences[b.query]):
                raise ValueError("Query block is outside its native molecule")
            if not 0 <= b.reference_start < b.reference_end <= len(self.sequences[self.reference]):
                raise ValueError("Reference block is outside its declared reference")
        return self


def score_blocks(graph_path, truth: BlockBenchmarkTruth):
    observed = {n: s.upper() for n, s in path_sequences(graph_path).items()}
    if observed != {n: s.upper() for n, s in truth.sequences.items()}:
        raise ValueError("Graph must exactly reconstruct all declared benchmark molecules")
    paths = path_steps(graph_path)
    if any(row["overlap_previous"] for rows in paths.values() for row in rows):
        raise ValueError("Block evaluation requires zero-overlap complete graph paths")
    references = defaultdict(list)
    for row in paths[truth.reference]:
        references[row["node"]].append(row)
    categories = ("consistent_unique", "inconsistent_unique", "ambiguous", "unmapped")
    total = Counter({key: 0 for key in categories})
    results = []
    for block in truth.blocks:
        rows = paths[block.query]
        starts = [r["start"] for r in rows]
        count = Counter({key: 0 for key in categories})
        spans = []
        for pos in range(block.query_start, block.query_end):
            row = rows[bisect_right(starts, pos) - 1]
            offset = pos - row["start"]
            physical = offset if row["orientation"] == "+" else row["node_length"] - 1 - offset
            placements = {
                (
                    r["start"] + physical if r["orientation"] == "+" else r["end"] - 1 - physical,
                    "+" if r["orientation"] == row["orientation"] else "-",
                )
                for r in references[row["node"]]
            }
            if not placements:
                category = "unmapped"
            elif len(placements) > 1:
                category = "ambiguous"
            else:
                ref_pos, strand = next(iter(placements))
                category = (
                    "consistent_unique"
                    if (
                        block.reference_start <= ref_pos < block.reference_end
                        and strand == block.strand
                    )
                    else "inconsistent_unique"
                )
            count[category] += 1
            if spans and spans[-1]["category"] == category:
                spans[-1]["end"] = pos + 1
            else:
                spans.append({"start": pos, "end": pos + 1, "category": category})
        total.update(count)
        results.append({**block.model_dump(), "counts": dict(count), "query_spans": spans})
    denominator = sum(total.values())
    return {
        "benchmark_id": truth.benchmark_id,
        "benchmark_version": truth.benchmark_version,
        "case_id": truth.case_id,
        "source_description": truth.source_description,
        "definition": "Published HSP interval and strand consistency of unique reference graph placements; no base-pair CIGAR or breakpoint error inferred. Overlapping HSP bases are separate block observations; unmapped variant bases are not automatically graph errors.",
        "blocks": results,
        "counts": dict(total),
        "block_base_observations": denominator,
        "metrics": {
            **{"block_" + key + "_fraction": total[key] / denominator for key in categories},
            "exact_input_paths": 1.0,
        },
    }


def evaluate_block_benchmark(result: OrganelleResult, *, truth_artifact_id: str) -> OrganelleResult:
    from .benchmark_evaluator import _evaluate_benchmark

    return _evaluate_benchmark(
        result,
        truth_artifact_id,
        BlockBenchmarkTruth,
        score_blocks,
        "pangenome.evaluate_block_benchmark",
        "Published alignment-block consistency",
    )
