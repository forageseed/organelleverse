from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("pysam") is None:
    pytest.skip("pysam is required for graph evidence tests", allow_module_level=True)

from organelleverse.quality_control.graph import collect_graph_evidence
from organelleverse.quality_control.policy import QC_POLICY_V5

from .test_graph_evidence import _evidence


def _cycle_records() -> tuple[tuple[str, int, str, int], ...]:
    records: list[tuple[str, int, str, int]] = []
    for index in range(3):
        records.append((f"ab-{index}", 150, "500M", 0))
        records.append((f"bc-{index}", 550, "500M", 0))
        records.append((f"wrap-{index}", 950, "250M250S", 0))
        records.append((f"wrap-{index}", 0, "250S250M", 2048))
    return tuple(records)


def test_rotating_circular_origin_preserves_graph_decision_inputs(tmp_path: Path) -> None:
    a = "A" * 400
    b = "C" * 400
    c = "G" * 400
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    segments = {"A": a, "B": b, "C": c}
    first, first_mapping = _evidence(
        first_dir,
        path_steps="A+,B+,C+",
        sequence=a + b + c,
        records=_cycle_records(),
        segment_sequences=segments,
    )
    second, second_mapping = _evidence(
        second_dir,
        path_steps="B+,C+,A+",
        sequence=b + c + a,
        records=_cycle_records(),
        segment_sequences=segments,
    )
    with first.assembly_graph_path.open("a") as handle:
        handle.write("L\tC\t+\tA\t+\t0M\n")
    with second.assembly_graph_path.open("a") as handle:
        handle.write("L\tA\t+\tB\t+\t0M\n")

    first_graph = collect_graph_evidence(first, first_mapping, QC_POLICY_V5)
    second_graph = collect_graph_evidence(second, second_mapping, QC_POLICY_V5)

    assert first_graph.gfa_summary == second_graph.gfa_summary
    assert sorted(
        (
            item.left_segment,
            item.right_segment,
            item.status,
            item.supporting_reads,
            item.contradicting_reads,
        )
        for item in first_graph.junction_support
    ) == sorted(
        (
            item.left_segment,
            item.right_segment,
            item.status,
            item.supporting_reads,
            item.contradicting_reads,
        )
        for item in second_graph.junction_support
    )
    assert len(first_graph.structural_error_candidates) == len(
        second_graph.structural_error_candidates
    )
