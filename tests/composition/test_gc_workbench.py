"""Numerical and file contracts used by the GC scientific workbench."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from organelleverse.composition.gc_core import compute_gc_content
from organelleverse.core.errors import OrganelleParameterError
from organelleverse.core.frozen import thaw_json
from organelleverse.operations.analysis_runners import ANALYSIS_TOOL_RUNNERS


def fasta(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "input.fasta"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_step_preserves_old_windows_and_tail_with_new_gc_fractions(tmp_path):
    source = fasta(tmp_path, ">one\nGGCCAAAANT\n")
    result = compute_gc_content(source, 4)
    assert result["gc_content"] == 0.4 and result["total_length"] == 10
    assert result["windows"] == [
        {"start": 1, "end": 4, "gc_skew": 0.0, "gc_content": 1.0},
        {"start": 5, "end": 8, "gc_skew": 0.0, "gc_content": 0.0},
    ]
    assert compute_gc_content(source, 4, step=None) == result


def test_explicit_step_emits_overlapping_full_linear_windows(tmp_path):
    source = fasta(tmp_path, ">one\nGGCCAAAANT\n")
    result = compute_gc_content(source, 4, step=2)
    assert result["windows"] == [
        {"start": 1, "end": 4, "gc_skew": 0.0, "gc_content": 1.0},
        {"start": 3, "end": 6, "gc_skew": -1.0, "gc_content": 0.5},
        {"start": 5, "end": 8, "gc_skew": 0.0, "gc_content": 0.0},
        {"start": 7, "end": 10, "gc_skew": 0.0, "gc_content": 0.0},
    ]


@pytest.mark.parametrize(("sequence", "gc", "skew"), [("gcNN", 0.5, 0.0), ("ANNN", 0.0, 0.0)])
def test_short_input_ambiguity_denominator_and_zero_gc_skew(tmp_path, sequence, gc, skew):
    result = compute_gc_content(fasta(tmp_path, f">one\n{sequence}\n"), 10, step=3)
    assert result["gc_content"] == gc
    assert result["windows"] == [{"start": 1, "end": 4, "gc_skew": skew, "gc_content": gc}]


def test_public_default_keeps_historical_multirecord_concatenation(tmp_path):
    result = compute_gc_content(fasta(tmp_path, ">one\nGGGG\n>two\nCCCC\n"), 6)
    assert result["gc_content"] == 1.0 and result["total_length"] == 8
    assert result["windows"] == [{"start": 1, "end": 6, "gc_skew": 0.3333, "gc_content": 1.0}]


@pytest.mark.parametrize("step", [0, -1, True, "2", 1.5])
def test_invalid_step_is_rejected_before_file_io(step):
    with pytest.raises(OrganelleParameterError):
        compute_gc_content("does-not-exist.fasta", step=step)


def test_registered_parameter_schema_and_invocation_accept_step(tmp_path):
    from organelleverse import operations

    schema = operations.parameter_schema("composition.compute_gc_content")
    assert "step" in schema["properties"]
    source = fasta(tmp_path, ">one\nGGCCAAAANT\n")
    native = compute_gc_content(source, window_size=4, step=2)
    registered = operations.invoke("composition.compute_gc_content", input=None,
                                   parameters={"fasta_path": str(source), "window_size": 4, "step": 2})
    assert thaw_json(registered.metrics)["gc_metrics"] == native


def test_runner_honors_step_and_materializes_old_tables_and_method_json(tmp_path):
    source = fasta(tmp_path, ">one\nGGGGGCCCCCAAAAA\n")
    outcome = ANALYSIS_TOOL_RUNNERS["mito_gc"](
        {"input": str(source), "window": 10, "step": 5}, output_dir=tmp_path / "out", log=lambda _: None,
    )
    files = {name: Path(path) for name, path in outcome["output_files"].items()}
    assert files["summary"].name == "gc_summary.tsv" and files["windows"].name == "gc_windows.tsv"
    with files["summary"].open(newline="") as handle:
        summary = {row["metric"]: row["value"] for row in csv.DictReader(handle, delimiter="\t")}
    assert float(summary["gc_content"]) == 0.6667 and int(summary["total_length"]) == 15
    with files["windows"].open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(int(row["start"]), int(row["end"])) for row in rows] == [(1, 10), (6, 15)]
    assert [float(row["gc_content"]) for row in rows] == [1.0, 0.5]
    assert [float(row["gc_skew"]) for row in rows] == [0.0, -1.0]
    analysis = json.loads(files["analysis"].read_text(encoding="utf-8"))
    assert files["analysis"].name == "gc_analysis.json"
    assert analysis["schema"] == "organelleverse.gc-analysis.v1"
    assert analysis["window_size"] == 10 and analysis["step"] == 5
    assert analysis["sequence_count"] == 1
    assert analysis["sequences"] == [{"id": "one", "length": 15, "start": 1, "end": 15}]
    assert analysis["method"] == {
        "algorithm": "organelleverse.composition.gc_core.compute_gc_content",
        "coordinates": "1-based-inclusive", "gc_denominator": "all_sequence_characters", "gc_numerator": "G+C",
        "ambiguous_bases": "included_in_length_not_counted_as_G_or_C",
        "gc_skew_formula": "(G-C)/(G+C)", "gc_skew_zero_denominator": 0.0, "rounding_decimal_places": 4,
        "window_boundary": "full_windows_only_except_one_short_window_for_short_input",
        "window_topology": "linear_no_wrap", "multiple_records": "concatenated_in_file_order_without_separator",
    }


def test_runner_keeps_legacy_multirecord_success_and_identifies_record_boundaries(tmp_path):
    source = fasta(tmp_path, ">one\nGGGGG\n>two\nCCCCC\n")
    outcome = ANALYSIS_TOOL_RUNNERS["mito_gc"](
        {"input": str(source), "window": 10}, output_dir=tmp_path / "out", log=lambda _: None,
    )
    analysis = json.loads(Path(outcome["output_files"]["analysis"]).read_text(encoding="utf-8"))
    assert analysis["step"] == 10 and analysis["sequence_count"] == 2
    assert analysis["sequences"] == [
        {"id": "one", "length": 5, "start": 1, "end": 5}, {"id": "two", "length": 5, "start": 6, "end": 10},
    ]
    assert analysis["windows"] == [{"start": 1, "end": 10, "gc_skew": 0.0, "gc_content": 1.0}]


@pytest.mark.parametrize("step", [0, -1, True, "5", 1.5])
def test_runner_rejects_step_instead_of_coercing_or_silently_ignoring_it(tmp_path, step):
    source = fasta(tmp_path, ">one\nGGGGGCCCCC\n")
    with pytest.raises(OrganelleParameterError):
        ANALYSIS_TOOL_RUNNERS["mito_gc"](
            {"input": str(source), "window": 10, "step": step}, output_dir=tmp_path / "out", log=lambda _: None,
        )


def test_runner_accepts_existing_small_genbank_fixture_without_changing_topology(tmp_path):
    source = Path(__file__).parents[1] / "data" / "mito.gbk"
    outcome = ANALYSIS_TOOL_RUNNERS["mito_gc"](
        {"input": str(source), "window": 100, "step": 50}, output_dir=tmp_path / "out", log=lambda _: None,
    )
    analysis = json.loads(Path(outcome["output_files"]["analysis"]).read_text(encoding="utf-8"))
    assert analysis["total_length"] == 600 and analysis["sequence_count"] == 1
    assert analysis["sequences"][0]["id"] == "MTTEST"
    assert analysis["windows"][0] == {"start": 1, "end": 100, "gc_skew": 0.0, "gc_content": 0.5}
    assert analysis["windows"][-1]["end"] == 600
    assert analysis["method"]["window_topology"] == "linear_no_wrap"
