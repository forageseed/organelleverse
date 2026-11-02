"""Independent small-data oracles for FASTQ metrics and selection semantics."""

from __future__ import annotations

import gzip
import json
import os
import shutil
from pathlib import Path

import pytest

from organelleverse import qc
from organelleverse.capabilities.adapters.qc import FIXTURES
from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
    OrganelleParameterError,
)
from organelleverse.quality_control import reads as impl


@pytest.fixture
def reads(tmp_path, monkeypatch):
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    path = tmp_path / "reads.fastq"
    path.write_text(FIXTURES["qc.read_statistics"][0].files[0].content)
    return path


def test_exact_statistics_plain_and_gzip(reads):
    expected = {
        "reads": 3,
        "bases": 12,
        "n50": 6,
        "mean_length": 4.0,
        "median_length": 4.0,
        "mean_quality": 280 / 12,
    }
    assert dict(qc.read_statistics(reads).metrics) == expected
    compressed = reads.with_suffix(".gz")
    with gzip.open(compressed, "wt") as out:
        out.write(reads.read_text())
    assert dict(qc.read_statistics(compressed).metrics) == expected


def test_empty_statistics_and_even_median(reads):
    reads.write_text("")
    result = qc.read_statistics(reads)
    assert result.metrics["reads"] == result.metrics["n50"] == 0
    assert result.metrics["mean_quality"] is None
    reads.write_text("@a\nAA\n+\nII\n@b\nAAAAAA\n+\n555555\n")
    assert qc.read_statistics(reads).metrics["median_length"] == 4


@pytest.mark.parametrize(
    "text",
    ["@a\nAC\n+\nI\n", "@a\nAC\n+\n", "bad\nAC\n+\nII\n", "@a\nAC\n-\nII\n", "@a\nAC\n+\n I\n"],
)
def test_malformed_fastq_fails(reads, text):
    reads.write_text(text)
    with pytest.raises(OrganelleInputError):
        qc.read_statistics(reads)


@pytest.mark.parametrize(
    "target,expected,delta",
    [
        (None, ["short", "long"], None),
        (4, ["short"], 0),
        (5, ["short", "long"], 5),
        (100, ["short", "long"], -90),
    ],
)
def test_filter_and_whole_read_target(reads, target, expected, delta):
    result = qc.filter_long_reads(reads, min_length=3, min_mean_quality=10, target_bases=target)
    output = Path(result.metrics["output_paths"][0])
    assert output.read_text().splitlines()[::4] == ["@" + name for name in expected]
    assert result.metrics["target_delta_bases"] == delta
    assert dict(result.metrics["after"]) == dict(qc.read_statistics(output).metrics)
    assert result.metrics["before"]["reads"] == 3
    assert result.metrics["eligible"]["bases"] == 10


def test_rank_quality_then_length_then_position_and_preserve_input_order(reads):
    reads.write_text(
        "@first\nAA\n+\nII\n@long\nAAAA\n+\nIIII\n@tie\nAAAA\n+\nIIII\n@lower\nAAAAAA\n+\n555555\n"
    )
    result = qc.filter_long_reads(reads, min_length=1, target_bases=5)
    text = Path(result.metrics["output_paths"][0]).read_text()
    assert text.splitlines()[::4] == ["@long", "@tie"]
    assert result.metrics["after"]["bases"] == 8
    result = qc.filter_long_reads(reads, min_length=1, target_bases=4)
    assert Path(result.metrics["output_paths"][0]).read_text().splitlines()[::4] == ["@long"]


def test_no_reads_retained_is_honest_warning(reads):
    result = qc.filter_long_reads(reads, min_length=100)
    assert result.status == "warning"
    assert result.metrics["after"]["reads"] == 0
    assert Path(result.metrics["output_paths"][0]).read_text() == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_length": 0},
        {"min_mean_quality": float("nan")},
        {"min_mean_quality": 94},
        {"target_bases": 0},
        {"target_bases": 2.5},
    ],
)
def test_invalid_parameters(reads, kwargs):
    with pytest.raises(OrganelleParameterError):
        qc.filter_long_reads(reads, **kwargs)


def test_missing_fastp_install_hint(reads, monkeypatch):
    monkeypatch.setenv("ORGANELLEVERSE_FASTP", "/not/installed/fastp")
    with pytest.raises(OrganelleDependencyError, match="micromamba create"):
        qc.filter_short_reads(reads)


def test_failed_fastp_is_not_success(reads, monkeypatch):
    monkeypatch.setattr(impl.shutil, "which", lambda _: "/fastp")

    def fail(*args, **kwargs):
        raise OrganelleExecutionError(code="qc.fastp_failed", message="fastp failed")

    monkeypatch.setattr(impl, "run_external", fail)
    with pytest.raises(OrganelleExecutionError, match="fastp failed"):
        qc.filter_short_reads(reads)


@pytest.mark.integration
@pytest.mark.parametrize("paired", [False, True])
def test_real_fastp_json_and_output(reads, paired):
    binary = shutil.which(os.environ.get("ORGANELLEVERSE_FASTP", "fastp"))
    assert binary, "Real fastp integration gate requires fastp"
    reads.write_text("@a\n" + "ACGT" * 30 + "\n+\n" + "I" * 120 + "\n")
    mate = reads.with_name("mate.fastq")
    mate.write_text(reads.read_text())
    from organelleverse.capabilities.discovery import discover_capabilities
    from organelleverse.operations.python_binding import bind_python_capability

    entry = discover_capabilities().describe("qc.filter_short_reads")
    bound = bind_python_capability(entry.bundle, qc.filter_short_reads, None)
    parameters = {"reads": str(reads)}
    if paired:
        parameters["reads2"] = str(mate)
    result = bound.invoke(None, parameters)
    metrics = result.metrics
    raw = json.loads(Path(metrics["json_report"]).read_text())
    assert metrics["before"]["reads"] == 2 if paired else metrics["before"]["reads"] == 1
    assert metrics["after"]["bases"] == raw["summary"]["after_filtering"]["total_bases"]
    assert metrics["duplication_rate"] == raw["duplication"]["rate"]
    assert len(metrics["output_paths"]) == (2 if paired else 1)


def test_fastp_fractional_threshold_fails_at_boundary(reads):
    with pytest.raises(OrganelleParameterError, match="integer"):
        qc.filter_short_reads(reads, min_mean_quality=20.5)


def test_missing_fastp_report_fails_clearly(reads, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(impl.shutil, "which", lambda _: "/fastp")
    monkeypatch.setattr(impl, "run_external", lambda *a, **kw: SimpleNamespace(stderr=""))
    with pytest.raises(OrganelleExecutionError, match="Invalid fastp outputs"):
        qc.filter_short_reads(reads)


@pytest.mark.integration
def test_real_fastp_known_adapter_and_all_filtered(reads):
    adapter = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
    sequence = "ACTGATCCTAGTCCATGACT" * 3 + adapter
    reads.write_text(f"@adapter\n{sequence}\n+\n{'I' * len(sequence)}\n")
    result = qc.filter_short_reads(reads, adapter_sequence=adapter)
    assert result.metrics["adapter_cutting"]["adapter_trimmed_reads"] == 1
    assert result.metrics["adapter_cutting"]["adapter_trimmed_bases"] == len(adapter)
    assert result.metrics["after"]["bases"] == 60
    result = qc.filter_short_reads(reads, min_length=100, adapter_sequence=adapter)
    assert result.status == "warning"
    assert result.metrics["after"]["reads"] == 0
    assert result.metrics["after"]["mean_quality"] is None


def test_statistics_of_read_data_summarize_each_library(reads, tmp_path):
    import organelleverse as ov

    hifi = ov.read(reads, technology="hifi")
    metrics = dict(qc.read_statistics(hifi).metrics)
    assert set(metrics) == {"long_0"}
    library = dict(metrics["long_0"])
    assert library["technology"] == "pacbio_hifi"
    assert library["quality_state"] == "ccs"
    assert library["reads"] == 3 and library["bases"] == 12 and library["n50"] == 6
    # the data form summarizes exactly what the path form does
    by_path = dict(qc.read_statistics(reads).metrics)
    assert {k: library[k] for k in by_path} == by_path

    mate = tmp_path / "reads2.fastq"
    mate.write_text(reads.read_text())
    paired = dict(qc.read_statistics(ov.read([reads, mate], technology="illumina")).metrics)
    assert set(paired) == {"short_0"}
    short = dict(paired["short_0"])
    assert short["layout"] == "paired_end"
    assert short["reads"] == 6 and short["bases"] == 24  # both mates pooled


def test_statistics_of_a_genome_are_refused(tmp_path):
    import organelleverse as ov
    from organelleverse.core.data import OrganelleData

    wrong = OrganelleData(modality="expression_matrix")
    with pytest.raises(OrganelleInputError):
        qc.read_statistics(wrong)
    assert ov  # the shorthand lives on ov.read; the data type is what qc checks


def test_qc_write_materializes_read_statistics_as_json_and_csv(reads, tmp_path):
    import csv
    import json

    import organelleverse as ov

    stats = qc.read_statistics(ov.read(reads, technology="hifi"))

    as_json = tmp_path / "stats.json"
    written = qc.write(stats, output=as_json)
    assert written.operation_id == "qc.write"
    assert written.summary_text == "Materialized read QC output."
    assert json.loads(as_json.read_text())["operation_id"] == "qc.read_statistics"

    as_csv = tmp_path / "stats.csv"
    qc.write(stats, output=as_csv)
    with as_csv.open(newline="") as handle:
        rows = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    assert rows["long_0.technology"] == "pacbio_hifi"
    assert rows["long_0.reads"] == "3" and rows["long_0.n50"] == "6"

    # the same content again is idempotent; different content is a conflict
    qc.write(stats, output=as_csv)
    as_csv.write_text("metric,value\n")
    with pytest.raises(OrganelleInputError) as conflict:
        qc.write(stats, output=as_csv)
    assert conflict.value.code == "qc.destination_conflict"


def test_qc_write_flattens_a_path_result_and_refuses_other_suffixes(reads, tmp_path):
    stats = qc.read_statistics(reads)
    out = tmp_path / "flat.csv"
    qc.write(stats, output=out)
    assert out.read_text().splitlines()[0] == "metric,value"
    assert "reads,3" in out.read_text().splitlines()

    with pytest.raises(OrganelleInputError) as refused:
        qc.write(stats, output=tmp_path / "stats.md")
    assert refused.value.code == "qc.unsupported_output_suffix"
    assert not (tmp_path / "stats.md").exists()
