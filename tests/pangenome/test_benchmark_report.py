import csv

from organelleverse.pangenome.benchmark_report import render_stage_benchmark


def test_report_preserves_measured_zero_and_omits_unknown_or_own_timing(tmp_path):
    stages = [
        {
            "name": "prepare",
            "status": "completed",
            "wall_seconds": 1.0,
            "cpu_seconds": 0.0,
            "peak_memory_bytes": 1048576,
        },
        {
            "name": "graph",
            "status": "completed",
            "wall_seconds": 12.0,
            "cpu_seconds": None,
            "peak_memory_bytes": None,
        },
        {"name": "report", "status": "running", "wall_seconds": None},
    ]
    outputs = render_stage_benchmark({"stages": stages}, tmp_path, formats=("svg", "pdf", "png"))
    with (tmp_path / "stage_benchmark.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["name"] for row in rows] == ["prepare", "graph"]
    assert rows[0]["cpu_seconds"] == "0.0" and rows[1]["cpu_seconds"] == ""
    assert (tmp_path / "stage_cpu_seconds.pdf").read_bytes().startswith(b"%PDF-")
    assert (tmp_path / "stage_cpu_seconds.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = (tmp_path / "stage_cpu_seconds.svg").read_text()
    assert "prepare" in svg and "before report generation" in svg
    assert len(outputs) == 11
