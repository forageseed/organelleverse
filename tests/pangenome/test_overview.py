"""Output validity and fail-closed boundaries for actual ODGI graph overviews."""

import json
import os
from pathlib import Path

import pytest

from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.pangenome import overview
from organelleverse.pangenome._runner import CommandRecord

GFA = "S\t1\tACGT\nS\t2\tAA\nS\t3\tTT\nL\t1\t+\t2\t+\t0M\nL\t1\t+\t3\t+\t0M\nP\ta#1#1\t1+,2+\t0M\nP\tb#1#1\t1+,3+\t0M\n"
PNG = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\0\x01\0\0\0\x01"
SVG = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 L1 1"/></svg>'


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / "source.gfa"
    path.write_text(GFA)
    return path


def backend(monkeypatch, *, failed=None, invalid_image=None, decoded=GFA):
    monkeypatch.setattr(overview.shutil, "which", lambda name: "/bin/odgi")

    def run(argv, *, cwd, stdout_path=None):
        command = argv[1]
        if command == failed:
            return CommandRecord(tuple(argv), 2, None, "reproduced backend error")
        if command == "version":
            Path(stdout_path).write_text("odgi-test-version")
        elif command in {"build", "sort"}:
            Path(argv[argv.index("-o") + 1]).write_bytes(bytes.fromhex("7680bdba") + b"test-og")
        elif command == "view":
            Path(stdout_path).write_text(decoded)
        elif command == "viz":
            Path(argv[argv.index("-o") + 1]).write_bytes(
                b"not png" if invalid_image == "png" else PNG
            )
        elif command == "layout":
            Path(argv[argv.index("-o") + 1]).write_bytes(b"test-lay")
            Path(argv[argv.index("-T") + 1]).write_text("idx\tX\tY\tcomponent\n0\t0\t0\t0\n")
        elif command == "draw":
            Path(argv[argv.index("-p") + 1]).write_bytes(PNG)
            Path(argv[argv.index("-s") + 1]).write_text(
                "not svg" if invalid_image == "svg" else SVG
            )
        return CommandRecord(
            tuple(argv), 0, str(stdout_path) if stdout_path else None, "", 0.01, 0.005, 4096
        )

    monkeypatch.setattr(overview, "run_command", run)


def test_retains_real_encoding_outputs_and_command_source_evidence(graph, tmp_path, monkeypatch):
    backend(monkeypatch)
    files = overview.render_odgi_overview(graph, tmp_path / "overview")
    data = json.loads(files[-1].read_text())
    assert len(files) == 9
    assert data["status"] == "succeeded"
    assert data["source"]["uri"] == str(graph)
    assert data["source"]["format"] == "gfa"
    assert data["software"]["version"] == "odgi-test-version"
    assert data["validation"] == {
        "source_roundtrip_equivalent": True,
        "sorted_ids_compact": True,
        "sorted_path_sequences_equivalent": True,
        "image_encodings_valid": True,
    }
    sort = next(c for c in data["commands"] if c["argv"][1] == "sort")
    assert "-b" in sort["argv"] and "-O" in sort["argv"]
    assert all(c["wall_seconds"] >= 0 for c in data["commands"])
    assert all(c["resource_usage"]["cpu_seconds"] == 0.005 for c in data["commands"])
    assert all(c["resource_usage"]["peak_memory_bytes"] == 4096 for c in data["commands"])
    assert all(d["reuses_source"] for d in data["decoded_graphs"].values())
    assert not list((tmp_path / "overview").glob("*.gfa"))
    assert {a["media_type"] for a in data["artifacts"]} >= {"image/png", "image/svg+xml"}
    assert data["graph_statistics"] == {
        "nodes": 3,
        "edges": 2,
        "paths": 2,
        "components": 1,
        "total_bp": 8,
    }
    assert (tmp_path / "overview/graph.og").read_bytes() != graph.read_bytes()


@pytest.mark.parametrize("invalid_image", ["png", "svg"])
def test_rejects_false_image_encodings_and_retains_failure_evidence(
    graph, tmp_path, monkeypatch, invalid_image
):
    backend(monkeypatch, invalid_image=invalid_image)
    with pytest.raises(OrganelleExecutionError, match="not a valid"):
        overview.render_odgi_overview(graph, tmp_path / "overview")
    data = json.loads((tmp_path / "overview/overview.evidence.json").read_text())
    assert data["status"] == "failed" and "artifacts" not in data


def test_failed_layout_preserves_command_failure_without_draw(graph, tmp_path, monkeypatch):
    backend(monkeypatch, failed="layout")
    with pytest.raises(OrganelleExecutionError, match="layout failed"):
        overview.render_odgi_overview(graph, tmp_path / "overview")
    data = json.loads((tmp_path / "overview/overview.evidence.json").read_text())
    assert data["commands"][-1]["returncode"] == 2
    assert data["commands"][-1]["stderr"] == "reproduced backend error"
    assert not (tmp_path / "overview/graph_2d.png").exists()


@pytest.mark.parametrize(
    "source", [GFA.replace("0M", "1M"), "S\t1\tACGT\nW\ts\t1\tchr\t0\t4\t>1\n", "S\t1\tACGT\n"]
)
def test_rejects_overlaps_walks_and_pathless_input_before_spawning(
    graph, tmp_path, monkeypatch, source
):
    graph.write_text(source)

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported input must not reach an external command")

    monkeypatch.setattr(overview, "run_command", unexpected)
    with pytest.raises(OrganelleInputError):
        overview.render_odgi_overview(graph, tmp_path / "overview")


def test_rejects_changed_roundtrip(graph, tmp_path, monkeypatch):
    backend(monkeypatch, decoded=GFA.replace("ACGT", "ACGA"))
    with pytest.raises(OrganelleExecutionError, match="changed source"):
        overview.render_odgi_overview(graph, tmp_path / "overview")


def test_missing_dependency_and_stale_outputs_are_explicit(graph, tmp_path, monkeypatch):
    monkeypatch.setattr(overview.shutil, "which", lambda name: None)
    with pytest.raises(OrganelleDependencyError):
        overview.render_odgi_overview(graph, tmp_path / "overview")
    backend(monkeypatch)
    out = tmp_path / "overview"
    out.mkdir()
    (out / "graph_2d.png").write_bytes(PNG)
    with pytest.raises(OrganelleInputError, match="must be empty"):
        overview.render_odgi_overview(graph, out)


@pytest.mark.skipif(
    os.environ.get("ORG_VERSE_TEST_REAL_ODGI") != "1", reason="Opt-in installed ODGI validation"
)
def test_installed_odgi_renders_all_nodes_with_native_layout(graph, tmp_path):
    files = overview.render_odgi_overview(graph, tmp_path / "real", threads=2)
    data = json.loads(files[-1].read_text())
    assert data["status"] == "succeeded"
    assert len((tmp_path / "real/graph_2d.tsv").read_text().splitlines()) == 1 + 2 * 3
    assert data["graph_statistics"]["nodes"] == 3


NAMED_GFA = (
    "S\tS1.a\tACGT\nS\tS1.b\tAA\nS\tS1.c\tTT\n"
    "L\tS1.a\t+\tS1.b\t+\t0M\nL\tS1.a\t+\tS1.c\t+\t0M\n"
    "P\ta#1#1\tS1.a+,S1.b+\t0M\nP\tb#1#1\tS1.a+,S1.c+\t0M\n"
)


def test_named_segments_reach_odgi_renumbered_with_a_name_table(tmp_path, monkeypatch):
    # ODGI 0.9 rejects non-numeric segment names; ODGI's roundtrip is the renumbered graph.
    source = tmp_path / "named.gfa"
    source.write_text(NAMED_GFA)
    seen = []
    backend(monkeypatch, decoded=GFA)
    run = overview.run_command

    def spy(argv, *, cwd, stdout_path=None):
        seen.append(list(argv))
        return run(argv, cwd=cwd, stdout_path=stdout_path)

    monkeypatch.setattr(overview, "run_command", spy)
    files = overview.render_odgi_overview(source, tmp_path / "out", threads=1)
    data = json.loads(files[-1].read_text())
    assert data["status"] == "succeeded"
    assert data["node_names"] == {"renumbered": True, "table": "node_names.tsv"}
    build = next(a for a in seen if a[1] == "build")
    assert build[build.index("-g") + 1].endswith("graph.numeric.gfa")
    assert (tmp_path / "out/node_names.tsv").read_text().splitlines()[1:] == [
        "1\tS1.a",
        "2\tS1.b",
        "3\tS1.c",
    ]


@pytest.mark.skipif(
    os.environ.get("ORG_VERSE_TEST_REAL_ODGI") != "1", reason="Opt-in installed ODGI validation"
)
def test_installed_odgi_renders_an_ov_gfa_with_named_segments(tmp_path):
    source = tmp_path / "named.gfa"
    source.write_text(NAMED_GFA)
    files = overview.render_odgi_overview(source, tmp_path / "real", threads=2)
    data = json.loads(files[-1].read_text())
    assert data["status"] == "succeeded"
    assert data["graph_statistics"]["nodes"] == 3


def test_circular_paths_survive_the_overview_as_evidence(tmp_path, monkeypatch):
    # ODGI drops TP:Z:circular; drawing does not depend on it, so the overview goes on and
    # records which paths were circular.
    source = tmp_path / "circular.gfa"
    source.write_text(GFA.replace("P\ta#1#1\t1+,2+\t0M\n", "P\ta#1#1\t1+,2+\t0M\tTP:Z:circular\n"))
    backend(monkeypatch, decoded=GFA)
    files = overview.render_odgi_overview(source, tmp_path / "out", threads=1)
    data = json.loads(files[-1].read_text())
    assert data["status"] == "succeeded"
    assert data["circular_paths"] == ["a#1#1"]
    assert data["node_names"]["renumbered"] is False
