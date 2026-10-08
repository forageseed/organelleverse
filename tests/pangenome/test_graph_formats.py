"""Conversion contracts and opt-in validation against ODGI/VG installations."""

import json
import os
from pathlib import Path

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.errors import (
    OrganelleDependencyError,
    OrganelleExecutionError,
    OrganelleInputError,
)
from organelleverse.pangenome import graph_formats as formats
from organelleverse.pangenome._runner import CommandRecord
from organelleverse.pangenome.graph import path_sequences

GFA = "H\tVN:Z:1.0\nS\t1\tACGT\nS\t2\tAA\nL\t1\t+\t2\t+\t0M\nP\ts#1#1\t1+,2+\t0M\n"


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.gfa"
    path.write_text(GFA)
    return ArtifactRef.from_path(path, kind="pangenome_graph", format="gfa")


def fake_backend(monkeypatch, decoded=GFA, failed=False, text_binary=False):
    monkeypatch.setattr(formats.shutil, "which", lambda name: f"/bin/{name}")

    def run(argv, *, cwd, stdout_path=None):
        if failed:
            return CommandRecord(tuple(argv), 2, str(stdout_path), "invalid graph")
        if argv[1] == "version":
            Path(stdout_path).write_text("test-version")
        elif argv[1] == "build":
            Path(argv[argv.index("-o") + 1]).write_bytes(
                GFA.encode() if text_binary else bytes.fromhex("7680bdba") + b"unit-test-graph"
            )
        elif argv[1] == "view" or "-f" in argv:
            Path(stdout_path).write_text(decoded)
        else:
            Path(stdout_path).write_bytes(b"\x02\x02VG" + b"unit-test-graph")
        return CommandRecord(tuple(argv), 0, str(stdout_path), "")

    monkeypatch.setattr(formats, "run_command", run)


def test_converter_publishes_only_converted_graph_and_evidence(source, monkeypatch):
    fake_backend(monkeypatch)
    result = formats.convert_graph(source, target_format="og")
    assert result.status == "ok"
    assert {a.format for a in result.artifacts} == {"og", "json"}
    evidence = json.loads(
        next(Path(a.uri).read_text() for a in result.artifacts if a.format == "json")
    )
    assert evidence["validation"]["source_equivalence"] is True
    assert evidence["path_mapping"] == {"s#1#1": "s#1#1"}
    assert source.sha256 in result.provenance.input_artifact_hashes


def test_vg_walk_encoding_preserves_pansn_semantics(source, monkeypatch):
    fake_backend(monkeypatch, decoded=GFA.replace("P\ts#1#1\t1+,2+\t0M", "W\ts\t1\t1\t0\t6\t>1>2"))
    result = formats.convert_graph(source, target_format="vg")
    evidence = json.loads(
        next(Path(a.uri).read_text() for a in result.artifacts if a.format == "json")
    )
    assert evidence["encoding"] == "VG Protobuf"
    assert evidence["path_mapping"] == {"s#1#1": "s#1#1:0-6"}


def test_conversion_must_not_drop_paths(source, monkeypatch):
    fake_backend(monkeypatch, decoded="S\t1\tACGT\nS\t2\tAA\nL\t1\t+\t2\t+\t0M\n")
    with pytest.raises(OrganelleExecutionError, match="changed"):
        formats.convert_graph(source, target_format="og")


def test_text_masquerading_as_binary_is_rejected(source, monkeypatch):
    fake_backend(monkeypatch, text_binary=True)
    with pytest.raises(OrganelleInputError, match="encoding"):
        formats.convert_graph(source, target_format="og")


def test_missing_backend_and_failure_are_explicit(source, monkeypatch):
    monkeypatch.setattr(formats.shutil, "which", lambda name: None)
    with pytest.raises(OrganelleDependencyError):
        formats.convert_graph(source, target_format="og")
    fake_backend(monkeypatch, failed=True)
    with pytest.raises(OrganelleExecutionError, match="failed"):
        formats.convert_graph(source, target_format="og")


def test_changed_input_and_unsupported_route_are_rejected(source):
    with pytest.raises(OrganelleInputError, match="Supported conversions"):
        formats.convert_graph(source, target_format="gfa")
    source.resolve().write_text("S\tx\tA\n")
    with pytest.raises(OrganelleInputError, match="changed"):
        formats.convert_graph(source, target_format="og")


@pytest.mark.skipif(
    os.environ.get("ORGANELLEVERSE_TEST_GRAPH_CONVERTERS") != "1",
    reason="explicit real backend opt-in",
)
def test_real_odgi_and_vg_roundtrip(source):
    og = formats.convert_graph(source, target_format="og")
    og_graph = next(a for a in og.artifacts if a.kind == "pangenome_graph")
    decoded = formats.convert_graph(og_graph, target_format="gfa")
    gfa_graph = next(a for a in decoded.artifacts if a.kind == "pangenome_graph")
    assert path_sequences(gfa_graph.uri) == path_sequences(source.uri)
    vg = formats.convert_graph(source, target_format="vg")
    assert vg.status == "ok"


def test_circular_path_flag_cannot_silently_disappear(tmp_path, monkeypatch):
    source = tmp_path / "circular.gfa"
    source.write_text(GFA.rstrip() + "\tTP:Z:circular\n")
    fake_backend(monkeypatch)
    artifact = ArtifactRef.from_path(source, kind="pangenome_graph", format="gfa")
    with pytest.raises(OrganelleExecutionError, match="changed"):
        formats.convert_graph(artifact, target_format="og")
