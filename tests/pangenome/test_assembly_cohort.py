import json
from pathlib import Path

import pytest

from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.result import ErrorDetail, OrganelleResult
from organelleverse.pangenome import AssemblySample, build_exact_pan, from_assemblies
from organelleverse.pangenome.assembly_cohort import AssemblyCohort
from organelleverse.pangenome.exact_pan import _membership


def sample(tmp_path, name, *, paths=True, failed=False, group="g"):
    directory = tmp_path / name
    directory.mkdir()
    graph = directory / "organelle.gfa"
    graph.write_text(f"H\tVN:Z:1.0\tsn:Z:{name}\nS\t{name}.a\tACGT\nS\t{name}.off\tGATTACA\n" +
                     (f"P\t{name}#1#mt\t{name}.a+\t*\nP\t{name}#2#alt\t{name}.a-\t*\n" if paths else ""))
    result = OrganelleResult(operation_id="assembly.assemble", scope="mitochondrion",
                             status="failed" if failed else "ok",
                             errors=(ErrorDetail(code="assembly.failed", message="test failure"),) if failed else (),
                             artifacts=(ArtifactRef.from_path(graph, kind="assembly_output", format="gfa"),))
    return AssemblySample(sample=name, species="test species", group=group, result=result)


def test_cohort_keeps_failed_graph_only_and_one_sample_per_configuration(tmp_path):
    inputs = [sample(tmp_path, "a"), sample(tmp_path, "b", paths=False), sample(tmp_path, "c", failed=True)]
    cohort = from_assemblies(inputs)
    assert [m.state for m in cohort.members] == ["representative", "graph_only", "failed"]
    assert len(cohort.members) == 3
    assert AssemblyCohort.model_validate_json(cohort.model_dump_json()) == cohort
    assert cohort.members[0].alternative_paths == ("a#2#alt",)
    assert all("absence_not_assessed" in m.limitations for m in cohort.members[:2])
    with pytest.raises(ValueError, match="unique"):
        from_assemblies([inputs[0], inputs[0]])
    graph = inputs[0].result.artifacts[0].resolve()
    graph.write_text(graph.read_text() + "S\tchanged\tTTT\n")
    with pytest.raises(ValueError, match="changed"):
        from_assemblies(inputs)


def test_membership_includes_off_path_nodes_and_rejects_missing_mapping(tmp_path):
    source = sample(tmp_path, "a").result.artifacts[0].resolve()
    pan = tmp_path / "pan.gfa"
    pan.write_text("S\t1\tACGT\nS\t2\tTGTAATC\n")
    mapping = tmp_path / "segments.tsv"
    mapping.write_text("sample\tsegment\tlength\tpieces\na\ta.a\t4\t1+\na\ta.off\t7\t2-\n")
    assert _membership(pan, mapping, {"a": source}, tmp_path / "membership.tsv") == 2
    assert "2\ta\t1" in (tmp_path / "membership.tsv").read_text()
    mapping.write_text("sample\tsegment\tlength\tpieces\na\ta.a\t4\t1+\n")
    with pytest.raises(ValueError, match="incomplete"):
        _membership(pan, mapping, {"a": source}, tmp_path / "membership.tsv")


def test_managed_pan_publishes_reuses_and_detects_tampering(tmp_path, monkeypatch):
    from organelleverse.pangenome import exact_pan
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    cohort = from_assemblies([sample(tmp_path, "a"), sample(tmp_path, "b", paths=False), sample(tmp_path, "c", failed=True)])
    calls = []
    def native(graphs, *, out_gfa, out_json, k):
        calls.append(graphs)
        out_gfa.write_text("S\t1\tACGT\nS\t2\tGATTACA\nP\ta#1#mt\t1+\t*\n")
        out_gfa.with_suffix(".segments.tsv").write_text("sample\tsegment\tlength\tpieces\n" + "".join(f"{s}\t{s}.a\t4\t1+\n{s}\t{s}.off\t7\t2+\n" for s in ("a", "b")))
        out_gfa.with_suffix(".links.tsv").write_text("sample\tfrom\tto\tpangenome_link\n")
        out_json.with_suffix(".inputs.txt").write_text("inputs")
        out_json.write_text(json.dumps({"verified_segments": 4}))
        return {"verified_segments": 4}
    monkeypatch.setattr(exact_pan, "run_pan", native)
    result = build_exact_pan(cohort, group="g", k=15)
    assert result.metrics["included_samples"] == 2
    assert result.metrics["declared_samples"] == 3
    assert build_exact_pan(cohort, group="g", k=15).object_id == result.object_id
    assert len(calls) == 1
    mapping = next(a.resolve() for a in result.artifacts if a.kind == "pangenome_source_segments")
    mapping.write_text("corrupted")
    with pytest.raises(Exception, match="match|digest|bytes"):
        build_exact_pan(cohort, group="g", k=15)


def test_native_exact_pan_preserves_bare_graph_and_alternative_paths(tmp_path, monkeypatch):
    from organelleverse._ovasm import resolve_ovasm
    from organelleverse.pangenome.graph import path_sequences
    if resolve_ovasm() is None:
        pytest.skip("ovasm binary not built")
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    cohort = from_assemblies([sample(tmp_path, "a"), sample(tmp_path, "b", paths=False)])
    result = build_exact_pan(cohort, group="g", k=15)
    graph = next(a.resolve() for a in result.artifacts if a.kind == "pangenome_graph")
    assert path_sequences(graph) == {"a#1#mt": "ACGT", "a#2#alt": "ACGT"}
    assert result.metrics["verified_source_segments"] == 4
    assert result.metrics["graph_only_samples"] == 1
    membership = next(a.resolve() for a in result.artifacts if a.kind == "pangenome_source_membership").read_text()
    assert "\tb\t1" in membership
