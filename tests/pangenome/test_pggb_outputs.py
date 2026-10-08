import gzip

import pytest

from organelleverse.pangenome.pggb_outputs import related_artifacts
from tests.pangenome.test_graph_builders import MINIMAL_GFA, _assert_failed_closed, _run_pggb


def deposit(directory):
    (directory / "out.smooth.final.gfa").write_text(MINIMAL_GFA)
    (directory / "out.og").write_bytes((1988148666).to_bytes(4, "big") + b"fixture-format-header")
    (directory / "out.maf").write_text(
        "##maf version=1\na score=0\ns ref 0 4 + 4 ACGT\ns alt 0 4 + 4 ACGA\n\n"
    )
    (directory / "out.paf").write_text("ref\t4\t0\t4\t+\talt\t4\t0\t4\t3\t4\t60\n")
    with gzip.open(directory / "out.vcf.gz", "wt") as handle:
        handle.write(
            "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nref\t4\t.\tT\tA\t.\tPASS\t.\n"
        )


def test_pggb_declares_produced_sidecar_artifacts(tmp_path, monkeypatch):
    result = _run_pggb(tmp_path, monkeypatch, deposit)
    assert result.status == "ok", result.errors
    assert {a.format for a in result.artifacts} == {"gfa", "og", "maf", "paf", "vcf", "txt", "json"}
    assert len([a for a in result.artifacts if a.kind == "pangenome_graph"]) == 1
    assert next(a for a in result.artifacts if a.format == "vcf").media_type == "application/gzip"
    assert len(result.metrics["related_artifact_validation"]) == 4
    assert all(a.resolve().is_file() for a in result.artifacts)


@pytest.mark.parametrize(
    "name,content",
    [
        ("bad.paf", "broken\n"),
        ("bad.vcf", "missing header"),
        ("bad.maf", "missing header"),
        ("bad.og", "not odgi"),
    ],
)
def test_invalid_declared_sidecars_fail_closed(tmp_path, monkeypatch, name, content):
    def invalid(directory):
        deposit(directory)
        (directory / name).write_text(content)

    result = _run_pggb(tmp_path, monkeypatch, invalid)
    _assert_failed_closed(result, code="pangenome.invalid_related_artifact")


def test_empty_paf_is_valid_and_absent_formats_not_fabricated(tmp_path):
    (tmp_path / "empty.paf").write_text("")
    artifacts, validation = related_artifacts(tmp_path)
    assert len(artifacts) == len(validation) == 1
    assert artifacts[0].size_bytes == 0


def test_pggb_rejects_malformed_walk_before_graph_success(tmp_path, monkeypatch):
    def invalid(directory):
        (directory / "out.smooth.final.gfa").write_text(
            MINIMAL_GFA + "W\ta\t0\tchr\t0\t5\t>missing\n"
        )

    result = _run_pggb(tmp_path, monkeypatch, invalid)
    _assert_failed_closed(result, code="pangenome.invalid_gfa")


def test_canonical_builder_publishes_related_files_with_graph(tmp_path, monkeypatch):
    from pathlib import Path

    import organelleverse.pangenome.pangenome as module
    from organelleverse.pangenome._runner import CommandRecord
    from organelleverse.pangenome.service import build_graph, require_graph_result
    from tests.pangenome.test_graph_builders import _two_genomes, _versioned_runner

    def runner(argv, *, stdout_path=None, cwd=None):
        if Path(argv[0]).name == "samtools":
            Path(f"{argv[-1]}.fai").write_text("index\n")
        else:
            deposit(Path(cwd))
        return CommandRecord(tuple(argv), 0, None, "")

    monkeypatch.setattr(module, "run_command", _versioned_runner(runner))
    monkeypatch.setenv("ORGANELLEVERSE_HOME", str(tmp_path / "home"))
    result = build_graph(_two_genomes(tmp_path), method="pggb")
    require_graph_result(result, backend="pggb")
    assert {a.format for a in result.artifacts} >= {"gfa", "og", "vcf", "maf", "paf"}
    for artifact in result.artifacts:
        actual = type(artifact).from_path(
            artifact.resolve(), kind=artifact.kind, format=artifact.format
        )
        assert actual.sha256 == artifact.sha256
        assert ".staging" not in artifact.uri


def test_truncated_gzip_sidecar_fails_closed(tmp_path, monkeypatch):
    def invalid(directory):
        deposit(directory)
        path = directory / "out.vcf.gz"
        path.write_bytes(path.read_bytes()[:-5])

    result = _run_pggb(tmp_path, monkeypatch, invalid)
    _assert_failed_closed(result, code="pangenome.invalid_related_artifact")
