from pathlib import Path

import pytest

import organelleverse as ov
from organelleverse.assembly.contracts import LongReadLibrary


def test_ovasm_public_api_uses_canonical_assembly_and_returns_its_result(monkeypatch, tmp_path):
    from organelleverse.assembly import api

    reads = tmp_path / "reads.fastq"
    reads.write_text("@read\nACGT\n+\nIIII\n")
    data = ov.asm.read(long_libraries=(LongReadLibrary(
        technology="pacbio_hifi", quality_state="ccs", reads=reads,
    ),))
    calls = []
    result = object()
    monkeypatch.setattr(api, "assemble", lambda actual, **kwargs: calls.append((actual, kwargs)) or result)

    assert ov.asm.ovasm(
        data, read_set="target_reads", graph_mode="multichrom",
        graph_min_count=80, threads=12, timeout_seconds=600,
    ) is result
    actual, kwargs = calls[0]
    assert actual is data
    assert kwargs["method"] == "ovasm"
    assert kwargs["threads"] == 12
    assert kwargs["timeout_seconds"] == 600
    params = kwargs["backend_parameters"]
    assert params.graph_mode == "multichrom"
    assert params.graph_min_count == 80
    assert params.read_set == "target_reads"
    assert ov.asm is ov.assembly


@pytest.mark.parametrize("tool", ["mito_assemble", "chloro_assemble"])
def test_desktop_preserves_supported_ovasm_hybrid_inputs(monkeypatch, tool, tmp_path):
    from organelleverse.assembly import ovasm_pipeline as ovasm_runner
    from organelleverse.operations.analysis_runners import ANALYSIS_TOOL_RUNNERS

    reads = tmp_path / "long.fastq"
    reads.write_text("@read\nACGT\n+\nIIII\n")
    calls = []
    monkeypatch.setattr(
        ovasm_runner, "run_ovasm",
        lambda params, **kwargs: calls.append((params, kwargs)) or {"output_files": {}},
    )
    ANALYSIS_TOOL_RUNNERS[tool](
        {"strategy": "hybrid", "tool": "ovasm", "input": str(reads),
         "short_reads": "short.fastq", "short_reads_2": "mate.fastq"},
        output_dir=tmp_path / "result", log=lambda _: None,
    )
    params, kwargs = calls[0]
    assert params["strategy"] == "hybrid"
    assert params["short_reads"] == "short.fastq"
    assert params["short_reads_2"] == "mate.fastq"
    assert kwargs["reads"] == reads


def test_public_workbench_reuses_gc_runner_and_writes_real_outputs(tmp_path):
    reads = tmp_path / "genome.fa"
    reads.write_text(">genome\n" + "ACGT" * 100 + "\n")
    result = ov.workbench.run("mito_gc", input=str(reads), window=20, output=tmp_path / "result")
    assert "GC content" in result["summary_text"]
    for path in result["output_files"].values():
        assert Path(path).is_file()
    assert "0.5" in Path(result["output_files"]["summary"]).read_text() or "50" in Path(result["output_files"]["summary"]).read_text()


def test_python_device_selection_does_not_claim_remote_execution(tmp_path):
    with pytest.raises(ValueError, match="selected device"):
        ov.workbench.run("mito_gc", output=tmp_path, compute_target_id="ssh:forageseed-235")
