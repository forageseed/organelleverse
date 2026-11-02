"""A closed recruited pool must not hide disconnected WGS sequence.

The native integration fixture isolates input routing: recruitment and taxonomic
classification are controlled, while assembly, read evidence and export are real.
It does not validate organelle classification or physical molecular topology.
"""
import json
import random
from pathlib import Path

import pytest

from organelleverse import _ovasm
from organelleverse.assembly.backends import ovasm as backend
from organelleverse.assembly.environment_specs import OVASM_ENVIRONMENT
from organelleverse.assembly import ovasm_pipeline as ovasm_runner


def test_multichrom_protocol_separates_legacy_cached_runs(tmp_path, monkeypatch):
    binary = tmp_path / "ovasm"
    binary.write_bytes(b"fixed native binary")
    monkeypatch.setattr(backend, "resolve_ovasm", lambda: binary)
    ordinary = backend.NativeOvasmEnvironment()
    current = backend.NativeOvasmEnvironment({"graph_mode": "multichrom"})
    assert current.expected_environment_digest(OVASM_ENVIRONMENT) != ordinary.expected_environment_digest(OVASM_ENVIRONMENT)


@pytest.mark.skipif(_ovasm.resolve_ovasm() is None, reason="requires native OVASM")
def test_closed_seed_pool_does_not_hide_second_wgs_component(tmp_path, monkeypatch):
    sequences = ["".join(random.Random(seed).choices("ACGT", k=6000)) for seed in (142, 243)]
    pools = [
        "".join(f"@c{i}_{start}\n{(sequence * 3)[start:start + 7000]}\n+\n{'I' * 7000}\n"
                for start in range(0, 6000, 250))
        for i, sequence in enumerate(sequences)
    ]
    original = tmp_path / "original.fastq"
    original.write_text("".join(pools))
    seed = tmp_path / "seed.fasta"
    seed.write_text(">seed\n" + sequences[0] + "\n")

    def recruit(inputs, **kwargs):
        assert inputs == [original]
        output = Path(kwargs["out_dir"])
        output.mkdir(parents=True)
        (output / "mitochondrion.fastq").write_text(pools[0])
        return {"targets": [{"name": "mitochondrion"}],
                "outputs": [{"kept_reads": 24, "kept_bases": 168000}]}

    # Random fixture sequence has no biological markers: retain both components
    # only to test the library passed into the native graph and read evidence.
    def keep(graph, organelle, output_dir, files, log, **kwargs):
        from organelleverse.assembly.graph_compare import parse_gfa
        segments, _ = parse_gfa(graph)
        length = sum(map(len, segments.values()))
        return graph, {"kept_length": length, "kept_components": len(segments),
                       "reference_depth": 28, "removed_components": 0,
                       "removed_length": 0, "components": []}

    monkeypatch.setattr(_ovasm, "run_recruit", recruit)
    monkeypatch.setattr(ovasm_runner, "_keep_target_components", keep)
    identity = {"closest": {"organism": "fixture", "family": None, "estimated_identity": 1.0}}
    monkeypatch.setattr(ovasm_runner, "_identify", lambda *a, **kw: (identity, []))
    result = ovasm_runner.run_ovasm(
        {"strategy": "hifi_only", "read_set": "whole_genome",
         "seed_source": "custom", "seed": str(seed),
         "graph_mode": "multichrom", "graph_min_count": 10, "threads": 1},
        organelle="mitochondrion", reads=original, output_dir=tmp_path / "result", log=lambda _: None,
    )
    summary = json.loads(Path(result["output_files"]["summary"]).read_text())
    assert summary["graph_mode"] == "multichrom"
    assert summary["ordinary_preflight"]["circular"] == [True]
    assert summary["graph_input_context"] == "original_wgs"
    assert summary["graph_input_read_count"] == 48
    assert summary["graph_input_bases"] == 336000
    assert summary["assembly_sequences"] == 2
    assert summary["physical_molecule_count"] is None
    assert summary["complete"] is False
    from Bio import SeqIO
    observed = [str(record.seq) for record in SeqIO.parse(result["output_files"]["assembly_fasta"], "fasta")]
    from Bio.Seq import Seq
    assert len(observed) == 2
    for sequence in sequences:
        assert any(len(value) == len(sequence) and
                   (value in sequence * 2 or value in str(Seq(sequence).reverse_complement()) * 2)
                   for value in observed)
