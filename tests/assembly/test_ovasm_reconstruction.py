"""Public graph-conditioned reconstruction boundary and real small native chain."""
import json
import os
from pathlib import Path
import random

import pytest

from organelleverse import _ovasm


def test_public_facade_passes_both_graphs_and_all_mates(monkeypatch, tmp_path):
    import organelleverse as ov
    calls = []
    report = {"physical_molecule_count": None, "physical_topology_identified": False}
    monkeypatch.setattr(_ovasm, "run_reconstruction", lambda reads, **kw: calls.append((reads, kw)) or report)
    mates = (tmp_path / "R1.fq.gz", tmp_path / "R2.fq.gz")
    assert ov.asm.reconstruct(reads=mates, mitochondrial_reference=tmp_path / "mt.gfa",
                              plastid_reference=tmp_path / "pt.gfa", sample="target",
                              output=tmp_path / "out", target_depth=150, threads=8) is report
    assert calls[0][0] == mates
    assert calls[0][1]["references"] == {"mitochondrion": tmp_path / "mt.gfa", "plastid": tmp_path / "pt.gfa"}


@pytest.mark.parametrize("changes, match", [
    ({"reads": ()}, "WGS read"),
    ({"references": {"mitochondrion": "mt.gfa"}}, "plastid reference"),
    ({"sample": "bad#sample"}, "PanSN"),
    ({"threads": 0}, "threads"),
    ({"target_depth": float("nan")}, "target depth"),
])
def test_input_errors_precede_missing_binary(monkeypatch, tmp_path, changes, match):
    monkeypatch.setenv("ORG_VERSE_OVASM_BIN", "none")
    args = dict(reads=("reads.fq",), references={"mitochondrion": "mt.gfa", "plastid": "pt.gfa"},
                out_dir=tmp_path, sample="target")
    args.update(changes)
    with pytest.raises(ValueError, match=match):
        _ovasm.run_reconstruction(**args)


def test_native_error_is_not_a_reconstructed_result(monkeypatch, tmp_path):
    from organelleverse.core.errors import OrganelleExecutionError
    monkeypatch.setattr(_ovasm, "_require", lambda: "ovasm")
    def fail(*args, **kwargs):
        raise OrganelleExecutionError(code=kwargs["code"], message="real native failure")
    monkeypatch.setattr(_ovasm, "run_external", fail)
    with pytest.raises(OrganelleExecutionError, match="real native failure"):
        _ovasm.run_reconstruction(("reads.fq",), references={"mitochondrion": "mt.gfa", "plastid": "pt.gfa"},
                                 sample="target", out_dir=tmp_path)


def test_real_native_both_organelle_chain_uses_second_source_allele(tmp_path, monkeypatch):
    import organelleverse as ov

    binary = os.environ.get("OVASM_RECONSTRUCT_TEST_BIN")
    if not binary:
        pytest.skip("explicit reconstruction binary required for this small native test")
    monkeypatch.setenv("ORG_VERSE_OVASM_BIN", binary)
    reads = tmp_path / "reads.fasta"
    targets = {}
    refs = {}
    records = []
    for j, organelle in enumerate(("mitochondrion", "plastid")):
        rng = random.Random(70 + j)
        left = "".join(rng.choices("ACGT", k=2000))
        right = "".join(rng.choices("ACGT", k=2000))
        target = left + "C" + right
        targets[organelle] = target
        gfa = tmp_path / f"{organelle}.gfa"
        gfa.write_text(f"H\tVN:Z:1.0\nS\tl\t{left}\nS\ta\tA\nS\tc\tC\nS\tr\t{right}\n"
                       "L\tl\t+\ta\t+\t0M\nL\ta\t+\tr\t+\t0M\n"
                       "L\tl\t+\tc\t+\t0M\nL\tc\t+\tr\t+\t0M\n"
                       f"P\tfirst#1#{organelle}\tl+,a+,r+\t*\n"
                       f"P\tsecond#1#{organelle}\tl+,c+,r+\t*\n")
        refs[organelle] = gfa
        # Dense k=127 assembly needs overlapping 127-mers. Adjacent 150-bp
        # records spaced by 50 bases only overlap 100 bp and cannot share one.
        starts = [*range(0, len(target) - 149, 10), len(target) - 150]
        for rep in range(12):
            records += [f">{organelle}.{rep}.{a}\n{target[a:a+150]}\n" for a in starts]
    reads.write_text("".join(records))
    out = tmp_path / "result"
    report = ov.asm.reconstruct(
        reads=(reads,), mitochondrial_reference=refs["mitochondrion"],
        plastid_reference=refs["plastid"], sample="target", output=out, threads=2,
        timeout_seconds=120,
    )
    assert report["physical_molecule_count"] is None
    assert report["strict_score"] is None
    assert report["recruitment"]["iterations"][0]["reads_scanned"] == len(records)
    for organelle in targets:
        data = report["organelles"][organelle]
        assert data["reference"]["source_paths"] == 2
        assert data["projection"]["certain_exact_reference_bases"] == 4001
        assert data["projection"]["reference_supported_links"] == 2
        assert data["individual"]["complete_assembly"] is False
        assert "error" not in data["de_novo"]
        graph = Path(data["individual"]["individual_gfa"]).read_text()
        nodes = {f[1]: f[2] for line in graph.splitlines() if (f := line.split("\t"))[0] == "S"}
        sequences = [line for line in Path(data["individual"]["individual_fasta"]).read_text().splitlines() if not line.startswith(">")]
        assert sum(map(len, sequences)) == sum(map(len, nodes.values()))
        assert targets[organelle] in sequences or targets[organelle].translate(str.maketrans("ACGT", "TGCA"))[::-1] in sequences
    reused = ov.asm.reconstruct(
        mitochondrial_reference=refs["mitochondrion"], plastid_reference=refs["plastid"],
        sample="reused", output=tmp_path / "target-stage", threads=2,
        target_reads={o: (out / "recruit" / f"{o}.fastq",) for o in targets},
        timeout_seconds=120,
    )
    assert reused["input_scope"] == "explicit_target_reads"
    assert reused["recruitment"]["status"] == "not_run_explicit_target_inputs"
    assert reused["workflow_completed"] is True
    for o in targets:
        assert reused["organelles"][o]["projection"]["certain_exact_reference_bases"] == 4001
