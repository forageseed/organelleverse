from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from organelleverse import _ovasm
from organelleverse._ovasm import resolve_ovasm, run_pipeline


def test_run_pipeline_passes_every_option_to_the_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> None:
        seen.append(argv)
        (tmp_path / "out").mkdir(exist_ok=True)
        (tmp_path / "out" / "summary.json").write_text(json.dumps({"k": 701}))

    monkeypatch.setattr(_ovasm, "_require", lambda: "ovasm")
    monkeypatch.setattr(_ovasm, "run_external", fake_run)
    summary = run_pipeline(
        ["a.fq.gz", "b.fq.gz"],
        organelle="mitochondrion",
        out_dir=tmp_path / "out",
        seeds={"mitochondrion": ["mt.fa"], "plastid": ["pt.fa"]},
        recruit="both",
        recruit_depth=150,
        mode="standard",
        replicates=4,
        max_replicates=5,
        replicate_depth=60,
        salt=7,
        sample="desktop",
        threads=8,
    )
    assert summary == {"k": 701}
    argv = seen[0]
    assert argv[:2] == ["ovasm", "run"]

    def value(flag: str) -> list[str]:
        return [argv[i + 1] for i, a in enumerate(argv) if a == flag]

    assert value("--reads") == ["a.fq.gz", "b.fq.gz"]
    assert value("--seeds") == ["mitochondrion=mt.fa", "plastid=pt.fa"]
    assert value("--organelle") == ["mitochondrion"]
    assert value("--read-type") == ["hifi"]
    assert value("--read-set") == ["whole-genome"]
    assert value("--recruit") == ["both"]
    assert value("--recruit-depth") == ["150"]
    assert value("--mode") == ["standard"]
    assert value("--replicates") == ["4"]
    assert value("--max-replicates") == ["5"]
    assert value("--replicate-depth") == ["60"]
    assert value("--salt") == ["7"]
    assert value("--sample") == ["desktop"]
    assert value("--threads") == ["8"]
    assert "--single-seed" not in argv


def test_run_pipeline_needs_reads_and_seed_files() -> None:
    with pytest.raises(ValueError, match="at least one read file"):
        run_pipeline([], organelle="plastid", out_dir="out")
    with pytest.raises(ValueError, match="has no files"):
        run_pipeline(["a.fq"], organelle="plastid", out_dir="out", seeds={"plastid": []})


def _revcomp(seq: str) -> str:
    return seq[::-1].translate(str.maketrans("ACGT", "TGCA"))


@pytest.mark.skipif(resolve_ovasm() is None, reason="ovasm binary not built")
def test_run_pipeline_assembles_target_reads_end_to_end(tmp_path: Path) -> None:
    binary = resolve_ovasm()
    assert binary is not None
    import subprocess

    if "run" not in subprocess.run([binary, "--help"], capture_output=True, text=True).stdout:
        pytest.skip("this ovasm binary predates `ovasm run`")
    rng = random.Random(11)
    genome = "".join(rng.choice("ACGT") for _ in range(20_000))
    circle = genome + genome[:3000]
    reads = tmp_path / "target.fastq"
    with reads.open("w") as out:
        for i, start in enumerate(range(0, len(genome), 100)):
            seq = circle[start : start + 3000]
            if i % 2:
                seq = _revcomp(seq)
            out.write(f"@r{i}\n{seq}\n+\n{'I' * len(seq)}\n")
    summary = run_pipeline(
        [reads], organelle="plastid", out_dir=tmp_path / "out", read_set="target", sample="desktop"
    )
    assert summary["backend"] == "ovasm"
    assert summary["organelle"] == "plastid"
    assert summary["read_set"] == "target_reads"
    assert summary["recruited_reads"] is None
    assert summary["molecules"] == 1
    assert summary["lengths"] == [len(genome)]
    assert summary["circular"] == [True]
    assert summary["k"] == summary["k_tried"][-1]["k"]
    assert summary["mode"] == "fast"
    for name in (
        "assembly.gfa",
        "evidence.json",
        "linearization.json",
        "molecules.fasta",
        "organelle.gfa",
        "organelle.json",
        "summary.json",
    ):
        assert (tmp_path / "out" / name).is_file(), name
