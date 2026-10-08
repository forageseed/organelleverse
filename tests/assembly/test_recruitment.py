from __future__ import annotations

import random
from pathlib import Path

import pytest

from organelleverse._ovasm import resolve_ovasm, run_recruit
from organelleverse.assembly.recruitment import _merge_unique, preset_for, recruit_long_reads


def _random_dna(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _write_fastq(path: Path, reads: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"@{name}\n{seq}\n+\n{'I' * len(seq)}\n" for name, seq in reads))


def test_preset_follows_read_accuracy() -> None:
    assert preset_for("pacbio_hifi", "ccs") == "hifi"
    assert preset_for("ont", "duplex") == "hifi"
    assert preset_for("ont", "raw") == "ont"
    assert preset_for("pacbio_clr", "raw") == "ont"


def test_env_can_disable_ovasm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORG_VERSE_OVASM_BIN", "off")
    assert resolve_ovasm() is None


def test_merge_keeps_one_copy_of_reads_shared_between_targets(tmp_path: Path) -> None:
    a, b, merged = tmp_path / "a.fastq", tmp_path / "b.fastq", tmp_path / "m.fastq"
    _write_fastq(a, [("r1", "ACGT"), ("shared", "GGCC")])
    _write_fastq(b, [("shared", "GGCC"), ("r2", "TTAA")])
    assert _merge_unique([a, b], merged) == 3
    assert [line for line in merged.read_text().splitlines() if line.startswith("@")] == [
        "@r1",
        "@shared",
        "@r2",
    ]


@pytest.mark.skipif(resolve_ovasm() is None, reason="ovasm binary not built")
def test_recruit_long_reads_end_to_end(tmp_path: Path) -> None:
    rng = random.Random(7)
    mito, nuclear = _random_dna(6000, rng), _random_dna(20000, rng)
    plastid = _random_dna(4000, rng)
    plastid = plastid[:1000] + mito[2000:2600] + plastid[1600:]  # MTPT-like identical segment
    (tmp_path / "mt.fa").write_text(f">mt\n{mito}\n")
    (tmp_path / "cp.fa").write_text(f">cp\n{plastid}\n")

    reads: list[tuple[str, str]] = []
    for i in range(40):
        s = rng.randrange(0, len(mito) - 800)
        reads.append((f"mt{i}", mito[s : s + 800]))
        s = rng.randrange(0, len(plastid) - 800)
        if 1000 - 800 < s < 1600:  # avoid ambiguous plastid reads overlapping the shared segment
            s = 1700
        reads.append((f"cp{i}", plastid[s : s + 800]))
        s = rng.randrange(0, len(nuclear) - 800)
        reads.append((f"nu{i}", nuclear[s : s + 800]))
    reads.append(("shared", mito[2050:2550]))  # identical in both genomes
    _write_fastq(tmp_path / "reads.fastq", reads)

    data, report = recruit_long_reads(
        tmp_path / "reads.fastq",
        technology="pacbio_hifi",
        quality_state="ccs",
        seeds={"mitochondrion": [tmp_path / "mt.fa"], "plastid": [tmp_path / "cp.fa"]},
        out_dir=tmp_path / "recruit",
        target_depth=None,
    )
    names = [
        line[1:]
        for line in (tmp_path / "recruit" / "recruited.fastq").read_text().splitlines()
        if line.startswith("@")
    ]
    assert not any(n.startswith("nu") for n in names)
    assert sum(n.startswith("mt") for n in names) == 40
    assert sum(n.startswith("cp") for n in names) == 40
    assert names.count("shared") == 1  # tied between targets, kept once
    assert report["merged_reads"] == 81
    assert report["iterations"][0]["multi_target_reads"] >= 1
    assert data.modality


@pytest.mark.skipif(resolve_ovasm() is None, reason="ovasm binary not built")
def test_recruited_library_path_is_absolute_for_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = random.Random(3)
    mito = _random_dna(3000, rng)
    (tmp_path / "mt.fa").write_text(f">mt\n{mito}\n")
    _write_fastq(
        tmp_path / "reads.fastq", [(f"mt{i}", mito[i * 100 : i * 100 + 600]) for i in range(20)]
    )
    monkeypatch.chdir(tmp_path)
    _, report = recruit_long_reads(
        Path("reads.fastq"),
        technology="pacbio_hifi",
        quality_state="ccs",
        seeds={"mitochondrion": [Path("mt.fa")]},
        out_dir=Path("relative_out"),
        target_depth=None,
    )
    assert Path(report["merged_output"]).is_absolute()


@pytest.mark.skipif(resolve_ovasm() is None, reason="ovasm binary not built")
def test_short_reads_are_recruited_by_long_reads_of_the_sample(tmp_path: Path) -> None:
    # No reference: the sample's own long reads (20x, one error each) are the seed.
    rng = random.Random(11)
    mito, nuclear = _random_dna(6000, rng), _random_dna(20000, rng)
    circ = mito + mito[:1500]
    long_reads = []
    for i in range(0, len(mito), 300):
        r = list(circ[i : i + 1500])
        p = rng.randrange(len(r))
        r[p] = "A" if r[p] != "A" else "C"
        long_reads.append((f"long{i}", "".join(r)))
    _write_fastq(tmp_path / "long.fastq", long_reads)
    short = [(f"mt{i}", circ[i * 40 : i * 40 + 150]) for i in range(150)]
    short += [(f"nu{i}", nuclear[i * 100 : i * 100 + 150]) for i in range(150)]
    _write_fastq(tmp_path / "short.fastq", short)

    report = run_recruit(
        [tmp_path / "short.fastq"],
        seeds={},
        seed_reads={"mitochondrion": [tmp_path / "long.fastq"]},
        out_dir=tmp_path / "recruit",
        preset="sr",
    )
    names = [
        line[1:]
        for line in (tmp_path / "recruit" / "mitochondrion.fastq").read_text().splitlines()[::4]
    ]
    assert sorted(names) == sorted(n for n, _ in short if n.startswith("mt"))
    (target,) = report["targets"]
    assert abs(target["seed_length"] - len(mito)) < 50, "genome size from solid k-mers"
