from __future__ import annotations

import random
from pathlib import Path

import pytest

from organelleverse._ovasm import resolve_ovasm
from organelleverse.assembly.graph_assessment import assess_assembly_graph
from organelleverse.assembly.unified_graph import export_representatives, unify_assembly_graph
from organelleverse.core.errors import OrganelleInputError
from organelleverse.pangenome._gfa import load_gfa
from organelleverse.pangenome.graph import path_sequences
from organelleverse.pangenome.graph_formats import (
    _fingerprint,  # pyright: ignore[reportPrivateUsage]
)

needs_ovasm = pytest.mark.skipif(resolve_ovasm() is None, reason="ovasm binary not built")


def _dna(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _revcomp(s: str) -> str:
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


def _write_ov(path: Path, sample: str, with_path: bool = True) -> Path:
    text = f"H\tVN:Z:1.0\nS\t{sample}.A\tACGT\nS\t{sample}.B\tGGCC\nL\t{sample}.A\t+\t{sample}.B\t+\t0M\n"
    if with_path:
        text += f"P\t{sample}#1#mt\t{sample}.A+,{sample}.B+\t0M\n"
    path.write_text(text)
    return path


def test_export_writes_haplotype_one_per_sample(tmp_path: Path) -> None:
    graphs = [_write_ov(tmp_path / f"{s}.gfa", s) for s in ("s1", "s2")]
    out = export_representatives(graphs, tmp_path / "pan")
    assert [p.name for p in out] == ["s1.fa", "s2.fa"]
    assert out[0].read_text() == ">s1\nACGTGGCC\n"


def test_export_rejects_duplicate_samples_and_graphs_without_a_path(tmp_path: Path) -> None:
    a = _write_ov(tmp_path / "a.gfa", "s1")
    b = _write_ov(tmp_path / "b.gfa", "s1")
    with pytest.raises(OrganelleInputError) as dup:
        export_representatives([a, b], tmp_path / "pan")
    assert dup.value.code == "assembly.duplicate_sample"
    bare = _write_ov(tmp_path / "c.gfa", "s3", with_path=False)
    with pytest.raises(ValueError, match="no paths"):
        export_representatives([bare], tmp_path / "pan2")


@needs_ovasm
def test_overlapping_graph_becomes_a_pangenome_ready_graph(tmp_path: Path) -> None:
    rng = random.Random(8)
    a, b, c, d, r = (_dna(3000, rng) for _ in range(5))
    # Oatk-like overlaps: A and C end with R's first 100 bp (100M), B ends with C's first 50 bp.
    segs = {"A": a + r[:100], "B": b + c[:50], "C": c + r[:100], "D": d, "R": r}
    (tmp_path / "g.gfa").write_text(
        "H\tVN:Z:1.0\n"
        + "".join(f"S\t{n}\t{s}\n" for n, s in segs.items())
        + "L\tA\t+\tR\t+\t100M\nL\tC\t+\tR\t+\t100M\nL\tR\t+\tB\t+\t0M\nL\tR\t+\tD\t+\t0M\n"
        + "L\tB\t+\tC\t+\t50M\n"
    )
    reads: list[tuple[str, str]] = []
    for i in range(20):
        reads += [
            (f"ab{i}", a[-1500:] + r + b[:1500]),
            (f"cd{i}", c[-1500:] + r + d[:1500]),
            (f"bc{i}", b[-1500:] + c[:1500]),
        ]
    (tmp_path / "reads.fa").write_text("".join(f">{n}\n{s}\n" for n, s in reads))
    gfa = tmp_path / "g.gfa"
    assessment = assess_assembly_graph(
        gfa, [tmp_path / "reads.fa"], out_dir=tmp_path / "ev", bootstrap=0
    )

    unified = unify_assembly_graph(
        gfa, sample="S1", out_dir=tmp_path / "ov", assessment=assessment, backend="oatk"
    )

    graph = load_gfa(unified.gfa)
    assert all(link.overlap == "0M" for link in graph.links)
    _fingerprint(unified.gfa)  # the pangenome conversion precheck
    assert unified.report["overlap_bp_removed"] == 250
    assert unified.report["links_with_evidence"] == 5
    stored = sum(len(s.sequence or "") for s in graph.segments.values())
    # every base once: the overlapping copies are one base
    assert stored == sum(len(s) for s in segs.values()) - 250, "no base is stored twice"
    (name,) = path_sequences(unified.gfa)
    assert name == "S1#1#mt"
    expected = a + r + b + c + r + d
    assert path_sequences(unified.gfa)[name] in (expected, _revcomp(expected))
    assert assessment.linear_fasta is not None
    linear = "".join(assessment.linear_fasta.read_text().splitlines()[1:])
    assert path_sequences(unified.gfa)[name] == linear
    (fa,) = export_representatives([unified.gfa], tmp_path / "pan")
    assert "".join(fa.read_text().splitlines()[1:]) == linear


@needs_ovasm
def test_assessment_of_another_graph_is_refused(tmp_path: Path) -> None:
    rng = random.Random(9)
    x, y = _dna(3000, rng), _dna(3000, rng)
    for name in ("g1", "g2"):
        (tmp_path / f"{name}.gfa").write_text(f"S\tX\t{x}\nS\tY\t{y}\nL\tX\t+\tY\t+\t0M\n")
    (tmp_path / "r.fa").write_text(f">r\n{x[-1500:] + y[:1500]}\n")
    other = assess_assembly_graph(
        tmp_path / "g2.gfa", [tmp_path / "r.fa"], out_dir=tmp_path / "e", bootstrap=0
    )
    with pytest.raises(OrganelleInputError) as err:
        unify_assembly_graph(
            tmp_path / "g1.gfa", sample="S", out_dir=tmp_path / "o", assessment=other
        )
    assert err.value.code == "assembly.assessment_mismatch"
