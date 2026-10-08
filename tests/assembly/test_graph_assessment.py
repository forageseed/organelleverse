from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from organelleverse._ovasm import resolve_ovasm
from organelleverse.assembly.graph_assessment import (
    NO_LINEAR_PATH,
    UNDECIDED,
    UNSUPPORTED_LINKS,
    assess_assembly_graph,
    findings_for,
    flags_for,
)

needs_ovasm = pytest.mark.skipif(resolve_ovasm() is None, reason="ovasm binary not built")


def _evidence(reads: list[int]) -> dict[str, Any]:
    return {
        "links": [{"from": f"S{i}+", "to": f"S{i + 1}+", "reads": n} for i, n in enumerate(reads)],
        "links_supported": sum(n > 0 for n in reads),
        "branches": [{"from": "S0+", "minority_fraction": 0.3}],
    }


def _linear(*, decisive: bool, best: bool = True) -> dict[str, Any]:
    return {
        "best": {"path": "S0+ S1+", "length": 900, "circular": False} if best else None,
        "decisive": decisive,
        "margin_ln": 0.2 if best else None,
        "repeats_not_placed": [],
    }


def test_unsupported_links_are_flagged_one_finding_each() -> None:
    ev = _evidence([12, 0, 7, 0])
    findings = findings_for(ev, None)
    unsupported = [f.metric for f in findings if f.code == "assembly.graph_link_unsupported"]
    assert unsupported == ["S1+ -> S2+", "S3+ -> S4+"]
    (summary,) = [f for f in findings if f.code == "assembly.graph_links_supported"]
    assert (summary.value, summary.unit) == (2, "of 4 links")
    assert flags_for(ev, None) == [UNSUPPORTED_LINKS]


def test_linearization_flags_distinguish_undecided_from_unresolved() -> None:
    ev = _evidence([5, 5])
    assert flags_for(ev, _linear(decisive=True)) == []
    assert flags_for(ev, _linear(decisive=False)) == [UNDECIDED]
    assert flags_for(ev, _linear(decisive=False, best=False)) == [NO_LINEAR_PATH]
    # one component unringed, another no cover of a few molecules explains
    assert flags_for(ev, {**_linear(decisive=True), "unsolved_components": 1}) == [NO_LINEAR_PATH]
    metrics = {
        f.metric: f.value
        for f in findings_for(ev, _linear(decisive=False))
        if f.code == "assembly.graph_linearization"
    }
    assert metrics == {
        "molecules": 1,
        "path": "S0+ S1+",
        "length": 900,
        "circular": False,
        "decisive": False,
        "margin_ln": 0.2,
    }


def _dna(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _revcomp(s: str) -> str:
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


@needs_ovasm
def test_assessment_finds_the_bad_link_and_threads_the_repeat_twice(tmp_path: Path) -> None:
    rng = random.Random(5)
    a, b, c, d, r = (_dna(3000, rng) for _ in range(5))
    # A->R->B, C->R->D around repeat R, B->C joins the two halves; A->D is an assembly error.
    (tmp_path / "g.gfa").write_text(
        "H\tVN:Z:1.0\n"
        + "".join(f"S\t{n}\t{s}\n" for n, s in zip("ABCDR", (a, b, c, d, r), strict=True))
        + "L\tA\t+\tR\t+\t0M\nL\tC\t+\tR\t+\t0M\nL\tR\t+\tB\t+\t0M\nL\tR\t+\tD\t+\t0M\n"
        + "L\tB\t+\tC\t+\t0M\nL\tA\t+\tD\t+\t0M\n"
    )
    reads: list[tuple[str, str]] = []
    for i in range(20):
        reads += [
            (f"ab{i}", a[-1500:] + r + b[:1500]),
            (f"cd{i}", c[-1500:] + r + d[:1500]),
            (f"bc{i}", b[-1500:] + c[:1500]),
        ]
    (tmp_path / "reads.fa").write_text("".join(f">{n}\n{s}\n" for n, s in reads))

    report = assess_assembly_graph(
        tmp_path / "g.gfa", [tmp_path / "reads.fa"], out_dir=tmp_path / "out", bootstrap=0
    )

    assert report.evidence["links_supported"] == 5
    bad = [f.metric for f in report.findings if f.code == "assembly.graph_link_unsupported"]
    assert bad == ["A+ -> D+"]
    assert report.flags == (UNSUPPORTED_LINKS,)
    assert report.linearization is not None and report.linearization["decisive"]
    assert report.linear_fasta is not None
    seq = "".join(report.linear_fasta.read_text().splitlines()[1:])
    expected = a + r + b + c + r + d
    assert seq in (expected, _revcomp(expected)), "R is laid once per crossing"


@needs_ovasm
def test_a_rerun_without_a_path_rewrites_the_fasta(tmp_path: Path) -> None:
    rng = random.Random(6)
    a, b = _dna(3000, rng), _dna(3000, rng)
    gfa = tmp_path / "g.gfa"
    gfa.write_text(f"H\tVN:Z:1.0\nS\tA\t{a}\nS\tB\t{b}\nL\tA\t+\tB\t+\t0M\n")
    spanning = tmp_path / "span.fa"
    spanning.write_text("".join(f">s{i}\n{a[-1500:] + b[:1500]}\n" for i in range(5)))
    out = tmp_path / "out"
    first = assess_assembly_graph(gfa, [spanning], out_dir=out, bootstrap=0)
    assert first.linear_fasta is not None and first.flags == ()
    joined = first.linear_fasta.read_text()

    elsewhere = tmp_path / "none.fa"
    elsewhere.write_text(f">x\n{a[:2000]}\n")
    second = assess_assembly_graph(gfa, [elsewhere], out_dir=out, bootstrap=0)
    # no read joins A and B: each is a linear molecule of its own, not the stale joined path
    assert second.linear_fasta is not None
    records = [r for r in second.linear_fasta.read_text().split(">") if r]
    assert second.linear_fasta.read_text() != joined
    assert sorted(len("".join(r.splitlines()[1:])) for r in records) == [3000, 3000]
    assert second.flags == (UNSUPPORTED_LINKS, UNDECIDED)
    unbridged = [f.metric for f in second.findings if f.code == "assembly.graph_anchor_unbridged"]
    assert unbridged == ["A", "B"], "the report names the segments no read connects"
