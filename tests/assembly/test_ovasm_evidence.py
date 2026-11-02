from __future__ import annotations

import random
from pathlib import Path

import pytest

from organelleverse._ovasm import resolve_ovasm, run_evidence, run_unify

pytestmark = pytest.mark.skipif(resolve_ovasm() is None, reason="ovasm binary not built")


def _dna(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_evidence_counts_links_branches_and_repeat_pairings(tmp_path: Path) -> None:
    rng = random.Random(11)
    a, b, c, d, r = (_dna(3000, rng) for _ in range(5))
    (tmp_path / "g.gfa").write_text(
        "H\tVN:Z:1.0\n"
        + "".join(f"S\t{n}\t{s}\n" for n, s in zip("ABCDR", (a, b, c, d, r), strict=True))
        + "L\tA\t+\tR\t+\t0M\nL\tC\t+\tR\t+\t0M\nL\tR\t+\tB\t+\t0M\nL\tR\t+\tD\t+\t0M\n"
    )
    # master pairing A-R-B / C-R-D x 30, recombinant A-R-D / C-R-B x 10
    reads = []
    for i in range(30):
        reads += [(f"m{i}a", a[-1500:] + r + b[:1500]), (f"m{i}c", c[-1500:] + r + d[:1500])]
    for i in range(10):
        reads += [(f"x{i}a", a[-1500:] + r + d[:1500]), (f"x{i}c", c[-1500:] + r + b[:1500])]
    (tmp_path / "reads.fa").write_text("".join(f">{n}\n{s}\n" for n, s in reads))

    report = run_evidence(
        tmp_path / "g.gfa", [tmp_path / "reads.fa"], out_json=tmp_path / "ev.json", bootstrap=200
    )

    assert report["links_supported"] == 4
    assert {(link["from"], link["to"]): link["reads"] for link in report["links"]} == {
        ("A+", "R+"): 40,
        ("C+", "R+"): 40,
        ("R+", "B+"): 40,
        ("R+", "D+"): 40,
    }
    (pairing,) = report["repeats"]
    assert pairing["repeat"] == "R+"
    assert sorted(pairing["pairing_reads"]) == [20, 60]
    assert pairing["minority_fraction"] == pytest.approx(0.25)
    lo, hi = pairing["minority_ci95"]
    assert lo < 0.25 < hi


def test_sr_api_supports_dense_overlap_without_claiming_repeat_phase(tmp_path: Path) -> None:
    sequence = _dna(500, random.Random(173))
    graph = tmp_path / "dense.gfa"
    graph.write_text(
        f"H\tVN:Z:1.0\nS\tA\t{sequence[:300]}\nS\tB\t{sequence[174:]}\n"
        "L\tA\t+\tB\t+\t126M\n"
    )
    reads = tmp_path / "reads.fa"
    reads.write_text(f">r1\n{sequence[160:310]}\n>r2\n{sequence[160:310]}\n")
    report = run_evidence(
        graph, [reads], out_json=tmp_path / "sr.json", preset="sr", min_anchor=40,
        bootstrap=0,
    )
    assert report["links_supported"] == 0
    assert report["adjacency_supported_links"] == 1
    assert report["links"][0]["adjacency_reads"] == 2
    assert report["traversals"] == []
    assert report["phased_paths"] == []
    unified = tmp_path / "organelle.gfa"
    run_unify(
        graph, out_gfa=unified, out_json=tmp_path / "unify.json", sample="sample",
        backend="ovasm", organelle="mitochondrion", evidence_json=tmp_path / "sr.json",
    )
    links = [line for line in unified.read_text().splitlines() if line.startswith("L\t")]
    assert any("\tev:i:0\taj:i:2" in line for line in links)
    long_report = run_evidence(
        graph, [reads], out_json=tmp_path / "hifi.json", min_anchor=40, bootstrap=0,
    )
    assert long_report["adjacency_supported_links"] == 0
