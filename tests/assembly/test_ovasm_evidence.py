from __future__ import annotations

import random
from pathlib import Path

import pytest

from organelleverse._ovasm import resolve_ovasm, run_evidence

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
