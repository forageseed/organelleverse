"""validate_transfers_longread on a simulated NUMT.

Real rice data showed the old check (one minimap2 run over the whole read file per
candidate, "alignment crosses the boundary by >= 1 bp") reported 8-376 linking reads
for candidates where 2-10 nuclear reads exist: organelle reads that are homologous
to a larger insert align across boundaries inside it. The reads here are exact
substrings of a simulated genome (the insert is 2% diverged from its organelle source), so
every count below is known.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import pytest

from organelleverse.transfer import transfer as transfer_module
from organelleverse.transfer.transfer import validate_transfers_longread

pytestmark = pytest.mark.skipif(shutil.which("minimap2") is None, reason="needs minimap2")

INSERT_AT = 9000  # 0-based; the insert is O[1000:4000], so it spans 9001..12000 (1-based)
INSERT_LEN = 3000


def _dna(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _diverge(seq: str, rate: float, seed: int) -> str:
    """Substitute ``rate`` of the bases: a real NUMT is never identical to its source, which is
    what lets a read inside the insert be told apart from an organelle read."""
    rng = random.Random(seed)
    swap = {"A": "C", "C": "G", "G": "T", "T": "A"}
    return "".join(swap[b] if rng.random() < rate else b for b in seq)


def _candidate(start: int, end: int, seqid: str = "chrN") -> dict:
    return {
        "nuclear_seqid": seqid,
        "nuclear_start": start,
        "nuclear_end": end,
        "organelle_seqid": "mt",
        "organelle_start": 1,
        "organelle_end": end - start + 1,
        "identity": 99.0,
        "length": end - start + 1,
        "evalue": 1e-50,
        "bitscore": 1000.0,
    }


@pytest.fixture
def world(tmp_path: Path) -> dict:
    organelle = _dna(6000, seed=1)
    flank = _dna(20000, seed=2)
    numt = _diverge(organelle[1000:4000], 0.02, seed=3)
    nuclear = flank[:INSERT_AT] + numt + flank[INSERT_AT:]
    (tmp_path / "nuclear.fa").write_text(f">chrN\n{nuclear}\n")
    (tmp_path / "mt.fa").write_text(f">mt\n{organelle}\n")
    reads: list[tuple[str, str]] = []

    def add(prefix: str, seq: str, copies: int) -> None:
        reads.extend((f"{prefix}{i}", seq) for i in range(copies))

    add("left", nuclear[8000:10000], 3)  # crosses the insert's left junction (1 kb each side)
    add("right", nuclear[11000:13000], 2)  # crosses the right junction
    add("inner", nuclear[9600:10900], 4)  # nuclear reads inside the insert (cross candidate B's boundaries)
    add("org", organelle[200:5000], 5)  # organelle reads covering the whole insert homology
    add("short", nuclear[8850:9300], 1)  # only 150 bp of flank before the insert
    (tmp_path / "reads.fa").write_text("".join(f">{n}\n{s}\n" for n, s in reads))
    return {
        "nuclear": tmp_path / "nuclear.fa",
        "organelle": tmp_path / "mt.fa",
        "reads": tmp_path / "reads.fa",
        "insert": _candidate(INSERT_AT + 1, INSERT_AT + INSERT_LEN),  # candidate A: the whole insert
        "inner": _candidate(10001, 10500),  # candidate B: both boundaries inside the insert
    }


def _linking(result, index: int) -> int:
    return result.metrics["candidates"][index]["linking_reads"]


def test_junction_reads_are_counted_and_organelle_reads_are_not(world: dict) -> None:
    result = validate_transfers_longread(
        world["nuclear"], [world["insert"], world["inner"]], world["reads"],
        organelle_fasta=world["organelle"],
    )
    assert result.status == "ok"
    assert _linking(result, 0) == 3 + 2  # left + right junction reads; organelle reads excluded
    assert _linking(result, 1) == 4  # only the nuclear reads that really cross B's boundaries
    assert result.metrics["organelle_reads_excluded"] >= 5
    assert "organelle_reads_not_excluded" not in result.flags


def test_without_the_organelle_genome_organelle_reads_inflate_internal_boundaries(world: dict) -> None:
    """The artefact seen on real data, now flagged instead of silently counted."""
    result = validate_transfers_longread(world["nuclear"], [world["insert"], world["inner"]], world["reads"])
    assert _linking(result, 1) == 4 + 5  # the 5 organelle reads cross B's boundaries too
    assert "organelle_reads_not_excluded" in result.flags
    # they do not reach 200 bp beyond the insert's own junctions
    assert _linking(result, 0) == 3 + 2


def test_a_boundary_needs_min_anchor_bases_on_both_sides(world: dict) -> None:
    only_left_junction = world["insert"]
    strict = validate_transfers_longread(
        world["nuclear"], [only_left_junction], world["reads"], organelle_fasta=world["organelle"], min_anchor=200
    )
    relaxed = validate_transfers_longread(
        world["nuclear"], [only_left_junction], world["reads"], organelle_fasta=world["organelle"], min_anchor=100
    )
    assert _linking(relaxed, 0) == _linking(strict, 0) + 1  # the read with 150 bp of flank


def test_both_boundaries_are_checked(world: dict) -> None:
    result = validate_transfers_longread(
        world["nuclear"], [world["insert"]], world["reads"], organelle_fasta=world["organelle"]
    )
    # 2 right-junction reads would be missed by a left-boundary-only rule
    assert _linking(result, 0) == 5


def test_reads_are_aligned_in_one_pass_whatever_the_number_of_candidates(
    world: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    real = transfer_module.run_external

    def counting(argv, **kwargs):
        calls.append([str(a) for a in argv])
        return real(argv, **kwargs)

    monkeypatch.setattr(transfer_module, "run_external", counting)
    candidates = [world["insert"], world["inner"], _candidate(10101, 10300), _candidate(10401, 10450)]
    validate_transfers_longread(world["nuclear"], candidates, world["reads"], organelle_fasta=world["organelle"])
    assert len([c for c in calls if "map-hifi" in c]) == 1


def test_candidate_on_an_unknown_sequence_has_no_links_and_nothing_crashes(world: dict) -> None:
    result = validate_transfers_longread(
        world["nuclear"], [world["insert"], _candidate(500, 900, seqid="nope")], world["reads"],
        organelle_fasta=world["organelle"],
    )
    assert _linking(result, 0) == 5
    assert _linking(result, 1) == 0
    assert result.metrics["candidates"][1]["longread_linked"] is False


def test_the_default_window_is_longer_than_a_read_so_nuclear_reads_are_not_lost(tmp_path: Path) -> None:
    """A read longer than its window is truncated there and loses to its full-length alignment on
    the organelle genome; real rice data lost 41 of 1,518 linked candidates with a 2 kb flank."""
    import inspect

    assert inspect.signature(validate_transfers_longread).parameters["flank"].default >= 10_000

    organelle = _dna(30000, seed=11)
    flank_seq = _dna(60000, seed=12)
    numt = _diverge(organelle[2000:26000], 0.02, seed=13)  # a 24 kb insert, 2% from its source
    nuclear = flank_seq[:INSERT_AT] + numt + flank_seq[INSERT_AT:]
    (tmp_path / "n.fa").write_text(f">chrN\n{nuclear}\n")
    (tmp_path / "o.fa").write_text(f">mt\n{organelle}\n")
    # four nuclear reads of 15 kb inside the insert, crossing candidate B (20001..20500)
    (tmp_path / "r.fa").write_text("".join(f">n{i}\n{nuclear[12000:27000]}\n" for i in range(4)))
    inner = _candidate(20001, 20500)

    default = validate_transfers_longread(tmp_path / "n.fa", [inner], tmp_path / "r.fa", organelle_fasta=tmp_path / "o.fa")
    narrow = validate_transfers_longread(
        tmp_path / "n.fa", [inner], tmp_path / "r.fa", organelle_fasta=tmp_path / "o.fa", flank=2000
    )
    assert _linking(default, 0) == 4
    assert _linking(narrow, 0) == 0  # the failure the default avoids
