"""Repeat conformations on an assembly graph: topology, molecules and spanning reads."""

import random
import shutil

import pytest

from organelleverse.structure.graph_conformation import (
    Alignment,
    analyze_conformations,
    build_probes,
    count_spanning_reads,
    find_repeats,
    parse_paf,
    read_assembly_graph,
    recombine_paths,
    reported_molecules,
    reverse_complement,
    spell_molecules,
)


def _random(length, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


A, B, R = _random(6000, 1), _random(5000, 2), _random(1500, 3)


def _direct_repeat_gfa(tmp_path, *, sequences=True):
    """Master circle A-R-B-R: R is a direct repeat; recombination gives A-R and B-R."""
    seq = (lambda s: s) if sequences else (lambda s: "*")
    text = (
        f"S\tA\t{seq(A)}\tLN:i:{len(A)}\tdp:f:40\n"
        f"S\tB\t{seq(B)}\tLN:i:{len(B)}\tdp:f:42\n"
        f"S\tR\t{seq(R)}\tLN:i:{len(R)}\tdp:f:81\n"
        "L\tA\t+\tR\t+\t0M\nL\tR\t+\tB\t+\t0M\nL\tB\t+\tR\t+\t0M\nL\tR\t+\tA\t+\t0M\n"
        "P\tmt\tA+,R+,B+,R+\t*\n"
    )
    path = tmp_path / "graph.gfa"
    path.write_text(text)
    return path


def test_repeat_is_a_segment_branching_on_both_ends(tmp_path):
    graph = read_assembly_graph(_direct_repeat_gfa(tmp_path))
    (repeat,) = find_repeats(graph)
    assert repeat.name == "R"
    assert repeat.kind == "repeat"
    assert sorted(repeat.predecessors) == [("A", "+"), ("B", "+")]
    assert sorted(repeat.successors) == [("A", "+"), ("B", "+")]
    assert len(repeat.configurations()) == 2


def test_depth_from_counts_is_per_base(tmp_path):
    path = tmp_path / "pmat.gfa"
    path.write_text("S\tx\t*\tLN:i:1000\tRC:i:45000\n")
    assert read_assembly_graph(path).segments["x"].depth == 45.0


def test_conformations_spell_master_circle_and_subcircles(tmp_path):
    result = analyze_conformations(_direct_repeat_gfa(tmp_path, sequences=False))
    (row,) = result["repeats"]
    assert row["status"] == "candidate_only"
    assembled, recombined = row["configurations"]
    assert assembled["assembled"] and not recombined["assembled"]
    assert [(m["circular"], m["length"]) for m in assembled["molecules"]] == [(True, 14000)]
    assert sorted(m["length"] for m in recombined["molecules"]) == [6500, 7500]
    assert all(m["circular"] for m in recombined["molecules"])
    assert assembled["proportion"] is None and assembled["supporting_reads"] is None


def test_inverted_repeat_conformation_is_an_isomer(tmp_path):
    # A-R-B-R(-): R is inverted, so recombination flips B instead of splitting.
    path = tmp_path / "ir.gfa"
    path.write_text(
        "S\tA\t*\tLN:i:6000\nS\tB\t*\tLN:i:5000\nS\tR\t*\tLN:i:1500\n"
        "L\tA\t+\tR\t+\t0M\nL\tR\t+\tB\t+\t0M\nL\tB\t+\tR\t-\t0M\nL\tR\t-\tA\t+\t0M\n"
    )
    graph = read_assembly_graph(path)
    (repeat,) = find_repeats(graph)
    for matching in repeat.configurations():
        molecules = spell_molecules(graph, {"R": matching})
        assert [(m["circular"], m["length"]) for m in molecules] == [(True, 14000)]


def test_unresolved_branching_spells_no_molecules(tmp_path):
    path = tmp_path / "bubble.gfa"
    path.write_text(
        "S\tA\t*\tLN:i:10\nS\tB\t*\tLN:i:10\nS\tC\t*\tLN:i:10\nS\tD\t*\tLN:i:10\n"
        "L\tA\t+\tB\t+\t0M\nL\tA\t+\tC\t+\t0M\nL\tB\t+\tD\t+\t0M\n"
        "L\tC\t+\tD\t+\t0M\nL\tD\t+\tA\t+\t0M\n"
    )
    graph = read_assembly_graph(path)
    assert find_repeats(graph) == []
    assert spell_molecules(graph, {}) is None


def test_spanning_needs_both_anchors_and_best_score_wins(tmp_path):
    graph = read_assembly_graph(_direct_repeat_gfa(tmp_path))
    (repeat,) = find_repeats(graph)
    probes = build_probes(graph, repeat, flank=1000)
    index = {probe.pair: i for i, probe in enumerate(probes)}
    ab = index[(("A", "+"), ("B", "+"))]
    aa = index[(("A", "+"), ("A", "+"))]
    full = 1000 + 1500 + 1000
    hits = [
        Alignment("span", 4000, ab, 0, full, full, full, 6000),
        Alignment("span", 4000, aa, 0, 2600, 2600, 2600, 5000),  # stops inside the flank
        Alignment("short", 2000, ab, 900, 2700, 1800, 1800, 3000),  # misses the anchors
        Alignment("tie", 4000, ab, 0, full, full, full, 6000),
        Alignment("tie", 4000, aa, 0, full, full, full, 6000),
        Alignment("noisy", 4000, ab, 0, full, 2000, full, 1000),  # below identity
    ]
    counts = count_spanning_reads(probes, hits, anchor=500, min_identity=0.95)["R"]
    assert dict(counts["pairs"]) == {ab: 1}
    assert counts["ambiguous"] == 1
    assert counts["touching"] == 4
    assert counts["longest"] == 4000


def test_parse_paf_reads_alignment_score():
    line = "r\t4000\t0\t4000\t+\t3\t3500\t0\t3500\t3490\t3500\t0\ttp:A:P\tAS:i:6900\n"
    (hit,) = parse_paf([line])
    assert (hit.target, hit.matches, hit.block, hit.score) == (3, 3490, 3500, 6900)


@pytest.mark.skipif(shutil.which("minimap2") is None, reason="minimap2 not installed")
def test_minimap2_spanning_reads_give_conformation_frequencies(tmp_path):
    reads = []
    flank = 800
    junctions = {("A", "B"): 15, ("B", "A"): 15, ("A", "A"): 5, ("B", "B"): 5}
    sequences = {"A": A, "B": B}
    for (left, right), copies in junctions.items():
        read = sequences[left][-flank:] + R + sequences[right][:flank]
        for i in range(copies):
            body = read if i % 2 == 0 else reverse_complement(read)
            reads.append(f"@{left}{right}{i}\n{body}\n+\n{'I' * len(body)}\n")
    fastq = tmp_path / "reads.fastq"
    fastq.write_text("".join(reads))
    result = analyze_conformations(_direct_repeat_gfa(tmp_path), reads=[fastq], threads=1)
    (row,) = result["repeats"]
    assert row["status"] == "supported"
    assert row["spanning_reads"] == 40
    assembled, recombined = row["configurations"]
    assert (assembled["supporting_reads"], recombined["supporting_reads"]) == (30, 10)
    assert (assembled["proportion"], recombined["proportion"]) == (0.75, 0.25)
    assert assembled["proportion_denominator"] == recombined["proportion_denominator"] == 40
    assert "not whole-molecule abundance" in row["proportion_definition"]
    assert result["read_evidence"]["preset"] == "map-hifi"


@pytest.mark.skipif(shutil.which("minimap2") is None, reason="minimap2 not installed")
def test_repeat_longer_than_reads_is_not_spannable(tmp_path):
    fastq = tmp_path / "reads.fastq"
    body = A[-700:] + R[:900]  # enters the repeat but cannot leave it
    fastq.write_text("".join(f"@r{i}\n{body}\n+\n{'I' * len(body)}\n" for i in range(5)))
    result = analyze_conformations(_direct_repeat_gfa(tmp_path), reads=[fastq], threads=1)
    (row,) = result["repeats"]
    assert row["status"] == "not_spannable"
    assert row["longest_touching_read"] == 1600 < row["required_read_length"]
    assert all(c["proportion"] is None for c in row["configurations"])


def _blocked_gfa(tmp_path, path_line):
    """Master circle A-R-B-R plus a tip C off A: A branches on one end, so the graph spells nothing."""
    path = tmp_path / "blocked.gfa"
    path.write_text(
        "S\tA\t*\tLN:i:6000\tdp:f:40\nS\tB\t*\tLN:i:5000\tdp:f:42\nS\tR\t*\tLN:i:1500\tdp:f:81\n"
        "S\tC\t*\tLN:i:800\tdp:f:5\n"
        "L\tA\t+\tR\t+\t0M\nL\tR\t+\tB\t+\t0M\nL\tB\t+\tR\t+\t0M\nL\tR\t+\tA\t+\t0M\nL\tA\t+\tC\t+\t0M\n"
        + path_line
    )
    return path


def test_reported_paths_give_molecules_when_the_graph_spells_none(tmp_path):
    gfa = _blocked_gfa(
        tmp_path,
        "P\ts#1#mt.1\tA+,R+,B+,R+\t0M,0M,0M\tTP:Z:circular\n"
        "P\ts#2#mt.1\tA+,R+,A+,R+\t0M,0M,0M\tTP:Z:circular\n",
    )
    graph = read_assembly_graph(gfa)
    assert spell_molecules(graph, {"R": [(("A", "+"), ("B", "+")), (("B", "+"), ("A", "+"))]}) is None
    assert reported_molecules(graph) == [([("A", "+"), ("R", "+"), ("B", "+"), ("R", "+")], True)]
    (row,) = analyze_conformations(gfa)["repeats"]
    assembled, recombined = row["configurations"]
    assert assembled["molecules_source"] == "paths"
    assert [(m["circular"], m["length"]) for m in assembled["molecules"]] == [(True, 14000)]
    assert recombined["molecules_source"] == "paths_recombined"
    assert [(m["circular"], m["length"]) for m in recombined["molecules"]] == [(True, 7500), (True, 6500)]


def test_repeat_passed_three_times_keeps_the_paths_that_use_each_conformation(tmp_path):
    # R passed three times: A>R>B, B>R>A and A>R>A; neither conformation covers all, re-pairing is not unique
    gfa = _blocked_gfa(
        tmp_path,
        "P\ts#1#mt.1\tA+,R+,B+,R+\t0M,0M,0M\tTP:Z:circular\nP\ts#1#mt.2\tA+,R+\t0M\tTP:Z:circular\n",
    )
    graph = read_assembly_graph(gfa)
    matching = [(("A", "+"), ("B", "+")), (("B", "+"), ("A", "+"))]
    assert recombine_paths(graph, reported_molecules(graph), "R", matching) is None
    (row,) = analyze_conformations(gfa)["repeats"]
    by_pairs = {tuple(map(tuple, conf["pairs"])): conf for conf in row["configurations"]}
    straight = by_pairs[(("A+", "B+"), ("B+", "A+"))]
    crossed = by_pairs[(("A+", "A+"), ("B+", "B+"))]
    assert straight["molecules_source"] == crossed["molecules_source"] == "paths_partial"
    assert [m["length"] for m in straight["molecules"]] == [14000]
    assert [m["length"] for m in crossed["molecules"]] == [7500]


def test_linear_topology_tag_wins_over_a_closing_link(tmp_path):
    gfa = _blocked_gfa(tmp_path, "P\tmt\tA+,R+,B+,R+\t0M,0M,0M\tTP:Z:linear\n")
    ((steps, circular),) = reported_molecules(read_assembly_graph(gfa))
    assert len(steps) == 4 and circular is False
