"""Dispersed-repeat fraction and assembly-graph complexity."""

from __future__ import annotations

import random
import shutil

import pytest

from organelleverse.structure import dispersed_repeats, graph_complexity
from organelleverse.structure.dispersed import SelfHit, filter_self_hits, repeat_landscape
from organelleverse.structure.graph_complexity import graph_complexity_metrics

# --------------------------------------------------------------- graph complexity


def _gfa(tmp_path, text: str):
    path = tmp_path / "graph.gfa"
    path.write_text(text.strip().replace(" ", "\t") + "\n")
    return path


def test_single_circular_segment_is_simple(tmp_path) -> None:
    result = graph_complexity(_gfa(tmp_path, "S u1 * LN:i:150000\nL u1 + u1 + 0M"))
    assert result.metrics["complexity"] == "simple"
    assert result.metrics["components"][0]["shape"] == "circular"
    assert result.metrics["n_dead_ends"] == 0


def test_linear_segment_has_two_dead_ends() -> None:
    metrics = graph_complexity_metrics({"u1": 5000}, [])
    assert metrics["complexity"] == "simple"
    assert metrics["n_dead_ends"] == 2
    assert metrics["components"][0]["shape"] == "linear"


def test_repeat_graph_is_complex_with_a_double_fork(tmp_path) -> None:
    # u1 -> r -> u2 -> r(rc)... : a 2x2 repeat r joining two single-copy arms.
    text = """
S u1 ACGTACGTAC
S u2 ACGTACGTACGTACGTACGT
S r ACG
L u1 + r + 0M
L r + u2 + 0M
L u2 + r + 0M
L r + u1 + 0M
"""
    result = graph_complexity(_gfa(tmp_path, text))
    metrics = result.metrics
    assert metrics["complexity"] == "complex"
    assert metrics["n_segments"] == 3 and metrics["n_links"] == 4
    assert list(metrics["double_forks"]) == ["r"]
    assert metrics["n_dead_ends"] == 0
    assert metrics["total_length"] == 33
    assert "has_double_forks" in result.flags


def test_reverse_complement_links_are_one_adjacency() -> None:
    links = [("a", "+", "b", "+"), ("b", "-", "a", "-"), ("b", "+", "a", "+")]
    metrics = graph_complexity_metrics({"a": 10, "b": 10}, links)
    assert metrics["n_links"] == 2
    assert metrics["components"][0]["shape"] == "closed_graph"


def test_disconnected_graphs_are_multicomponent(tmp_path) -> None:
    text = """
S c1 * LN:i:124987
S c2 * LN:i:109138
L c1 + c1 + 1000M
L c2 - c2 - 1000M
"""
    metrics = graph_complexity(_gfa(tmp_path, text)).metrics
    assert metrics["complexity"] == "multicomponent"
    assert metrics["n_components"] == 2
    assert [row["length"] for row in metrics["components"]] == [124987, 109138]
    assert metrics["log_mean_segments_per_component"] == 0.0


# --------------------------------------------------------------- dispersed repeats


def _hit(q, qs, qe, t, ts, te, strand="+", identity=99.0, length=None, mapq=60):
    return SelfHit(q, qs, qe, t, ts, te, strand, identity, length or qe - qs + 1, mapq)


def test_filter_drops_diagonal_short_low_identity_and_reciprocal() -> None:
    hits = [
        _hit("m", 1, 10_000, "m", 1, 10_000),  # the trivial self-match
        _hit("m", 101, 2100, "m", 120, 2119),  # near-diagonal, same strand
        _hit("m", 1001, 3000, "m", 7001, 9000),  # a real direct repeat
        _hit("m", 7001, 9000, "m", 1001, 3000),  # its reciprocal
        _hit("m", 4001, 4050, "m", 8001, 8050),  # too short
        _hit("m", 5001, 5500, "m", 6001, 6500, identity=80.0),  # too divergent
        _hit("m", 5001, 5500, "m", 6001, 6500, mapq=0),  # unplaceable
        _hit("m", 101, 600, "m", 101, 600, strand="-"),  # a palindrome is kept
    ]
    kept = filter_self_hits(hits, min_length=100, min_identity=90, min_mapq=1, diagonal_band=50)
    assert [(h.qstart, h.tstart, h.strand) for h in kept] == [(1001, 7001, "+"), (101, 101, "-")]


def test_landscape_merges_both_sides() -> None:
    hits = [
        _hit("m", 1001, 3000, "m", 7001, 9000),
        _hit("m", 2501, 3500, "m", 9501, 10_500, strand="-"),
    ]
    landscape = repeat_landscape(hits, {"m": 20_000})
    # 1001-3500 (2500) + 7001-9000 (2000) + 9501-10500 (1000)
    assert landscape["repeat_bases"] == 5500
    assert landscape["repeat_fraction"] == pytest.approx(5500 / 20_000)
    assert landscape["n_direct"] == 1 and landscape["n_inverted"] == 1
    assert landscape["pairs_by_length"] == {"100-1000": 0, "1000-5000": 2, ">=5000": 0}


def _revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


@pytest.mark.skipif(
    not (shutil.which("minimap2") or shutil.which("blastn")),
    reason="needs minimap2 or blastn",
)
def test_end_to_end_on_a_synthetic_genome(tmp_path) -> None:
    rng = random.Random(7)

    def random_dna(n: int) -> str:
        return "".join(rng.choice("ACGT") for _ in range(n))

    direct = random_dna(2000)
    inverted = random_dna(1500)
    genome = (
        random_dna(10_000) + direct + random_dna(15_000) + inverted
        + random_dna(12_000) + direct + random_dna(9_000) + _revcomp(inverted) + random_dna(8_000)
    )
    fasta = tmp_path / "mito.fa"
    fasta.write_text(">mito\n" + genome + "\n")
    result = dispersed_repeats(fasta)
    metrics = result.metrics
    expected = (2 * 2000 + 2 * 1500) / len(genome)
    assert metrics["repeat_fraction"] == pytest.approx(expected, rel=0.02)
    assert metrics["n_direct"] >= 1 and metrics["n_inverted"] >= 1
    assert metrics["longest_repeat"] >= 1950
    assert metrics["repeats"][0]["positions"][0] >= 1


def test_sequences_restrict_to_the_mitochondrial_components(tmp_path) -> None:
    rng = random.Random(11)

    def dna(n: int) -> str:
        return "".join(rng.choice("ACGT") for _ in range(n))

    m1, m2, plastid, stray = dna(3000), dna(2000), dna(4000), dna(500)
    text = "\n".join(
        [
            f"S\tm1\t{m1}", f"S\tm2\t{m2}", f"S\tp1\t{plastid}", f"S\tx1\t{stray}",
            "L\tm1\t+\tm2\t+\t0M", "L\tm2\t+\tm1\t+\t0M", "L\tp1\t+\tp1\t+\t0M",
        ]
    )
    gfa = tmp_path / "raw.gfa"
    gfa.write_text(text + "\n")
    mito = tmp_path / "mito.fa"
    # the mitochondrial FASTA spells m1 then m2 (m2 on the reverse strand)
    revcomp = m2.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    mito.write_text(">ctg1\n" + m1 + revcomp + "\n")
    raw = graph_complexity(gfa).metrics
    assert raw["complexity"] == "multicomponent" and raw["n_components"] == 3
    picked = graph_complexity(gfa, sequences=mito).metrics
    assert picked["complexity"] == "complex"
    assert picked["n_segments"] == 2 and picked["n_components"] == 1
    assert picked["selection"]["components_dropped"] == 2
    assert picked["selection"]["length_dropped"] == 4500
    with pytest.raises(Exception):
        graph_complexity(gfa, sequences=mito, min_kmer_fraction=0)


# ------------------------------------------------------------------ intron classes

TBLOUT = """#target name  accession query name accession mdl mdl from mdl to seq from seq to strand trunc pass gc bias score E-value inc description of target
#------------ --------- ---------- --------- --- -------- ------ -------- ------ ------ ----- ---- ---- ---- ----- ------- --- ---------------------
ctg1  -  Intron_gpII  RF00029  cm  1  77  343815  343754  -  no  1  0.43  0.0  63.3  5.8e-13  !  -
ctg1  -  Intron_gpII  RF00029  cm  1  77  141399  141533  +  no  1  0.40  0.0  52.7  2.3e-10  !  -
ctg1  -  Intron_gpII  RF00029  cm  1  77  9000  9080  +  no  1  0.40  0.0  20.1  4.0e-01  ?  -
"""


def test_tblout_keeps_included_hits_with_forward_coordinates() -> None:
    from organelleverse.structure.intron_classes import parse_tblout

    hits = parse_tblout(TBLOUT)
    assert [(h["start"], h["end"], h["strand"]) for h in hits] == [
        (141399, 141533, "+"),
        (343754, 343815, "-"),
    ]
    assert hits[1]["bits"] == 63.3


def test_intron_models_are_shipped() -> None:
    from organelleverse.structure.intron_classes import MODELS

    for path in MODELS.values():
        head = path.read_text().splitlines()[:3]
        assert head[0].startswith("INFERNAL1/a")
    assert "RF00028" in MODELS["group_I"].read_text()[:400]


@pytest.mark.skipif(not shutil.which("cmsearch"), reason="needs Infernal cmsearch")
def test_intron_classes_end_to_end(tmp_path) -> None:
    from organelleverse.structure import intron_classes

    rng = random.Random(5)
    fasta = tmp_path / "random.fa"
    fasta.write_text(">r\n" + "".join(rng.choice("ACGT") for _ in range(20_000)) + "\n")
    result = intron_classes(fasta)
    assert result.status == "ok"
    assert result.metrics["intron_count"] == 0  # random sequence holds no intron cores


def test_subgraph_is_written_and_suggested_for_drawing(tmp_path) -> None:
    from organelleverse.structure.graph_complexity import _read_gfa, write_subgraph

    rng = random.Random(13)

    def dna(n: int) -> str:
        return "".join(rng.choice("ACGT") for _ in range(n))

    m1, m2, plastid = dna(3000), dna(2000), dna(4000)
    gfa = tmp_path / "raw.gfa"
    gfa.write_text("\n".join([
        "H\tVN:Z:1.0", f"S\tm1\t{m1}", f"S\tm2\t{m2}", f"S\tp1\t{plastid}",
        "L\tm1\t+\tm2\t+\t0M", "L\tm2\t+\tm1\t+\t0M", "L\tp1\t+\tp1\t+\t0M", "P\tpt\tp1+\t*",
    ]) + "\n")
    mito = tmp_path / "mito.fa"
    mito.write_text(">ctg1\n" + m1 + m2 + "\n")
    sub = write_subgraph(gfa, mito, tmp_path / "sub.gfa")
    segments, links = _read_gfa(sub)
    assert set(segments) == {"m1", "m2"} and len(links) == 2
    assert sub.read_text().startswith("H\tVN:Z:1.0") and "\tpt\t" not in sub.read_text()
    [suggestion] = graph_complexity(gfa, sequences=mito).suggested_operations
    assert suggestion.operation_id == "visualization.plot_assembly_graph"
    assert suggestion.parameter_changes["sequences"] == str(mito)
