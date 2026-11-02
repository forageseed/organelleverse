"""ovasm run inputs and plastid-derived (MTPT) pieces of a mitochondrial graph."""

import json
import random
import shutil

import pytest

from organelleverse._losat import resolve_losat
from organelleverse.structure.graph_conformation import analyze_conformations
from organelleverse.structure.graph_origin import load_ovasm_run


def _random(length, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


PLASTID = _random(20000, 9)
MTPT = PLASTID[8000:9500]
A = _random(2000, 1) + MTPT + _random(2500, 4)
B, R = _random(5000, 2), _random(1500, 3)
LINKS = "L\t{a}\t+\t{r}\t+\t0M\nL\t{r}\t+\t{b}\t+\t0M\nL\t{b}\t+\t{r}\t+\t0M\nL\t{r}\t+\t{a}\t+\t0M\n"
HAS_ALIGNER = resolve_losat() is not None or shutil.which("blastn") is not None


def _gfa(prefix):
    a, b, r = f"{prefix}A", f"{prefix}B", f"{prefix}R"
    return (
        f"S\t{a}\t{A}\tdp:f:40\nS\t{b}\t{B}\tdp:f:42\nS\t{r}\t{R}\tdp:f:81\n"
        + LINKS.format(a=a, b=b, r=r)
        + f"P\t{prefix}1\t{a}+,{r}+,{b}+,{r}+\t*\n"
    )


def _ovasm_run(tmp_path, *, both=True):
    root = tmp_path / "run"
    mito = root / "mitochondrion" if both else root
    mito.mkdir(parents=True)
    (mito / "assembly.gfa").write_text(_gfa(""))
    (mito / "organelle.gfa").write_text(_gfa("s."))
    (mito / "organelle.json").write_text(json.dumps({"sample": "s"}))
    evidence = {
        "repeats": [
            {"repeat": "R+", "entries": ["A+", "B+"], "exits": ["A+", "B+"], "pairing_reads": [30, 10]}
        ]
    }
    (mito / "evidence.json").write_text(json.dumps(evidence))
    (mito / "linearization.json").write_text(json.dumps({"components": [{"decisive": False}]}))
    if both:
        (root / "plastid").mkdir()
        (root / "plastid" / "molecules.fasta").write_text(f">pt circular=true\n{PLASTID}\n")
    return root


def test_ovasm_run_pairings_become_conformation_frequencies(tmp_path):
    run = load_ovasm_run(_ovasm_run(tmp_path, both=False))
    assert run["plastid"] is None and run["decisive"] is False
    result = analyze_conformations(run["gfa"], pairings=run["pairings"])
    (row,) = result["repeats"]
    assert row["support_source"] == "pairings" and row["status"] == "supported"
    assembled, recombined = row["configurations"]
    # straight pairing (A>A, B>B) is the sub-circle split; crossed is the master circle
    assert (assembled["supporting_reads"], recombined["supporting_reads"]) == (10, 30)
    assert (assembled["proportion"], recombined["proportion"]) == (0.25, 0.75)
    assert len(recombined["molecules"]) == 2


def test_cut_segments_map_by_end(tmp_path):
    root = _ovasm_run(tmp_path, both=False)
    gfa = (root / "organelle.gfa").read_text()
    head, tail = A[:3000], A[3000:]
    gfa = gfa.replace(f"S\ts.A\t{A}\tdp:f:40\n", f"S\ts.A.1\t{head}\tdp:f:40\nS\ts.A.2\t{tail}\tdp:f:40\n")
    gfa = gfa.replace("L\ts.A\t+\ts.R", "L\ts.A.2\t+\ts.R").replace("L\ts.R\t+\ts.A\t", "L\ts.R\t+\ts.A.1\t")
    gfa = gfa.replace("s.A+,", "s.A.1+,s.A.2+,") + "L\ts.A.1\t+\ts.A.2\t+\t0M\n"
    (root / "organelle.gfa").write_text(gfa)
    pairings = load_ovasm_run(root)["pairings"]["s.R"]
    straight = dict(pairings[0][0])
    assert straight == {"s.A.2+": "s.A.1+", "s.B+": "s.B+"}


@pytest.mark.skipif(not HAS_ALIGNER, reason="needs LOSAT or blastn")
def test_mtpt_found_with_donor_block_and_origin(tmp_path):
    run = load_ovasm_run(_ovasm_run(tmp_path))
    result = analyze_conformations(run["gfa"], pairings=run["pairings"], plastid=run["plastid"])
    plastid = result["plastid"]
    assert plastid["reference"]["length"] == len(PLASTID)
    (fragment,) = plastid["fragments"]
    assert fragment["segment"] == "s.A"
    assert abs(fragment["start"] - 2001) <= 5 and abs(fragment["end"] - 3500) <= 5
    (block,) = fragment["donor_blocks"]
    assert abs(block["start"] - 8001) <= 5 and abs(block["end"] - 9500) <= 5
    assert fragment["integration"] == "not_assessed"
    origins = {row["name"]: row["origin"] for row in result["segments"]}
    assert origins == {"s.A": "mitochondrial", "s.B": "mitochondrial", "s.R": "mitochondrial"}


@pytest.mark.skipif(
    not HAS_ALIGNER or shutil.which("minimap2") is None, reason="needs an aligner and minimap2"
)
def test_reads_spanning_an_mtpt_into_mitochondrial_flanks(tmp_path):
    run = load_ovasm_run(_ovasm_run(tmp_path))
    integrated = A[1200:4300]  # MTPT plus 800 bp of mitochondrial flank on each side
    plastid_only = PLASTID[7500:10000]  # plastid reads cover the MTPT but not the flanks
    reads = [integrated] * 4 + [plastid_only] * 20
    fastq = tmp_path / "reads.fastq"
    fastq.write_text("".join(f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(reads)))
    result = analyze_conformations(run["gfa"], reads=[fastq], plastid=run["plastid"], threads=1)
    (fragment,) = result["plastid"]["fragments"]
    assert fragment["spanning_reads"] == 4
    assert fragment["integration"] == "supported"
