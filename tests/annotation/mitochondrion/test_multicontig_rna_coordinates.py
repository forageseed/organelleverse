"""tRNA and rRNA coordinates must live in the pipeline's merged-genome space.

``load_fasta`` joins the contigs of a multi-contig mitogenome with 200 N and every writer reads
coordinates in that space. The tRNA and rRNA stages used to concatenate the records with no spacer
(tRNA) or scan each record separately (rRNA), which put genes on the second and later contigs at the
wrong place: on the first contig for rRNA, 200 x k bp to the left for tRNA. Mitochondrial tRNA
recall of the main benchmark was understated by about a third because of it.
"""

from __future__ import annotations

import random
from pathlib import Path

from Bio import SeqIO

from organelleverse.annotation.mitochondrion.db import DBManager
from organelleverse.annotation.mitochondrion.fasta import load_fasta
from organelleverse.annotation.mitochondrion.rrna_hmm import annotate_rrna_hmm
from organelleverse.annotation.mitochondrion.rrna_native import annotate_rrna_native
from organelleverse.annotation.mitochondrion.trna_native import (
    annotate_trna_cm,
    annotate_trna_native,
)

GAP = 200


def _random(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _write(path: Path, contigs: list[tuple[str, str]]) -> Path:
    path.write_text("".join(f">{name}\n{seq}\n" for name, seq in contigs))
    return path


def _rrna_reference() -> str:
    ref_file = sorted(DBManager().rrna_ref_dir.glob("*.fasta"))[0]
    return str(next(SeqIO.parse(str(ref_file), "fasta")).seq).upper().replace("U", "T")


def test_rrna_hmm_on_a_later_contig_is_reported_in_merged_coordinates(tmp_path: Path):
    ref = _rrna_reference()
    first = _random(400, 1)
    second = _random(150, 2) + ref + _random(90, 3)
    fasta = _write(tmp_path / "g.fasta", [("c1", first), ("c2", second)])

    hits = annotate_rrna_hmm(fasta, tmp_path, DBManager())

    best = max(hits, key=lambda h: h.score)
    offset = len(first) + GAP + 150
    assert (best.start, best.end) == (offset + 1, offset + len(ref))
    merged = load_fasta(fasta).sequence
    assert merged[best.start - 1 : best.end] == ref


def test_rrna_native_on_a_later_contig_is_reported_in_merged_coordinates(tmp_path: Path):
    ref = _rrna_reference()
    first = _random(400, 4)
    second = _random(150, 5) + ref + _random(90, 6)
    fasta = _write(tmp_path / "g.fasta", [("c1", first), ("c2", second)])

    hits = annotate_rrna_native(fasta, tmp_path, DBManager())

    assert hits
    merged = load_fasta(fasta).sequence
    best = max(hits, key=lambda h: (h.end - h.start, h.score))
    # whatever the engine reports, the span has to sit on the reference in the merged genome
    assert abs(best.start - (len(first) + GAP + 150 + 1)) <= 5
    assert merged[best.start - 1 : best.end].count("N") == 0


def test_trna_reference_scan_on_a_later_contig_is_reported_in_merged_coordinates(tmp_path: Path):
    class MiniDB:
        def __init__(self, trna_ref_dir: Path):
            self.trna_ref_dir = trna_ref_dir

    ref = "G" * 7 + "T" * 5 + "ACGATCGATCGATCGATCGATCGATCG" + "TTC" + "A" * 32
    first = "N" * 30 + _random(300, 7)
    second = "N" * 30 + ref + "N" * 20
    fasta = _write(tmp_path / "g.fasta", [("c1", first), ("c2", second)])
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    (ref_dir / "trna_refs.fasta").write_text(f">Ref_tRNA_trnF-GAA_1\n{ref}\n")

    hits = annotate_trna_native(fasta, tmp_path / "out", MiniDB(ref_dir))

    start = len(first) + GAP + 30 + 1
    assert [(h.start, h.end) for h in hits] == [(start, start + len(ref) - 1)]


def test_trna_cm_stage_sees_the_merged_genome(tmp_path: Path, monkeypatch):
    """The CM stage is fed the merged sequence, so a hit's position is already in merged space."""
    import organelleverse.annotation.cmsearch.genome as cm_genome

    seen = {}

    def fake_find_trnas(genome, **kwargs):
        seen["genome"] = genome
        return []

    monkeypatch.setattr(cm_genome, "find_trnas", fake_find_trnas)
    fasta = _write(
        tmp_path / "g.fasta",
        [("c1", _random(120, 8)), ("c2", _random(90, 9)), ("c3", _random(60, 10))],
    )

    annotate_trna_cm(fasta, tmp_path)

    assert seen["genome"] == load_fasta(fasta).sequence
    assert len(seen["genome"]) == 120 + 90 + 60 + 2 * GAP


def test_single_contig_input_is_unchanged(tmp_path: Path, monkeypatch):
    import organelleverse.annotation.cmsearch.genome as cm_genome

    seen = {}
    monkeypatch.setattr(
        cm_genome, "find_trnas", lambda genome, **kw: seen.setdefault("genome", genome) and []
    )
    seq = _random(200, 11)
    fasta = _write(tmp_path / "g.fasta", [("only", seq)])

    annotate_trna_cm(fasta, tmp_path)

    assert seen["genome"] == seq


def test_empty_fasta_returns_no_hits(tmp_path: Path):
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    assert annotate_trna_cm(empty, tmp_path) == []
    assert annotate_rrna_hmm(empty, tmp_path, DBManager()) == []
