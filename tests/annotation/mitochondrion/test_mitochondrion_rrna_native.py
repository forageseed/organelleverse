from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from organelleverse.annotation.mitochondrion.db import DBManager
from organelleverse.annotation.mitochondrion.rrna import (
    RawRRNA,
    _merge_rrna_fragments,
    _select_rrna_engine,
)


def test_rrna_engine_defaults_to_native(monkeypatch):
    monkeypatch.delenv("ORG_VERSE_RRNA_ENGINE", raising=False)
    assert _select_rrna_engine() == "native"


def test_rrna_engine_external_opt_in(monkeypatch):
    monkeypatch.setenv("ORG_VERSE_RRNA_ENGINE", "external")
    assert _select_rrna_engine() == "external"


class MiniDB:
    def __init__(self, rrna_ref_dir: Path):
        self.rrna_ref_dir = rrna_ref_dir


def _reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


def _write_fasta(path: Path, name: str, seq: str) -> None:
    path.write_text(f">{name}\n{seq}\n")


def test_native_rrna_finds_plus_and_minus_seed_hits(tmp_path: Path):
    from organelleverse.annotation.mitochondrion.rrna_native import annotate_rrna_native

    ref = (
        "AAACCGGGCACTACGGTGAGACGTGAAAACACCCGATCCCATTCCGACCTCGATAT"
        "GTGGAATCGTCTTGCGCCATATGTACTGAGATTGTTCGGGAGACATGGTCCAAGCCCGGTGA"
    )
    genome = "N" * 50 + ref + "N" * 80 + _reverse_complement(ref) + "N" * 50
    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", genome)
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    _write_fasta(ref_dir / "rrn5.mito.fasta", "rrn5", ref)

    hits = annotate_rrna_native(fasta, tmp_path / "out", MiniDB(ref_dir))

    assert [(hit.gene_name, hit.start, hit.end, hit.strand) for hit in hits] == [
        ("rrn5", 51, 51 + len(ref) - 1, 1),
        ("rrn5", 51 + len(ref) + 80, 51 + len(ref) + 80 + len(ref) - 1, -1),
    ]


def test_native_rrna_refines_long_rrna_boundaries_across_indels(tmp_path: Path):
    from organelleverse.annotation.mitochondrion.rrna_native import annotate_rrna_native

    rng = random.Random(3)
    ref = "".join(rng.choice("ACGT") for _ in range(220))
    observed = ref[:95] + "GGGGG" + ref[100:]
    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", "N" * 40 + observed + "N" * 30)
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    _write_fasta(ref_dir / "rrn18.mito.fasta", "rrn18", ref)

    hits = annotate_rrna_native(fasta, tmp_path / "out", MiniDB(ref_dir))

    assert [(hit.gene_name, hit.start, hit.end, hit.strand) for hit in hits] == [
        ("rrn18", 41, 40 + len(observed), 1)
    ]


def test_native_rrna_recovers_pmga_ranunculus_long_rrnas(tmp_path: Path):
    from organelleverse.annotation.mitochondrion.rrna import annotate_rrna

    fasta_value = os.environ.get("ORG_VERSE_PMGA_RANUNCULUS_FASTA")
    if not fasta_value:
        pytest.skip("Set ORG_VERSE_PMGA_RANUNCULUS_FASTA to run this real-data test")
    fasta = Path(fasta_value)
    if not fasta.exists():
        pytest.skip("ORG_VERSE_PMGA_RANUNCULUS_FASTA does not exist")

    annotations = annotate_rrna(fasta, tmp_path / "out", db_manager=DBManager())
    found = {(a.gene_name, a.genomic_start, a.genomic_end, a.strand.symbol) for a in annotations}

    assert ("rrn18", 395149, 397002, "-") in found
    assert ("rrn18", 108799, 110652, "-") in found
    assert ("rrn26", 65687, 69353, "+") in found
    assert ("rrn26", 443595, 447261, "-") in found
    assert ("rrn26", 1097376, 1101042, "-") in found


def test_rrna_merge_rules_keep_pmga_thresholds():
    hits = [
        RawRRNA("rrn5", 1, 79, 1, 90.0, "5S"),
        RawRRNA("rrn5", 100, 179, 1, 90.0, "5S"),
        RawRRNA("rrn18", 1000, 1699, 1, 80.0, "18S"),
        RawRRNA("rrn18", 1900, 2499, 1, 81.0, "18S"),
        RawRRNA("rrn26", 5000, 6400, -1, 82.0, "26S"),
        RawRRNA("rrn26", 6350, 7600, -1, 83.0, "26S"),
    ]

    merged = _merge_rrna_fragments(hits)

    assert [(hit.gene_name, hit.start, hit.end, hit.strand) for hit in merged] == [
        ("rrn5", 100, 179, 1),
        ("rrn18", 1000, 2499, 1),
        ("rrn26", 5000, 7600, -1),
    ]


def test_annotate_rrna_uses_native_before_external_tools(tmp_path: Path, monkeypatch):
    from organelleverse.annotation.mitochondrion import rrna, rrna_hmm, rrna_native

    fasta = tmp_path / "genome.fasta"
    _write_fasta(fasta, "mini", "ACGT" * 100)

    def fail_barrnap(*args, **kwargs):
        raise AssertionError("barrnap should not run before native rRNA")

    def fail_blastn(*args, **kwargs):
        raise AssertionError("BLASTN fallback should not run when native rRNA succeeds")

    monkeypatch.setattr(rrna, "_run_barrnap", fail_barrnap)
    monkeypatch.setattr(rrna, "_rrna_by_blastn", fail_blastn)
    # pyhmmer path finds nothing here; native reference-align tier should supply the hit.
    monkeypatch.setattr(rrna_hmm, "annotate_rrna_hmm", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        rrna_native,
        "annotate_rrna_native",
        lambda *args, **kwargs: [RawRRNA("rrn5", 10, 120, 1, 91.0, "5S", source_tool="Native")],
    )

    annotations = rrna.annotate_rrna(fasta, tmp_path / "out", db_manager=MiniDB(tmp_path))

    assert len(annotations) == 1
    assert annotations[0].gene_name == "rrn5"
    assert annotations[0].source_tool == "Native"
