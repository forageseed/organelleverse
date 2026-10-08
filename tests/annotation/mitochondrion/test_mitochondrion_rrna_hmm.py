from __future__ import annotations

import random
from pathlib import Path

from Bio import SeqIO

from organelleverse.annotation.mitochondrion.db import DBManager
from organelleverse.annotation.mitochondrion.rrna_hmm import annotate_rrna_hmm


def test_rrna_hmm_no_crash_on_random_sequence(tmp_path: Path):
    db = DBManager()
    fasta = tmp_path / "g.fasta"
    fasta.write_text(">seq\n" + ("ACGT" * 500) + "\n")
    hits = annotate_rrna_hmm(fasta, tmp_path, db)
    assert isinstance(hits, list)


def test_rrna_hmm_finds_embedded_reference_with_exact_coordinates(tmp_path: Path):
    db = DBManager()
    ref_file = sorted(db.rrna_ref_dir.glob("*.fasta"))[0]  # rrn18 SSU
    ref = str(next(SeqIO.parse(str(ref_file), "fasta")).seq).upper().replace("U", "T")

    random.seed(7)
    left = "".join(random.choice("ACGT") for _ in range(300))
    right = "".join(random.choice("ACGT") for _ in range(120))
    genome = left + ref + right
    fasta = tmp_path / "g.fasta"
    fasta.write_text(">genome\n" + genome + "\n")

    hits = annotate_rrna_hmm(fasta, tmp_path, db)
    assert hits, "expected at least one rRNA hit for the embedded reference"
    best = max(hits, key=lambda h: h.score)
    assert best.start == 301
    assert best.end == 300 + len(ref)
    assert best.strand == 1
    assert best.rrna_type == "18S"
    assert best.gene_name.startswith("rrn18")
    assert best.source_tool == "pyhmmer-nhmmer"


def test_rrna_hmm_finds_reverse_strand_reference(tmp_path: Path):
    db = DBManager()
    ref_file = sorted(db.rrna_ref_dir.glob("*.fasta"))[0]
    ref = str(next(SeqIO.parse(str(ref_file), "fasta")).seq).upper().replace("U", "T")
    rc = ref.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    random.seed(11)
    left = "".join(random.choice("ACGT") for _ in range(150))
    genome = left + rc + "".join(random.choice("ACGT") for _ in range(90))
    fasta = tmp_path / "g.fasta"
    fasta.write_text(">genome\n" + genome + "\n")

    hits = annotate_rrna_hmm(fasta, tmp_path, db)
    assert hits
    best = max(hits, key=lambda h: h.score)
    assert best.strand == -1
    assert best.start == 151
    assert best.end == 150 + len(ref)
