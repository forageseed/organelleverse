"""Plant mitochondrial tRNA calls must look like whole mature tRNAs.

In seven RefSeq mitogenomes every annotated tRNA found was 70-88 nt and scored
>=37 bits; 21 of 51 extra calls were truncated fragments or weak/misaligned
hits (43-63 nt, <30 bits, or 120 nt) and are no longer reported.
"""

from __future__ import annotations

from pathlib import Path

from organelleverse.annotation.cmsearch import genome as cm_genome
from organelleverse.annotation.cmsearch import naming
from organelleverse.annotation.cmsearch.models import CMHit
from organelleverse.annotation.mitochondrion.trna_native import annotate_trna_cm


def test_fragments_long_and_weak_hits_are_not_reported(tmp_path: Path, monkeypatch) -> None:
    fasta = tmp_path / "mito.fasta"
    fasta.write_text(">m\n" + "ACGT" * 500 + "\n")
    hits = [
        CMHit(start=100, end=173, strand=1, score=45.0),  # whole tRNA: kept
        CMHit(start=300, end=354, strand=1, score=40.0),  # 55 nt fragment
        CMHit(start=500, end=619, strand=-1, score=40.0),  # 120 nt: not one tRNA
        CMHit(start=800, end=873, strand=1, score=25.0),  # weak
    ]
    monkeypatch.setattr(cm_genome, "find_trnas", lambda *args, **kwargs: hits)
    monkeypatch.setattr(naming, "name_trna", lambda model, seq: ("GTA", "Y", "trnY(gta)"))

    calls = annotate_trna_cm(fasta, tmp_path / "out")

    assert [(c.start, c.end) for c in calls] == [(100, 173)]


def test_trna_products_use_ncbi_three_letter_names() -> None:
    from organelleverse.annotation.mitochondrion.trna import trna_product

    assert trna_product("S") == "tRNA-Ser"
    assert trna_product("fM") == "tRNA-fMet"
    assert trna_product("Ile") == "tRNA-Ile"
    assert trna_product("") == "tRNA-Xxx"
    assert trna_product("?") == "tRNA-Xxx"
