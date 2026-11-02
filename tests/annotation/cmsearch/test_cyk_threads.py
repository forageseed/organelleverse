"""Threaded covariance-model search returns exactly the serial result."""

from pathlib import Path

from Bio import SeqIO

from organelleverse.annotation.cmsearch.genome import find_trnas
from organelleverse.annotation.cmsearch.parallel import cyk_threads, pmap


def test_pmap_keeps_order_and_falls_back_to_serial() -> None:
    assert pmap(lambda x: x * x, range(50)) == [x * x for x in range(50)]
    with cyk_threads(4):
        assert pmap(lambda x: x * x, range(50)) == [x * x for x in range(50)]
        assert pmap(lambda x: x, []) == []


def test_threaded_trna_search_equals_serial() -> None:
    data = Path(__file__).resolve().parents[3] / "src/annotation/data/plastome/references"
    gb = sorted(data.glob("*.gb"))[0]
    seq = str(SeqIO.read(gb, "genbank").seq)[:40000]  # trnK-UUU (intron tRNA) sits near the start of most plastomes
    serial = find_trnas(seq, recover_intron_pairs=True)
    with cyk_threads(4):
        threaded = find_trnas(seq, recover_intron_pairs=True)
    assert serial
    assert threaded == serial
