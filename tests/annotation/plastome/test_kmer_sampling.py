"""The fast k-mer sampling and sequence cleaning return exactly what the original per-k-mer implementation did."""

import random
import zlib
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq

from organelleverse.annotation.plastome import references as refs


def _old_clean(seq: str) -> str:
    return "".join(base if base in "ACGT" else "N" for base in seq.upper().replace("U", "T") if not base.isspace())


def _old_sample(seq: str) -> set[str]:
    def canonical(kmer):
        rc = str(Seq(kmer).reverse_complement())
        return kmer if kmer <= rc else rc

    clean = _old_clean(seq)
    if len(clean) < 31:
        return {clean}
    circular = clean + clean[:30]
    ranked = sorted(
        (zlib.crc32(canonical(circular[i : i + 31]).encode("ascii")), canonical(circular[i : i + 31]))
        for i in range(len(clean))
        if "N" not in circular[i : i + 31]
    )
    if not ranked:
        return {clean[i : i + 31] for i in range(0, len(clean) - 30)}
    return {kmer for _, kmer in ranked[:1200]}


def _random_seq(rng: random.Random, n: int) -> str:
    alphabet = "ACGT" * 20 + "acgtNnRYKMuU \n\t-"
    return "".join(rng.choice(alphabet) for _ in range(n))


@pytest.mark.parametrize("n", [0, 5, 30, 31, 32, 200, 5000, 40000])
def test_random_sequences_match_the_original(n: int) -> None:
    rng = random.Random(n)
    for _ in range(5):
        s = _random_seq(rng, n)
        assert refs.clean_seq(s) == _old_clean(s)
        assert refs.sample_kmers(s) == _old_sample(s)


def test_all_n_and_short_sequences() -> None:
    for s in ("N" * 100, "ACGTN" * 3, "", "acgu"):
        assert refs.sample_kmers(s) == _old_sample(s)


def test_reference_choice_reuses_the_parsed_kmers() -> None:
    data = Path(refs.__file__).resolve().parents[1] / "data/plastome/references"
    files = sorted(data.glob("*.gb"))[:4]
    for reference in refs.load_references(files):
        stat = reference.path.stat()
        reread = refs._reference_choice_kmers_cached(str(reference.path.resolve()), stat.st_size, stat.st_mtime_ns)
        assert refs._reference_choice_kmers(reference) == reread, reference.path.name


def test_shipped_references_match_the_original() -> None:
    data = Path(refs.__file__).resolve().parents[1] / "data/plastome/references"
    files = sorted(data.glob("*.gb"))[:6]
    assert files
    for gb in files:
        seq = str(SeqIO.read(gb, "genbank").seq)
        assert refs.sample_kmers(seq) == _old_sample(seq), gb.name
