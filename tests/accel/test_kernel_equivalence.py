"""Equivalence fixtures pinning the Rust accel kernels to their references.

Each test compares a Rust kernel against the pure-Python / scipy reference it
replaces, following the cm_cyk precedent in
``tests/annotation/cmsearch/test_cmsearch_rust.py``. A kernel that passes here
is promoted to ``VERIFIED_KERNELS`` and becomes a default backend.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess

import pytest

import organelleverse.accel as accel
from organelleverse.transfer import transfer as transfer_mod

# The kernels under test are gated behind the opt-in env var until they pass
# here; promote them (accel.VERIFIED_KERNELS) once the fixtures are green.
pytestmark = pytest.mark.skipif(
    accel.kmer_overlap is None or accel.mafft_align is None or accel.erc_correlation is None,
    reason="unverified kernels not enabled in this process (ORGANELLEVERSE_ACCEL_UNVERIFIED)",
)

_MAFFT = shutil.which("mafft")


def _random_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


# ---------------------------------------------------------------------------
# kmer_overlap  (MTPT/NUMT fragment detection — transfer module)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_kmer_overlap_matches_python_scan(seed: int) -> None:
    rng = random.Random(seed)
    source = _random_seq(rng, rng.randint(120, 400))
    # Splice pieces of the source into a random target so real overlaps exist.
    target = _random_seq(rng, 200)
    for _ in range(rng.randint(1, 4)):
        start = rng.randrange(0, len(source) - 60)
        length = rng.randint(25, 80)
        pos = rng.randrange(0, len(target))
        target = target[:pos] + source[start : start + length] + target[pos:]

    index = transfer_mod._build_stranded_kmer_index(source, 15)
    for min_len in (25, 40):
        python_runs = transfer_mod._scan_kmer_runs(target, index, 15, min_len)
        rust_fragments = accel.kmer_overlap(target, source, 15, min_len)
        assert sorted((int(s), int(e)) for s, e in rust_fragments) == sorted(
            (run["start"], run["end"]) for run in python_runs
        ), f"seed={seed} min_len={min_len}"


# ---------------------------------------------------------------------------
# mafft_align  (phylogeny / codeml alignment)
# ---------------------------------------------------------------------------


def _mafft_cli(seqs: list[tuple[str, str]], strategy: str) -> list[tuple[str, str]]:
    """Run the mafft CLI reference. "fft-ns-2" is mafft's default strategy —
    it has no dedicated flag (unlike l-INS-i's --localpair)."""
    fasta_in = "".join(f">{name}\n{seq}\n" for name, seq in seqs)
    argv = ["mafft", "--auto", "-"] if strategy == "auto" else ["mafft", "-"]
    proc = subprocess.run(argv, input=fasta_in, capture_output=True, text=True, check=True)
    names: list[str] = []
    chunks: dict[str, list[str]] = {}
    for line in proc.stdout.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            names.append(name)
            chunks[name] = []
        elif names:
            chunks[names[-1]].append(line.strip().upper())
    return [(name, "".join(chunks[name])) for name in names]


def _score_alignment(seqs: list[tuple[str, str]]) -> int:
    """Match columns across all rows — comparable even if gap placement differs."""
    columns = zip(*(seq for _, seq in seqs))
    return sum(1 for column in columns if len(set(column)) == 1)


@pytest.mark.skipif(_MAFFT is None, reason="mafft CLI not installed")
@pytest.mark.parametrize("strategy", ["auto", "fft-ns-2"])
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_mafft_align_matches_mafft_cli(seed: int, strategy: str) -> None:
    rng = random.Random(seed)
    ancestor = _random_seq(rng, 240)
    seqs: list[tuple[str, str]] = []
    for i in range(4):
        mutated = list(ancestor)
        for _ in range(rng.randint(6, 18)):
            pos = rng.randrange(len(mutated))
            mutated[pos] = rng.choice("ACGT")
        seqs.append((f"s{i}", "".join(mutated)))

    rust = accel.mafft_align(seqs, strategy)
    reference = _mafft_cli(seqs, strategy)

    assert [name for name, _ in rust] == [name for name, _ in reference]
    assert len({len(aligned) for _, aligned in rust}) == 1  # aligned to one width
    rust_score = _score_alignment(rust)
    ref_score = _score_alignment(reference)
    # Identical match-column score; gap placement may legitimately differ.
    assert rust_score == ref_score, (
        f"seed={seed} strategy={strategy}: rust={rust_score} mafft={ref_score}"
    )


@pytest.mark.skipif(_MAFFT is None, reason="mafft CLI not installed")
def test_mafft_align_preserves_sequence_content() -> None:
    seqs = [("a", "ACGTACGTACGTTTTTACGTAC"), ("b", "ACGTTCGTACGTTTTTACGTACG")]
    aligned = dict(accel.mafft_align(seqs, "auto"))
    for name, original in seqs:
        assert aligned[name].replace("-", "") == original


# ---------------------------------------------------------------------------
# erc_correlation  (coevolution rate covariation)
# ---------------------------------------------------------------------------


def test_erc_correlation_matches_scipy() -> None:
    import numpy as np
    from scipy import stats as st

    rng = random.Random(7)
    genes = [f"g{i}" for i in range(6)]
    vectors = [[round(rng.uniform(0.01, 0.5), 4) for _ in range(10)] for _ in genes]

    pairs = accel.erc_correlation(genes, vectors, "pearson", "none", 5, 0.05)
    assert len(pairs) == 15  # C(6,2)

    by_pair = {(p["gene_a"], p["gene_b"]): p for p in pairs}
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            pair = by_pair[(genes[i], genes[j])]
            a = np.array(vectors[i])
            b = np.array(vectors[j])
            expected_r = st.pearsonr(a, b)
            expected_rho = st.spearmanr(a, b)
            expected_tau = st.kendalltau(a, b)
            assert abs(pair["r"] - expected_r.statistic) < 1e-3
            assert abs(pair["rho"] - expected_rho.statistic) < 1e-3
            assert abs(pair["tau"] - expected_tau.statistic) < 1e-3
            assert abs(pair["pval"] - expected_r.pvalue) < 1e-4
            assert pair["n_branches"] == 10
            assert pair["significant"] == (pair["fdr_pval"] < 0.05)


def test_erc_correlation_min_overlap_drops_short() -> None:
    genes = ["g1", "g2"]
    vectors = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]
    pairs = accel.erc_correlation(genes, vectors, "pearson", "none", 5, 0.05)
    assert pairs == []  # 3 branches < min_overlap 5


# ---------------------------------------------------------------------------
# kmer_jaccard — strand-specific, pinned to the barcode reference
# ---------------------------------------------------------------------------


def test_kmer_jaccard_strand_specific_edge() -> None:
    """The historical divergence, now fixed: the kernel must NOT fold reverse
    complements (that is kmer_overlap's job for MTPT/NUMT detection); the
    barcode reference is strand-specific."""
    if accel.kmer_jaccard is None:
        pytest.skip("rust kernels unavailable in this process")
    assert accel.kmer_jaccard("AAAA", "TTTT", 3) == 0.0
    assert accel.kmer_jaccard("AAAA", "AAAA", 3) == 1.0
    assert accel.kmer_jaccard("aaaa", "AAAA", 3) == 1.0  # upper-cased


def test_kmer_jaccard_matches_barcode_reference() -> None:
    """Equivalence pin: the Rust kernel reproduces the Python barcode
    Jaccard exactly (forward k-mers, N-windows skipped, upper-cased)."""
    if accel.kmer_jaccard is None:
        pytest.skip("rust kernels unavailable in this process")
    import random

    def py_jac(a: str, b: str, k: int) -> float:
        a, b = a.upper(), b.upper()
        A = {a[i : i + k] for i in range(len(a) - k + 1) if "N" not in a[i : i + k]}
        B = {b[i : i + k] for i in range(len(b) - k + 1) if "N" not in b[i : i + k]}
        return len(A & B) / len(A | B) if A | B else 0.0

    rng = random.Random(7)
    for _ in range(500):
        length = rng.randint(20, 300)
        k = rng.randint(2, 7)
        a = "".join(rng.choice("ACGTN") for _ in range(length))
        b = "".join(rng.choice("ACGTN") for _ in range(length))
        assert accel.kmer_jaccard(a, b, k) == py_jac(a, b, k)
