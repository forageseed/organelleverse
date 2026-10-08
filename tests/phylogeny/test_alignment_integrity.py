"""Regression cases for false-success alignments and backend failures."""

from subprocess import CompletedProcess

import pytest

from organelleverse import accel
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.phylogeny import phylo
from organelleverse.phylogeny.phylo_core import compute_alignment


@pytest.fixture
def fasta(tmp_path):
    p = tmp_path / "indel.fa"
    p.write_text(">a\nACGT\n>b\nACGGT\n")
    return p


def test_missing_forced_backend_is_failure(fasta, monkeypatch):
    monkeypatch.setattr(phylo, "_which", lambda _: None)
    monkeypatch.setattr(accel, "mafft_align", None)
    for method in ("auto", "mafft", "rust"):
        result = phylo.align(fasta, method=method)
        assert result.status == "failed"
        assert result.errors[0].code == "phylogeny.align.backend_missing"
        assert "alignment" not in result.metrics
    with pytest.raises(OrganelleExecutionError, match="No alignment backend"):
        compute_alignment(fasta)


def test_failed_cli_preserves_stderr_without_success_fallback(fasta, monkeypatch):
    monkeypatch.setattr(phylo, "_which", lambda _: "/mafft")

    def fail(*args, **kwargs):
        raise OrganelleExecutionError(
            code="external_command_failed",
            message="MAFFT failed: invalid input",
            details={"stderr_tail": "invalid input", "returncode": 2},
        )

    monkeypatch.setattr(phylo, "run_external", fail)
    result = phylo.align(fasta, method="mafft")
    assert result.status == "failed"
    assert result.errors[0].details["stderr_tail"] == "invalid input"
    assert "alignment" not in result.metrics


def test_failed_rust_does_not_silently_change_backends(fasta, monkeypatch):
    def fail(*args):
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(accel, "mafft_align", fail)
    result = phylo.align(fasta)
    assert result.status == "failed"
    assert "kernel failed" in result.errors[0].message
    assert result.provenance.actual_backend == "rust_mafft"


@pytest.mark.parametrize(
    "output",
    [
        "",
        ">a\nACGT\n>b\nACGGT\n",
        ">a\nACG-T\n>b\nACGAT\n",
        ">a\nACG-T\n>c\nACGGT\n",
    ],
)
def test_rejects_incomplete_unequal_or_changed_alignment(fasta, monkeypatch, output):
    monkeypatch.setattr(phylo, "_which", lambda _: "/mafft")
    monkeypatch.setattr(phylo, "run_external", lambda *a, **kw: CompletedProcess(a, 0, output, ""))
    result = phylo.align(fasta, method="mafft")
    assert result.status == "failed"
    assert result.errors[0].code == "phylogeny.align.invalid_output"


def test_real_alignment_indel_and_typed_core(tmp_path):
    if accel.mafft_align is None and phylo._which("mafft") is None:
        pytest.skip("A real MAFFT kernel or CLI is required")
    p = tmp_path / "indel.fa"
    p.write_text(">a\nACGTTGCACTAGGATCCGTA\n>b\nACGTTGCACTAGGGATCCGTA\n")
    result = phylo.align(p)
    assert result.status == "ok"
    aligned = dict(result.metrics["alignment"])
    assert len(aligned["a"]) == len(aligned["b"])
    assert "-" in aligned["a"]
    assert aligned["a"].replace("-", "") == "ACGTTGCACTAGGATCCGTA"
    core = compute_alignment(p)
    assert core["method"] == result.metrics["method"]
    assert core["aligned_length"] == result.metrics["aligned_length"]


def test_rust_fft_anchors_preserve_repetitive_sequences():
    if accel.mafft_align is None:
        pytest.skip("Rust MAFFT kernel is required")
    seed = "ATGCGCGTATTAACCTAAGGTTAGCCTAAGGCATCGATCGGATT"
    reference = (seed * (600 // len(seed) + 1))[:600]
    sequences = [
        ("ref", reference),
        ("alt1", reference[:100] + "C" + reference[101:300] + "G" + reference[301:]),
        ("alt2", reference[:250] + "T" + reference[251:400] + "A" + reference[401:]),
    ]
    aligned = dict(accel.mafft_align(sequences, "auto"))
    assert len({len(s) for s in aligned.values()}) == 1
    for name, original in sequences:
        assert aligned[name].replace("-", "") == original
