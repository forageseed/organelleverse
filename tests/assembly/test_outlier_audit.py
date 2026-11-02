import importlib.util
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def counter(tmp_path, sequence, k):
    script = (
        Path(__file__).resolve().parents[2] / "scripts/assembly_benchmarks/audit_ovasm_outliers.py"
    )
    spec = importlib.util.spec_from_file_location("outlier_audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fasta = tmp_path / "sequence.fa"
    SeqIO.write(SeqRecord(Seq(sequence), id="test"), fasta, "fasta")
    return module.sequence_counts(fasta, [True], k)


def test_circular_counts_are_rotation_and_strand_invariant(tmp_path):
    sequence = "ACGTTTAGCGAC"
    expected = counter(tmp_path, sequence, 5)
    assert counter(tmp_path, sequence[4:] + sequence[:4], 5) == expected
    assert counter(tmp_path, str(Seq(sequence).reverse_complement()), 5) == expected


def test_copy_counts_detect_collapsed_repeat_with_identical_unique_content(tmp_path):
    one = counter(tmp_path, "ACGTAC", 5)
    two = counter(tmp_path, "ACGTACACGTAC", 5)
    assert one.keys() == two.keys()
    assert sum((two - one).values()) == 6


def test_tiny_circle_emits_one_word_per_genomic_start(tmp_path):
    assert sum(counter(tmp_path, "ACGT", 101).values()) == 4
