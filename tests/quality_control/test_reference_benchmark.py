"""Benchmark reference denominator includes unaligned records, with separate unions."""

from organelleverse._bio import write_fasta
from organelleverse.assembly import graph_compare as benchmark


def test_linear_comparison_multiple_references_and_reverse_secondary(tmp_path, monkeypatch):
    ref = write_fasta(tmp_path / "r.fa", [("a", "A" * 10), ("b", "C" * 10), ("c", "G" * 10)])

    def run(argv):
        assert "--secondary=no" not in argv
        return (
            "q\t20\t0\t10\t+\ta\t10\t0\t10\t10\t10\t60\ttp:A:P\n"
            "q\t20\t10\t20\t-\tb\t10\t0\t10\t10\t10\t0\ttp:A:S\n"
        )

    monkeypatch.setattr(benchmark, "_run", run)
    result = benchmark.compare_linear(ref, tmp_path / "q.fa", "minimap2")
    assert result["gold_length"] == 30
    assert result["gold_aligned_fraction"] == 2 / 3
    assert result["alignment_identity"] == 1
    assert result["inverted_blocks"][0]["reference"] == "b"
    assert result["inverted_blocks"][0]["is_primary"] is False


def test_linear_no_alignment_keeps_reference_length(tmp_path, monkeypatch):
    ref = write_fasta(tmp_path / "r.fa", [("r", "ACGT")])
    monkeypatch.setattr(benchmark, "_run", lambda argv: "")
    result = benchmark.compare_linear(ref, tmp_path / "q.fa", "minimap2")
    assert result["gold_length"] == 4
    assert result["gold_aligned_fraction"] == 0
    assert result["alignment_identity"] is None


def test_large_indels_are_structural_and_gaps_count_once(tmp_path, monkeypatch):
    ref = write_fasta(tmp_path / "r.fa", [("r", "A" * 5000)])
    # 1000 '=' + 2 'X' + a 1500 bp deletion + 1 bp insertion + 997 '='
    row = "q\t2000\t0\t2000\t+\tr\t5000\t0\t3500\t1997\t3501\t60\ttp:A:P\tcg:Z:1000=2X1500D1I997=\n"
    monkeypatch.setattr(benchmark, "_run", lambda argv: row)
    result = benchmark.compare_linear(ref, tmp_path / "q.fa", "minimap2")
    (indel,) = result["large_indels"]
    assert indel["type"] == "deletion_in_candidate"
    assert indel["length"] == 1500 and indel["reference_pos"] == 1002
    assert result["gap_compressed_identity"] == 1997 / (1997 + 2 + 2)
    assert result["inverted_blocks"] == []
