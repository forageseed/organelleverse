from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.assembly.normalization import (
    FastaStats,
    GfaStats,
    normalize_fasta,
    normalize_gfa,
    parse_gene_markers,
    validate_fasta_against_gfa,
)
from organelleverse.core.errors import OrganelleExecutionError


def test_normalize_fasta_preserves_order_ids_descriptions_and_sequence(tmp_path: Path) -> None:
    source = tmp_path / "raw.fa"
    source.write_text(">ctg2 second\nacgtnryk\n>ctg1 first\nTTAA\n")
    destination = tmp_path / "assembly.fasta"

    stats = normalize_fasta(source, destination)

    assert stats.record_count == 2
    assert stats.total_bases == 12
    assert destination.read_text() == ">ctg2 second\nACGTNRYK\n>ctg1 first\nTTAA\n"


def test_normalize_fasta_wraps_at_80_columns(tmp_path: Path) -> None:
    source = tmp_path / "raw.fa"
    seq = "acgt" * 30  # 120 bases
    source.write_text(f">ctg1 long\n{seq}\n")
    destination = tmp_path / "assembly.fasta"

    stats = normalize_fasta(source, destination)

    lines = destination.read_text().rstrip("\n").split("\n")
    assert lines[0] == ">ctg1 long"
    body = lines[1:]
    assert all(len(line) <= 80 for line in body)
    assert stats.total_bases == 120


def test_normalize_fasta_rejects_duplicate_ids(tmp_path: Path) -> None:
    source = tmp_path / "raw.fa"
    source.write_text(">ctg1\nACGT\n>ctg1\nTTAA\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        normalize_fasta(source, tmp_path / "out.fa")
    assert raised.value.code == "assembly.parse_failed"


def test_normalize_fasta_rejects_empty_records(tmp_path: Path) -> None:
    source = tmp_path / "raw.fa"
    source.write_text(">ctg1\n\n>ctg2\nACGT\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        normalize_fasta(source, tmp_path / "out.fa")
    assert raised.value.code == "assembly.parse_failed"


def test_normalize_fasta_rejects_invalid_iupac(tmp_path: Path) -> None:
    source = tmp_path / "raw.fa"
    source.write_text(">ctg1\nACGTXYZ\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        normalize_fasta(source, tmp_path / "out.fa")
    assert raised.value.code == "assembly.parse_failed"


def test_normalize_fasta_preserves_iupac_ambiguity_uppercased(tmp_path: Path) -> None:
    source = tmp_path / "raw.fa"
    source.write_text(">ctg1\nacgtryswkmbdhvn\n")
    destination = tmp_path / "out.fa"
    stats = normalize_fasta(source, destination)
    assert stats.total_bases == 15
    assert "ACGTRYSWKMBDHVN" in destination.read_text()


def test_normalize_gfa_rejects_a_path_with_an_unknown_segment(tmp_path: Path) -> None:
    source = tmp_path / "raw.gfa"
    source.write_text("H\tVN:Z:1.0\nS\ts1\tACGT\nP\tp1\ts1+,missing+\t*\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        normalize_gfa(source, tmp_path / "assembly.gfa")
    assert raised.value.code == "assembly.validation_failed"


def test_normalize_gfa_normalizes_crlf_to_lf(tmp_path: Path) -> None:
    source = tmp_path / "raw.gfa"
    source.write_bytes(b"H\tVN:Z:1.0\r\nS\ts1\tACGT\r\n")
    destination = tmp_path / "assembly.gfa"
    normalize_gfa(source, destination)
    text = destination.read_text()
    assert "\r" not in text
    assert text.endswith("\n")


def test_normalize_gfa_rejects_link_with_unknown_segment(tmp_path: Path) -> None:
    source = tmp_path / "raw.gfa"
    source.write_text("H\tVN:Z:1.0\nS\ts1\tACGT\nL\ts1\t+\tmissing\t+\t0M\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        normalize_gfa(source, tmp_path / "out.gfa")
    assert raised.value.code == "assembly.validation_failed"


def test_normalize_gfa_requires_sequence_bearing_segments(tmp_path: Path) -> None:
    source = tmp_path / "raw.gfa"
    source.write_text("H\tVN:Z:1.0\nS\ts1\t*\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        normalize_gfa(source, tmp_path / "out.gfa")
    assert raised.value.code == "assembly.validation_failed"


def test_normalize_gfa_preserves_tags_and_record_order(tmp_path: Path) -> None:
    source = tmp_path / "raw.gfa"
    source.write_text("H\tVN:Z:1.0\nS\ts1\tACGT\tSN:Z:tag1\nS\ts2\tTTGG\nL\ts1\t+\ts2\t+\t2M\n")
    destination = tmp_path / "out.gfa"
    stats = normalize_gfa(source, destination)
    text = destination.read_text()
    assert "SN:Z:tag1" in text
    assert text.index("S\ts1") < text.index("S\ts2") < text.index("L\ts1")
    assert stats.segment_count == 2


def test_fasta_sequence_must_be_spellable_by_the_target_graph(tmp_path: Path) -> None:
    fasta = tmp_path / "assembly.fasta"
    gfa = tmp_path / "assembly.gfa"
    fasta.write_text(">ctg1\nAAACCG\n")
    gfa.write_text("H\tVN:Z:1.0\nS\ts1\tAAAC\nS\ts2\tACCG\nL\ts1\t+\ts2\t+\t2M\n")

    fasta_stats = normalize_fasta(fasta, tmp_path / "normalized.fasta")
    gfa_stats = normalize_gfa(gfa, tmp_path / "normalized.gfa")

    validate_fasta_against_gfa(fasta_stats, gfa_stats)

    disconnected = gfa.read_text().replace("L\ts1\t+\ts2\t+\t2M\n", "")
    gfa.write_text(disconnected)
    disconnected_stats = normalize_gfa(gfa, tmp_path / "disconnected.gfa")
    with pytest.raises(OrganelleExecutionError) as raised:
        validate_fasta_against_gfa(fasta_stats, disconnected_stats)
    assert raised.value.code == "assembly.validation_failed"


def test_graph_spelling_accepts_reverse_complement_orientation(tmp_path: Path) -> None:
    # forward walk AAACCG; its reverse complement CGGTTT must also be spellable
    # by walking the segments in their '-' orientation.
    gfa = tmp_path / "assembly.gfa"
    gfa.write_text("H\tVN:Z:1.0\nS\ts1\tAAAC\nS\ts2\tACCG\nL\ts1\t+\ts2\t+\t2M\n")
    gfa_stats = normalize_gfa(gfa, tmp_path / "n.gfa")
    fasta_rc = tmp_path / "rc.fasta"
    fasta_rc.write_text(">ctg1\nCGGTTT\n")  # reverse complement of AAACCG
    stats_rc = normalize_fasta(fasta_rc, tmp_path / "rc_n.fasta")
    validate_fasta_against_gfa(stats_rc, gfa_stats)


def test_graph_spelling_rejects_overlap_mismatch(tmp_path: Path) -> None:
    fasta = tmp_path / "assembly.fasta"
    gfa = tmp_path / "assembly.gfa"
    fasta.write_text(">ctg1\nAAACCG\n")
    gfa.write_text("H\tVN:Z:1.0\nS\ts1\tAAAC\nS\ts2\tTTGG\nL\ts1\t+\ts2\t+\t2M\n")
    fasta_stats = normalize_fasta(fasta, tmp_path / "n.fasta")
    gfa_stats = normalize_gfa(gfa, tmp_path / "n.gfa")
    with pytest.raises(OrganelleExecutionError) as raised:
        validate_fasta_against_gfa(fasta_stats, gfa_stats)
    assert raised.value.code == "assembly.validation_failed"


def test_graph_spelling_rejects_unrepresented_fasta_record(tmp_path: Path) -> None:
    fasta = tmp_path / "assembly.fasta"
    gfa = tmp_path / "assembly.gfa"
    fasta.write_text(">ctg1\nAAACCG\n>ctg2\nGGGGGG\n")
    gfa.write_text("H\tVN:Z:1.0\nS\ts1\tAAAC\nS\ts2\tACCG\nL\ts1\t+\ts2\t+\t2M\n")
    fasta_stats = normalize_fasta(fasta, tmp_path / "n.fasta")
    gfa_stats = normalize_gfa(gfa, tmp_path / "n.gfa")
    with pytest.raises(OrganelleExecutionError) as raised:
        validate_fasta_against_gfa(fasta_stats, gfa_stats)
    assert raised.value.code == "assembly.validation_failed"


def test_graph_spelling_terminates_on_cycle(tmp_path: Path) -> None:
    fasta = tmp_path / "assembly.fasta"
    gfa = tmp_path / "assembly.gfa"
    fasta.write_text(">ctg1\nACGTTT\n")
    gfa.write_text(
        "H\tVN:Z:1.0\nS\ts1\tACGT\nS\ts2\tTT\nL\ts1\t+\ts2\t+\t0M\nL\ts2\t+\ts2\t+\t0M\n"
    )
    fasta_stats = normalize_fasta(fasta, tmp_path / "n.fasta")
    gfa_stats = normalize_gfa(gfa, tmp_path / "n.gfa")
    validate_fasta_against_gfa(fasta_stats, gfa_stats)


def test_graph_spelling_accepts_circular_closure_with_seam_trim(tmp_path: Path) -> None:
    """A circular contig's reported sequence is one full lap, cut at an
    arbitrary segment boundary; the seam where the last segment's tail
    overlaps the first segment's head is trimmed from the reported FASTA
    rather than repeated. This mirrors real Oatk ``circular=true`` contigs,
    whose path may revisit a repeat segment and whose final hop overshoots
    the record length by exactly the closing link's declared overlap."""
    fasta = tmp_path / "assembly.fasta"
    gfa = tmp_path / "assembly.gfa"
    fasta.write_text(">ctg1\nAACCGTT\n")
    gfa.write_text(
        "H\tVN:Z:1.0\nS\ts1\tAACCG\nS\ts2\tCGTTA\nL\ts1\t+\ts2\t+\t2M\nL\ts2\t+\ts1\t+\t1M\n"
    )
    fasta_stats = normalize_fasta(fasta, tmp_path / "n.fasta")
    gfa_stats = normalize_gfa(gfa, tmp_path / "n.gfa")

    validate_fasta_against_gfa(fasta_stats, gfa_stats)


def test_graph_spelling_accepts_a_rotated_circular_start(tmp_path: Path) -> None:
    """A revisited (repeat) segment can make a clean offset-0 start
    impossible even though the contig is genuinely circular: real Oatk
    ``circular=true`` contigs (observed on the ddAraThal4 mitochondrion,
    where segment ``u2`` is traversed twice per loop) may report a FASTA
    that starts partway into the segment that closes the loop, not at that
    segment's own offset 0. The search must also try starting at the offset
    implied by each declared incoming link, not only offset 0."""
    fasta = tmp_path / "assembly.fasta"
    gfa = tmp_path / "assembly.gfa"
    fasta.write_text(">ctg1\nTGGTGTATTCAGCTTGGTGTATTCACTCGATTT\n")
    gfa.write_text(
        "H\tVN:Z:1.0\n"
        "S\tA\tTTGGTGTATTCA\n"
        "S\tB\tTCAGCTTG\n"
        "S\tC\tACTCGATTT\n"
        "L\tA\t+\tB\t+\t3M\n"
        "L\tB\t+\tA\t+\t3M\n"
        "L\tA\t+\tC\t+\t1M\n"
        "L\tC\t+\tA\t+\t1M\n"
    )
    fasta_stats = normalize_fasta(fasta, tmp_path / "n.fasta")
    gfa_stats = normalize_gfa(gfa, tmp_path / "n.gfa")

    validate_fasta_against_gfa(fasta_stats, gfa_stats)


def test_graph_spelling_rejects_a_false_circular_closure(tmp_path: Path) -> None:
    """The overshoot must be backed by a real declared link whose overlap
    content actually matches; a record that merely happens to be shorter
    than the linear walk must still fail if no such link closes it."""
    fasta = tmp_path / "assembly.fasta"
    gfa = tmp_path / "assembly.gfa"
    fasta.write_text(">ctg1\nAACCGTA\n")
    gfa.write_text("H\tVN:Z:1.0\nS\ts1\tAACCG\nS\ts2\tCGTTA\nL\ts1\t+\ts2\t+\t2M\n")
    fasta_stats = normalize_fasta(fasta, tmp_path / "n.fasta")
    gfa_stats = normalize_gfa(gfa, tmp_path / "n.gfa")
    with pytest.raises(OrganelleExecutionError) as raised:
        validate_fasta_against_gfa(fasta_stats, gfa_stats)
    assert raised.value.code == "assembly.validation_failed"


def test_parse_gene_markers_collects_case_insensitive_tokens(tmp_path: Path) -> None:
    bed = tmp_path / "anno.bed"
    bed.write_text(
        "ctg1\t0\t100\tCOX1\t.\t+\nctg1\t200\t300\tatp6\t.\t-\nctg1\t400\t500\tRbcl\t.\t+\n"
    )
    markers = parse_gene_markers(bed)
    assert markers == ("atp6", "cox1", "rbcl")


def test_parse_gene_markers_rejects_malformed_columns(tmp_path: Path) -> None:
    bed = tmp_path / "anno.bed"
    bed.write_text("ctg1\t0\tnotanint\tgeneA\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        parse_gene_markers(bed)
    assert raised.value.code == "assembly.parse_failed"


def test_parse_gene_markers_rejects_end_not_greater_than_start(tmp_path: Path) -> None:
    bed = tmp_path / "anno.bed"
    bed.write_text("ctg1\t100\t50\tgeneA\t.\t+\n")
    with pytest.raises(OrganelleExecutionError) as raised:
        parse_gene_markers(bed)
    assert raised.value.code == "assembly.parse_failed"


def test_stats_models_are_strict_frozen() -> None:
    assert FastaStats.model_config.get("frozen") is True
    assert FastaStats.model_config.get("extra") == "forbid"
    assert GfaStats.model_config.get("frozen") is True
    assert GfaStats.model_config.get("extra") == "forbid"
