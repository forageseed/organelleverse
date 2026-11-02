import json

import pytest

from organelleverse.assembly.read_origins import (
    anchored_nuclear_read_contexts,
    assess_read_origins,
    candidate_and_flank,
    collect_read_origin_evidence,
    compare_periodic_context,
    extend_nuclear_context,
    extended_nuclear_contexts,
    nuclear_context_costs,
    read_blocks,
    realign_read_context,
)


def test_whole_query_realigns_a_misplaced_repeat_middle():
    read = "TTTTTACGTACGTGGGGG"
    fields = ["r", "18", "5", "13", "+", "chr", "22", "8", "16"]
    reference = "CC" + read + "AA"
    before = extend_nuclear_context(read, reference, fields, 8)
    assert compare_periodic_context(read, "ACGT", before["cost_upper_bound"]) == "periodic_compatible"
    after = realign_read_context(read, reference, fields, 8)
    assert after["cost_upper_bound"] == 0
    assert (after["reference_start"], after["reference_end"]) == (2, 20)
    assert compare_periodic_context(read, "ACGT", after["cost_upper_bound"]) == "nuclear_context_better"


@pytest.mark.parametrize("strand,read,a,b", [
    ("+", "TTTTTACGTACGT", 5, 13), ("-", "ACGTACGTAAAAA", 0, 8),
])
def test_one_available_flank_resolves_read_ending_in_repeat(tmp_path, strand, read, a, b):
    p = tmp_path / "p.paf"
    p.write_text(f"r\t13\t0\t13\t{strand}\tchr\t17\t2\t15\t13\t13\t0\tNM:i:0\tcg:Z:13M\n")
    hit = {"query_start": a, "query_end": b, "read_length": 13}
    context = extended_nuclear_contexts(
        p, {"r": hit}, {"chr": "assembled-molecule"}, {"r": read},
        {"chr": "CCTTTTTACGTACGTAA"},
    )["r"]["assigned"]
    assert context["cost_upper_bound"] == 0
    assert candidate_and_flank(context, hit)
    assert compare_periodic_context(read, "ACGT", 0) == "nuclear_context_better"


def test_inserted_flank_and_candidate_contained_read_do_not_supply_context():
    read = "TTTTTACGTACGT"
    fields = ["r", "13", "5", "13", "+", "chr", "8", "0", "8"]
    context = realign_read_context(read, "ACGTACGT", fields, 0)
    assert not candidate_and_flank(context, {"query_start": 5, "query_end": 13})
    assert not candidate_and_flank(context, {"query_start": 0, "query_end": 13})


def test_nuclear_read_haplotype_requires_donor_flank_and_retains_provenance(tmp_path):
    p = tmp_path / "read.paf"
    p.write_text(
        "r\t13\t0\t13\t+\td\t18\t0\t13\t13\t13\t0\tNM:i:0\tcg:Z:13M\n"
        "repeat\t8\t0\t8\t+\td\t18\t5\t13\t8\t8\t0\tNM:i:0\tcg:Z:8M\n"
    )
    proof = {"call": "nuclear_context_better", "nuclear_contexts": {"assigned": {"reference": "chr"}},
             "periodic_cost_at_nuclear_bound": None}
    args = (
        p, {"r": {"query_start": 5, "query_end": 13},
            "repeat": {"query_start": 2, "query_end": 6}},
        {"r": "TTTTTACGTACGT", "repeat": "ACGTACGT"},
        {"d": "TTTTTACGTACGTGGGGG"}, {"d": {"query_start": 5, "query_end": 13}},
        {"d": proof},
    )
    result = anchored_nuclear_read_contexts(*args)
    assert set(result) == {"r"}
    assert result["r"]["cost_upper_bound"] == 0
    assert result["r"]["donor_nuclear_contexts"] == proof["nuclear_contexts"]
    assert anchored_nuclear_read_contexts(*args, incumbent_costs={"r": 1}) == result
    # A zero-cost retained nuclear witness cannot be improved. The new search
    # returns no replacement; the caller retains its original valid witness.
    assert anchored_nuclear_read_contexts(*args, incumbent_costs={"r": 0}) == {}
    proof["call"] = "periodic_compatible"
    with pytest.raises(ValueError, match="lacks a nuclear-better"):
        anchored_nuclear_read_contexts(*args)


def test_observed_organelle_context_vetoes_nuclear_read_haplotype(tmp_path):
    read = "TTTTTACGTACGT"
    graph, sam, raw = (tmp_path / name for name in ("g.gfa", "c.sam", "raw.fastq"))
    graph.write_text("S\ts\tACGTACGT\nL\ts\t+\ts\t+\t4M\n")
    sam.write_text(f"r\t0\ts\t1\t60\t5S8M\t*\t0\t0\t{read}\t{'I'*13}\n")
    raw.write_text(f"@r\n{read}\n+\n{'I'*13}\n")
    args = dict(
        hits={"r": {"component": "s", "query_start": 5, "query_end": 13, "read_length": 13}},
        nuclear_roles={}, graph_periods={"s": "ACGT"}, original_reads=raw, scanned_reads=1,
        decisions_jsonl=tmp_path / "d.jsonl", out_json=tmp_path / "o.json",
        retained_nuclear_contexts={},
        nuclear_read_contexts={"r": {"cost_upper_bound": 0, "reference": "donor"}},
    )
    result = assess_read_origins(graph, sam, tmp_path / "unused.paf", **args)
    assert result["segments"][0]["nuclear_context_reads"] == 1
    protected = assess_read_origins(
        graph, sam, tmp_path / "unused.paf",
        organelle_contexts={"r": {"cost_upper_bound": 0, "reference": "marked-path"}}, **args,
    )
    assert protected["segments"][0]["unresolved_reads"] == 1
    assert protected["segments"][0]["nuclear_context_reads"] == 0
    decision = json.loads(args["decisions_jsonl"].read_text())
    assert decision["organelle_context"]["reference"] == "marked-path"


def test_exact_clipped_extensions_resolve_loose_periodic_bound_on_both_strands():
    read = "TTTTTACGTACGTGGGGG"
    reference = "CC" + read + "AA"
    plus = ["r", "18", "6", "12", "+", "chr", "22", "8", "14"]
    minus = ["r", "18", "6", "12", "-", "chr", "22", "8", "14"]
    assert compare_periodic_context(read, "ACGT", 12) == "periodic_compatible"
    for query, fields in [(read, plus), ("CCCCCACGTACGTAAAAA", minus)]:
        witness = extend_nuclear_context(query, reference, fields, 0)
        assert witness["cost_upper_bound"] == 0
        assert witness["unextended_cost_upper_bound"] == 12
        assert (witness["reference_start"], witness["reference_end"]) == (2, 20)
        assert compare_periodic_context(query, "ACGT", witness["cost_upper_bound"]) == (
            "nuclear_context_better"
        )


def test_extension_handles_reference_boundary_and_insertion_without_a_flank_window():
    fields = ["r", "8", "4", "8", "+", "chr", "4", "0", "4"]
    witness = extend_nuclear_context("TTTTACGT", "ACGT", fields, 0)
    assert witness["left_extension_edits"] == witness["cost_upper_bound"] == 4
    assert witness["reference_start"] == 0
    # Consuming > L reference bases must be possible when deletions are optimal.
    fields = ["r", "12", "0", "4", "+", "chr", "14", "0", "4"]
    witness = extend_nuclear_context("ACGTACGTGCTA", "ACGTACGTTTGCTA", fields, 0)
    assert witness["right_extension_edits"] == 2
    assert witness["reference_end"] == 14


def test_extended_context_retains_unplaced_tie_and_all_observed_secondary_placements(tmp_path):
    read = "TTTTTACGTACGTGGGGG"
    fields = "r\t18\t6\t12\t+\tchr\t22\t8\t14\t6\t6\t0\tNM:i:0\tcg:Z:6M\n"
    p = tmp_path / "p.paf"
    p.write_text(fields + fields.replace("\tchr\t", "\tscaffold\t"))
    contexts = extended_nuclear_contexts(
        p,
        {"r": {"query_start": 7, "query_end": 11, "read_length": 18}},
        {"chr": "assembled-molecule", "scaffold": "unplaced-scaffold"},
        {"r": read},
        {"chr": "CC" + read + "AA", "scaffold": "CC" + read + "AA"},
    )
    assert contexts["r"]["assigned"]["cost_upper_bound"] == 0
    assert contexts["r"]["unplaced"]["cost_upper_bound"] == 0


def test_periodic_competition_preserves_true_gene_free_circle_and_both_strands():
    assert compare_periodic_context("ACGTACGTACGT", "ACGT", 0) == "periodic_compatible"
    assert compare_periodic_context("CTCTCTCT", "AG", 0) == "periodic_compatible"
    assert compare_periodic_context("ACGTACGTACGT", "ACGT", 4) == "periodic_compatible"
    assert compare_periodic_context("ACGTACGT", "ACGT", None) == "unresolved"


def test_nuclear_flanks_win_against_the_entire_periodic_read_hypothesis():
    read = "TTTTTACGTACGTGGGGG"
    assert compare_periodic_context(read, "ACGT", 0) == "nuclear_context_better"
    assert compare_periodic_context(read, "ACGT", 12) == "periodic_compatible"


def test_reverse_cigar_insertion_is_not_reference_aligned():
    f = ["r", "30", "5", "25", "-"]
    assert read_blocks(f, "5M10I5M") == [(20, 25), (5, 10)]
    with pytest.raises(ValueError):
        read_blocks(f, "20M1I")
    with pytest.raises(ValueError):
        read_blocks(f, "20Mgarbage")


def test_both_flanks_required_and_unplaced_context_vetoes(tmp_path):
    p = tmp_path / "n.paf"
    h = {"r": {"query_start": 5, "query_end": 10, "read_length": 15}}
    prefix = "r\t15\t0\t15\t+\tchr\t100\t0\t15\t15\t15\t60\tNM:i:0\tcg:Z:15M\n"
    p.write_text(prefix)
    assert nuclear_context_costs(p, h, {"chr": "assembled-molecule"}) == {"r": 0}
    p.write_text(prefix + prefix.replace("\tchr\t", "\tscaffold\t"))
    assert (
        nuclear_context_costs(p, h, {"chr": "assembled-molecule", "scaffold": "unplaced-scaffold"})
        == {}
    )
    p.write_text(prefix.replace("15M", "5M5I5M"))
    assert nuclear_context_costs(p, h, {"chr": "assembled-molecule"}) == {}


def test_retained_decisions_materialize_only_aligned_base_coverage(tmp_path):
    graph = tmp_path / "g.gfa"
    graph.write_text("S\ts\tACGTACGTACGT\nL\ts\t+\ts\t+\t8M\n")
    sam = tmp_path / "reads.sam"
    sam.write_text("r\t0\ts\t1\t60\t2S4M2D4M2S\t*\t0\t0\tTTACGTACGTTT\tIIIIIIIIIIII\n")
    raw = tmp_path / "raw.fastq"
    raw.write_text("@r\nTTACGTACGTTT\n+\nIIIIIIIIIIII\n")
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps({"read": "r", "segment": "s", "call": "nuclear_context_better"}) + "\n"
    )
    result = collect_read_origin_evidence(
        graph,
        sam,
        decisions,
        original_reads=raw,
        scanned_reads=1,
        out_json=tmp_path / "origin.json",
    )
    assert result["segments"][0]["nuclear_covered_intervals"] == [(0, 4), (6, 10)]
    assert result["segments"][0]["sequence"] == "ACGTACGTACGT"
    assert result["original_reads"] == str(raw.resolve())
    sam.write_text(sam.read_text().replace("2S4M2D4M2S", "2H4M2D4M2S"))
    with pytest.raises(ValueError, match="complete"):
        collect_read_origin_evidence(
            graph,
            sam,
            decisions,
            original_reads=raw,
            scanned_reads=1,
            out_json=tmp_path / "origin.json",
        )


def test_complete_read_comparison_to_component_input(tmp_path):
    graph = tmp_path / "g.gfa"
    graph.write_text(
        "S\tnuclear\tACGTACGT\nL\tnuclear\t+\tnuclear\t+\t4M\n"
        "S\tcircle\tACGAACGA\nL\tcircle\t+\tcircle\t+\t4M\n"
    )
    raw = tmp_path / "raw.fastq"
    raw.write_text(
        "@n\nTTTTTACGTACGTGGGGG\n+\nIIIIIIIIIIIIIIIIII\n"
        "@c\nACGAACGAACGAACGAACGA\n+\nIIIIIIIIIIIIIIIIIIII\n"
    )
    sam = tmp_path / "candidate.sam"
    sam.write_text(
        "n\t0\tnuclear\t1\t60\t5S8M5S\t*\t0\t0\tTTTTTACGTACGTGGGGG\tIIIIIIIIIIIIIIIIII\n"
        "c\t0\tcircle\t1\t60\t8M12S\t*\t0\t0\tACGAACGAACGAACGAACGA\tIIIIIIIIIIIIIIIIIIII\n"
    )
    paf = tmp_path / "nuclear.paf"
    paf.write_text("n\t18\t0\t18\t+\tchr\t100\t0\t18\t18\t18\t60\tNM:i:0\tcg:Z:18M\n")
    hits = {
        "n": {"component": "nuclear", "query_start": 5, "query_end": 13, "read_length": 18},
        "c": {"component": "circle", "query_start": 0, "query_end": 8, "read_length": 20},
    }
    result = assess_read_origins(
        graph,
        sam,
        paf,
        hits=hits,
        nuclear_roles={"chr": "assembled-molecule"},
        graph_periods={"nuclear": "ACGT", "circle": "ACGA"},
        original_reads=raw,
        scanned_reads=2,
        decisions_jsonl=tmp_path / "decisions.jsonl",
        out_json=tmp_path / "origins.json",
    )
    by_name = {s["segment"]: s for s in result["segments"]}
    assert by_name["nuclear"]["nuclear_context_reads"] == 1
    assert by_name["nuclear"]["nuclear_covered_intervals"] == [(0, 8)]
    assert by_name["circle"]["unresolved_reads"] == 1


def test_assessment_uses_extended_context_and_preserves_unplaced_ambiguity(tmp_path):
    read = "TTTTTACGTACGTGGGGG"
    graph = tmp_path / "g.gfa"
    graph.write_text("S\ts\tACGTACGT\nL\ts\t+\ts\t+\t4M\n")
    raw = tmp_path / "raw.fastq"
    raw.write_text(f"@r\n{read}\n+\n{'I' * len(read)}\n")
    sam = tmp_path / "candidate.sam"
    sam.write_text(f"r\t0\ts\t1\t60\t5S8M5S\t*\t0\t0\t{read}\t{'I' * len(read)}\n")
    paf = tmp_path / "n.paf"
    seed = "r\t18\t4\t14\t+\tchr\t22\t6\t16\t10\t10\t0\tNM:i:0\tcg:Z:10M\n"
    paf.write_text(seed)
    args = dict(
        hits={"r": {"component": "s", "query_start": 5, "query_end": 13, "read_length": 18}},
        nuclear_roles={"chr": "assembled-molecule", "scaffold": "unplaced-scaffold"},
        graph_periods={"s": "ACGT"},
        original_reads=raw,
        scanned_reads=1,
        decisions_jsonl=tmp_path / "decisions.jsonl",
        out_json=tmp_path / "origins.json",
    )
    before = assess_read_origins(graph, sam, paf, **args)
    assert before["segments"][0]["periodic_compatible_reads"] == 1
    refs = {"chr": "CC" + read + "AA", "scaffold": "CC" + read + "AA"}
    after = assess_read_origins(graph, sam, paf, nuclear_sequences=refs, **args)
    assert after["segments"][0]["nuclear_context_reads"] == 1
    decision = json.loads(args["decisions_jsonl"].read_text())
    assert decision["nuclear_contexts"]["assigned"]["cost_upper_bound"] == 0
    assert decision["periodic_cost_at_nuclear_bound"] is None
    paf.write_text(seed + seed.replace("\tchr\t", "\tscaffold\t"))
    tied = assess_read_origins(graph, sam, paf, nuclear_sequences=refs, **args)
    assert tied["segments"][0]["unresolved_reads"] == 1
