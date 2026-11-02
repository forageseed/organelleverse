import pytest
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature

from organelleverse.annotation.plastome.genomic_product_projection import (
    compare_genomic_products,
    genomic_alignment_from_paf,
)


def feature(*parts, phase=1):
    locations = [FeatureLocation(a, b, strand=s) for a, b, s in parts]
    loc = locations[0] if len(locations) == 1 else CompoundLocation(locations)
    return SeqFeature(loc, type="CDS", qualifiers={"codon_start": [str(phase)]})


def paf(chain, qs, qe, ts, te, strand="+", cigar=None):
    return genomic_alignment_from_paf(
        f"source\t1000\t{qs}\t{qe}\t{strand}\ttarget\t1000\t{ts}\t{te}\t0\t0\t0\tcg:Z:{cigar or str(qe - qs) + 'M'}",
        chain,
    )


def compare(source, target, chains, pairs):
    return compare_genomic_products(
        "source-product",
        source,
        "target-copy",
        target,
        tuple(chains),
        source_record="source",
        target_record="target",
        part_pairs=pairs,
    )


def statuses(result):
    return [s.status for p in result.parts for s in p.states]


def test_product_can_be_covered_by_separate_trans_blocks():
    source = feature((0, 6, 1), (100, 106, -1), phase=2)
    target = feature((20, 26, 1), (200, 206, -1), phase=2)
    result = compare(
        source,
        target,
        [paf("first", 0, 6, 20, 26), paf("second", 100, 106, 200, 206)],
        ((0, 0), (1, 1)),
    )
    assert statuses(result) == ["single_agreeing_requested_part_mapping"] * 2
    assert [p.source_coding_interval for p in result.parts] == [(-1, 5), (5, 11)]
    assert not any(p.target_uncovered_intervals for p in result.parts)
    assert {m.chain_id for p in result.parts for s in p.states for m in s.mappings} == {
        "first",
        "second",
    }


def test_split_chain_cis_domain_keeps_an_actual_insertion_gap():
    result = compare(
        feature((10, 20, 1)),
        feature((100, 111, 1)),
        [paf("indels", 10, 20, 100, 111, cigar="3M2I2M3D3M")],
        ((0, 0),),
    )
    assert (13, 15) in [s.source_interval for s in result.parts[0].states if s.status == "unmapped"]
    assert result.parts[0].target_uncovered_intervals == ((105, 108),)


def test_reverse_block_with_negative_feature_projects_correctly():
    result = compare(
        feature((12, 16, -1)),
        feature((104, 108, 1)),
        [paf("reverse", 10, 20, 100, 110, "-")],
        ((0, 0),),
    )
    assert statuses(result) == ["single_agreeing_requested_part_mapping"]
    assert result.parts[0].states[0].mappings[0].target_interval == (104, 108)


def test_ir_outside_alternative_is_preserved_for_requested_copy():
    result = compare(
        feature((0, 6, 1)),
        feature((20, 26, 1)),
        [paf("copyA", 0, 6, 20, 26), paf("copyB", 0, 6, 200, 206, "-")],
        ((0, 0),),
    )
    assert statuses(result) == ["single_agreeing_requested_part_mapping"]
    mappings = result.parts[0].states[0].mappings
    assert [(m.chain_id, m.inside_requested_part) for m in mappings] == [
        ("copyA", True),
        ("copyB", False),
    ]


def test_distinct_overlapping_chains_do_not_fabricate_unique_coordinates():
    result = compare(
        feature((0, 6, 1)),
        feature((20, 30, 1)),
        [paf("A", 0, 6, 20, 26), paf("B", 0, 6, 21, 27)],
        ((0, 0),),
    )
    assert statuses(result) == ["ambiguous_requested_part_mapping"]
    assert len(result.parts[0].states[0].mappings) == 2


def test_duplicate_chain_ids_are_evidence_not_extra_distinct_mappings():
    result = compare(
        feature((0, 6, 1)),
        feature((20, 26, 1)),
        [paf("A", 0, 6, 20, 26), paf("B", 0, 6, 20, 26)],
        ((0, 0),),
    )
    assert statuses(result) == ["single_agreeing_requested_part_mapping"]
    assert len(result.parts[0].states[0].mappings) == 2


def test_opposed_first_part_is_not_hidden_by_matching_second_part():
    result = compare(
        feature((0, 6, 1), (10, 16, 1)),
        feature((20, 26, -1), (30, 36, 1)),
        [paf("A", 0, 6, 20, 26), paf("B", 10, 16, 30, 36)],
        ((0, 0), (1, 1)),
    )
    assert statuses(result) == [
        "opposed_requested_part_mapping",
        "single_agreeing_requested_part_mapping",
    ]


def test_origin_split_biological_parts_can_map_to_one_physical_domain():
    result = compare(
        feature((994, 1000, 1), (0, 6, 1)),
        feature((20, 32, 1)),
        [paf("end", 994, 1000, 20, 26), paf("start", 0, 6, 26, 32)],
        ((0, 0), (1, 0)),
    )
    assert statuses(result) == ["single_agreeing_requested_part_mapping"] * 2
    assert [p.source_coding_interval for p in result.parts] == [(0, 6), (6, 12)]
    # Coverage is explicitly per part-pair, not inferred merged exon boundaries.
    assert result.parts[0].target_uncovered_intervals == ((26, 32),)


def test_partial_target_overlap_splits_mapped_and_outside_domains():
    result = compare(
        feature((0, 10, 1)), feature((102, 108, 1)), [paf("A", 0, 10, 100, 110)], ((0, 0),)
    )
    assert statuses(result) == [
        "outside_requested_part",
        "single_agreeing_requested_part_mapping",
        "outside_requested_part",
    ]
    assert [s.source_interval for s in result.parts[0].states] == [(0, 2), (2, 8), (8, 10)]


def test_unpaired_and_absent_parts_are_explicit():
    result = compare(
        feature((0, 6, 1), (10, 16, 1)), feature((20, 26, 1), (30, 36, 1)), [], ((0, 0),)
    )
    assert statuses(result) == ["unmapped"]
    assert result.unpaired_source_parts == result.unpaired_target_parts == (1,)


def test_invalid_cigar_consumption_fails_without_fabricated_mapping():
    with pytest.raises(ValueError, match="consume"):
        paf("bad", 0, 10, 20, 30, cigar="9M")
