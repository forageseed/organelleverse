"""Short duplicate copies of a gene that is also annotated in full are dropped; lone pieces are kept."""

from organelleverse.annotation.mitochondrion.fragment_filter import drop_duplicate_fragments
from organelleverse.annotation.mitochondrion.models.gene import ExonRecord, GeneAnnotation, Strand


def cds(name, *parts, partial3=False):
    return GeneAnnotation(
        gene_name=name,
        gene_type="CDS",
        strand=Strand.PLUS,
        is_partial_3prime=partial3,
        exons=[
            ExonRecord(start=s, end=e, strand=Strand.PLUS, number=i + 1)
            for i, (s, e) in enumerate(parts)
        ],
    )


REF = {"ccmb": 207.0, "nad1": 325.0}


def test_short_copy_of_gene_with_full_copy_is_dropped():
    full = cds("ccmB", (1000, 1624))  # 207 aa
    piece = cds("ccmB", (50000, 50122))  # 40 aa elsewhere
    kept, dropped = drop_duplicate_fragments([full, piece], REF)
    assert kept == [full] and dropped == [piece]


def test_lone_short_piece_is_kept():
    # a trans-spliced segment the assembly could not join: no full-length nad1 exists
    pieces = [cds("nad1", (1000, 1384)), cds("nad1", (90000, 90300))]
    kept, dropped = drop_duplicate_fragments(pieces, REF)
    assert kept == pieces and dropped == []


def test_overlapping_short_model_and_partial_copy_are_kept():
    full = cds("ccmB", (1000, 1624))
    inside = cds("ccmB", (1100, 1222))  # shares bases with the full copy
    edge = cds("ccmB", (70000, 70120), partial3=True)  # cut by a sequence end
    kept, dropped = drop_duplicate_fragments([full, inside, edge], REF)
    assert dropped == [] and len(kept) == 3


def test_genes_without_reference_and_rna_genes_pass():
    trna = GeneAnnotation(
        gene_name="trnK-UUU",
        gene_type="tRNA",
        strand=Strand.PLUS,
        exons=[ExonRecord(start=5, end=77, strand=Strand.PLUS)],
    )
    orf = cds("orf25", (10, 90))
    kept, dropped = drop_duplicate_fragments([trna, orf], REF)
    assert kept == [trna, orf] and dropped == []


FAMILY_REF = {"ccmfc": 450.0, "ccmfc1": 355.0, "ccmfc2": 353.0, "rpl10": 160.0, "nad1": 325.0}
SPLICED = frozenset({"nad1", "ccmfc"})


def test_short_split_gene_family_member_is_dropped_when_the_gene_is_full():
    full = cds("ccmFC", (1000, 2350))  # 450 aa
    piece = cds("ccmFC1", (60000, 60160))  # 53 aa, own reference 355 aa
    kept, dropped = drop_duplicate_fragments([full, piece], FAMILY_REF, SPLICED)
    assert kept == [full] and dropped == [piece]


def test_a_family_member_is_kept_when_no_member_is_full():
    piece = cds("ccmFC1", (60000, 60160))
    kept, dropped = drop_duplicate_fragments([piece], FAMILY_REF, SPLICED)
    assert kept == [piece] and dropped == []


def test_short_single_exon_remnant_of_a_single_exon_gene_is_dropped():
    remnant = cds("rpl10", (5000, 5120))  # 40 aa against 160
    full_elsewhere = cds("nad1", (1000, 1980))
    _kept, dropped = drop_duplicate_fragments([remnant, full_elsewhere], FAMILY_REF, SPLICED)
    assert dropped == [remnant]


def test_short_single_exon_piece_of_a_spliced_gene_is_kept():
    piece = cds("nad1", (90000, 90300))  # one exon of a multi-exon gene
    kept, dropped = drop_duplicate_fragments([piece], FAMILY_REF, SPLICED)
    assert kept == [piece] and dropped == []


def test_the_remnant_rule_needs_the_exon_references():
    remnant = cds("rpl10", (5000, 5120))
    kept, dropped = drop_duplicate_fragments([remnant], FAMILY_REF)
    assert kept == [remnant] and dropped == []


def test_a_remnant_cut_by_a_sequence_end_is_kept():
    edge = cds("rpl10", (5000, 5120), partial3=True)
    kept, dropped = drop_duplicate_fragments([edge], FAMILY_REF, SPLICED)
    assert kept == [edge] and dropped == []


def test_lineage_typical_short_gene_is_kept_against_the_short_end():
    # rps2-like: 220 aa, 55% of an angiosperm-heavy median of 400 aa, but references of that
    # length exist (short end 210 aa) so it is a real gene of the lineage
    ref = {"rps2": 400.0, "nad1": 325.0}
    floors = {"rps2": 210.0, "nad1": 325.0}
    gene = cds("rps2", (5000, 5662))  # 220 aa
    kept, dropped = drop_duplicate_fragments([gene], ref, SPLICED, floors)
    assert kept == [gene] and dropped == []
    # judged against the median it would have been a remnant
    kept, dropped = drop_duplicate_fragments([gene], ref, SPLICED)
    assert dropped == [gene]


def test_a_cds_below_the_short_end_is_still_a_remnant():
    ref = {"rps2": 400.0}
    floors = {"rps2": 210.0}
    remnant = cds("rps2", (5000, 5300))  # 99 aa, under 60% of 210
    _kept, dropped = drop_duplicate_fragments([remnant], ref, SPLICED, floors)
    assert dropped == [remnant]
