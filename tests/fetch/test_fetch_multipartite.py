"""Multipartite genomes: one genome, several accessions.

The cucumber mitochondrion is the canonical case — three chromosomes, three
records, one genome. Everything here guards against the three ways that breaks:
silent truncation, a length filter acting as a shredder, and an inflated n.
"""

from __future__ import annotations

from typing import Any

from organelleverse.fetch.assembly import group_records, molecule_sort_key
from organelleverse.fetch.filters import FetchFilters, apply_filters_to_units
from organelleverse.fetch.ncbi import chromosome_from_title


def molecule(
    accession: str,
    chromosome: str,
    length: int,
    *,
    organism: str = "Cucumis sativus",
    cultivar: str = "Calypso",
    doi: str = "10.1105/tpc.111.087189",
    gene_count: int = 20,
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "accession": accession,
        "chromosome": chromosome,
        "length": length,
        "organism": organism,
        "cultivar": cultivar,
        "doi": doi,
        "gene_count": gene_count,
        "protein_count": gene_count // 2,
    }
    record.update(extra)
    return record


# The real cucumber mitochondrion (verified against NCBI 2026-07-13).
CUCUMBER = [
    molecule("HQ860794.1", "2", 83_817),
    molecule("HQ860792.1", "1", 1_555_935),
    molecule("HQ860793.1", "3", 44_840),
]
CUCUMBER_TOTAL = 1_684_592

# A single-circle mitogenome carries no /chromosome qualifier at all.
RICE = [
    molecule(
        "NC_011033.1",
        "",
        490_520,
        organism="Oryza sativa Japonica Group",
        cultivar="Nipponbare",
        doi="",
        gene_count=81,
    )
]


# ─────────────────────────────────────────────────────────────
# detection
# ─────────────────────────────────────────────────────────────


def test_the_esummary_title_reveals_a_multipartite_genome_for_free() -> None:
    """No need to pay for GenBank XML just to discover we must group."""
    assert (
        chromosome_from_title("Cucumis sativus mitochondrion chromosome 1, complete sequence")
        == "1"
    )
    assert chromosome_from_title("Oryza sativa mitochondrion, complete genome") == ""


# ─────────────────────────────────────────────────────────────
# grouping
# ─────────────────────────────────────────────────────────────


def test_three_accessions_become_one_genome() -> None:
    (unit,) = group_records(CUCUMBER)
    assert unit.molecule_count == 3
    assert unit.is_multipartite is True
    assert unit.total_length == CUCUMBER_TOTAL


def test_molecules_are_ordered_by_chromosome_not_by_accession() -> None:
    """Accession order is NOT chromosome order, and this is the real data:

        HQ860792.1 -> chromosome 1
        HQ860794.1 -> chromosome 2   <- note
        HQ860793.1 -> chromosome 3

    Sorting by accession would hand the caller molecules 1, 3, 2.
    """
    (unit,) = group_records(CUCUMBER)
    assert [m["chromosome"] for m in unit.molecules] == ["1", "2", "3"]
    assert unit.accessions == ("HQ860792.1", "HQ860794.1", "HQ860793.1")
    assert sorted(unit.accessions) != list(unit.accessions)  # the trap


def test_a_single_circle_genome_is_its_own_unit() -> None:
    (unit,) = group_records(RICE)
    assert unit.molecule_count == 1
    assert unit.is_multipartite is False
    assert unit.total_length == 490_520


def test_a_multipartite_and_a_single_circle_genome_do_not_merge() -> None:
    units = group_records([*CUCUMBER, *RICE])
    assert len(units) == 2
    assert {u.molecule_count for u in units} == {1, 3}


def test_two_single_circle_genomes_from_one_project_stay_apart() -> None:
    """Without /chromosome the record IS the genome — never merge on project."""
    a = molecule("A.1", "", 500_000, organism="Zea mays", cultivar="B73", doi="10.1/x")
    b = molecule("B.1", "", 501_000, organism="Zea mays", cultivar="B73", doi="10.1/x")
    assert len(group_records([a, b])) == 2


def test_different_cultivars_are_different_genomes() -> None:
    """RefSeq's curated copy and a second cultivar must not collapse together."""
    other = [
        molecule("XX000001.1", "1", 1_500_000, cultivar="Gy14"),
        molecule("XX000002.1", "2", 80_000, cultivar="Gy14"),
    ]
    units = group_records([*CUCUMBER, *other])
    assert len(units) == 2
    assert {u.cultivar for u in units} == {"Calypso", "Gy14"}


def test_biosample_outranks_doi_as_a_grouping_key() -> None:
    same_sample = [
        molecule("A.1", "1", 900_000, biosample="SAMN001", doi="10.1/a"),
        molecule("B.1", "2", 90_000, biosample="SAMN001", doi="10.1/b"),
    ]
    (unit,) = group_records(same_sample)
    assert unit.molecule_count == 2
    assert unit.unit_id.startswith("biosample:")


# ─────────────────────────────────────────────────────────────
# incompleteness — the guard against silent truncation
# ─────────────────────────────────────────────────────────────


def test_a_gap_in_the_chromosome_numbering_is_reported() -> None:
    """Holding chromosomes 1 and 3 means chromosome 2 exists and we lack it."""
    partial = [CUCUMBER[1], CUCUMBER[2]]  # chromosome 1 and 3
    (unit,) = group_records(partial)
    assert unit.missing_chromosomes == (2,)
    assert unit.complete is False


def test_a_complete_genome_reports_no_gaps() -> None:
    (unit,) = group_records(CUCUMBER)
    assert unit.missing_chromosomes == ()
    assert unit.complete is True


# ─────────────────────────────────────────────────────────────
# the filter must not shred a genome
# ─────────────────────────────────────────────────────────────


def test_a_length_floor_keeps_every_molecule_of_a_genome_that_qualifies() -> None:
    """THE bug this module exists for.

    min_length=200_000 is a sane floor for a plant mitogenome. Applied per
    accession it deletes cucumber chromosomes 2 (83 kb) and 3 (44 kb) and leaves
    chromosome 1 posing as the whole genome. Applied per genome (1,684,592 bp)
    all three molecules survive together.
    """
    units = group_records(CUCUMBER)
    kept = apply_filters_to_units(
        units, FetchFilters(min_length=200_000), have_genbank_metadata=True
    )
    assert len(kept) == 1
    assert kept[0].molecule_count == 3  # not 1
    assert set(kept[0].accessions) == {"HQ860792.1", "HQ860793.1", "HQ860794.1"}


def test_a_genome_below_the_floor_is_dropped_whole() -> None:
    units = group_records(CUCUMBER)
    kept = apply_filters_to_units(
        units, FetchFilters(min_length=2_000_000), have_genbank_metadata=True
    )
    assert kept == []


def test_gene_count_is_summed_across_molecules() -> None:
    """A genome's gene count is the genome's, not the largest molecule's."""
    units = group_records(CUCUMBER)
    assert units[0].gene_count == 60  # 3 x 20
    kept = apply_filters_to_units(
        units, FetchFilters(min_gene_count=50), have_genbank_metadata=True
    )
    assert len(kept) == 1


def test_the_genome_view_reports_n_genomes_not_n_accessions() -> None:
    """Three accessions, sample size one. Reporting 3 inflates every statistic."""
    units = group_records([*CUCUMBER, *RICE])
    assert len(units) == 2  # two genomes
    assert sum(u.molecule_count for u in units) == 4  # four accessions


# ─────────────────────────────────────────────────────────────
# odd chromosome labels
# ─────────────────────────────────────────────────────────────


def test_non_numeric_chromosome_labels_sort_biggest_first() -> None:
    odd = [
        molecule("A.1", "MT-B", 100),
        molecule("B.1", "MT-A", 900),
    ]
    ordered = sorted(odd, key=molecule_sort_key)
    assert [m["accession"] for m in ordered] == ["B.1", "A.1"]


def test_numeric_labels_sort_before_non_numeric_ones() -> None:
    mixed = [molecule("A.1", "unnamed", 10), molecule("B.1", "1", 5)]
    ordered = sorted(mixed, key=molecule_sort_key)
    assert ordered[0]["chromosome"] == "1"


# ─────────────────────────────────────────────────────────────
# When the title lies — the Begonia case, from real data
# ─────────────────────────────────────────────────────────────


def test_nine_chromosomes_each_titled_complete_genome_are_still_suspected() -> None:
    """Begonia fimbristipula deposits 9 mitochondrial chromosomes as 9 records,
    every one of them titled "mitochondrion, complete genome". The title carries
    no chromosome number, so title-matching alone sees nine complete genomes and
    lets a 2,354 bp fragment through as a mitogenome. Three-plus records for one
    organism is the fallback signal.
    """
    from organelleverse.fetch.ncbi import suspect_multipartite

    begonia = [
        {
            "accession": f"PX0683{n:02d}.1",
            "length": size,
            "organism": "Begonia fimbristipula",
            "title": "Begonia fimbristipula mitochondrion, complete genome",
        }
        for n, size in enumerate([2_354, 2_956, 6_507, 11_921, 31_576], start=12)
    ]
    assert suspect_multipartite(begonia) is True


def test_a_refseq_plus_genbank_pair_is_not_suspicious() -> None:
    """Two records for one organism is the normal RefSeq/GenBank mirror, not a
    multipartite genome. Do not pay for GenBank XML over it."""
    from organelleverse.fetch.ncbi import suspect_multipartite

    pair = [
        {
            "accession": "NC_011033.1",
            "organism": "Oryza sativa",
            "title": "Oryza sativa mitochondrion, complete genome",
        },
        {
            "accession": "BA000029.3",
            "organism": "Oryza sativa",
            "title": "Oryza sativa mitochondrial DNA, complete genome",
        },
    ]
    assert suspect_multipartite(pair) is False


# ─────────────────────────────────────────────────────────────
# Over-grouping — the potato case, from real data
# ─────────────────────────────────────────────────────────────


def test_a_bioproject_holding_six_cultivars_is_not_one_genome() -> None:
    """PRJNA718240 holds six potato cultivars, each a complete 3-chromosome
    mitogenome. Grouping on BioProject fused them into one 18-molecule chimera
    of 2.79 Mb — a genome that does not exist. A project is a submission, not a
    genome.
    """
    potato = [
        molecule(
            f"MZ0307{n:02d}.1",
            str(chrom),
            size,
            organism="Solanum tuberosum",
            cultivar=cv,
            doi="",
            bioproject="PRJNA718240",
        )
        for n, (cv, chrom, size) in enumerate(
            [
                (cv, c, s)
                for cv in ("Spunta", "Atlantic", "Altus")
                for c, s in ((1, 49_230), (2, 112_832), (3, 312_533))
            ],
            start=25,
        )
    ]
    units = group_records(potato)
    assert len(units) == 3  # three cultivars, three genomes
    assert {u.molecule_count for u in units} == {3}
    assert all(u.total_length < 500_000 for u in units)  # not 2.79 Mb


def test_a_repeated_chromosome_label_proves_the_grouping_is_wrong() -> None:
    """A genome cannot contain two chromosome 1s. When it appears to, the group
    is a chimera — split it rather than report a genome that does not exist.
    """
    chimera = [
        molecule("A.1", "1", 100, cultivar="", doi="", biosample="SAME"),
        molecule("B.1", "1", 200, cultivar="", doi="", biosample="SAME"),
        molecule("C.1", "2", 300, cultivar="", doi="", biosample="SAME"),
    ]
    units = group_records(chimera)
    assert len(units) == 3  # fell back to one molecule per unit
    assert all(u.molecule_count == 1 for u in units)


def test_a_bioproject_without_a_sample_id_does_not_group_on_the_project() -> None:
    """The project must not be the key — but the molecules still belong to one
    genome, and accession order recovers it. What must never happen is grouping
    on PRJNA alone, which would fuse a whole cultivar panel.
    """
    anonymous = [
        molecule("MN000001.1", "1", 100, cultivar="", doi="", bioproject="PRJNA1"),
        molecule("MN000002.1", "2", 200, cultivar="", doi="", bioproject="PRJNA1"),
    ]
    (unit,) = group_records(anonymous)
    assert unit.molecule_count == 2
    assert not unit.unit_id.startswith("bioproject:")  # never keyed on the project

    # Add a second genome from the same project: the chromosome numbers repeat,
    # so the run is cut and the two genomes stay apart.
    panel = [
        *anonymous,
        molecule("MN000003.1", "1", 105, cultivar="", doi="", bioproject="PRJNA1"),
        molecule("MN000004.1", "2", 205, cultivar="", doi="", bioproject="PRJNA1"),
    ]
    assert len(group_records(panel)) == 2  # two genomes, not one four-molecule chimera


# ─────────────────────────────────────────────────────────────
# Naming schemes — every one of these is from the real seed-plant data
# ─────────────────────────────────────────────────────────────


def test_every_naming_scheme_seen_in_real_data_yields_a_number() -> None:
    """Observed labels across 71 real seed-plant mitogenomes."""
    from organelleverse.fetch.assembly import chromosome_number

    assert chromosome_number("3") == 3
    assert chromosome_number("108") == 108
    assert chromosome_number("contig17") == 17  # Camellia lanceoleosa
    assert chromosome_number("cir6") == 6  # Coptis chinensis
    assert chromosome_number("M01") == 1
    assert chromosome_number("LS1") == 1
    assert chromosome_number("ge1") == 1
    assert chromosome_number("A") is None  # not seen, but possible


def test_molecules_named_cir1_cir6_order_and_group_correctly() -> None:
    coptis = [
        molecule(
            f"OP4667{n}.1",
            f"cir{n}",
            100_000 * n,
            organism="Coptis chinensis",
            cultivar="",
            doi="",
            biosample="SAMN_COPTIS",
        )
        for n in (3, 1, 2)
    ]
    (unit,) = group_records(coptis)
    assert [m["chromosome"] for m in unit.molecules] == ["cir1", "cir2", "cir3"]
    assert unit.complete is True


def test_cir1_and_1_are_the_same_molecule_number_so_the_chimera_is_caught() -> None:
    """Two genomes, one labelled cir1..cir2 and one 1..2, wrongly grouped.
    Comparing raw labels would see four distinct molecules and pass the chimera
    through. Comparing numbers catches it.
    """
    chimera = [
        molecule("A.1", "cir1", 100, cultivar="", doi="", biosample="SAME"),
        molecule("B.1", "1", 200, cultivar="", doi="", biosample="SAME"),
        molecule("C.1", "cir2", 300, cultivar="", doi="", biosample="SAME"),
        molecule("D.1", "2", 400, cultivar="", doi="", biosample="SAME"),
    ]
    units = group_records(chimera)
    assert len(units) == 4  # split, not reported as one 4-molecule genome


def test_an_unnumbered_label_makes_completeness_unknowable_not_true() -> None:
    """A/B/C gives no way to detect a missing molecule. Say "cannot tell",
    never "complete"."""
    lettered = [
        molecule("A.1", "A", 100, cultivar="x", doi="", biosample="S1"),
        molecule("B.1", "B", 200, cultivar="x", doi="", biosample="S1"),
    ]
    (unit,) = group_records(lettered)
    assert unit.completeness_checkable is False
    assert unit.complete is None  # not True
    assert unit.missing_chromosomes == ()


def test_a_numbered_genome_still_reports_completeness_normally() -> None:
    (unit,) = group_records(CUCUMBER)
    assert unit.completeness_checkable is True
    assert unit.complete is True


# ─────────────────────────────────────────────────────────────
# Accession-run recovery — for records with no BioSample and no cultivar
# ─────────────────────────────────────────────────────────────


def test_consecutive_accessions_recover_a_genome_with_no_sample_id() -> None:
    """Amorphophallus/Begonia-style records: /chromosome present, but no
    BioSample and no cultivar. Left ungrouped they each become a "complete
    genome" of one molecule — a fragment reported as whole. GenBank assigns
    consecutive accessions to one submission, so accession order recovers it.
    """
    anon = [
        molecule(
            f"MN10480{n}.1",
            str(n),
            size,
            organism="Solanum tuberosum",
            cultivar="",
            doi="",
            biosample="",
        )
        for n, size in [(1, 312_441), (2, 112_800), (3, 49_230)]
    ]
    (unit,) = group_records(anon)
    assert unit.molecule_count == 3
    assert unit.total_length == 474_471
    assert unit.complete is True


def test_a_consecutive_run_holding_six_cultivars_is_cut_at_each_repeat() -> None:
    """MZ030725..742 is six potato cultivars in one consecutive block. The
    chromosome numbers run 1,2,3,1,2,3,... and each repeat is a genome boundary.
    """
    panel = []
    acc = 725
    for _ in range(6):
        for chrom, size in ((1, 49_230), (2, 112_832), (3, 312_533)):
            panel.append(
                molecule(
                    f"MZ030{acc}.1",
                    str(chrom),
                    size,
                    organism="Solanum tuberosum",
                    cultivar="",
                    doi="",
                    biosample="",
                )
            )
            acc += 1
    units = group_records(panel)
    assert len(units) == 6  # six genomes, not one 18-molecule chimera
    assert {u.molecule_count for u in units} == {3}
