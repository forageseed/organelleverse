"""Dedup: the same genome deposited twice. Offline.

The line this module must never cross: technical redundancy is removed,
biological variation is not. Japonica and Indica rice mitogenomes differ by
995 bp and are two genomes. NC_011033.1 and BA000029.3 are the same 490,520 bp
and are one.
"""

from __future__ import annotations

from typing import Any

from organelleverse.fetch.dedup import dedup_genomes, quality_rank, select_representative


def rec(accession: str, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "accession": accession,
        "organism": "Oryza sativa",
        "length": 490_520,
        "cultivar": "",
        "completeness": "full",
        "topology": "circular",
        "gene_count": 80,
        "ambiguous_count": 0,
        "is_refseq": accession.startswith("NC_"),
        "derived_from": "",
    }
    base.update(kw)
    return base


# ─────────────────────────────────────────────────────────────
# L1 — the RefSeq mirror
# ─────────────────────────────────────────────────────────────


def test_a_refseq_record_merges_with_the_submission_it_curates() -> None:
    """NC_011033.1's COMMENT says it was derived from BA000029. One genome."""
    records = [
        rec("BA000029.3", is_refseq=False),
        rec("NC_011033.1", derived_from="BA000029"),
    ]
    (cluster,) = dedup_genomes(records)
    assert cluster.is_duplicated
    assert cluster.reason == "refseq_mirror"
    assert set(cluster.accessions) == {"BA000029.3", "NC_011033.1"}


def test_the_derived_from_link_matches_across_versions() -> None:
    """The COMMENT names BA000029; the record we hold is BA000029.3."""
    records = [rec("BA000029.3", is_refseq=False), rec("NC_011033.1", derived_from="BA000029")]
    assert len(dedup_genomes(records)) == 1


def test_records_with_no_link_stay_apart() -> None:
    records = [rec("BA000029.3", is_refseq=False), rec("NC_011033.1")]
    assert len(dedup_genomes(records)) == 2


# ─────────────────────────────────────────────────────────────
# L2 — identical sequence
# ─────────────────────────────────────────────────────────────


def test_identical_sequence_digests_merge() -> None:
    records = [rec("A.1", sha256="f" * 64), rec("B.1", sha256="f" * 64)]
    (cluster,) = dedup_genomes(records)
    assert cluster.reason == "identical_sequence"


def test_different_digests_do_not_merge() -> None:
    records = [rec("A.1", sha256="a" * 64), rec("B.1", sha256="b" * 64)]
    assert len(dedup_genomes(records)) == 2


# ─────────────────────────────────────────────────────────────
# The veto — biological variation is data, not redundancy
# ─────────────────────────────────────────────────────────────


def test_different_cultivars_never_merge_even_with_identical_sequence() -> None:
    """The hard veto. Whatever the evidence says, two named samples are two
    genomes — subspecies divergence is the subject of study, not noise."""
    records = [
        rec("A.1", cultivar="Nipponbare", sha256="f" * 64),
        rec("B.1", cultivar="93-11", sha256="f" * 64),
    ]
    assert len(dedup_genomes(records)) == 2


def test_different_vouchers_never_merge() -> None:
    records = [
        rec("A.1", specimen_voucher="SUMG 004", derived_from=""),
        rec("NC_1.1", specimen_voucher="SUMG 005", derived_from="A"),
    ]
    assert len(dedup_genomes(records)) == 2


def test_an_absent_label_does_not_veto() -> None:
    """Most records name no cultivar. Absence must not block a real mirror."""
    records = [
        rec("BA000029.3", cultivar="Nipponbare", is_refseq=False),
        rec("NC_011033.1", cultivar="", derived_from="BA000029"),
    ]
    assert len(dedup_genomes(records)) == 1


# ─────────────────────────────────────────────────────────────
# Representative selection — quality outranks RefSeq status
# ─────────────────────────────────────────────────────────────


def test_a_complete_genbank_record_beats_a_partial_refseq_one() -> None:
    """The caller's rule: prefer the chromosome-level assembly even when the
    contig-level one is the RefSeq copy. Completeness is the criterion; curation
    status is only the tie-breaker.
    """
    partial_refseq = rec("NC_1.1", completeness="partial", is_refseq=True)
    complete_genbank = rec("XX1.1", completeness="full", is_refseq=False)

    chosen = select_representative([partial_refseq, complete_genbank])
    assert chosen["accession"] == "XX1.1"


def test_a_whole_genome_beats_a_truncated_one() -> None:
    truncated = rec("A.1", complete=False, is_refseq=True)
    whole = rec("B.1", complete=True, is_refseq=False)
    assert select_representative([truncated, whole])["accession"] == "B.1"


def test_a_better_annotated_record_wins() -> None:
    sparse = rec("A.1", gene_count=12, is_refseq=True)
    rich = rec("B.1", gene_count=130, is_refseq=False)
    assert select_representative([sparse, rich])["accession"] == "B.1"


def test_fewer_ambiguous_bases_wins() -> None:
    gappy = rec("A.1", ambiguous_count=5_000, is_refseq=True)
    clean = rec("B.1", ambiguous_count=0, is_refseq=False)
    assert select_representative([gappy, clean])["accession"] == "B.1"


def test_refseq_only_breaks_a_tie() -> None:
    """When everything else is equal, the curated copy wins — and only then."""
    genbank = rec("XX1.1", is_refseq=False)
    refseq = rec("NC_1.1", is_refseq=True)
    assert select_representative([genbank, refseq])["accession"] == "NC_1.1"


def test_circular_topology_is_a_weak_signal_not_a_criterion() -> None:
    """Rice's complete mitogenome is deposited `linear`. Topology must never
    outrank completeness."""
    linear_complete = rec("A.1", topology="linear", completeness="full")
    circular_partial = rec("B.1", topology="circular", completeness="partial")
    assert select_representative([linear_complete, circular_partial])["accession"] == "A.1"


def test_the_representative_ranks_above_its_duplicates() -> None:
    records = [
        rec("BA000029.3", is_refseq=False, gene_count=80),
        rec("NC_011033.1", derived_from="BA000029", is_refseq=True, gene_count=80),
    ]
    (cluster,) = dedup_genomes(records)
    assert quality_rank(cluster.representative) >= max(quality_rank(d) for d in cluster.duplicates)
    assert cluster.representative["accession"] == "NC_011033.1"  # tie -> RefSeq


# ─────────────────────────────────────────────────────────────
# The audit trail
# ─────────────────────────────────────────────────────────────


def test_a_cluster_reports_why_it_merged() -> None:
    records = [rec("BA000029.3", is_refseq=False), rec("NC_011033.1", derived_from="BA000029")]
    (cluster,) = dedup_genomes(records)
    record = cluster.as_record()
    assert record["representative"] == "NC_011033.1"
    assert record["duplicates"] == ["BA000029.3"]
    assert record["reason"] == "refseq_mirror"


def test_a_singleton_cluster_carries_no_reason() -> None:
    (cluster,) = dedup_genomes([rec("A.1")])
    assert cluster.is_duplicated is False
    assert cluster.reason == ""


# ─────────────────────────────────────────────────────────────
# Both RefSeq wordings — from real records
#
# NCBI writes the same fact two ways, and a parser that knows only one silently
# misses every mirror worded the other. All three cucumber chromosomes are
# "identical to"; rice is "derived from".
# ─────────────────────────────────────────────────────────────


def test_both_refseq_wordings_are_recognised() -> None:
    from organelleverse.fetch.genbank_meta import derived_from

    assert derived_from("The reference sequence was derived from BA000029.") == "BA000029"
    assert derived_from("The reference sequence is identical to HQ860792.") == "HQ860792"


def test_a_version_replacement_notice_is_not_a_mirror() -> None:
    """ "this sequence version replaced AB076665.2" is a version history note,
    not a curation link. Reading it as one would merge unrelated records."""
    from organelleverse.fetch.genbank_meta import derived_from

    assert derived_from("On or before Nov 5, 2004 this version replaced AB076665.2") == ""


def test_the_cucumber_mirrors_collapse_to_three_molecules() -> None:
    """Real data: NC_016005/04/06 are 'identical to' HQ860792/94/93. Six records,
    three molecules of one genome — not six genomes.
    """
    pairs = [
        ("HQ860792.1", "NC_016005.1", "1"),
        ("HQ860794.1", "NC_016004.1", "2"),
        ("HQ860793.1", "NC_016006.1", "3"),
    ]
    records = []
    for genbank, refseq, chrom in pairs:
        records.append(
            rec(
                genbank,
                organism="Cucumis sativus",
                cultivar="Calypso",
                chromosome=chrom,
                is_refseq=False,
                completeness="",
            )
        )
        records.append(
            rec(
                refseq,
                organism="Cucumis sativus",
                cultivar="Calypso",
                chromosome=chrom,
                is_refseq=True,
                completeness="full",
                derived_from=genbank.split(".")[0],
            )
        )

    clusters = dedup_genomes(records)
    assert len(clusters) == 3  # one per molecule, not six
    assert all(c.is_duplicated for c in clusters)
    assert all(c.reason == "refseq_mirror" for c in clusters)
    # RefSeq wins each pair here: it states full length, the GenBank copy does not.
    assert {c.representative["accession"] for c in clusters} == {
        "NC_016005.1",
        "NC_016004.1",
        "NC_016006.1",
    }


# ─────────────────────────────────────────────────────────────
# The wording guard
#
# Measured, not assumed: 400 sampled Viridiplantae organelle RefSeq records,
# 100% of their curation links parsed. 300 of them said "is identical to";
# "was derived from" is real but rare (Oryza NC_011033.1).
#
# That is a claim about a sample. If NCBI changes the wording, the failure is
# SILENT — a mirror goes unseen, a genome is counted twice, and n is wrong.
# So an unparsed RefSeq COMMENT is reported rather than ignored.
# ─────────────────────────────────────────────────────────────


def test_an_unparsed_refseq_comment_is_reported_not_ignored() -> None:
    from organelleverse.fetch.dedup import unlinked_refseq_records

    records = [
        rec(
            "NC_1.1",
            is_refseq=True,
            comment="The reference sequence is identical to XX000001.",
            derived_from="XX000001",
        ),
        # A wording we do not know. It must not pass unnoticed.
        rec(
            "NC_2.1",
            is_refseq=True,
            comment="This record supersedes an earlier assembly.",
            derived_from="",
        ),
    ]
    assert unlinked_refseq_records(records) == ["NC_2.1"]


def test_a_genbank_record_without_a_link_is_not_flagged() -> None:
    """Only RefSeq records are expected to name a source."""
    from organelleverse.fetch.dedup import unlinked_refseq_records

    records = [rec("XX1.1", is_refseq=False, comment="Assembly Method: SPAdes", derived_from="")]
    assert unlinked_refseq_records(records) == []


def test_a_refseq_record_with_no_comment_at_all_is_not_flagged() -> None:
    from organelleverse.fetch.dedup import unlinked_refseq_records

    records = [rec("NC_1.1", is_refseq=True, comment="", derived_from="")]
    assert unlinked_refseq_records(records) == []
