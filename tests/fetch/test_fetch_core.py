"""Pure-function core of the fetch suite: batching, hashing, parsing, retry.

All offline. No network, no monkeypatching of urllib.
"""

from __future__ import annotations

import pytest

from organelleverse.core.errors import OrganelleExecutionError, OrganelleParameterError
from organelleverse.fetch._http import (
    MAX_URL_LENGTH,
    batch_accessions_for_url,
    retry_with_backoff,
)
from organelleverse.fetch.manifest import accession_set_sha256, build_manifest
from organelleverse.fetch.ncbi import (
    build_organelle_term,
    parse_esearch_xml,
    parse_release_number,
)

# ─────────────────────────────────────────────────────────────
# accession set hashing — the determinism primitive
# ─────────────────────────────────────────────────────────────


def test_accession_hash_is_order_independent() -> None:
    """The hash identifies a *set*, so input order must not change it."""
    a = accession_set_sha256(["NC_000932.1", "NC_037304.1", "MT_000001.1"])
    b = accession_set_sha256(["MT_000001.1", "NC_000932.1", "NC_037304.1"])
    assert a == b


def test_accession_hash_deduplicates() -> None:
    """A duplicated accession is the same set."""
    once = accession_set_sha256(["NC_000932.1"])
    twice = accession_set_sha256(["NC_000932.1", "NC_000932.1"])
    assert once == twice


def test_accession_hash_is_sensitive_to_membership() -> None:
    """Adding a record must change the hash, or the benchmark is worthless."""
    a = accession_set_sha256(["NC_000932.1"])
    b = accession_set_sha256(["NC_000932.1", "NC_037304.1"])
    assert a != b


def test_accession_hash_is_a_sha256_hexdigest() -> None:
    digest = accession_set_sha256(["NC_000932.1"])
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_empty_accession_set_is_rejected() -> None:
    """An empty set has no scientific meaning; fail loudly rather than hash ''."""
    with pytest.raises(OrganelleParameterError):
        accession_set_sha256([])


# ─────────────────────────────────────────────────────────────
# URL batching — NCBI returns HTTP 414 above ~2000 chars
# ─────────────────────────────────────────────────────────────


def test_batching_keeps_every_url_under_the_limit() -> None:
    accessions = [f"NC_{i:06d}.1" for i in range(500)]
    base = 120
    batches = batch_accessions_for_url(accessions, base_url_length=base)
    for batch in batches:
        url_len = base + len(",".join(batch))
        assert url_len <= MAX_URL_LENGTH


def test_batching_loses_no_accession_and_preserves_no_duplicates() -> None:
    accessions = [f"NC_{i:06d}.1" for i in range(500)]
    batches = batch_accessions_for_url(accessions, base_url_length=120)
    flat = [a for batch in batches for a in batch]
    assert flat == accessions  # order preserved, nothing dropped


def test_small_input_is_a_single_batch() -> None:
    batches = batch_accessions_for_url(["NC_000932.1", "NC_037304.1"], base_url_length=100)
    assert len(batches) == 1


def test_batching_rejects_a_base_url_that_leaves_no_room() -> None:
    with pytest.raises(OrganelleParameterError):
        batch_accessions_for_url(["NC_000932.1"], base_url_length=MAX_URL_LENGTH)


# ─────────────────────────────────────────────────────────────
# retry / backoff — NCBI rate-limits aggressively
# ─────────────────────────────────────────────────────────────


def test_retry_returns_the_value_on_first_success() -> None:
    calls: list[int] = []

    def op() -> str:
        calls.append(1)
        return "ok"

    assert retry_with_backoff(op, max_attempts=3, initial_delay=0.0) == "ok"
    assert len(calls) == 1


def test_retry_recovers_from_transient_failures() -> None:
    calls: list[int] = []

    def op() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise OSError("transient")
        return "ok"

    assert retry_with_backoff(op, max_attempts=5, initial_delay=0.0) == "ok"
    assert len(calls) == 3


def test_retry_raises_a_structured_retryable_error_when_exhausted() -> None:
    def op() -> str:
        raise OSError("always down")

    with pytest.raises(OrganelleExecutionError) as excinfo:
        retry_with_backoff(op, max_attempts=2, initial_delay=0.0)

    error = excinfo.value
    assert error.retryable is True
    assert error.code.startswith("network.")
    assert error.as_dict()["details"]["attempts"] == 2


def test_retry_does_not_swallow_a_parameter_error() -> None:
    """A bad request will never succeed on retry — fail fast, don't hammer NCBI."""

    def op() -> str:
        raise OrganelleParameterError(code="input.bad", message="nope")

    with pytest.raises(OrganelleParameterError):
        retry_with_backoff(op, max_attempts=5, initial_delay=0.0)


# ─────────────────────────────────────────────────────────────
# NCBI response parsing
# ─────────────────────────────────────────────────────────────


def test_parse_release_number() -> None:
    assert parse_release_number("236\n") == 236


def test_parse_release_number_rejects_garbage() -> None:
    with pytest.raises(OrganelleExecutionError):
        parse_release_number("<html>404</html>")


ESEARCH_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<eSearchResult>
  <Count>58082</Count>
  <RetMax>3</RetMax>
  <IdList>
    <Id>2760155037</Id>
    <Id>2760155036</Id>
    <Id>2760155035</Id>
  </IdList>
</eSearchResult>
"""


def test_parse_esearch_xml() -> None:
    count, uids = parse_esearch_xml(ESEARCH_XML)
    assert count == 58082
    assert uids == ["2760155037", "2760155036", "2760155035"]


def test_parse_esearch_xml_handles_zero_hits() -> None:
    xml = "<eSearchResult><Count>0</Count><IdList></IdList></eSearchResult>"
    count, uids = parse_esearch_xml(xml)
    assert count == 0
    assert uids == []


def test_parse_esearch_xml_rejects_an_error_page() -> None:
    with pytest.raises(OrganelleExecutionError):
        parse_esearch_xml("<html>Service unavailable</html>")


# ─────────────────────────────────────────────────────────────
# Entrez query construction — the scoping of a scientific claim
# ─────────────────────────────────────────────────────────────


def test_organelle_term_is_organelle_specific() -> None:
    plastid = build_organelle_term(organelle="plastid")
    mito = build_organelle_term(organelle="mitochondrion")
    assert "chloroplast[filter]" in plastid
    assert "mitochondrion[filter]" in mito
    assert plastid != mito


def test_organelle_term_defaults_to_complete_genomes() -> None:
    term = build_organelle_term(organelle="plastid")
    assert '"complete genome"[Title]' in term


def test_organelle_term_scopes_to_taxon() -> None:
    term = build_organelle_term(organelle="plastid", taxon="Viridiplantae")
    assert "Viridiplantae[Organism]" in term


def test_organelle_term_can_drop_the_completeness_filter() -> None:
    """Plant mitochondria are hard to assemble; recall sometimes beats precision."""
    term = build_organelle_term(organelle="mitochondrion", complete_only=False)
    assert "complete genome" not in term


def test_organelle_term_rejects_an_unknown_organelle() -> None:
    with pytest.raises(OrganelleParameterError):
        build_organelle_term(organelle="nucleus")


# ─────────────────────────────────────────────────────────────
# esummary parsing — fixture is a real nuccore response
#
# Regression guard: the first version of this parser read `Slen` and `Organism`,
# neither of which nuccore returns. Every offline test passed and every real
# record came back with length=0 — which would have silently emptied any
# min_length filter. Fixtures come from the wire, not from memory.
# ─────────────────────────────────────────────────────────────

REAL_ESUMMARY = """<?xml version="1.0" encoding="UTF-8" ?>
<eSummaryResult>
<DocSum>
	<Id>2441074688</Id>
	<Item Name="Caption" Type="String">OP474144</Item>
	<Item Name="Title" Type="String">Arabidopsis thaliana chloroplast, complete genome</Item>
	<Item Name="Gi" Type="Integer">2441074688</Item>
	<Item Name="UpdateDate" Type="String">2024/06/11</Item>
	<Item Name="TaxId" Type="Integer">3702</Item>
	<Item Name="Length" Type="Integer">154478</Item>
	<Item Name="Status" Type="String">live</Item>
	<Item Name="AccessionVersion" Type="String">OP474144.1</Item>
</DocSum>
</eSummaryResult>"""


def test_esummary_reads_the_length_item_that_nuccore_actually_sends() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    (record,) = parse_esummary_xml(REAL_ESUMMARY)
    assert record["length"] == 154478  # `Length`, not `Slen`
    assert record["accession"] == "OP474144.1"
    assert record["taxid"] == "3702"


def test_esummary_derives_organism_from_title_since_nuccore_sends_none() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    (record,) = parse_esummary_xml(REAL_ESUMMARY)
    assert record["organism"] == "Arabidopsis thaliana"


def test_esummary_tolerates_the_legacy_slen_item() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    xml = """<eSummaryResult><DocSum>
      <Item Name="AccessionVersion" Type="String">X1.1</Item>
      <Item Name="Title" Type="String">Zea mays mitochondrion, complete genome</Item>
      <Item Name="Slen" Type="Integer">569630</Item>
    </DocSum></eSummaryResult>"""
    (record,) = parse_esummary_xml(xml)
    assert record["length"] == 569630
    assert record["organism"] == "Zea mays"


# ─────────────────────────────────────────────────────────────
# The completeness clause is organelle-specific
#
# `"complete sequence"[Title]` means opposite things per organelle:
#   mitochondrion — how a multipartite chromosome is titled (needed)
#   chloroplast   — 99.6% gene/spacer fragments (poison)
# ─────────────────────────────────────────────────────────────


def test_mitochondrion_query_admits_multipartite_chromosome_records() -> None:
    term = build_organelle_term(organelle="mitochondrion")
    assert '"complete sequence"[Title]' in term
    assert "chromosome[Title]" in term


def test_plastid_query_excludes_complete_sequence_which_is_236k_fragments() -> None:
    """Adding it turns 58,082 plastomes into 295,091 mostly-fragments."""
    term = build_organelle_term(organelle="plastid")
    assert '"complete sequence"[Title]' not in term


def test_plastid_query_admits_the_titles_that_never_say_complete() -> None:
    """`genome assembly, organelle: plastid:chloroplast` is a real 154 kb
    plastome. The old clause missed 3,136 genomes titled this way."""
    term = build_organelle_term(organelle="plastid")
    assert '"genome assembly"[Title]' in term
    assert '"whole genome shotgun"[Title]' in term
    assert "chromosome[Title]" in term  # Homalomena perplexa: 12 plastid chromosomes


def test_sources_tried_is_recorded_when_given() -> None:
    manifest = build_manifest(
        source="gir",
        organelle="nuclear",
        accessions=["gir:Rosaceae/v1.2_Malus_domestica"],
        scope={"taxon": "Malus domestica"},
        sources_tried=[
            {"source": "tair", "status": "miss", "detail": "not arabidopsis"},
            {"source": "ncbi", "status": "miss", "detail": "no assembly"},
            {"source": "gir", "status": "hit", "detail": "matched"},
        ],
    )
    assert manifest["sources_tried"] == [
        {"source": "tair", "status": "miss", "detail": "not arabidopsis"},
        {"source": "ncbi", "status": "miss", "detail": "no assembly"},
        {"source": "gir", "status": "hit", "detail": "matched"},
    ]


def test_sources_tried_is_absent_when_not_given() -> None:
    manifest = build_manifest(
        source="gir",
        organelle="nuclear",
        accessions=["gir:x"],
        scope={},
    )
    assert "sources_tried" not in manifest


def test_the_source_literal_accepts_every_new_source() -> None:
    for source in ("ngdc_gwh", "gir", "imp", "pgd", "tair"):
        manifest = build_manifest(source=source, organelle="nuclear", accessions=["x:1"], scope={})
        assert manifest["source"] == source


# ─────────────────────────────────────────────────────────────
# The mitochondrion clause excludes fragments, and only fragments
#
# Re-checked 2026-10-16: the old clause returned 15,191 Viridiplantae records,
# 4,748 of them gene, pseudogene, mRNA or spacer records that say "complete
# sequence" about the gene, plus 91 "partial sequence/genome" molecules.
# ─────────────────────────────────────────────────────────────


def test_mitochondrion_term_excludes_fragment_vocabulary_unless_a_molecule_is_named() -> None:
    term = build_organelle_term(organelle="mitochondrion")
    assert "pseudogene[Title]" in term
    assert "mrna[Title]" in term
    assert "orf*[Title]" in term
    assert "NOT (genome[Title] OR chromosome[Title] OR plasmid[Title])" in term
    assert '"partial sequence"[Title]' in term


def test_every_word_the_mitochondrion_term_excludes_is_one_the_post_filter_rejects() -> None:
    """The term is a cheaper copy of the post-filter, never a stricter one.

    A word the term excludes but looks_like_gene_record keeps would drop real
    genomes before anyone could see them.
    """
    from organelleverse.fetch.filters import _GENE_RECORD_WORDS, looks_like_gene_record

    term = build_organelle_term(organelle="mitochondrion")
    for word in (*_GENE_RECORD_WORDS, "ribosomal RNA", "transfer RNA", "orf25", "orfB"):
        title = f"Zea mays {word}, complete sequence; mitochondrial"
        assert looks_like_gene_record(title), word
        assert looks_like_gene_record(title.replace("; mitochondrial", " chromosome 3")) is False
    for word in _GENE_RECORD_WORDS:
        assert f"{word}[Title]" in term


def test_the_partial_molecule_rule_reads_the_molecule_not_the_gene() -> None:
    from organelleverse.fetch.ncbi import _PARTIAL_MOLECULE

    # PQ492199.1, 6,648 bp
    assert _PARTIAL_MOLECULE.search(
        "Stipa sibirica isolate A_2_1 chromosome 29 mitochondrion, partial sequence"
    )
    assert not _PARTIAL_MOLECULE.search(
        "Cymbidium kanran cultivar Jing Zi chromosome 5 nad1 gene, partial cds; mitochondrial"
    )


def test_plastid_term_is_untouched_by_the_mitochondrion_exclusions() -> None:
    assert "NOT" not in build_organelle_term(organelle="plastid")


# ─────────────────────────────────────────────────────────────
# esummary 2.0 carries the species; the title does not
# ─────────────────────────────────────────────────────────────

# Trimmed from a real version=2.0 reply (2026-10-16); <Statistics> dropped.
REAL_ESUMMARY_V2 = """<?xml version="1.0" encoding="UTF-8" ?>
<eSummaryResult>
<DocumentSummarySet status="OK">
<DocumentSummary uid="194033210"><Caption>NC_011033</Caption><Title>Oryza sativa Japonica Group mitochondrion, complete genome</Title><Extra>gi|194033210|ref|NC_011033.1|</Extra><Gi>194033210</Gi><CreateDate>2008/07/10</CreateDate><UpdateDate>2023/04/03</UpdateDate><TaxId>39947</TaxId><Slen>490520</Slen><Biomol>genomic</Biomol><MolType>dna</MolType><Topology>linear</Topology><SourceDb>refseq</SourceDb><Genome>mitochondrion</Genome><SubType>cultivar</SubType><SubName>Nipponbare</SubName><AssemblyGi>47118326</AssemblyGi><AssemblyAcc>BA000029</AssemblyAcc><Completeness>complete</Completeness><GeneticCode>1</GeneticCode><Organism>Oryza sativa Japonica Group</Organism><Statistics><Stat type="Length" count="490520"/></Statistics><Flags>512</Flags><Properties na="1">1</Properties><OSLT indexed="yes">NC_011033.1</OSLT><AccessionVersion>NC_011033.1</AccessionVersion></DocumentSummary>
<DocumentSummary uid="287376"><Caption>D13097</Caption><Title>Oryza sativa Japonica Group mitochondrial gene for tRNA-Cys, complete sequence, clone: pMTVSCT1</Title><Extra>gi|287376|dbj|D13097.1|RICMT01</Extra><Gi>287376</Gi><CreateDate>1993/06/11</CreateDate><UpdateDate>2007/12/11</UpdateDate><TaxId>39947</TaxId><Slen>2177</Slen><SourceDb>insd</SourceDb><Genome>mitochondrion</Genome><Organism>Oryza sativa Japonica Group</Organism><AccessionVersion>D13097.1</AccessionVersion></DocumentSummary>
</DocumentSummarySet>
</eSummaryResult>"""


def test_esummary_v2_takes_the_organism_from_ncbi_taxonomy() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    refseq, clone = parse_esummary_xml(REAL_ESUMMARY_V2)
    # The fields every caller already reads are unchanged.
    assert {k: refseq[k] for k in _BASE_FIELDS} == {
        "accession": "NC_011033.1",
        "title": "Oryza sativa Japonica Group mitochondrion, complete genome",
        "organism": "Oryza sativa Japonica Group",
        "taxid": "39947",
        "length": 490520,
        "status": "live",
        "create_date": "2008/07/10",
        "update_date": "2023/04/03",
    }
    assert refseq["cultivar"] == "Nipponbare"
    assert refseq["assembly_accession"] == "BA000029"  # the INSDC record RefSeq curated
    assert refseq["summary"]["Completeness"] == "complete"
    assert refseq["summary"]["Statistics"] == [{"type": "Length", "count": "490520"}]
    assert clone["organism"] == "Oryza sativa Japonica Group"
    assert clone["length"] == 2177


@pytest.mark.parametrize(
    ("title", "organism"),
    [
        # OZ568514.1 — no organelle word before the comma
        ("Oxalis debilis genome assembly, organelle: mitochondrion", "Oxalis debilis"),
        # EU497799.1 — used to come back as the whole gene description
        (
            "Abies bracteata NADH dehydrogenase subunit 3 (nad3) gene, partial cds; "
            "nad3-rps12 intergenic spacer, complete sequence",
            "",
        ),
    ],
)
def test_a_v1_title_without_a_reliable_cut_gives_no_organism(title: str, organism: str) -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    xml = f"""<eSummaryResult><DocSum>
      <Item Name="AccessionVersion" Type="String">X1.1</Item>
      <Item Name="Title" Type="String">{title}</Item>
      <Item Name="Length" Type="Integer">1000</Item>
    </DocSum></eSummaryResult>"""
    (record,) = parse_esummary_xml(xml)
    assert record["organism"] == organism


# ─────────────────────────────────────────────────────────────
# Every esummary field is kept, and the sample is structured
#
# `organism` is the species only, so 16 cucumber samples (Calypso, 9930,
# 9110gt, Cu2, XTMC, ...) all read "Cucumis sativus"; grouping on it fused
# their chromosomes into one 148-molecule, 21.7 Mb "genome". The sample lives
# in SubType/SubName. Fixtures: real version=2.0 DocumentSummary elements
# (2026-10-17), <Statistics> removed.
# ─────────────────────────────────────────────────────────────

_BASE_FIELDS = (
    "accession",
    "title",
    "organism",
    "taxid",
    "length",
    "status",
    "create_date",
    "update_date",
)

CUCUMBER_V2 = """<eSummaryResult><DocumentSummarySet status="OK">
<DocumentSummary uid="346683357" term="346683357"><Caption>NC_016005</Caption><Title>Cucumis sativus mitochondrion chromosome 1, complete sequence</Title><Extra>gi|346683357|ref|NC_016005.1|</Extra><Gi>346683357</Gi><CreateDate>2011/09/14</CreateDate><UpdateDate>2023/04/03</UpdateDate><TaxId>3659</TaxId><Slen>1555935</Slen><Biomol>genomic</Biomol><MolType>dna</MolType><Topology>circular</Topology><SourceDb>refseq</SourceDb><ProjectId>927338</ProjectId><Genome>mitochondrion</Genome><SubType>cultivar|chromosome</SubType><SubName>Calypso|1</SubName><AssemblyGi>325305577</AssemblyGi><AssemblyAcc>HQ860792</AssemblyAcc><Completeness>complete</Completeness><GeneticCode>1</GeneticCode><Organism>Cucumis sativus</Organism><Flags>768</Flags><Properties na="1">1</Properties><OSLT indexed="yes">NC_016005.1</OSLT><AccessionVersion>NC_016005.1</AccessionVersion></DocumentSummary>
<DocumentSummary uid="3356463025" term="3356463025"><Caption>PV686508</Caption><Title>UNVERIFIED: Cucumis sativus isolate 9110gt chromosome 12 mitochondrion sequence</Title><Extra>gi|3356463025|gb|PV686508.1|</Extra><Gi>3356463025</Gi><CreateDate>2026/07/13</CreateDate><UpdateDate>2026/07/13</UpdateDate><TaxId>3659</TaxId><Slen>20516</Slen><Biomol>genomic</Biomol><MolType>dna</MolType><Topology>linear</Topology><SourceDb>insd</SourceDb><Genome>mitochondrion</Genome><SubType>isolate|chromosome</SubType><SubName>9110gt|12</SubName><GeneticCode>1</GeneticCode><Strand>ds</Strand><Organism>Cucumis sativus</Organism><Properties na="1">1</Properties><OSLT indexed="yes">PV686508.1</OSLT><AccessionVersion>PV686508.1</AccessionVersion></DocumentSummary>
<DocumentSummary uid="3356463037" term="3356463037"><Caption>PV686520</Caption><Title>UNVERIFIED: Cucumis sativus isolate 9930 chromosome 12 mitochondrion sequence</Title><Extra>gi|3356463037|gb|PV686520.1|</Extra><Gi>3356463037</Gi><CreateDate>2026/07/13</CreateDate><UpdateDate>2026/07/13</UpdateDate><TaxId>3659</TaxId><Slen>284</Slen><Biomol>genomic</Biomol><MolType>dna</MolType><Topology>linear</Topology><SourceDb>insd</SourceDb><Genome>mitochondrion</Genome><SubType>isolate|chromosome</SubType><SubName>9930|12</SubName><GeneticCode>1</GeneticCode><Strand>ds</Strand><Organism>Cucumis sativus</Organism><Properties na="1">1</Properties><OSLT indexed="yes">PV686520.1</OSLT><AccessionVersion>PV686520.1</AccessionVersion></DocumentSummary>
<DocumentSummary uid="3356463049" term="3356463049"><Caption>PV686532</Caption><Title>UNVERIFIED: Cucumis sativus isolate Cu2 chromosome 12 mitochondrion sequence</Title><Extra>gi|3356463049|gb|PV686532.1|</Extra><Gi>3356463049</Gi><CreateDate>2026/07/13</CreateDate><UpdateDate>2026/07/13</UpdateDate><TaxId>3659</TaxId><Slen>44815</Slen><Biomol>genomic</Biomol><MolType>dna</MolType><Topology>linear</Topology><SourceDb>insd</SourceDb><Genome>mitochondrion</Genome><SubType>isolate|chromosome</SubType><SubName>Cu2|12</SubName><GeneticCode>1</GeneticCode><Strand>ds</Strand><Organism>Cucumis sativus</Organism><Properties na="1">1</Properties><OSLT indexed="yes">PV686532.1</OSLT><AccessionVersion>PV686532.1</AccessionVersion></DocumentSummary>
</DocumentSummarySet></eSummaryResult>"""

SAMPLES_V2 = """<eSummaryResult><DocumentSummarySet status="OK">
<DocumentSummary uid="2419836744" term="2419836744"><Caption>MN787757</Caption><Title>Abelia chinensis var. chinensis isolate C144 chloroplast, complete genome</Title><Extra>gi|2419836744|gb|MN787757.1|</Extra><Gi>2419836744</Gi><CreateDate>2023/01/10</CreateDate><UpdateDate>2023/01/10</UpdateDate><TaxId>3018643</TaxId><Slen>157247</Slen><Biomol>genomic</Biomol><MolType>dna</MolType><Topology>circular</Topology><SourceDb>insd</SourceDb><Genome>chloroplast</Genome><SubType>isolate</SubType><SubName>C144</SubName><Completeness>complete</Completeness><GeneticCode>1</GeneticCode><Organism>Abelia chinensis var. chinensis</Organism><Flags>256</Flags><Properties na="1">1</Properties><OSLT indexed="yes">MN787757.1</OSLT><AccessionVersion>MN787757.1</AccessionVersion></DocumentSummary>
<DocumentSummary uid="2801150296" term="2801150296"><Caption>PQ303447</Caption><Title>Phoebe chekiangensis mitochondrion, complete genome</Title><Extra>gi|2801150296|gb|PQ303447.1|</Extra><Gi>2801150296</Gi><CreateDate>2026/08/19</CreateDate><UpdateDate>2026/08/19</UpdateDate><TaxId>610235</TaxId><Slen>865019</Slen><Biomol>genomic</Biomol><MolType>dna</MolType><Topology>circular</Topology><SourceDb>insd</SourceDb><Genome>mitochondrion</Genome><Completeness>complete</Completeness><GeneticCode>1</GeneticCode><Strand>ds</Strand><Organism>Phoebe chekiangensis</Organism><Flags>256</Flags><Properties na="1">1</Properties><OSLT indexed="yes">PQ303447.1</OSLT><AccessionVersion>PQ303447.1</AccessionVersion></DocumentSummary>
<DocumentSummary uid="669017085" term="669017085"><Caption>NC_024514</Caption><Title>Ptychomnion cygnisetum mitochondrion, complete genome</Title><Extra>gi|669017085|ref|NC_024514.1||gnl|NCBI_GENOMES|36669</Extra><Gi>669017085</Gi><CreateDate>2014/07/28</CreateDate><UpdateDate>2023/04/04</UpdateDate><TaxId>245469</TaxId><Slen>104480</Slen><Biomol>genomic</Biomol><MolType>dna</MolType><Topology>circular</Topology><SourceDb>refseq</SourceDb><ProjectId>927338</ProjectId><Genome>mitochondrion</Genome><SubType>specimen_voucher|specimen_voucher</SubType><SubName>CONN:Buck 58936|NYBG:Buck 58936</SubName><AssemblyGi>511370171</AssemblyGi><AssemblyAcc>KC784949</AssemblyAcc><Tech>standard</Tech><Completeness>complete</Completeness><GeneticCode>1</GeneticCode><Strand>ds</Strand><Organism>Ptychomnion cygnisetum</Organism><Flags>768</Flags><Properties na="1">1</Properties><OSLT indexed="yes">NC_024514.1</OSLT><AccessionVersion>NC_024514.1</AccessionVersion></DocumentSummary>
<DocumentSummary uid="2957582959" term="2957582959"><Caption>CM113015</Caption><Title>Cannabis sativa cultivar Punto Rojo x Cherry Pie isolate BP-2024a mitochondrion, complete sequence, whole genome shotgun sequence</Title><Extra>gi|2957582959|gb|CM113015.1||gnl|WGS:JBDLLE|PR:1:mt</Extra><Gi>2957582959</Gi><CreateDate>2025/04/17</CreateDate><UpdateDate>2026/10/14</UpdateDate><TaxId>3483</TaxId><Slen>414691</Slen><Biomol>genomic</Biomol><MolType>dna</MolType><Topology>circular</Topology><SourceDb>insd</SourceDb><ProjectId>1058441</ProjectId><Genome>mitochondrion</Genome><SubType>cultivar|isolate|country|isolation_source|lat_lon|collection_date</SubType><SubName>Punto Rojo x Cherry Pie|BP-2024a|Colombia: Fuente de Oro, Meta|Agricultural soil [ENVO:00005897]||3.5 N 73.6 W|2021</SubName><Tech>wgs</Tech><Completeness>complete</Completeness><GeneticCode>1</GeneticCode><Organism>Cannabis sativa</Organism><BioSample>SAMN39175052</BioSample><Flags>288</Flags><Properties na="1">1</Properties><OSLT indexed="yes">CM113015.1</OSLT><AccessionVersion>CM113015.1</AccessionVersion></DocumentSummary>
</DocumentSummarySet></eSummaryResult>"""


def test_every_v2_field_is_kept_verbatim() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    calypso = parse_esummary_xml(CUCUMBER_V2)[0]
    summary = calypso["summary"]
    for field in (
        "Organism", "TaxId", "SubType", "SubName", "ProjectId", "AssemblyAcc", "Genome",
        "Completeness", "Topology", "SourceDb", "Biomol", "MolType", "CreateDate",
        "UpdateDate", "Extra", "GeneticCode", "Flags", "OSLT",
    ):  # fmt: skip
        assert field in summary, field
    assert summary["Extra"] == "gi|346683357|ref|NC_016005.1|"
    assert summary["Topology"] == "circular"
    assert "Status" not in summary  # not sent, so not invented


def test_cucumber_samples_are_told_apart_by_their_qualifiers() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    records = parse_esummary_xml(CUCUMBER_V2)
    assert {r["organism"] for r in records} == {"Cucumis sativus"}
    assert [r["source_qualifiers"] for r in records] == [
        {"cultivar": "Calypso", "chromosome": "1"},
        {"isolate": "9110gt", "chromosome": "12"},
        {"isolate": "9930", "chromosome": "12"},
        {"isolate": "Cu2", "chromosome": "12"},
    ]
    assert [(r["cultivar"], r["isolate"], r["chromosome"]) for r in records] == [
        ("Calypso", "", "1"),
        ("", "9110gt", "12"),
        ("", "9930", "12"),
        ("", "Cu2", "12"),
    ]
    assert records[0]["assembly_accession"] == "HQ860792"


def test_esummary_qualifiers_keep_cucumber_samples_as_separate_genomes() -> None:
    """The grouping keys are the GenBank-tier names, so esummary alone is
    enough to keep four samples as four genomes rather than one chimera."""
    from organelleverse.fetch.assembly import group_records
    from organelleverse.fetch.ncbi import parse_esummary_xml

    units = group_records(parse_esummary_xml(CUCUMBER_V2))
    assert len(units) == 4
    assert all(unit.molecule_count == 1 for unit in units)


def test_isolate_missing_subtype_repeats_and_an_unpairable_record() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    abelia, phoebe, ptychomnion, cannabis = parse_esummary_xml(SAMPLES_V2)

    assert abelia["organism"] == "Abelia chinensis var. chinensis"
    assert abelia["isolate"] == "C144"
    assert abelia["source_qualifiers"] == {"isolate": "C144"}

    # No SubType at all: nothing to structure, nothing invented.
    assert phoebe["source_qualifiers"] == {}
    assert (phoebe["cultivar"], phoebe["isolate"], phoebe["biosample"]) == ("", "", "")

    # A type named twice keeps both values, as NCBI sent them.
    assert ptychomnion["specimen_voucher"] == "CONN:Buck 58936|NYBG:Buck 58936"

    # Six types, seven values: the pairing cannot be known, so it is not
    # guessed. The raw strings and the other fields survive.
    assert cannabis["source_qualifiers"] == {}
    assert cannabis["summary"]["SubType"].count("|") == 5
    assert cannabis["biosample"] == "SAMN39175052"


def test_source_qualifiers_pairs_type_and_name() -> None:
    from organelleverse.fetch.ncbi import source_qualifiers

    assert source_qualifiers("ecotype", "Col-0") == {"ecotype": "Col-0"}  # NC_037304.1
    assert source_qualifiers("", "") == {}
    assert source_qualifiers("isolate|note|note", "95|male sterility line|with M. cytoplasm") == {
        "isolate": "95",
        "note": "male sterility line|with M. cytoplasm",
    }  # MG872829.1, shortened


def test_a_v1_summary_keeps_its_items_and_has_no_qualifiers() -> None:
    from organelleverse.fetch.ncbi import parse_esummary_xml

    (record,) = parse_esummary_xml(REAL_ESUMMARY)
    assert record["summary"]["Caption"] == "OP474144"
    assert record["summary"]["Status"] == "live"
    assert record["source_qualifiers"] == {}
    assert record["cultivar"] == "" and record["biosample"] == ""


def test_the_csv_export_keeps_scalar_fields_and_leaves_the_maps_to_jsonl(tmp_path) -> None:
    import csv

    from organelleverse.fetch.export import write_metadata_csv
    from organelleverse.fetch.ncbi import parse_esummary_xml

    path = write_metadata_csv(parse_esummary_xml(CUCUMBER_V2), tmp_path / "m.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "summary" not in rows[0] and "source_qualifiers" not in rows[0]
    assert {r["isolate"] for r in rows} == {"", "9110gt", "9930", "Cu2"}
