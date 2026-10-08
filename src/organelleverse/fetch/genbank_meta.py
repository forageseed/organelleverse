"""GenBank XML metadata — the tier that makes the good filters possible.

``esummary`` gives length and dates. It does not give gene counts, CDS counts,
ambiguous-base counts, collection date, country, or submitter. Those live in the
full GBSeq record, so a filter on any of them has to pull ``rettype=gb&
retmode=xml`` first.

For organelles this tier is not a luxury: a plastome carries ~130 genes and a
plant mitogenome ~60, so a record claiming 12 is a fragment mislabelled as a
complete genome — and gene count is the cheapest way to see that.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any
from xml.etree import ElementTree

from ..core.errors import OrganelleExecutionError
from ._http import Transport, eutils_delay, ncbi_api_key, retry_with_backoff
from .filters import count_ambiguous

__all__ = [
    "GENBANK_ONLY_FIELDS",
    "derived_from",
    "enrich_with_genbank_metadata",
    "parse_gbseq_xml",
]

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Fields that exist only once GBSeq XML has been fetched. They are written even
# when empty, so a consumer can distinguish "we looked and it says nothing" from
# "we never looked". dedup depends on that distinction: a record with no
# COMPLETENESS statement is not the same as a record we never enriched.
GENBANK_ONLY_FIELDS = frozenset(
    {
        "gene_count",
        "protein_count",
        "annotated",
        "ambiguous_count",
        "chromosome",
        "bioproject",
        "biosample",
        "doi",
        "collection_date",
        "geo_location",
        "isolate",
        "isolation_source",
        "cultivar",
        "specimen_voucher",
        "submitter_name",
        "submitter_institution",
        "submitter_country",
        "topology",
        "completeness",
        "is_refseq",
        "derived_from",
        "comment",
    }
)

# "Submitted (12-FEB-2023) Institute of Botany, CAS, Beijing, China"
_SUBMITTED = re.compile(r"Submitted\s*\([^)]*\)\s*(.+)", re.IGNORECASE)

# A RefSeq record naming the GenBank submission it curates. This is the only
# machine-readable evidence that two accessions are the same genome, and it is
# what makes L1 dedup exact rather than a guess.
#
# NCBI uses TWO wordings, and both appear in real records:
#
#   "The reference sequence was derived from BA000029."     (Oryza)
#   "The reference sequence is identical to HQ860792."      (Cucumis)
#
# Matching only the first silently misses every mirror worded the second way —
# which is all three cucumber chromosomes.
_DERIVED_FROM = re.compile(
    r"(?:derived from|identical to)\s+([A-Z]{1,2}\d{5,8}(?:\.\d+)?)",
    re.IGNORECASE,
)


def derived_from(comment: str) -> str:
    """The source accession a RefSeq record was curated from, if it says so."""
    match = _DERIVED_FROM.search(comment or "")
    return match.group(1) if match else ""


def _text(node: ElementTree.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def _qualifiers(feature: ElementTree.Element) -> dict[str, str]:
    quals: dict[str, str] = {}
    for qual in feature.findall("./GBFeature_quals/GBQualifier"):
        name = _text(qual.find("GBQualifier_name"))
        if name:
            quals[name] = _text(qual.find("GBQualifier_value"))
    return quals


def _xrefs(seq: ElementTree.Element) -> dict[str, str]:
    """DBLINK entries (BioProject / BioSample / doi) — candidate grouping keys."""
    out: dict[str, str] = {}
    for xref in seq.findall("./GBSeq_xrefs/GBXref"):
        name = _text(xref.find("GBXref_dbname")).lower()
        value = _text(xref.find("GBXref_id"))
        if name and value:
            out[name] = value
    return out


def _submitter(seq: ElementTree.Element) -> tuple[str, str, str]:
    """Return ``(name, institution, country)`` from the submission reference."""
    for reference in seq.findall("./GBSeq_references/GBReference"):
        journal = _text(reference.find("GBReference_journal"))
        match = _SUBMITTED.search(journal)
        if not match:
            continue
        address = match.group(1).strip().rstrip(".")
        parts = [p.strip() for p in address.split(",") if p.strip()]
        institution = parts[0] if parts else ""
        country = parts[-1] if len(parts) > 1 else ""
        authors = reference.findall("./GBReference_authors/GBAuthor")
        name = _text(authors[0]) if authors else ""
        return name, institution, country
    return "", "", ""


def parse_gbseq_xml(text: str | bytes) -> list[dict[str, Any]]:
    """Parse ``efetch(rettype=gb, retmode=xml)`` into metadata records."""
    raw = text.decode(errors="replace") if isinstance(text, bytes) else text
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"GenBank XML did not parse: {exc}",
            details={"body": raw[:200]},
            retryable=True,
        ) from exc

    records: list[dict[str, Any]] = []
    for seq in root.iter("GBSeq"):
        accession = _text(seq.find("GBSeq_accession-version"))
        if not accession:
            continue

        gene_count = 0
        protein_count = 0
        source: dict[str, str] = {}
        for feature in seq.findall("./GBSeq_feature-table/GBFeature"):
            key = _text(feature.find("GBFeature_key"))
            if key == "gene":
                gene_count += 1
            elif key == "CDS":
                protein_count += 1
            elif key == "source" and not source:
                source = _qualifiers(feature)

        sequence = _text(seq.find("GBSeq_sequence"))
        name, institution, country = _submitter(seq)
        xrefs = _xrefs(seq)
        comment = _text(seq.find("GBSeq_comment"))
        keywords = [_text(k) for k in seq.findall("./GBSeq_keywords/GBKeyword")]
        lowered = comment.lower()

        records.append(
            {
                "accession": accession,
                "length": int(_text(seq.find("GBSeq_length")) or 0),
                "create_date": _text(seq.find("GBSeq_create-date")),
                "update_date": _text(seq.find("GBSeq_update-date")),
                "organism": _text(seq.find("GBSeq_organism")),
                "gene_count": gene_count,
                "protein_count": protein_count,
                "annotated": gene_count > 0 or protein_count > 0,
                "ambiguous_count": count_ambiguous(sequence) if sequence else 0,
                # A `/chromosome` qualifier means this record is ONE molecule of a
                # multipartite genome, not a whole genome. Single-circle mitogenomes
                # (rice) have no such qualifier; multipartite ones (cucumber) do.
                "chromosome": source.get("chromosome", ""),
                # Quality signals. Organelle nuccore records carry no
                # assembly_level (that belongs to nuclear assemblies), but they
                # do state completeness outright, and that is the stronger claim.
                "topology": _text(seq.find("GBSeq_topology")),
                "completeness": (
                    "full"
                    if "full length" in lowered
                    else "partial"
                    if "partial" in lowered
                    else ""
                ),
                "is_refseq": "RefSeq" in keywords,
                "derived_from": derived_from(comment),
                "comment": comment[:300],
                "bioproject": xrefs.get("bioproject", ""),
                "biosample": xrefs.get("biosample", ""),
                "doi": xrefs.get("doi", ""),
                "collection_date": source.get("collection_date", ""),
                # NCBI renamed `country` to `geo_loc_name` in 2024; accept both.
                "geo_location": source.get("geo_loc_name") or source.get("country", ""),
                "isolate": source.get("isolate", ""),
                "isolation_source": source.get("isolation_source", ""),
                "cultivar": source.get("cultivar", ""),
                "specimen_voucher": source.get("specimen_voucher", ""),
                "submitter_name": name,
                "submitter_institution": institution,
                "submitter_country": country,
            }
        )
    return records


def _with_key(url: str) -> str:
    key = ncbi_api_key()
    return f"{url}&api_key={key}" if key else url


def fetch_genbank_metadata(
    http: Transport,
    *,
    accessions: Sequence[str],
    batch_size: int = 10,
) -> dict[str, dict[str, Any]]:
    """Fetch GBSeq metadata for accessions, keyed by accession.

    Batched over POST-free GET. The batch default is 50: real plant-mitochondria
    queries return thousands of accessions, and larger batches exceed the
    E-utilities GET URI limit (HTTP 414) once accession lengths and the API-key
    parameter are included.
    """
    import time

    out: dict[str, dict[str, Any]] = {}
    ordered = list(accessions)
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        url = _with_key(
            f"{EUTILS}/efetch.fcgi?db=nuccore&rettype=gb&retmode=xml&id={','.join(batch)}"
        )
        body = retry_with_backoff(lambda url=url: http.get(url, timeout=600.0))
        for record in parse_gbseq_xml(body):
            out[record["accession"]] = record
        delay = eutils_delay()
        if delay:
            time.sleep(delay)
    return out


def enrich_with_genbank_metadata(
    http: Transport,
    *,
    records: list[dict[str, Any]],
    batch_size: int = 200,
) -> list[dict[str, Any]]:
    """Merge GBSeq fields into esummary records, keeping esummary as the base."""
    if not records:
        return records
    accessions = [str(r["accession"]) for r in records]
    rich = fetch_genbank_metadata(http, accessions=accessions, batch_size=batch_size)
    merged: list[dict[str, Any]] = []
    for record in records:
        extra = rich.get(str(record["accession"]), {})
        combined = dict(record)
        for key, value in extra.items():
            # A GenBank-only field is written even when empty. "Absent" would
            # mean we never looked; "empty" means we looked and the submitter
            # said nothing — and a consumer (dedup) has to tell those apart.
            # Fields esummary already supplies are only overwritten by a real
            # value, so a blank GBSeq organism cannot erase a good one.
            if key in GENBANK_ONLY_FIELDS or value not in ("", None):
                combined[key] = value
        merged.append(combined)
    return merged
