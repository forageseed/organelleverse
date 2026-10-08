"""NCBI channels: a deterministic RefSeq snapshot and a live Entrez query.

Two channels, two operations — because they differ in the one property that
matters for a scientific claim:

* ``refseq_snapshot`` verifies ``RELEASE_NUMBER`` before reading NCBI's live
  release directory. Historical release sequence files are not archived there,
  so unavailable releases are rejected rather than relabelled.
* ``entrez_query`` reads a live database that changes daily. It is **not**.

Merging them would force one of the two specs to lie. The contract field
``deterministic`` is what forced this split, which is the whole point of
declaring it.

Staged retrieval (after gget virus, arXiv 2606.06749): metadata is fetched and
filtered *before* any sequence is downloaded, so a record that will be discarded
is never transferred.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

from ..core.artifacts import ArtifactRef
from ..core.data import OrganelleData
from ..core.errors import OrganelleExecutionError, OrganelleParameterError
from ._http import (
    Transport,
    default_transport,
    eutils_delay,
    ncbi_api_key,
    retry_with_backoff,
)
from .assembly import group_records
from .cache import read_baseline, write_jsonl
from .export import genbank_to_fasta, write_metadata_csv
from .filters import FetchFilters, apply_filters, apply_filters_to_units
from .genbank_meta import enrich_with_genbank_metadata
from .manifest import build_manifest

__all__ = [
    "History",
    "build_organelle_term",
    "entrez_query",
    "fetch_accessions",
    "name_variants",
    "parse_esearch_xml",
    "parse_esummary_xml",
    "parse_history_xml",
    "parse_release_number",
    "refseq_snapshot",
]

Organelle = Literal["mitochondrion", "plastid"]

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REFSEQ_RELEASE = "https://ftp.ncbi.nlm.nih.gov/refseq/release"

# Entrez calls organelles by their own filter names.
_ORGANELLE_FILTER = {
    "plastid": "chloroplast[filter]",
    "mitochondrion": "mitochondrion[filter]",
}

# The completeness clause CANNOT be shared between the two organelles, because
# `"complete sequence"[Title]` means opposite things in each (verified against
# Viridiplantae, 2026-07-13):
#
#   mitochondrion  11,607 hits — this is how a multipartite chromosome is titled
#                  ("Cucumis sativus mitochondrion chromosome 2, complete
#                  sequence"). Without it, every multi-chromosome mitogenome is
#                  invisible: 2,699 hits instead of 14,651.
#
#   chloroplast   237,010 hits — and 236,172 of them (99.6%) are under 10 kb.
#                  They are gene and spacer records ("trnD-trnY intergenic
#                  spacer, complete sequence"), not genomes. Including it turns
#                  58,082 plastomes into 295,091 mostly-fragments.
#
# Plastids are also titled in ways that never say "complete": `genome assembly,
# organelle: plastid:chloroplast` and whole-genome-shotgun submissions are real
# 154 kb plastomes. Those are worth 3,136 genomes the old clause silently missed.
_COMPLETENESS_CLAUSE = {
    "mitochondrion": (
        '("complete genome"[Title] OR "complete sequence"[Title] OR chromosome[Title])'
    ),
    "plastid": (
        '("complete genome"[Title] OR "genome assembly"[Title] '
        'OR "whole genome shotgun"[Title] OR chromosome[Title])'
    ),
}
# RefSeq release directories use a third naming again.
_REFSEQ_DIR = {"plastid": "plastid", "mitochondrion": "mitochondrion"}

# NCBI's own per-request ceiling for esummary/efetch.
_PAGE_SIZE = 500
#: efetch GET batches must stay small on two axes: URI length (HTTP 414 at
#: hundreds of accessions) and response size - full GBSeq XML for ten plant
#: organelle genomes is ~3-5 MB, comfortably inside NCBI's chunked delivery;
#: fifty hit IncompleteRead truncation in real plant-mitochondria queries.
_EFETCH_BATCH = 10


def _merge_filters(filters: FetchFilters | None, *, min_length: int | None) -> FetchFilters:
    """``min_length=`` stays as a shortcut for the most common single filter."""
    if filters is None:
        return FetchFilters(min_length=min_length)
    if min_length is not None and filters.min_length is None:
        return FetchFilters(**{**filters.__dict__, "min_length": min_length})
    return filters


# Plant nomenclature is messier than viral nomenclature: subspecies, hybrid
# crosses, and abbreviated genera all fail an exact [Organism] match.
def name_variants(taxon: str) -> list[str]:
    """Fallbacks to try when an exact organism match returns nothing."""
    cleaned = " ".join(taxon.split())
    out: list[str] = []
    # U+00D7 is deliberate: it is the hybrid sign in botanical names
    # ("Triticum x aestivum" written with the multiplication sign), which
    # nuccore does not match literally.
    without_x = cleaned.replace("×", "").replace(" x ", " ").strip()  # noqa: RUF001
    if without_x != cleaned:
        out.append(without_x)
    words = cleaned.split()
    # "Genus species subsp. foo" -> "Genus species"
    if len(words) > 2:
        out.append(" ".join(words[:2]))
    # "Genus species" -> "Genus" (whole genus)
    if len(words) >= 2:
        out.append(words[0])
    seen: set[str] = {cleaned}
    unique: list[str] = []
    for candidate in out:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


# ─────────────────────────────────────────────────────────────
# Query construction — this is where a claim gets its scope
# ─────────────────────────────────────────────────────────────


def build_organelle_term(
    *,
    organelle: str,
    taxon: str | None = None,
    complete_only: bool = True,
) -> str:
    """Build the Entrez term for one organelle.

    ``complete_only=False`` matters for plant mitochondria: there are ~21x fewer
    complete plant mitogenomes than plastomes, so recall sometimes beats
    precision and the completeness filter has to come off.
    """
    try:
        organelle_filter = _ORGANELLE_FILTER[organelle]
    except KeyError:
        raise OrganelleParameterError(
            code="input.unknown_organelle",
            message=f"organelle must be 'mitochondrion' or 'plastid', got {organelle!r}",
            details={"organelle": organelle},
        ) from None

    parts = [organelle_filter]
    if taxon:
        parts.insert(0, f"{taxon}[Organism]")
    if complete_only:
        parts.append(_COMPLETENESS_CLAUSE[organelle])
    return " AND ".join(parts)


# ─────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────


def parse_release_number(text: str | bytes) -> int:
    """Parse ``refseq/release/RELEASE_NUMBER`` — the snapshot identity."""
    raw = text.decode() if isinstance(text, bytes) else text
    candidate = raw.strip()
    if not candidate.isdigit():
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"RELEASE_NUMBER is not a number: {candidate[:60]!r}",
            details={"body": candidate[:200]},
            retryable=True,
        )
    return int(candidate)


def _parse_xml(text: str | bytes, *, expect: str) -> ElementTree.Element:
    raw = text.decode() if isinstance(text, bytes) else text
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"NCBI returned unparseable XML: {exc}",
            details={"body": raw[:200]},
            retryable=True,
        ) from exc
    if root.tag != expect:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"expected <{expect}>, got <{root.tag}> (likely an NCBI error page)",
            details={"root": root.tag, "body": raw[:200]},
            retryable=True,
        )
    return root


def parse_esearch_xml(text: str | bytes) -> tuple[int, list[str]]:
    """Return ``(total_count, uids_on_this_page)``."""
    root = _parse_xml(text, expect="eSearchResult")
    count_node = root.find("Count")
    count = int(count_node.text or 0) if count_node is not None else 0
    uids = [node.text or "" for node in root.findall("./IdList/Id")]
    return count, [uid for uid in uids if uid]


@dataclass(frozen=True)
class History:
    """An NCBI server-side result set (``WebEnv`` + ``query_key``).

    This is what makes a full-corpus pull possible. Without it, a caller is
    limited to whatever one ``retmax`` can carry, and the 58,082 complete
    Viridiplantae plastomes cannot be retrieved at all.
    """

    count: int
    web_env: str
    query_key: str

    def pages(self, page_size: int) -> list[tuple[int, int]]:
        """``(retstart, retmax)`` pairs covering the whole result set."""
        return [
            (start, min(page_size, self.count - start)) for start in range(0, self.count, page_size)
        ]


def parse_history_xml(text: str | bytes, *, expect: str) -> History:
    """Pull ``Count`` / ``WebEnv`` / ``QueryKey`` from an esearch or epost reply."""
    root = _parse_xml(text, expect=expect)
    count_node = root.find("Count")
    web_env_node = root.find("WebEnv")
    query_key_node = root.find("QueryKey")

    if web_env_node is None or query_key_node is None:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"<{expect}> carried no WebEnv/QueryKey (history was not created)",
            details={"root": root.tag},
            retryable=True,
        )
    # epost has no Count; the caller knows how many it posted.
    count = int(count_node.text or 0) if count_node is not None else 0
    return History(
        count=count,
        web_env=web_env_node.text or "",
        query_key=query_key_node.text or "",
    )


def _organism_from_title(title: str) -> str:
    """Best-effort species from a nuccore title.

    nuccore esummary has no ``Organism`` item — only ``TaxId`` and a ``Title``
    like "Arabidopsis thaliana chloroplast, complete genome". Cut at the
    organelle word. This is a heuristic; ``taxid`` is the authoritative field.
    """
    lowered = title.lower()
    for marker in (" chloroplast", " mitochondrion", " plastid", " mitochondrial"):
        index = lowered.find(marker)
        if index > 0:
            return title[:index].strip()
    return title.split(",")[0].strip()


def parse_esummary_xml(text: str | bytes) -> list[dict[str, Any]]:
    """Return one metadata record per DocSum — no sequence is touched here.

    Field names come from a real nuccore response: the length item is
    ``Length`` (``Slen`` is accepted as a fallback), and there is no
    ``Organism`` item at all.
    """
    root = _parse_xml(text, expect="eSummaryResult")
    records: list[dict[str, Any]] = []
    for docsum in root.findall("DocSum"):
        item: dict[str, Any] = {}
        for node in docsum.findall("Item"):
            item[node.get("Name", "")] = node.text or ""
        accession = item.get("AccessionVersion") or item.get("Caption") or ""
        if not accession:
            continue
        try:
            length = int(item.get("Length") or item.get("Slen") or 0)
        except ValueError:
            length = 0
        title = item.get("Title", "")
        records.append(
            {
                "accession": accession,
                "title": title,
                "organism": _organism_from_title(title),
                "taxid": item.get("TaxId", ""),
                "length": length,
                "status": item.get("Status", ""),
                "create_date": item.get("CreateDate", ""),
                "update_date": item.get("UpdateDate", ""),
            }
        )
    return records


# "Cucumis sativus mitochondrion chromosome 1, complete sequence"
# A free signal from esummary that a genome spans several accessions — no need
# to pay for GenBank XML just to find out whether we must.
_TITLE_CHROMOSOME = re.compile(r"\bchromosome\s+(\S+?)[,\s]", re.IGNORECASE)


def chromosome_from_title(title: str) -> str:
    match = _TITLE_CHROMOSOME.search(title or "")
    return match.group(1).strip() if match else ""


# How many records of one organism it takes before "several complete genomes of
# the same species" stops being plausible and starts meaning "one genome in
# several molecules". Two is normal (a RefSeq copy plus its GenBank original);
# three or more is not.
_SUSPICIOUS_RECORD_COUNT = 3


def suspect_multipartite(records: Sequence[dict[str, Any]]) -> bool:
    """Decide whether the genomes here might span multiple accessions.

    The obvious signal is a title that says ``chromosome 1``. It is not enough.
    Begonia fimbristipula deposits its nine mitochondrial chromosomes as nine
    records **each titled "mitochondrion, complete genome"** — the title lies,
    and only the ``/chromosome`` qualifier in the GenBank record tells the
    truth. Downloading the 2,354 bp one and calling it a mitogenome is exactly
    the failure this guard exists to prevent.

    So a second, cheaper signal: one organism with three or more records that
    all claim to be complete genomes is not three genomes, it is one genome in
    pieces. False positives here only cost a metadata fetch; a false negative
    costs a truncated genome nobody notices.
    """
    if any(chromosome_from_title(str(r.get("title", ""))) for r in records):
        return True
    counts: dict[str, int] = {}
    for record in records:
        organism = str(record.get("organism", "")).strip()
        if organism:
            counts[organism] = counts.get(organism, 0) + 1
    return any(n >= _SUSPICIOUS_RECORD_COUNT for n in counts.values())


_VERSION_RE = re.compile(r"^VERSION\s+(\S+)", re.MULTILINE)
_LOCUS_RE = re.compile(r"^LOCUS\s+(\S+)", re.MULTILINE)


def _accessions_in_genbank(text: str) -> list[str]:
    found = _VERSION_RE.findall(text)
    return found or _LOCUS_RE.findall(text)


# ─────────────────────────────────────────────────────────────
# Channel B: live Entrez query (NOT deterministic)
# ─────────────────────────────────────────────────────────────


def entrez_query(
    *,
    organelle: str,
    dest: str | Path,
    taxon: str | None = None,
    complete_only: bool = True,
    filters: FetchFilters | None = None,
    min_length: int | None = None,
    max_records: int | None = None,
    page_size: int = _PAGE_SIZE,
    genbank_metadata: bool = False,
    group_genomes: bool = True,
    baseline: str | Path | None = None,
    formats: Sequence[str] = ("genbank",),
    try_name_variants: bool = True,
    transport: Transport | None = None,
) -> OrganelleData:
    """Fetch organelle records from live nuccore, metadata first.

    ``max_records=None`` means *all of them*: the search is run on NCBI's
    history server and paged, so a whole-corpus pull (58,082 complete
    Viridiplantae plastomes) is a normal call, not a special case.

    Not reproducible across time by construction — nuccore changes daily. Use
    :func:`refseq_snapshot` when you need a claim that holds.
    """
    http = transport or default_transport()
    active = _merge_filters(filters, min_length=min_length)
    if active.requires_genbank_metadata() and not genbank_metadata:
        genbank_metadata = True  # the filter asked for it; fetch what it needs

    term = build_organelle_term(organelle=organelle, taxon=taxon, complete_only=complete_only)
    history = _esearch_history(http, term=term)

    # A plant synonym that nuccore does not know: try the usual variants before
    # reporting zero. Plant nomenclature is messier than viral nomenclature.
    tried = [term]
    if history.count == 0 and taxon and try_name_variants:
        for variant in name_variants(taxon):
            candidate = build_organelle_term(
                organelle=organelle, taxon=variant, complete_only=complete_only
            )
            tried.append(candidate)
            history = _esearch_history(http, term=candidate)
            if history.count:
                term = candidate
                taxon = variant
                break

    scope: dict[str, Any] = {
        "organelle": organelle,
        "taxon": taxon or "",
        "complete_only": complete_only,
        "genbank_metadata": genbank_metadata,
        **active.active(),
    }

    def empty() -> OrganelleData:
        return _as_data(
            manifest=build_manifest(
                source="entrez_query",
                organelle=organelle,
                accessions=[],
                scope=scope,
                query=term,
                hits_before_filter=history.count,
                examined=examined,
                names_tried=tried,
            ),
            artifact=None,
        )

    examined = 0
    if history.count == 0:
        return empty()

    # Stage 2 — metadata only, paged. No sequence has crossed the wire yet.
    wanted = history.count if max_records is None else min(max_records, history.count)
    records = _esummary_paged(http, history=history, limit=wanted, page_size=page_size)
    # What we actually looked at — NOT history.count. Conflating the two would
    # claim the filter rejected records it never fetched metadata for.
    examined = len(records)

    # Stage 2b — is any genome multipartite? The esummary title says so for free
    # ("... chromosome 1, complete sequence"), so we only pay for GenBank XML
    # when a genome actually spans several accessions and must be grouped.
    multipartite = suspect_multipartite(records)
    if group_genomes and multipartite and not genbank_metadata:
        genbank_metadata = True
        scope["genbank_metadata"] = True
    scope["multipartite_detected"] = multipartite

    # Stage 2c — richer metadata, when a filter or the grouping needs it.
    if genbank_metadata:
        records = enrich_with_genbank_metadata(
        http, records=records, batch_size=min(page_size, _EFETCH_BATCH)
    )

    # Stage 3 — filter, then keep every molecule of a genome that survived.
    #
    # Filtering molecule-by-molecule is what shreds a multipartite genome: a
    # 200 kb floor would delete the cucumber mitochondrion's 83 kb and 44 kb
    # chromosomes and leave chromosome 1 masquerading as the whole genome. So
    # the predicate runs on the genome's totals, and a surviving genome brings
    # all of its molecules with it.
    if group_genomes:
        units = group_records(records)
        kept_units = apply_filters_to_units(units, active, have_genbank_metadata=genbank_metadata)
        records = [molecule for unit in kept_units for molecule in unit.molecules]
        genomes = [unit.as_record() for unit in kept_units]
        incomplete = [g for g in genomes if not g["complete"]]
    else:
        records = apply_filters(records, active, have_genbank_metadata=genbank_metadata)
        genomes, incomplete = [], []

    # Stage 3b — drop what a previous run already has (resume / incremental).
    seen: set[str] = read_baseline(baseline) if baseline else set()
    if seen:
        records = [r for r in records if str(r["accession"]) not in seen]

    accessions = sorted({str(r["accession"]) for r in records})
    if not accessions:
        return empty()

    # Stage 4 — sequences for the survivors only.
    path = _efetch_accessions(
        http, accessions=accessions, dest=Path(dest), stem=organelle, page_size=page_size
    )
    exports = _write_exports(
        formats=formats, records=records, genbank=path, dest=Path(dest), stem=organelle
    )

    return _as_data(
        manifest=build_manifest(
            source="entrez_query",
            organelle=organelle,
            accessions=accessions,
            scope=scope,
            query=term,
            hits_before_filter=history.count,
            examined=examined,
            names_tried=tried,
            baseline_skipped=len(seen),
            genomes=genomes,
            incomplete_genomes=incomplete,
        ),
        artifact=path,
        extra_artifacts=exports,
        records=records,
        genomes=genomes,
    )


def fetch_accessions(
    *,
    accessions: Sequence[str],
    dest: str | Path,
    organelle: str = "plastid",
    genbank_metadata: bool = False,
    formats: Sequence[str] = ("genbank",),
    page_size: int = _PAGE_SIZE,
    transport: Transport | None = None,
) -> OrganelleData:
    """Fetch an explicit accession list (gget's ``--is_accession``).

    This is how you reproduce someone else's dataset: give the accessions, get
    exactly those records, and a set hash to prove it.
    """
    http = transport or default_transport()
    wanted = sorted({a.strip() for a in accessions if a and a.strip()})
    if not wanted:
        raise OrganelleParameterError(
            code="input.empty_accession_list",
            message="fetch_accessions() needs at least one accession",
        )

    records = _esummary_by_accession(http, accessions=wanted, page_size=page_size)
    if genbank_metadata:
        records = enrich_with_genbank_metadata(http, records=records, batch_size=page_size)

    found = sorted({str(r["accession"]) for r in records})
    missing = [a for a in wanted if not any(f.startswith(a.split(".")[0]) for f in found)]

    path = _efetch_accessions(
        http, accessions=found, dest=Path(dest), stem=organelle, page_size=page_size
    )
    exports = _write_exports(
        formats=formats, records=records, genbank=path, dest=Path(dest), stem=organelle
    )
    return _as_data(
        manifest=build_manifest(
            source="accession_list",
            organelle=organelle,
            accessions=found,
            scope={"organelle": organelle, "requested": len(wanted)},
            query=f"{len(wanted)} accessions",
            missing=missing,
        ),
        artifact=path,
        extra_artifacts=exports,
        records=records,
    )


# ─────────────────────────────────────────────────────────────
# Channel A: pinned RefSeq release (deterministic)
# ─────────────────────────────────────────────────────────────


def refseq_snapshot(
    *,
    organelle: str,
    dest: str | Path,
    release: int | None = None,
    transport: Transport | None = None,
) -> OrganelleData:
    """Download the RefSeq release snapshot for one organelle.

    NCBI only keeps sequence files for the current release in this directory.
    An explicit release is therefore accepted only while it matches
    ``RELEASE_NUMBER``; historical releases are rejected rather than silently
    relabelling current data.
    """
    if organelle not in _REFSEQ_DIR:
        raise OrganelleParameterError(
            code="input.unknown_organelle",
            message=f"organelle must be 'mitochondrion' or 'plastid', got {organelle!r}",
            details={"organelle": organelle},
        )

    http = transport or default_transport()
    current = _current_release(http)
    if release is not None and release != current:
        raise OrganelleParameterError(
            code="input.refseq_release_unavailable",
            message=(
                f"RefSeq release {release} cannot be fetched from NCBI's current-only "
                f"release directory (current release: {current})"
            ),
            details={"requested": release, "current": current},
        )
    pinned = current

    directory = f"{REFSEQ_RELEASE}/{_REFSEQ_DIR[organelle]}/"
    index = retry_with_backoff(lambda: http.get(directory, timeout=120.0))
    names = _genbank_files_in_index(index)

    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"refseq_{organelle}_release{pinned}.gbff"

    accessions: list[str] = []
    with out.open("w", encoding="utf-8") as handle:
        for name in names:
            url = f"{directory}{name}"
            blob = retry_with_backoff(lambda url=url: http.get(url, timeout=600.0))
            text = _maybe_gunzip(blob)
            accessions.extend(_accessions_in_genbank(text))
            handle.write(text)

    return _as_data(
        manifest=build_manifest(
            source="refseq_release",
            organelle=organelle,
            accessions=accessions,
            scope={"organelle": organelle, "refseq_release": pinned, "complete_only": True},
            query=f"refseq/release/{_REFSEQ_DIR[organelle]}",
            refseq_release=pinned,
        ),
        artifact=out if accessions else None,
    )


def _current_release(http: Transport) -> int:
    body = retry_with_backoff(lambda: http.get(f"{REFSEQ_RELEASE}/RELEASE_NUMBER", timeout=60.0))
    return parse_release_number(body)


_HREF_RE = re.compile(r'href="([^"]+\.genomic\.gbff\.gz)"')


def _genbank_files_in_index(index: bytes | str) -> list[str]:
    raw = index.decode(errors="replace") if isinstance(index, bytes) else index
    return sorted(set(_HREF_RE.findall(raw)))


def _maybe_gunzip(blob: bytes) -> str:
    if blob[:2] == b"\x1f\x8b":
        return gzip.decompress(blob).decode(errors="replace")
    return blob.decode(errors="replace")


# ─────────────────────────────────────────────────────────────
# E-utilities calls
# ─────────────────────────────────────────────────────────────


def _with_key(url: str) -> str:
    key = ncbi_api_key()
    return f"{url}&api_key={key}" if key else url


def _esearch_history(http: Transport, *, term: str) -> History:
    """Run the search on NCBI's history server so the result set can be paged."""
    from urllib.parse import quote_plus

    url = _with_key(
        f"{EUTILS}/esearch.fcgi?db=nuccore&term={quote_plus(term)}&usehistory=y&retmax=0"
    )
    body = retry_with_backoff(lambda: http.get(url, timeout=120.0))
    return parse_history_xml(body, expect="eSearchResult")


def _epost_history(http: Transport, *, accessions: Sequence[str]) -> History:
    """Upload an accession list to the history server (POST — never the URL)."""
    from urllib.parse import urlencode

    url = _with_key(f"{EUTILS}/epost.fcgi?db=nuccore")
    payload = urlencode({"db": "nuccore", "id": ",".join(accessions)}).encode()
    body = retry_with_backoff(lambda: http.post(url, payload, timeout=120.0))
    history = parse_history_xml(body, expect="ePostResult")
    return History(count=len(accessions), web_env=history.web_env, query_key=history.query_key)


def _esummary_paged(
    http: Transport, *, history: History, limit: int, page_size: int
) -> list[dict[str, Any]]:
    """Page esummary over a history result set. This is what lifts the ceiling."""
    records: list[dict[str, Any]] = []
    for retstart in range(0, limit, page_size):
        retmax = min(page_size, limit - retstart)
        url = _with_key(
            f"{EUTILS}/esummary.fcgi?db=nuccore"
            f"&WebEnv={history.web_env}&query_key={history.query_key}"
            f"&retstart={retstart}&retmax={retmax}"
        )
        body = retry_with_backoff(lambda url=url: http.get(url, timeout=180.0))
        records.extend(parse_esummary_xml(body))
        _throttle()
    return records


def _esummary_by_accession(
    http: Transport, *, accessions: Sequence[str], page_size: int
) -> list[dict[str, Any]]:
    history = _epost_history(http, accessions=accessions)
    return _esummary_paged(http, history=history, limit=len(accessions), page_size=page_size)


def _efetch_accessions(
    http: Transport, *, accessions: Sequence[str], dest: Path, stem: str, page_size: int
) -> Path:
    """Download GenBank records for an explicit accession set, via epost history.

    Posting the ids and paging the history avoids both the 2000-character URL
    ceiling and NCBI's per-request record cap.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{stem}.gbff"
    if not accessions:
        return out

    history = _epost_history(http, accessions=accessions)
    with out.open("w", encoding="utf-8") as handle:
        for retstart in range(0, len(accessions), page_size):
            retmax = min(page_size, len(accessions) - retstart)
            url = _with_key(
                f"{EUTILS}/efetch.fcgi?db=nuccore&rettype=gbwithparts&retmode=text"
                f"&WebEnv={history.web_env}&query_key={history.query_key}"
                f"&retstart={retstart}&retmax={retmax}"
            )
            body = retry_with_backoff(lambda url=url: http.get(url, timeout=600.0))
            handle.write(_maybe_gunzip(body))
            _throttle()
    return out


def _throttle() -> None:
    import time

    delay = eutils_delay()
    if delay:
        time.sleep(delay)


# ─────────────────────────────────────────────────────────────
# Result assembly
# ─────────────────────────────────────────────────────────────


def _as_data(
    *,
    manifest: dict[str, Any],
    artifact: Path | None = None,
    extra_artifacts: dict[str, tuple[Path, str, str]] | None = None,
    records: list[dict[str, Any]] | None = None,
    genomes: list[dict[str, Any]] | None = None,
    modality: str = "organelle_records",
) -> OrganelleData:
    artifacts: dict[str, ArtifactRef] = {}
    if artifact is not None and artifact.is_file():
        artifacts["records"] = ArtifactRef.from_path(
            artifact, kind="records", format="genbank", media_type="text/plain"
        )
    for name, (path, fmt, media) in (extra_artifacts or {}).items():
        if path.is_file():
            artifacts[name] = ArtifactRef.from_path(path, kind=name, format=fmt, media_type=media)
    payload: dict[str, Any] = {"manifest": manifest}
    if records is not None:
        payload["records"] = records
    if genomes is not None:
        payload["genomes"] = genomes
    return OrganelleData(
        modality=modality,
        artifacts=artifacts,
        payload=payload,
    )


def _write_exports(
    *,
    formats: Sequence[str],
    records: list[dict[str, Any]],
    genbank: Path,
    dest: Path,
    stem: str,
) -> dict[str, tuple[Path, str, str]]:
    """Materialise the requested side-products. GenBank is always written."""
    unknown = set(formats) - {"genbank", "fasta", "csv", "jsonl"}
    if unknown:
        raise OrganelleParameterError(
            code="input.unknown_format",
            message=f"unknown formats: {sorted(unknown)}",
            details={"supported": ["genbank", "fasta", "csv", "jsonl"]},
        )
    out: dict[str, tuple[Path, str, str]] = {}
    if "fasta" in formats and genbank.is_file():
        fasta = dest / f"{stem}.fasta"
        genbank_to_fasta(genbank, fasta)
        out["sequences"] = (fasta, "fasta", "text/plain")
    if "csv" in formats and records:
        csv_path = write_metadata_csv(records, dest / f"{stem}_metadata.csv")
        out["metadata_table"] = (csv_path, "csv", "text/csv")
    if "jsonl" in formats and records:
        jsonl = write_jsonl(records, dest / f"{stem}_metadata.jsonl")
        out["metadata"] = (jsonl, "jsonl", "application/x-ndjson")
    return out
