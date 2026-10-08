"""Exact-species nuclear genome-size candidate lookup against NCBI Datasets v2.

This is the input-recovery side of low-coverage PMAT2 routing: when a plant
mitochondrial HiFi run needs a nuclear genome size and the caller will not
guess, this operation resolves an exact species through NCBI Taxonomy and lists
the ranked nuclear-assembly candidates for that one TaxID. It never substitutes
a near relative and never auto-selects a scaffold/contig or phased record.

It is deliberately separate from :func:`organelleverse.fetch.nuclear`: that
operation downloads proteins from the ``v2alpha`` download endpoint, while this
one only reads NCBI Datasets **v2** report metadata and stores one canonical
report artifact per candidate. No sequence is downloaded.

Canonical report bytes are UTF-8 JSON with sorted keys, compact separators, and
exactly one trailing newline. The artifact SHA256 is the hash of those bytes, so
evidence bound to an invocation cannot be silently replaced by a later API
response.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, NoReturn, TypedDict, cast
from urllib.parse import quote

from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict, Field

from ..core.artifacts import ArtifactRef
from ..core.data import OrganelleData
from ..core.errors import OrganelleExecutionError, OrganelleInputError, OrganelleParameterError
from ._http import Transport, default_transport, get_with_retry

__all__ = ["DATASETS_V2", "GenomeSizeCandidate", "genome_size_candidates"]

DATASETS_V2 = "https://api.ncbi.nlm.nih.gov/datasets/v2"

_PROVIDER = "ncbi_datasets"
_SCHEMA = "datasets/v2"
# Request the largest documented page and follow every returned page token
# before ranking, so a later page cannot hide the best assembly.
_TAXON_PAGE_SIZE = 1000

# Stable rank tables. Higher rank sorts first (negated in the sort key).
_LEVEL_RANK: dict[str, int] = {
    "Complete Genome": 3,
    "Telomere-to-Telomere": 3,
    "Chromosome": 2,
    "Scaffold": 1,
    "Contig": 0,
}
_REFSEQ_RANK: dict[str, int] = {
    "reference genome": 2,
    "representative genome": 1,
}
# Ploidy markers that disqualify auto-selection. ``hap2`` is NCBI's phased pair
# marker; a plain ``haploid`` assembly never contains these substrings.
_DIPLOID_TYPE_MARKERS = ("diploid", "pseudohaplotype", "primary-hap2", "alternate-hap2", "hap2")
_LINKED_PHASING_ROLES = frozenset({"primary", "alternate", "maternal", "paternal"})


class _RecordFields(TypedDict):
    accession: str
    assembly_name: str
    assembly_level: str
    assembly_type: object
    refseq_category: object
    diploid_role: object
    linked_assemblies: list[object]
    atypical: object
    assembly_status: object
    suppression_reason: object
    release_date: str
    taxid: int | None
    scientific_name: str
    genome_size_bp: int
    contig_n50: int | None
    scaffold_n50: int | None
    unplaced_proportion: float | None


class GenomeSizeCandidate(BaseModel):
    """One ranked nuclear-assembly genome-size candidate for an exact species.

    Carries only the scientific identity a person or Agent needs to choose a
    genome size, plus the content address of its authoritative NCBI report
    artifact. Retrieval metadata (provider, schema, timestamp) lives on the
    enclosing :class:`~organelleverse.core.data.OrganelleData`, never here.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )

    taxon_id: int = Field(gt=0)
    scientific_name: str = Field(min_length=1)
    assembly_accession: str = Field(min_length=1)
    assembly_name: str
    assembly_level: str
    refseq_category: str | None
    genome_size_bp: int = Field(gt=0)
    contig_n50: int | None = Field(default=None, ge=0)
    scaffold_n50: int | None = Field(default=None, ge=0)
    unplaced_proportion: float | None = Field(default=None, ge=0, le=1)
    release_date: str
    selectable: bool
    ambiguity_reason: str | None
    report_uri: str = Field(min_length=1)
    report_artifact_role: str = Field(min_length=1)
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _default_cache_root() -> Path:
    """Persistent content-addressed home for canonical report artifacts.

    Overridable for hermetic tests via ``ORGANELLEVERSE_GENOME_SIZE_CACHE``;
    otherwise honors ``XDG_DATA_HOME`` and falls back to the platform data dir.
    """

    override = os.environ.get("ORGANELLEVERSE_GENOME_SIZE_CACHE")
    if override:
        return Path(override)
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "organelleverse" / "genome_size_reports"


def _canonical_report_bytes(record: Mapping[str, object]) -> bytes:
    """Canonical UTF-8 JSON: sorted keys, compact separators, one trailing newline."""
    payload = json.dumps(
        dict(record),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return payload.encode("utf-8") + b"\n"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_taxid(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = int(text)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _parse_int_required(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else 0
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _parse_int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() and value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _parse_float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) and 0 <= value <= 1 else None
    if isinstance(value, int):
        return float(value) if 0 <= value <= 1 else None
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) and 0 <= parsed <= 1 else None
    return None


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number {value!r}")


def _string_object_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, object], value)


def _validated_scientific_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrganelleParameterError(
            code="input.empty_scientific_name",
            message="scientific_name must be a non-empty species name",
        )
    return value


def _validated_max_candidates(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 50:
        raise OrganelleParameterError(
            code="parameter.invalid_max_candidates",
            message="max_candidates must be an integer from 1 through 50",
            details={"max_candidates": value},
        )
    return value


def _parse_json_body(body: bytes, *, url: str) -> dict[str, object]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"NCBI response was not UTF-8: {error}",
            details={"url": url},
            retryable=True,
        ) from error
    try:
        decoded: object = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"NCBI did not return JSON: {error}",
            details={"url": url, "body": text[:200]},
            retryable=True,
        ) from error
    if not isinstance(decoded, dict):
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message="NCBI response was not a JSON object",
            details={"url": url},
            retryable=True,
        )
    return cast(dict[str, object], decoded)


def _unexpected_status(url: str, status: int) -> OrganelleExecutionError:
    return OrganelleExecutionError(
        code="network.unexpected_status",
        message=f"NCBI returned HTTP {status}",
        details={"url": url, "status": status},
        retryable=False,
    )


def _resolve_taxonomy(transport: Transport, scientific_name: str) -> tuple[str, int]:
    """Resolve an exact species to (canonical name, TaxID), fail-closed.

    A 404 or empty node list is a confirmed exact-name miss, not a transient
    failure. A node whose name or TaxID is missing is internally inconsistent.
    """

    requested_name = " ".join(scientific_name.split())
    url = f"{DATASETS_V2}/taxonomy/taxon/{quote(requested_name, safe='')}"
    response = get_with_retry(transport, url)
    if response.status == 404:
        raise OrganelleInputError(
            code="fetch.genome_size_taxon_miss",
            message=f"NCBI Taxonomy has no exact match for {scientific_name!r}",
            details={"scientific_name": scientific_name},
            retryable=False,
        )
    if response.status != 200:
        raise _unexpected_status(url, response.status)
    body = _parse_json_body(response.body, url=url)
    raw_nodes: object = body.get("taxonomy_nodes")
    if raw_nodes is None:
        raw_nodes = []
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise OrganelleInputError(
            code="fetch.genome_size_taxon_miss",
            message=f"NCBI Taxonomy returned no node for {scientific_name!r}",
            details={"scientific_name": scientific_name},
            retryable=False,
        )
    nodes = cast(list[object], raw_nodes)
    exact_matches: list[tuple[str, int]] = []
    for raw_node in nodes:
        raw_node_map = _string_object_mapping(raw_node)
        if raw_node_map is None:
            raise OrganelleExecutionError(
                code="fetch.genome_size_taxon_inconsistent",
                message="NCBI Taxonomy node was not an object",
                details={"url": url},
                retryable=True,
            )
        nested = _string_object_mapping(raw_node_map.get("taxonomy"))
        node = nested if nested is not None else raw_node_map
        name = node.get("scientific_name") or node.get("organism_name")
        taxid = _parse_taxid(node.get("tax_id"))
        if not isinstance(name, str) or not name.strip() or taxid is None:
            raise OrganelleExecutionError(
                code="fetch.genome_size_taxon_inconsistent",
                message="NCBI Taxonomy node lacks a consistent name/TaxID pair",
                details={"url": url, "scientific_name": scientific_name},
                retryable=True,
            )
        canonical_name = " ".join(name.split())
        if canonical_name.casefold() == requested_name.casefold():
            exact_matches.append((canonical_name, taxid))

    if not exact_matches:
        raise OrganelleInputError(
            code="fetch.genome_size_taxon_miss",
            message=f"NCBI Taxonomy returned no exact scientific-name match for {scientific_name!r}",
            details={"scientific_name": scientific_name},
            retryable=False,
        )
    if len(set(exact_matches)) != 1:
        raise OrganelleExecutionError(
            code="fetch.genome_size_taxon_inconsistent",
            message="NCBI Taxonomy returned conflicting TaxIDs for one exact scientific name",
            details={"url": url, "scientific_name": scientific_name},
            retryable=True,
        )
    return exact_matches[0]


def _fetch_taxon_reports(transport: Transport, taxid: int) -> list[dict[str, object]]:
    base_url = f"{DATASETS_V2}/genome/taxon/{quote(str(taxid), safe='')}/dataset_report"
    reports: list[dict[str, object]] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        url = f"{base_url}?page_size={_TAXON_PAGE_SIZE}"
        if page_token is not None:
            url += f"&page_token={quote(page_token, safe='')}"
        response = get_with_retry(transport, url)
        if response.status == 404:
            return []
        if response.status != 200:
            raise _unexpected_status(url, response.status)
        body = _parse_json_body(response.body, url=url)
        raw_page_reports = body.get("reports", [])
        if not isinstance(raw_page_reports, list):
            raise OrganelleExecutionError(
                code="network.malformed_response",
                message="NCBI Datasets reports page was not a list of objects",
                details={"url": url},
                retryable=True,
            )
        page_reports: list[dict[str, object]] = []
        for raw_report in cast(list[object], raw_page_reports):
            report = _string_object_mapping(raw_report)
            if report is None:
                raise OrganelleExecutionError(
                    code="network.malformed_response",
                    message="NCBI Datasets reports page was not a list of objects",
                    details={"url": url},
                    retryable=True,
                )
            page_reports.append(dict(report))
        reports.extend(page_reports)
        raw_next = body.get("next_page_token")
        if raw_next in (None, ""):
            return reports
        if not isinstance(raw_next, str) or raw_next in seen_tokens:
            raise OrganelleExecutionError(
                code="network.malformed_response",
                message="NCBI Datasets returned an invalid or repeated page token",
                details={"url": url},
                retryable=True,
            )
        seen_tokens.add(raw_next)
        page_token = raw_next


def _fetch_accession_report(transport: Transport, accession: str) -> dict[str, object]:
    url = f"{DATASETS_V2}/genome/accession/{quote(accession, safe='')}/dataset_report"
    response = get_with_retry(transport, url)
    if response.status == 404:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"NCBI Datasets report vanished for accession {accession!r}",
            details={"url": url, "accession": accession},
            retryable=True,
        )
    if response.status != 200:
        raise _unexpected_status(url, response.status)
    body = _parse_json_body(response.body, url=url)
    raw_reports: object = body.get("reports")
    if raw_reports is None:
        raw_reports = []
    if not isinstance(raw_reports, list) or not raw_reports:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"NCBI Datasets returned no report for accession {accession!r}",
            details={"url": url, "accession": accession},
            retryable=True,
        )
    reports = cast(list[object], raw_reports)
    if len(reports) != 1:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"NCBI Datasets did not return exactly one report for {accession!r}",
            details={"url": url, "accession": accession},
            retryable=True,
        )
    report = _string_object_mapping(reports[0])
    if report is None:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"NCBI Datasets report for {accession!r} was not an object",
            details={"url": url, "accession": accession},
            retryable=True,
        )
    return dict(report)


def _record_fields(record: Mapping[str, object]) -> _RecordFields:
    info_map = _string_object_mapping(record.get("assembly_info")) or {}
    stats_map = _string_object_mapping(record.get("assembly_stats")) or {}
    organism_map = _string_object_mapping(record.get("organism")) or {}
    linked = info_map.get("linked_assemblies")
    return {
        "accession": str(record.get("accession") or ""),
        "assembly_name": str(info_map.get("assembly_name") or ""),
        "assembly_level": str(info_map.get("assembly_level") or ""),
        "assembly_type": info_map.get("assembly_type"),
        "refseq_category": info_map.get("refseq_category"),
        "diploid_role": info_map.get("diploid_role"),
        "linked_assemblies": cast(list[object], linked) if isinstance(linked, list) else [],
        "atypical": info_map.get("atypical"),
        "assembly_status": info_map.get("assembly_status"),
        "suppression_reason": info_map.get("suppression_reason"),
        "release_date": str(
            info_map.get("release_date")
            or info_map.get("released_date")
            or info_map.get("submission_date")
            or ""
        ),
        "taxid": _parse_taxid(organism_map.get("tax_id")),
        "scientific_name": str(
            organism_map.get("scientific_name") or organism_map.get("organism_name") or ""
        ),
        "genome_size_bp": _parse_int_required(stats_map.get("total_sequence_length")),
        "contig_n50": _parse_int_or_none(stats_map.get("contig_n50")),
        "scaffold_n50": _parse_int_or_none(stats_map.get("scaffold_n50")),
        "unplaced_proportion": _parse_float_or_none(stats_map.get("unplaced_proportion")),
    }


def _is_excluded(fields: _RecordFields) -> bool:
    atypical = fields["atypical"]
    atypical_map = _string_object_mapping(atypical)
    if atypical_map is not None:
        if atypical_map.get("is_atypical") is True:
            return True
        official_text: list[str] = []
        warnings = atypical_map.get("warnings")
        if isinstance(warnings, list):
            official_text.extend(str(item) for item in cast(list[object], warnings))
        if any(
            marker in text.casefold()
            for text in official_text
            for marker in ("partial", "contaminat", "incomplete", "anomal")
        ):
            return True

    status = fields["assembly_status"]
    if status is not None and str(status).strip().casefold() not in {"", "current"}:
        return True
    suppression_reason = fields["suppression_reason"]
    return isinstance(suppression_reason, str) and bool(suppression_reason.strip())


def _assembly_level_rank(level: str) -> int:
    return _LEVEL_RANK.get(level, 0)


def _refseq_category_rank(category: object) -> int:
    if category is None:
        return 0
    return _REFSEQ_RANK.get(str(category), 0)


def _completeness_rank(fields: _RecordFields) -> int:
    return (1 if fields["contig_n50"] is not None else 0) + (
        1 if fields["scaffold_n50"] is not None else 0
    )


def _date_ordinal(release_date: str) -> int:
    if not release_date:
        return 0
    try:
        return date.fromisoformat(release_date[:10]).toordinal()
    except ValueError:
        return 0


def _ploidy_reason(fields: _RecordFields) -> str | None:
    assembly_type = fields["assembly_type"]
    type_text = str(assembly_type).lower() if assembly_type is not None else ""
    if any(marker in type_text for marker in _DIPLOID_TYPE_MARKERS):
        return f"assembly_type:{assembly_type}"
    diploid_role = fields["diploid_role"]
    if diploid_role is not None and str(diploid_role).lower() != "haploid":
        return f"diploid_role:{diploid_role}"
    for entry in fields["linked_assemblies"]:
        entry_map = _string_object_mapping(entry)
        if entry_map is not None:
            raw_role = entry_map.get("assembly_type") or entry_map.get("role")
            role = str(raw_role or "").lower().replace("-", "_").replace(" ", "_")
            if role in _LINKED_PHASING_ROLES or "haplotype" in role or "pseudohaplotype" in role:
                return f"linked_assembly_role:{raw_role or 'unknown'}"
    # Fail-closed: haploid ploidy must be positively confirmed.
    if assembly_type is None or type_text != "haploid":
        return "ploidy_not_confirmed"
    return None


def _selectability(fields: _RecordFields) -> tuple[bool, str | None]:
    reasons: list[str] = []
    if _assembly_level_rank(str(fields["assembly_level"])) < 2:
        reasons.append(f"assembly_level:{fields['assembly_level'] or 'unknown'}")
    ploidy = _ploidy_reason(fields)
    if ploidy is not None:
        reasons.append(ploidy)
    if reasons:
        return False, ";".join(reasons)
    return True, None


def _sort_key(fields: _RecordFields) -> tuple[object, ...]:
    unplaced = fields["unplaced_proportion"]
    unplaced_value = unplaced if unplaced is not None else 2.0
    return (
        -_assembly_level_rank(str(fields["assembly_level"])),
        -_refseq_category_rank(fields["refseq_category"]),
        -_completeness_rank(fields),
        unplaced_value,
        -_date_ordinal(str(fields["release_date"])),
        str(fields["accession"]),
    )


def _materialize_report(cache_root: Path, sha256: str, canonical: bytes) -> ArtifactRef:
    """Write canonical report bytes to a content-addressed path and reference them.

    The file is named by its SHA256, so writes are idempotent and the artifact's
    hash is the hash of exactly the bytes on disk.
    """

    shard = cache_root / sha256[:2]
    shard.mkdir(parents=True, exist_ok=True)
    path = shard / f"{sha256}.json"
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise OrganelleExecutionError(
                code="fetch.genome_size_cache_corrupt",
                message="Could not verify an existing genome-size report artifact",
                details={"path": str(path), "expected_sha256": sha256},
                retryable=False,
            ) from error
        if existing != canonical or hashlib.sha256(existing).hexdigest() != sha256:
            raise OrganelleExecutionError(
                code="fetch.genome_size_cache_corrupt",
                message="Existing genome-size report artifact does not match its content address",
                details={"path": str(path), "expected_sha256": sha256},
                retryable=False,
            )
    else:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=shard, prefix=".report-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(canonical)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return ArtifactRef(
        kind="genome_size_report",
        uri=str(path),
        format="json",
        media_type="application/json",
        sha256=sha256,
        size_bytes=len(canonical),
        validated=True,
    )


def genome_size_candidates(
    scientific_name: str,
    *,
    max_candidates: Annotated[int, Ge(1), Le(50)] = 10,
    transport: Transport | None = None,
) -> OrganelleData:
    """Return ranked exact-species nuclear genome-size candidates from NCBI.

    Resolves ``scientific_name`` through NCBI Taxonomy, lists the TaxID's nuclear
    assemblies, keeps only exact-TaxID records, excludes anomalous ones, ranks
    the rest deterministically, and stores one canonical report artifact per
    returned candidate. A near relative is never substituted; a scaffold/contig
    or phased record is displayed but never auto-selected.
    """

    query_name = _validated_scientific_name(scientific_name)
    candidate_limit = _validated_max_candidates(max_candidates)

    http = transport or default_transport()
    resolved_name, taxid = _resolve_taxonomy(http, query_name)
    raw_reports = _fetch_taxon_reports(http, taxid)

    candidates_fields: list[_RecordFields] = []
    for record in raw_reports:
        fields = _record_fields(record)
        if not str(fields["accession"]):
            continue
        if fields["taxid"] != taxid:
            continue
        if _is_excluded(fields):
            continue
        if fields["genome_size_bp"] <= 0:
            continue
        candidates_fields.append(fields)

    if not candidates_fields:
        raise OrganelleInputError(
            code="fetch.genome_size_taxon_miss",
            message=(
                f"NCBI Datasets has no exact-TaxID nuclear assembly for "
                f"{query_name!r} (TaxID {taxid})"
            ),
            details={"scientific_name": query_name, "taxon_id": taxid},
            retryable=False,
        )

    candidates_fields.sort(key=_sort_key)
    kept = candidates_fields[:candidate_limit]

    cache_root = _default_cache_root()
    artifacts: dict[str, ArtifactRef] = {}
    records: list[dict[str, object]] = []
    for fields in kept:
        accession = str(fields["accession"])
        report = _fetch_accession_report(http, accession)
        report_fields = _record_fields(report)
        if report_fields != fields:
            raise OrganelleExecutionError(
                code="fetch.genome_size_report_inconsistent",
                message="NCBI accession report does not match the ranked taxon report",
                details={"accession": accession},
                retryable=True,
            )
        if (
            report_fields["taxid"] != taxid
            or " ".join(str(report_fields["scientific_name"]).split()).casefold()
            != resolved_name.casefold()
        ):
            raise OrganelleExecutionError(
                code="fetch.genome_size_report_inconsistent",
                message="NCBI accession report does not match the resolved species identity",
                details={"accession": accession, "taxon_id": taxid},
                retryable=True,
            )
        canonical = _canonical_report_bytes(report)
        digest = hashlib.sha256(canonical).hexdigest()
        role = f"genome_size_report:{accession}"
        report_uri = f"{DATASETS_V2}/genome/accession/{quote(accession, safe='')}/dataset_report"
        artifacts[role] = _materialize_report(cache_root, digest, canonical)
        selectable, ambiguity_reason = _selectability(fields)
        candidate = GenomeSizeCandidate(
            taxon_id=taxid,
            scientific_name=resolved_name,
            assembly_accession=accession,
            assembly_name=str(fields["assembly_name"]),
            assembly_level=str(fields["assembly_level"]),
            refseq_category=(
                str(fields["refseq_category"]) if fields["refseq_category"] is not None else None
            ),
            genome_size_bp=fields["genome_size_bp"],
            contig_n50=fields["contig_n50"],
            scaffold_n50=fields["scaffold_n50"],
            unplaced_proportion=fields["unplaced_proportion"],
            release_date=str(fields["release_date"]),
            selectable=selectable,
            ambiguity_reason=ambiguity_reason,
            report_uri=report_uri,
            report_artifact_role=role,
            report_sha256=digest,
        )
        records.append(candidate.model_dump(mode="json"))

    return OrganelleData.model_validate(
        {
            "modality": "genome_size_candidates",
            "artifacts": artifacts,
            "payload": {
                "scientific_name": resolved_name,
                "taxon_id": taxid,
                "query": query_name.strip(),
                "candidates": records,
            },
            "metadata": {
                "provider": _PROVIDER,
                "schema": _SCHEMA,
                "retrieved_at": _now_iso(),
                "endpoints": {
                    "taxonomy": f"{DATASETS_V2}/taxonomy/taxon/{{scientific_name}}",
                    "taxon_report": f"{DATASETS_V2}/genome/taxon/{{taxid}}/dataset_report",
                    "accession_report": (
                        f"{DATASETS_V2}/genome/accession/{{accession}}/dataset_report"
                    ),
                },
            },
        }
    )
