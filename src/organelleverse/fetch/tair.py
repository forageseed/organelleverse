"""TAIR (arabidopsis.org) — Arabidopsis thaliana only.

The real download API, found by reading the SPA's own JS bundle (grep for
``download-files`` in ``/js/app.*.js``) rather than assuming the design's
original ``index-auto.jsp`` guess still worked — direct content inspection
showed that guess returns an identical 2192-byte Vue-router HTML shell for
every path, real or fabricated. The real API:
``GET /api/download-files/list?dir=<dir>`` (JSON directory listing) and
``GET /api/download-files/download?filePath=<path>`` (streams real content).

TAIR reuses dated filenames across releases with no stable "current" alias
for protein/CDS/GFF3 (confirmed: protein moved directories and filenames
between two snapshots taken minutes apart while writing this plan), so this
module discovers the current file by listing and picking the newest
``lastModified`` top-level entry, rather than hardcoding one dated name.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import cast
from urllib.parse import quote

from ..core.artifacts import ArtifactRef
from ..core.data import OrganelleData
from ..core.errors import OrganelleParameterError
from ..core.frozen import FrozenJson, FrozenMap
from ._content import SHARED_INCLUDE_KINDS, validate_downloaded_content
from ._http import Transport, default_transport, download_with_retry, get_with_retry
from .manifest import build_manifest
from .taxon import resolve_taxon

__all__ = ["fetch_tair"]

_LIST_API = "https://www.arabidopsis.org/api/download-files/list"
_DOWNLOAD_API = "https://www.arabidopsis.org/api/download-files/download"
_USER_AGENT = "Mozilla/5.0 (compatible; OrganelleVerse-fetch/1.0)"

# kind -> (directory to list, filename pattern for a "current", non-archived file)
_DISCOVERED_KINDS: dict[str, tuple[str, re.Pattern[str]]] = {
    "gff3": (
        "Genes/Araport11_genome_release",
        re.compile(r"^Araport11_GFF3_genes_transposons\.\d{8}\.gff\.gz$"),
    ),
    "protein": (
        "Sequences/Araport11_blastsets",
        re.compile(r"^Araport11_pep_\d{8}\.gz$"),
    ),
    "cds": (
        "Genes/Araport11_genome_release/Araport11_blastsets",
        re.compile(r"^Araport11_cds_\d{8}\.gz$"),
    ),
}
# TAIR10's assembly itself, unlike Araport11's gene models, is not
# re-released on a rolling date-stamped schedule — a fixed path is correct.
_GENOME_PATH = "Genes/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas.gz"


def _discover_current_file(http: Transport, dir_path: str, pattern: re.Pattern[str]) -> str | None:
    """The ``path`` of the newest-``lastModified`` top-level file in ``dir_path``
    matching ``pattern`` — top-level only, so an ``archived``/``Archived``
    subdirectory (present in every real TAIR listing) is never picked."""
    response = get_with_retry(
        http, f"{_LIST_API}?dir={quote(dir_path)}", headers={"User-Agent": _USER_AGENT}
    )
    if response.status != 200:
        return None
    try:
        raw_listing = json.loads(response.body)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw_listing, dict):
        return None
    listing = cast("dict[str, object]", raw_listing)

    candidates: list[tuple[str, str]] = []
    for name, raw_entry in listing.items():
        if not isinstance(raw_entry, dict):
            continue
        entry = cast("dict[str, object]", raw_entry)
        if entry.get("type") != "file":
            continue
        if not pattern.match(name):
            continue
        path = entry.get("path")
        last_modified = entry.get("lastModified")
        if isinstance(path, str):
            candidates.append((str(last_modified) if last_modified is not None else "", path))

    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def fetch_tair(
    *,
    taxon: str,
    dest: str | Path,
    include: tuple[str, ...] = ("protein",),
    transport: Transport | None = None,
) -> OrganelleData:
    """Fetch the current Araport11/TAIR10 release — the only species TAIR serves."""
    unknown = set(include) - SHARED_INCLUDE_KINDS
    if unknown:
        raise OrganelleParameterError(
            code="input.unknown_include",
            message=f"unknown include: {sorted(unknown)}",
            details={"supported": sorted(SHARED_INCLUDE_KINDS)},
        )

    resolved = resolve_taxon(taxon)
    if resolved.genus != "Arabidopsis" or resolved.species != "thaliana":
        raise OrganelleParameterError(
            code="input.non_arabidopsis_taxon",
            message=f"TAIR only serves Arabidopsis thaliana, got {resolved.binomial!r}",
            details={"taxon": taxon},
        )

    http = transport or default_transport()
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    artifacts: dict[str, ArtifactRef] = {}
    missing: list[dict[str, str]] = []
    discovered_paths: dict[str, str] = {}

    for kind in include:
        if kind == "genome":
            remote_path: str | None = _GENOME_PATH
        elif kind in _DISCOVERED_KINDS:
            dir_path, pattern = _DISCOVERED_KINDS[kind]
            remote_path = _discover_current_file(http, dir_path, pattern)
        else:
            missing.append({"kind": kind, "reason": "not_offered_by_this_source"})
            continue

        if remote_path is None:
            missing.append({"kind": kind, "reason": "not_found_in_current_release"})
            continue

        final_path = dest_dir / f"{kind}.gz"
        # Validate a same-filesystem temp copy before renaming into place, so
        # a WAF/app-shell page saved with a 200 status never lands at the
        # real output path — see this plan's "fifth adjustment" note above.
        with tempfile.TemporaryDirectory(dir=dest_dir) as tmp:
            tmp_path = Path(tmp) / final_path.name
            status = download_with_retry(
                http,
                f"{_DOWNLOAD_API}?filePath={quote(remote_path)}",
                tmp_path,
                headers={"User-Agent": _USER_AGENT},
                timeout=1800.0,
            )
            if status != 200:
                missing.append({"kind": kind, "reason": "not_found_in_current_release"})
                continue
            validate_downloaded_content(tmp_path, kind=kind, gzipped=True, source="tair")
            tmp_path.replace(final_path)
        files[kind] = str(final_path)
        artifact_format = "gff3_gz" if kind == "gff3" else "fasta_gz"
        artifacts[kind] = ArtifactRef.from_path(
            final_path, kind=kind, format=artifact_format, media_type="application/gzip"
        )
        discovered_paths[kind] = remote_path

    accession = "tair:Araport11"
    record = {
        "accession": accession,
        "organism": resolved.binomial,
        "source": "tair",
        "confidence": None,
        "files": files,
        "missing": missing,
    }
    manifest = build_manifest(
        source="tair",
        organelle="nuclear",
        accessions=[accession],
        scope={"taxon": taxon, "include": list(include), "discovered_paths": discovered_paths},
        hits_before_filter=1,
        examined=1,
    )
    return OrganelleData(
        modality="nuclear_assemblies",
        # `OrganelleData`'s `mode="before"` validators accept a plain dict and
        # freeze it themselves at runtime; the cast only satisfies the static
        # type checker, which otherwise sees the field's frozen declared type
        # (`FrozenMap[...]`) as incompatible with a plain `dict`.
        artifacts=cast("FrozenMap[ArtifactRef]", artifacts),
        payload=cast("FrozenMap[FrozenJson]", {"manifest": manifest, "records": [record]}),
    )
