"""NGDC/GWH — a supplementary sample source, not a source of new species.

China's National Genomics Data Center holds plant genome *assemblies* submitted
there under ``GWH*`` accessions, which NCBI does not mint. Measured on 2026-07-14
against ``/gwh/Plants/`` (12,166 genome directories):

* **1,270 plastid genomes.** A random sample of 40 was checked against NCBI:
  **0 of 40** belonged to a species NCBI lacked — every species already had a
  chloroplast genome there. So the value is **sample-level, not species-level**:
  these are 1,270 additional assemblies (useful for population and phylogenetic
  sampling), not 1,270 organisms found nowhere else. Many are medicinal plants;
  the submitter is often the National Resource Center for Chinese Materia Medica.
* **5 mitochondrial genomes** — Asparagus taliensis, Camellia sinensis,
  Chlorella vulgaris (x2), and *Panax notoginseng* (sanqi). Same index, same API,
  so they cost nothing extra.

What GWH has that NCBI's organelle records do not is an ``assemblyLevel``
(Complete / Chromosome / Scaffold) — a quality signal absent from nuccore.

The earlier claim that ~400 species existed here and nowhere else was wrong: it
rested on a table that was never actually queried. ``manifest["scope"]["ngdc_only"]``
means the *accession* is absent from NCBI, never the species.

**Classification is by token, not substring.** ``Ilex_vomitoria_24`` contains
the letters "mito" (vo-MITO-ria) and is not a mitochondrion. Splitting the
folder name into tokens and matching whole tokens is what makes the difference,
and a size guard catches whatever the names still get wrong.

Only sequence and annotation are taken. CGIR also publishes 16M SSRs, 1M DNA
barcodes and 38M "DNA signature sequences". Those third-party computed results
are outside the released fetch contract and are not imported into the evidence
chain.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from ..core.artifacts import ArtifactRef
from ..core.data import OrganelleData
from ..core.errors import OrganelleExecutionError, OrganelleParameterError
from ..core.frozen import FrozenJson, FrozenMap
from ._http import Transport, default_transport, retry_with_backoff
from .manifest import build_manifest

__all__ = [
    "GWH_API",
    "GWH_PLANTS",
    "classify_folder",
    "fetch_ngdc",
    "gwh_assembly_metadata",
    "parse_plants_index",
]

# `download.big.ac.cn` serves a certificate that does not match its own
# hostname, and 301s here anyway. Go straight to the destination.
GWH_PLANTS = "https://download.cncb.ac.cn/gwh/Plants"
GWH_API = "https://ngdc.cncb.ac.cn/gwh/api/public/assembly"

Organelle = Literal["mitochondrion", "plastid"]

# Whole tokens, never substrings. `Ilex_vomitoria` is not a mitochondrion.
_MITO_TOKENS = frozenset(
    {"mt", "mtdna", "mito", "mitochondrion", "mitochondria", "mitochondrial", "mitogenome"}
)
_PLASTID_TOKENS = frozenset(
    {"cp", "cpdna", "chloroplast", "chloroplasts", "plastid", "plastids", "plastome"}
)

_ACCESSION = re.compile(r"(GWH[A-Z]+\d+)")
_TOKENS = re.compile(r"[_.\-\s]+")

# A compressed organelle genome is small. A nuclear one is not: wheat is 17 Gb.
# Any name-based classifier will eventually be wrong; the size is the backstop.
_MAX_ORGANELLE_GZ_BYTES = 12 * 1024 * 1024


def classify_folder(folder: str) -> Organelle | None:
    """Which organelle a GWH folder holds, or ``None`` if it is not one.

    Token-wise, because substring matching is wrong: ``Ilex_vomitoria_24``
    contains "mito" and is a holly.
    """
    tokens = {token.lower() for token in _TOKENS.split(folder) if token}
    if tokens & _MITO_TOKENS:
        return "mitochondrion"
    if tokens & _PLASTID_TOKENS:
        return "plastid"
    return None


def _organism_from_folder(folder: str) -> str:
    """Best-effort binomial from ``Genus_species_whatever_GWH...``.

    A guess, and labelled as one: ``gwh_assembly_metadata`` returns the
    authoritative name.
    """
    parts = [p for p in folder.split("_") if p]
    if len(parts) >= 2 and parts[0][:1].isupper():
        return f"{parts[0]} {parts[1]}".rstrip(".")
    return parts[0] if parts else ""


def parse_plants_index(html: str | bytes) -> list[dict[str, Any]]:
    """Every organelle genome in the GWH plants directory listing."""
    raw = html.decode(errors="replace") if isinstance(html, bytes) else html
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for match in re.finditer(r'href="([^"?][^"]*)/"', raw):
        folder = match.group(1).strip("/")
        if not folder or folder.startswith("."):
            continue
        organelle = classify_folder(folder)
        if organelle is None:
            continue
        accession_match = _ACCESSION.search(folder)
        if not accession_match:
            continue
        accession = accession_match.group(1)
        if accession in seen:
            continue
        seen.add(accession)
        records.append(
            {
                "accession": accession,
                "folder": folder,
                "organelle": organelle,
                "organism": _organism_from_folder(folder),
                "source": "ngdc_gwh",
            }
        )
    return sorted(records, key=lambda r: str(r["accession"]))


def gwh_assembly_metadata(http: Transport, accession: str) -> dict[str, Any]:
    """Authoritative metadata for one GWH assembly.

    The versioned accession is required: ``GWHABWV00000000`` (no version)
    answers with ``organism: None``, while ``GWHABWV01000000`` answers properly.
    """
    body = retry_with_backoff(lambda: http.get(f"{GWH_API}/{accession}", timeout=60.0))
    raw = body.decode(errors="replace")
    try:
        decoded: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"GWH did not return JSON for {accession}: {exc}",
            details={"accession": accession, "body": raw[:160]},
            retryable=True,
        ) from exc
    if not isinstance(decoded, dict):
        return {}
    payload = cast(dict[str, object], decoded)
    return {
        "organism": payload.get("organism") or "",
        "taxid": str(payload.get("taxId") or ""),
        "assembly_name": payload.get("assemblyName") or "",
        # NCBI's organelle nuccore records carry no assembly level. GWH's do.
        "assembly_level": payload.get("assemblyLevel") or "",
        "bioproject": payload.get("bioprojectAccession") or "",
        "biosample": payload.get("biosampleAccession") or "",
        "submitter_organization": payload.get("submitterOrganization") or "",
        "release_time": payload.get("releaseTime") or "",
        "ftp_dna": payload.get("ftpPathDna") or "",
        "ftp_gff": payload.get("ftpPathGff") or "",
        "ftp_cds": payload.get("ftpPathCds") or "",
        "ftp_protein": payload.get("ftpPathProtein") or "",
    }


_SUFFIX = {
    "fasta": "genome.fasta.gz",
    "gff": "gff.gz",
    "cds": "CDS.fasta.gz",
    "protein": "Protein.faa.gz",
}


def _download_url(record: dict[str, Any], fmt: str) -> str:
    """The file URL, derivable from folder + accession without the API."""
    return f"{GWH_PLANTS}/{record['folder']}/{record['accession']}.{_SUFFIX[fmt]}"


def fetch_ngdc(
    *,
    organelle: Organelle,
    dest: str | Path,
    taxon: str | None = None,
    formats: Sequence[str] = ("fasta",),
    with_metadata: bool = True,
    max_records: int | None = None,
    max_gz_bytes: int = _MAX_ORGANELLE_GZ_BYTES,
    transport: Transport | None = None,
) -> OrganelleData:
    """Fetch organelle genomes from NGDC/GWH.

    Every record returned carries a ``GWH*`` accession, so none of it is
    reachable through NCBI. ``taxon`` filters on the folder name first (free)
    and on the authoritative organism afterwards (when ``with_metadata``).
    """
    unknown = set(formats) - set(_SUFFIX)
    if unknown:
        raise OrganelleParameterError(
            code="input.unknown_format",
            message=f"unknown formats: {sorted(unknown)}",
            details={"supported": sorted(_SUFFIX)},
        )
    if organelle not in ("mitochondrion", "plastid"):
        raise OrganelleParameterError(
            code="input.unknown_organelle",
            message=f"organelle must be 'mitochondrion' or 'plastid', got {organelle!r}",
            details={"organelle": organelle},
        )

    http = transport or default_transport()
    index_html = retry_with_backoff(lambda: http.get(f"{GWH_PLANTS}/", timeout=300.0))
    records = [r for r in parse_plants_index(index_html) if r["organelle"] == organelle]
    total_for_organelle = len(records)

    # Cheap pre-filter on the folder-derived name, before paying for metadata.
    if taxon:
        needle = taxon.strip().lower()
        records = [r for r in records if needle in str(r["organism"]).lower()]

    if max_records is not None:
        records = records[:max_records]

    if with_metadata:
        enriched: list[dict[str, Any]] = []
        for record in records:
            record = {**record, **gwh_assembly_metadata(http, str(record["accession"]))}
            # The authoritative organism can disagree with the folder guess.
            if taxon and taxon.strip().lower() not in str(record.get("organism", "")).lower():
                continue
            enriched.append(record)
        records = enriched

    dest_dir = Path(dest)
    artifacts: dict[str, ArtifactRef] = {}
    rejected: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []

    if records:
        dest_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        accession = str(record["accession"])
        paths: dict[str, str] = {}
        oversize = False

        for fmt in formats:
            url = _download_url(record, fmt)
            try:
                blob = retry_with_backoff(lambda url=url: http.get(url, timeout=600.0))
            except OrganelleExecutionError:
                continue  # a genome without a GFF is normal; a missing FASTA is not
            # Size guard. A name-based classifier will eventually be wrong, and a
            # nuclear genome must never enter an organelle dataset silently.
            if fmt == "fasta" and len(blob) > max_gz_bytes:
                rejected.append(
                    {
                        "accession": accession,
                        "reason": "too large for an organelle genome",
                        "gz_bytes": len(blob),
                    }
                )
                oversize = True
                break
            path = dest_dir / f"{accession}.{_SUFFIX[fmt]}"
            path.write_bytes(blob)
            paths[fmt] = str(path)

        if oversize or "fasta" not in paths:
            continue

        kept.append({**record, "files": paths})
        artifacts[accession] = ArtifactRef.from_path(
            Path(paths["fasta"]),
            kind="sequence",
            format="fasta_gz",
            media_type="application/gzip",
        )

    manifest = build_manifest(
        source="ngdc_gwh",
        organelle=organelle,
        accessions=[str(r["accession"]) for r in kept],
        scope={
            "organelle": organelle,
            "taxon": taxon or "",
            "formats": list(formats),
            # Every GWH accession is absent from NCBI by construction.
            "ngdc_only": True,
        },
        query=f"{GWH_PLANTS}/ [{organelle}]",
        hits_before_filter=total_for_organelle,
        examined=len(records),
    )
    if rejected:
        # Never silent: a folder named like an organelle that is too big to be
        # one is a classification failure, and it gets said out loud.
        manifest["rejected_by_size"] = rejected

    payload: dict[str, Any] = {"manifest": manifest, "records": kept}
    return OrganelleData(
        modality="organelle_records",
        # `OrganelleData`'s `mode="before"` validators accept a plain dict and
        # freeze it themselves at runtime; the cast only satisfies the
        # static type checker, which sees the field's frozen declared type.
        artifacts=cast("FrozenMap[ArtifactRef]", artifacts),
        payload=cast("FrozenMap[FrozenJson]", payload),
    )
