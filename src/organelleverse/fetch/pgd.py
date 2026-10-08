"""PGD (biobigdata.nju.edu.cn) — POST+ZIP against the real ``pgdatabaseAPI``.

Confirmed directly against the live endpoint while writing this plan (not
assumed from the reference script alone): ``POST .../pgdatabaseAPI/download``
with a JSON body ``{"files": "<lowercased_genus_species>.<suffix>"}`` returns
a real ZIP containing the requested ``.gz`` file plus a ``.md5sum`` sidecar.
PGD offers only ``protein``/``genome``/``gff3`` — never ``cds``.

There is no separate "does this species exist" endpoint, so ``fetch_pgd``
always probes the ``protein`` file (the cheapest offered kind) to decide
hit-vs-miss, even when the caller's ``include`` never names ``"protein"``.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import cast

from ..core.artifacts import ArtifactRef
from ..core.data import OrganelleData
from ..core.errors import OrganelleExecutionError, OrganelleParameterError
from ..core.frozen import FrozenJson, FrozenMap
from ._content import SHARED_INCLUDE_KINDS, validate_downloaded_content
from ._http import Transport, default_transport, download_with_retry
from .manifest import build_manifest
from .taxon import resolve_taxon

__all__ = ["fetch_pgd"]

_PGD_API = "https://biobigdata.nju.edu.cn/pgdatabaseAPI/download"
_SUFFIXES = {"protein": "pep.fa.gz", "genome": "genomic.fa.gz", "gff3": "genomic.gff.gz"}
_GFF_FALLBACK_SUFFIX = "longest.gff.gz"
# A real PGD ZIP always carries a real .gz payload plus the .md5sum sidecar,
# so it is never anywhere near this small in practice — this just rejects an
# obviously-empty or truncated body (e.g. a minimal JSON/HTML error response)
# before attempting to unzip it. Matches the threshold used by the reference
# script (``download_from_three_dbs.py``'s ``pgd_download_zip_stream``).
_MIN_ZIP_BYTES = 1000


def _verify_md5sum(sidecar_text: bytes, content: bytes) -> None:
    fields = sidecar_text.decode(errors="replace").split()
    if not fields:
        return  # no sidecar shipped this time — nothing to check against
    expected = fields[0].strip().lower()
    actual = hashlib.md5(content).hexdigest()
    if expected != actual:
        raise OrganelleExecutionError(
            code="network.pgd.content_invalid",
            message="downloaded content does not match PGD's own md5sum sidecar",
            details={"expected": expected, "actual": actual},
            retryable=True,
        )


def _download_pgd_zip(http: Transport, dest_dir: Path, pgd_filename: str) -> bytes | None:
    """Returns the extracted ``.gz`` member's bytes, or ``None`` on a plain miss."""
    with tempfile.TemporaryDirectory(dir=dest_dir) as tmp:
        tmp_zip = Path(tmp) / "response.zip"
        status = download_with_retry(
            http,
            _PGD_API,
            tmp_zip,
            method="POST",
            data=json.dumps({"files": pgd_filename}).encode(),
            headers={"Content-Type": "application/json"},
            timeout=7200.0,
        )
        if status != 200 or not tmp_zip.is_file() or tmp_zip.stat().st_size < _MIN_ZIP_BYTES:
            return None
        try:
            with zipfile.ZipFile(tmp_zip) as archive:
                members = [m for m in archive.infolist() if not m.filename.endswith(".md5sum")]
                if not members:
                    return None
                gz_bytes = archive.read(members[0])
                sidecar = next(
                    (m for m in archive.infolist() if m.filename.endswith(".md5sum")), None
                )
                if sidecar is not None:
                    _verify_md5sum(archive.read(sidecar), gz_bytes)
        except zipfile.BadZipFile:
            raise OrganelleExecutionError(
                code="network.pgd.content_invalid",
                message=f"{pgd_filename} did not come back as a valid ZIP",
                details={"filename": pgd_filename},
                retryable=True,
            ) from None
    return gz_bytes


def _fetch_one_kind(http: Transport, dest_dir: Path, sp_lower: str, kind: str) -> Path | None:
    candidates = [_SUFFIXES[kind]]
    if kind == "gff3":
        candidates.append(_GFF_FALLBACK_SUFFIX)

    for suffix in candidates:
        gz_bytes = _download_pgd_zip(http, dest_dir, f"{sp_lower}.{suffix}")
        if gz_bytes is None:
            continue
        final_path = dest_dir / f"{kind}.gz"
        # Validate a same-filesystem temp copy before renaming into place, so
        # bad content never lands at the real output path.
        with tempfile.TemporaryDirectory(dir=dest_dir) as tmp:
            tmp_path = Path(tmp) / final_path.name
            tmp_path.write_bytes(gz_bytes)
            validate_downloaded_content(tmp_path, kind=kind, gzipped=True, source="pgd")
            tmp_path.replace(final_path)
        return final_path
    return None


def fetch_pgd(
    *,
    taxon: str,
    dest: str | Path,
    include: tuple[str, ...] = ("protein",),
    transport: Transport | None = None,
) -> OrganelleData:
    """Fetch a nuclear assembly from PGD via its real POST+ZIP download endpoint."""
    unknown = set(include) - SHARED_INCLUDE_KINDS
    if unknown:
        raise OrganelleParameterError(
            code="input.unknown_include",
            message=f"unknown include: {sorted(unknown)}",
            details={"supported": sorted(SHARED_INCLUDE_KINDS)},
        )

    resolved = resolve_taxon(taxon)
    http = transport or default_transport()
    sp_lower = f"{resolved.genus}_{resolved.species}".lower()
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    fetched: dict[str, Path | None] = {}

    def ensure(kind: str) -> Path | None:
        if kind not in fetched:
            fetched[kind] = _fetch_one_kind(http, dest_dir, sp_lower, kind)
        return fetched[kind]

    existence = ensure("protein")

    files: dict[str, str] = {}
    artifacts: dict[str, ArtifactRef] = {}
    missing: list[dict[str, str]] = []

    for kind in include:
        if kind not in _SUFFIXES:
            missing.append({"kind": kind, "reason": "not_offered_by_this_source"})
            continue
        path = ensure(kind)
        if path is None:
            missing.append({"kind": kind, "reason": "no_genomic_file_for_this_species"})
            continue
        files[kind] = str(path)
        artifact_format = "gff3_gz" if kind == "gff3" else "fasta_gz"
        artifacts[kind] = ArtifactRef.from_path(
            path, kind=kind, format=artifact_format, media_type="application/gzip"
        )

    # "protein" was fetched purely as the existence probe (see this function's
    # docstring) and the caller never actually asked for it — don't leave an
    # unrequested file sitting in dest.
    probed_protein_path = fetched.get("protein")
    if "protein" not in include and probed_protein_path is not None:
        probed_protein_path.unlink(missing_ok=True)

    if existence is None:
        manifest = build_manifest(
            source="pgd",
            organelle="nuclear",
            accessions=[],
            scope={"taxon": taxon, "include": list(include)},
            hits_before_filter=0,
            examined=0,
        )
        return OrganelleData(
            modality="nuclear_assemblies",
            payload=cast("FrozenMap[FrozenJson]", {"manifest": manifest, "records": []}),
        )

    accession = f"pgd:{resolved.genus}_{resolved.species}"
    record = {
        "accession": accession,
        "organism": resolved.binomial,
        "source": "pgd",
        "confidence": None,
        "files": files,
        "missing": missing,
    }
    manifest = build_manifest(
        source="pgd",
        organelle="nuclear",
        accessions=[accession],
        scope={"taxon": taxon, "include": list(include)},
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
