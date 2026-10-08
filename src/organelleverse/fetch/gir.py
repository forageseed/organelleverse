"""GIR (plantgir.cn) — FTP-only, no browsable index.

Organized ``/<Family>/(vX.Y)Genus_species/``, one directory per species
carrying a version tag. There is no species→family index anywhere on the
site (checked directly: the site has no reachable HTTPS service at all), so
every lookup walks every family's listing — exactly what the prior manual
workflow's own script (``download_from_three_dbs.py``'s
``find_gir_species_online``) does. That cost is inherent to the source, not
a defect in this module.
"""

from __future__ import annotations

import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Protocol, cast

from ..core.artifacts import ArtifactRef
from ..core.data import OrganelleData
from ..core.errors import OrganelleExecutionError, OrganelleParameterError
from ..core.frozen import FrozenJson, FrozenMap
from ._content import SHARED_INCLUDE_KINDS, validate_downloaded_content
from ._http import Transport
from .manifest import build_manifest
from .taxon import resolve_taxon

__all__ = ["FtpClient", "FtplibClient", "fetch_gir"]

_GIR_HOST = "plantgir.cn"
_GIR_PORT = 21
_GIR_USER = "root"
_GIR_PASSWORD_ENV = "ORGANELLEVERSE_GIR_FTP_PASSWORD"

# include kind -> (file suffix, miss reason when a matching file never showed
# up in the species directory's own listing).
_SUFFIX = {
    "protein": "pro.fa.tar.gz",
    "cds": "cds.fa.tar.gz",
    "gff3": "gff.tar.gz",
    "genome": "genome.fa.tar.gz",
}
_TAR_MEMBER_SUFFIX = {
    "protein": ".fa",
    "cds": ".fa",
    "gff3": ".gff3",
    "genome": ".fa",
}
_VERSION_DIR = re.compile(r"^\(v([\d.]+)\)(.+)$")


class FtpClient(Protocol):
    """GIR's FTP surface. Stays private to this module — no other source needs FTP."""

    def list_families(self) -> list[str]: ...

    def list_species_dirs(self, family: str) -> list[str]: ...

    def list_files(self, family: str, species_dir: str) -> list[str]: ...

    def download(self, remote_path: str, dest: Path) -> None: ...


class FtplibClient:
    """The real GIR FTP client. Imports ``ftplib`` lazily, per the discovery-safety rule."""

    def __init__(self, *, password: str) -> None:
        self._password = password

    def _connect(self):  # ftplib.FTP, imported lazily
        import ftplib

        ftp = ftplib.FTP()
        try:
            ftp.connect(_GIR_HOST, _GIR_PORT, timeout=30)
            ftp.login(_GIR_USER, self._password)
        except ftplib.error_perm as exc:
            ftp.close()
            raise OrganelleExecutionError(
                code="network.gir.ftp_login_failed",
                message=f"GIR FTP login failed: {exc}",
                retryable=False,
            ) from exc
        except OSError as exc:
            raise OrganelleExecutionError(
                code="network.gir.ftp_unreachable",
                message=f"GIR FTP unreachable: {exc}",
                retryable=True,
            ) from exc
        return ftp

    def list_families(self) -> list[str]:
        ftp = self._connect()
        try:
            return sorted(name.strip("/") for name in ftp.nlst("/"))
        finally:
            ftp.quit()

    def list_species_dirs(self, family: str) -> list[str]:
        ftp = self._connect()
        try:
            return sorted(name.rsplit("/", 1)[-1] for name in ftp.nlst(f"/{family}"))
        finally:
            ftp.quit()

    def list_files(self, family: str, species_dir: str) -> list[str]:
        ftp = self._connect()
        try:
            return sorted(name.rsplit("/", 1)[-1] for name in ftp.nlst(f"/{family}/{species_dir}"))
        finally:
            ftp.quit()

    def download(self, remote_path: str, dest: Path) -> None:
        import ftplib

        ftp = self._connect()
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as handle:
                ftp.retrbinary(f"RETR {remote_path}", handle.write)
        except ftplib.error_perm as exc:
            raise OrganelleExecutionError(
                code="network.gir.ftp_file_missing",
                message=f"GIR FTP file not found: {remote_path}",
                details={"remote_path": remote_path},
                retryable=False,
            ) from exc
        finally:
            ftp.quit()


def _default_ftp_client() -> FtpClient:
    import os

    password = os.environ.get(_GIR_PASSWORD_ENV)
    if not password:
        from ..core.errors import OrganelleDependencyError

        raise OrganelleDependencyError(
            code="dependency.gir_ftp_credential_missing",
            message=f"{_GIR_PASSWORD_ENV} is not set",
            details={"env_var": _GIR_PASSWORD_ENV},
        )
    return FtplibClient(password=password)


def _parse_version(dir_name: str) -> tuple[tuple[int, ...], str]:
    match = _VERSION_DIR.match(dir_name)
    if not match:
        return (0,), dir_name
    parts = tuple(int(piece) for piece in match.group(1).split(".") if piece.isdigit())
    return parts or (0,), match.group(2)


def _find_species_directory(ftp_client: FtpClient, target: str) -> tuple[str, str] | None:
    """Every family's listing, scanned for the highest-version match of ``target``."""
    best: tuple[tuple[int, ...], str, str] | None = None
    for family in ftp_client.list_families():
        for dir_name in ftp_client.list_species_dirs(family):
            version, bare_name = _parse_version(dir_name)
            if bare_name != target:
                continue
            if best is None or version > best[0]:
                best = (version, family, dir_name)
    if best is None:
        return None
    return best[1], best[2]


def fetch_gir(
    *,
    taxon: str,
    dest: str | Path,
    include: tuple[str, ...] = ("protein",),
    transport: Transport | None = None,
    ftp_client: FtpClient | None = None,
) -> OrganelleData:
    """Fetch a nuclear assembly from GIR by walking every family's FTP listing."""
    unknown = set(include) - SHARED_INCLUDE_KINDS
    if unknown:
        raise OrganelleParameterError(
            code="input.unknown_include",
            message=f"unknown include: {sorted(unknown)}",
            details={"supported": sorted(SHARED_INCLUDE_KINDS)},
        )

    resolved = resolve_taxon(taxon)
    client = ftp_client or _default_ftp_client()
    target = f"{resolved.genus}_{resolved.species}"

    located = _find_species_directory(client, target)
    dest_dir = Path(dest)

    if located is None:
        manifest = build_manifest(
            source="gir",
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

    family, species_dir = located
    available_files = set(client.list_files(family, species_dir))
    dest_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    artifacts: dict[str, ArtifactRef] = {}
    missing: list[dict[str, str]] = []

    for kind in include:
        if kind not in _SUFFIX:
            missing.append({"kind": kind, "reason": "not_offered_by_this_source"})
            continue
        suffix = _SUFFIX[kind]
        filename = f"{target}.{suffix}"
        if filename not in available_files:
            missing.append({"kind": kind, "reason": "not_present_for_this_species"})
            continue

        with tempfile.TemporaryDirectory(dir=dest_dir) as tmp:
            tmp_tar = Path(tmp) / filename
            client.download(f"/{family}/{species_dir}/{filename}", tmp_tar)
            extracted = _extract_member(tmp_tar, _TAR_MEMBER_SUFFIX[kind])
            if extracted is None:
                raise OrganelleExecutionError(
                    code="network.gir.content_invalid",
                    message=f"{filename} did not contain a {_TAR_MEMBER_SUFFIX[kind]} member",
                    details={"filename": filename},
                    retryable=True,
                )
            final_path = dest_dir / f"{kind}{_TAR_MEMBER_SUFFIX[kind]}"
            tmp_extracted = Path(tmp) / f"{kind}{_TAR_MEMBER_SUFFIX[kind]}"
            tmp_extracted.write_bytes(extracted)
            validate_downloaded_content(tmp_extracted, kind=kind, gzipped=False, source="gir")
            tmp_extracted.replace(final_path)

        files[kind] = str(final_path)
        artifact_format = "gff3" if kind == "gff3" else "fasta"
        artifacts[kind] = ArtifactRef.from_path(
            final_path, kind=kind, format=artifact_format, media_type="text/plain"
        )

    accession = f"gir:{family}/{species_dir}"
    record: dict[str, Any] = {
        "accession": accession,
        "organism": resolved.binomial,
        "source": "gir",
        "confidence": None,
        "files": files,
        "missing": missing,
    }
    manifest = build_manifest(
        source="gir",
        organelle="nuclear",
        accessions=[accession],
        scope={
            "taxon": taxon,
            "include": list(include),
            "family": family,
            "version_dir": species_dir,
        },
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


def _extract_member(tar_path: Path, suffix: str) -> bytes | None:
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith(suffix)), None)
            if member is None:
                return None
            extracted = tar.extractfile(member)
            return extracted.read() if extracted is not None else None
    except tarfile.TarError:
        raise OrganelleExecutionError(
            code="network.gir.content_invalid",
            message=f"{tar_path.name} is not a valid tar.gz archive",
            details={"path": str(tar_path)},
            retryable=True,
        ) from None
