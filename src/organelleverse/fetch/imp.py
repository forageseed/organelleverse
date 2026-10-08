"""IMP (bic.ac.cn/IMP) — cleaned whitelist first, formula-probe fallback.

No per-record metadata endpoint states an authoritative species name for a
given ID (checked directly: every guessed metadata path returned 404), so
automated ID resolution cannot verify species identity on its own. The
cleaned cache (Task 3) resolves what can be resolved by internal
self-consistency; anything not in it falls back to the same
probe-and-accept-first-existing-ID the prior workflow used, tagged with a
strictly lower confidence — see the design doc's "IMP cache cleaning".
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from ..core.artifacts import ArtifactRef
from ..core.data import OrganelleData
from ..core.errors import OrganelleParameterError
from ..core.frozen import FrozenJson, FrozenMap
from ._content import SHARED_INCLUDE_KINDS, validate_downloaded_content
from ._http import Transport, default_transport, download_with_retry
from .imp_cache import imp_id_formula, load_cleaned_cache
from .manifest import build_manifest
from .taxon import resolve_taxon

__all__ = ["fetch_imp"]

_IMP_IGV = "https://www.bic.ac.cn/IMP/public/data/igv"
_IMP_BLAST = "https://www.bic.ac.cn/IMP/public/data/blast"
_IMP_EXPR = "https://www.bic.ac.cn/IMP/public/data/expr_matrix"

# include kind -> (url template, gzipped?)
_SOURCE: dict[str, tuple[Callable[[str], str], bool]] = {
    "protein": (lambda imp_id: f"{_IMP_BLAST}/{imp_id}.prot.fasta", False),
    "cds": (lambda imp_id: f"{_IMP_BLAST}/{imp_id}.gene.fasta", False),
    "gff3": (lambda imp_id: f"{_IMP_IGV}/{imp_id}/{imp_id}.gff3.gz", True),
    "genome": (lambda imp_id: f"{_IMP_IGV}/{imp_id}/{imp_id}.fa.gz", True),
    "tpm": (lambda imp_id: f"{_IMP_EXPR}/{imp_id}.all.rnaseq.TPM.txt", False),
}
_MAX_PROBE_CANDIDATES = 10
# The content-integrity check (Task 2's ``_content.py``) only knows how to
# recognize FASTA and GFF shapes — TPM is a plain TSV table, out of scope for
# it (the design's own content-integrity language names only ``.fa``/``.gff``
# file kinds). Skip the check for that one kind rather than misapplying a
# FASTA check that would reject every genuine TPM file.
_CONTENT_VALIDATED_KINDS = frozenset({"protein", "cds", "gff3", "genome"})


def _resolve_imp_id(transport: Transport, species_key: str) -> tuple[str, str] | None:
    """``(imp_id, confidence)``, or ``None`` if the species is on neither the
    cleaned whitelist nor answers any of the ten formula-derived candidates."""
    cleaned = load_cleaned_cache()
    if species_key in cleaned:
        return cleaned[species_key]["imp_id"], "heuristic_consistent"

    prefix = imp_id_formula(species_key)
    if prefix is None:
        return None
    for i in range(1, _MAX_PROBE_CANDIDATES + 1):
        candidate = f"{prefix}{i}"
        status = transport.head(f"{_IMP_IGV}/{candidate}/{candidate}.fa.gz")
        if status == 200:
            return candidate, "unverified_probe"
    return None


def fetch_imp(
    *,
    taxon: str,
    dest: str | Path,
    include: tuple[str, ...] = ("protein",),
    transport: Transport | None = None,
) -> OrganelleData:
    """Fetch a nuclear assembly from IMP, resolving its internal ID first."""
    unknown = set(include) - SHARED_INCLUDE_KINDS
    if unknown:
        raise OrganelleParameterError(
            code="input.unknown_include",
            message=f"unknown include: {sorted(unknown)}",
            details={"supported": sorted(SHARED_INCLUDE_KINDS)},
        )

    resolved = resolve_taxon(taxon)
    http = transport or default_transport()
    species_key = f"{resolved.genus}_{resolved.species}"

    found = _resolve_imp_id(http, species_key)
    dest_dir = Path(dest)

    if found is None:
        manifest = build_manifest(
            source="imp",
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

    imp_id, confidence = found
    dest_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    artifacts: dict[str, ArtifactRef] = {}
    missing: list[dict[str, str]] = []

    for kind in include:
        if kind not in _SOURCE:
            missing.append({"kind": kind, "reason": "not_offered_by_this_source"})
            continue
        url_for, gzipped = _SOURCE[kind]
        url = url_for(imp_id)
        suffix = ".gz" if gzipped else Path(url).suffix
        final_path = dest_dir / f"{kind}{suffix}"
        # Download and validate inside a same-filesystem temp directory first;
        # only rename into `final_path` after validation passes, so a
        # content-invalid response never lands at the real output path.
        with tempfile.TemporaryDirectory(dir=dest_dir) as tmp:
            tmp_path = Path(tmp) / final_path.name
            status = download_with_retry(http, url, tmp_path)
            if status != 200:
                missing.append({"kind": kind, "reason": "not_found_for_this_id"})
                continue
            if kind in _CONTENT_VALIDATED_KINDS:
                validate_downloaded_content(tmp_path, kind=kind, gzipped=gzipped, source="imp")
            tmp_path.replace(final_path)
        files[kind] = str(final_path)
        base_format = "gff3" if kind == "gff3" else "tsv" if kind == "tpm" else "fasta"
        artifact_format = f"{base_format}_gz" if gzipped else base_format
        artifact_media_type = "application/gzip" if gzipped else "text/plain"
        artifacts[kind] = ArtifactRef.from_path(
            final_path, kind=kind, format=artifact_format, media_type=artifact_media_type
        )

    accession = f"imp:{imp_id}"
    record: dict[str, Any] = {
        "accession": accession,
        "organism": resolved.binomial,
        "source": "imp",
        "confidence": confidence,
        "files": files,
        "missing": missing,
    }
    manifest = build_manifest(
        source="imp",
        organelle="nuclear",
        accessions=[accession],
        scope={"taxon": taxon, "include": list(include), "confidence": confidence},
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
