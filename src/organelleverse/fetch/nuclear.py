"""Nuclear genomes — a different database, not a different query.

Organelle genomes are **nuccore records**; nuclear genomes are **assemblies**
(``GCF_``/``GCA_``). They live behind the NCBI Datasets API by default, and no
Entrez query will return them. This is the mirror image of the organelle
rule: Datasets is useless for organelles, and Entrez is useless for nuclear
assemblies.

The reason OrganelleVerse needs this at all is cytonuclear coevolution: ERC
compares organelle genes against **nuclear** genes, so a run needs both sides.

And it needs the nuclear side as **proteins**, not as sequence. Arabidopsis
TAIR10 is 119 Mb of genome and ~35 MB of protein FASTA; OrthoFinder (the ERC
upstream) consumes the proteins. Downloading the genome to get the proteins is
two orders of magnitude of waste, so ``include`` defaults to proteins only.

``source`` extends this to four supplementary databases (GIR/IMP/PGD/TAIR)
for species NCBI lacks or annotates poorly. The default stays ``"ncbi"``:
every caller, including ``_agent_ops.py``'s Agent-facing operation, is
unaffected until it opts in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, cast

from ..core.artifacts import ArtifactRef
from ..core.data import OrganelleData
from ..core.errors import OrganelleError, OrganelleExecutionError, OrganelleParameterError
from ..core.frozen import FrozenJson, FrozenMap, thaw_json
from ._http import Transport, default_transport, retry_with_backoff
from .gir import fetch_gir
from .imp import fetch_imp
from .manifest import build_manifest
from .pgd import fetch_pgd
from .tair import fetch_tair
from .taxon import ResolvedTaxon, resolve_taxon

__all__ = [
    "DATASETS_API",
    "AssemblyLevel",
    "ResolvedTaxon",
    "fetch_nuclear_genome",
    "parse_assembly_reports",
    "resolve_taxon",
]

DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"

AssemblyLevel = Literal["complete_genome", "chromosome", "scaffold", "contig"]

# What a Datasets download may contain. Proteins are the default because that is
# what ERC's upstream (OrthoFinder) eats; `genome` is the 119 Mb file nobody
# asked for.
_NCBI_INCLUDE = frozenset({"protein", "cds", "gff3", "rna", "genome", "seq-report"})

_KNOWN_SOURCES = frozenset({"auto", "ncbi", "gir", "imp", "pgd", "tair"})


def _as_dict(value: object) -> dict[str, Any]:
    """Narrow an arbitrary JSON-decoded value to a plain dict, or ``{}``.

    ``json.loads`` returns ``Any``, and pyright (this file is in the repo's
    strict-typecheck scope) otherwise infers every chained ``.get(...) or {}``
    as a partially-``Unknown`` type from the bare ``{}``/``[]`` literal
    fallbacks. Routing every nested-dict read through this one narrowing
    point is a pure type-checker satisfaction — it changes no runtime value.
    """
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def parse_assembly_reports(payload: bytes | str) -> list[dict[str, Any]]:
    """Flatten a Datasets ``dataset_report`` into one record per assembly."""
    raw = payload.decode(errors="replace") if isinstance(payload, bytes) else payload
    try:
        body: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OrganelleExecutionError(
            code="network.malformed_response",
            message=f"Datasets did not return JSON: {exc}",
            details={"body": raw[:200]},
            retryable=True,
        ) from exc

    reports: list[dict[str, Any]] = []
    raw_reports = cast("list[object]", body.get("reports") or [])
    for raw_report in raw_reports:
        report = _as_dict(raw_report)
        info = _as_dict(report.get("assembly_info"))
        stats = _as_dict(report.get("assembly_stats"))
        organism = _as_dict(report.get("organism"))
        annotation_info = _as_dict(report.get("annotation_info"))
        annotation_stats = _as_dict(annotation_info.get("stats"))
        gene_counts = _as_dict(annotation_stats.get("gene_counts"))
        accession = str(report.get("accession", ""))
        if not accession:
            continue
        reports.append(
            {
                "accession": accession,
                "assembly_name": info.get("assembly_name", ""),
                "assembly_level": info.get("assembly_level", ""),
                "refseq_category": info.get("refseq_category", ""),
                "organism": organism.get("organism_name", ""),
                "taxid": str(organism.get("tax_id", "")),
                "length": int(stats.get("total_sequence_length", 0) or 0),
                "contig_n50": int(stats.get("contig_n50", 0) or 0),
                "scaffold_n50": int(stats.get("scaffold_n50", 0) or 0),
                "gene_count": int(gene_counts.get("total", 0) or 0),
                "submission_date": info.get("submission_date", ""),
            }
        )
    return reports


def _records_of(data: OrganelleData) -> list[dict[str, Any]]:
    """Thaw a fetcher's own ``records`` list back out of its (frozen)
    ``OrganelleData.payload`` so this dispatcher can inspect
    ``confidence``/``missing``/``files``.

    These records are OrganelleVerse's own output from a moment earlier in
    the same call — never externally supplied input — so the shape is
    already known; the ``cast`` only documents that for the type checker
    without changing what is read.
    """
    thawed = thaw_json(data.payload.get("records", ()))
    return cast("list[dict[str, Any]]", thawed) if isinstance(thawed, list) else []


def fetch_nuclear_genome(
    *,
    taxon: str,
    dest: str | Path,
    source: Literal["auto", "ncbi", "gir", "imp", "pgd", "tair"] = "ncbi",
    assembly_level: AssemblyLevel | None = "chromosome",
    reference_only: bool = True,
    include: tuple[str, ...] = ("protein",),
    max_records: int = 20,
    download: bool = True,
    allow_partial: bool = False,
    transport: Transport | None = None,
) -> OrganelleData:
    """Fetch nuclear assembly metadata, and optionally its protein/CDS files.

    ``source`` defaults to ``"ncbi"`` — every existing caller keeps today's
    behavior unchanged until it explicitly opts into ``"auto"`` or a specific
    alternate source. ``"auto"`` tries TAIR (Arabidopsis only) → NCBI → IMP →
    GIR → PGD in that order, stopping at the first full hit; a source that
    only partially satisfies ``include`` is remembered and returned only if
    every later source also misses. ``allow_partial`` (only meaningful for
    ``"auto"``) treats a partial hit as good enough to stop the chain.

    ``download`` (survey metadata only vs. actually fetch files) applies to
    the NCBI leg only — GIR/IMP/PGD/TAIR always transfer their files
    regardless of ``download``'s value, since none of them has a
    metadata-only survey mode.
    """
    if source not in _KNOWN_SOURCES:
        raise OrganelleParameterError(
            code="input.unknown_source",
            message=f"unknown source: {source!r}",
            details={"supported": sorted(_KNOWN_SOURCES)},
        )

    if source == "ncbi":
        return _fetch_ncbi(
            taxon=taxon,
            dest=dest,
            assembly_level=assembly_level,
            reference_only=reference_only,
            include=include,
            max_records=max_records,
            download=download,
            transport=transport,
        )

    if source in ("gir", "imp", "pgd", "tair"):
        fetcher = {"gir": fetch_gir, "imp": fetch_imp, "pgd": fetch_pgd, "tair": fetch_tair}[source]
        data = fetcher(taxon=taxon, dest=dest, include=include, transport=transport)
        return _annotate_ncbi_only_params(
            data, assembly_level=assembly_level, reference_only=reference_only
        )

    return _fetch_auto(
        taxon=taxon,
        dest=dest,
        assembly_level=assembly_level,
        reference_only=reference_only,
        include=include,
        max_records=max_records,
        download=download,
        allow_partial=allow_partial,
        transport=transport,
    )


def _require_real_dataset_package(blob: bytes, *, accessions, include) -> None:
    """Reject the anonymous-access README stub before it masquerades as data.

    NCBI Datasets v2 answers unauthenticated POST downloads with a valid zip
    that contains only README/catalog/metadata files - no sequence payload.
    Writing that as the requested assembly would silently deliver an empty
    package, so the stub is detected and refused with the exact remedy.
    """
    import io
    import zipfile

    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        return  # not a zip at all; downstream consumers will report it
    names = archive.namelist()
    # the stub's data/ directory carries only package metadata files; a real
    # package also holds sequence/annotation payload files beneath them
    _METADATA_NAMES = {
        "assembly_data_report.jsonl",
        "dataset_catalog.json",
        "md5sum.txt",
        "README.md",
    }
    payload = [
        n
        for n in names
        if n.startswith("ncbi_dataset/data/")
        and not n.endswith("/")
        and n.rsplit("/", 1)[-1] not in _METADATA_NAMES
    ]
    if names and not payload:
        raise OrganelleExecutionError(
            code="fetch.nuclear_download_stub",
            message=(
                "NCBI returned a metadata-only dataset package (no sequence "
                "payload). Anonymous v2 downloads are stubbed; supply an "
                "API key via NCBI_API_KEY, or use download=False for "
                "metadata-only, or pick another source (gir/imp/pgd/tair)."
            ),
            details={
                "accessions": list(accessions),
                "include": list(include),
                "package_entries": names[:8],
            },
        )


def _fetch_ncbi(
    *,
    taxon: str,
    dest: str | Path,
    assembly_level: AssemblyLevel | None,
    reference_only: bool,
    include: tuple[str, ...],
    max_records: int,
    download: bool,
    transport: Transport | None,
) -> OrganelleData:
    """NCBI Datasets — unchanged from before ``source`` existed."""
    from urllib.parse import quote

    unknown = set(include) - _NCBI_INCLUDE
    if unknown:
        raise OrganelleParameterError(
            code="input.unknown_include",
            message=f"unknown include: {sorted(unknown)}",
            details={"supported": sorted(_NCBI_INCLUDE)},
        )

    http = transport or default_transport()
    url = f"{DATASETS_API}/genome/taxon/{quote(taxon)}/dataset_report?page_size={max_records}"
    if assembly_level:
        url += f"&filters.assembly_level={assembly_level}"
    if reference_only:
        # The representative assembly for the species, not all 265 of them.
        url += "&filters.reference_only=true"

    body = retry_with_backoff(lambda: http.get(url, timeout=120.0))
    reports = parse_assembly_reports(body)

    scope: dict[str, Any] = {
        "taxon": taxon,
        "assembly_level": assembly_level or "any",
        "reference_only": reference_only,
        "include": list(include),
    }

    artifacts: dict[str, tuple[Path, str, str]] = {}
    if download and reports:
        dest_dir = Path(dest)
        dest_dir.mkdir(parents=True, exist_ok=True)
        accession_ids = [r["accession"] for r in reports]
        include_q = "&".join(f"include_annotation_type={kind.upper()}" for kind in include)
        # v2 download is POST + JSON body; the legacy GET-by-accession-path form
        # returns HTTP 400 for every caller (verified live 2026-08-15).
        download_url = f"{DATASETS_API}/genome/download?{include_q}"
        archive = dest_dir / f"{taxon.replace(' ', '_')}_nuclear.zip"
        import json as _json

        request_body = _json.dumps({"accessions": accession_ids}).encode("utf-8")
        blob = retry_with_backoff(
            lambda: http.post(
                download_url,
                request_body,
                timeout=1800.0,
                headers={"Content-Type": "application/json"},
            )
        )
        _require_real_dataset_package(blob, accessions=accession_ids, include=include)
        archive.write_bytes(blob)
        artifacts["assemblies"] = (archive, "zip", "application/zip")

    manifest = build_manifest(
        source="nuclear_assembly",
        organelle="nuclear",
        accessions=[r["accession"] for r in reports],
        scope=scope,
        query=url,
        hits_before_filter=len(reports),
        examined=len(reports),
    )

    refs: dict[str, ArtifactRef] = {}
    for name, (path, fmt, media) in artifacts.items():
        if path.is_file():
            refs[name] = ArtifactRef.from_path(path, kind=name, format=fmt, media_type=media)

    return OrganelleData(
        modality="nuclear_assemblies",
        # `OrganelleData`'s `mode="before"` validators accept a plain dict and
        # freeze it themselves at runtime; the cast only satisfies the static
        # type checker, which otherwise sees the field's frozen declared type
        # (`FrozenMap[...]`) as incompatible with a plain `dict`.
        artifacts=cast("FrozenMap[ArtifactRef]", refs),
        payload=cast("FrozenMap[FrozenJson]", {"manifest": manifest, "records": reports}),
    )


def _thawed_dict(value: object) -> dict[str, Any]:
    """Thaw one nested frozen-JSON field (``manifest``, ``scope``, ...) back
    into a plain, mutable dict for a read-modify-refreeze round trip, or
    ``{}`` if it is missing/not an object.

    ``thaw_json`` recursively converts any nested ``FrozenMap``/``tuple`` it
    finds and passes anything else through unchanged, so it is safe to call
    here whether ``value`` is a still-frozen field straight out of
    ``OrganelleData.payload`` or an already-plain dict.
    """
    if value is None:
        return {}
    thawed = thaw_json(cast("FrozenJson", value))
    return cast("dict[str, Any]", thawed) if isinstance(thawed, dict) else {}


def _as_list(value: object) -> list[Any]:
    """Narrow an arbitrary decoded value to a plain list, or ``[]``. Same
    type-checker-only purpose as ``_as_dict``."""
    return cast("list[Any]", value) if isinstance(value, list) else []


def _annotate_ncbi_only_params(
    data: OrganelleData, *, assembly_level: AssemblyLevel | None, reference_only: bool
) -> OrganelleData:
    """Record that ``assembly_level``/``reference_only`` were requested but
    don't apply outside NCBI, rather than silently dropping them."""
    payload: dict[str, Any] = dict(data.payload)
    manifest = _thawed_dict(payload.get("manifest"))
    scope = _thawed_dict(manifest.get("scope"))
    scope["assembly_level"] = {"value": assembly_level or "any", "applies_to": "ncbi_only"}
    scope["reference_only"] = {"value": reference_only, "applies_to": "ncbi_only"}
    manifest["scope"] = scope
    payload["manifest"] = manifest
    return OrganelleData(
        modality=data.modality,
        artifacts=cast("FrozenMap[ArtifactRef]", dict(data.artifacts)),
        payload=cast("FrozenMap[FrozenJson]", payload),
    )


def _is_full_hit(data: OrganelleData, allow_partial: bool) -> bool:
    records = _records_of(data)
    if not records:
        return False
    missing = _as_list(records[0].get("missing"))
    return not missing or allow_partial


def _candidate_dir(dest_dir: Path, source: str) -> Path:
    return dest_dir / ".auto_candidates" / source


def _cleanup_candidates(dest_dir: Path) -> None:
    import shutil

    candidates_root = dest_dir / ".auto_candidates"
    if candidates_root.is_dir():
        shutil.rmtree(candidates_root, ignore_errors=True)


def _publish(
    dest_dir: Path, source: str, data: OrganelleData, sources_tried: list[dict[str, Any]]
) -> OrganelleData:
    """Promote ``source``'s files from its candidate temp directory into
    ``dest_dir``, and delete every candidate's temp directory — the winner's
    (now empty) and every loser's alike. Resolves the design doc's Known
    gap #1: only the finally-selected source ever writes to ``dest``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    new_artifacts: dict[str, ArtifactRef] = {}
    new_paths: dict[str, str] = {}
    for kind, artifact in data.artifacts.items():
        old_path = Path(artifact.uri)
        new_path = dest_dir / old_path.name
        if old_path.resolve() != new_path.resolve():
            os.replace(old_path, new_path)
        new_artifacts[kind] = artifact.model_copy(update={"uri": str(new_path)})
        new_paths[kind] = str(new_path)

    records = _records_of(data)
    for entry in records:
        files = dict(entry.get("files") or {})
        for kind in list(files):
            if kind in new_paths:
                files[kind] = new_paths[kind]
        entry["files"] = files

    payload: dict[str, Any] = dict(data.payload)
    payload["records"] = records
    manifest = _thawed_dict(payload.get("manifest"))
    manifest["sources_tried"] = list(sources_tried)
    payload["manifest"] = manifest

    _cleanup_candidates(dest_dir)
    return OrganelleData(
        modality=data.modality,
        artifacts=cast("FrozenMap[ArtifactRef]", new_artifacts),
        payload=cast("FrozenMap[FrozenJson]", payload),
    )


def _fetch_auto(
    *,
    taxon: str,
    dest: str | Path,
    assembly_level: AssemblyLevel | None,
    reference_only: bool,
    include: tuple[str, ...],
    max_records: int,
    download: bool,
    allow_partial: bool,
    transport: Transport | None,
) -> OrganelleData:
    """TAIR (Arabidopsis only) → NCBI → IMP → GIR → PGD, stopping at the
    first full hit. Priority order per the design doc's citation of
    ``download_from_three_dbs.py``'s own header: "IMP(本地) > GIR(本地) >
    IMP(在线) > GIR(在线) > PGD(在线)" — IMP before GIR, not after.
    """
    dest_dir = Path(dest)
    sources_tried: list[dict[str, Any]] = []
    fallback: tuple[str, OrganelleData] | None = None

    def record(entry_source: str, status: str, detail: str) -> None:
        sources_tried.append({"source": entry_source, "status": status, "detail": detail})

    try:
        resolved: ResolvedTaxon | None = resolve_taxon(taxon)
    except OrganelleParameterError:
        resolved = None  # each source's own leg re-resolves and may reject independently

    if resolved is not None and resolved.genus == "Arabidopsis" and resolved.species == "thaliana":
        try:
            data = fetch_tair(
                taxon=taxon,
                dest=_candidate_dir(dest_dir, "tair"),
                include=include,
                transport=transport,
            )
            if _is_full_hit(data, allow_partial):
                record("tair", "hit", "arabidopsis thaliana")
                return _publish(dest_dir, "tair", data, sources_tried)
            if data.payload.get("records"):
                record("tair", "partial", "missing requested kinds")
                fallback = fallback or ("tair", data)
            else:
                record("tair", "miss", "no current release files found")
        except OrganelleError as error:
            record("tair", "error", str(error))

    try:
        data = _fetch_ncbi(
            taxon=taxon,
            dest=_candidate_dir(dest_dir, "ncbi"),
            assembly_level=assembly_level,
            reference_only=reference_only,
            include=include,
            max_records=max_records,
            download=download,
            transport=transport,
        )
        if data.payload.get("records"):
            record("ncbi", "hit", "matched by NCBI Datasets")
            return _publish(dest_dir, "ncbi", data, sources_tried)
        record("ncbi", "miss", "no NCBI assembly")
    except OrganelleError as error:
        record("ncbi", "error", str(error))

    try:
        data = fetch_imp(
            taxon=taxon, dest=_candidate_dir(dest_dir, "imp"), include=include, transport=transport
        )
        records = _records_of(data)
        # `_resolve_imp_id` (imp.py) only ever returns "heuristic_consistent"
        # (cleaned whitelist match) or "unverified_probe" (fresh formula
        # probe) — never "verified": nothing in this codebase can
        # independently confirm IMP species identity (see imp_cache.py's
        # module docstring). So this branch cannot fire today, and IMP always
        # falls through to the "unconfirmed" fallback below, letting a later
        # GIR/PGD full hit win even over a whitelisted IMP match. That is
        # deliberate, not dead code: it is kept so that if IMP ever gains a
        # real verification mechanism, a confirmed hit would correctly
        # short-circuit `auto` the same way GIR/PGD do today. Do not "fix"
        # this to `== "heuristic_consistent"` — that would silently drop the
        # safety-motivated precedence order.
        if records and records[0].get("confidence") == "verified":
            record("imp", "hit", "verified")
            return _publish(dest_dir, "imp", data, sources_tried)
        if records:
            record("imp", "unconfirmed", str(records[0].get("confidence")))
            fallback = fallback or ("imp", data)
        else:
            record("imp", "miss", "no candidate ID answered")
    except OrganelleError as error:
        record("imp", "error", str(error))

    try:
        data = fetch_gir(
            taxon=taxon, dest=_candidate_dir(dest_dir, "gir"), include=include, transport=transport
        )
        if _is_full_hit(data, allow_partial):
            record("gir", "hit", "matched by family/version listing")
            return _publish(dest_dir, "gir", data, sources_tried)
        if data.payload.get("records"):
            record("gir", "partial", "missing requested kinds")
            fallback = fallback or ("gir", data)
        else:
            record("gir", "miss", "species not found under any family")
    except OrganelleError as error:
        record("gir", "error", str(error))

    try:
        data = fetch_pgd(
            taxon=taxon, dest=_candidate_dir(dest_dir, "pgd"), include=include, transport=transport
        )
        if _is_full_hit(data, allow_partial):
            record("pgd", "hit", "matched by species-name probe")
            return _publish(dest_dir, "pgd", data, sources_tried)
        if data.payload.get("records"):
            record("pgd", "partial", "missing requested kinds")
            fallback = fallback or ("pgd", data)
        else:
            record("pgd", "miss", "no genomic file for this species")
    except OrganelleError as error:
        record("pgd", "error", str(error))

    if fallback is not None:
        name, data = fallback
        return _publish(dest_dir, name, data, sources_tried)

    _cleanup_candidates(dest_dir)
    raise OrganelleExecutionError(
        code="network.auto_exhausted",
        message=f"no source found a nuclear assembly for {taxon!r}",
        details={"sources_tried": sources_tried},
        retryable=False,
    )
