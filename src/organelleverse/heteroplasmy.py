"""Site-level alternate allele fraction estimator for supplied organelle SNVs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from organelleverse.core.provenance import ResultProvenance
from organelleverse.core.result import ErrorDetail, Finding, OrganelleResult

_VERSION = "1.0"


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _provenance(
    op: str, *, parameters: dict, backend: str, software_versions: dict | None = None
) -> ResultProvenance:
    payload = json.dumps(
        parameters, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return ResultProvenance(
        operation_id=f"heteroplasmy.{op}",
        operation_version=_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=hashlib.sha256(payload).hexdigest(),
        requested_backend=backend,
        actual_backend=backend,
        attempted_backends=(backend,),
        software_versions=software_versions or {},
    )


def _failed(op: str, *, scope: str, code: str, message: str, backend: str) -> OrganelleResult:
    return OrganelleResult(
        operation_id=f"heteroplasmy.{op}",
        operation_version=_VERSION,
        scope=scope,
        status="failed",
        summary_text=message,
        errors=(ErrorDetail(code=code, message=message),),
        provenance=_provenance(op, parameters={}, backend=backend),
    )


def quantify_variant_fractions(
    bam_path: str | Path,
    sites_tsv: str | Path,
    *,
    scope: Literal["mitochondrion", "plastid"] = "mitochondrion",
    min_mapping_quality: int = 20,
    min_base_quality: int = 20,
    min_depth: int = 10,
    exclude_duplicates: bool = True,
    reference_fasta: str | Path | None = None,
    mask_regions_bed: str | Path | None = None,
    homology_reference_fasta: str | Path | None = None,
    homology_min_identity: float = 80.0,
    homology_min_length: int = 100,
    auto_homology_mask: bool = True,
    nuclear_reference_fasta: str | Path | None = None,
) -> OrganelleResult:
    """Measure fractions only at supplied sites; TSV: site_id,seqid,position,ref,alt.
    Uses indexed BAM/CRAM, quality-filtered A/C/G/T bases and Wilson 95% intervals.
    Not a variant caller; does not infer structural paths or model mapping bias.

    Reads from the other organelle (MTPT/NUPT-like inserts) map onto homologous
    stretches and look like heteroplasmy: in a real Nipponbare run all 111
    high-quality mitochondrial candidates sat in plastid-derived inserts. Sites
    inside masked regions are therefore not piled up and get status ``masked``.
    Masked regions come from ``mask_regions_bed`` (BED, 0-based half-open) and/or
    from ``homology_reference_fasta`` (the other organelle genome; regions of
    ``reference_fasta`` that match it at >= ``homology_min_identity`` percent over
    >= ``homology_min_length`` bp, found with ``transfer.detect_transfers_blast``).
    By default (``auto_homology_mask=True``) a mitochondrial run that has a
    ``reference_fasta`` but no ``homology_reference_fasta`` searches the plastomes
    bundled with the package (annotation reference library, 46 species) instead.
    That matches masking with the sample's own plastome when a close relative is
    bundled and degrades with distance (Nipponbare MTPT bases found: 99.9% without
    Oryza, 82% without any Poaceae, 23% with only non-angiosperms), so pass the
    sample's own plastome whenever you have it. If the automatic search cannot run,
    the result stays unmasked and is flagged ``homology_mask_unavailable`` (an
    explicit request that fails is an error instead). With no masking at all the
    result is flagged ``numt_mapping_not_resolved``. Masking only covers homology
    that the search finds; it does not prove the remaining sites are organelle-only.

    Nuclear copies (NUMT for a mitochondrion, NUPT for a plastid) attract reads the
    same way. Pass ``nuclear_reference_fasta`` (the nuclear genome with the organelle
    sequences removed; a record with the same name as a ``reference_fasta`` contig is
    skipped) to also mask the regions of ``reference_fasta`` that have a copy there at
    >= ``homology_min_identity`` percent over >= ``homology_min_length`` bp (mappy,
    ``asm20``; about 20 s for the 370 Mb rice genome). Nothing is bundled for this,
    so there is no automatic nuclear mask. A mask covering >= 80% of the reference is
    flagged ``homology_mask_covers_most_of_reference``; check then that the nuclear
    FASTA does not still contain organelle sequence.
    """
    op = "quantify_variant_fractions"
    params = {
        "bam_path": str(bam_path),
        "sites_tsv": str(sites_tsv),
        "scope": scope,
        "min_mapping_quality": min_mapping_quality,
        "min_base_quality": min_base_quality,
        "min_depth": min_depth,
        "exclude_duplicates": exclude_duplicates,
        "reference_fasta": str(reference_fasta) if reference_fasta is not None else None,
        "mask_regions_bed": str(mask_regions_bed) if mask_regions_bed is not None else None,
        "homology_reference_fasta": (
            str(homology_reference_fasta) if homology_reference_fasta is not None else None
        ),
        "homology_min_identity": homology_min_identity,
        "homology_min_length": homology_min_length,
        "auto_homology_mask": auto_homology_mask,
        "nuclear_reference_fasta": (
            str(nuclear_reference_fasta) if nuclear_reference_fasta is not None else None
        ),
    }
    if scope not in ("mitochondrion", "plastid"):
        return _failed(
            op,
            scope="mitochondrion",
            code="heteroplasmy.invalid_scope",
            message="scope must be mitochondrion or plastid.",
            backend="pysam",
        )
    if min_mapping_quality < 0 or min_base_quality < 0 or min_depth < 1:
        return _failed(
            op,
            scope=scope,
            code="heteroplasmy.invalid_threshold",
            message="Quality thresholds must be non-negative and min_depth >= 1.",
            backend="pysam",
        )
    if not 0 < homology_min_identity <= 100 or homology_min_length < 1:
        return _failed(
            op,
            scope=scope,
            code="heteroplasmy.invalid_threshold",
            message="homology_min_identity must be in (0, 100] and homology_min_length >= 1.",
            backend="pysam",
        )
    if (
        homology_reference_fasta is not None or nuclear_reference_fasta is not None
    ) and reference_fasta is None:
        return _failed(
            op,
            scope=scope,
            code="heteroplasmy.mask_needs_reference",
            message=(
                "homology masking searches reference_fasta (the genome the BAM was "
                "aligned to) against homology_reference_fasta / nuclear_reference_fasta; "
                "pass reference_fasta."
            ),
            backend="pysam",
        )
    try:
        sites = _read_sites(sites_tsv)
    except (OSError, ValueError) as exc:
        return _failed(
            op,
            scope=scope,
            code="heteroplasmy.invalid_sites",
            message=f"Could not read candidate-site TSV: {exc}",
            backend="pysam",
        )
    try:
        import pysam
    except ImportError:
        return _failed(
            op,
            scope=scope,
            code="heteroplasmy.pysam_missing",
            message="Install organelleverse[qc] for BAM/CRAM pileups.",
            backend="pysam",
        )
    try:
        masked, mask_versions = _masked_intervals(
            mask_regions_bed,
            homology_reference_fasta,
            reference_fasta,
            homology_min_identity,
            homology_min_length,
            other="plastid" if scope == "mitochondrion" else "mitochondrion",
        )
    except _MaskError as exc:
        return _failed(op, scope=scope, code=exc.code, message=str(exc), backend="pysam")
    homology_mode = "explicit_fasta" if homology_reference_fasta is not None else "off"
    homology_note: str | None = None
    homology_references: int | None = None
    if homology_reference_fasta is None and auto_homology_mask:
        if scope != "mitochondrion":
            homology_note = (
                "automatic homology masking covers mitochondrial runs only "
                "(no mitochondrial genomes are bundled)"
            )
        elif reference_fasta is None:
            homology_note = "automatic homology masking needs reference_fasta"
        else:
            try:
                auto_masked, auto_versions, homology_references = _auto_plastome_intervals(
                    reference_fasta, homology_min_identity, homology_min_length
                )
            except _MaskError as exc:
                homology_note = str(exc)
            else:
                for contig, items in auto_masked.items():
                    masked.setdefault(contig, []).extend(items)
                mask_versions.update(auto_versions)
                homology_mode = "bundled_plastomes"
    nuclear_interval_count = 0
    if nuclear_reference_fasta is not None:
        try:
            nuclear_masked, nuclear_versions = _nuclear_intervals(
                reference_fasta, nuclear_reference_fasta, homology_min_identity, homology_min_length
            )
        except _MaskError as exc:
            return _failed(op, scope=scope, code=exc.code, message=str(exc), backend="pysam")
        for contig, items in nuclear_masked.items():
            masked.setdefault(contig, []).extend(items)
            nuclear_interval_count += len(items)
        mask_versions.update(nuclear_versions)
    mask_applied = (
        mask_regions_bed is not None
        or homology_mode != "off"
        or nuclear_reference_fasta is not None
    )
    mask_unavailable = homology_note is not None and scope == "mitochondrion"
    masked_sites: dict[str, str] = {}
    ref_lengths: dict[str, int] = {}
    counts = {s["site_id"]: Counter() for s in sites}
    try:
        kw = {"reference_filename": str(reference_fasta)} if reference_fasta is not None else {}
        with pysam.AlignmentFile(str(bam_path), "r", **kw) as bam:
            ref_lengths = dict(zip(bam.references, bam.lengths, strict=True))
            for site in sites:
                contig, pos = site["seqid"], site["position"] - 1
                if contig not in bam.references:
                    raise ValueError(f"BAM has no reference sequence {contig!r}")
                source = _mask_source(masked, contig, pos)
                if source is not None:
                    masked_sites[site["site_id"]] = source
                    continue
                for col in bam.pileup(
                    contig,
                    pos,
                    pos + 1,
                    truncate=True,
                    stepper="all",
                    flag_filter=(
                        pysam.FUNMAP
                        | pysam.FSECONDARY
                        | pysam.FQCFAIL
                        | (pysam.FDUP if exclude_duplicates else 0)
                    ),
                    min_mapping_quality=0,
                    min_base_quality=0,
                    max_depth=1000000,
                ):
                    if col.reference_pos != pos:
                        continue
                    for item in col.pileups:
                        r = item.alignment
                        if (
                            item.is_del
                            or item.is_refskip
                            or r.is_unmapped
                            or r.is_secondary
                            or r.is_supplementary
                            or r.is_qcfail
                        ):
                            continue
                        if exclude_duplicates and r.is_duplicate:
                            continue
                        q = item.query_position
                        if (
                            r.mapping_quality < min_mapping_quality
                            or q is None
                            or r.query_qualities is None
                            or r.query_qualities[q] < min_base_quality
                        ):
                            continue
                        b = r.query_sequence[q].upper()
                        if b in "ACGT":
                            counts[site["site_id"]][b] += 1
    except (OSError, ValueError, KeyError) as exc:
        return _failed(
            op,
            scope=scope,
            code="heteroplasmy.pileup_failed",
            message=f"Could not quantify BAM pileup: {exc}",
            backend="pysam",
        )
    rows = []
    for s in sites:
        c = counts[s["site_id"]]
        ref, alt = c[s["ref"]], c[s["alt"]]
        depth = sum(c.values())
        f = alt / depth if depth >= min_depth else None
        lo, hi = _wilson(alt, depth) if f is not None else (None, None)
        is_masked = s["site_id"] in masked_sites
        rows.append(
            {
                **s,
                "ref_count": ref,
                "alt_count": alt,
                "other_count": depth - ref - alt,
                "depth": depth,
                "alt_fraction": round(f, 6) if f is not None else None,
                "ci95_lower": round(lo, 6) if lo is not None else None,
                "ci95_upper": round(hi, 6) if hi is not None else None,
                "status": "masked" if is_masked else ("ok" if f is not None else "low_depth"),
                "mask_source": masked_sites.get(s["site_id"]),
            }
        )
    n = sum(row["status"] == "ok" for row in rows)
    n_masked = len(masked_sites)
    sv = {"pysam": pysam.__version__, **mask_versions}
    total_bp = sum(ref_lengths.values())
    masked_bp = sum(
        _merged_bp(items, ref_lengths.get(contig, 0)) for contig, items in masked.items()
    )
    masked_fraction = round(masked_bp / total_bp, 4) if total_bp else 0.0
    mask_excessive = mask_applied and masked_fraction >= _EXCESSIVE_MASK_FRACTION
    return OrganelleResult(
        operation_id=f"heteroplasmy.{op}",
        operation_version="1.0",
        scope=scope,
        status="ok",
        summary_text=(
            f"Estimated alternate-allele fractions at {n}/{len(rows)} supplied sites"
            + (f"; {n_masked} masked in homologous regions." if mask_applied else ".")
            + (
                f" Automatic homology mask unavailable: {homology_note}."
                if mask_unavailable
                else ""
            )
            + (
                f" {masked_fraction:.0%} of the reference is masked; check that the nuclear "
                "FASTA does not contain organelle sequence."
                if mask_excessive
                else ""
            )
        ),
        metrics={
            "site_count": len(rows),
            "evaluated_sites": n,
            "masked_sites": n_masked,
            "masked_interval_count": sum(len(v) for v in masked.values()),
            "masked_reference_fraction": masked_fraction,
            "homology_mask": {
                "mode": homology_mode,
                "bed": mask_regions_bed is not None,
                "bundled_references": homology_references,
                "nuclear": nuclear_reference_fasta is not None,
                "nuclear_intervals": nuclear_interval_count,
                "note": homology_note,
            },
            "sites": rows,
        },
        findings=(
            Finding(code="heteroplasmy.evaluated_sites", metric="evaluated_sites", value=n),
            *(
                (Finding(code="heteroplasmy.masked_sites", metric="masked_sites", value=n_masked),)
                if mask_applied
                else ()
            ),
        ),
        flags=(
            "candidate_sites_only",
            "read_fraction_not_molecule_fraction",
            "homologous_regions_masked" if mask_applied else "numt_mapping_not_resolved",
            *(
                ("homology_mask_from_bundled_references",)
                if homology_mode == "bundled_plastomes"
                else ()
            ),
            *(("numt_regions_masked",) if nuclear_reference_fasta is not None else ()),
            *(("homology_mask_covers_most_of_reference",) if mask_excessive else ()),
            *(("homology_mask_unavailable",) if mask_unavailable else ()),
        ),
        provenance=_provenance(
            op, parameters=params, backend="pysam", software_versions=sv
        ),
    )


def _read_sites(path: str | Path) -> list[dict]:
    import csv

    need = {"site_id", "seqid", "position", "ref", "alt"}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=chr(9))
        if reader.fieldnames is None or not need.issubset(reader.fieldnames):
            raise ValueError("required TSV header: site_id,seqid,position,ref,alt")
        out = []
        seen = set()
        for line, row in enumerate(reader, 2):
            sid = (row.get("site_id") or "").strip()
            seq = (row.get("seqid") or "").strip()
            ref = (row.get("ref") or "").strip().upper()
            alt = (row.get("alt") or "").strip().upper()
            try:
                pos = int(row.get("position", ""))
            except ValueError as exc:
                raise ValueError(f"line {line}: position must be integer") from exc
            if not sid or not seq or pos < 1:
                raise ValueError(f"line {line}: site_id/seqid required and position >= 1")
            if sid in seen:
                raise ValueError(f"line {line}: duplicate site_id {sid}")
            if (
                len(ref) != 1
                or len(alt) != 1
                or ref not in "ACGT"
                or alt not in "ACGT"
                or ref == alt
            ):
                raise ValueError(f"line {line}: ref and alt must be distinct A/C/G/T")
            seen.add(sid)
            out.append({"site_id": sid, "seqid": seq, "position": pos, "ref": ref, "alt": alt})
    return out


class _MaskError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _masked_intervals(
    bed: str | Path | None,
    homology_fasta: str | Path | None,
    reference_fasta: str | Path | None,
    min_identity: float,
    min_length: int,
    *,
    other: str,
) -> tuple[dict[str, list[tuple[int, int, str]]], dict[str, str]]:
    """Masked ``(start0, end0, source)`` intervals per contig, 0-based half-open."""
    intervals: dict[str, list[tuple[int, int, str]]] = {}
    versions: dict[str, str] = {}
    if bed is not None:
        try:
            for seqid, start, end in _read_bed(bed):
                intervals.setdefault(seqid, []).append((start, end, "bed"))
        except (OSError, ValueError) as exc:
            raise _MaskError("heteroplasmy.invalid_mask_bed", f"Could not read mask BED: {exc}") from exc
    if homology_fasta is not None:
        from organelleverse.transfer.transfer import detect_transfers_blast

        result = detect_transfers_blast(
            reference_fasta,
            homology_fasta,
            min_identity=min_identity,
            min_length=min_length,
            organelle=other,
        )
        if result.status == "failed":
            # Never fall back to an unmasked estimate: that would silently
            # report the very false positives masking is meant to remove.
            raise _MaskError(
                "heteroplasmy.homology_mask_failed",
                f"Homology search against {other} genome failed: {result.summary_text}",
            )
        for cand in result.metrics["candidates"]:
            lo, hi = sorted((int(cand["nuclear_start"]), int(cand["nuclear_end"])))
            intervals.setdefault(str(cand["nuclear_seqid"]), []).append((lo - 1, hi, "homology"))
        versions = {str(k): str(v) for k, v in dict(result.provenance.software_versions).items()}
    return intervals, versions


# Per-process cache of homology searches (bundled plastomes and nuclear genome).
_AUTO_CACHE: dict[tuple, tuple] = {}


def _auto_plastome_intervals(
    reference_fasta: str | Path, min_identity: float, min_length: int
) -> tuple[dict[str, list[tuple[int, int, str]]], dict[str, str], int]:
    """Regions of ``reference_fasta`` homologous to the package's bundled plastomes.

    Cached per process because many BAMs share one reference; the key covers the
    reference file's size and mtime and the bundled library location.
    """
    from organelleverse.annotation.plastome.db import default_plastome_reference_dir

    library = default_plastome_reference_dir()
    try:
        stat = Path(reference_fasta).stat()
    except OSError as exc:
        raise _MaskError(
            "heteroplasmy.homology_mask_failed", f"Could not read reference_fasta: {exc}"
        ) from exc
    key = (
        str(library),
        str(Path(reference_fasta).resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        min_identity,
        min_length,
    )
    hit = _AUTO_CACHE.get(key)
    if hit is None:
        paths = sorted(library.glob("*_chloroplast.gb"))
        if not paths:
            raise _MaskError(
                "heteroplasmy.no_bundled_plastomes", f"No bundled plastome references in {library}"
            )
        import tempfile

        from Bio import SeqIO

        with tempfile.TemporaryDirectory(prefix="ov_hetero_plastomes_") as tmp:
            fasta = Path(tmp) / "bundled_plastomes.fa"
            try:
                with fasta.open("w", encoding="utf-8", newline="\n") as out:
                    for path in paths:
                        record = SeqIO.read(path, "genbank")
                        out.write(f">{path.stem}\n{record.seq}\n")
            except (OSError, ValueError) as exc:
                raise _MaskError(
                    "heteroplasmy.no_bundled_plastomes",
                    f"Could not read the bundled plastomes: {exc}",
                ) from exc
            intervals, versions = _masked_intervals(
                None, fasta, reference_fasta, min_identity, min_length, other="plastid"
            )
        if len(_AUTO_CACHE) >= 16:
            _AUTO_CACHE.clear()
        hit = (intervals, versions, len(paths))
        _AUTO_CACHE[key] = hit
    cached_intervals, cached_versions, count = hit
    return {c: list(v) for c, v in cached_intervals.items()}, dict(cached_versions), count


_EXCESSIVE_MASK_FRACTION = 0.8


def _merged_bp(items: list[tuple[int, int, str]], limit: int) -> int:
    """Bases covered by the union of ``items``, clipped to ``[0, limit)``."""
    total = end = 0
    for start, stop, _ in sorted(items):
        lo, hi = max(start, end), min(stop, limit)
        if hi > lo:
            total += hi - lo
            end = hi
    return total


def _nuclear_intervals(
    reference_fasta: str | Path, nuclear_fasta: str | Path, min_identity: float, min_length: int
) -> tuple[dict[str, list[tuple[int, int, str]]], dict[str, str]]:
    """Regions of ``reference_fasta`` that have a copy in the nuclear genome (NUMT/NUPT).

    The organelle reference is the small mappy index and the nuclear sequences are
    streamed through it one record at a time, so memory stays small and the
    coordinates come back on the organelle reference directly. Do not reuse
    ``transfer.detect_transfers_blast`` here: it merges hits by nuclear overlap and
    keeps the min-max organelle hull, which can span unrelated stretches.
    A nuclear record named like a reference contig is the organelle itself and is skipped.
    """
    try:
        import mappy
    except ImportError as exc:
        raise _MaskError(
            "heteroplasmy.mappy_missing",
            "Nuclear homology masking needs mappy; install it with "
            "pip install 'organelleverse[align]' (no Windows wheel is published).",
        ) from exc
    try:
        ref_stat = Path(reference_fasta).stat()
        nuc_stat = Path(nuclear_fasta).stat()
    except OSError as exc:
        raise _MaskError(
            "heteroplasmy.homology_mask_failed",
            f"Could not read a FASTA for the nuclear homology search: {exc}",
        ) from exc
    key = (
        "nuclear",
        str(Path(reference_fasta).resolve()),
        ref_stat.st_size,
        ref_stat.st_mtime_ns,
        str(Path(nuclear_fasta).resolve()),
        nuc_stat.st_size,
        nuc_stat.st_mtime_ns,
        min_identity,
        min_length,
    )
    hit = _AUTO_CACHE.get(key)
    if hit is None:
        aligner = mappy.Aligner(str(reference_fasta), preset="asm20", best_n=10)
        if not aligner:
            raise _MaskError(
                "heteroplasmy.homology_mask_failed", "mappy could not index reference_fasta."
            )
        own = set(aligner.seq_names)
        raw: dict[str, list[tuple[int, int]]] = {}
        for name, seq, _ in mappy.fastx_read(str(nuclear_fasta)):
            if name in own:
                continue
            for aln in aligner.map(seq):
                if aln.blen >= min_length and aln.mlen / aln.blen * 100 >= min_identity:
                    raw.setdefault(aln.ctg, []).append((aln.r_st, aln.r_en))
        found: dict[str, list[tuple[int, int, str]]] = {}
        for contig, spans in raw.items():
            merged: list[list[int]] = []
            for start, stop in sorted(spans):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], stop)
                else:
                    merged.append([start, stop])
            found[contig] = [(a, b, "nuclear") for a, b in merged]
        if len(_AUTO_CACHE) >= 16:
            _AUTO_CACHE.clear()
        hit = (found, {"mappy": mappy.__version__})
        _AUTO_CACHE[key] = hit
    cached, versions = hit
    return {c: list(v) for c, v in cached.items()}, dict(versions)


def _read_bed(path: str | Path) -> list[tuple[str, int, int]]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"line {line_no}: BED needs seqid, start, end")
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(f"line {line_no}: BED start/end must be integers") from exc
            if start < 0 or end <= start:
                raise ValueError(f"line {line_no}: BED needs 0 <= start < end")
            out.append((fields[0], start, end))
    return out


def _mask_source(
    intervals: dict[str, list[tuple[int, int, str]]], contig: str, pos0: int
) -> str | None:
    for start, end, source in intervals.get(contig, ()):
        if start <= pos0 < end:
            return source
    return None


def _wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
