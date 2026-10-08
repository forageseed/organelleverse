"""Organelle DNA transfer detection and NUMT/NUPT evidence validation.

- detect_mtpt(): finds chloroplast-derived fragments in a mitochondrial genome
  by local alignment (LOSAT, then NCBI blastn; blastn task, word size 11), on
  both strands. An exact-k-mer-run scan is only the last resort when neither
  aligner is installed: it recovers just the near-identical copies (59% / 29%
  of the BLAST bases on real rice / Arabidopsis MTPTs).

``_detect_transfer`` (dual-strand exact k-mer runs, optional Rust acceleration
via ``accel.kmer_overlap``) remains for that fallback and for
``population.detect_numt``. 1-based inclusive coordinates; a k-mer fragment's
last position is always ``last_matching_kmer_start + k``.

Note: NUMT/NUPT should use ``detect_transfers_evidence`` so nuclear
chromosome coordinates, source genes and evidence layers flow into plotting.
The population suite still reuses ``_detect_transfer`` for its k-mer NUMT scan.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from .._bio import read_fasta
from .._sequtil import reverse_complement
from ..annotation.genbank import parse_genbank
from ..core.data import OrganelleData
from ..core.external import run_external
from ..core.frozen import FrozenMap, thaw_json
from ..core.genome import OrganelleGenome
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult, ResultScope

_SUITE = "transfer"
_OPERATION_VERSION = "1.0"

_SCOPE_BY_ORGANELLE: dict[str, ResultScope] = {
    "mitochondrion": "mitochondrion",
    "mito": "mitochondrion",
    "plastid": "plastid",
    "chloro": "plastid",
    "chloroplast": "plastid",
}


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance(op: str, parameters: Mapping[str, Any], *, method: str) -> ResultProvenance:
    """Build canonical provenance for one ``transfer`` operation.

    ``method`` is the external toolchain that produced the result (``blastn``,
    ``pysam``, ``minimap2``, ``blastn+pysam+minimap2``, ``organelleverse_kmer``)
    and is recorded as the actual backend.
    """
    package_version = _package_version()
    return ResultProvenance(
        operation_id=f"{_SUITE}.{op}",
        operation_version=_OPERATION_VERSION,
        package_version=package_version,
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=_sha256_json(dict(parameters)),
        actual_backend=method,
        attempted_backends=(method,),
        software_versions=FrozenMap({"organelleverse": package_version}),
    )


def _scope(organelle: str | None) -> ResultScope:
    if organelle is None:
        return "none"
    return _SCOPE_BY_ORGANELLE.get(str(organelle).casefold(), "none")


def _failed(op: str, *, scope: ResultScope, message: str, code: str) -> OrganelleResult:
    return OrganelleResult(
        operation_id=f"{_SUITE}.{op}",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="failed",
        summary_text=message,
        errors=(ErrorDetail(code=code, message=message),),
    )


def _result_metrics(result: OrganelleResult | Mapping[str, Any]) -> Mapping[str, Any]:
    """Return plain-JSON metrics from a canonical result or a raw mapping."""
    if isinstance(result, OrganelleResult):
        return cast(Mapping[str, Any], thaw_json(result.metrics))
    return dict(result)


def detect(
    nuclear_fasta: Any,
    organelle_fasta: Any,
    *,
    organelle_annotation: Any = None,
    nuclear_annotation: Any = None,
    bam: Any = None,
    hifi: Any = None,
    **kwargs: Any,
) -> OrganelleResult:
    """Run NUMT/NUPT detection from individually read input objects."""
    if bam is not None and "bam_path" not in kwargs:
        kwargs["bam_path"] = bam
    if hifi is not None and "hifi_reads" not in kwargs:
        kwargs["hifi_reads"] = hifi
    return detect_transfers_evidence(
        nuclear_fasta,
        organelle_fasta,
        organelle_annotation=organelle_annotation,
        nuclear_annotation=nuclear_annotation,
        **kwargs,
    )


def detect_mtpt(
    mito_fasta: str | Path,
    cp_fasta: str | Path,
    *,
    k: int = 31,
    min_len: int = 100,
    min_identity: float = 80.0,
) -> OrganelleResult:
    """Detect mitochondrial plastid-derived DNA transfers (MTPTs) by local alignment.

    The plastome is aligned against the mitochondrial genome with LOSAT (NCBI
    ``blastn`` when LOSAT is unavailable); an HSP of >= ``min_len`` alignment
    columns and >= ``min_identity`` percent identity on either strand is a
    fragment, and overlapping HSPs are merged on mitochondrial coordinates.
    ``k`` only matters for the exact-k-mer fallback used when neither aligner
    is installed (flag ``kmer_fallback``; it misses diverged copies). An
    organelle-separated assembly legitimately returns zero fragments
    (independently confirmed by blastn on a PMAT master ring) - corroborate a
    zero with ``detect_transfers_blast`` before reading it as absence.
    """
    return _detect_transfer_alignment(
        mito_fasta,
        cp_fasta,
        k=k,
        min_len=min_len,
        min_identity=min_identity,
        op="detect_mtpt",
        organelle="mitochondrion",
    )


def _build_stranded_kmer_index(source: str, k: int) -> set[str]:
    """Build a k-mer set covering both strands of ``source``.

    A transfer fragment can integrate as the reverse complement of the donor
    sequence, so the index carries every forward k-mer AND its reverse
    complement. k-mers containing N are skipped.
    """
    index: set[str] = set()
    for i in range(len(source) - k + 1):
        km = source[i : i + k]
        if "N" in km:
            continue
        index.add(km)
        index.add(reverse_complement(km))
    return index


def _scan_kmer_runs(target: str, index: set[str], k: int, min_len: int) -> list[dict]:
    """Scan ``target`` for runs of consecutive k-mers present in ``index``.

    Returns fragments as dicts with 1-based inclusive ``start``/``end``/``length``
    where ``end = last_matching_kmer_start + k`` (the last base covered by the
    last matching k-mer). The tail run uses the identical formula, so no run
    overshoots the target.
    """
    if len(target) < k:
        return []
    fragments: list[dict] = []
    run_start: int | None = None
    last_match_end = 0  # last base index (0-based) covered by the run
    n = len(target)
    for i in range(n - k + 1):
        km = target[i : i + k]
        if km in index:
            if run_start is None:
                run_start = i
            last_match_end = i + k  # 0-based exclusive end == 0-based last-base + 1
        elif run_start is not None:
            run_len = last_match_end - run_start
            if run_len >= min_len:
                fragments.append({"start": run_start + 1, "end": last_match_end, "length": run_len})
            run_start = None
    # tail
    if run_start is not None:
        run_len = last_match_end - run_start
        if run_len >= min_len:
            fragments.append({"start": run_start + 1, "end": last_match_end, "length": run_len})
    return fragments


def _detect_transfer(
    target_fasta: str | Path,
    source_fasta: str | Path,
    k: int,
    min_len: int,
    op: str,
    organelle: str,
) -> OrganelleResult:
    """Generic dual-strand k-mer-overlap transfer detection.

    Indexes both strands of the source, scans the target for k-mer runs, and
    reports fragments (1-based inclusive, never exceeding target length).
    """
    scope = _scope(organelle)
    target = "".join(s.upper() for _, s in read_fasta(Path(target_fasta)))
    source = "".join(s.upper() for _, s in read_fasta(Path(source_fasta)))
    if len(target) < k or len(source) < k:
        return _failed(
            op,
            scope=scope,
            message=f"{op} requires sequences >= {k} bp.",
            code="sequences_too_short",
        )

    # --- Rust fast path (optional) ---
    # The Rust kernel indexes both strands of the source too, so its fragments
    # are strand-agnostic (orientation is not distinguished). For the per-strand
    # breakdown we still compute the Python index here; when Rust is available
    # we use its faster scan for the fragment list.
    from .. import accel

    kmer_overlap = getattr(accel, "kmer_overlap", None)
    if accel.HAS_RUST and kmer_overlap is not None:
        raw_fragments = kmer_overlap(target, source, k, min_len)
        fragments = [{"start": s, "end": e, "length": e - s + 1} for s, e in raw_fragments]
        method = "organelleverse_kmer_rust"
        rust = True
    else:
        index = _build_stranded_kmer_index(source, k)
        fragments = _scan_kmer_runs(target, index, k, min_len)
        method = "organelleverse_kmer"
        rust = False

    # Per-strand classification: a fragment is "-" (reverse integration) when
    # its forward k-mers are NOT in the forward source index but ARE in the
    # reverse-complement index. Cheap to compute and biologically informative.
    forward_index: set[str] = set()
    rev_index: set[str] = set()
    for i in range(len(source) - k + 1):
        km = source[i : i + k]
        if "N" not in km:
            forward_index.add(km)
            rev_index.add(reverse_complement(km))
    direct_bp = reverse_bp = 0
    for f in fragments:
        s = f["start"] - 1
        head = target[s : s + k]
        if head in forward_index:
            f["strand"] = "+"
            direct_bp += f["length"]
        elif head in rev_index:
            f["strand"] = "-"
            reverse_bp += f["length"]
        else:
            f["strand"] = "?"
            direct_bp += f["length"]  # ambiguous; count as forward

    total_bp = sum(f["length"] for f in fragments)
    flags = []
    if fragments:
        flags.append("transfer_detected")
    if rust:
        flags.append("rust_accelerated")
    if reverse_bp:
        flags.append("reverse_strand_transfer")
    return OrganelleResult(
        operation_id=f"{_SUITE}.{op}",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="ok",
        summary_text=(
            f"{len(fragments)} {op} fragments, {total_bp} bp total "
            f"(+{direct_bp} bp forward, -{reverse_bp} bp reverse)" + (" (Rust)." if rust else ".")
        ),
        metrics=FrozenMap(
            {
                "fragment_count": len(fragments),
                "total_transferred_bp": total_bp,
                "direct_strand_bp": direct_bp,
                "reverse_strand_bp": reverse_bp,
                "fraction": round(total_bp / len(target), 6) if target else 0,
                "fragments": fragments,
            }
        ),
        findings=(
            Finding(code="fragments", metric="fragments", value=len(fragments)),
            Finding(code="total_bp", metric="total_bp", value=total_bp),
            Finding(code="reverse_strand_bp", metric="reverse_strand_bp", value=reverse_bp),
        ),
        flags=tuple(flags),
        artifacts=(),
        provenance=_provenance(op, {"k": k, "min_len": min_len}, method=method),
    )


def _detect_transfer_alignment(
    target_fasta: str | Path,
    source_fasta: str | Path,
    *,
    k: int,
    min_len: int,
    min_identity: float,
    op: str,
    organelle: str,
) -> OrganelleResult:
    """Alignment-based transfer detection with the exact k-mer scan as last resort."""
    from .transfer_core import WORD_SIZE, alignment_transfer

    scope = _scope(organelle)
    target_path, source_path = Path(target_fasta), Path(source_fasta)
    from .._losat import fasta_sequence_lengths

    target_lengths = fasta_sequence_lengths(target_path)
    target_len = sum(target_lengths.values())
    source_len = sum(fasta_sequence_lengths(source_path).values())
    if target_len < WORD_SIZE or source_len < WORD_SIZE:
        return _failed(
            op,
            scope=scope,
            message=f"{op} requires sequences >= {WORD_SIZE} bp.",
            code="sequences_too_short",
        )
    found = alignment_transfer(
        target_path,
        source_path,
        min_len=min_len,
        min_identity=min_identity,
        target_lengths=target_lengths,
    )
    parameters = {"k": k, "min_len": min_len, "min_identity": min_identity}
    if found is None:
        fallback = _detect_transfer(target_path, source_path, k, min_len, op, organelle)
        return fallback.model_copy(
            update={
                "flags": (*fallback.flags, "kmer_fallback"),
                "provenance": fallback.provenance.model_copy(
                    update={
                        "attempted_backends": (
                            "losat",
                            "ncbi_blastn",
                            fallback.provenance.actual_backend,
                        )
                    }
                ),
            }
        )

    fragments = found["fragments"]
    total_bp = sum(f["length"] for f in fragments)
    reverse_bp = sum(f["length"] for f in fragments if f["strand"] == "-")
    direct_bp = total_bp - reverse_bp
    mean_identity = (
        round(sum(f["identity"] * f["length"] for f in fragments) / total_bp, 2) if total_bp else None
    )
    flags = []
    if fragments:
        flags.append("transfer_detected")
    if reverse_bp:
        flags.append("reverse_strand_transfer")
    if found["backend"] != found["attempted"][0]:
        flags.append("aligner_fallback")
    backend = found["backend"]
    provenance = _provenance(op, parameters, method=backend).model_copy(
        update={"attempted_backends": tuple(found["attempted"])}
    )
    return OrganelleResult(
        operation_id=f"{_SUITE}.{op}",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="ok",
        summary_text=(
            f"{len(fragments)} {op} fragments, {total_bp} bp total "
            f"(+{direct_bp} bp forward, -{reverse_bp} bp reverse; "
            f">= {min_identity:g}% identity, {backend})."
        ),
        metrics=FrozenMap(
            {
                "fragment_count": len(fragments),
                "total_transferred_bp": total_bp,
                "direct_strand_bp": direct_bp,
                "reverse_strand_bp": reverse_bp,
                "fraction": round(total_bp / found["target_bp"], 6) if found["target_bp"] else 0,
                "mean_identity": mean_identity,
                "target_records": found["target_records"],
                "fragments": fragments,
            }
        ),
        findings=(
            Finding(code="fragments", metric="fragments", value=len(fragments)),
            Finding(code="total_bp", metric="total_bp", value=total_bp),
            Finding(code="reverse_strand_bp", metric="reverse_strand_bp", value=reverse_bp),
        ),
        flags=tuple(flags),
        artifacts=(),
        provenance=provenance,
    )


def write_fragments(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write k-mer transfer fragments from ``detect_mtpt()``."""
    metrics = _result_metrics(result)
    fragments = list(metrics.get("fragments", ()))
    path = _resolve_output_path(output, "transfer_fragments.tsv")
    lines = ["start\tend\tlength\tstrand\tseqid\tidentity"]
    for fragment in fragments:
        lines.append(
            f"{fragment.get('start', '')}\t{fragment.get('end', '')}\t"
            f"{fragment.get('length', '')}\t{fragment.get('strand', '')}\t"
            f"{fragment.get('seqid', '')}\t{fragment.get('identity', '')}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _resolve_output_path(output: str | Path, default_name: str) -> Path:
    path = Path(output)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name


def _input_path(value: Any, role: str = "sequence") -> Path | None:
    if value is None:
        return None
    if isinstance(value, OrganelleGenome):
        artifact = value.annotation if role == "annotation" else value.sequence
        if artifact is None:
            artifact = value.sequence if role == "annotation" else value.annotation
        return artifact.resolve() if artifact is not None else None
    if isinstance(value, OrganelleData):
        return _data_path(value, role)
    if isinstance(value, (list, tuple)):
        for item in value:
            resolved = _input_path(item, role)
            if resolved is not None:
                return resolved
        return None
    return Path(value)


def _data_path(data: OrganelleData, role: str) -> Path | None:
    keys_by_role = {
        "sequence": ("sequence", "fasta", "genome", "nuclear_fasta", "organelle_fasta"),
        "annotation": (
            "annotation",
            "genbank",
            "gbk",
            "gff",
            "gff3",
            "organelle_annotation",
            "nuclear_annotation",
        ),
        "bam": ("bam", "bam_path", "alignment"),
        "reads": ("hifi_reads", "hifi", "reads", "fastq", "fasta"),
    }
    return _data_artifact(data, *keys_by_role.get(role, (role,)))


def _data_artifact(data: OrganelleData, *keys: str) -> Path | None:
    """Resolve the first artifact registered under any of ``keys``."""
    for key in keys:
        artifact = data.artifacts.get(key)
        if artifact is not None:
            return artifact.resolve()
    return None


def _data_value(data: OrganelleData, key: str, default: Any) -> Any:
    """Read one scalar from the container payload, then its metadata."""
    for source in (data.payload, data.metadata):
        if key in source:
            return source[key]
    return default


def _first_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _first_path(value[0]) if value else None
    return Path(value)


def _resolve_transfer_data_inputs(
    nuclear_fasta: Any,
    organelle_fasta: Any,
    *,
    organelle: str,
    organelle_annotation: Any,
    organelle_genbank: Any,
    nuclear_annotation: Any,
    bam_path: Any,
    hifi_reads: Any,
) -> dict[str, Any]:
    if isinstance(nuclear_fasta, OrganelleData) and nuclear_fasta.modality == "transfer":
        data = nuclear_fasta
        organelle = str(_data_value(data, "organelle", organelle))
        nuclear_fasta = _data_artifact(data, "nuclear_fasta", "nuclear", "genome")
        organelle_fasta = organelle_fasta or _data_artifact(data, "organelle_fasta", "organelle")
        organelle_annotation = (
            organelle_annotation
            or organelle_genbank
            or _data_artifact(data, "organelle_annotation", "organelle_genbank")
        )
        nuclear_annotation = nuclear_annotation or _data_artifact(data, "nuclear_annotation")
        bam_path = bam_path or _data_artifact(data, "bam", "bam_path")
        hifi_reads = hifi_reads or _data_artifact(data, "hifi_reads", "hifi")

    if isinstance(organelle_fasta, OrganelleGenome):
        organelle = organelle_fasta.organelle
    return {
        "nuclear_fasta": _input_path(nuclear_fasta, "sequence"),
        "organelle_fasta": _input_path(organelle_fasta, "sequence"),
        "organelle": organelle,
        "organelle_annotation": _input_path(
            organelle_annotation or organelle_genbank, "annotation"
        ),
        "nuclear_annotation": _input_path(nuclear_annotation, "annotation"),
        "bam_path": _input_path(bam_path, "bam"),
        "hifi_reads": _input_path(hifi_reads, "reads"),
    }


# ===========================================================================
# Three-evidence NUMT/NUPT pipeline (BLASTN + read depth + long-read linkage)
#
# Implements the organelle→nuclear transfer validation workflow used in
# organelle-genome papers:
#   1. BLASTN: organelle query vs nuclear subject (identity > 80%, len > 100)
#   2. Read depth: NUMT/NUPT regions carry ~nuclear + organelle depth
#   3. Long-read linkage: HiFi reads spanning the insert-nuclear junction
# ===========================================================================


@dataclass(frozen=True)
class TransferCandidate:
    """One BLASTN-supported organelle→nuclear transfer candidate.

    Coordinates are 1-based inclusive. ``nuclear_*`` describe where the
    fragment sits in the nuclear assembly; ``organelle_*`` describe the
    homologous region of the organelle genome. ``organelle_gene`` is the
    organelle gene overlapping that region (filled by gene annotation).
    """

    nuclear_seqid: str
    nuclear_start: int
    nuclear_end: int
    organelle_seqid: str
    organelle_start: int
    organelle_end: int
    identity: float
    length: int
    evalue: float
    bitscore: float
    strand: str = "+"
    organelle_gene: str = ""
    nuclear_gene: str = ""
    # evidence-layer annotations (filled by depth/longread validation)
    observed_depth: float | None = None
    depth_supported: bool | None = None
    linking_reads: int | None = None
    longread_linked: bool | None = None

    @property
    def evidence_level(self) -> int:
        """0 = BLAST only, +1 depth, +1 long-read linkage."""
        return int(bool(self.depth_supported)) + int(bool(self.longread_linked))

    def as_dict(self) -> dict:
        return {
            "nuclear_seqid": self.nuclear_seqid,
            "nuclear_start": self.nuclear_start,
            "nuclear_end": self.nuclear_end,
            "organelle_seqid": self.organelle_seqid,
            "organelle_start": self.organelle_start,
            "organelle_end": self.organelle_end,
            "identity": round(self.identity, 4),
            "length": self.length,
            "evalue": self.evalue,
            "bitscore": self.bitscore,
            "strand": self.strand,
            "organelle_gene": self.organelle_gene,
            "nuclear_gene": self.nuclear_gene,
            "observed_depth": self.observed_depth,
            "depth_supported": self.depth_supported,
            "linking_reads": self.linking_reads,
            "longread_linked": self.longread_linked,
            "evidence_level": self.evidence_level,
        }


def annotate_organelle_genes(
    candidates: list[TransferCandidate],
    organelle_annotation: str | Path | None,
) -> list[TransferCandidate]:
    """Annotate each candidate with the organelle gene it overlaps.

    Parses CDS/tRNA/rRNA/gene features from the organelle annotation
    (GenBank ``.gb``/``.gbk`` or GFF3 ``.gff``/``.gff3``) and assigns
    ``organelle_gene`` to the gene whose coordinates overlap the candidate's
    ``organelle_start..organelle_end`` range. Candidates with no overlap keep
    an empty gene string. When ``organelle_annotation`` is None, candidates are
    returned unchanged.
    """
    if organelle_annotation is None:
        return list(candidates)
    intervals = _load_gene_intervals(organelle_annotation)
    if not intervals:
        return list(candidates)
    out: list[TransferCandidate] = []
    for c in candidates:
        gene = _overlap_gene(intervals, c.organelle_start, c.organelle_end)
        out.append(_replace(c, organelle_gene=gene))
    return out


def annotate_nuclear_locus(
    candidates: list[TransferCandidate],
    nuclear_annotation: str | Path | None,
) -> list[TransferCandidate]:
    """Annotate each candidate with the nuclear gene it overlaps.

    Parses gene features from the nuclear annotation (GFF3 or GenBank) and
    assigns ``nuclear_gene`` to the gene whose coordinates overlap the
    candidate's ``nuclear_start..nuclear_end`` range on the same chromosome.
    Useful for reporting whether a NUMT/NUPT landed inside, upstream of, or
    downstream of a nuclear gene. When ``nuclear_annotation`` is None,
    candidates are returned unchanged.
    """
    if nuclear_annotation is None:
        return list(candidates)
    intervals_by_seq = _load_gene_intervals_by_seqid(nuclear_annotation)
    if not intervals_by_seq:
        return list(candidates)
    out: list[TransferCandidate] = []
    for c in candidates:
        intervals = intervals_by_seq.get(c.nuclear_seqid, [])
        gene = _overlap_gene(intervals, c.nuclear_start, c.nuclear_end)
        out.append(_replace(c, nuclear_gene=gene))
    return out


def _load_gene_intervals(
    path: str | Path,
) -> list[tuple[str, int, int]]:
    """Load (gene, start, end) intervals (1-based inclusive) from GenBank or GFF3.

    Auto-detects by file suffix/extension. Returns a flat list (single-genome
    organelle use case); use ``_load_gene_intervals_by_seqid`` for multi-chromosome
    nuclear annotations.
    """
    by_seq = _load_gene_intervals_by_seqid(path)
    flat: list[tuple[str, int, int]] = []
    for intervals in by_seq.values():
        flat.extend(intervals)
    return flat


def _load_gene_intervals_by_seqid(
    path: str | Path,
) -> dict[str, list[tuple[str, int, int]]]:
    """Load gene intervals grouped by sequence id from GenBank or GFF3.

    Returns ``{seqid: [(gene, start, end), ...]}`` with 1-based inclusive
    coordinates. Recognized feature types: CDS, tRNA, rRNA, gene. Gene names
    come from the ``gene`` qualifier (GenBank) or ``Name``/``gene``/``ID``
    attributes (GFF3).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    out: dict[str, list[tuple[str, int, int]]] = {}
    try:
        out = _load_gff_intervals(p) if suffix in (".gff", ".gff3") else _load_genbank_intervals(p)
    except Exception:
        return {}
    return out


def _load_genbank_intervals(path: Path) -> dict[str, list[tuple[str, int, int]]]:
    """Parse gene intervals from a GenBank file, grouped by seqid."""

    out: dict[str, list[tuple[str, int, int]]] = {}
    for record in parse_genbank(path).records:
        for feature in record.features:
            if feature.type.casefold() not in ("cds", "trna", "rrna", "gene"):
                continue
            values = feature.qualifier_values("gene")
            if not values or not values[0]:
                continue
            bounding_start = min(part.start for part in feature.parts)
            bounding_end = max(part.end for part in feature.parts)
            out.setdefault(record.seqid, []).append((values[0], bounding_start + 1, bounding_end))
    return out


def _load_gff_intervals(path: Path) -> dict[str, list[tuple[str, int, int]]]:
    """Parse gene intervals from a GFF3 file, grouped by seqid.

    Uses BCBio.GFF when available; falls back to a minimal line parser. Keeps
    CDS/exon/gene/mRNA records and derives the gene name from the ``Name``,
    ``gene``, or ``ID`` attribute.
    """
    kept = {"cds", "exon", "gene", "mrna", "trna", "rrna"}
    try:
        from BCBio import GFF

        out: dict[str, list[tuple[str, int, int]]] = {}
        with open(path) as handle:
            for rec in GFF.parse(handle):
                for f in rec.features:
                    if f.type.lower() not in kept:
                        continue
                    gene = (
                        f.qualifiers.get("Name")
                        or f.qualifiers.get("gene")
                        or f.qualifiers.get("ID")
                        or [""]
                    )[0]
                    if not gene:
                        continue
                    out.setdefault(rec.id, []).append(
                        (gene, int(f.location.start) + 1, int(f.location.end))
                    )
        return out
    except ImportError:
        # minimal GFF3 line parser fallback
        out = {}
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9 or cols[2].lower() not in kept:
                continue
            seqid = cols[0]
            start, end = int(cols[3]), int(cols[4])
            attrs = dict(kv.split("=", 1) for kv in cols[8].split(";") if "=" in kv)
            gene = attrs.get("Name") or attrs.get("gene") or attrs.get("ID") or ""
            if gene:
                out.setdefault(seqid, []).append((gene, start, end))
        return out


def _overlap_gene(
    intervals: list[tuple[str, int, int]],
    start: int,
    end: int,
) -> str:
    """Return the gene name of the interval with the largest overlap."""
    best_gene = ""
    best_overlap = 0
    for gene, g_start, g_end in intervals:
        ov = max(0, min(end, g_end) - max(start, g_start) + 1)
        if ov > best_overlap:
            best_overlap = ov
            best_gene = gene
    return best_gene


def _replace(c: TransferCandidate, **changes) -> TransferCandidate:
    """dataclass.replace shim (avoids importing dataclasses.replace at call site)."""
    from dataclasses import replace as _r

    return _r(c, **changes)


def detect_transfers_blast(
    nuclear_fasta: str | Path,
    organelle_fasta: str | Path,
    *,
    min_identity: float = 80.0,
    min_length: int = 100,
    evalue: float = 1e-5,
    organelle: str = "mitochondrion",
    blastn_path: str | Path | None = None,
    makeblastdb_path: str | Path | None = None,
) -> OrganelleResult:
    """Identify NUMT/NUPT candidates via BLASTN (layer 1).

    Builds a BLAST database from the nuclear genome and queries it with the
    organelle genome. Reports nuclear loci with identity > ``min_identity`` and
    alignment length > ``min_length`` (paper defaults: 80%, 100 bp).
    """
    scope = _scope(organelle)
    result = _detect_transfers_blast_core(
        nuclear_fasta,
        organelle_fasta,
        min_identity=min_identity,
        min_length=min_length,
        evalue=evalue,
        blastn_path=blastn_path,
        makeblastdb_path=makeblastdb_path,
    )
    if isinstance(result, str):
        code = "blastn_missing" if result == "blastn_missing" else "blast_failed"
        return _failed(
            "detect_transfers_blast",
            scope=scope,
            message=result,
            code=code,
        )
    candidates = result
    total_bp = sum(c.length for c in candidates)
    return OrganelleResult(
        operation_id=f"{_SUITE}.detect_transfers_blast",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="ok",
        summary_text=(
            f"BLASTN: {len(candidates)} transfer candidate(s), "
            f"{total_bp} bp (identity>{min_identity}%, len>{min_length})."
        ),
        metrics=FrozenMap(
            {
                "candidate_count": len(candidates),
                "total_bp": total_bp,
                "min_identity": min_identity,
                "min_length": min_length,
                "candidates": [c.as_dict() for c in candidates],
            }
        ),
        findings=(
            Finding(code="candidate_count", metric="candidate_count", value=len(candidates)),
            Finding(code="total_bp", metric="total_bp", value=total_bp, unit="bp"),
        ),
        flags=("transfer_detected",) if candidates else (),
        artifacts=(),
        provenance=_provenance(
            "detect_transfers_blast",
            {
                "min_identity": min_identity,
                "min_length": min_length,
                "evalue": evalue,
            },
            method="blastn",
        ),
    )


def _parse_transfer_blast_line(line: str) -> TransferCandidate | None:
    f = line.rstrip("\n").split("\t")
    if len(f) < 12:
        return None
    qstart, qend = int(f[6]), int(f[7])
    sstart, send = int(f[8]), int(f[9])
    strand = "-" if sstart > send else "+"
    return TransferCandidate(
        nuclear_seqid=f[1],
        nuclear_start=min(sstart, send),
        nuclear_end=max(sstart, send),
        organelle_seqid=f[0],
        organelle_start=min(qstart, qend),
        organelle_end=max(qstart, qend),
        identity=float(f[2]),
        length=int(f[3]),
        evalue=float(f[10]),
        bitscore=float(f[11]),
        strand=strand,
    )


def _collinear(a: TransferCandidate, b: TransferCandidate) -> bool:
    """True when ``b`` continues ``a``: overlapping nuclear interval and the same
    organelle record, strand and (overlapping or touching) organelle interval."""
    return (
        a.nuclear_seqid == b.nuclear_seqid
        and b.nuclear_start <= a.nuclear_end
        and a.nuclear_start <= b.nuclear_end
        and a.organelle_seqid == b.organelle_seqid
        and a.strand == b.strand
        and b.organelle_start <= a.organelle_end + 1
        and a.organelle_start <= b.organelle_end + 1
    )


def _merge_overlapping(cands: list[TransferCandidate]) -> list[TransferCandidate]:
    """Merge hits that are one transfer reported as several overlapping HSPs.

    Overlap on the nuclear chromosome alone is not enough: two hits from
    different organelle loci (or from opposite strands, e.g. the two copies of
    a plastid inverted repeat) can land on the same nuclear bases. Merging
    those would make the organelle side the min..max hull between unrelated
    regions (347 kb for a 286 bp nuclear locus on real rice data) and mislabel
    the strand, so they stay separate candidates. Merged identity is the
    span-weighted mean, not the maximum.
    """
    if len(cands) < 2:
        return list(cands)
    entries: list[list] = []  # [candidate, identity * span, span]
    for c in sorted(cands, key=lambda c: (c.nuclear_seqid, c.nuclear_start, c.nuclear_end)):
        span = c.nuclear_end - c.nuclear_start + 1
        joined = False
        for entry in reversed(entries):
            if entry[0].nuclear_seqid != c.nuclear_seqid:
                break
            if _collinear(entry[0], c):
                entry[0] = _union(entry[0], c)
                entry[1] += c.identity * span
                entry[2] += span
                joined = True
                break
        if not joined:
            entries.append([c, c.identity * span, span])
    # A hit can bridge two earlier entries; repeat until nothing merges.
    changed = len(entries) > 1
    while changed:
        changed = False
        for i, left in enumerate(entries):
            for j in range(i + 1, len(entries)):
                right = entries[j]
                if _collinear(left[0], right[0]):
                    left[0] = _union(left[0], right[0])
                    left[1] += right[1]
                    left[2] += right[2]
                    del entries[j]
                    changed = True
                    break
            if changed:
                break
    return [_replace(cand, identity=round(ident / span, 4)) for cand, ident, span in entries]


def _union(a: TransferCandidate, b: TransferCandidate) -> TransferCandidate:
    start = min(a.nuclear_start, b.nuclear_start)
    end = max(a.nuclear_end, b.nuclear_end)
    return _replace(
        a,
        nuclear_start=start,
        nuclear_end=end,
        organelle_start=min(a.organelle_start, b.organelle_start),
        organelle_end=max(a.organelle_end, b.organelle_end),
        length=end - start + 1,
        evalue=min(a.evalue, b.evalue),
        bitscore=max(a.bitscore, b.bitscore),
    )


# ---------------------------------------------------------------------------
# Layer 2: read-depth validation (NUMT/NUPT regions are ~2x nuclear depth)
# ---------------------------------------------------------------------------


def validate_transfers_depth(
    candidates: list[Mapping[str, str | int | float | None]],
    bam_path: str | Path,
    nuclear_mean_depth: float | None = None,
    organelle_mean_depth: float | None = None,
    *,
    flank: int = 0,
    tolerance: float = 0.5,
    organelle: str = "mitochondrion",
) -> OrganelleResult:
    """Validate transfer candidates by read depth (layer 2).

    Measures the mean depth over each candidate with pysam (agrees with
    ``samtools bedcov``, Pearson 0.9999 on real data) and sets ``depth_supported``:

    * ``organelle_mean_depth`` not given (default; the BAM's reference contains the
      organelle genome, so organelle reads settle on it): a real nuclear insert is
      covered by its own nuclear reads, so it is supported when its depth is at
      least ``nuclear_mean_depth * (1 - tolerance)``. A higher depth only means
      organelle reads also multi-map onto it and is not held against it. At low
      coverage a short candidate can fall under the bound by sampling noise alone
      (real rice data at 4.6x: 200 of 1,532 unsupported, 195 of them long-read
      linked, none of the 74 inserts >= 2 kb).
    * ``organelle_mean_depth`` given (the reference has NO organelle genome, so
      organelle reads pile onto the inserts): the two-sided formula, supported when
      the depth is within ``tolerance`` of ``nuclear_mean_depth +
      organelle_mean_depth``. Only long, near-identical inserts carry the organelle
      depth: on real rice data it supports 32 of the 74 inserts >= 2 kb and 129 of
      1,532 candidates overall.

    A depth check cannot expose an organelle-derived contaminant that is present
    in the reference with the organelle genome (its reads split between the two
    copies, giving it about half the organelle depth); the long-read layer with
    ``organelle_fasta`` does.
    """
    scope = _scope(organelle)
    cands = [_to_candidate(c) for c in candidates]
    if not cands:
        return _failed(
            "validate_transfers_depth",
            scope=scope,
            message="no candidates to validate.",
            code="no_candidates",
        )
    two_sided = organelle_mean_depth is not None  # before the core overwrites it
    try:
        annotated, nuclear_mean_depth, organelle_mean_depth, expected = _validate_depth_core(
            cands,
            bam_path,
            nuclear_mean_depth,
            organelle_mean_depth,
            flank=flank,
            tolerance=tolerance,
        )
    except ImportError:
        return _failed(
            "validate_transfers_depth",
            scope=scope,
            message="pysam not installed; install with pip install pysam.",
            code="pysam_missing",
        )
    n_supported = sum(1 for c in annotated if c.depth_supported)
    return OrganelleResult(
        operation_id=f"{_SUITE}.validate_transfers_depth",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="ok",
        summary_text=(
            f"Depth: {n_supported}/{len(annotated)} candidates supported "
            + (
                f"(expected ~{expected:.1f}x, +/-{tolerance:.0%})."
                if two_sided
                else f"(>= {expected * (1 - tolerance):.1f}x)."
            )
        ),
        metrics=FrozenMap(
            {
                "nuclear_mean_depth": round(nuclear_mean_depth, 2),
                "organelle_mean_depth": round(organelle_mean_depth or 0, 2),
                "expected_depth": round(expected, 2),
                "criterion": "two_sided" if two_sided else "lower_bound",
                "minimum_depth": round(expected * (1 - tolerance), 2),
                "candidate_count": len(annotated),
                "depth_supported": n_supported,
                "candidates": [c.as_dict() for c in annotated],
            }
        ),
        findings=(
            Finding(code="candidate_count", metric="candidate_count", value=len(annotated)),
            Finding(code="depth_supported", metric="depth_supported", value=n_supported),
            Finding(code="expected_depth", metric="expected_depth", value=round(expected, 2)),
        ),
        flags=("depth_evidence",) if n_supported else (),
        artifacts=(),
        provenance=_provenance(
            "validate_transfers_depth",
            {"flank": flank, "tolerance": tolerance},
            method="pysam",
        ),
    )


def _validate_depth_core(
    candidates: list[TransferCandidate],
    bam_path: str | Path,
    nuclear_mean_depth: float | None,
    organelle_mean_depth: float | None,
    *,
    flank: int,
    tolerance: float,
) -> tuple[list[TransferCandidate], float, float, float]:
    """Depth core: returns (annotated candidates, nuclear_depth, organelle_depth, expected).

    Raises ImportError if pysam is unavailable. This is the chainable core used
    by the pipeline; candidates flow in and out as ``TransferCandidate`` objects.
    """
    import pysam

    if nuclear_mean_depth is None:
        nuclear_mean_depth = _bam_mean_depth(bam_path)
    two_sided = organelle_mean_depth is not None  # organelle depth supplied -> legacy formula
    organelle_mean_depth = organelle_mean_depth or 0.0
    expected = nuclear_mean_depth + organelle_mean_depth
    lo = expected * (1 - tolerance)
    hi = expected * (1 + tolerance)

    out: list[TransferCandidate] = []
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for c in candidates:
            start = max(1, c.nuclear_start - flank)
            end = c.nuclear_end + flank
            depth = _region_depth(bam, c.nuclear_seqid, start, end)
            if not expected:
                supported = depth > 0
            elif two_sided:
                supported = lo <= depth <= hi
            else:
                supported = depth >= lo
            out.append(_replace(c, observed_depth=round(depth, 2), depth_supported=supported))
    return out, nuclear_mean_depth, organelle_mean_depth, expected


def _bam_mean_depth(bam_path: str | Path) -> float:
    """Cheap genome-wide mean depth via a sparse sample of positions."""
    import pysam

    total = 0
    n = 0
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for ref in bam.references:
            length = bam.get_reference_length(ref)
            if length == 0:
                continue
            # sample up to 1000 evenly-spaced columns
            step = max(1, length // 1000)
            for pos in range(0, length, step):
                total += bam.count(ref, pos, pos + 1)
                n += 1
                if n >= 5000:
                    break
            if n >= 5000:
                break
    return total / n if n else 0.0


def _region_depth(bam, seqid: str, start: int, end: int) -> float:
    """Mean depth over [start, end] (1-based inclusive)."""
    length = end - start + 1
    if length <= 0:
        return 0.0
    try:
        col_iter = bam.pileup(seqid, start - 1, end)
        total = sum(col.n for col in col_iter if start - 1 <= col.reference_pos < end)
    except (ValueError, OSError):
        total = 0
    return total / length


# ---------------------------------------------------------------------------
# Layer 3: long-read linkage validation (HiFi reads span the junction)
# ---------------------------------------------------------------------------


def validate_transfers_longread(
    nuclear_fasta: str | Path,
    candidates: list[Mapping[str, str | int | float | None]],
    hifi_reads: str | Path,
    *,
    flank: int = 20000,
    organelle: str = "mitochondrion",
    minimap2_path: str | Path | None = None,
    organelle_fasta: str | Path | None = None,
    min_anchor: int = 200,
    min_mapq: int = 20,
) -> OrganelleResult:
    """Validate transfer candidates by long-read linkage (layer 3).

    The ±``flank`` bp around every candidate are merged into windows and the
    HiFi reads are aligned to them with ONE minimap2 pass. ``flank`` must be
    longer than the reads (default 20 kb for HiFi, 10-25 kb): a read that
    extends past its window is truncated there and then loses to its full-length
    alignment on the organelle genome, so nuclear reads are discarded (real rice
    data, 100k HiFi reads: with 2 kb, 41 of 1,518 linked candidates were lost;
    with 20 kb the linked/not-linked call matched pysam on all 1,532). A
    read links a candidate when its primary alignment (MAPQ >= ``min_mapq``)
    covers at least ``min_anchor`` bp on each side of the candidate's left or
    right boundary - physical evidence that the insert is embedded in the
    nuclear locus rather than assembled from organelle reads.

    Pass ``organelle_fasta``: its records are aligned together with the windows
    and a read whose best alignment is on the organelle genome is not counted.
    Without it, organelle reads that are homologous to the insert align across
    boundaries that lie *inside* a larger insert and inflate the count (real rice
    data: 8-376 "linking" reads where 2-10 nuclear reads exist); the result is
    then flagged ``organelle_reads_not_excluded``.
    """
    scope = _scope(organelle)
    exe = str(minimap2_path) if minimap2_path else shutil.which("minimap2")
    if not exe:
        return _failed(
            "validate_transfers_longread",
            scope=scope,
            message="minimap2 not found; install minimap2.",
            code="minimap2_missing",
        )
    cands = [_to_candidate(c) for c in candidates]
    if not cands:
        return _failed(
            "validate_transfers_longread",
            scope=scope,
            message="no candidates to validate.",
            code="no_candidates",
        )

    annotated, stats = _validate_longread_core(
        cands,
        nuclear_fasta,
        hifi_reads,
        exe,
        flank=flank,
        organelle_fasta=organelle_fasta,
        min_anchor=min_anchor,
        min_mapq=min_mapq,
    )
    n_supported = sum(1 for c in annotated if c.longread_linked)
    return OrganelleResult(
        operation_id=f"{_SUITE}.validate_transfers_longread",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="ok",
        summary_text=(
            f"Long-read: {n_supported}/{len(annotated)} candidates have "
            f"junction-spanning HiFi reads."
        ),
        metrics=FrozenMap(
            {
                "candidate_count": len(annotated),
                "longread_linked": n_supported,
                "flank": flank,
                "min_anchor": min_anchor,
                "min_mapq": min_mapq,
                "organelle_reads_excluded": stats["organelle_reads_excluded"],
                "candidates": [c.as_dict() for c in annotated],
            }
        ),
        findings=(
            Finding(code="candidate_count", metric="candidate_count", value=len(annotated)),
            Finding(code="longread_linked", metric="longread_linked", value=n_supported),
        ),
        flags=(
            *(("longread_evidence",) if n_supported else ()),
            *(() if organelle_fasta is not None else ("organelle_reads_not_excluded",)),
        ),
        artifacts=(),
        provenance=_provenance(
            "validate_transfers_longread",
            {"flank": flank, "min_anchor": min_anchor, "min_mapq": min_mapq},
            method="minimap2",
        ),
    )


def _validate_longread_core(
    candidates: list[TransferCandidate],
    nuclear_fasta: str | Path,
    hifi_reads: str | Path,
    minimap2_exe: str,
    *,
    flank: int,
    organelle_fasta: str | Path | None = None,
    min_anchor: int = 200,
    min_mapq: int = 20,
) -> tuple[list[TransferCandidate], dict[str, Any]]:
    """Long-read core: returns (annotated candidates, stats); chainable.

    One minimap2 pass over all reads against the merged candidate windows (plus
    the organelle genome when given, so organelle reads settle there). A read
    links a candidate when its primary alignment, MAPQ >= ``min_mapq``, reaches
    ``min_anchor`` bp past the candidate's left or right boundary on both sides.
    """
    from bisect import bisect_left, bisect_right

    from Bio import SeqIO

    by_seqid: dict[str, list[int]] = {}
    for i, c in enumerate(candidates):
        by_seqid.setdefault(c.nuclear_seqid, []).append(i)

    linked: dict[int, set[str]] = {}
    organelle_reads: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="ov_longread_") as tmp:
        ref, paf = Path(tmp) / "ref.fa", Path(tmp) / "hits.paf"
        regions: list[tuple[str, int]] = []  # (seqid, window start, 0-based) by region number
        with open(ref, "w", encoding="utf-8") as out:
            for record in SeqIO.parse(str(nuclear_fasta), "fasta"):  # one chromosome at a time
                indexes = by_seqid.get(record.id)
                if not indexes:
                    continue
                sequence = str(record.seq)
                windows = sorted(
                    (
                        max(0, candidates[i].nuclear_start - 1 - flank),
                        min(len(sequence), candidates[i].nuclear_end + flank),
                    )
                    for i in indexes
                )
                merged: list[list[int]] = []
                for lo, hi in windows:
                    if merged and lo <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], hi)
                    else:
                        merged.append([lo, hi])
                for lo, hi in merged:
                    if hi - lo < 100:
                        continue
                    out.write(f">R{len(regions)}\n{sequence[lo:hi]}\n")
                    regions.append((record.id, lo))
            if organelle_fasta is not None:
                for record in SeqIO.parse(str(organelle_fasta), "fasta"):
                    out.write(f">ORG:{record.id}\n{record.seq}\n")
        if regions:
            run_external(
                [
                    minimap2_exe,
                    "-x",
                    "map-hifi",
                    "-c",
                    "--secondary=no",
                    "-t",
                    "4",
                    "-o",
                    str(paf),
                    str(ref),
                    str(hifi_reads),
                ],
                timeout=7200,
                tool="minimap2",
            )
            # boundary tables: a boundary b is spanned by an alignment [a0, a1) when
            # a0 + min_anchor <= b <= a1 - min_anchor
            lefts = {
                sid: sorted((candidates[i].nuclear_start - 1, i) for i in idxs)
                for sid, idxs in by_seqid.items()
            }
            rights = {
                sid: sorted((candidates[i].nuclear_end, i) for i in idxs)
                for sid, idxs in by_seqid.items()
            }
            with open(paf, encoding="utf-8") as handle:
                for line in handle:
                    f = line.rstrip("\n").split("\t")
                    if len(f) < 12:
                        continue
                    if next((t[5:] for t in f[12:] if t.startswith("tp:A:")), "P") != "P":
                        continue
                    if f[5].startswith("ORG:"):
                        organelle_reads.add(f[0])
                        continue
                    if int(f[11]) < min_mapq:
                        continue
                    seqid, offset = regions[int(f[5][1:])]
                    lo = offset + int(f[7]) + min_anchor
                    hi = offset + int(f[8]) - min_anchor
                    if lo > hi:
                        continue
                    for table in (lefts[seqid], rights[seqid]):
                        first = bisect_left(table, (lo, -1))
                        last = bisect_right(table, (hi, len(candidates)))
                        for _, index in table[first:last]:
                            linked.setdefault(index, set()).add(f[0])

    out_candidates = [
        _replace(c, linking_reads=len(linked.get(i, ())), longread_linked=i in linked)
        for i, c in enumerate(candidates)
    ]
    stats = {"organelle_reads_excluded": len(organelle_reads) if organelle_fasta is not None else None}
    return out_candidates, stats


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


def detect_transfers_evidence(
    nuclear_fasta: str | Path | OrganelleData,
    organelle_fasta: str | Path | OrganelleGenome | OrganelleData | None = None,
    *,
    organelle: str = "mitochondrion",
    organelle_annotation: str | Path | None = None,
    organelle_genbank: str | Path | None = None,
    nuclear_annotation: str | Path | None = None,
    bam_path: str | Path | None = None,
    hifi_reads: str | Path | None = None,
    min_identity: float = 80.0,
    min_length: int = 100,
    nuclear_mean_depth: float | None = None,
    organelle_mean_depth: float | None = None,
    depth_tolerance: float = 0.5,
    longread_flank: int = 20000,
    blastn_path: str | Path | None = None,
    makeblastdb_path: str | Path | None = None,
    minimap2_path: str | Path | None = None,
) -> OrganelleResult:
    """Three-evidence NUMT/NUPT pipeline (BLASTN → gene map → depth → long-read).

    A clean chained pipeline. Candidates flow as ``TransferCandidate`` objects
    through each stage; the wrapper returns the final ``OrganelleResult``:

        layer 1  BLASTN candidates          → list[TransferCandidate]
        gene     annotate organelle gene    → list[TransferCandidate]
        locus    annotate nuclear locus     → list[TransferCandidate]
        layer 2  read-depth validation      → list[TransferCandidate]   (if bam)
        layer 3  long-read linkage          → list[TransferCandidate]   (if hifi)

    Each candidate carries an ``evidence_level`` (0 = BLAST only,
    +1 depth, +1 long-read). When ``organelle_annotation`` (GenBank or GFF3)
    is given, each candidate reports the ``organelle_gene`` it overlaps;
    when ``nuclear_annotation`` (GFF3 or GenBank) is given, it also reports the
    ``nuclear_gene`` it overlaps on the nuclear chromosome.
    """
    resolved_inputs = _resolve_transfer_data_inputs(
        nuclear_fasta,
        organelle_fasta,
        organelle=organelle,
        organelle_annotation=organelle_annotation,
        organelle_genbank=organelle_genbank,
        nuclear_annotation=nuclear_annotation,
        bam_path=bam_path,
        hifi_reads=hifi_reads,
    )
    nuclear_fasta = resolved_inputs["nuclear_fasta"]
    organelle_fasta = resolved_inputs["organelle_fasta"]
    organelle = resolved_inputs["organelle"]
    organelle_annotation = resolved_inputs["organelle_annotation"]
    nuclear_annotation = resolved_inputs["nuclear_annotation"]
    bam_path = resolved_inputs["bam_path"]
    hifi_reads = resolved_inputs["hifi_reads"]
    scope = _scope(organelle)
    parameters = {
        "min_identity": min_identity,
        "min_length": min_length,
        "evalue": 1e-5,
        "depth_tolerance": depth_tolerance,
        "longread_flank": longread_flank,
        "nuclear_mean_depth": nuclear_mean_depth,
        "organelle_mean_depth": organelle_mean_depth,
    }
    if nuclear_fasta is None or organelle_fasta is None:
        return _failed(
            "detect_transfers_evidence",
            scope=scope,
            message="transfer detection needs nuclear_fasta and organelle_fasta.",
            code="missing_transfer_inputs",
        )

    # --- layer 1: BLASTN ---
    candidates = _detect_transfers_blast_core(
        nuclear_fasta,
        organelle_fasta,
        min_identity=min_identity,
        min_length=min_length,
        evalue=1e-5,
        blastn_path=blastn_path,
        makeblastdb_path=makeblastdb_path,
    )
    if isinstance(candidates, str):
        # a string return signals a tool-missing/error code
        return _failed(
            "detect_transfers_evidence",
            scope=scope,
            message=candidates,
            code=(
                candidates.split("_")[0] + "_missing" if "missing" in candidates else "blast_failed"
            ),
        )
    if not candidates:
        return OrganelleResult(
            operation_id=f"{_SUITE}.detect_transfers_evidence",
            operation_version=_OPERATION_VERSION,
            scope=scope,
            status="ok",
            summary_text="Evidence pipeline: 0 candidates.",
            metrics=FrozenMap(
                {
                    "candidate_count": 0,
                    "depth_supported": 0,
                    "longread_linked": 0,
                    "max_evidence_level": 0,
                    "organelle": scope,
                    "candidates": [],
                }
            ),
            findings=(Finding(code="candidate_count", metric="candidate_count", value=0),),
            flags=(),
            artifacts=(),
            provenance=_provenance("detect_transfers_evidence", parameters, method="blastn"),
        )

    # --- gene annotation (both sides) ---
    # organelle side: which organelle gene does the insert come from?
    org_ann = organelle_annotation if organelle_annotation else organelle_genbank
    candidates = annotate_organelle_genes(candidates, org_ann)
    # nuclear side: which nuclear gene does the insert land in/near?
    candidates = annotate_nuclear_locus(candidates, nuclear_annotation)

    # --- layer 2: read depth ---
    if bam_path:
        try:
            candidates, _nuc_depth, _org_depth, _expected = _validate_depth_core(
                candidates,
                bam_path,
                nuclear_mean_depth,
                organelle_mean_depth,
                flank=0,
                tolerance=depth_tolerance,
            )
        except ImportError:
            return _failed(
                "detect_transfers_evidence",
                scope=scope,
                message="pysam not installed.",
                code="pysam_missing",
            )

    # --- layer 3: long-read linkage ---
    if hifi_reads:
        exe = str(minimap2_path) if minimap2_path else shutil.which("minimap2")
        if not exe:
            return _failed(
                "detect_transfers_evidence",
                scope=scope,
                message="minimap2 not found.",
                code="minimap2_missing",
            )
        candidates, _longread_stats = _validate_longread_core(
            candidates,
            nuclear_fasta,
            hifi_reads,
            exe,
            flank=longread_flank,
            organelle_fasta=organelle_fasta,
        )

    # --- assemble result ---
    candidates.sort(key=lambda c: c.evidence_level, reverse=True)
    n_depth = sum(1 for c in candidates if c.depth_supported)
    n_longread = sum(1 for c in candidates if c.longread_linked)
    methods = ["blastn"]
    if bam_path:
        methods.append("pysam")
    if hifi_reads:
        methods.append("minimap2")
    return OrganelleResult(
        operation_id=f"{_SUITE}.detect_transfers_evidence",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="ok",
        summary_text=(
            f"Evidence pipeline: {len(candidates)} candidates; "
            f"{n_depth} depth-supported, {n_longread} long-read-linked."
        ),
        metrics=FrozenMap(
            {
                "candidate_count": len(candidates),
                "depth_supported": n_depth,
                "longread_linked": n_longread,
                "max_evidence_level": max((c.evidence_level for c in candidates), default=0),
                "organelle": scope,
                "candidates": [c.as_dict() for c in candidates],
            }
        ),
        findings=(
            Finding(code="candidate_count", metric="candidate_count", value=len(candidates)),
            Finding(code="depth_supported", metric="depth_supported", value=n_depth),
            Finding(code="longread_linked", metric="longread_linked", value=n_longread),
        ),
        flags=("three_evidence",)
        if n_depth and n_longread
        else ("depth_evidence",)
        if n_depth
        else ("transfer_detected",)
        if candidates
        else (),
        artifacts=(),
        provenance=_provenance("detect_transfers_evidence", parameters, method="+".join(methods)),
    )


def _detect_transfers_blast_core(
    nuclear_fasta: str | Path,
    organelle_fasta: str | Path,
    *,
    min_identity: float,
    min_length: int,
    evalue: float,
    blastn_path: str | Path | None,
    makeblastdb_path: str | Path | None,
) -> list[TransferCandidate] | str:
    """BLASTN core: returns candidate list, or a string error message.

    Prefers the vendored LOSAT ``blastn`` (no makeblastdb, FASTA subject)
    unless ``blastn_path`` is given or ``ORG_VERSE_LOSAT_BIN`` opts out;
    NCBI BLAST+ is the fallback. String return is a sentinel for
    tool-missing / failure so the pipeline can surface it as a failed
    OrganelleResult without raising.
    """
    from .._losat import resolve_losat, run_losat_blastn

    if blastn_path is None and resolve_losat() is not None:
        try:
            rows = run_losat_blastn(organelle_fasta, nuclear_fasta, evalue=evalue)
        except RuntimeError as error:
            return f"blastn_failed: {error}"
        losat_candidates: list[TransferCandidate] = []
        for cells in rows:
            # LOSAT emits the fixed NCBI 12-column layout; the parser below
            # consumes only those columns (qlen/slen were never read).
            c = _parse_transfer_blast_line("\t".join(cells))
            if c is None:
                continue
            if c.identity < min_identity or c.length < min_length:
                continue
            if c.organelle_seqid == c.nuclear_seqid:
                continue
            losat_candidates.append(c)
        return _merge_overlapping(losat_candidates)

    blastn = str(blastn_path) if blastn_path else shutil.which("blastn")
    makeblastdb = str(makeblastdb_path) if makeblastdb_path else shutil.which("makeblastdb")
    if not blastn or not makeblastdb:
        return "blastn_missing"

    with tempfile.TemporaryDirectory(prefix="ov_transfer_blast_") as tmp:
        db_prefix = str(Path(tmp) / "nuclear_db")
        db_run = subprocess.run(
            [makeblastdb, "-in", str(nuclear_fasta), "-dbtype", "nucl", "-out", db_prefix],
            capture_output=True,
            text=True,
            check=False,
        )
        if db_run.returncode != 0:
            return f"makeblastdb_failed: {db_run.stderr.strip()}"
        blast_run = subprocess.run(
            [
                blastn,
                "-query",
                str(organelle_fasta),
                "-db",
                db_prefix,
                "-dust",
                "no",
                "-evalue",
                str(evalue),
                "-outfmt",
                "6 qseqid sseqid pident length mismatch gapopen "
                "qstart qend sstart send evalue bitscore qlen slen",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if blast_run.returncode != 0:
        return f"blastn_failed: {blast_run.stderr.strip()}"

    candidates: list[TransferCandidate] = []
    for line in blast_run.stdout.splitlines():
        if not line.strip():
            continue
        c = _parse_transfer_blast_line(line)
        if c is None:
            continue
        if c.identity < min_identity or c.length < min_length:
            continue
        if c.organelle_seqid == c.nuclear_seqid:
            continue
        candidates.append(c)
    return _merge_overlapping(candidates)


# ---------------------------------------------------------------------------
# helpers for the pipeline
# ---------------------------------------------------------------------------


def _to_candidate(c) -> TransferCandidate:
    if isinstance(c, TransferCandidate):
        return c
    if isinstance(c, dict):
        return TransferCandidate(**_candidate_fields(c))
    raise TypeError(f"cannot coerce {type(c)} to TransferCandidate")


def _candidate_fields(d: dict) -> dict:
    keys = (
        "nuclear_seqid",
        "nuclear_start",
        "nuclear_end",
        "organelle_seqid",
        "organelle_start",
        "organelle_end",
        "identity",
        "length",
        "evalue",
        "bitscore",
        "strand",
    )
    out = {k: d[k] for k in keys if k in d}
    out.setdefault("strand", "+")
    return out
