"""Mitochondrial tRNA annotation with a fixed native release engine.

The released path uses packaged CM/HMM profiles and the native CYK engine
without algorithm fallback. External tRNA tools remain available only through
an explicit internal ``external`` mode.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from organelleverse._losat import (
    resolve_losat,
    run_losat_with_query_coverage,
)
from organelleverse.core.errors import OrganelleDependencyError, OrganelleExecutionError
from organelleverse.core.external import MESSAGE_TAIL_LINES, run_external, tail_lines

from ..execution import CommandRunner
from .db import DBManager
from .models.feature import tRNAAnnotation
from .models.gene import ExonRecord, Strand

logger = logging.getLogger(__name__)


AMINO_ACID_CODES = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "fMet": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Sec": "U",
    "Undet": "?",
}


_AA1_TO_3 = {
    one: three for three, one in AMINO_ACID_CODES.items() if three not in ("fMet", "Undet")
}
_AA1_TO_3["fM"] = "fMet"


def trna_product(amino_acid: str) -> str:
    """NCBI tRNA product (``tRNA-Ser``) from a one-letter code or three-letter name.

    GenBank validation expects ``tRNA-`` plus a three-letter amino acid;
    ``tRNA-S`` or ``tRNA-S-TGA`` are rejected. Unknown identity is ``tRNA-Xxx``.
    """
    if amino_acid in AMINO_ACID_CODES and amino_acid != "Undet":
        return f"tRNA-{amino_acid}"
    return f"tRNA-{_AA1_TO_3.get(amino_acid, 'Xxx')}"


def _standardize_trna_name(amino_acid: str, anticodon: str) -> str:
    """Standardize tRNA name to NCBI-compatible format.

    NCBI format: trnI(cau) - lowercase, T notation, parentheses
    Legacy format: trnI-AAU - uppercase, U notation, hyphen

    Args:
        amino_acid: Amino acid letter (I, F, M, etc.) or full name (Ile, fMet)
        anticodon: Anticodon sequence (AAU, GAA, etc.) - may contain U or T

    Returns:
        Standardized name like "trnI(aat)"
    """
    # Handle fMet special case (formylmethionine -> Met)
    # ARAGORN outputs "fMet" which should map to "M", not "F"
    if amino_acid == "fM":
        aa_upper = "fM"
    elif amino_acid in AMINO_ACID_CODES:
        aa_upper = AMINO_ACID_CODES[amino_acid]
    else:
        aa_upper = amino_acid.upper() if len(amino_acid) <= 1 else amino_acid[0].upper()

    # Convert U to T in anticodon (RNA -> DNA notation)
    anticodon_t = anticodon.upper().replace("U", "T")

    # Format: trnX(abc) with lowercase anticodon
    return f"trn{aa_upper}({anticodon_t.lower()})"


@dataclass
class RawTRNA:
    """Raw tRNA hit from a prediction tool."""

    gene_name: str
    start: int  # 1-based
    end: int  # inclusive
    strand: int  # +1 or -1
    anticodon: str
    amino_acid: str
    score: float
    source: str  # "tRNAscan-SE" or "ARAGORN"
    intron_start: int | None = None
    intron_end: int | None = None


def _select_trna_engine() -> str:
    """Return the tRNA engine to use.

    The packaged native engine is the default; ``ORG_VERSE_TRNA_ENGINE=external``
    opts into the external tRNAscan-SE/ARAGORN cross-check path.
    """
    value = os.environ.get("ORG_VERSE_TRNA_ENGINE", "").strip().lower()
    if value in {"external", "tools"}:
        return "external"
    return "native"


def annotate_trna(
    fasta_path: Path,
    output_dir: Path,
    threads: int = 4,
    db_manager: DBManager | None = None,
    *,
    engine: str | None = None,
    tool_paths: Mapping[str, str] | None = None,
    command_runner: CommandRunner | None = None,
) -> list[tRNAAnnotation]:
    """Annotate tRNA genes with one explicitly selected engine.

    Args:
        fasta_path: Input genome FASTA
        output_dir: Directory for tool output files
        threads: Number of threads for an explicitly selected external engine
        db_manager: Database manager used only by explicit external fallback

    Returns:
        List of tRNAAnnotation objects
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    merged: list[RawTRNA] = []
    selected_engine = _select_trna_engine() if engine is None else engine

    if selected_engine == "trnascan_se":
        tool = tool_paths.get("tRNAscan-SE") if tool_paths is not None else None
        cmsearch = tool_paths.get("cmsearch") if tool_paths is not None else None
        if tool is None:
            raise OrganelleDependencyError(
                code="dependency_missing",
                message="external tRNA mode requires tRNAscan-SE",
                details={"missing": ["tRNAscan-SE"]},
            )
        merged = _run_trnascan_se(
            fasta_path,
            output_dir,
            threads,
            tool_path=tool,
            cmsearch_path=cmsearch,
            command_runner=command_runner,
        )
        return _raw_trna_to_annotations(merged)

    if selected_engine == "native":
        from .trna_native import annotate_trna_cm

        merged = annotate_trna_cm(fasta_path, output_dir, db_manager)
        logger.info(f"Native tRNA engine: {len(merged)} tRNA genes")
        return _raw_trna_to_annotations(merged)

    if selected_engine == "external":
        trnascan_hits = _run_trnascan_se(fasta_path, output_dir, threads)
        aragorn_hits = _run_aragorn(fasta_path, output_dir)
        if trnascan_hits or aragorn_hits:
            logger.info(
                f"External tRNA cross-check: tRNAscan-SE {len(trnascan_hits)} hits, "
                f"ARAGORN {len(aragorn_hits)} hits"
            )
            merged = _merge_trna_results(trnascan_hits, aragorn_hits) or merged

    if selected_engine == "external" and not merged and db_manager is not None:
        logger.info("No tRNAs from engine. Trying BLASTN fallback...")
        blastn_hits = _trna_by_blastn(fasta_path, output_dir, db_manager)
        logger.info(f"BLASTN fallback: {len(blastn_hits)} tRNA hits")
        merged = blastn_hits

    logger.info(f"After merge: {len(merged)} tRNA genes")

    return _raw_trna_to_annotations(merged)


def _raw_trna_to_annotations(merged: list[RawTRNA]) -> list[tRNAAnnotation]:
    annotations = []
    for hit in merged:
        exons = [ExonRecord(start=hit.start, end=hit.end, strand=Strand(hit.strand))]
        if hit.intron_start is not None:
            # tRNA with intron: split into two exons
            exons = [
                ExonRecord(start=hit.start, end=hit.intron_start - 1, strand=Strand(hit.strand)),
                ExonRecord(start=hit.intron_end + 1, end=hit.end, strand=Strand(hit.strand)),
            ]

        annotations.append(
            tRNAAnnotation(
                gene_name=hit.gene_name,
                product=trna_product(hit.amino_acid),
                exons=exons,
                strand=Strand(hit.strand),
                anticodon=hit.anticodon,
                amino_acid=hit.amino_acid,
                source_tool=hit.source,
                source_method=hit.source,
                trnascan_score=hit.score if "tRNAscan" in hit.source else None,
                score=hit.score,
            )
        )

    return annotations


def _run_trnascan_se(
    fasta_path: Path,
    output_dir: Path,
    threads: int,
    *,
    tool_path: str | None = None,
    cmsearch_path: str | None = None,
    command_runner: CommandRunner | None = None,
) -> list[RawTRNA]:
    """Run tRNAscan-SE 2.0 via subprocess."""
    tool = tool_path if tool_path is not None else shutil.which("tRNAscan-SE")
    if tool is None:
        logger.warning("tRNAscan-SE not found. Skipping tRNAscan annotation.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "trnascan.out"
    cmd = [
        tool,
        "--thread",
        str(threads),
    ]
    if conf := _patched_trnascan_conf(Path(tool), output_dir, cmsearch_path=cmsearch_path):
        cmd.extend(["-c", str(conf)])
    cmd.extend(["-O", "-q", "-Q", "-o", str(output_file), str(fasta_path)])

    if command_runner is not None:
        command_runner.run(
            tuple(cmd),
            stage="trna:trnascan_se",
            timeout=900,
        )
    else:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if result.returncode != 0:
                if output_file.exists():
                    hits = _parse_trnascan_output(output_file)
                    if hits:
                        logger.warning(
                            "tRNAscan-SE returned non-zero but produced parseable output."
                        )
                        return hits
                logger.warning(
                    f"tRNAscan-SE failed: {tail_lines(result.stderr, MESSAGE_TAIL_LINES)}"
                )
                return []
        except subprocess.TimeoutExpired:
            logger.warning("tRNAscan-SE timed out")
            return []

    if not output_file.exists():
        return []

    return _parse_trnascan_output(output_file)


def _patched_trnascan_conf(
    tool: Path,
    output_dir: Path,
    *,
    cmsearch_path: str | None = None,
) -> Path | None:
    conf = tool.parent / "tRNAscan-SE.conf"
    cmsearch = cmsearch_path if cmsearch_path is not None else shutil.which("cmsearch")
    if not conf.exists() or not cmsearch:
        return None
    text = conf.read_text()
    patched = re.sub(
        r"^infernal_dir:\s*.*$",
        f"infernal_dir: {Path(cmsearch).parent}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if patched == text:
        return None
    target = output_dir / "tRNAscan-SE.organelleverse.conf"
    target.write_text(patched)
    return target


def _parse_trnascan_output(output_file: Path) -> list[RawTRNA]:
    """Parse standard tRNAscan-SE 2.x output."""
    hits = []
    for line in output_file.read_text().strip().split("\n"):
        if line.startswith("#") or line.startswith("Sequence") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 9 or not parts[1].isdigit():
            continue

        try:
            start = int(parts[2])
            end = int(parts[3])
            aa = parts[4].strip()
            anticodon_raw = parts[5].strip()
            intron_begin = int(parts[6])
            intron_end_raw = int(parts[7])
            score = float(parts[8])

            if start > end:
                strand = -1
                start, end = end, start
            else:
                strand = 1

            intron_start = None
            intron_end = None
            if intron_begin > 0 and intron_end_raw > 0:
                intron_start, intron_end = sorted((intron_begin, intron_end_raw))

            gene_name = _standardize_trna_name(aa, anticodon_raw)
            anticodon_t = anticodon_raw.upper().replace("U", "T")

            hits.append(
                RawTRNA(
                    gene_name=gene_name,
                    start=start,
                    end=end,
                    strand=strand,
                    anticodon=anticodon_t,
                    amino_acid=aa,
                    score=score,
                    source="tRNAscan-SE",
                    intron_start=intron_start,
                    intron_end=intron_end,
                )
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping tRNAscan line: {e}")
            continue

    return hits


def _run_aragorn(fasta_path: Path, output_dir: Path) -> list[RawTRNA]:
    """Run ARAGORN via subprocess."""
    tool = shutil.which("aragorn")
    if tool is None:
        logger.warning("ARAGORN not found. Skipping ARAGORN annotation.")
        return []

    output_file = output_dir / "aragorn.txt"
    cmd = [
        tool,
        "-l",  # linear sequence
        "-gcstd",  # standard genetic code
        "-t",  # search for tRNA, not tmRNA
        "-w",  # one-line records that are stable to parse
        "-o",
        str(output_file),
        str(fasta_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"ARAGORN failed: {tail_lines(result.stderr, MESSAGE_TAIL_LINES)}")
            return []
    except subprocess.TimeoutExpired:
        logger.warning("ARAGORN timed out")
        return []

    if not output_file.exists():
        return []

    return _parse_aragorn_output(output_file)


def _parse_aragorn_output(output_file: Path) -> list[RawTRNA]:
    """Parse ARAGORN output."""
    hits = []
    text = output_file.read_text()
    in_results = False

    for line in text.split("\n"):
        if line.startswith(">"):
            in_results = True
            continue
        if not in_results or not line.strip():
            continue

        match = re.search(
            r"tRNA-([A-Za-z?]+)\s+"
            r"(c?)\[(\d+),(\d+)\]\s+"
            r"([0-9.]+)\s+\(([A-Za-z?]+)\)",
            line,
        )
        if not match:
            continue

        try:
            aa_info, complement, start_text, end_text, score_text, anticodon_raw = match.groups()
            start, end = sorted((int(start_text), int(end_text)))
            strand = -1 if complement else 1

            # Standardize tRNA name to NCBI format: trnX(abc) with T notation
            gene_name = _standardize_trna_name(aa_info, anticodon_raw)
            # Store anticodon in T notation (DNA) for consistency
            anticodon_t = anticodon_raw.upper().replace("U", "T")

            hits.append(
                RawTRNA(
                    gene_name=gene_name,
                    start=start,
                    end=end,
                    strand=strand,
                    anticodon=anticodon_t,
                    amino_acid=aa_info,
                    score=float(score_text),
                    source="ARAGORN",
                )
            )
        except (ValueError, IndexError):
            continue

    return hits


def _merge_trna_results(
    trnascan: list[RawTRNA],
    aragorn: list[RawTRNA],
    overlap_threshold: float = 0.8,
) -> list[RawTRNA]:
    """Merge tRNA results from both tools.

    Priority:
    1. Both tools find same tRNA → use higher confidence, mark "merged"
    2. Only tRNAscan-SE → use tRNAscan result
    3. Only ARAGORN → use ARAGORN result

    Handles anticodon notation differences (U vs T) between tools.
    """
    if not trnascan:
        return _dedup_trnas(aragorn)
    if not aragorn:
        return _dedup_trnas(trnascan)

    # Normalize anticodons: replace U with T for consistent comparison
    def norm_anticodon(ac: str) -> str:
        return ac.upper().replace("U", "T")

    merged = []
    used_aragorn = set()

    for ts in trnascan:
        best_match = None
        best_overlap = 0

        for j, ar in enumerate(aragorn):
            if j in used_aragorn:
                continue
            if norm_anticodon(ts.anticodon) != norm_anticodon(ar.anticodon):
                continue

            # Check overlap (regardless of strand)
            o_start = max(ts.start, ar.start)
            o_end = min(ts.end, ar.end)
            overlap = max(0, o_end - o_start + 1)
            min_len = min(ts.end - ts.start, ar.end - ar.start) + 1
            if min_len > 0 and overlap / min_len > overlap_threshold and overlap > best_overlap:
                best_match = j
                best_overlap = overlap

        if best_match is not None:
            # Both found it — prefer tRNAscan score, keep tRNAscan strand
            used_aragorn.add(best_match)
            merged.append(
                RawTRNA(
                    gene_name=ts.gene_name,
                    start=ts.start,
                    end=ts.end,
                    strand=ts.strand,
                    anticodon=ts.anticodon,
                    amino_acid=ts.amino_acid,
                    score=ts.score,
                    source="merged",
                    intron_start=ts.intron_start,
                    intron_end=ts.intron_end,
                )
            )
        else:
            merged.append(ts)

    # Add ARAGORN-only hits
    for j, ar in enumerate(aragorn):
        if j not in used_aragorn:
            merged.append(ar)

    return _dedup_trnas(merged)


def _dedup_trnas(trnas: list[RawTRNA], overlap_bp: int = 30) -> list[RawTRNA]:
    """Remove duplicate tRNAs at the same genomic location.

    If two tRNAs overlap significantly (>overlap_bp), keep the one with
    higher score. If same score, prefer tRNAscan-SE > merged > ARAGORN.
    """
    if not trnas:
        return []

    source_priority = {
        "Native": 4,
        "tRNAscan-SE": 3,
        "merged": 2,
        "ARAGORN": 1,
        "BLASTN": 0,
        "BLASTN-Python": 0,
    }

    # Sort by score desc, then source priority
    sorted_trnas = sorted(
        trnas, key=lambda t: (t.score, source_priority.get(t.source, 0)), reverse=True
    )

    kept: list[RawTRNA] = []
    for t in sorted_trnas:
        is_dup = False
        for k in kept:
            overlap = max(0, min(t.end, k.end) - max(t.start, k.start) + 1)
            if overlap > overlap_bp:
                is_dup = True
                break
        if not is_dup:
            kept.append(t)

    return sorted(kept, key=lambda t: t.start)


# ── BLASTN fallback ─────────────────────────────────────────────


def _trna_by_blastn(
    fasta_path: Path,
    output_dir: Path,
    db_manager: DBManager,
    evalue: float = 1e-3,
    min_identity: float = 65.0,
    min_coverage: float = 60.0,
) -> list[RawTRNA]:
    """Detect tRNA genes using BLASTN against reference tRNA database.

    Uses the packaged tRNA_STD_3086_Uniq.fasta (8682 tRNA sequences).
    """
    trna_ref_dir = db_manager.trna_ref_dir
    if not trna_ref_dir.exists():
        logger.warning(f"tRNA reference directory not found: {trna_ref_dir}")
        return []

    # Find reference FASTA files
    ref_files = sorted(trna_ref_dir.glob("*.fasta")) + sorted(trna_ref_dir.glob("*.fa"))
    if not ref_files:
        logger.warning(f"No tRNA reference files in {trna_ref_dir}")
        return []

    # Combine all reference sequences into one query file
    combined_ref = output_dir / "trna_refs_combined.fasta"
    with open(combined_ref, "w") as out:
        for rf in ref_files:
            out.write(rf.read_text())

    losat = resolve_losat()
    blastn = shutil.which("blastn")
    makeblastdb = shutil.which("makeblastdb")

    if losat is not None or (blastn and makeblastdb):
        return _trna_blastn_external(
            fasta_path,
            output_dir,
            combined_ref,
            blastn,
            makeblastdb,
            evalue,
            min_identity,
            min_coverage,
            losat=losat,
        )
    # Neither search binary: keep the pure-Python local alignment path, which
    # returns real (slower) hits rather than an empty result.
    logger.info("LOSAT and BLAST+ not found. Using pure-Python alignment fallback for tRNA.")
    return _trna_python_fallback(fasta_path, combined_ref, min_identity, min_coverage)


def _trna_blastn_external(
    fasta_path: Path,
    output_dir: Path,
    ref_fasta: Path,
    blastn: str | None,
    makeblastdb: str | None,
    evalue: float,
    min_identity: float,
    min_coverage: float,
    *,
    losat: str | None = None,
) -> list[RawTRNA]:
    """Run BLASTN using external BLAST+ tools for tRNA detection."""
    if losat is None:
        # Create BLAST database from genome
        db_path = output_dir / "genome_blastdb_trna"
        try:
            run_external(
                [makeblastdb, "-in", str(fasta_path), "-dbtype", "nucl", "-out", str(db_path)],
                timeout=120,
            )
        except OrganelleExecutionError as e:
            logger.warning(f"makeblastdb failed for tRNA: {e.message}")
            return []

    if losat is not None:
        raw_rows = run_losat_with_query_coverage(
            "blastn",
            ref_fasta,
            fasta_path,
            evalue=evalue,
            max_target_seqs=5,
            executable=losat,
            timeout=600,
            stage="trna:blastn",
        )
        rows = [
            [r[i] for i in (0, 1, 6, 7, 8, 9, 10, 11, 3, 2)] + [r[12]]
            for r in raw_rows
            if len(r) >= 13
        ]
    else:
        out_file = output_dir / "trna_blastn.tsv"
        cmd = [
            blastn,
            "-query",
            str(ref_fasta),
            "-db",
            str(db_path),
            "-out",
            str(out_file),
            "-outfmt",
            "6 qseqid sseqid qstart qend sstart send evalue bitscore length pident qcovs",
            "-evalue",
            str(evalue),
            "-strand",
            "both",
            "-max_target_seqs",
            "5",
        ]

        try:
            run_external(cmd, timeout=600)
        except OrganelleExecutionError as e:
            logger.warning(f"BLASTN failed for tRNA: {e.message}")
            return []

        rows = (
            [line.split("\t") for line in out_file.read_text().splitlines() if line.strip()]
            if out_file.exists()
            else []
        )
    hits = []
    seen_regions = []  # For deduplication

    for parts in rows:
        if len(parts) < 11:
            continue

        try:
            qseqid = parts[0]
            sstart = int(parts[4])
            send = int(parts[5])
            score = float(parts[7])
            pident = float(parts[9])
            qcovs = float(parts[10])
        except (ValueError, IndexError):
            continue

        if pident < min_identity or qcovs < min_coverage:
            continue

        # Determine strand and coordinates
        if sstart <= send:
            strand = 1
            start, end = sstart, send
        else:
            strand = -1
            start, end = send, sstart

        # Parse tRNA name from query ID
        aa, anticodon_raw = _parse_trna_name(qseqid)

        # Deduplicate overlapping hits
        is_dup = False
        for sr_start, sr_end, sr_strand in seen_regions:
            if strand == sr_strand:
                overlap = max(0, min(end, sr_end) - max(start, sr_start) + 1)
                if overlap > 30:  # Overlap >30bp = duplicate
                    is_dup = True
                    break
        if is_dup:
            continue

        # Standardize tRNA name to NCBI format: trnX(abc) with T notation
        gene_name = _standardize_trna_name(aa, anticodon_raw)
        # Store anticodon in T notation (DNA) for consistency
        anticodon_t = anticodon_raw.upper().replace("U", "T")

        seen_regions.append((start, end, strand))
        hits.append(
            RawTRNA(
                gene_name=gene_name,
                start=start,
                end=end,
                strand=strand,
                anticodon=anticodon_t,
                amino_acid=aa,
                score=score,
                source="BLASTN",
            )
        )

    return hits


def _trna_python_fallback(
    fasta_path: Path,
    ref_fasta: Path,
    min_identity: float,
    min_coverage: float,
) -> list[RawTRNA]:
    """Pure-Python tRNA detection using k-mer seed + extend."""
    from Bio import SeqIO
    from Bio.Seq import Seq

    record = SeqIO.read(str(fasta_path), "fasta")
    genome_seq = str(record.seq).upper()
    genome_len = len(genome_seq)
    rc_genome = str(Seq(genome_seq).reverse_complement()).upper()

    kmer_size = 12
    hits = []
    seen_regions = []

    for ref_record in SeqIO.parse(str(ref_fasta), "fasta"):
        ref_seq = str(ref_record.seq).upper()
        ref_len = len(ref_seq)
        if ref_len < 50:
            continue

        # Parse tRNA info from name
        aa, anticodon_raw = _parse_trna_name(ref_record.id)
        # Standardize tRNA name and convert U to T
        gene_name = _standardize_trna_name(aa, anticodon_raw)
        anticodon_t = anticodon_raw.upper().replace("U", "T")

        # Build k-mer index
        kmer_idx: dict[str, list[int]] = {}
        for i in range(0, ref_len - kmer_size + 1, 3):
            kmer = ref_seq[i : i + kmer_size]
            if "N" not in kmer:
                kmer_idx.setdefault(kmer, []).append(i)

        # Scan genome
        for pos in range(0, genome_len - kmer_size + 1, 50):
            kmer = genome_seq[pos : pos + kmer_size]
            if kmer not in kmer_idx:
                continue

            for q_pos in kmer_idx[kmer]:
                g_start = max(0, pos - q_pos)
                g_end = min(genome_len, g_start + ref_len + 10)
                candidate = genome_seq[g_start:g_end]

                compare_len = min(len(candidate), ref_len)
                matches = sum(1 for i in range(compare_len) if candidate[i] == ref_seq[i])
                identity = matches / compare_len * 100
                coverage = compare_len / ref_len * 100

                if identity >= min_identity and coverage >= min_coverage:
                    start = g_start + 1
                    end = g_start + compare_len

                    # Dedup
                    is_dup = any(min(end, se) - max(start, ss) > 30 for ss, se, _ in seen_regions)
                    if not is_dup:
                        seen_regions.append((start, end, 1))
                        hits.append(
                            RawTRNA(
                                gene_name=gene_name,
                                start=start,
                                end=end,
                                strand=1,
                                anticodon=anticodon_t,
                                amino_acid=aa,
                                score=identity,
                                source="BLASTN-Python",
                            )
                        )

            # Also check RC
            kmer_rc = rc_genome[pos : pos + kmer_size]
            if kmer_rc not in kmer_idx:
                continue

            for q_pos in kmer_idx[kmer_rc]:
                g_start = max(0, pos - q_pos)
                g_end = min(genome_len, g_start + ref_len + 10)
                candidate = rc_genome[g_start:g_end]

                compare_len = min(len(candidate), ref_len)
                matches = sum(1 for i in range(compare_len) if candidate[i] == ref_seq[i])
                identity = matches / compare_len * 100
                coverage = compare_len / ref_len * 100

                if identity >= min_identity and coverage >= min_coverage:
                    start_rc = g_start + 1
                    end_rc = g_start + compare_len
                    # Convert to forward coordinates
                    start = genome_len - end_rc + 1
                    end = genome_len - start_rc + 1

                    is_dup = any(min(end, se) - max(start, ss) > 30 for ss, se, _ in seen_regions)
                    if not is_dup:
                        seen_regions.append((start, end, -1))
                        hits.append(
                            RawTRNA(
                                gene_name=gene_name,
                                start=start,
                                end=end,
                                strand=-1,
                                anticodon=anticodon_t,
                                amino_acid=aa,
                                score=identity,
                                source="BLASTN-Python",
                            )
                        )

    return hits


def _parse_trna_name(qseqid: str) -> tuple[str, str]:
    """Parse amino acid and anticodon from tRNA reference sequence name.

    Examples:
        RefVi3086_tRNA_trnA-UGC_0001 -> ("A", "UGC")
        trnF-GAA -> ("F", "GAA")
        trnI(aat) -> ("I", "AAT")  # NCBI lowercase format
    """
    m = re.search(r"trnfM[-(]?([A-Z]{3})\)?", qseqid, re.IGNORECASE)
    if m:
        return "fM", m.group(1).upper()

    # Try NCBI standard format: trnX(abc) with lowercase anticodon
    m = re.search(r"trn([A-Z][a-z]?)\(([A-Z]{3})\)", qseqid, re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).upper()

    # Try hyphen format: trnX-ABC or trnX-abc
    m = re.search(r"trn([A-Z][a-z]?)-?([A-Z]{3})", qseqid, re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).upper()

    m = re.search(r"tRNA[_-]?([A-Z][a-z]{1,2})", qseqid, re.IGNORECASE)
    if m:
        aa_map = {
            "Ala": "A",
            "Arg": "R",
            "Asn": "N",
            "Asp": "D",
            "Cys": "C",
            "Gln": "Q",
            "Glu": "E",
            "Gly": "G",
            "His": "H",
            "Ile": "I",
            "Leu": "L",
            "Lys": "K",
            "Met": "M",
            "fMet": "M",
            "Phe": "F",
            "Pro": "P",
            "Ser": "S",
            "Thr": "T",
            "Trp": "W",
            "Tyr": "Y",
            "Val": "V",
            "Sec": "U",
        }
        aa_name = m.group(1)
        aa = aa_map.get(aa_name, aa_name[0].upper())
        return aa, "NNN"

    return "X", "NNN"
