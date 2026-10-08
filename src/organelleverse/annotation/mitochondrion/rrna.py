"""Mitochondrial rRNA annotation with native references and optional fallbacks.

Primary: packaged references with native seed/align refinement.
Fallbacks: external rRNA tools, then BLASTN against packaged references.

Merging rules (from PMGA v1 editBoundary.py lines 917-1063):
  - rrn26: merge fragments within [-200, 1000] bp; reject if <2000 bp
  - rrn18: merge fragments within [0, 500] bp; reject if <1000 bp
  - rrn5:  reject if <80 bp
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from Bio.Seq import Seq

from organelleverse._losat import (
    resolve_losat,
    run_losat_with_query_coverage,
)
from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.core.external import MESSAGE_TAIL_LINES, run_external, tail_lines

from .db import DBManager
from .models.feature import rRNAAnnotation
from .models.gene import ExonRecord, Strand

logger = logging.getLogger(__name__)


@dataclass
class RawRRNA:
    """Raw rRNA hit from Barrnap or BLASTN fallback."""

    gene_name: str
    start: int
    end: int
    strand: int
    score: float
    rrna_type: str  # "5S", "18S", "26S"
    source_tool: str = ""


def _select_rrna_engine() -> str:
    """Native packaged engine is default; 'external' opts into barrnap."""
    value = os.environ.get("ORG_VERSE_RRNA_ENGINE", "").strip().lower()
    return "external" if value in {"external", "barrnap"} else "native"


def annotate_rrna(
    fasta_path: Path,
    output_dir: Path,
    kingdom: str = "mito",
    db_manager: DBManager | None = None,
    *,
    engine: str | None = None,
) -> list[rRNAAnnotation]:
    """Annotate rRNA genes using packaged native engines; barrnap is opt-in."""
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_hits: list[RawRRNA] = []
    selected_engine = _select_rrna_engine() if engine is None else engine

    if selected_engine == "pyhmmer":
        if db_manager is None:
            raise ValueError("pyhmmer rRNA annotation requires a database manager")
        from .rrna_hmm import annotate_rrna_hmm

        raw_hits = annotate_rrna_hmm(fasta_path, output_dir, db_manager)
        return _raw_rrna_to_annotations(_merge_rrna_fragments(raw_hits))

    if selected_engine == "native" and db_manager is not None:
        try:
            from .rrna_hmm import annotate_rrna_hmm

            raw_hits = annotate_rrna_hmm(fasta_path, output_dir, db_manager)
            logger.info(f"Native rRNA (pyhmmer): {len(raw_hits)} raw rRNA hits")
        except Exception as e:
            logger.warning(f"pyhmmer rRNA failed: {e}")
        if not raw_hits:
            try:
                from .rrna_native import annotate_rrna_native

                raw_hits = annotate_rrna_native(fasta_path, output_dir, db_manager)
                logger.info(f"Native rRNA (reference align): {len(raw_hits)} raw rRNA hits")
            except Exception as e:
                logger.warning(f"Native rRNA reference-align failed: {e}")

    if selected_engine == "external" or not raw_hits:
        barrnap_hits = _run_barrnap(fasta_path, output_dir, kingdom)
        if barrnap_hits:
            logger.info(f"Barrnap cross-check: {len(barrnap_hits)} raw rRNA hits")
            raw_hits = barrnap_hits or raw_hits

    if not raw_hits and db_manager is not None:
        logger.info("No rRNA from engine. Trying BLASTN fallback...")
        raw_hits = _rrna_by_blastn(fasta_path, output_dir, db_manager)
        logger.info(f"BLASTN fallback: {len(raw_hits)} raw rRNA hits")

    merged = _merge_rrna_fragments(raw_hits)
    logger.info(f"After merge: {len(merged)} rRNA genes")

    return _raw_rrna_to_annotations(merged)


def _raw_rrna_to_annotations(merged: list[RawRRNA]) -> list[rRNAAnnotation]:
    annotations = []
    for hit in merged:
        strand = Strand.PLUS if hit.strand == 1 else Strand.MINUS
        source_tool = hit.source_tool or ("Barrnap" if hit.score > 0 else "BLASTN")
        annotations.append(
            rRNAAnnotation(
                gene_name=hit.gene_name,
                product=f"{hit.rrna_type} ribosomal RNA",
                exons=[ExonRecord(start=hit.start, end=hit.end, strand=strand)],
                strand=strand,
                rrna_type=hit.rrna_type,
                source_tool=source_tool,
                source_method=source_tool,
                score=hit.score,
            )
        )

    return annotations


def _run_barrnap(fasta_path: Path, output_dir: Path, kingdom: str) -> list[RawRRNA]:
    """Run Barrnap via subprocess."""
    tool = shutil.which("barrnap")
    if tool is None:
        logger.warning("Barrnap not found. Skipping Barrnap rRNA annotation.")
        return []

    output_file = output_dir / "barrnap.gff"
    cmd = [
        tool,
        "--kingdom",
        kingdom,
        "--outseq",
        str(output_file),
        str(fasta_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(
                f"Barrnap failed: {tail_lines(result.stderr, MESSAGE_TAIL_LINES)}"
            )
            return []
    except subprocess.TimeoutExpired:
        logger.warning("Barrnap timed out")
        return []
    except Exception as e:
        logger.warning(f"Barrnap error: {e}")
        return []

    return _parse_barrnap_gff(output_file)


def _parse_barrnap_gff(gff_file: Path) -> list[RawRRNA]:
    """Parse Barrnap GFF output."""
    if not gff_file.exists():
        return []

    hits = []
    for line in gff_file.read_text().split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        if parts[2] != "rRNA":
            continue

        start = int(parts[3])
        end = int(parts[4])
        strand_val = 1 if parts[6] == "+" else -1
        attrs = parts[8]

        # Parse attributes
        name = ""
        product = ""
        for attr in attrs.split(";"):
            if "=" in attr:
                k, v = attr.split("=", 1)
                if k == "Name":
                    name = v
                elif k == "product":
                    product = v

        # Determine rRNA type
        rrna_type = _classify_rrna(name, product, end - start + 1)

        # Extract score
        score = float(parts[5]) if parts[5] != "." else 0.0

        hits.append(
            RawRRNA(
                gene_name=f"rrn{rrna_type.replace('S', '')}",
                start=start,
                end=end,
                strand=strand_val,
                score=score,
                rrna_type=rrna_type,
                source_tool="Barrnap",
            )
        )

    return hits


def _rrna_by_blastn(
    fasta_path: Path,
    output_dir: Path,
    db_manager: DBManager,
    evalue: float = 1e-5,
    min_identity: float = 70.0,
    min_coverage: float = 50.0,
) -> list[RawRRNA]:
    """Detect rRNA genes using BLASTN against reference rRNA sequences.

    Uses packaged reference FASTA files (rrn5/18/26) as query sequences.
    Uses LOSAT when available, otherwise NCBI BLAST+, otherwise a pure-Python
    local alignment fallback.
    """
    rrna_ref_dir = db_manager.rrna_ref_dir
    if not rrna_ref_dir.exists():
        logger.warning(f"rRNA reference directory not found: {rrna_ref_dir}")
        return []

    # Collect all reference rRNA FASTA files
    ref_files = sorted(rrna_ref_dir.glob("*.fasta"))
    if not ref_files:
        logger.warning(f"No rRNA reference FASTA files in {rrna_ref_dir}")
        return []

    losat = resolve_losat()
    blastn = shutil.which("blastn")
    makeblastdb = shutil.which("makeblastdb")

    if losat is not None or (blastn and makeblastdb):
        return _rrna_blastn_external(
            fasta_path,
            output_dir,
            ref_files,
            blastn,
            makeblastdb,
            evalue,
            min_identity,
            min_coverage,
            losat=losat,
        )
    # Neither search binary: keep the pure-Python local alignment path, which
    # returns real (slower) hits rather than an empty result.
    logger.info("LOSAT and BLAST+ not found. Using pure-Python alignment fallback for rRNA.")
    return _rrna_python_fallback(fasta_path, ref_files, min_identity, min_coverage)


def _rrna_blastn_external(
    fasta_path: Path,
    output_dir: Path,
    ref_files: list[Path],
    blastn: str | None,
    makeblastdb: str | None,
    evalue: float,
    min_identity: float,
    min_coverage: float,
    *,
    losat: str | None = None,
) -> list[RawRRNA]:
    """Run BLASTN using external BLAST+ tools."""
    hits = []

    if losat is None:
        # Create BLAST database from input genome
        db_path = output_dir / "rrna_genome_blastdb"
        try:
            cmd = [
                makeblastdb,
                "-in",
                str(fasta_path),
                "-dbtype",
                "nucl",
                "-out",
                str(db_path),
                "-parse_seqids",
            ]
            run_external(cmd, timeout=120)
        except OrganelleExecutionError as e:
            logger.warning(f"makeblastdb failed: {e.message}")
            return []

    for ref_file in ref_files:
        # Determine rRNA type from filename
        ref_name = ref_file.stem.lower()
        if "26s" in ref_name or "rrn26" in ref_name or "lsu" in ref_name:
            rrna_type = "26S"
        elif "18s" in ref_name or "rrn18" in ref_name or "ssu" in ref_name:
            rrna_type = "18S"
        elif "5s" in ref_name or "rrn5" in ref_name:
            rrna_type = "5S"
        else:
            continue

        if losat is not None:
            raw_rows = run_losat_with_query_coverage(
                "blastn",
                ref_file,
                fasta_path,
                evalue=evalue,
                max_target_seqs=10,
                executable=losat,
                timeout=300,
                stage="rrna:blastn",
            )
            rows = [
                [r[i] for i in (0, 1, 6, 7, 8, 9, 10, 11, 3, 2)] + [r[12]]
                for r in raw_rows
                if len(r) >= 13
            ]
        else:
            out_file = output_dir / f"blastn_{rrna_type}.tsv"
            cmd = [
                blastn,
                "-query",
                str(ref_file),
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
                "10",
            ]

            try:
                run_external(cmd, timeout=300)
            except OrganelleExecutionError as e:
                logger.warning(f"BLASTN failed for {rrna_type}: {e.message}")
                continue

            rows = (
                [line.split("\t") for line in out_file.read_text().splitlines() if line.strip()]
                if out_file.exists()
                else []
            )
        # Parse BLASTN output
        for parts in rows:
            if len(parts) < 10:
                continue

            try:
                qstart = int(parts[2])
                qend = int(parts[3])
                sstart = int(parts[4])
                send = int(parts[5])
                float(parts[6])
                score_val = float(parts[7])
                aln_len = int(parts[8])
                pident = float(parts[9])
                qcovs = float(parts[10]) if len(parts) > 10 else 0.0
            except (ValueError, IndexError):
                continue

            if pident < min_identity:
                continue

            # Determine strand and coordinates
            if sstart <= send:
                strand = 1
                start, end = sstart, send
            else:
                strand = -1
                start, end = send, sstart

            # Calculate coverage from query coverage or alignment length
            if qcovs > 0:
                coverage = qcovs
            else:
                ref_len = abs(qend - qstart) + 1
                coverage = aln_len / ref_len * 100 if ref_len > 0 else 0
            if coverage < min_coverage:
                continue

            hits.append(
                RawRRNA(
                    gene_name=f"rrn{rrna_type.replace('S', '')}",
                    start=start,
                    end=end,
                    strand=strand,
                    score=score_val,
                    rrna_type=rrna_type,
                    source_tool="BLASTN",
                )
            )

    return hits


def _rrna_python_fallback(
    fasta_path: Path,
    ref_files: list[Path],
    min_identity: float,
    min_coverage: float,
) -> list[RawRRNA]:
    """Pure-Python rRNA detection using local alignment.

    Uses a simple sliding-window k-mer match + extension approach.
    """
    from Bio import SeqIO

    # Load genome
    record = SeqIO.read(str(fasta_path), "fasta")
    genome_seq = str(record.seq).upper()

    hits = []
    for ref_file in ref_files:
        # Determine rRNA type
        ref_name = ref_file.stem.lower()
        if "26s" in ref_name or "rrn26" in ref_name or "lsu" in ref_name:
            rrna_type = "26S"
        elif "18s" in ref_name or "rrn18" in ref_name or "ssu" in ref_name:
            rrna_type = "18S"
        elif "5s" in ref_name or "rrn5" in ref_name:
            rrna_type = "5S"
        else:
            continue

        # Load reference rRNA sequences
        for ref_record in SeqIO.parse(str(ref_file), "fasta"):
            ref_seq = str(ref_record.seq).upper()
            ref_len = len(ref_seq)
            if ref_len < 50:
                continue

            # Search forward strand
            fwd_hits = _local_align_search(genome_seq, ref_seq, min_identity, min_coverage)
            for start, end, score in fwd_hits:
                hits.append(
                    RawRRNA(
                        gene_name=f"rrn{rrna_type.replace('S', '')}",
                        start=start,
                        end=end,
                        strand=1,
                        score=score,
                        rrna_type=rrna_type,
                        source_tool="PythonFallback",
                    )
                )

            # Search reverse strand
            rc_genome = str(Seq(genome_seq).reverse_complement()).upper()
            rev_hits = _local_align_search(rc_genome, ref_seq, min_identity, min_coverage)
            for start_rc, end_rc, score in rev_hits:
                # Convert RC coordinates to forward
                start = len(genome_seq) - end_rc + 1
                end = len(genome_seq) - start_rc + 1
                hits.append(
                    RawRRNA(
                        gene_name=f"rrn{rrna_type.replace('S', '')}",
                        start=start,
                        end=end,
                        strand=-1,
                        score=score,
                        rrna_type=rrna_type,
                        source_tool="PythonFallback",
                    )
                )

    return hits


def _local_align_search(
    genome: str,
    query: str,
    min_identity: float,
    min_coverage: float,
    kmer_size: int = 15,
    step: int = 100,
) -> list[tuple[int, int, float]]:
    """Simple k-mer seed + extend search for homologous regions.

    Returns list of (start, end, score) tuples (1-based inclusive).
    """
    query_len = len(query)
    genome_len = len(genome)
    # Build k-mer index for query
    kmer_idx: dict[str, list[int]] = {}
    for i in range(0, query_len - kmer_size + 1):
        kmer = query[i : i + kmer_size]
        if "N" not in kmer:
            kmer_idx.setdefault(kmer, []).append(i)

    # Scan genome with step
    found_regions = []

    for pos in range(0, genome_len - kmer_size + 1, step):
        kmer = genome[pos : pos + kmer_size]
        if kmer in kmer_idx:
            for q_pos in kmer_idx[kmer]:
                # Extend from this seed
                g_start = max(0, pos - q_pos)
                g_end = min(genome_len, g_start + query_len)
                candidate = genome[g_start:g_end]

                # Count matches
                compare_len = min(len(candidate), query_len)
                matches = sum(1 for i in range(compare_len) if candidate[i] == query[i])
                identity = matches / compare_len * 100 if compare_len > 0 else 0
                coverage = compare_len / query_len * 100

                if identity >= min_identity and coverage >= min_coverage:
                    found_regions.append((g_start + 1, g_end, identity))

    # Deduplicate overlapping hits (keep best score)
    if not found_regions:
        return []

    found_regions.sort(key=lambda x: x[2], reverse=True)
    kept = []
    for start, end, score in found_regions:
        overlap = False
        for ks, ke, _ in kept:
            if max(start, ks) <= min(end, ke):
                overlap = True
                break
        if not overlap:
            kept.append((start, end, score))

    return kept


def _classify_rrna(name: str, product: str, length: int) -> str:
    """Classify rRNA type from name/product/length."""
    combined = (name + " " + product).lower()

    if "26s" in combined or "lsu" in combined or "large" in combined:
        return "26S"
    if "18s" in combined or "ssu" in combined or "small" in combined:
        return "18S"
    if "5s" in combined:
        return "5S"

    # Fallback: by length
    if length > 2000:
        return "26S"
    elif length > 1000:
        return "18S"
    else:
        return "5S"


def _merge_rrna_fragments(hits: list[RawRRNA]) -> list[RawRRNA]:
    """Merge rRNA fragments based on v1 rules.

    Rules from PMGA v1 (02.editBoundary.py):
    - rrn26: merge if fragments within [-200, 1000] bp; reject if <2000 bp
    - rrn18: merge if fragments within [0, 500] bp; reject if <1000 bp
    - rrn5: reject if <80 bp
    """
    if not hits:
        return []

    # Group by type
    by_type: dict[str, list[RawRRNA]] = {}
    for h in hits:
        by_type.setdefault(h.rrna_type, []).append(h)

    merged = []
    for rrna_type, type_hits in by_type.items():
        type_hits = _drop_contained_duplicates(type_hits)
        # Sort by start
        type_hits.sort(key=lambda h: h.start)

        if rrna_type == "26S":
            merged.extend(_merge_by_proximity(type_hits, -200, 1000, 2000))
        elif rrna_type == "18S":
            merged.extend(_merge_by_proximity(type_hits, 0, 500, 1000))
        elif rrna_type == "5S":
            # Just filter by length
            for h in type_hits:
                if h.end - h.start + 1 >= 80:
                    merged.append(h)

    return merged


# A hit this far inside a longer hit of the same type and strand adds no locus of its own.
_CONTAINED_FRACTION = 0.9


def _drop_contained_duplicates(
    hits: list[RawRRNA], containment: float = _CONTAINED_FRACTION
) -> list[RawRRNA]:
    """Remove hits that sit inside a longer hit of the same type and strand.

    Every packaged reference is a separate nhmmer query, so a genome with many close relatives in the
    reference set reports the same rRNA once per query: Arabidopsis gave five 5S, five 18S and two 26S
    hits at three loci. Longest-first, a hit is kept unless at least ``containment`` of it is already
    covered by kept hits of its strand. Distinct copies (other loci) and fragments that only overlap
    their neighbour at the ends, which the proximity merge joins, are not touched.
    """
    kept: list[RawRRNA] = []
    for hit in sorted(hits, key=lambda h: (-(h.end - h.start + 1), -h.score, h.start)):
        length = hit.end - hit.start + 1
        spans = sorted((k.start, k.end) for k in kept if k.strand == hit.strand)
        covered, edge = 0, hit.start - 1
        for start, end in spans:
            if end <= edge or start > hit.end:
                continue
            covered += min(end, hit.end) - max(start, edge + 1) + 1
            edge = max(edge, min(end, hit.end))
        if covered / length < containment:
            kept.append(hit)
    return kept


def _merge_by_proximity(
    hits: list[RawRRNA],
    min_gap: int,
    max_gap: int,
    min_length: int,
) -> list[RawRRNA]:
    """Merge adjacent rRNA fragments within gap range."""
    if len(hits) <= 1:
        result = []
        for h in hits:
            if h.end - h.start + 1 >= min_length:
                result.append(h)
        return result

    merged = [hits[0]]
    for h in hits[1:]:
        prev = merged[-1]
        gap = h.start - prev.end - 1

        if min_gap <= gap <= max_gap and prev.strand == h.strand:
            # Merge
            merged[-1] = RawRRNA(
                gene_name=prev.gene_name,
                start=prev.start,
                end=h.end,
                strand=prev.strand,
                score=max(prev.score, h.score),
                rrna_type=prev.rrna_type,
            )
        else:
            merged.append(h)

    # Filter by minimum length
    return [h for h in merged if h.end - h.start + 1 >= min_length]
