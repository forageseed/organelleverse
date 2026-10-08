"""BLAST execution and parsing for the OrganelleVerse plastome backend."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from organelleverse.core.errors import OrganelleExecutionError

from .models import BlastHit, BlastTools, ReferenceQuery
from .references import clean_seq
from .tools import run_command

logger = logging.getLogger(__name__)


def _ncbi_fallback(tools: BlastTools, *, need_tblastn: bool) -> BlastTools | None:
    """The same tools without LOSAT, when NCBI BLAST+ can do the search instead."""
    if tools.losat is None or tools.blastn is None or tools.makeblastdb is None:
        return None
    if need_tblastn and tools.tblastn is None:
        return None
    return replace(tools, losat=None)


def run_plastome_blasts(
    target_seq: str,
    queries: tuple[ReferenceQuery, ...],
    *,
    work_dir: Path,
    tools: BlastTools,
    min_identity: float,
    qcoverage_range: tuple[float, float],
    threads: int = 1,
) -> dict[str, BlastHit]:
    """Reference transfer searches; a LOSAT failure falls back to NCBI BLAST+ when installed.

    LOSAT aborts on some inputs (a blastn identity walk reaching the first base
    of the subject underflowed in builds before 2026-10-04: Welwitschia and
    Cuscuta plastomes), which used to fail the whole annotation.
    """
    try:
        return _run_plastome_blasts(target_seq, queries, work_dir=work_dir, tools=tools, min_identity=min_identity,
                                    qcoverage_range=qcoverage_range, threads=threads)
    except OrganelleExecutionError as exc:
        fallback = _ncbi_fallback(tools, need_tblastn=True)
        if fallback is None:
            raise
        logger.warning("LOSAT failed (%s); repeating the plastome searches with NCBI BLAST+", exc)
        return _run_plastome_blasts(target_seq, queries, work_dir=work_dir, tools=fallback,
                                    min_identity=min_identity, qcoverage_range=qcoverage_range, threads=threads)


def _run_plastome_blasts(
    target_seq: str,
    queries: tuple[ReferenceQuery, ...],
    *,
    work_dir: Path,
    tools: BlastTools,
    min_identity: float,
    qcoverage_range: tuple[float, float],
    threads: int = 1,
) -> dict[str, BlastHit]:
    use_losat = tools.losat is not None
    if not use_losat and (
        tools.blastn is None or tools.makeblastdb is None or tools.tblastn is None
    ):
        raise RuntimeError("NCBI BLAST+ executables blastn, makeblastdb, and tblastn are required")
    work_dir.mkdir(parents=True, exist_ok=True)
    target_fasta = work_dir / "target.fasta"
    db_prefix = work_dir / "target_db"
    target_changed = write_fasta(target_fasta, (("target", target_seq),), clean=True)
    db_changed = target_changed or not _blast_db_exists(db_prefix)
    if not use_losat and db_changed:
        logger.info("Building plastome BLAST database in %s", work_dir)
        run_command(
            [
                tools.makeblastdb,
                "-in",
                str(target_fasta),
                "-dbtype",
                "nucl",
                "-out",
                str(db_prefix),
            ],
            timeout=300,
        )
    hits: dict[str, BlastHit] = {}
    grouped = {group: [q for q in queries if q.group == group] for group in _query_groups()}
    for group, group_queries in grouped.items():
        if not group_queries:
            continue
        query_path = work_dir / f"{group}.fasta"
        out_path = work_dir / f"{group}.tsv"
        query_changed = write_fasta(
            query_path,
            ((q.query_id, q.sequence) for q in group_queries),
            clean=False,
        )
        protein_group = group not in {"reference1", "reference2"}
        if use_losat:
            cmd = _losat_group_command(
                tools.losat,
                query_path=query_path,
                target_fasta=target_fasta,
                out_path=out_path,
                protein=protein_group,
                threads=threads,
            )
        else:
            exe = tools.blastn if not protein_group else tools.tblastn
            task = "blastn" if not protein_group else "tblastn"
            cmd = [
                exe,
                "-task",
                task,
                "-query",
                str(query_path),
                "-db",
                str(db_prefix),
                "-outfmt",
                "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen",
                "-max_hsps",
                "1",
                "-max_target_seqs",
                "1",
                *_thread_args(threads),
                "-out",
                str(out_path),
            ]
            if task == "blastn":
                cmd[3:3] = ["-word_size", "7", "-dust", "no"]
            else:
                # Same reason as the LOSAT branch: keep transmembrane residues.
                cmd[3:3] = ["-seg", "no"]
        # Search parameters are part of the cache key: a changed command (e.g.
        # a new masking flag) must not reuse hits produced by the old one.
        cmd_path = out_path.with_suffix(".cmd")
        cmd_text = " ".join(str(part) for part in cmd if part != str(out_path))
        cmd_changed = not cmd_path.exists() or cmd_path.read_text() != cmd_text
        if (
            cmd_changed
            or (use_losat and (target_changed or query_changed))
            or (not use_losat and (db_changed or query_changed or not out_path.exists()))
            or not out_path.exists()
        ):
            logger.info(
                "Running %s for plastome query group %s with %s thread(s)",
                "losat " + ("tblastn" if protein_group else "blastn")
                if use_losat
                else ("tblastn" if protein_group else "blastn"),
                group,
                _normalize_threads(threads),
            )
            run_command(cmd, timeout=900)
            cmd_path.write_text(cmd_text)
        hits.update(parse_blast_hits(out_path, min_identity, qcoverage_range, group_queries))
    return hits


def _losat_group_command(
    losat: str,
    *,
    query_path: Path,
    target_fasta: Path,
    out_path: Path,
    protein: bool,
    threads: int,
) -> list[str]:
    """Build a LOSAT command for one plastome query group.

    LOSAT searches a FASTA subject directly (-s), so no makeblastdb step is
    needed. Its tabular output is the fixed NCBI default 12 columns; qlen is
    recovered from the query sequences during parsing. Scoring parameters for
    the nucleotide groups mirror NCBI `-task blastn` defaults explicitly
    because the LOSAT binary defaults to megablast scoring.
    """
    if protein:
        return [
            losat,
            "tblastn",
            # SEG masks the hydrophobic transmembrane stretches that make up most
            # of the small photosystem proteins (psbJ keeps 11 of 40 residues),
            # dropping their query coverage below the transfer threshold.
            "--seg=false",
            "-q",
            str(query_path),
            "-s",
            str(target_fasta),
            "--max-target-seqs",
            "1",
            *_losat_thread_args(threads),
            "-o",
            str(out_path),
        ]
    return [
        losat,
        "blastn",
        "--task",
        "blastn",
        # Equals form: clap would parse a bare "-3" as a flag.
        "--reward=2",
        "--penalty=-3",
        "--gap-open=5",
        "--gap-extend=2",
        "--word-size",
        "7",
        "-q",
        str(query_path),
        "-s",
        str(target_fasta),
        "--max-target-seqs",
        "1",
        *_losat_thread_args(threads),
        "-o",
        str(out_path),
    ]


def _losat_thread_args(threads: int) -> list[str]:
    return ["-n", str(_normalize_threads(threads))]


def detect_ir_regions(
    target_seq: str,
    *,
    work_dir: Path,
    tools: BlastTools,
    min_ir_length: int,
    threads: int = 1,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Inverted repeats by self-search; a LOSAT failure falls back to NCBI blastn when installed."""
    try:
        return _detect_ir_regions(target_seq, work_dir=work_dir, tools=tools, min_ir_length=min_ir_length,
                                  threads=threads)
    except OrganelleExecutionError as exc:
        fallback = _ncbi_fallback(tools, need_tblastn=False)
        if fallback is None:
            raise
        logger.warning("LOSAT failed (%s); repeating the IR self-search with NCBI blastn", exc)
        return _detect_ir_regions(target_seq, work_dir=work_dir, tools=fallback, min_ir_length=min_ir_length,
                                  threads=threads)


def _detect_ir_regions(
    target_seq: str,
    *,
    work_dir: Path,
    tools: BlastTools,
    min_ir_length: int,
    threads: int = 1,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    use_losat = tools.losat is not None
    if not use_losat and (tools.blastn is None or tools.makeblastdb is None):
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    doubled = target_seq + target_seq
    fasta = work_dir / "ir_target.fasta"
    db_prefix = work_dir / "ir_db"
    out_path = work_dir / "ir_self.tsv"
    target_changed = write_fasta(fasta, (("target2x", doubled),), clean=True)
    db_changed = target_changed or not _blast_db_exists(db_prefix)
    if not use_losat and db_changed:
        logger.info("Building plastome IR BLAST database in %s", work_dir)
        run_command(
            [tools.makeblastdb, "-in", str(fasta), "-dbtype", "nucl", "-out", str(db_prefix)],
            timeout=300,
        )
    if use_losat:
        if target_changed or not out_path.exists():
            logger.info(
                "Running plastome IR self-search with LOSAT blastn (%s thread(s))",
                _normalize_threads(threads),
            )
            run_command(
                [
                    tools.losat,
                    "blastn",
                    "--task",
                    "blastn",
                    "--reward=2",
                    "--penalty=-3",
                    "--gap-open=5",
                    "--gap-extend=2",
                    "--percent-identity",
                    "99",
                    "-q",
                    str(fasta),
                    "-s",
                    str(fasta),
                    *_losat_thread_args(threads),
                    "-o",
                    str(out_path),
                ],
                timeout=1800,
            )
    elif db_changed or not out_path.exists():
        logger.info("Running plastome IR self-BLAST with %s thread(s)", _normalize_threads(threads))
        run_command(
            [
                tools.blastn,
                "-task",
                "blastn",
                "-query",
                str(fasta),
                "-db",
                str(db_prefix),
                "-perc_identity",
                "99",
                "-outfmt",
                "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen",
                *_thread_args(threads),
                "-out",
                str(out_path),
            ],
            timeout=900,
        )
    best: tuple[int, int, int, int, int] | None = None
    genome_len = len(target_seq)
    for line in out_path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) not in {12, 14}:
            continue
        length = int(fields[3])
        qs, qe, ss, se = (int(fields[6]), int(fields[7]), int(fields[8]), int(fields[9]))
        if length in {genome_len, genome_len * 2}:
            continue
        if length < min_ir_length or qe > genome_len or se > genome_len or qs >= ss:
            continue
        if best is None or length > best[0]:
            best = (length, qs, qe, ss, se)
    if best is None:
        return None
    _, qs, qe, ss, se = best
    return ((qs - 1, qe, 1), (min(ss, se) - 1, max(ss, se), -1))


def parse_blast_hits(
    path: Path,
    min_identity: float,
    qcoverage_range: tuple[float, float],
    queries: list[ReferenceQuery],
) -> dict[str, BlastHit]:
    query_by_id = {query.query_id: query for query in queries}
    # LOSAT tabular output is the fixed NCBI default 12 columns (no qlen/slen
    # columns); the query length is identical to the sequence we wrote, so it
    # is recovered from the query map instead.
    qlen_by_id = {query.query_id: len(query.sequence) for query in queries}
    best: dict[str, BlastHit] = {}
    if not path.exists():
        return best
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) not in {12, 14}:
            continue
        qid = fields[0]
        query = query_by_id.get(qid)
        if query is None:
            continue
        pident = float(fields[2])
        length = int(fields[3])
        qstart = int(fields[6])
        qend = int(fields[7])
        sstart = int(fields[8])
        send = int(fields[9])
        evalue = float(fields[10])
        bitscore = float(fields[11])
        qlen = int(fields[12]) if len(fields) == 14 else qlen_by_id.get(qid, 0)
        if qlen <= 0:
            continue
        qcov = length / qlen
        gene = query.reference_feature.gene
        if pident / 100 < min_identity and gene not in {"ycf1", "ycf2"}:
            continue
        if not (qcoverage_range[0] <= qcov <= qcoverage_range[1]):
            continue
        hit = BlastHit(
            query_id=qid,
            pident=pident,
            qcov=qcov,
            start=min(sstart, send) - 1,
            end=max(sstart, send),
            strand=1 if sstart <= send else -1,
            bitscore=bitscore,
            evalue=evalue,
            align_length=length,
            qstart=qstart,
            qend=qend,
            qlen=qlen,
        )
        previous = best.get(qid)
        if previous is None or (hit.bitscore, hit.qcov, hit.pident) > (
            previous.bitscore,
            previous.qcov,
            previous.pident,
        ):
            best[qid] = hit
    return best


def write_fasta(path: Path, records: Iterable[tuple[str, str]], *, clean: bool) -> bool:
    text = _format_fasta(records, clean=clean)
    if path.exists() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


def _format_fasta(records: Iterable[tuple[str, str]], *, clean: bool) -> str:
    lines: list[str] = []
    for record_id, seq in records:
        sequence = clean_seq(seq) if clean else seq.strip().upper()
        if not sequence:
            continue
        lines.append(f">{record_id}")
        lines.extend(sequence[i : i + 80] for i in range(0, len(sequence), 80))
    return "\n".join(lines) + ("\n" if lines else "")


def _blast_db_exists(prefix: Path) -> bool:
    return any(prefix.parent.glob(f"{prefix.name}.*"))


def _thread_args(threads: int) -> list[str]:
    return ["-num_threads", str(_normalize_threads(threads))]


def _normalize_threads(threads: int) -> int:
    try:
        value = int(threads)
    except (TypeError, ValueError):
        return 1
    return max(1, value)


def _query_groups() -> tuple[str, str, str, str]:
    return ("reference1", "reference2", "reference3", "reference4")
