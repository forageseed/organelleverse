"""Shared LOSAT (vendored Rust BLAST-compatible) resolution and invocation.

Modules that BLAST prefer LOSAT's direct FASTA search over NCBI BLAST+
where the command surface overlaps: no database build step (``--query`` /
``--subject`` FASTA directly), one binary, subcommands ``blastn`` /
``blastp`` / ``tblastn`` / ``tblastx``. Output is the fixed NCBI default
12-column tabular layout::

    qseqid sseqid pident length mismatch gapopen
    qstart qend sstart send evalue bitscore

Callers wanting columns outside the 12 (qlen/slen/qcovs/sframe) derive them
from the FASTA inputs, the way the annotation plastome backend does: query
length from the query FASTA record, coverage from qstart/qend, and the
subject strand from the sstart/send order (tblastn ``sframe`` sign).

Scoring presets mirror NCBI task defaults and are selected with the same
``--task`` vocabulary (``megablast`` is the default for ``blastn`` on both
sides). Every invocation runs through :mod:`organelleverse.core.external`
(``run_external``), or through a managed ``CommandRunner`` when one is given,
so failures carry the same stderr-rich error shape as other externals.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .core.errors import OrganelleDependencyError, OrganelleExecutionError
from .core.external import run_external

__all__ = [
    "LOSAT_COLUMNS",
    "fasta_sequence_lengths",
    "losat_from_tool_paths",
    "query_coverage_percent",
    "resolve_losat",
    "run_losat",
    "run_losat_blastn",
    "run_losat_with_query_coverage",
    "union_query_coverage",
]

#: Fixed NCBI default 12-column tabular layout emitted by LOSAT.
LOSAT_COLUMNS: tuple[str, ...] = (
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
)

#: Explicit opt-out tokens: the checkout otherwise always finds its vendored
#: build, so A/B runs against NCBI BLAST+ need a hard switch.
_DISABLED_TOKENS = {"none", "off", "disable", "ncbi", "0"}

#: LOSAT defaults to megablast scoring (reward=1 penalty=-2), matching the
#: NCBI ``blastn`` executable's default task, so a bare LOSAT ``blastn`` run is
#: behavior-aligned with plain ``blastn -query .. -db ..`` runs.
_DEFAULT_WORD_SIZE = "28"


#: Programs whose query is nucleotide: LOSAT reverse-complements it for the minus strand.
_NUCLEOTIDE_QUERY_PROGRAMS = {"blastn", "tblastx"}


def _has_lowercase_sequence(path: str | Path) -> bool:
    """True when a FASTA sequence line has lower-case bases (unreadable paths: False, LOSAT reports them)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith(">") and any(ch.islower() for ch in line):
                    return True
    except OSError:
        return False
    return False


@contextmanager
def _uppercase_query(program: str, query: str | Path) -> Iterator[str | Path]:
    """Yield a query path LOSAT can search on both strands.

    LOSAT's ``reverse_complement`` (``blastn/lookup.rs``) maps lower-case bases
    to the gap code, so a lower-case or soft-masked nucleotide query silently
    loses every minus-strand hit: on the rice mitochondrion (lower-case FASTA)
    against the rice nuclear genome LOSAT reported 776 HSPs where NCBI blastn
    reported 1,538, including a 46.7 kb, 99.9% identical reverse-strand NUMT.
    NCBI treats case as plain sequence (no ``-lcase_masking``), so the search
    runs on an upper-cased copy, which is removed afterwards.
    """
    if program not in _NUCLEOTIDE_QUERY_PROGRAMS or not _has_lowercase_sequence(query):
        yield query
        return
    with tempfile.TemporaryDirectory(prefix="losat_upper_") as tmpdir:
        upper = Path(tmpdir) / "query_upper.fasta"
        with open(query, encoding="utf-8", errors="replace") as source, open(
            upper, "w", encoding="utf-8", newline="\n"
        ) as target:
            for line in source:
                target.write(line if line.startswith(">") else line.upper())
        yield upper


def resolve_losat() -> str | None:
    """Resolve the LOSAT binary: env override, then checkout, then PATH."""
    env_value = os.environ.get("ORG_VERSE_LOSAT_BIN")
    if env_value:
        if env_value.strip().casefold() in _DISABLED_TOKENS:
            return None
        resolved = shutil.which(env_value) or (
            str(Path(env_value)) if Path(env_value).is_file() else None
        )
        if resolved:
            return resolved
    candidate = (
        Path(__file__).resolve().parents[2]
        / "external_tools"
        / "LOSAT"
        / "LOSAT"
        / "target"
        / "release"
        / "LOSAT"
    )
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("losat") or shutil.which("LOSAT")


def losat_from_tool_paths(tool_paths: Mapping[str, str] | None) -> str | None:
    """Pick the LOSAT binary for one BLAST call site.

    Managed runs (a ``tool_paths`` mapping) trust the backend preflight: the
    mapping either carries the resolved ``losat`` path or deliberately omits
    it after the NCBI fallback was chosen. Unmanaged runs (``tool_paths`` is
    ``None``) resolve LOSAT from the environment / vendored checkout, the
    same way the hgt/transfer call sites do.
    """
    if tool_paths is not None:
        return tool_paths.get("losat")
    return resolve_losat()


def fasta_sequence_lengths(path: str | Path) -> dict[str, int]:
    """Map every FASTA record id (first header token) to its sequence length.

    Used to recover the qlen/slen columns that LOSAT's fixed 12-column output
    does not emit.
    """
    lengths: dict[str, int] = {}
    record_id: str | None = None
    size = 0
    # bytes, not text: a 380 MB nuclear genome takes ~4x longer to decode line by line
    with open(path, "rb") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(b">"):
                if record_id is not None:
                    lengths[record_id] = size
                words = line[1:].split()
                record_id = words[0].decode("utf-8", errors="replace") if words else ""
                size = 0
            else:
                size += len(line)
    if record_id is not None:
        lengths[record_id] = size
    return lengths


def query_coverage_percent(qstart: int, qend: int, qlen: int) -> float:
    """Recover NCBI ``qcovs`` for a single-HSP tabular row.

    ``qcovs`` is the percent of the query covered by the alignment; for the
    single-HSP hits these call sites select, the covered span is
    ``qend - qstart + 1``.
    """
    if qlen <= 0:
        return 0.0
    return (abs(qend - qstart) + 1) * 100.0 / qlen


def union_query_coverage(
    rows: list[list[str]],
    qlen_by_id: Mapping[str, int],
) -> dict[str, float]:
    """Recover NCBI ``qcovs`` (query coverage per subject) for LOSAT rows.

    NCBI counts each query position once across every HSP the subject
    reports; mirror that by merging the qstart..qend spans of all rows that
    share a query id before dividing by the query length.
    """
    spans: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        if len(row) < 12:
            continue
        try:
            qstart, qend = int(row[6]), int(row[7])
        except ValueError:
            continue
        spans.setdefault(row[0], []).append((min(qstart, qend), max(qstart, qend)))
    coverage: dict[str, float] = {}
    for qid, intervals in spans.items():
        qlen = qlen_by_id.get(qid, 0)
        if qlen <= 0:
            continue
        merged = 0
        ordered = sorted(intervals)
        start, end = ordered[0]
        for next_start, next_end in ordered[1:]:
            if next_start <= end + 1:
                end = max(end, next_end)
            else:
                merged += end - start + 1
                start, end = next_start, next_end
        merged += end - start + 1
        coverage[qid] = merged * 100.0 / qlen
    return coverage


def run_losat(
    program: str,
    query: str | Path,
    subject: str | Path,
    *,
    evalue: float = 10.0,
    task: str | None = None,
    word_size: int | None = None,
    reward: int | None = None,
    penalty: int | None = None,
    gap_open: int | None = None,
    gap_extend: int | None = None,
    percent_identity: float | None = None,
    max_target_seqs: int | None = None,
    threads: int | None = None,
    seg: bool | None = None,
    timeout: int = 300,
    stage: str = "losat",
    command_runner=None,
    executable: str | None = None,
) -> list[list[str]]:
    """Run one LOSAT search and split the fixed 12 output columns.

    ``program`` is one of ``blastn``/``blastp``/``tblastn``/``tblastx``.
    Scoring options (``task``/``word_size``/``reward``/...) are forwarded
    only when set, so program-specific flags stay the caller's choice.

    ``executable`` is the LOSAT binary the call site already resolved (for
    managed runs, the backend preflight's ``tool_paths["losat"]``); when omitted
    the binary is resolved from the environment / vendored checkout.

    With a managed ``command_runner`` the invocation is recorded as evidence
    under ``stage``; otherwise :func:`organelleverse.core.external.run_external`
    executes it. Raises ``OrganelleDependencyError`` when LOSAT is unavailable;
    execution failures propagate as
    :class:`~organelleverse.core.errors.OrganelleExecutionError` (the same
    type the NCBI branches raise), so per-site fallback handling stays uniform.
    """
    if program == "tblastn" and any(
        value is not None
        for value in (task, word_size, reward, penalty, gap_open, gap_extend, percent_identity)
    ):
        raise ValueError("LOSAT tblastn does not support nucleotide task/scoring options")
    losat = executable or resolve_losat()
    if losat is None:
        raise OrganelleDependencyError(
            code="dependency_missing",
            message="Install LOSAT and set ORG_VERSE_LOSAT_BIN to its executable",
            details={"missing": ["losat"]},
        )
    stack = ExitStack()
    query = stack.enter_context(_uppercase_query(program, query))
    argv: list[str] = [losat, program]
    if program == "tblastn":
        argv += ["--db-gencode", "1"]
    if task is not None:
        argv += ["--task", task]
    if word_size is not None:
        argv += ["--word-size", str(word_size)]
    if reward is not None:
        argv += ["--reward", str(reward)]
    if penalty is not None:
        # clap otherwise treats negative scores as a new option.
        argv += [f"--penalty={penalty}"]
    if gap_open is not None:
        argv += ["--gap-open", str(gap_open)]
    if gap_extend is not None:
        argv += ["--gap-extend", str(gap_extend)]
    argv += ["--evalue", str(evalue)]
    if percent_identity is not None:
        argv += ["--percent-identity", str(percent_identity)]
    if max_target_seqs is not None:
        argv += ["--max-target-seqs", str(max_target_seqs)]
    if seg is not None and program == "tblastn":
        # SEG masks the hydrophobic helices of small membrane proteins (atp9).
        argv += [f"--seg={'true' if seg else 'false'}"]
    if threads is not None:
        argv += ["-n", str(threads)]
    argv += ["-q", str(query), "-s", str(subject)]

    try:
        if command_runner is not None:
            evidence = command_runner.run(tuple(argv), stage=stage, timeout=timeout)
            output = evidence.stdout
        else:
            completed = run_external(
                argv,
                timeout=timeout,
                code=f"losat_{program}_failed",
                tool="losat",
            )
            output = completed.stdout
    finally:
        stack.close()
    return [line.split("\t") for line in output.splitlines() if line.strip()]


def run_losat_with_query_coverage(
    program: str,
    query: str | Path,
    subject: str | Path,
    **kwargs,
) -> list[list[str]]:
    """Return the 12 LOSAT fields plus NCBI's integer ``qcovs``.

    Packaged mitochondrial references contain repeated IDs with different
    sequences and lengths. Search those records in separate batches with
    unique IDs within each batch, so each hit has an unambiguous query length.
    No query names are rewritten; results retain the original record order.
    Every batch searches the same subject with the same search parameters.
    Mitochondrial callers supply a single genome sequence as the subject.
    """
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord

    records = list(SeqIO.parse(query, "fasta"))
    occurrences: dict[str, int] = {}
    batches: list[list[tuple[int, SeqRecord]]] = []
    for index, record in enumerate(records):
        batch_index = occurrences.get(record.id, 0)
        occurrences[record.id] = batch_index + 1
        if batch_index == len(batches):
            batches.append([])
        batches[batch_index].append((index, record))

    by_record: list[list[list[str]]] = [[] for _ in records]
    with tempfile.TemporaryDirectory(prefix="losat_queries_") as tmpdir:
        for batch_index, batch in enumerate(batches):
            batch_query = Path(query)
            if len(batches) > 1:
                batch_query = Path(tmpdir) / f"queries_{batch_index}.fasta"
                SeqIO.write([record for _, record in batch], batch_query, "fasta")
            rows = run_losat(program, batch_query, subject, **kwargs)
            lengths = {record.id: len(record.seq) for _, record in batch}
            coverage = union_query_coverage(rows, lengths)
            indices = {record.id: index for index, record in batch}
            for row in rows:
                if len(row) < 12:
                    continue
                qcovs = int(coverage[row[0]] + 0.5)
                by_record[indices[row[0]]].append([*row, str(qcovs)])
    return [row for rows in by_record for row in rows]


def run_losat_blastn(
    query: str | Path,
    subject: str | Path,
    *,
    evalue: float = 10.0,
    word_size: int = 28,
) -> list[list[str]]:
    """Run LOSAT blastn (default/megablast scoring) and split the 12 columns.

    Raises OrganelleDependencyError when unavailable, or RuntimeError on failure.
    """
    try:
        return run_losat(
            "blastn",
            query,
            subject,
            evalue=evalue,
            task="megablast",
            word_size=word_size or int(_DEFAULT_WORD_SIZE),
        )
    except OrganelleExecutionError as error:
        tail = error.details.get("stderr_tail", "") if error.details else ""
        raise RuntimeError(f"losat_blastn_failed: {tail or error.message}") from error
