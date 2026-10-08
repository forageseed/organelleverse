"""Alignment backends for HGT candidate detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile


@dataclass(frozen=True)
class HGTAlignment:
    """One donor-to-recipient local alignment candidate."""

    recipient_id: str
    donor_id: str
    recipient_start: int
    recipient_end: int
    donor_start: int
    donor_end: int
    strand: str
    length: int
    identity: float
    mapq: int | None
    method: str

    def as_dict(self) -> dict[str, object]:
        return {
            "recipient_id": self.recipient_id,
            "donor_id": self.donor_id,
            "recipient_start": self.recipient_start,
            "recipient_end": self.recipient_end,
            "donor_start": self.donor_start,
            "donor_end": self.donor_end,
            "strand": self.strand,
            "length": self.length,
            "identity": round(self.identity, 4),
            "mapq": self.mapq,
            "method": self.method,
        }


@dataclass(frozen=True)
class HGTBlastHit:
    """One BLASTN confirmation hit for an HGT candidate."""

    recipient_id: str
    donor_id: str
    recipient_start: int
    recipient_end: int
    donor_start: int
    donor_end: int
    length: int
    identity: float
    evalue: float
    bitscore: float

    def as_dict(self) -> dict[str, object]:
        return {
            "recipient_id": self.recipient_id,
            "donor_id": self.donor_id,
            "recipient_start": self.recipient_start,
            "recipient_end": self.recipient_end,
            "donor_start": self.donor_start,
            "donor_end": self.donor_end,
            "length": self.length,
            "identity": round(self.identity, 4),
            "evalue": self.evalue,
            "bitscore": self.bitscore,
        }


class HGTBackendError(RuntimeError):
    """Raised when an HGT alignment backend cannot run."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def run_hgt_alignments(
    donor_fasta: str | Path,
    recipient_fasta: str | Path,
    *,
    backend: str = "auto",
    min_len: int = 200,
    min_identity: float = 90.0,
    preset: str = "asm5",
    minimap2_path: str | Path | None = None,
) -> tuple[str, list[HGTAlignment]]:
    """Return filtered donor/recipient alignments from the requested backend."""
    selected = backend.lower()
    if selected == "mappy":
        return "mappy", _run_mappy_alignments(
            donor_fasta,
            recipient_fasta,
            min_len=min_len,
            min_identity=min_identity,
            preset=preset,
        )
    if selected in {"minimap2", "minimap2_cli"}:
        return "minimap2", _run_minimap2_cli(
            donor_fasta,
            recipient_fasta,
            min_len=min_len,
            min_identity=min_identity,
            preset=preset,
            minimap2_path=minimap2_path,
        )
    if selected != "auto":
        raise HGTBackendError("unsupported_hgt_backend", f"Unsupported HGT backend: {backend}")

    try:
        return "mappy", _run_mappy_alignments(
            donor_fasta,
            recipient_fasta,
            min_len=min_len,
            min_identity=min_identity,
            preset=preset,
        )
    except HGTBackendError:
        return "minimap2", _run_minimap2_cli(
            donor_fasta,
            recipient_fasta,
            min_len=min_len,
            min_identity=min_identity,
            preset=preset,
            minimap2_path=minimap2_path,
        )


def run_blastn_confirmation(
    donor_fasta: str | Path,
    recipient_fasta: str | Path,
    *,
    min_len: int = 200,
    min_identity: float = 90.0,
    blastn_path: str | Path | None = None,
    makeblastdb_path: str | Path | None = None,
) -> list[HGTBlastHit]:
    """Return BLASTN hits supporting donor/recipient sequence similarity.

    Prefers the vendored LOSAT ``blastn`` (no makeblastdb, FASTA subject)
    unless ``blastn_path`` is given or ``ORG_VERSE_LOSAT_BIN`` opts out;
    NCBI BLAST+ is the fallback. Both paths produce the same 10-field hits.
    """
    from .._losat import resolve_losat, run_losat_blastn

    if blastn_path is None:
        if resolve_losat() is not None:
            # LOSAT tabular is the fixed 12-column NCBI layout; this parser
            # wants the 10-field subset (drop the mismatch/gapopen columns).
            rows = run_losat_blastn(recipient_fasta, donor_fasta, evalue=10.0)
            losat_hits: list[HGTBlastHit] = []
            for cells in rows:
                if len(cells) < 12:
                    continue
                hit = _parse_blastn_line(
                    "\t".join(cells[i] for i in (0, 1, 2, 3, 6, 7, 8, 9, 10, 11))
                )
                if hit is None:
                    continue
                if hit.length < min_len or hit.identity < min_identity:
                    continue
                losat_hits.append(hit)
            return losat_hits

    blastn = str(blastn_path) if blastn_path is not None else shutil.which("blastn")
    makeblastdb = (
        str(makeblastdb_path) if makeblastdb_path is not None else shutil.which("makeblastdb")
    )
    if not blastn or not makeblastdb:
        raise HGTBackendError("blastn_missing", "blastn or makeblastdb executable was not found.")

    with tempfile.TemporaryDirectory(prefix="organelleverse_hgt_blast_") as tmp:
        db_prefix = str(Path(tmp) / "donor_db")
        db_run = subprocess.run(
            [makeblastdb, "-in", str(donor_fasta), "-dbtype", "nucl", "-out", db_prefix],
            check=False,
            capture_output=True,
            text=True,
        )
        if db_run.returncode != 0:
            raise HGTBackendError(
                "makeblastdb_failed",
                db_run.stderr.strip() or f"makeblastdb exited with status {db_run.returncode}",
            )

        blast_run = subprocess.run(
            [
                blastn,
                "-query",
                str(recipient_fasta),
                "-db",
                db_prefix,
                "-dust",
                "no",
                "-outfmt",
                "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if blast_run.returncode != 0:
        raise HGTBackendError(
            "blastn_failed",
            blast_run.stderr.strip() or f"blastn exited with status {blast_run.returncode}",
        )

    hits: list[HGTBlastHit] = []
    for line in blast_run.stdout.splitlines():
        if not line.strip():
            continue
        hit = _parse_blastn_line(line)
        if hit is None:
            continue
        if hit.length < min_len or hit.identity < min_identity:
            continue
        hits.append(hit)
    return hits


def _run_mappy_alignments(
    donor_fasta: str | Path,
    recipient_fasta: str | Path,
    *,
    min_len: int,
    min_identity: float,
    preset: str,
) -> list[HGTAlignment]:
    try:
        import mappy as mp
    except ImportError as exc:
        raise HGTBackendError("mappy_missing", "mappy is not installed.") from exc

    aligner = mp.Aligner(str(donor_fasta), preset=preset)
    if not aligner:
        raise HGTBackendError("mappy_index_failed", f"Could not index donor FASTA: {donor_fasta}")

    alignments: list[HGTAlignment] = []
    for recipient_id, sequence, _quality in mp.fastx_read(str(recipient_fasta)):
        for hit in aligner.map(sequence, cs=True):
            if not bool(getattr(hit, "is_primary", True)):
                continue
            block_len = int(hit.blen or (hit.q_en - hit.q_st))
            if block_len <= 0:
                continue
            identity = 100.0 * float(hit.mlen) / float(block_len)
            length = int(hit.q_en - hit.q_st)
            if length < min_len or identity < min_identity:
                continue
            alignments.append(
                HGTAlignment(
                    recipient_id=recipient_id,
                    donor_id=str(hit.ctg),
                    recipient_start=int(hit.q_st) + 1,
                    recipient_end=int(hit.q_en),
                    donor_start=int(hit.r_st) + 1,
                    donor_end=int(hit.r_en),
                    strand="+" if int(hit.strand) >= 0 else "-",
                    length=length,
                    identity=identity,
                    mapq=int(hit.mapq),
                    method="mappy_minimap2",
                )
            )
    return alignments


def _run_minimap2_cli(
    donor_fasta: str | Path,
    recipient_fasta: str | Path,
    *,
    min_len: int,
    min_identity: float,
    preset: str,
    minimap2_path: str | Path | None,
) -> list[HGTAlignment]:
    executable = str(minimap2_path) if minimap2_path is not None else shutil.which("minimap2")
    if not executable:
        raise HGTBackendError("minimap2_missing", "minimap2 executable was not found.")

    completed = subprocess.run(
        [executable, "-x", preset, "-c", str(donor_fasta), str(recipient_fasta)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise HGTBackendError(
            "minimap2_failed",
            completed.stderr.strip() or f"minimap2 exited with status {completed.returncode}",
        )

    alignments: list[HGTAlignment] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        alignment = _parse_paf_line(line)
        if alignment is None:
            continue
        if alignment.length < min_len or alignment.identity < min_identity:
            continue
        alignments.append(alignment)
    return alignments


def _parse_blastn_line(line: str) -> HGTBlastHit | None:
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 10:
        return None
    qstart, qend = int(fields[4]), int(fields[5])
    sstart, send = int(fields[6]), int(fields[7])
    return HGTBlastHit(
        recipient_id=fields[0],
        donor_id=fields[1],
        recipient_start=min(qstart, qend),
        recipient_end=max(qstart, qend),
        donor_start=min(sstart, send),
        donor_end=max(sstart, send),
        length=int(fields[3]),
        identity=float(fields[2]),
        evalue=float(fields[8]),
        bitscore=float(fields[9]),
    )


def _parse_paf_line(line: str) -> HGTAlignment | None:
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 12:
        return None
    tag_values = {field[:5]: field[5:] for field in fields[12:] if len(field) >= 5}
    if tag_values.get("tp:A:") not in {None, "P"}:
        return None
    block_len = int(fields[10])
    if block_len <= 0:
        return None
    identity = 100.0 * float(fields[9]) / float(block_len)
    return HGTAlignment(
        recipient_id=fields[0],
        donor_id=fields[5],
        recipient_start=int(fields[2]) + 1,
        recipient_end=int(fields[3]),
        donor_start=int(fields[7]) + 1,
        donor_end=int(fields[8]),
        strand=fields[4],
        length=int(fields[3]) - int(fields[2]),
        identity=identity,
        mapq=int(fields[11]),
        method="minimap2_cli",
    )
