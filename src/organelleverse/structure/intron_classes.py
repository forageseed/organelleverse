"""Group I and group II introns in an organelle genome, by Rfam covariance models.

Following Brown et al. 2026, who found the number of group I/II introns the
most consistent correlate of plant mitogenome size, the whole genome is
searched with Rfam RF00028 (Intron_gpI) and RF00029 (Intron_gpII, domains
V-VI) using Infernal ``cmsearch --cut_ga`` (each family's curated gathering
threshold). Unlike ``structure.introns``, which counts the ``intron`` features
an annotation happens to record, this works on bare sequence and separates
the two classes.

A hit is one intron's conserved core: RF00029 models domains V-VI, so a group
II intron is counted once wherever its 3' end lies, including the pieces of
trans-spliced introns that keep domain V-VI. OV's native CYK search is far too
slow for whole genomes, so Infernal is required.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from organelleverse._public_validation import validated_public

from ..core.errors import OrganelleDependencyError, OrganelleInputError, OrganelleParameterError
from ..core.result import Finding, OrganelleResult
from ..sequences import Sequences, as_path
from .structure import _OPERATION_VERSION, _provenance, _sha256_file

__all__ = ["intron_classes", "parse_tblout"]

_DATA = Path(__file__).resolve().parent.parent / "annotation" / "data"
MODELS = {
    "group_I": _DATA / "models" / "introns" / "RF00028.cm",
    "group_II": _DATA / "plastome" / "group_ii" / "RF00029.cm",
}


def parse_tblout(text: str) -> list[dict[str, object]]:
    """Rows of an Infernal ``--tblout`` table that pass the inclusion threshold (``!``)."""
    hits = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 17 or fields[16] != "!":
            continue
        start, end = int(fields[7]), int(fields[8])
        hits.append(
            {
                "sequence_id": fields[0],
                "start": min(start, end),
                "end": max(start, end),
                "strand": fields[9],
                "bits": float(fields[14]),
                "evalue": float(fields[15]),
            }
        )
    return sorted(hits, key=lambda hit: (str(hit["sequence_id"]), int(hit["start"])))


@validated_public("structure.intron_classes")
def intron_classes(
    genome_fasta: str | Path | Sequences,
    *,
    threads: int = 1,
) -> OrganelleResult:
    """Count group I and group II introns in an organelle genome with Rfam models.

    Runs Infernal ``cmsearch --cut_ga`` with RF00028 (group I) and RF00029
    (group II domains V-VI) over every sequence of ``genome_fasta``. Reports
    ``group_I_count``, ``group_II_count``, their sum ``intron_count`` and each
    hit (sequence, 1-based span, strand, bit score, E-value).

    Needs Infernal (``cmsearch``) on PATH; it is not replaced by a slower
    search when missing. The group I scan dominates the run time (tens of
    seconds for a 400 kb mitogenome).
    """
    if threads < 1:
        raise OrganelleParameterError(code="structure.intron_threads", message="threads must be >= 1")
    cmsearch = shutil.which("cmsearch")
    if cmsearch is None:
        raise OrganelleDependencyError(
            code="structure.cmsearch_missing",
            message="intron_classes needs Infernal cmsearch",
            suggested_action={"install": "mamba install -c bioconda infernal"},
        )
    genome_fasta = as_path(genome_fasta)
    fasta = Path(genome_fasta)
    if not fasta.is_file():
        raise OrganelleInputError(code="structure.missing_fasta", message=f"{fasta} does not exist")
    started_at = datetime.now(UTC)
    found: dict[str, list[dict[str, object]]] = {}
    version = ""
    for label, model in MODELS.items():
        argv = [
            cmsearch, "--cut_ga", "--noali", "--cpu", str(threads),
            "--tblout", "/dev/stdout", "-o", "/dev/null", str(model), str(fasta),
        ]
        process = subprocess.run(argv, capture_output=True, text=True)
        if process.returncode != 0:
            raise OrganelleInputError(
                code="structure.cmsearch_failed",
                message=process.stderr.strip()[-1000:] or "cmsearch failed",
                details={"argv": argv},
            )
        found[label] = parse_tblout(process.stdout)
        for line in process.stdout.splitlines():
            if line.startswith("# Program:") or line.startswith("# Version:"):
                version = version or line.split(":", 1)[1].strip()
    group_i, group_ii = len(found["group_I"]), len(found["group_II"])
    return OrganelleResult(
        operation_id="structure.intron_classes",
        operation_version=_OPERATION_VERSION,
        scope="mitochondrion",
        status="ok",
        summary_text=f"{group_i} group I and {group_ii} group II introns (Rfam RF00028/RF00029, cmsearch --cut_ga).",
        metrics={
            "group_I_count": group_i,
            "group_II_count": group_ii,
            "intron_count": group_i + group_ii,
            "group_I_hits": found["group_I"],
            "group_II_hits": found["group_II"],
            "models": {label: path.name for label, path in MODELS.items()},
        },
        findings=(
            Finding(code="structure.group_I_count", metric="group_I_count", value=group_i),
            Finding(code="structure.group_II_count", metric="group_II_count", value=group_ii),
        ),
        flags=(),
        provenance=_provenance(
            "structure.intron_classes",
            parameters={"threads": threads, "cmsearch": version or cmsearch},
            input_artifact_hashes=(_sha256_file(fasta),),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
    )
