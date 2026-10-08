"""Submission-readiness validation of annotation output through NCBI table2asn.

The five-column TBL is written so that ``table2asn`` can consume it, but
until now nothing ever ran the consumer: an annotation whose features violate
GenBank submission rules (internal stop codons without ``/transl_except``,
out-of-range or frame-suspicious CDS, pseudo/product conflicts) looked
identical to a clean one. This module runs the real tool on the materialized
TBL plus a FASTA of the annotated sequence, parses its ``.val`` findings, and
separates feature-level errors — the ones that describe genes — from
record-level descriptor noise (missing publications, BioSource descriptors)
that only matters at actual deposit time.

The binary is optional, like every external tool here: missing means no
report, never a failed annotation. Resolution order mirrors LOSAT —
``ORG_VERSE_TABLE2ASN_BIN`` (with a ``none`` opt-out), then ``PATH``, then
the checkout's ``external_tools/table2asn/table2asn``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .mitochondrion.tbl import render_tbl_document
from .models import AnnotationDocument

_SCHEMA = "organelleverse.table2asn-validation.v1"
_LINE_RE = re.compile(r"^(Error|Warning|Info|Reject|Fatal):\s+valid\s+\[([^\]]+)\]\s*(.*)$")
_SEVERITY_ORDER = {"Fatal": 0, "Reject": 1, "Error": 2, "Warning": 3, "Info": 4}
_FASTA_WIDTH = 70


@dataclass(frozen=True)
class ValidationFinding:
    severity: str
    code: str
    message: str

    @property
    def feature_level(self) -> bool:
        """Feature findings describe genes; the rest is record/descriptor bookkeeping."""
        return self.code.startswith("SEQ_FEAT") or "FEATURE:" in self.message


def find_table2asn() -> str | None:
    """Locate the table2asn binary, or None when it is not provisioned."""
    env_value = os.environ.get("ORG_VERSE_TABLE2ASN_BIN")
    if env_value:
        if env_value.strip().casefold() in {"none", "off", "disable", "0"}:
            return None
        resolved = shutil.which(env_value)
        if resolved:
            return resolved
    found = shutil.which("table2asn")
    if found:
        return found
    candidate = Path(__file__).resolve().parents[3] / "external_tools" / "table2asn" / "table2asn"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def parse_val_text(text: str) -> list[ValidationFinding]:
    """Parse ``.val`` lines into findings; unparseable lines are dropped."""
    findings = []
    for line in text.splitlines():
        match = _LINE_RE.match(line.strip())
        if match:
            findings.append(ValidationFinding(*match.groups()))
    return sorted(findings, key=lambda f: (_SEVERITY_ORDER.get(f.severity, 99), f.code))


def write_submission_fasta(document: AnnotationDocument, path: str | Path) -> Path:
    """Write the FASTA table2asn pairs with the feature table.

    ``[organism=...]`` in the definition line suppresses the BioSource
    descriptor errors for records that know their species, which keeps the
    report focused on feature problems.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    organism = document.source_metadata.get("species") or document.source_metadata.get("organism")
    lines = []
    for record in document.records:
        definition = record.seqid
        if organism:
            definition = f"{definition} [organism={organism}] [moltype=genomic DNA]"
        lines.append(f">{definition}")
        sequence = record.sequence.upper()
        lines += [sequence[i : i + _FASTA_WIDTH] for i in range(0, len(sequence), _FASTA_WIDTH)]
    destination.write_text("\n".join(lines) + "\n")
    return destination


def _tool_version(binary: str) -> str:
    try:
        out = subprocess.run(
            [binary, "-version"], capture_output=True, text=True, timeout=60, check=False
        )
        return (out.stdout + out.stderr).strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def run_table2asn_validation(
    document: AnnotationDocument,
    work_dir: str | Path,
    *,
    binary: str | None = None,
    timeout: int = 900,
) -> dict | None:
    """Validate the document's TBL rendering with table2asn; None when unavailable.

    Runs in ``work_dir`` (created), leaves ``annotation.fsa``/``annotation.tbl``
    inputs there, and returns a JSON-serializable report. The verdict fields
    are ``feature_errors``/``feature_findings``; record-level findings stay in
    ``findings`` for completeness.
    """
    executable = binary or find_table2asn()
    if executable is None:
        return None
    scratch = Path(work_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    fsa = write_submission_fasta(document, scratch / "annotation.fsa")
    tbl = scratch / "annotation.tbl"
    render_tbl_document(document, tbl)
    output = scratch / "annotation.sqn"
    command = [
        executable,
        "-i",
        str(fsa),
        "-f",
        str(tbl),
        "-o",
        str(output),
        "-V",
        "vb",
        "-outdir",
        str(scratch),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    val_path = scratch / "annotation.val"
    findings = parse_val_text(val_path.read_text()) if val_path.is_file() else []
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    feature_findings = [f for f in findings if f.feature_level]
    return {
        "schema_version": _SCHEMA,
        "tool_version": _tool_version(executable),
        "command": command,
        "returncode": completed.returncode,
        "counts": counts,
        "feature_error_count": sum(
            1 for f in feature_findings if f.severity in {"Error", "Reject", "Fatal"}
        ),
        "feature_findings": [
            {"severity": f.severity, "code": f.code, "message": f.message} for f in feature_findings
        ],
        "findings": [
            {"severity": f.severity, "code": f.code, "message": f.message} for f in findings
        ],
        "stdout_tail": (completed.stdout or "").strip().splitlines()[-20:],
        "stderr_tail": (completed.stderr or "").strip().splitlines()[-20:],
    }


def write_validation_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(
        json.dumps(report, sort_keys=True, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
    )
    return destination
