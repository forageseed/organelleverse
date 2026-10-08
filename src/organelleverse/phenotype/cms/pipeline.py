"""CMS (cytoplasmic male sterility) candidate gene prediction. Self-contained.

Scans a mitochondrial genome FASTA for novel ORFs matching CMS candidate
screening criteria: length >= threshold and a hydrophobic stretch.
This screen does not test chimerism or establish CMS causality.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..._bio import read_fasta
from ..._sequtil import reverse_complement
from ...annotation.research import _predict_orfs
from ...core.provenance import ResultProvenance
from ...core.result import Finding, OrganelleResult
from ...selection.kaks import _CODONS as _STD_CODONS

_OPERATION_VERSION = "1.0"


def cms(
    genome_fasta: str | Path,
    *,
    min_orf_aa: int = 50,
    tm_window: int = 21,
    tm_threshold: float = 0.5,
) -> OrganelleResult:
    """Predict CMS candidate genes from a mitochondrial FASTA.

    A candidate is a long ORF (>= min_orf_aa) with a hydrophobic stretch
    (transmembrane hint). No chimerism, homology, or CMS causality is tested. Compute-only;
    persist candidates through an explicit writer (e.g. ``ov.write``).
    """
    seqs = read_fasta(Path(genome_fasta))
    candidates = _screen_candidates(seqs, min_orf_aa, tm_window, tm_threshold)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return OrganelleResult(
        operation_id="phenotype.cms",
        operation_version=_OPERATION_VERSION,
        scope="mitochondrion",
        status="ok",
        summary_text=f"{len(candidates)} CMS candidate ORFs.",
        metrics={
            "candidates": len(candidates),
            "candidate_table": candidates,
            "interpretation": "Length and hydrophobicity screen only; CMS association is untested.",
        },
        findings=tuple(
            Finding(code="cms_candidate", metric=c["id"], value=c["score"]) for c in candidates[:5]
        ),
        flags=("hydrophobicity_screen_only",) + (("cms_candidates_found",) if candidates else ()),
        provenance=_provenance(
            "cms",
            parameters={
                "min_orf_aa": min_orf_aa,
                "tm_window": tm_window,
                "tm_threshold": tm_threshold,
            },
            backend="organelleverse",
        ),
    )


def _screen_candidates(
    records: list[tuple[str, str]], min_orf_aa: int, tm_window: int, tm_threshold: float
) -> list[dict[str, object]]:
    """Use one coordinate and strand interpretation for both CMS entry points."""
    candidates = []
    for contig_id, seq in records:
        for orf in _predict_orfs(contig_id, seq.upper(), min_orf_aa):
            sub = seq[orf.start - 1 : orf.end]
            if orf.strand == "-":
                sub = reverse_complement(sub)
            aa = _translate(sub)
            if len(aa) >= min_orf_aa and _has_tm(aa, tm_window, tm_threshold):
                candidates.append(
                    {
                        "id": orf.attributes.get("ID", ""),
                        "sequence_id": contig_id,
                        "start": orf.start,
                        "end": orf.end,
                        "strand": orf.strand,
                        "length_aa": len(aa),
                        "protein_sequence": aa,
                        "has_tm": True,
                        "tm_method": "kyte_doolittle_window",
                        "score": round(len(aa) / 100 + 2.0, 2),
                        "evidence": "hydrophobicity_screen_only",
                    }
                )
    return candidates


_CODON_TABLE = _STD_CODONS


def _translate(dna: str) -> str:
    aa = []
    for i in range(0, len(dna) - 2, 3):
        a = _CODON_TABLE.get(dna[i : i + 3].upper())
        if a is None:
            return ""
        if a == "*":
            break
        aa.append(a)
    return "".join(aa)


_KD = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}


def _has_tm(aa: str, window: int, threshold: float) -> bool:
    """Cheap transmembrane heuristic: sliding-window mean Kyte-Doolittle >= threshold."""
    if len(aa) < window:
        return False
    for i in range(len(aa) - window + 1):
        scores = [_KD.get(aa[j], 0.0) for j in range(i, i + window)]
        if sum(scores) / window >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# canonical contract adapters (I/O boundary only — no scientific logic here)
# ---------------------------------------------------------------------------


def _provenance(
    op: str,
    *,
    parameters: Mapping[str, Any],
    backend: str = "",
) -> ResultProvenance:
    """Build canonical provenance for one phenotype operation."""
    return ResultProvenance(
        operation_id=f"phenotype.{op}",
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=_sha256_json(dict(parameters)),
        requested_backend=backend,
        actual_backend=backend,
        attempted_backends=(backend,) if backend else (),
    )


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"
