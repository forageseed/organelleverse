"""Debug output for clean-room tRNA candidate decisions."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from .models import TRNACall

FIELDS = [
    "gene",
    "start",
    "end",
    "strand",
    "anticodon",
    "source",
    "confidence",
    "passed",
    "reason",
    "total",
    "sequence_score",
    "structure_score",
    "covariance_score",
    "anticodon_score",
    "pseudogene_penalty",
    "overcall_penalty",
]


def write_debug_candidates(path: Path, calls: Iterable[TRNACall]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for call in calls:
            candidate = call.candidate
            components = call.score.components
            writer.writerow(
                {
                    "gene": candidate.gene_name,
                    "start": str(candidate.start),
                    "end": str(candidate.end),
                    "strand": "+" if candidate.strand == 1 else "-",
                    "anticodon": candidate.anticodon,
                    "source": candidate.source,
                    "confidence": call.decision.confidence,
                    "passed": "true" if call.decision.passed else "false",
                    "reason": call.decision.reason,
                    "total": f"{components['total']:g}",
                    "sequence_score": f"{components['sequence_score']:g}",
                    "structure_score": f"{components['structure_score']:g}",
                    "covariance_score": f"{components['covariance_score']:g}",
                    "anticodon_score": f"{components['anticodon_score']:g}",
                    "pseudogene_penalty": f"{components['pseudogene_penalty']:g}",
                    "overcall_penalty": f"{components['overcall_penalty']:g}",
                }
            )
