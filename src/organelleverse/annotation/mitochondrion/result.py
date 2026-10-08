"""Mitochondrial annotation data models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cds import RejectedCDSCandidate


@dataclass(slots=True)
class MitochondrialAnnotationStats:
    sample_name: str
    output_paths: tuple[Path, ...]
    warnings: tuple[str, ...]
    contig_count: int
    cds_predicted: int
    trna_predicted: int
    rrna_predicted: int
    missing_core_genes: tuple[str, ...]
    invalid_cds: int
    rejected_cds_candidates: tuple[RejectedCDSCandidate, ...] = ()

    @property
    def features_total(self) -> int:
        return self.cds_predicted + self.trna_predicted + self.rrna_predicted
