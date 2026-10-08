"""Immutable assembly-QC policy (version 5).

These are evidence-processing thresholds (mapping-quality floors, coverage depth
levels, junction support rules, HMM thresholds, k-mer settings), recorded in the
report and run manifest. They are *not* universal pass/fail thresholds for N50,
size, GC, circularity, mapping QV, or contig count. Changing a decision rule
requires a new policy version.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from organelleverse.core.base import StrictFrozenModel

from .contracts import FractionFloat, PosFloat, StrictNonNegInt, StrictPosInt


class QcPolicy(StrictFrozenModel[Literal["assembly_qc_policy"]]):
    """Frozen evidence-processing policy for assembly QC."""

    kind: Literal["assembly_qc_policy"] = "assembly_qc_policy"
    policy_version: str
    minimum_mapping_quality: StrictNonNegInt
    minimum_base_quality: StrictNonNegInt
    coverage_depth_levels: tuple[StrictNonNegInt, ...] = Field(min_length=1)
    coverage_window_bases: StrictPosInt
    minimum_base_error_support: StrictNonNegInt = 3
    minimum_base_error_depth: StrictPosInt = 5
    minimum_base_error_fraction: FractionFloat = 0.8
    minimum_junction_support: StrictNonNegInt
    minimum_contradiction_support: StrictNonNegInt
    minimum_anchor_bases: StrictNonNegInt
    maximum_anchor_bases: StrictPosInt
    error_cluster_window_bases: StrictPosInt
    marker_evalue: PosFloat
    marker_complete_fraction: FractionFloat = 0.8
    kmer_size: StrictPosInt
    log_capture_limit_bytes: StrictPosInt

    @property
    def schema_version(self) -> str:
        return self.policy_version


QC_POLICY_V5 = QcPolicy(
    policy_version="organelleverse.assembly-qc-policy.v5",
    minimum_mapping_quality=20,
    minimum_base_quality=20,
    coverage_depth_levels=(1, 5, 10),
    coverage_window_bases=1_000,
    minimum_base_error_support=3,
    minimum_base_error_depth=5,
    minimum_base_error_fraction=0.8,
    minimum_junction_support=3,
    minimum_contradiction_support=3,
    minimum_anchor_bases=250,
    maximum_anchor_bases=5_000,
    error_cluster_window_bases=50,
    marker_evalue=1e-5,
    marker_complete_fraction=0.8,
    kmer_size=21,
    log_capture_limit_bytes=1_048_576,
)


__all__ = ["QC_POLICY_V5", "QcPolicy"]
