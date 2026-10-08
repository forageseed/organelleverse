"""Variation suite: snp() + snp_density(). Backend: scikit-allel when available."""

from __future__ import annotations
from .snp import snp, snp_density, write_snp, write_snp_density
from .snp_core import compute_snp

__all__ = ["snp", "snp_density", "write_snp", "write_snp_density", "compute_snp"]
