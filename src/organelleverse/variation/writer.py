"""Materialization of ``variation`` results: each operation writes with its own ``write_*``."""

from __future__ import annotations

from ..materialize import make_materializer


def _writers():
    from .snp import write_snp, write_snp_density

    return {
        "variation.snp": write_snp,
        "variation.snp_density": write_snp_density,
    }


materialize_result = make_materializer("variation", _writers)

__all__ = ["materialize_result"]
