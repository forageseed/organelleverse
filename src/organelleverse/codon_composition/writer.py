"""Materialization of ``codon_composition`` results: each operation writes with its own ``write_*``."""

from __future__ import annotations

from ..materialize import make_materializer


def _writers():
    from .codon import write_amino_acid, write_usage

    return {
        "codon_composition.codon_usage": write_usage,
        "codon_composition.amino_acid": write_amino_acid,
    }


materialize_result = make_materializer("codon_composition", _writers)

__all__ = ["materialize_result"]
