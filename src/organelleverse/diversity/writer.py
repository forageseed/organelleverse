"""Materialization of ``diversity`` results: each operation writes with its own ``write_*``."""

from __future__ import annotations

from ..materialize import make_materializer


def _writers():
    from .diversity import write_neutral_tests, write_nucleotide, write_sliding_window_diversity

    return {
        "diversity.nucleotide_diversity": write_nucleotide,
        "diversity.neutral_tests": write_neutral_tests,
        "diversity.sliding_window_diversity": write_sliding_window_diversity,
    }


materialize_result = make_materializer("diversity", _writers)

__all__ = ["materialize_result"]
