"""Materialization of ``pangenome`` results: each operation writes with its own ``write_*``."""

from __future__ import annotations

from ..materialize import make_materializer


def _writers():
    from .pangenome import write_pav

    return {
        "pangenome.gene_pav": write_pav,
    }


materialize_result = make_materializer("pangenome", _writers)

__all__ = ["materialize_result"]
