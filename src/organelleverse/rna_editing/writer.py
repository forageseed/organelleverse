"""Materialization of ``rna_editing`` results: each operation writes with its own ``write_*``."""

from __future__ import annotations

from ..materialize import make_materializer


def _writers():
    from .rna_edit import write_sites

    return {
        "rna_editing.predict_edits": write_sites,
        "rna_editing.predict_edits_deepred": write_sites,
        "rna_editing.predict_edits_plantc2u": write_sites,
    }


materialize_result = make_materializer("rna_editing", _writers)

__all__ = ["materialize_result"]
