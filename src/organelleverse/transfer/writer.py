"""Materialization of ``transfer`` results: each operation writes with its own ``write_*``."""

from __future__ import annotations

from ..materialize import make_materializer


def _writers():
    from .transfer import write_fragments

    return {
        "transfer.detect_mtpt": write_fragments,
    }


materialize_result = make_materializer("transfer", _writers)

__all__ = ["materialize_result"]
