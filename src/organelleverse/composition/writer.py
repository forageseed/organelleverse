"""Materialization of ``composition`` results: each operation writes with its own ``write_*``."""

from __future__ import annotations

from ..materialize import make_materializer


def _writers():
    from .gc import write_content

    return {
        "composition.gc_content": write_content,
    }


materialize_result = make_materializer("composition", _writers)

__all__ = ["materialize_result"]
