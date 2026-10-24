"""Materialization of ``localization`` results as plain tables."""

from __future__ import annotations

from ..materialize import make_materializer

materialize_result = make_materializer("localization", generic=True)

__all__ = ["materialize_result"]
