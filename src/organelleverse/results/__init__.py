"""Result DAG and canonical output-path surfaces (spec: docs/superpowers/
specs/2026-08-15-result-dag-output-paths-design.md)."""

from __future__ import annotations

from .dag import DagEdge, DagNode, ResultDag, build_result_dag
from .paths import (
    PathSpecError,
    is_canonical,
    relative_address,
)

__all__ = [
    "DagEdge",
    "DagNode",
    "PathSpecError",
    "ResultDag",
    "build_result_dag",
    "is_canonical",
    "relative_address",
]
