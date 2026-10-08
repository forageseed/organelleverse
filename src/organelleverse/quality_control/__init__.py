"""Annotation and assembly quality-control contracts, operations, and lazy API."""

from __future__ import annotations

from .api import (
    annotation,
    assembly,
    compare_assembly_to_reference,
    filter_long_reads,
    filter_short_reads,
    read_statistics,
    write,
)
from .operations import ANNOTATION_QC_SPEC, ASSEMBLY_QC_SPEC, QC_WRITE_SPEC

__all__ = [
    "ANNOTATION_QC_SPEC",
    "ASSEMBLY_QC_SPEC",
    "QC_WRITE_SPEC",
    "annotation",
    "assembly",
    "compare_assembly_to_reference",
    "filter_long_reads",
    "filter_short_reads",
    "read_statistics",
    "write",
]
