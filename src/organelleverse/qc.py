"""Public facade for annotation, assembly and read quality control."""

from .quality_control.api import (
    annotation,
    assembly,
    compare_assembly_to_reference,
    filter_long_reads,
    filter_short_reads,
    read_statistics,
    write,
)

__all__ = [
    "annotation",
    "assembly",
    "compare_assembly_to_reference",
    "filter_long_reads",
    "filter_short_reads",
    "read_statistics",
    "write",
]
