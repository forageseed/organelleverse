"""Composition suite: gc_content(). Self-contained."""

from __future__ import annotations

from ..inputs import make_reader
from .gc import gc_content, write_content
from .gc_core import compute_gc_content

read = make_reader("composition", "fasta")

__all__ = ["read", "gc_content", "write_content", "compute_gc_content"]
