"""Composition suite: gc_content(). Self-contained."""

from __future__ import annotations
from .gc import gc_content, write_content
from .gc_core import compute_gc_content

__all__ = ["gc_content", "write_content", "compute_gc_content"]
