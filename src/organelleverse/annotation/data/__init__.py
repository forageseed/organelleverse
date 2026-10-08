"""Packaged annotation reference data."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def annotation_data_dir() -> Path:
    """Return the packaged annotation data root."""
    return Path(str(files(__name__)))


def mitochondrion_data_dir() -> Path:
    """Return the packaged mitochondrial annotation data root."""
    return annotation_data_dir() / "mitochondrion"


def plastome_data_dir() -> Path:
    """Return the packaged plastome annotation data root."""
    return annotation_data_dir() / "plastome"


def plastome_reference_dir() -> Path:
    """Return the packaged plastome GenBank reference directory."""
    return plastome_data_dir() / "references"


def plastome_reference_manifest() -> Path:
    """Return the packaged plastome reference-set manifest."""
    return plastome_data_dir() / "reference_set.json"


__all__ = [
    "annotation_data_dir",
    "mitochondrion_data_dir",
    "plastome_data_dir",
    "plastome_reference_dir",
    "plastome_reference_manifest",
]
