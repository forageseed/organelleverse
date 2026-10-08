"""Packaged CMS reference resources."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def cms_data_dir() -> Path:
    """Return the packaged CMS data directory."""
    return Path(str(files(__package__).joinpath("data")))


def cms_model_dir() -> Path:
    """Return the packaged CMS model directory."""
    return cms_data_dir() / "models"


def cms_reference_json() -> Path:
    """Return metadata for the accession-checked ORF79/ORF138 seed."""
    return cms_data_dir() / "curated" / "cms_reference.json"


def cms_protein_fasta() -> Path:
    """Return the accession-checked seed, excluding the unverified legacy catalog."""
    return cms_data_dir() / "curated" / "cms_proteins.fasta"


__all__ = [
    "cms_data_dir",
    "cms_model_dir",
    "cms_protein_fasta",
    "cms_reference_json",
]
