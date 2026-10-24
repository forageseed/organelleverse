"""Phenotype suite with separate CMS screening and evidence assessment."""

from __future__ import annotations

from ..inputs import make_reader

from .cms import (
    assess_cms_annotation,
    assess_cms_candidates,
    cms,
    compute_cms_candidates,
    predict_cms_topology,
    write_cms_evidence,
)

read = make_reader("phenotype", "fasta")

__all__ = [
    "read",
    "assess_cms_annotation",
    "assess_cms_candidates",
    "cms",
    "compute_cms_candidates",
    "predict_cms_topology",
    "write_cms_evidence",
]
