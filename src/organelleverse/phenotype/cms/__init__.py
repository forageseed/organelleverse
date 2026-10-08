"""CMS phenotype prediction."""

from __future__ import annotations

from .annotation_evidence import assess_cms_annotation
from .core import compute_cms_candidates
from .evidence import assess_cms_candidates, write_cms_evidence
from .pipeline import cms
from .resources import cms_data_dir, cms_model_dir, cms_protein_fasta, cms_reference_json
from .topology import predict_cms_topology

__all__ = [
    "assess_cms_annotation",
    "assess_cms_candidates",
    "cms",
    "cms_data_dir",
    "cms_model_dir",
    "cms_protein_fasta",
    "cms_reference_json",
    "compute_cms_candidates",
    "predict_cms_topology",
    "write_cms_evidence",
]
