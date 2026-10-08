"""Released canonical annotation API and value types."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from . import api as api
from .api import FeatureType, annotate, extract, write
from .backends.base import AnnotationRequest, AnnotationStage
from .models import (
    AnnotationDocument,
    AnnotationFeature,
    AnnotationRecord,
    FeatureQualifier,
    LocationPart,
)
from .readers import ReaderOrganelle, read_fasta, read_genbank

if TYPE_CHECKING:
    from .edited_cds import translate_edited_cds, write_edited_cds
    from .orfs import find_orfs, write_orfs


def __getattr__(name: str):
    modules = {
        "find_orfs": ".orfs",
        "write_orfs": ".orfs",
        "translate_edited_cds": ".edited_cds",
        "write_edited_cds": ".edited_cds",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "AnnotationDocument",
    "AnnotationFeature",
    "AnnotationRecord",
    "AnnotationRequest",
    "AnnotationStage",
    "FeatureQualifier",
    "FeatureType",
    "LocationPart",
    "ReaderOrganelle",
    "annotate",
    "api",
    "extract",
    "find_orfs",
    "read_fasta",
    "read_genbank",
    "translate_edited_cds",
    "write",
    "write_edited_cds",
    "write_orfs",
]


def __getattr__(name: str):
    from importlib import import_module

    modules = {
        "translate_edited_cds": ".edited_cds",
        "write_edited_cds": ".edited_cds",
        "find_orfs": ".orfs",
        "write_orfs": ".orfs",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
