"""Strict v1 annotation API shared by Python callers and Agents."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from annotated_types import Ge, Le

from organelleverse.core.genome import OrganelleGenome
from organelleverse.core.result import OrganelleResult
from organelleverse.runtime import managed_runs_root

from .backends.base import AnnotationRequest, AnnotationStage

FeatureType = Literal["CDS", "Protein", "tRNA", "rRNA", "Gene", "intron"]

__all__ = ["FeatureType", "annotate", "extract", "write"]


def run_annotation(
    genome: OrganelleGenome,
    request: AnnotationRequest,
) -> OrganelleResult:
    """Load and invoke the annotation service only when execution is requested."""

    from .service import run_annotation as execute

    return execute(genome, request)


def run_extraction(
    genome: OrganelleGenome,
    workspace: Path,
    feature_types: tuple[str, ...],
) -> OrganelleResult:
    """Load and invoke extraction only when execution is requested."""

    from .service import run_extraction as execute

    return execute(genome, workspace=workspace, feature_types=feature_types)


def materialize_result(result: OrganelleResult, output: Path) -> OrganelleResult:
    """Load the annotation writer only when materialization is requested."""

    from .writer import materialize_result as materialize

    return materialize(result, output=output)


def annotate(
    genome: OrganelleGenome,
    *,
    backend: Literal["auto", "mitochondrion", "plastome"] = "auto",
    threads: Annotated[int, Ge(1), Le(256)] = 4,
    call_trna: bool = True,
    call_rrna: bool = True,
    call_orfs: bool = False,
    orf_min_aa: Annotated[int, Ge(1)] = 30,
    orf_circular: bool = False,
    allow_partial: bool = False,
    rotate_to_trnh: bool = False,
) -> OrganelleResult:
    """Annotate known genes and optionally add sequence-only candidate ORF CDSs.

    ORFs use complete ATG-start candidates and the genome metadata genetic code.
    Circular scanning is explicit; sequence-only candidates do not establish
    expression, RNA editing, splicing or CMS causality.

    allow_partial is for fragments and incomplete assemblies. A requested stage that ran but found
    nothing (for example no rRNA on a 49 kb piece) is then reported in the result
    (status "warning", flag "partial_annotation", metric "stages_without_features") instead of
    failing the whole run. A run in which no stage found anything still fails.

    rotate_to_trnh (plastid only) presents a circular plastome the usual GenBank way: trnH-GUG on the
    reverse strand at the start of the sequence, reverse-complementing and rotating the input as needed.
    The output then refers to that presentation, and the change is listed in the pipeline warnings.
    """

    stages: tuple[AnnotationStage, ...] = (
        "pcg",
        *(("trna",) if call_trna else ()),
        *(("rrna",) if call_rrna else ()),
    )
    return run_annotation(
        genome,
        AnnotationRequest(
            backend=backend,
            workspace=managed_runs_root() / "annotation.annotate",
            threads=threads,
            stages=stages,
            call_orfs=call_orfs,
            orf_min_aa=orf_min_aa,
            orf_circular=orf_circular,
            allow_partial=allow_partial,
            rotate_to_trnh=rotate_to_trnh,
        ),
    )


def extract(
    genome: OrganelleGenome,
    *,
    feature_types: tuple[FeatureType, ...] = (
        "CDS",
        "Protein",
        "tRNA",
        "rRNA",
        "Gene",
        "intron",
    ),
) -> OrganelleResult:
    """Extract canonical feature sequences without rerunning annotation."""

    return run_extraction(
        genome,
        workspace=managed_runs_root() / "annotation.extract",
        feature_types=feature_types,
    )


def write(result: OrganelleResult, *, output: Path) -> OrganelleResult:
    """Atomically materialize a non-failed canonical annotation result."""

    return materialize_result(result, output=output)
