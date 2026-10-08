"""Released annotation- and assembly-QC operation specifications.

Every QC operation is ``CONSUME`` Result -> Result, idempotent, non-cacheable,
and prohibits fallback. ``qc.assembly`` declares Python contract dependencies on
``pysam`` and ``pyhmmer``; minimap2, samtools, and optional meryl belong to a
versioned QC environment manifest rather than mandatory host-executable
dependencies. ``qc.annotation`` consumes an already-published mitochondrial
``annotation.annotate`` Result and declares no host-executable dependency.

The lazy API keeps discovery free of environment resolution and scientific I/O.
"""

from organelleverse.operations import (
    CoreKind,
    DependencyKind,
    DependencySpec,
    ExecutionMode,
    OperationSpec,
    OperationStage,
    SideEffect,
)

ANNOTATION_QC_SPEC = OperationSpec(
    operation_id="qc.annotation",
    contract_version="1.0",
    title="Assess annotation quality",
    description=(
        "Consumes an already-published mitochondrial annotation result and "
        "computes quality-control metrics and findings over its gene models."
    ),
    keywords=("annotation", "assessment", "quality-control"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.CONSUME,
    input_kind=CoreKind.RESULT,
    output_kind=CoreKind.RESULT,
    organelle_types=("mitochondrion",),
    callable_locator="organelleverse.quality_control.api:annotation",
    side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES),
    deterministic=True,
    idempotent=True,
    cacheable=False,
)


ASSEMBLY_QC_SPEC = OperationSpec(
    operation_id="qc.assembly",
    contract_version="1.5",
    title="Assess assembly quality",
    description=(
        "Consumes an existing assembly result and computes quality-control "
        "metrics and findings (completeness, contamination, structure) over it."
    ),
    keywords=("assembly", "assessment", "quality-control"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.CONSUME,
    input_kind=CoreKind.RESULT,
    output_kind=CoreKind.RESULT,
    organelle_types=("mitochondrion", "plastid"),
    callable_locator="organelleverse.quality_control.api:assembly",
    dependencies=(
        DependencySpec(kind=DependencyKind.PYTHON, name="pysam"),
        DependencySpec(kind=DependencyKind.PYTHON, name="pyhmmer"),
    ),
    side_effects=(
        SideEffect.READ_FILES,
        SideEffect.WRITE_FILES,
        SideEffect.SUBPROCESS,
        SideEffect.NETWORK,
    ),
    deterministic=True,
    idempotent=True,
    cacheable=False,
)


QC_WRITE_SPEC = OperationSpec(
    operation_id="qc.write",
    contract_version="1.2",
    title="Write a quality-control result",
    description=(
        "Atomically materializes a non-failed canonical quality-control result "
        "to a caller-selected destination."
    ),
    keywords=("publish", "quality-control", "write"),
    execution_mode=ExecutionMode.INLINE,
    stage=OperationStage.CONSUME,
    input_kind=CoreKind.RESULT,
    output_kind=CoreKind.RESULT,
    organelle_types=("mitochondrion", "plastid"),
    callable_locator="organelleverse.quality_control.api:write",
    side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES),
    deterministic=True,
    idempotent=True,
    cacheable=False,
)


__all__ = ["ANNOTATION_QC_SPEC", "ASSEMBLY_QC_SPEC", "QC_WRITE_SPEC"]
