"""Assembly operation contracts."""

from organelleverse.operations import (
    CoreKind,
    ExecutionMode,
    OperationSpec,
    OperationStage,
    SideEffect,
)

ASSEMBLE_SPEC = OperationSpec(
    operation_id="assembly.assemble",
    contract_version="1.0",
    title="Assemble an organelle genome from sequencing reads",
    description=(
        "Assembles a mitochondrial or plastid genome from long-read sequencing "
        "data using a selected backend, producing a structured assembly result."
    ),
    keywords=("assembly", "genome", "sequencing"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.ANALYZE,
    input_kind=CoreKind.DATA,
    output_kind=CoreKind.RESULT,
    organelle_types=("mitochondrion", "plastid"),
    input_modalities=("sequencing_reads",),
    callable_locator="organelleverse.assembly.api:assemble",
    side_effects=(
        SideEffect.READ_FILES,
        SideEffect.WRITE_FILES,
        SideEffect.SUBPROCESS,
        SideEffect.NETWORK,
    ),
    deterministic=False,
    deterministic_reason="Assembler heuristics, thread scheduling, and optional remote index lookups can produce different contigs or ordering between runs.",
    idempotent=True,
    cacheable=False,
)


PMAT_GRAPH_BUILD_SPEC = OperationSpec(
    operation_id="assembly.pmat_graph_build",
    contract_version="1.0",
    title="Build a PMAT assembly graph",
    description=(
        "Builds the PMAT2 assembly graph from prepared input data, producing a "
        "structured graph-build result used as PMAT's assembly starting point."
    ),
    keywords=("assembly", "genome", "graph"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.ANALYZE,
    input_kind=CoreKind.DATA,
    output_kind=CoreKind.RESULT,
    organelle_types=("mitochondrion", "plastid"),
    input_modalities=("pmat_graph_input",),
    callable_locator="organelleverse.assembly.api:pmat_graph_build",
    side_effects=(
        SideEffect.READ_FILES,
        SideEffect.WRITE_FILES,
        SideEffect.SUBPROCESS,
        SideEffect.NETWORK,
    ),
    deterministic=False,
    deterministic_reason="Graph construction relies on read-order and heuristic pruning that may vary between runs.",
    idempotent=True,
    cacheable=False,
)


ASSEMBLY_WRITE_SPEC = OperationSpec(
    operation_id="assembly.write",
    contract_version="1.0",
    title="Write an assembly result",
    description=(
        "Atomically materializes a non-failed canonical assembly result to a "
        "caller-selected destination."
    ),
    keywords=("assembly", "publish", "write"),
    execution_mode=ExecutionMode.INLINE,
    stage=OperationStage.CONSUME,
    input_kind=CoreKind.RESULT,
    output_kind=CoreKind.RESULT,
    organelle_types=("mitochondrion", "plastid"),
    callable_locator="organelleverse.assembly.api:write",
    side_effects=(SideEffect.READ_FILES, SideEffect.WRITE_FILES),
    deterministic=True,
    idempotent=True,
    cacheable=False,
)
