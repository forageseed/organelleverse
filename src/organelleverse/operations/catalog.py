"""Residual four-operation catalog.

Sixteen of the twenty released operations migrated to checked-in core
capability bundles under ``src/organelleverse/capabilities/`` (Plan 03
Task 1, ``scripts/capabilities/migrate_release_catalog.py``). Four remain
here because the bundle contract cannot yet express something their
released contract needs (owner-ruling addendum on Decision 004, 2026-08-08):

- ``annotation.annotate`` declares three BLAST *executable* dependencies.
  The bundle validator requires an interface probe per executable
  dependency while forbidding probes on ``native`` implementations, and the
  ``external`` shape that implies needs a content-addressed bundle-local
  worker identity that package-resident code does not have.
- ``io.read_long_reads``, ``assembly.assemble`` and
  ``assembly.pmat_graph_build`` carry register-time suite
  :class:`DataContract`s (``sequencing_reads`` twice, ``pmat_graph_input``)
  that the bundle contract cannot yet reference; without them their
  bindings reject their own declared input modality.

Both gaps are the core-external / bundle data-contract lane tracked under
Decision 004 item 4. Until it lands, this module keeps serving the four
operations exactly as before, pinned by the same oracle fixture
(``tests/capabilities/fixtures/release_catalog_v0.json``).
"""

from __future__ import annotations

from organelleverse.annotation import api

from .data_contracts import DataContract
from .registry import OperationRegistry
from .registry import registry as default_registry
from .spec import (
    CoreKind,
    DependencyKind,
    DependencySpec,
    ExecutionMode,
    OperationSpec,
    OperationStage,
    SideEffect,
)

_ANNOTATION_REFERENCE = "docs/reports/annotation_release/README.md"

ANNOTATE_SPEC = OperationSpec(
    operation_id="annotation.annotate",
    contract_version="1.0",
    title="Annotate an organelle genome",
    description=(
        "Predicts protein-coding, tRNA, and rRNA gene models on an assembled "
        "mitochondrial or chloroplast genome and returns a structured "
        "annotation result. Optional sequence-only ORFs are candidate CDSs, not confirmed genes."
    ),
    keywords=("annotation", "gene-prediction", "mitochondrion", "plastome"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.ANALYZE,
    input_kind=CoreKind.GENOME,
    output_kind=CoreKind.RESULT,
    organelle_types=("mitochondrion", "plastid"),
    callable_locator="organelleverse.annotation.api:annotate",
    dependencies=(
        DependencySpec(
            kind=DependencyKind.PYTHON,
            name="Bio",
            locator="biopython",
            version_spec=">=1.85",
        ),
        DependencySpec(kind=DependencyKind.PYTHON, name="pyhmmer", version_spec=">=0.7"),
        DependencySpec(kind=DependencyKind.EXECUTABLE, name="blastn"),
        DependencySpec(kind=DependencyKind.EXECUTABLE, name="makeblastdb"),
        DependencySpec(kind=DependencyKind.EXECUTABLE, name="tblastn"),
    ),
    side_effects=(
        SideEffect.READ_FILES,
        SideEffect.WRITE_FILES,
        SideEffect.SUBPROCESS,
    ),
    deterministic=True,
    idempotent=True,
    cacheable=False,
    references=(_ANNOTATION_REFERENCE,),
)


def _released_reads_contract() -> DataContract:
    # Imported here, not at module scope: the assembly suite imports the registry,
    # and the registry imports this catalog.
    from ..assembly.data_contract import released_assembly_sequencing_reads_data_contract

    return released_assembly_sequencing_reads_data_contract()


READ_LONG_READS_SPEC = OperationSpec(
    operation_id="io.read_long_reads",
    contract_version="1.0",
    title="Read a long-read sequencing library",
    description=(
        "Reads one long-read library (PacBio HiFi/CLR or ONT) from disk as "
        "released sequencing-reads data, tagged with its technology and quality "
        "state. Verification is streaming with bounded memory (a 30 GB gzipped "
        "HiFi library verifies at ~50 MB RSS), gzip is accepted transparently, "
        "and the recorded digest describes the bytes on disk; the size cap is the "
        "64 GiB module default."
    ),
    keywords=("fastq", "long-reads", "sequencing"),
    execution_mode=ExecutionMode.INLINE,
    stage=OperationStage.READ,
    input_kind=CoreKind.NONE,
    output_kind=CoreKind.DATA,
    output_modalities=("sequencing_reads",),
    side_effects=(SideEffect.READ_FILES,),
    callable_locator="organelleverse.io_reads:read_long_reads",
)


def load_release_catalog(
    target_registry: OperationRegistry = default_registry,
) -> None:
    """Register the residual operations without executing scientific code or I/O."""

    from ..io_reads import read_long_reads

    target_registry.register_authoritative(
        READ_LONG_READS_SPEC,
        read_long_reads,
        data_contracts=(_released_reads_contract(),),
    )

    target_registry.register_authoritative(ANNOTATE_SPEC, api.annotate)

    # Assembly discovery imports only contracts and the lazy canonical API. The
    # execution service (and therefore environment preparation) stays unloaded
    # until the operation is actually invoked.
    from ..assembly.api import assemble, pmat_graph_build
    from ..assembly.data_contract import (
        pmat_graph_input_data_contract,
        released_assembly_sequencing_reads_data_contract,
    )
    from ..assembly.operations import (
        ASSEMBLE_SPEC,
        PMAT_GRAPH_BUILD_SPEC,
    )

    target_registry.register_authoritative(
        ASSEMBLE_SPEC,
        assemble,
        data_contracts=(released_assembly_sequencing_reads_data_contract(),),
    )
    target_registry.register_authoritative(
        PMAT_GRAPH_BUILD_SPEC,
        pmat_graph_build,
        data_contracts=(pmat_graph_input_data_contract(),),
    )
