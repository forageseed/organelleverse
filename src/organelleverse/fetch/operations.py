"""OperationSpecs for the fetch suite.

Declaring the contract is what split this suite in two. ``deterministic`` cannot
honestly be ``True`` for a live database query, and cannot honestly be ``False``
for a pinned RefSeq release — so they are two operations, not one operation with
a backend flag.

The registry is release-only (a2a5bcc), so neither spec carries a maturity
tier. ``references`` stays empty: these operations have no benchmark yet, and an
empty citation list is the honest way to say so. See
``docs/superpowers/specs/2026-07-13-organelle-fetch-design.md`` §5 for the scoped
claim ``fetch.refseq_snapshot`` is positioned to make once a benchmark exists.
"""

from __future__ import annotations

from ..core.errors import OrganelleContractError
from ..operations import (
    CoreKind,
    ExecutionMode,
    OperationRegistry,
    OperationSpec,
    OperationStage,
    RetryPolicy,
    SideEffect,
)

__all__ = [
    "ACCESSION_LIST_SPEC",
    "DEDUP_SPEC",
    "ENTREZ_QUERY_SPEC",
    "FETCH_SPECS",
    "GENE_RECORDS_SPEC",
    "GENOME_SIZE_CANDIDATES_SPEC",
    "NGDC_GWH_SPEC",
    "NUCLEAR_ASSEMBLY_SPEC",
    "REFSEQ_SNAPSHOT_SPEC",
    "register_fetch_operations",
]

_RETRY = RetryPolicy(
    max_attempts=5,
    retryable_error_codes=("network.transient", "network.malformed_response"),
)

# Downloading touches the network and the disk. Saying so is what forbids
# caching (the validator rejects a cacheable operation with either effect).
_SIDE_EFFECTS = (SideEffect.NETWORK, SideEffect.WRITE_FILES)


REFSEQ_SNAPSHOT_SPEC = OperationSpec(
    operation_id="fetch.refseq_snapshot",
    contract_version="1.0",
    title="Fetch a pinned RefSeq organelle snapshot",
    description=(
        "Downloads organelle records from a pinned RefSeq release number, "
        "giving a reproducible accession set for the chosen organelle types."
    ),
    keywords=("fetch", "ncbi", "refseq"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.READ,
    input_kind=CoreKind.NONE,
    output_kind=CoreKind.DATA,
    organelle_types=("mitochondrion", "plastid"),
    output_modalities=("organelle_records",),
    callable_locator="organelleverse.fetch._agent_ops:op_refseq_snapshot",
    side_effects=_SIDE_EFFECTS,
    # Pinning RELEASE_NUMBER makes the accession set reproducible.
    deterministic=True,
    idempotent=True,
    cacheable=False,
    retry=_RETRY,
    # No benchmark yet: no citation to make. Empty is honest.
    references=(),
)


ENTREZ_QUERY_SPEC = OperationSpec(
    operation_id="fetch.entrez_query",
    contract_version="1.0",
    title="Query NCBI nuccore for organelle records",
    description=(
        "Runs a live Entrez nuccore search term and returns matching organelle "
        "records; not reproducible since the underlying database changes over time."
    ),
    keywords=("entrez", "fetch", "ncbi"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.READ,
    input_kind=CoreKind.NONE,
    output_kind=CoreKind.DATA,
    organelle_types=("mitochondrion", "plastid"),
    output_modalities=("organelle_records",),
    callable_locator="organelleverse.fetch._agent_ops:op_entrez_query",
    side_effects=_SIDE_EFFECTS,
    # nuccore is a live database: the same term returns different records over
    # time. Claiming reproducibility here would be false.
    deterministic=False,
    deterministic_reason="nuccore is a live database; matching records change over time.",
    idempotent=False,
    cacheable=False,
    retry=_RETRY,
    # No benchmark yet: no citation to make. Empty is honest.
    references=(),
)


ACCESSION_LIST_SPEC = OperationSpec(
    operation_id="fetch.accession_list",
    contract_version="1.0",
    title="Fetch organelle records by accession",
    description=(
        "Downloads organelle records for a caller-supplied, explicit list of "
        "accessions, reproducing someone else's named dataset exactly."
    ),
    keywords=("accession", "fetch", "ncbi"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.READ,
    input_kind=CoreKind.NONE,
    output_kind=CoreKind.DATA,
    organelle_types=("mitochondrion", "plastid"),
    output_modalities=("organelle_records",),
    callable_locator="organelleverse.fetch._agent_ops:op_accession_list",
    side_effects=_SIDE_EFFECTS,
    # An explicit accession list is the one live-database call that IS
    # reproducible: the caller named the records, so the same names return the
    # same records. This is how someone else's dataset gets reproduced.
    deterministic=True,
    idempotent=True,
    cacheable=False,
    retry=_RETRY,
    references=(),
)


GENE_RECORDS_SPEC = OperationSpec(
    operation_id="fetch.gene_records",
    contract_version="1.0",
    title="Fetch gene records from NCBI",
    description=(
        "Runs a live NCBI gene-record search and returns matching gene records "
        "for either nuclear or organelle genes; not reproducible over time."
    ),
    keywords=("fetch", "gene", "ncbi"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.READ,
    input_kind=CoreKind.NONE,
    output_kind=CoreKind.DATA,
    # Gene records exist for the nucleus too, so this operation is not confined
    # to the organelles.
    organelle_types=("mitochondrion", "plastid"),
    output_modalities=("gene_records",),
    callable_locator="organelleverse.fetch._agent_ops:op_gene_records",
    side_effects=_SIDE_EFFECTS,
    # Live nuccore, same as entrez_query.
    deterministic=False,
    deterministic_reason="NCBI Gene records are updated continuously; identical queries can return different records.",
    idempotent=False,
    cacheable=False,
    retry=_RETRY,
    references=(),
)


NUCLEAR_ASSEMBLY_SPEC = OperationSpec(
    operation_id="fetch.nuclear_assembly",
    contract_version="1.0",
    title="Fetch a nuclear genome assembly",
    description=(
        "Downloads a nuclear genome assembly record from NCBI Datasets; not "
        "reproducible since new assemblies land and reference picks can change."
    ),
    keywords=("assembly", "fetch", "ncbi"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.READ,
    input_kind=CoreKind.NONE,
    output_kind=CoreKind.DATA,
    # Nuclear assemblies are not an organelle type. Declaring one would be a lie
    # the contract can see through.
    organelle_types=(),
    output_modalities=("nuclear_assemblies",),
    callable_locator="organelleverse.fetch._agent_ops:op_nuclear_assembly",
    side_effects=_SIDE_EFFECTS,
    # NCBI Datasets is live: new assemblies land, and `reference_only` can point
    # at a different assembly next month.
    deterministic=False,
    deterministic_reason="NCBI Datasets assembly metadata changes as new assemblies are released and reference picks are updated.",
    idempotent=False,
    cacheable=False,
    retry=_RETRY,
    references=(),
)


GENOME_SIZE_CANDIDATES_SPEC = OperationSpec(
    operation_id="fetch.genome_size_candidates",
    contract_version="1.0",
    title="Fetch nuclear genome-size candidates",
    description=(
        "Resolves a species through NCBI Taxonomy and returns ranked nuclear "
        "genome-size candidates; the live ranking can shift between calls."
    ),
    keywords=("fetch", "genome-size", "ncbi"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.READ,
    input_kind=CoreKind.NONE,
    output_kind=CoreKind.DATA,
    # A nuclear genome-size lookup, not an organelle record. The exact species
    # is resolved through NCBI Taxonomy; no organelle type applies.
    organelle_types=(),
    output_modalities=("genome_size_candidates",),
    callable_locator="organelleverse.fetch._agent_ops:op_genome_size_candidates",
    side_effects=_SIDE_EFFECTS,
    # NCBI Datasets is live: assemblies land, statuses change, and the ranked
    # candidate set can shift between calls. Reproducibility lives in the
    # content-addressed report artifacts, not in the live list.
    deterministic=False,
    deterministic_reason="Ranked candidates depend on live NCBI Taxonomy/Datasets records that change over time.",
    idempotent=False,
    cacheable=False,
    retry=_RETRY,
    references=(),
)


DEDUP_SPEC = OperationSpec(
    operation_id="fetch.dedup_genomes",
    contract_version="1.0",
    title="Deduplicate organelle genome records",
    description=(
        "Removes duplicate organelle records from an already-fetched set using "
        "only metadata already in hand; a pure, cacheable computation."
    ),
    keywords=("dedup", "genome", "transform"),
    execution_mode=ExecutionMode.INLINE,
    stage=OperationStage.TRANSFORM,
    input_kind=CoreKind.DATA,
    output_kind=CoreKind.DATA,
    organelle_types=("mitochondrion", "plastid"),
    input_modalities=("organelle_records",),
    output_modalities=("organelle_records",),
    callable_locator="organelleverse.fetch._agent_ops:op_dedup_genomes",
    # Pure computation over metadata already in hand.
    side_effects=(),
    deterministic=True,
    idempotent=True,
    cacheable=True,
    references=(),
)


NGDC_GWH_SPEC = OperationSpec(
    operation_id="fetch.ngdc_gwh",
    contract_version="1.0",
    title="Fetch organelle records from NGDC GWH",
    description=(
        "Downloads organelle records from NGDC's Genome Warehouse (CNCB/CGIR), "
        "a live database covering both mitochondrial and plastid genomes."
    ),
    keywords=("fetch", "ngdc", "plastid"),
    execution_mode=ExecutionMode.DURABLE,
    stage=OperationStage.READ,
    input_kind=CoreKind.NONE,
    output_kind=CoreKind.DATA,
    # CGIR/GWH holds both, though the plastid side (1,270) dwarfs the
    # mitochondrial one (5). Both are declared; the spec does not overclaim.
    organelle_types=("mitochondrion", "plastid"),
    output_modalities=("organelle_records",),
    callable_locator="organelleverse.fetch._agent_ops:op_ngdc_gwh",
    side_effects=_SIDE_EFFECTS,
    # NGDC is a live database, same as nuccore.
    deterministic=False,
    deterministic_reason="NGDC Genome Warehouse content is updated continuously; identical queries can return different records.",
    idempotent=False,
    cacheable=False,
    retry=_RETRY,
    references=(),
)


FETCH_SPECS = (
    REFSEQ_SNAPSHOT_SPEC,
    ENTREZ_QUERY_SPEC,
    ACCESSION_LIST_SPEC,
    GENE_RECORDS_SPEC,
    NUCLEAR_ASSEMBLY_SPEC,
    GENOME_SIZE_CANDIDATES_SPEC,
    NGDC_GWH_SPEC,
    DEDUP_SPEC,
)


def register_fetch_operations(target: OperationRegistry | None = None) -> tuple[str, ...]:
    """Bind every fetch spec to its agent-facing callable in a registry.

    The release catalog calls this for the default registry during operation
    discovery. Callers may also use it to populate an isolated registry.

    Idempotent: re-registering the same ID is a no-op rather than an error, so
    loading the release catalog twice does not explode.
    """
    from ..operations.registry import registry as default_registry
    from . import _agent_ops

    into = default_registry if target is None else target
    bound: list[str] = []
    for spec in FETCH_SPECS:
        # Every FETCH_SPECS entry is defined above in this file with a literal,
        # non-None locator; callable_locator is only ever None for a future
        # composite bundle's derived spec, never for these native specs.
        assert spec.callable_locator is not None
        _, attribute = spec.callable_locator.split(":")
        try:
            into.register(spec, getattr(_agent_ops, attribute))
        except OrganelleContractError as error:
            if error.code != "contract.duplicate_operation_id":
                raise
        bound.append(spec.operation_id)
    return tuple(bound)
