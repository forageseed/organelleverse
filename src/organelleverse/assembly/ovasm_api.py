"""OVASM convenience entry point using the canonical compute-then-write API."""

from __future__ import annotations

from typing import Literal

from organelleverse.core.data import OrganelleData
from organelleverse.core.result import OrganelleResult


def ovasm(
    data: OrganelleData,
    *,
    organelle: Literal["mitochondrion", "plastid"] = "mitochondrion",
    read_set: Literal["whole_genome", "target_reads"] = "whole_genome",
    seed_source: Literal["builtin", "custom", "discover"] = "builtin",
    graph_mode: Literal["sparse", "multichrom"] = "sparse",
    graph_min_count: int | Literal["auto"] | None = None,
    nuclear_ploidy: int = 2,
    fallback: bool = True,
    threads: int = 8,
    timeout_seconds: int | None = None,
) -> OrganelleResult:
    """Assemble structured reads with OVASM and return a managed Result.

    Read inputs with ``ov.asm.read`` and export the returned result explicitly
    with ``ov.asm.write``. Read technology and custom seed artifacts belong to
    the structured input, as for ``ov.asm.assemble(method="ovasm")``. For
    reference-assisted source screening, supply ``AssemblyAuxiliary(read_origins=...)``
    to ``ov.asm.read``. Its JSON binds to the original read ArtifactRef's existing
    ``sha256`` as ``original_reads_sha256``. Unassigned sequences are published
    separately and remain part of complete-output acceptance.
    ``fallback`` lets an unaccepted HiFi whole-genome run escalate to seed
    discovery, recording attempts and reasons; ``fallback=False`` runs once.
    """
    from .api import assemble
    from .contracts import OvasmParameters

    return assemble(
        data,
        organelle=organelle,
        method="ovasm",
        backend_parameters=OvasmParameters(
            read_set=read_set,
            seed_source=seed_source,
            graph_mode=graph_mode,
            graph_min_count=graph_min_count,
            nuclear_ploidy=nuclear_ploidy,
            fallback=fallback,
        ),
        threads=threads,
        timeout_seconds=timeout_seconds,
    )
