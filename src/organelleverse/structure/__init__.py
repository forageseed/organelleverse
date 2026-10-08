"""Structure suite: multiconf(), introns(), repeats(), resolve_configs(). Self-contained."""

from __future__ import annotations

from .structure import (
    introns,
    materialize_result,
    multiconf,
    repeats,
    resolve_configs,
    write_introns,
    write_multiconf,
    write_repeats,
    write_resolve_configs,
)
from .structure_core import compute_multiconf, compute_repeats

# Canonical write boundary for this suite. ``organelleverse.writer.write`` only
# dispatches released operations (assembly. / annotation. / qc.); until it grows
# a ``structure.`` branch this materializer is the suite-local entry point, with
# the same signature as the released ones: (OrganelleResult, output) -> Result.
write = materialize_result
save = write

__all__ = [
    "compute_multiconf",
    "compute_repeats",
    "introns",
    "materialize_result",
    "multiconf",
    "repeats",
    "resolve_configs",
    "save",
    "write",
    "write_introns",
    "write_multiconf",
    "write_repeats",
    "write_resolve_configs",
]
