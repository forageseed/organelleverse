"""Structure suite: multiconf(), introns(), repeats(), dispersed_repeats(), graph_complexity(), resolve_configs()."""

from __future__ import annotations

from ..writer import deprecated_alias

from ..inputs import make_reader

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
from .dispersed import dispersed_repeats
from .graph_complexity import graph_complexity
from .intron_classes import intron_classes

# Canonical write boundary for this suite. ``organelleverse.writer.write`` only
# dispatches released operations (assembly. / annotation. / qc.); until it grows
# a ``structure.`` branch this materializer is the suite-local entry point, with
# the same signature as the released ones: (OrganelleResult, output) -> Result.
write = materialize_result
save = deprecated_alias(write, "save")

read = make_reader("structure", "fasta", "gfa", genome=True)

__all__ = [
    "read",
    "compute_multiconf",
    "compute_repeats",
    "dispersed_repeats",
    "graph_complexity",
    "intron_classes",
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
