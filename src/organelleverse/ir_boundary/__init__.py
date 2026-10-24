"""IR boundary suite: chloroplast inverted-repeat junction analysis.

Self-contained IRscope-equivalent: locates LSC/IRb/SSC/IRa junctions from
GenBank-annotated chloroplast genomes, compares boundary gene positions across
genomes, and generates an SVG visualization. Pure Python.
"""

from __future__ import annotations

from ..writer import deprecated_alias

from ..inputs import make_reader

from .ir_boundary import ir_boundary, materialize_result, write_boundary_svg
from .ir_boundary_core import compute_ir_boundary

# Canonical write boundary for this suite. ``organelleverse.writer.write`` only
# dispatches released operations (assembly. / annotation. / qc.); until it grows
# an ``ir_boundary.`` branch this materializer is the suite-local entry point,
# with the same signature: (OrganelleResult, output) -> OrganelleResult.
write = materialize_result
save = deprecated_alias(write, "save")

read = make_reader("ir_boundary", genome=True)

__all__ = [
    "read",
    "compute_ir_boundary",
    "ir_boundary",
    "materialize_result",
    "save",
    "write",
    "write_boundary_svg",
]
