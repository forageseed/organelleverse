"""Barcode suite: DNA barcode design + species identification. Self-contained.

``write`` is the canonical package-level output boundary
(:func:`organelleverse.writer.write`), not a barcode-specific writer. That
boundary only materializes results from released materializing operations, so it
currently rejects ``barcode.*`` results with ``output.unsupported_operation``
until a barcode materializer is registered in :mod:`organelleverse.writer`.

``design_marker_primers`` adds molecular-marker primer design on the alignment's
hypervariable regions (高变区) via the primer3-py bindings: primer pairs are
designed in the conserved flanks of each hotspot and filtered for cross-sample
conservation of their binding sites.
"""

from __future__ import annotations

from ..writer import write
from .barcode import design_barcode, identify
from .barcode_core import compute_barcode_candidates, compute_jaccard_identity
from .primers import design_marker_primers

design = design_barcode
save = write

__all__ = [
    "compute_barcode_candidates",
    "compute_jaccard_identity",
    "design",
    "design_barcode",
    "design_marker_primers",
    "identify",
    "save",
    "write",
]
