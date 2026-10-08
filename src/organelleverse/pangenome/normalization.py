"""Explicit reversible molecule orientation/origin changes; no inferred anchors."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Literal

from Bio.Seq import Seq

from .annotation_projection import Annotation


@dataclass(frozen=True)
class MoleculeTransform:
    """Rotate the oriented sequence left by origin (zero-based bases).

    A reverse transform reverse-complements first. Nonzero rotation is only
    meaningful for an explicitly circular molecule. Molecules are never joined.
    """

    source_path: str
    output_path: str
    length: int
    topology: Literal["linear", "circular", "unknown"] = "unknown"
    orientation: Literal["+", "-"] = "+"
    origin: int = 0

    def __post_init__(self):
        if not self.source_path or not self.output_path:
            raise ValueError("Source and output molecule paths must be nonempty")
        if isinstance(self.length, bool) or not isinstance(self.length, int) or self.length < 1:
            raise ValueError("Molecule length must be a positive integer")
        if self.topology not in {"linear", "circular", "unknown"}:
            raise ValueError("Topology must be linear, circular or unknown")
        if self.orientation not in {"+", "-"}:
            raise ValueError("Orientation must be + or -")
        if isinstance(self.origin, bool) or not isinstance(self.origin, int):
            raise ValueError("Origin must be an integer")
        if not 0 <= self.origin < self.length:
            raise ValueError("Origin must be within the molecule")
        if self.origin and self.topology != "circular":
            raise ValueError("Origin rotation requires explicitly circular topology")

    def sequence(self, sequence: str) -> str:
        if len(sequence) != self.length or set(sequence.upper()) - set("ACGTRYSWKMBDHVN"):
            raise ValueError("Sequence must match molecule length and use IUPAC DNA symbols")
        oriented = str(Seq(sequence).reverse_complement()) if self.orientation == "-" else sequence
        return oriented[self.origin :] + oriented[: self.origin]

    def interval(self, start: int, end: int) -> tuple[tuple[int, int], ...]:
        """Map one half-open interval, splitting only at the new circular origin.

        Returned pieces follow the forward strand of the oriented source.
        Annotation mapping additionally preserves each part's biological strand.
        """
        if not 0 <= start < end <= self.length:
            raise ValueError("Interval must be nonempty and inside the source molecule")
        if self.orientation == "-":
            start, end = self.length - end, self.length - start
        shifted = (start - self.origin) % self.length
        extent = end - start
        if shifted + extent <= self.length:
            return ((shifted, shifted + extent),)
        return ((shifted, self.length), (0, shifted + extent - self.length))

    def inverse(self) -> MoleculeTransform:
        return MoleculeTransform(
            source_path=self.output_path,
            output_path=self.source_path,
            length=self.length,
            topology=self.topology,
            orientation=self.orientation,
            origin=self.origin if self.orientation == "-" else (-self.origin) % self.length,
        )

    def annotation(self, annotation: Annotation) -> Annotation:
        if annotation.path != self.source_path:
            raise ValueError("Annotation path does not match transformation source")
        parts, strands = [], []
        for index, (start, end) in enumerate(annotation.parts):
            strand = annotation.strand_for_part(index)
            if strand != "." and self.orientation == "-":
                strand = "+" if strand == "-" else "-"
            pieces = self.interval(start, end)
            if strand == "-":
                pieces = tuple(reversed(pieces))
            parts.extend(pieces)
            strands.extend([strand] * len(pieces))
        return replace(
            annotation,
            path=self.output_path,
            parts=tuple(parts),
            strand=strands[0] if len(set(strands)) == 1 else ".",
            part_strands=tuple(strands),
        )

    def coordinate_rows(self) -> list[dict]:
        """Piecewise linear source→output mapping, zero-based half-open."""
        boundary = self.origin if self.orientation == "+" else self.length - self.origin
        boundaries = sorted({0, boundary, self.length})
        rows = []
        for start, end in itertools.pairwise(boundaries):
            for out_start, out_end in self.interval(start, end):
                rows.append(
                    {
                        "source_path": self.source_path,
                        "output_path": self.output_path,
                        "source_start": start,
                        "source_end": end,
                        "output_start": out_start,
                        "output_end": out_end,
                        "orientation": self.orientation,
                        "topology": self.topology,
                        "origin_after_orientation": self.origin,
                        "coordinate_system": "zero_based_half_open",
                    }
                )
        return rows
