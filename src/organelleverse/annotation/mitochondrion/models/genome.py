"""Genome data models."""

from __future__ import annotations

from pydantic import BaseModel, computed_field


class ContigInfo(BaseModel):
    """Track original contig positions after merging."""

    original_id: str
    start: int  # 1-based in merged sequence
    end: int  # inclusive
    length: int


class GenomeSequence(BaseModel):
    """Represents an assembled mitochondrial genome."""

    seqid: str
    sequence: str
    is_circular: bool = True
    contig_map: list[ContigInfo] | None = None

    @computed_field
    @property
    def length(self) -> int:
        return len(self.sequence)

    @computed_field
    @property
    def gc_content(self) -> float:
        seq = self.sequence.upper()
        gc = seq.count("G") + seq.count("C")
        valid = sum(1 for b in seq if b in "ATGCN")
        return gc / valid * 100 if valid > 0 else 0.0

    @computed_field
    @property
    def reverse_complement(self) -> str:
        comp = str.maketrans("ATGCatgcNn", "TACGtacgNn")
        return self.sequence.translate(comp)[::-1]

    def circular_distance(self, a: int, b: int) -> int:
        """Shortest distance between two positions on circular genome (1-based)."""
        diff = abs(a - b)
        return min(diff, self.length - diff)

    def circular_span(self, start: int, end: int) -> int:
        """Forward span from start to end on circular genome (1-based, inclusive).

        Handles wrap-around: if end < start, span crosses the origin.
        """
        if end >= start:
            return end - start + 1
        return self.length - start + end + 1

    def wrap_position(self, pos: int) -> int:
        """Normalize position to [1, length] range (1-based)."""
        return ((pos - 1) % self.length) + 1

    def circular_positions_between(self, start: int, end: int) -> list[int]:
        """All positions from start to end going forward (handles wrap-around)."""
        positions = []
        pos = start
        while True:
            positions.append(pos)
            if pos == end:
                break
            pos = (pos % self.length) + 1
            if pos == start:
                break  # safety: prevent infinite loop
        return positions

    def subsequence(self, start: int, end: int) -> str:
        """Extract subsequence (1-based, inclusive).

        Handles circular wrap-around when end < start.
        """
        if end >= start:
            return self.sequence[start - 1 : end]
        # Wrap-around: from start to end of genome, then beginning to end
        return self.sequence[start - 1 :] + self.sequence[:end]

    def get_sequence_for_range(self, start: int, end: int, strand: int = 1) -> str:
        """Get sequence for a range on given strand."""
        seq = self.subsequence(start, end)
        if strand == -1:
            comp = str.maketrans("ATGCatgcNn", "TACGtacgNn")
            return seq.translate(comp)[::-1]
        return seq
