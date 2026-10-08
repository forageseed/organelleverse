"""Exact reference-coordinate binding for external variant benchmarks.

A circular transform is accepted only when it explains every base of both
references. Allele agreement is an audit of benchmark inputs, not an accuracy
score, and is never used to choose a transform.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CircularReferenceTransform:
    length: int
    strand: str
    offset: int

    def source_position(self, target_position: int) -> int:
        """Map a zero-based target base to the original source reference."""
        if not 0 <= target_position < self.length:
            raise ValueError("Target position is outside the reference")
        oriented = (self.offset + target_position) % self.length
        return oriented if self.strand == "+" else self.length - 1 - oriented


def exact_circular_transform(source: str, target: str) -> CircularReferenceTransform:
    """Find the unique rotation/orientation explaining the complete sequences.

    Repeated/palindromic references with multiple transforms are ambiguous;
    sequence differences, including one-base indels, cannot be rotated away.
    """
    source, target = source.upper(), target.upper()
    if not source or len(source) != len(target) or (set(source) | set(target)) - set("ACGT"):
        raise ValueError("Exact circular binding requires equal-length A/C/G/T references")
    reverse = source.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    matches = []
    for strand, sequence in (("+", source), ("-", reverse)):
        doubled = sequence + sequence[:-1]
        offset = doubled.find(target)
        while offset >= 0:
            matches.append(CircularReferenceTransform(len(source), strand, offset))
            if len(matches) > 1:
                raise ValueError("Circular reference binding is ambiguous")
            offset = doubled.find(target, offset + 1)
    if not matches:
        raise ValueError("References differ beyond a circular rotation or reverse complement")
    return matches[0]


def audit_reference_alleles(sequence: str, rows: list[dict]) -> dict:
    """Retain every supplied one-based literal REF assertion, including failures.

    Published table coordinates are linear: out-of-bounds REF alleles are
    reported, never wrapped or normalized implicitly.
    """
    sequence = sequence.upper()
    if not sequence or set(sequence) - set("ACGT"):
        raise ValueError("Reference must be a nonempty A/C/G/T sequence")
    observations = []
    for number, row in enumerate(rows, 3):
        position, ref = row["pos"], row["ref"]
        observed = None
        if (
            type(position) is not int
            or position < 1
            or not isinstance(ref, str)
            or not ref
            or set(ref.upper()) - set("ACGT")
        ):
            state = "unsupported_literal_reference_allele"
        elif position - 1 + len(ref) > len(sequence):
            state = "outside_reference"
        else:
            observed = sequence[position - 1 : position - 1 + len(ref)]
            state = "matched" if observed == ref.upper() else "mismatched"
        observations.append(
            {
                "sheet_row": number,
                "position_1_based": position,
                "reference_allele": ref,
                "observed_allele": observed,
                "status": state,
            }
        )
    states = ("matched", "mismatched", "unsupported_literal_reference_allele", "outside_reference")
    counts = {state: sum(row["status"] == state for row in observations) for state in states}
    return {
        "status": "passed" if observations and counts["matched"] == len(observations) else "failed",
        "reference_length": len(sequence),
        "counts": counts,
        "rows": observations,
    }
