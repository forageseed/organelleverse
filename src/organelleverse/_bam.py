"""Shared BAM/CRAM observation semantics for the optional Rust alignment reader.

The caller owns the rubam reader (including any CRAM reference). No HTSlib or
samtools runtime is loaded. Coordinates here are 0-based, half-open.
"""

from __future__ import annotations

from collections import Counter


def alignment_provenance(operation_id: str, parameters: dict):
    """Record the actual optional reader using the package's result contract."""
    import hashlib
    import json
    import os
    from importlib.metadata import version

    import rubam

    from .core.provenance import ResultProvenance

    payload = json.dumps(
        parameters, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return ResultProvenance(
        operation_id=operation_id,
        operation_version="1.0",
        package_version=version("organelleverse"),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=hashlib.sha256(payload).hexdigest(),
        requested_backend="rubam",
        actual_backend="rubam",
        attempted_backends=("rubam",),
        software_versions={"rubam": rubam.__version__},
    )


def base_qualities(read):
    """Return numeric Phred values, treating BAM's 0xff sentinel as missing."""
    values = read.query_qualities
    if values is None:
        return None
    values = list(values)
    return None if not values or values[0] == 255 else values


def _usable(read, min_mapping_quality: int, exclude_duplicates: bool) -> bool:
    return not (
        read.is_unmapped
        or read.is_secondary
        or read.is_supplementary
        or read.is_qcfail
        or (exclude_duplicates and read.is_duplicate)
        or read.mapping_quality < min_mapping_quality
    )


def site_base_counts(
    bam,
    contig: str,
    position: int,
    *,
    min_mapping_quality: int,
    min_base_quality: int,
    exclude_duplicates: bool,
) -> Counter[str]:
    """Count passing aligned observations; mates count independently.

    Walking aligned CIGAR pairs excludes D/N and soft clips. Quality and flag
    filters are explicit, without backend BAQ/overlap adjustment or depth caps.
    """
    counts: Counter[str] = Counter()
    for read in bam.fetch(contig, position, position + 1):
        if not _usable(read, min_mapping_quality, exclude_duplicates):
            continue
        qualities = base_qualities(read)
        sequence = read.query_sequence
        if qualities is None or sequence is None:
            continue
        for query, reference in read.get_aligned_pairs(matches_only=True):
            if reference == position and query is not None and qualities[query] >= min_base_quality:
                base = sequence[query].upper()
                if base in "ACGT":
                    counts[base] += 1
                break
    return counts


def aligned_depth(
    bam, contig: str, length: int, *, min_mapping_quality: int, exclude_duplicates: bool
):
    """Uncapped primary-alignment depth, including zeros and excluding D/N."""
    import numpy as np

    differences = np.zeros(length + 1, dtype=np.int64)
    for read in bam.fetch(contig, 0, length):
        if not _usable(read, min_mapping_quality, exclude_duplicates):
            continue
        for start, end in read.get_blocks():
            start, end = max(0, start), min(length, end)
            if start < end:
                differences[start] += 1
                differences[end] -= 1
    return np.cumsum(differences)[:length]
