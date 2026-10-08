"""Typed cores for IR boundary (OmicVerse-style data-in/data-out)."""

from __future__ import annotations

from .ir_boundary import _analyze_one


def compute_ir_boundary(genomes: list) -> dict:
    """Analyze IR junctions. Returns dict per-genome."""
    per_genome = [_analyze_one(g) for g in genomes]
    return {
        "genome_count": len(genomes),
        "ir_lengths": [pg["ir_length"] for pg in per_genome],
        "lsc_lengths": [pg["lsc_length"] for pg in per_genome],
        "ssc_lengths": [pg["ssc_length"] for pg in per_genome],
        "per_genome": per_genome,
    }
