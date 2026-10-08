"""Dedup — the same genome deposited more than once.

Two records can be the same genome. ``NC_011033.1`` and ``BA000029.3`` are
490,520 bp of identical sequence: NCBI curated the second into the first. Counted
as two genomes they double the sample size, and every statistic computed on that
sample is wrong.

**Only technical redundancy is removed.** Two records of *different sequence* are
two genomes, however similar — the Japonica and Indica rice mitogenomes differ by
995 bp and that difference is the subject of study, not noise. Cultivar, isolate
and voucher act as a hard veto: if they disagree, the records are never merged,
whatever else matches.

So dedup does exactly two things, both exact:

* **L1** — a RefSeq record naming its source in ``COMMENT: derived from X``.
* **L2** — records with identical sequence digests.

There is no L3. "These look similar enough" is not dedup, it is sampling, and it
is a scientific decision that belongs to the caller, not to a retrieval library.

Representative selection follows the caller's stated rule: **prefer the
chromosome-level assembly even when the contig-level one is the RefSeq copy.**
Completeness outranks curation status. An organelle nuccore record carries no
``assembly_level`` (that is a nuclear-assembly field), but it states
``COMPLETENESS: full length`` outright, and that is the stronger claim anyway.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GenomeCluster",
    "dedup_genomes",
    "quality_rank",
    "select_representative",
    "unlinked_refseq_records",
]


def unlinked_refseq_records(records: Sequence[dict[str, Any]]) -> list[str]:
    """RefSeq records whose curation link we failed to parse.

    Every one of 400 sampled Viridiplantae organelle RefSeq records states its
    source, in one of two wordings ("is identical to X" in all 400; "was derived
    from X" exists too, e.g. Oryza NC_011033.1). The parser handles both, and on
    that sample it misses nothing.

    But that is a claim about a sample, not about NCBI. If the wording changes,
    the failure mode is silent: a mirror goes unrecognised, the genome is counted
    twice, and the sample size is quietly wrong. So a RefSeq record that has a
    COMMENT and no parsed link is reported, not ignored.
    """
    return sorted(
        str(r["accession"])
        for r in records
        if r.get("is_refseq")
        and str(r.get("comment", "")).strip()
        and not str(r.get("derived_from", "")).strip()
    )


def _sample_fingerprint(record: dict[str, Any]) -> tuple[str, ...]:
    """The fields that make a record a *different biological sample*.

    Disagreement on any of these is a veto on merging. A cultivar difference is
    data, not redundancy.
    """
    return tuple(
        str(record.get(field, "")).strip().lower()
        for field in ("cultivar", "isolate", "specimen_voucher", "strain", "ecotype")
    )


def _vetoes_merge(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """True when two records name different samples.

    An *absent* label never vetoes — most records carry none. Only two records
    that both name a sample, and name different ones, are kept apart.
    """
    for a, b in zip(_sample_fingerprint(left), _sample_fingerprint(right), strict=True):
        if a and b and a != b:
            return True
    return False


def quality_rank(record: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    """Sort key for choosing which record represents a genome. Higher is better.

    The caller's rule, in order:

    1. **Completeness** — ``full length`` beats an unstated claim beats
       ``partial``. This is the organelle analogue of chromosome-level over
       contig-level, and it is asserted by the submitter rather than inferred.
    2. **Whole genome** — a genome with all of its molecules beats a truncated
       one.
    3. **Annotation depth** — more genes means a more complete record.
    4. **Fewer ambiguous bases** — N-runs mean a gappier assembly.
    5. **Circular topology** — weak, and deliberately weak: rice's complete
       mitogenome is deposited ``linear``.
    6. **RefSeq last.** Curation status is the tie-breaker, never the criterion.
       A chromosome-level GenBank record beats a contig-level RefSeq one, which
       is exactly the instruction.
    """
    completeness = {"full": 2, "": 1, "partial": 0}.get(str(record.get("completeness", "")), 1)
    whole = 0 if record.get("complete") is False else 1
    genes = int(record.get("gene_count") or 0)
    ambiguous = -int(record.get("ambiguous_count") or 0)
    circular = 1 if str(record.get("topology", "")).lower() == "circular" else 0
    refseq = 1 if record.get("is_refseq") else 0
    return (completeness, whole, genes, ambiguous, circular, refseq)


@dataclass(frozen=True)
class GenomeCluster:
    """One genome, and every record that describes it."""

    representative: dict[str, Any]
    duplicates: tuple[dict[str, Any], ...] = ()
    reason: str = ""

    @property
    def accessions(self) -> tuple[str, ...]:
        return tuple(str(r["accession"]) for r in (self.representative, *self.duplicates))

    @property
    def is_duplicated(self) -> bool:
        return bool(self.duplicates)

    def as_record(self) -> dict[str, Any]:
        return {
            "representative": str(self.representative["accession"]),
            "duplicates": [str(r["accession"]) for r in self.duplicates],
            "reason": self.reason,
            "organism": self.representative.get("organism", ""),
            "cultivar": self.representative.get("cultivar", ""),
        }


def select_representative(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The best record of a cluster, by :func:`quality_rank`."""
    return max(records, key=quality_rank)


def dedup_genomes(records: Sequence[dict[str, Any]]) -> list[GenomeCluster]:
    """Cluster records that describe the same genome, and pick a representative.

    Exact evidence only. Records whose sample labels disagree are never merged.
    """
    by_accession = {str(r["accession"]): dict(r) for r in records}
    # Match on the bare accession too: a COMMENT says "derived from BA000029",
    # while the record we hold is BA000029.3.
    by_base = {a.split(".")[0]: a for a in by_accession}

    parent: dict[str, str] = {a: a for a in by_accession}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str, why: str, reasons: dict[str, str]) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if _vetoes_merge(by_accession[ra], by_accession[rb]):
            return  # different samples: never merge, whatever the evidence says
        parent[rb] = ra
        reasons[ra] = why

    reasons: dict[str, str] = {}

    # L1 — a RefSeq record names the submission it curates.
    for accession, record in by_accession.items():
        source = str(record.get("derived_from", "")).strip()
        if not source:
            continue
        target = (by_accession.get(source) and source) or by_base.get(source.split(".")[0])
        if target and target != accession:
            union(target, accession, "refseq_mirror", reasons)

    # L2 — identical sequence digests.
    by_digest: dict[str, str] = {}
    for accession, record in by_accession.items():
        digest = str(record.get("sha256", "")).strip()
        if not digest:
            continue
        first = by_digest.setdefault(digest, accession)
        if first != accession:
            union(first, accession, "identical_sequence", reasons)

    groups: dict[str, list[dict[str, Any]]] = {}
    for accession in by_accession:
        groups.setdefault(find(accession), []).append(by_accession[accession])

    clusters: list[GenomeCluster] = []
    for root, members in groups.items():
        representative = select_representative(members)
        duplicates = tuple(m for m in members if m["accession"] != representative["accession"])
        clusters.append(
            GenomeCluster(
                representative=representative,
                duplicates=duplicates,
                reason=reasons.get(root, "") if duplicates else "",
            )
        )
    clusters.sort(key=lambda c: str(c.representative["accession"]))
    return clusters
