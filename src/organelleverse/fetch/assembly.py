"""Group records into genomes — because one genome is not always one record.

Plant mitochondrial genomes are frequently **multipartite**: the genome is
deposited one chromosome per accession. The cucumber mitochondrion is three
records::

    NC_016005.1  chromosome 1  1,555,935 bp
    NC_016004.1  chromosome 2     83,817 bp
    NC_016006.1  chromosome 3     44,840 bp
                                ───────────
                                1,684,592 bp   ← the genome

Treating an accession as a genome breaks three ways at once:

1. **Silent truncation** — take only chromosome 1 and you have 92% of the
   genome and none of the genes on the other two, while believing you have all
   of it.
2. **A length filter becomes a shredder** — ``min_length=200_000`` is a sane
   floor for a plant mitogenome, and it deletes chromosomes 2 and 3 outright.
   So the filter must be applied to the *genome*, never to a molecule.
3. **Inflated n** — three accessions counted as three genomes. Every statistic
   downstream is then computed on a sample size that does not exist.

The signal is the ``/chromosome`` source qualifier: a single-circle mitogenome
(rice) has none; a multipartite one (cucumber) has it on every molecule.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "GenomeUnit",
    "group_records",
    "molecule_sort_key",
]

_LEADING_INT = re.compile(r"(\d+)")

# Submitters do not agree on how to name a molecule. Observed in real seed-plant
# mitogenomes: 1..108, contig1..contig17, cir1..cir6, M01, LS1, ge1. What they
# all share is a number — so the number, not the label, is the identity.
# A label with no number at all (A/B/C, roman numerals) is possible, and then
# completeness simply cannot be checked; say so rather than assume the genome is
# whole.


def chromosome_number(label: str) -> int | None:
    """The molecule's number, whatever the submitter called it.

    ``"3"`` -> 3, ``"contig12"`` -> 12, ``"cir6"`` -> 6, ``"M01"`` -> 1.
    ``None`` when the label carries no number, which means neither ordering nor
    completeness can be established from it.
    """
    match = _LEADING_INT.search(str(label or ""))
    return int(match.group(1)) if match else None


def _identity(record: dict[str, Any]) -> str:
    """A comparable molecule identity, normalised across naming schemes.

    ``"cir1"`` and ``"1"`` are the same molecule of two genomes that must not be
    merged; comparing raw labels would call them different and let the chimera
    through.
    """
    label = str(record.get("chromosome", "")).strip()
    number = chromosome_number(label)
    return str(number) if number is not None else label.lower()


def molecule_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    """Order molecules within a genome: chromosome 1, 2, 3 … then by size.

    Chromosome labels are not always integers (``"MT1"``, ``"II"``, ``"A"``), so
    a numeric prefix sorts first and anything else falls back to descending
    length — the main molecule leads either way.
    """
    label = str(record.get("chromosome", "")).strip()
    number = chromosome_number(label)
    if number is not None:
        return (0, number, label)
    # No number: keep a stable order, biggest molecule first.
    return (1, -int(record.get("length") or 0), label)


@dataclass(frozen=True)
class GenomeUnit:
    """One organelle genome, which may span several accessions."""

    unit_id: str
    organism: str
    cultivar: str
    molecules: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def accessions(self) -> tuple[str, ...]:
        return tuple(str(m["accession"]) for m in self.molecules)

    @property
    def total_length(self) -> int:
        """The length of the *genome* — what a length filter must look at."""
        return sum(int(m.get("length") or 0) for m in self.molecules)

    @property
    def molecule_count(self) -> int:
        return len(self.molecules)

    @property
    def is_multipartite(self) -> bool:
        return any(str(m.get("chromosome", "")).strip() for m in self.molecules)

    @property
    def gene_count(self) -> int:
        return sum(int(m.get("gene_count") or 0) for m in self.molecules)

    @property
    def protein_count(self) -> int:
        return sum(int(m.get("protein_count") or 0) for m in self.molecules)

    @property
    def missing_chromosomes(self) -> tuple[int, ...]:
        """Gaps in the chromosome numbering — evidence the genome is incomplete.

        If we hold chromosomes 1 and 3, chromosome 2 exists and we do not have
        it. That is detectable without knowing in advance how many molecules the
        genome has, and it is the cheapest guard against silent truncation.
        """
        if not self.completeness_checkable:
            return ()
        numbers = [
            n
            for n in (chromosome_number(m.get("chromosome", "")) for m in self.molecules)
            if n is not None
        ]
        if len(numbers) < 2:
            return ()
        present = set(numbers)
        return tuple(n for n in range(1, max(numbers)) if n not in present)

    @property
    def completeness_checkable(self) -> bool:
        """Can a gap even be detected here?

        Only if every molecule carries a number. A genome labelled A/B/C gives
        no way to know a molecule is missing — and reporting ``complete=True``
        for it would be a claim we cannot support.
        """
        labels = [str(m.get("chromosome", "")).strip() for m in self.molecules]
        return all(chromosome_number(label) is not None for label in labels if label)

    @property
    def complete(self) -> bool | None:
        """``None`` means "cannot tell" — never silently means "yes"."""
        if not self.completeness_checkable:
            return None
        return not self.missing_chromosomes

    def as_record(self) -> dict[str, Any]:
        """Genome-level summary — one row per genome, not per accession."""
        return {
            "unit_id": self.unit_id,
            "organism": self.organism,
            "cultivar": self.cultivar,
            "accessions": list(self.accessions),
            "molecule_count": self.molecule_count,
            "total_length": self.total_length,
            "gene_count": self.gene_count,
            "protein_count": self.protein_count,
            "multipartite": self.is_multipartite,
            "complete": self.complete,
            "completeness_checkable": self.completeness_checkable,
            "missing_chromosomes": list(self.missing_chromosomes),
            "chromosome_labels": [
                str(m.get("chromosome", "")) for m in self.molecules if m.get("chromosome")
            ],
        }


def _sample_id(record: dict[str, Any]) -> str:
    """A sample-level identifier — what actually distinguishes one genome."""
    for field_name in ("cultivar", "isolate", "specimen_voucher", "strain", "ecotype"):
        value = str(record.get(field_name, "")).strip()
        if value:
            return value
    return ""


def _grouping_key(record: dict[str, Any]) -> tuple[str, str]:
    """(kind, value) identifying the genome a molecule belongs to.

    **A BioProject is not a genome.** PRJNA718240 holds six potato cultivars,
    each a complete three-chromosome mitogenome; grouping on the project fused
    six genomes into one 18-molecule chimera of 2.79 Mb. A project is a
    submission, and a submission can carry an entire panel. So BioProject only
    groups when paired with a sample-level identifier.

    Preference order, strongest evidence first:

    * ``biosample`` — one biological sample, unambiguous.
    * ``bioproject`` + cultivar/isolate/voucher — one sample within a project.
    * ``doi`` + organism + sample — pre-BioSample records (the 2011 cucumber
      mitochondrion groups here).
    * organism + sample.
    * accession — refuse to merge.

    When no sample-level identifier exists at all, we refuse to group. Under-
    grouping splits one genome into several units and understates its length;
    over-grouping fuses several genomes into one, inflating length and
    destroying the sample size. The first is visible and recoverable, the
    second is neither.
    """
    organism = str(record.get("organism", "")).strip()
    chromosome = str(record.get("chromosome", "")).strip()
    sample = _sample_id(record)

    # Without a /chromosome qualifier the record IS the genome; never merge it
    # with anything, or two single-circle genomes from one project collapse
    # into one.
    if not chromosome:
        return ("accession", str(record.get("accession", "")))

    if biosample := str(record.get("biosample", "")).strip():
        return ("biosample", biosample)
    if (bioproject := str(record.get("bioproject", "")).strip()) and sample:
        return ("bioproject_sample", f"{bioproject}|{sample}")
    if (doi := str(record.get("doi", "")).strip()) and sample:
        return ("doi_sample", f"{doi}|{organism}|{sample}")
    if sample:
        return ("organism_sample", f"{organism}|{sample}")
    return ("accession", str(record.get("accession", "")))


def _split_on_duplicate_chromosomes(
    molecules: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """A genome cannot contain two chromosome 1s.

    A repeated chromosome label is proof the grouping key merged genomes that
    are not the same genome — it is the cheapest self-check available, and it
    catches over-grouping that no key hierarchy anticipated. When it fires, fall
    back to the safest possible split: one molecule, one unit.
    """
    seen: set[str] = set()
    for molecule in molecules:
        if not str(molecule.get("chromosome", "")).strip():
            continue
        # Compare normalised numbers, not raw labels: "cir1" and "1" are the
        # same molecule of two different genomes, and comparing the strings
        # would let that chimera through.
        identity = _identity(molecule)
        if identity in seen:
            return [[m] for m in molecules]
        seen.add(identity)
    return [molecules]


_ACC_PREFIX_NUM = re.compile(r"^([A-Za-z_]+)(\d+)")


def _accession_order(record: dict[str, Any]) -> tuple[str, int, str]:
    """Sort by accession the way GenBank assigns them: prefix, then number."""
    accession = str(record.get("accession", ""))
    match = _ACC_PREFIX_NUM.match(accession)
    if match:
        return (match.group(1).upper(), int(match.group(2)), accession)
    return (accession.upper(), 0, accession)


def _greedy_split_by_chromosome(
    molecules: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Cut a run of consecutive accessions wherever a chromosome number repeats.

    GenBank hands out consecutive accessions to one submission, so a genome's
    molecules are usually adjacent: MN104801/802/803 are chromosomes 1/2/3 of one
    potato mitogenome. But one submission can hold a whole panel —
    MZ030725..MZ030742 is six potato cultivars, and its chromosome numbers run
    1,2,3,1,2,3,... A repeat is therefore the boundary between genomes, and
    cutting there recovers both cases from nothing but accession order.

    This is the fallback for records that carry no BioSample and no cultivar —
    without it they each become their own "genome" and a one-molecule fragment
    is reported as a complete genome, which is the very failure this module
    exists to stop.
    """
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    seen: set[str] = set()

    for molecule in sorted(molecules, key=_accession_order):
        identity = _identity(molecule)
        if identity in seen:
            groups.append(current)
            current, seen = [], set()
        current.append(molecule)
        seen.add(identity)

    if current:
        groups.append(current)
    return groups


def group_records(records: Sequence[dict[str, Any]]) -> list[GenomeUnit]:
    """Group molecule-level records into genome-level units.

    A record with no ``/chromosome`` qualifier is its own genome. Records that
    share a genome are ordered chromosome 1, 2, 3 …
    """
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    anonymous: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        key = _grouping_key(record)
        # A molecule of a multipartite genome that carries no sample-level id:
        # do not strand it as its own "complete genome". Recover the genome from
        # accession order instead.
        if key[0] == "accession" and str(record.get("chromosome", "")).strip():
            anonymous.setdefault(str(record.get("organism", "")), []).append(dict(record))
        else:
            buckets.setdefault(key, []).append(dict(record))

    for organism, molecules in anonymous.items():
        for index, group in enumerate(_greedy_split_by_chromosome(molecules)):
            buckets[("accession_run", f"{organism}|{index}")] = group

    # Self-check before anything downstream trusts these groups.
    checked: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, molecules in buckets.items():
        parts = _split_on_duplicate_chromosomes(molecules)
        if len(parts) == 1:
            checked[key] = parts[0]
            continue
        for part in parts:
            checked[("accession", str(part[0].get("accession", "")))] = part

    units: list[GenomeUnit] = []
    for (kind, value), molecules in checked.items():
        molecules.sort(key=molecule_sort_key)
        first = molecules[0]
        units.append(
            GenomeUnit(
                unit_id=f"{kind}:{value}",
                organism=str(first.get("organism", "")),
                cultivar=str(first.get("cultivar", "")),
                molecules=tuple(molecules),
            )
        )
    units.sort(key=lambda u: (u.organism, u.cultivar, u.accessions[:1]))
    return units
