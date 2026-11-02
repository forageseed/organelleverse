"""Grade MTPT candidates against other species' plastid and mitochondrial genomes.

A raw MTPT call (``detect_mtpt``: plastome aligned to the mitogenome) also
picks up mitochondrial sequence that merely resembles plastid sequence, e.g.
native rRNA/tRNA regions. Following Brown et al. 2026 (bioRxiv
2026.10.26.761860), each candidate is aligned to a reference panel of other
species' plastid and mitochondrial genomes and graded on its best hits:

- ``clean_plastid``: a strong plastid hit (query coverage > 50%, identity
  > 90%) and no strong mitochondrial hit;
- ``ambiguous``: strong hits of both kinds (e.g. an old transfer shared by
  many mitogenomes);
- ``conserved_mtdna``: a strong mitochondrial hit (coverage > 30%, identity
  > 85%) and no strong plastid hit - native mitochondrial sequence, excluded;
- ``undetermined``: neither threshold met.

``ambiguous`` is refined by which kind of hit the candidate is closer to (an
OV extension, not in the paper): ``closer_to_plastid`` when its best plastid
identity exceeds its best mitochondrial identity by ``identity_margin``
points - a recent or lineage-specific transfer; ``closer_to_mito`` in the
opposite case - a transfer inherited with the mitogenomes of relatives (still
plastid-derived, only older); ``equidistant`` otherwise. Each candidate also
reports how many other species carry a strong mitochondrial copy.

The panel replaces the paper's search of NCBI nt with RefSeq organelle
genomes (:func:`organelle_reference_fasta` builds it from
``fetch.refseq_snapshot`` GBFF), which is all the grading needs and avoids
nt's mislabelled and nuclear records. The query species must not be in the
mitochondrial panel (its own mitogenome contains every candidate): pass
``exclude_organism``; a near-identical full-length mitochondrial hit without
it is flagged ``possible_self_in_reference``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from organelleverse._public_validation import validated_public

from .._bio import read_fasta
from ..core.errors import OrganelleDependencyError, OrganelleInputError, OrganelleParameterError
from ..core.frozen import FrozenMap, thaw_json
from ..core.result import Finding, OrganelleResult
from ..sequences import Sequences, as_path
from .transfer import _OPERATION_VERSION, _SUITE, _provenance, detect_mtpt

__all__ = ["classify_fragment", "grade_mtpt", "organelle_reference_fasta", "refine_ambiguous"]

TIERS = ("clean_plastid", "ambiguous", "conserved_mtdna", "undetermined")
AMBIGUOUS_SUBTYPES = ("closer_to_plastid", "closer_to_mito", "equidistant")
_OUTFMT = "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen"


# ------------------------------------------------------------------ panel


def organelle_reference_fasta(
    gbff: str | Path,
    dest_fasta: str | Path,
    *,
    lineage: str | None = "Viridiplantae",
) -> dict[str, int]:
    """Stream a RefSeq GBFF into a FASTA of ``>ACCESSION organism=<name>`` records.

    ``lineage`` keeps only records whose taxonomy contains it (default green
    plants; None keeps all). Returns ``{"records": n, "bases": n}``.
    """
    kept = bases = 0
    out = Path(dest_fasta)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(gbff).open(encoding="utf-8", errors="replace") as handle, out.open("w") as sink:
        accession = organism = ""
        taxonomy: list[str] = []
        sequence: list[str] = []
        state = ""
        for line in handle:
            if line.startswith("LOCUS"):
                accession, organism, taxonomy, sequence, state = line.split()[1], "", [], [], ""
            elif line.startswith("VERSION"):
                parts = line.split()
                if len(parts) > 1:
                    accession = parts[1]
            elif line.startswith("  ORGANISM"):
                organism = line[12:].strip()
                state = "taxonomy"
            elif state == "taxonomy" and line.startswith("            "):
                taxonomy.append(line.strip())
            elif line.startswith("ORIGIN"):
                state = "sequence"
            elif line.startswith("//"):
                lineage_text = " ".join(taxonomy)
                if sequence and (lineage is None or lineage in lineage_text):
                    seq = "".join(sequence).upper()
                    sink.write(f">{accession} organism={organism}\n")
                    for i in range(0, len(seq), 80):
                        sink.write(seq[i : i + 80] + "\n")
                    kept += 1
                    bases += len(seq)
                state = ""
            elif state == "sequence":
                sequence.append("".join(line.split()[1:]))
            elif state == "taxonomy" and not line.startswith(" "):
                state = ""
    return {"records": kept, "bases": bases}


def _organisms(fasta: Path) -> dict[str, str]:
    organisms: dict[str, str] = {}
    with fasta.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                head, _, rest = line[1:].rstrip("\n").partition(" ")
                organisms[head] = rest.split("organism=", 1)[1] if "organism=" in rest else rest
    return organisms


def _ensure_blast_db(fasta: Path, makeblastdb: str) -> str:
    """Build (once) a nucleotide BLAST database beside ``fasta``."""
    if not any(fasta.with_name(fasta.name + suffix).exists() for suffix in (".nsq", ".nal", ".00.nsq")):
        process = subprocess.run(
            [makeblastdb, "-in", str(fasta), "-dbtype", "nucl", "-out", str(fasta)],
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise OrganelleInputError(
                code="transfer.makeblastdb_failed",
                message=process.stderr.strip()[-1000:] or "makeblastdb failed",
                details={"fasta": str(fasta)},
            )
    return str(fasta)


# --------------------------------------------------------------- grading


def _hits_by_query(
    rows: Iterable[Sequence[str]],
    organisms: Mapping[str, str],
    exclude: str | None,
) -> dict[str, list[dict[str, Any]]]:
    """Per query, every subject's query coverage (union of HSP spans) and identity."""
    spans: dict[tuple[str, str], list[tuple[int, int, float]]] = defaultdict(list)
    qlens: dict[str, int] = {}
    for row in rows:
        query, subject = row[0], row[1]
        organism = organisms.get(subject, "")
        if exclude and organism.lower().startswith(exclude.lower()):
            continue
        qstart, qend = sorted((int(row[4]), int(row[5])))
        spans[(query, subject)].append((qstart, qend, float(row[2])))
        qlens[query] = int(row[10])
    per_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (query, subject), hsps in spans.items():
        merged: list[list[int]] = []
        for start, end, _ in sorted(hsps):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        covered = sum(end - start + 1 for start, end in merged)
        weight = sum(end - start + 1 for start, end, _ in hsps)
        identity = sum((end - start + 1) * ident for start, end, ident in hsps) / weight
        per_query[query].append(
            {
                "accession": subject,
                "organism": organisms.get(subject, ""),
                "coverage": round(100.0 * covered / qlens[query], 2),
                "identity": round(identity, 2),
            }
        )
    return per_query


def _best(hits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(max(hits, key=lambda h: (h["coverage"] * h["identity"], h["coverage"])))


def _best_hits(
    rows: Iterable[Sequence[str]],
    organisms: Mapping[str, str],
    exclude: str | None,
) -> dict[str, dict[str, Any]]:
    """Per query, the best subject by coverage x identity."""
    return {query: _best(hits) for query, hits in _hits_by_query(rows, organisms, exclude).items()}


def _species(organism: str) -> str:
    return " ".join(organism.split()[:2])


def refine_ambiguous(
    plastid_hit: Mapping[str, float],
    mito_hit: Mapping[str, float],
    *,
    identity_margin: float = 1.0,
) -> str:
    """Which kind of hit an ambiguous candidate is closer to, by best-hit identity."""
    delta = plastid_hit["identity"] - mito_hit["identity"]
    if delta > identity_margin:
        return "closer_to_plastid"
    if delta < -identity_margin:
        return "closer_to_mito"
    return "equidistant"


def classify_fragment(
    plastid_hit: Mapping[str, float] | None,
    mito_hit: Mapping[str, float] | None,
    *,
    plastid_min_coverage: float = 50.0,
    plastid_min_identity: float = 90.0,
    mito_min_coverage: float = 30.0,
    mito_min_identity: float = 85.0,
) -> str:
    """The four-tier grade from the best plastid and mitochondrial hits."""
    plastid = bool(
        plastid_hit
        and plastid_hit["coverage"] > plastid_min_coverage
        and plastid_hit["identity"] > plastid_min_identity
    )
    mito = bool(
        mito_hit and mito_hit["coverage"] > mito_min_coverage and mito_hit["identity"] > mito_min_identity
    )
    if plastid and mito:
        return "ambiguous"
    if plastid:
        return "clean_plastid"
    if mito:
        return "conserved_mtdna"
    return "undetermined"


def _blast(blastn: str, query: Path, db: str, threads: int) -> list[list[str]]:
    argv = [
        blastn, "-task", "blastn", "-query", str(query), "-db", db, "-evalue", "1e-10",
        "-max_target_seqs", "50", "-num_threads", str(threads), "-outfmt", _OUTFMT,
    ]
    process = subprocess.run(argv, capture_output=True, text=True)
    if process.returncode != 0:
        raise OrganelleInputError(
            code="transfer.blastn_failed",
            message=process.stderr.strip()[-1000:] or "blastn failed",
            details={"argv": argv},
        )
    return [line.split("\t") for line in process.stdout.splitlines() if line.strip()]


@validated_public("transfer.grade_mtpt", python_only=("mtpt",))
def grade_mtpt(
    mito_fasta: str | Path | Sequences,
    cp_fasta: str | Path | Sequences | None = None,
    *,
    plastid_reference: str | Path,
    mitochondrion_reference: str | Path,
    mtpt: OrganelleResult | None = None,
    exclude_organism: str | None = None,
    plastid_min_coverage: float = 50.0,
    plastid_min_identity: float = 90.0,
    mito_min_coverage: float = 30.0,
    mito_min_identity: float = 85.0,
    identity_margin: float = 1.0,
    threads: int = 1,
) -> OrganelleResult:
    """Grade MTPT candidates as clean plastid / ambiguous / conserved mtDNA / undetermined.

    Candidates come from ``mtpt`` (a ``detect_mtpt`` result) or, when it is
    omitted, from ``detect_mtpt(mito_fasta, cp_fasta)``. Each candidate is
    searched (NCBI blastn, task blastn) against ``plastid_reference`` and
    ``mitochondrion_reference`` FASTA panels (build them with
    :func:`organelle_reference_fasta`; a BLAST database is made beside each on
    first use). ``exclude_organism`` (e.g. "Oryza sativa") drops the query
    species from both panels.

    ``retained_fraction`` drops only ``conserved_mtdna`` candidates, as
    published; ``clean_fraction`` keeps only ``clean_plastid``.

    Ambiguous candidates get ``ambiguous_subtype`` (``closer_to_plastid`` /
    ``closer_to_mito`` / ``equidistant``, by best plastid minus best
    mitochondrial identity against ``identity_margin``); every candidate
    reports ``mito_species_with_copy``, the other species whose mitogenome
    holds a strong copy. With a densely sampled panel most transfers are
    ambiguous because relatives share them; the subtype separates recent from
    inherited ones.
    """
    mito_path = Path(as_path(mito_fasta))
    if mtpt is None:
        if cp_fasta is None:
            raise OrganelleParameterError(
                code="transfer.grade_mtpt_inputs",
                message="give either an mtpt result or the plastome (cp_fasta)",
            )
        mtpt = detect_mtpt(mito_path, cp_fasta)
    if mtpt.status == "failed":
        return mtpt
    if not mtpt.operation_id.endswith("detect_mtpt"):
        raise OrganelleParameterError(
            code="transfer.grade_mtpt_inputs", message="mtpt must be a detect_mtpt result"
        )
    blastn, makeblastdb = shutil.which("blastn"), shutil.which("makeblastdb")
    if not blastn or not makeblastdb:
        raise OrganelleDependencyError(
            code="transfer.blastn_missing",
            message="grading needs NCBI blastn and makeblastdb",
            suggested_action={"install": "mamba install -c bioconda blast"},
        )
    plastid_ref, mito_ref = Path(plastid_reference), Path(mitochondrion_reference)
    for reference in (plastid_ref, mito_ref):
        if not reference.is_file():
            raise OrganelleInputError(
                code="transfer.reference_missing",
                message=f"reference panel {reference} does not exist",
            )

    fragments = [dict(f) for f in thaw_json(mtpt.metrics)["fragments"]]
    records = dict(read_fasta(mito_path))
    genome_bp = sum(len(sequence) for sequence in records.values())
    graded: list[dict[str, Any]] = []
    if fragments:
        with tempfile.TemporaryDirectory(prefix="ov_grade_mtpt_") as tmp:
            query = Path(tmp) / "candidates.fa"
            with query.open("w") as handle:
                for index, fragment in enumerate(fragments):
                    sequence = records[fragment["seqid"]][fragment["start"] - 1 : fragment["end"]]
                    handle.write(f">c{index}\n{sequence}\n")
            plastid_hits = _hits_by_query(
                _blast(blastn, query, _ensure_blast_db(plastid_ref, makeblastdb), threads),
                _organisms(plastid_ref),
                exclude_organism,
            )
            mito_hits = _hits_by_query(
                _blast(blastn, query, _ensure_blast_db(mito_ref, makeblastdb), threads),
                _organisms(mito_ref),
                exclude_organism,
            )
        for index, fragment in enumerate(fragments):
            p_list, m_list = plastid_hits.get(f"c{index}", []), mito_hits.get(f"c{index}", [])
            p_hit = _best(p_list) if p_list else None
            m_hit = _best(m_list) if m_list else None
            sharing = sorted(
                {
                    _species(h["organism"])
                    for h in m_list
                    if h["coverage"] > mito_min_coverage and h["identity"] > mito_min_identity
                }
            )
            tier = classify_fragment(
                p_hit,
                m_hit,
                plastid_min_coverage=plastid_min_coverage,
                plastid_min_identity=plastid_min_identity,
                mito_min_coverage=mito_min_coverage,
                mito_min_identity=mito_min_identity,
            )
            fragment.pop("hsps", None)
            subtype = (
                refine_ambiguous(p_hit, m_hit, identity_margin=identity_margin)
                if tier == "ambiguous" and p_hit and m_hit
                else None
            )
            graded.append(
                {
                    **fragment,
                    "tier": tier,
                    "ambiguous_subtype": subtype,
                    "best_plastid_hit": p_hit,
                    "best_mito_hit": m_hit,
                    "identity_delta": (
                        round(p_hit["identity"] - m_hit["identity"], 2) if p_hit and m_hit else None
                    ),
                    "mito_species_with_copy": len(sharing),
                    "mito_species_examples": sharing[:10],
                }
            )

    bp = {tier: sum(f["length"] for f in graded if f["tier"] == tier) for tier in TIERS}
    count = {tier: sum(1 for f in graded if f["tier"] == tier) for tier in TIERS}
    retained_bp = sum(bp.values()) - bp["conserved_mtdna"]
    subtype_bp = {
        name: sum(f["length"] for f in graded if f["ambiguous_subtype"] == name)
        for name in AMBIGUOUS_SUBTYPES
    }
    flags = []
    if any(
        f["best_mito_hit"] and f["best_mito_hit"]["identity"] >= 99.5 and f["best_mito_hit"]["coverage"] >= 95
        for f in graded
    ) and exclude_organism is None:
        flags.append("possible_self_in_reference")
    if bp["conserved_mtdna"]:
        flags.append("conserved_mtdna_removed")
    raw_bp = sum(f["length"] for f in graded)
    return OrganelleResult(
        operation_id=f"{_SUITE}.grade_mtpt",
        operation_version=_OPERATION_VERSION,
        scope="mitochondrion",
        status="ok",
        summary_text=(
            f"{len(graded)} MTPT candidates, {raw_bp} bp: "
            + ", ".join(f"{tier} {count[tier]} ({bp[tier]} bp)" for tier in TIERS)
            + (
                " [ambiguous: " + ", ".join(f"{k} {v} bp" for k, v in subtype_bp.items()) + "]"
                if bp["ambiguous"]
                else ""
            )
            + f"; retained {retained_bp / genome_bp * 100 if genome_bp else 0:.2f}% of the mitogenome."
        ),
        metrics=FrozenMap(
            {
                "candidate_count": len(graded),
                "candidate_bp": raw_bp,
                "bp_by_tier": bp,
                "count_by_tier": count,
                "bp_by_ambiguous_subtype": subtype_bp,
                "mitogenome_bp": genome_bp,
                "raw_fraction": round(raw_bp / genome_bp, 6) if genome_bp else 0.0,
                "retained_fraction": round(retained_bp / genome_bp, 6) if genome_bp else 0.0,
                "clean_fraction": round(bp["clean_plastid"] / genome_bp, 6) if genome_bp else 0.0,
                "thresholds": {
                    "plastid_min_coverage": plastid_min_coverage,
                    "plastid_min_identity": plastid_min_identity,
                    "mito_min_coverage": mito_min_coverage,
                    "mito_min_identity": mito_min_identity,
                    "identity_margin": identity_margin,
                },
                "exclude_organism": exclude_organism,
                "fragments": graded,
            }
        ),
        findings=(
            Finding(code="retained_fraction", metric="retained_fraction", value=round(retained_bp / genome_bp, 6) if genome_bp else 0.0),
            Finding(code="conserved_mtdna_bp", metric="conserved_mtdna_bp", value=bp["conserved_mtdna"]),
        ),
        flags=tuple(flags),
        artifacts=(),
        provenance=_provenance(
            "grade_mtpt",
            {
                "plastid_reference": str(plastid_ref),
                "mitochondrion_reference": str(mito_ref),
                "exclude_organism": exclude_organism,
                "plastid_min_coverage": plastid_min_coverage,
                "plastid_min_identity": plastid_min_identity,
                "mito_min_coverage": mito_min_coverage,
                "mito_min_identity": mito_min_identity,
                "identity_margin": identity_margin,
            },
            method="ncbi_blastn",
        ),
    )
