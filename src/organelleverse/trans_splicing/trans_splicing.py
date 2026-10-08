"""Trans-splicing detection in plant mitochondrial genomes. Self-contained.

In plant mitochondria, genes like nad1, nad2, nad5 have exons scattered across
the genome (trans-splicing). This function parses CDS/exon features from GenBank
and classifies genes as cis-splicing (all exons contiguous) vs trans-splicing
(exons on distant regions / different contigs).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from ..annotation.genbank import parse_genbank
from ..core.genome import OrganelleGenome
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult

# Known trans-splicing genes in plant mitochondria
_KNOWN_TRANS = {"nad1", "nad2", "nad5"}

_OPERATION_VERSION = "1.0"


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provenance(
    *,
    parameters: dict[str, object],
    started_at: datetime,
    finished_at: datetime,
    input_object_ids: tuple[str, ...] = (),
    input_artifact_hashes: tuple[str, ...] = (),
) -> ResultProvenance:
    return ResultProvenance(
        operation_id="trans_splicing.detect_trans_splicing",
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        input_object_ids=input_object_ids,
        input_artifact_hashes=input_artifact_hashes,
        parameters_hash=_sha256_json(parameters),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=max(0.0, (finished_at - started_at).total_seconds()),
    )


def detect_trans_splicing(
    genome: OrganelleGenome,
    *,
    min_exon_gap: int = 5000,
) -> OrganelleResult:
    """Detect trans-splicing genes from GenBank annotation, per locus.

    Same-gene features are clustered at ``min_exon_gap`` before the gap test,
    so the two IR copies of a chloroplast gene (rpl2, ycf2, ... tens of kb
    apart) count as cis; only exons of one locus split far apart (rps12 in
    plastids, nad1/2/5 in mito) are trans. A gene is trans-splicing if it has
    multiple exons separated by >= min_exon_gap bp, or exons on different
    contigs.
    """
    started_at = datetime.now(UTC)
    if genome.annotation is None:
        return OrganelleResult(
            operation_id="trans_splicing.detect_trans_splicing",
            operation_version=_OPERATION_VERSION,
            scope=genome.organelle,
            status="failed",
            summary_text="detect_trans_splicing() needs an annotation artifact.",
            provenance=_provenance(
                parameters={"min_exon_gap": min_exon_gap},
                input_object_ids=(genome.object_id,),
                started_at=started_at,
                finished_at=datetime.now(UTC),
            ),
            errors=(
                ErrorDetail(
                    code="trans_splicing.missing_annotation",
                    message=(
                        "detect_trans_splicing() requires an OrganelleGenome "
                        "carrying an annotation."
                    ),
                    details={"organelle": genome.organelle},
                ),
            ),
        )
    document = parse_genbank(genome.annotation.resolve())
    # Collect exon-bearing features (one CDS/exon feature = one annotation
    # unit), then cluster same-gene features into loci: features closer than
    # min_exon_gap merge, while the two IR copies of a gene (tens of kb apart)
    # stay separate loci. Pooling exons by gene name alone misread every
    # IR-duplicated chloroplast gene (rpl2, ycf2, ndhB, ...) as trans-spliced.
    gene_features: dict[str, list[dict]] = defaultdict(list)
    for record in document.records:
        for feature in record.features:
            ft = feature.type.casefold()
            if ft not in ("exon", "cds"):
                continue
            values = feature.qualifier_values("gene")
            if not values or not values[0]:
                continue
            parts = sorted(
                ({"start": p.start, "end": p.end} for p in feature.parts),
                key=lambda p: p["start"],
            )
            if not parts:
                continue
            gene_features[values[0].lower()].append(
                {
                    "seqid": record.seqid,
                    "start": parts[0]["start"],
                    "end": parts[-1]["end"],
                    "parts": parts,
                }
            )

    def _loci(features: list[dict]) -> list[list[dict]]:
        clustered: list[list[dict]] = []
        for feature in sorted(features, key=lambda f: f["start"]):
            if clustered and feature["start"] - clustered[-1][-1]["end"] < min_exon_gap:
                clustered[-1].append(feature)
            else:
                clustered.append([feature])
        return clustered

    # classify
    cis_genes: list[str] = []
    trans_genes: list[dict] = []
    for gene, features in gene_features.items():
        is_trans = False
        reason = ""
        exon_count = 0
        for locus in _loci(features):
            if len({feature["seqid"] for feature in locus}) > 1:
                is_trans = True
                reason = "multi_contig"
                break
            exons_sorted = sorted(
                (part for feature in locus for part in feature["parts"]),
                key=lambda p: p["start"],
            )
            exon_count = max(exon_count, len(exons_sorted))
            for i in range(len(exons_sorted) - 1):
                gap = exons_sorted[i + 1]["start"] - exons_sorted[i]["end"]
                if gap >= min_exon_gap:
                    is_trans = True
                    reason = f"large_gap:{gap}bp"
                    break
            if is_trans:
                break
        if is_trans:
            trans_genes.append(
                {
                    "gene": gene,
                    "exon_count": exon_count,
                    "reason": reason,
                    "known": gene in _KNOWN_TRANS,
                }
            )
        else:
            cis_genes.append(gene)
    return OrganelleResult(
        operation_id="trans_splicing.detect_trans_splicing",
        operation_version=_OPERATION_VERSION,
        scope=genome.organelle,
        status="ok",
        summary_text=f"{len(trans_genes)} trans-splicing gene(s), {len(cis_genes)} cis-splicing.",
        metrics={
            "cis_splicing": len(cis_genes),
            "trans_splicing": len(trans_genes),
            "total_genes": len(cis_genes) + len(trans_genes),
        },
        findings=tuple(
            Finding(
                code="trans_splicing.trans_spliced_gene",
                metric=g["gene"],
                value=g["exon_count"],
                unit="exons",
            )
            for g in trans_genes
        )
        or (
            Finding(
                code="trans_splicing.trans_splicing_count",
                metric="trans_splicing_count",
                value=0,
            ),
        ),
        flags=("trans_splicing_detected",) if trans_genes else (),
        provenance=_provenance(
            parameters={"min_exon_gap": min_exon_gap},
            input_object_ids=(genome.object_id,),
            input_artifact_hashes=(genome.annotation.sha256,),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        ),
    )
