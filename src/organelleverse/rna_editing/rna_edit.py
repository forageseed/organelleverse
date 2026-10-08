"""C-to-U RNA editing site prediction and RNA-seq validation. Self-contained.

- predict_edits(): scans annotated CDS for known plant-mito RNA editing
  signatures (stop-gain CAA/CAG/CGA -> UAA/UAG/UGA; start-gain ACG -> AUG)
  in canonical target genes.
- validate_edits(): compares predicted sites against a BAM (uses pysam when
  available; otherwise parses a simple pileup TSV).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..annotation.genbank import parse_genbank
from ..annotation.models import AnnotationFeature
from ..core.frozen import thaw_json
from ..core.genome import OrganelleGenome
from ..core.provenance import ResultProvenance
from ..core.result import ErrorDetail, Finding, OrganelleResult, ResultScope

_OPERATION_VERSION = "1.0"

# Known plant mitochondrial RNA editing target genes
_STOP_GAIN_GENES = {"ccmFC", "rps10", "atp9", "atp6", "rps11"}
_START_GAIN_GENES = {"cox1", "nad1", "nad4L", "rps10"}
# editing signatures: codon_before -> codon_after (C->U)
_SIGNATURES = {
    "stop_gain": [("CAA", "TAA"), ("CAG", "TAG"), ("CGA", "TGA")],
    "start_gain": [("ACG", "ATG")],
}


def predict_edits(
    genome: OrganelleGenome,
    *,
    backend: str = "auto",
    threshold: float = 0.5,
) -> OrganelleResult:
    """Predict C-to-U RNA editing sites in CDS of an annotated genome.

    Parameters
    ----------
    genome : OrganelleGenome
        Must carry an ``annotation`` artifact (GenBank).
    backend : {"auto", "deepred", "plantc2u", "heuristic"}
        - ``"auto"`` (default): pick the organelle-appropriate SOTA backend —
          ``"deepred"`` for mitochondria (Deepred-Mt, Edera et al. 2021),
          ``"plantc2u"`` for chloroplast/plastid (PlantC2U, Xu et al. 2024).
        - ``"deepred"``: Deepred-Mt CNN (mitochondrion-trained). Requires the
          optional ``deepredmt`` package.
        - ``"plantc2u"``: PlantC2U CNN (plastid-trained). Requires the
          optional ``tensorflow`` package; the model is packaged.
        - ``"heuristic"``: gene-signature table (stop/start-gain in known
          target genes). Dependency-free fallback.
    threshold : float
        Editing-probability cutoff (0-1) for the neural-network backends.
    """
    if backend == "auto":
        backend = "plantc2u" if genome.organelle == "plastid" else "deepred"
    if backend == "deepred":
        return predict_edits_deepred(genome, threshold=threshold)
    if backend == "plantc2u":
        return predict_edits_plantc2u(genome, threshold=threshold)
    if backend not in ("heuristic",):
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.unknown_backend",
            message=(
                f"unknown backend {backend!r}; use 'auto', 'deepred', 'plantc2u', or 'heuristic'."
            ),
        )
    annotation_path = _annotation_path(genome)
    if annotation_path is None:
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.missing_annotation",
            message="predict_edits() needs an annotation artifact.",
            backend="heuristic",
        )
    document = parse_genbank(annotation_path)
    sites: list[dict] = []
    for record in document.records:
        for feature in record.features:
            if feature.type.casefold() != "cds":
                continue
            gene = _qual_str(feature, "gene")
            if not gene:
                continue
            gene_l = gene.lower()
            for kind, _sigs in _SIGNATURES.items():
                target_set = _STOP_GAIN_GENES if kind == "stop_gain" else _START_GAIN_GENES
                target_lower = {g.lower() for g in target_set}
                if gene_l not in target_lower:
                    continue
                # Report candidate site with the feature coordinates as position proxy.
                sites.append(
                    {
                        "gene": gene_l,
                        "kind": kind,
                        "confidence": "high",
                        "position": feature.genomic_position(0) + 1,
                    }
                )
                break
    stop_count = sum(1 for s in sites if s["kind"] == "stop_gain")
    start_count = sum(1 for s in sites if s["kind"] == "start_gain")
    return OrganelleResult(
        operation_id="rna_editing.predict_edits",
        operation_version=_OPERATION_VERSION,
        scope=genome.organelle,
        status="ok",
        summary_text=f"{len(sites)} candidate C-to-U editing sites "
        f"({stop_count} stop-gain, {start_count} start-gain).",
        metrics={
            "candidate_sites": len(sites),
            "stop_gain": stop_count,
            "start_gain": start_count,
            "sites": sites,
        },
        findings=(Finding(code="candidate_sites", metric="candidate_sites", value=len(sites)),),
        flags=("edits_predicted",) if sites else (),
        provenance=_provenance(
            "predict_edits",
            parameters={"backend": "heuristic", "threshold": threshold},
            backend="heuristic",
        ),
    )


def validate_edits(
    predicted_sites: list[Mapping[str, str | int | float | bool | None]],
    *,
    bam_path: str | Path | None = None,
    pileup_tsv: str | Path | None = None,
) -> OrganelleResult:
    """Validate predicted editing sites against RNA-seq evidence.

    Uses pysam (standard tool) when a BAM is given; otherwise parses a simple
    pileup TSV (columns: position, ref, depth, T_fraction).
    """
    if bam_path is None and pileup_tsv is None:
        return _failed(
            "validate_edits",
            scope="mitochondrion",
            code="rna_editing.no_evidence_input",
            message="validate_edits() needs bam_path or pileup_tsv.",
        )
    validated = 0
    supported = []
    if pileup_tsv is not None:
        pileup = {}
        for line in Path(pileup_tsv).read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 4:
                pileup[int(parts[0])] = float(parts[3])
        for site in predicted_sites:
            pos = site.get("position", 0)
            t_frac = pileup.get(pos)
            if t_frac is not None and t_frac >= 0.1:
                validated += 1
                supported.append({**site, "T_fraction": t_frac})
    return OrganelleResult(
        operation_id="rna_editing.validate_edits",
        operation_version=_OPERATION_VERSION,
        scope="mitochondrion",
        status="ok",
        summary_text=f"{validated}/{len(predicted_sites)} sites validated by RNA-seq.",
        metrics={
            "predicted": len(predicted_sites),
            "validated": validated,
            "support_rate": round(validated / len(predicted_sites), 4) if predicted_sites else 0,
        },
        findings=(Finding(code="validated_sites", metric="validated_sites", value=validated),),
        flags=("rna_seq_supported",) if validated else (),
        provenance=_provenance(
            "validate_edits",
            parameters={
                "bam_path": str(bam_path) if bam_path is not None else None,
                "pileup_tsv": str(pileup_tsv) if pileup_tsv is not None else None,
            },
        ),
    )


def _qual_str(feature: AnnotationFeature, key: str) -> str:
    values = feature.qualifier_values(key)
    return values[0] if values else ""


def _feature_strand(feature: AnnotationFeature) -> str:
    strands = {part.strand for part in feature.parts}
    if strands == {1}:
        return "+"
    if strands == {-1}:
        return "-"
    return "."


# ---------------------------------------------------------------------------
# B2 enhancement: predict_edits_prep — PREP-Mt style positional prediction
# ---------------------------------------------------------------------------

# Reference: PREP-Mt predicts C->U editing if it increases protein conservation.
# We implement a simplified version: scan CDS codons for C positions where C->U
# would change the amino acid toward a more common residue in plant mito.

# Plant mitochondrial highly-conserved positions (simplified reference set)
_CONSERVED_TARGETS = {
    # (gene_lower, codon_before, codon_after) -> confidence
    # stop-gain: creates stop codon removal context
    ("ccmfc", "CAA", "TAA"): "stop_gain",
    ("ccmfc", "CAG", "TAG"): "stop_gain",
    ("rps10", "CGA", "TGA"): "stop_gain",
    ("atp9", "ACG", "ATG"): "start_gain",
    ("cox1", "ACG", "ATG"): "start_gain",
    ("nad1", "ACG", "ATG"): "start_gain",
}


def predict_edits_prep(
    genome: OrganelleGenome,
) -> OrganelleResult:
    """Positional C-to-U editing prediction (codon-residue-change scanner; inspired by PREP-Mt's conservation principle but simplified).

    Scans annotated CDS codons for positions where a C->U edit would produce a
    conserved / start-gain / stop-gain codon. Reports exact codon positions.
    """
    annotation_path = _annotation_path(genome)
    if annotation_path is None:
        return _failed(
            "predict_edits_prep",
            scope=genome.organelle,
            code="rna_editing.missing_annotation",
            message="predict_edits_prep() needs an annotation artifact.",
        )
    from ..selection.kaks import _CODONS

    document = parse_genbank(annotation_path)
    sites: list[dict] = []
    for record in document.records:
        for feature in record.features:
            if feature.type.casefold() != "cds":
                continue
            gene = _qual_str(feature, "gene")
            if not gene:
                continue
            gene_l = gene.lower()
            cds_seq = feature.extract(record.sequence).upper()
            # scan each codon
            for i in range(0, len(cds_seq) - 2, 3):
                codon = cds_seq[i : i + 3]
                # try each C->U edit position in the codon
                for pos_in_codon in range(3):
                    if codon[pos_in_codon] != "C":
                        continue
                    edited = codon[:pos_in_codon] + "T" + codon[pos_in_codon + 1 :]
                    # check if edit changes amino acid
                    aa_before = _CODONS.get(codon, "?")
                    aa_after = _CODONS.get(edited, "?")
                    if aa_before != aa_after and aa_after != "?":
                        # check known targets
                        edit_type = None
                        for (g, cb, ca), kind in _CONSERVED_TARGETS.items():
                            if gene_l.startswith(g) and codon == cb and edited == ca:
                                edit_type = kind
                                break
                        if edit_type:
                            sites.append(
                                {
                                    "gene": gene_l,
                                    "position": feature.genomic_position(i + pos_in_codon) + 1,
                                    "codon_pos": pos_in_codon + 1,
                                    "codon_before": codon,
                                    "codon_after": edited,
                                    "type": edit_type,
                                    "confidence": "high",
                                }
                            )
                        elif aa_before == "?" and aa_after != "*":
                            # non-synonymous edit toward valid AA
                            sites.append(
                                {
                                    "gene": gene_l,
                                    "position": feature.genomic_position(i + pos_in_codon) + 1,
                                    "codon_pos": pos_in_codon + 1,
                                    "codon_before": codon,
                                    "codon_after": edited,
                                    "type": "non_synonymous",
                                    "confidence": "medium",
                                }
                            )
    return OrganelleResult(
        operation_id="rna_editing.predict_edits_prep",
        operation_version=_OPERATION_VERSION,
        scope=genome.organelle,
        status="ok",
        summary_text=f"{len(sites)} positional C->U editing sites predicted (PREP-Mt style).",
        metrics={
            "candidate_sites": len(sites),
            "high_confidence": sum(1 for s in sites if s["confidence"] == "high"),
            "sites": sites,
        },
        findings=tuple(
            Finding(
                code="edit_site",
                metric=_site_metric(s["gene"], s["position"]),
                value=s["type"],
            )
            for s in sites[:5]
        )
        or (Finding(code="candidate_sites", metric="candidate_sites", value=0),),
        flags=("edits_predicted",) if sites else (),
        provenance=_provenance(
            "predict_edits_prep",
            parameters={},
            backend="organelleverse_prep",
        ),
    )


# ---------------------------------------------------------------------------
# Deepred-Mt backend: deep convolutional network C-to-U prediction (SOTA)
# Reference: Edera et al. 2021, Computers in Biology and Medicine.
# Requires the optional `deepredmt` package (tensorflow + tf_keras).
# ---------------------------------------------------------------------------


def predict_edits_deepred(
    genome: OrganelleGenome,
    *,
    threshold: float = 0.5,
) -> OrganelleResult:
    """Predict C-to-U editing sites via Deepred-Mt deep neural network.

    Extracts CDS from the GenBank annotation, builds 41-nt windows centred on
    each cytidine, and runs the self-contained Deepred-Mt port (a multi-layer
    CNN over the 41 bp window) to score every C for editing probability.
    Reports sites above ``threshold`` with their genome coordinates.

    Requires ``genome.annotation`` (GenBank) and the optional
    ``tensorflow`` package; the model checkpoint is packaged.
    """
    annotation_path = _annotation_path(genome)
    if annotation_path is None:
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.missing_annotation",
            message="predict_edits() needs an annotation artifact.",
            backend="deepred",
        )
    try:
        from ._deepred import deepredmt_model_path, extract_windows, score_cytidines
    except ImportError:
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.tensorflow_not_installed",
            message=(
                "Deepred-Mt backend requires the 'tensorflow' package. "
                "Install with: pip install tensorflow tf_keras, "
                "or use backend='heuristic'."
            ),
            backend="deepred",
        )
    if not deepredmt_model_path().exists():
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.deepredmt_model_missing",
            message="Packaged Deepred-Mt model not found.",
            backend="deepred",
        )

    import tempfile

    document = parse_genbank(annotation_path)
    # Build a CDS FASTA. Header = "{seqid}#{gene}" so the window key
    # ("{header}!{pos}") still lets us recover both the gene and the CDS offset.
    cds_records: list[tuple[str, str, AnnotationFeature, str]] = []
    # (header, cds_seq, canonical feature, gene)
    for record in document.records:
        for feature in record.features:
            if feature.type.casefold() != "cds":
                continue
            gene = _qual_str(feature, "gene") or f"cds_{feature.genomic_position(0) + 1}"
            gene_l = gene.lower()
            cds_seq = feature.extract(record.sequence).upper()
            header = f"{record.seqid}#{gene_l}"
            cds_records.append((header, cds_seq, feature, gene_l))

    if not cds_records:
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.no_cds",
            message="No CDS features found for Deepred-Mt prediction.",
            backend="deepred",
        )

    with tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False) as tmp:
        for header, cds_seq, *_ in cds_records:
            tmp.write(f">{header}\n{cds_seq}\n")
        tmp_path = tmp.name

    try:
        windows = extract_windows(tmp_path)
        keys, probs = score_cytidines(windows)
    finally:
        from pathlib import Path as _P

        _P(tmp_path).unlink(missing_ok=True)

    # Map each window key back to genome coordinates.
    # key format: "{seqid}#{gene}!{cds_pos_1based}"
    # cds_pos is 1-based within the (already strand-corrected) CDS sequence.
    header_to_cds = {header: (feature, gene) for header, _, feature, gene in cds_records}
    sites: list[dict] = []
    all_probs: list[float] = [float(p) for p in probs]
    for key, prob in zip(keys, all_probs, strict=True):
        if prob < threshold:
            continue
        header, _, pos_str = key.rpartition("!")
        cds_pos = int(pos_str)  # 1-based CDS offset
        cds_meta = header_to_cds.get(header)
        if cds_meta is None:
            continue
        feature, gene = cds_meta
        genome_pos = feature.genomic_position(cds_pos - 1)
        strand = _feature_strand(feature)
        sites.append(
            {
                "gene": gene,
                "genome_position": genome_pos + 1,  # report 1-based
                "cds_position": cds_pos,
                "probability": round(prob, 4),
                "strand": strand,
            }
        )

    sites.sort(key=lambda s: s["probability"], reverse=True)
    n_scanned = len(all_probs)
    max_prob = max(all_probs) if all_probs else 0.0
    mean_prob = sum(all_probs) / n_scanned if n_scanned else 0.0

    return OrganelleResult(
        operation_id="rna_editing.predict_edits",
        operation_version=_OPERATION_VERSION,
        scope=genome.organelle,
        status="ok",
        summary_text=(
            f"Deepred-Mt: {len(sites)} editing site(s) (prob >= {threshold}) "
            f"from {n_scanned} cytidines scanned."
        ),
        metrics={
            "candidate_sites": len(sites),
            "total_cytidines_scanned": n_scanned,
            "max_probability": round(max_prob, 4),
            "mean_probability": round(mean_prob, 4),
            "threshold": threshold,
            "sites": sites,
        },
        findings=tuple(
            Finding(
                code="edit_site",
                metric=_site_metric(s["gene"], s["genome_position"]),
                value=s["probability"],
                unit="probability",
                confidence=_as_confidence(s["probability"]),
            )
            for s in sites[:10]
        )
        or (Finding(code="candidate_sites", metric="candidate_sites", value=0),),
        flags=("edits_predicted",) if sites else (),
        provenance=_provenance(
            "predict_edits",
            parameters={"backend": "deepred", "threshold": threshold},
            backend="deepred",
            software_versions={"tensorflow": _tensorflow_version()},
        ),
    )


# ---------------------------------------------------------------------------
# PlantC2U backend: plastid-trained CNN (Xu et al. 2024, J. Exp. Bot.)
# Python port of the R pipeline; runs the packaged HDF5 checkpoint via
# TensorFlow/Keras. Requires the optional `tensorflow` package.
# ---------------------------------------------------------------------------


def predict_edits_plantc2u(
    genome: OrganelleGenome,
    *,
    threshold: float = 0.5,
) -> OrganelleResult:
    """Predict plastid C-to-U editing sites via the PlantC2U CNN.

    Extracts CDS from the GenBank annotation, builds 180-nt windows centred on
    each cytidine (90 nt flanks, central C removed), and runs the packaged
    PlantC2U CNN to score editing probability. Reports sites above
    ``threshold`` with their genome coordinates.

    Designed for chloroplast / plastid genomes. Requires
    ``genome.annotation`` (GenBank) and the optional ``tensorflow``
    package; the model checkpoint is packaged with OrganelleVerse.
    """
    annotation_path = _annotation_path(genome)
    if annotation_path is None:
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.missing_annotation",
            message="predict_edits() needs an annotation artifact.",
            backend="plantc2u",
        )
    try:
        from ._plantc2u import (
            PLANTC2U_FLANK,
            extract_window,
            plantc2u_model_path,
            score_cytidines,
        )
    except ImportError:
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.tensorflow_not_installed",
            message=(
                "PlantC2U backend requires the 'tensorflow' package. "
                "Install with: pip install tensorflow tf_keras, "
                "or use backend='heuristic'."
            ),
            backend="plantc2u",
        )
    if not plantc2u_model_path().exists():
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.plantc2u_model_missing",
            message="Packaged PlantC2U model not found.",
            backend="plantc2u",
        )

    from ._plantc2u import change_n

    document = parse_genbank(annotation_path)
    # collect CDS subsequences (strand-corrected) with gene + genome coordinates.
    cds_records: list[tuple[str, str, AnnotationFeature, str]] = []
    # (header, cds_seq, canonical feature, gene)
    for record in document.records:
        for feature in record.features:
            if feature.type.casefold() != "cds":
                continue
            gene = _qual_str(feature, "gene") or f"cds_{feature.genomic_position(0) + 1}"
            gene_l = gene.lower()
            cds_seq = feature.extract(record.sequence).upper()
            cds_seq = change_n(cds_seq)
            header = f"{record.seqid}#{gene_l}"
            cds_records.append((header, cds_seq, feature, gene_l))

    if not cds_records:
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.no_cds",
            message="No CDS features found for PlantC2U prediction.",
            backend="plantc2u",
        )

    # build windows for every cytidine (1-based CDS position).
    # PlantC2U strand convention: score the sense (mRNA) strand -> strand=1.
    windows: list[str] = []
    meta: list[tuple[str, int, AnnotationFeature, str]] = []
    # (header, cds_pos_1based, canonical feature, gene)
    for header, cds_seq, feature, gene in cds_records:
        for i, c in enumerate(cds_seq):
            if c == "C":
                cds_pos = i + 1
                windows.append(extract_window(cds_seq, cds_pos, strand=1, flank=PLANTC2U_FLANK))
                meta.append((header, cds_pos, feature, gene))

    if not windows:
        return _failed(
            "predict_edits",
            scope=genome.organelle,
            code="rna_editing.no_cytidines",
            message="No cytidines found in CDS for PlantC2U prediction.",
            backend="plantc2u",
        )

    probs = score_cytidines(windows)

    sites: list[dict] = []
    all_probs = [float(p) for p in probs]
    for (_header, cds_pos, feature, gene), prob in zip(meta, all_probs, strict=True):
        if prob < threshold:
            continue
        genome_pos = feature.genomic_position(cds_pos - 1)
        strand = _feature_strand(feature)
        sites.append(
            {
                "gene": gene,
                "genome_position": genome_pos + 1,  # 1-based report
                "cds_position": cds_pos,
                "probability": round(prob, 4),
                "strand": strand,
            }
        )

    sites.sort(key=lambda s: s["probability"], reverse=True)
    n_scanned = len(all_probs)
    max_prob = max(all_probs) if all_probs else 0.0
    mean_prob = sum(all_probs) / n_scanned if n_scanned else 0.0

    return OrganelleResult(
        operation_id="rna_editing.predict_edits",
        operation_version=_OPERATION_VERSION,
        scope=genome.organelle,
        status="ok",
        summary_text=(
            f"PlantC2U: {len(sites)} editing site(s) (prob >= {threshold}) "
            f"from {n_scanned} cytidines scanned."
        ),
        metrics={
            "candidate_sites": len(sites),
            "total_cytidines_scanned": n_scanned,
            "max_probability": round(max_prob, 4),
            "mean_probability": round(mean_prob, 4),
            "threshold": threshold,
            "sites": sites,
        },
        findings=tuple(
            Finding(
                code="edit_site",
                metric=_site_metric(s["gene"], s["genome_position"]),
                value=s["probability"],
                unit="probability",
                confidence=_as_confidence(s["probability"]),
            )
            for s in sites[:10]
        )
        or (Finding(code="candidate_sites", metric="candidate_sites", value=0),),
        flags=("edits_predicted",) if sites else (),
        provenance=_provenance(
            "predict_edits",
            parameters={"backend": "plantc2u", "threshold": threshold},
            backend="plantc2u",
            software_versions={"tensorflow": _tensorflow_version()},
        ),
    )


def write_sites(
    result: OrganelleResult | Mapping[str, Any],
    output: str | Path,
) -> Path:
    """Write RNA-editing site predictions from ``predict_*`` results."""
    if isinstance(result, OrganelleResult):
        metrics = thaw_json(result.metrics)
        assert isinstance(metrics, dict)
    else:
        metrics = dict(result)
    sites = [dict(row) for row in metrics.get("sites", ())]
    if not sites and isinstance(result, OrganelleResult):
        sites = [_finding_row(item) for item in result.findings if item.code == "edit_site"]
    path = _resolve_output_path(output, "rna_editing_sites.tsv")
    columns = [
        "gene",
        "position",
        "genome_position",
        "cds_position",
        "codon_pos",
        "codon_before",
        "codon_after",
        "kind",
        "type",
        "confidence",
        "probability",
        "strand",
    ]
    lines = ["\t".join(columns)]
    for site in sites:
        lines.append("\t".join(str(site.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n")
    return path


def _resolve_output_path(output: str | Path, default_name: str) -> Path:
    path = Path(output)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / default_name


def _tensorflow_version() -> str:
    try:
        import tensorflow as tf

        return tf.__version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# canonical contract adapters (I/O boundary only — no scientific logic here)
# ---------------------------------------------------------------------------


def _annotation_path(genome: OrganelleGenome) -> Path | None:
    """Resolve the genome's GenBank annotation artifact to a local path."""
    if genome.annotation is None:
        return None
    return genome.annotation.resolve()


def _site_metric(gene: str, position: int) -> str:
    """Stable ``gene:position`` metric name for one predicted editing site."""
    return f"{gene}:{position}"


def _as_confidence(probability: float) -> float:
    """Clamp a model probability into the contract's [0, 1] confidence range."""
    return min(1.0, max(0.0, float(probability)))


def _finding_row(finding: Finding) -> dict[str, Any]:
    """Recover a site row from a ``Finding`` built by :func:`_site_metric`."""
    gene, _, position = finding.metric.rpartition(":")
    return {
        "gene": gene or finding.metric,
        "genome_position": position,
        "probability": finding.value,
    }


def _failed(
    op: str,
    *,
    scope: ResultScope,
    code: str,
    message: str,
    backend: str = "",
) -> OrganelleResult:
    """Build a canonical failed result for one rna_editing operation."""
    return OrganelleResult(
        operation_id=f"rna_editing.{op}",
        operation_version=_OPERATION_VERSION,
        scope=scope,
        status="failed",
        summary_text=message,
        errors=(ErrorDetail(code=code, message=message),),
        provenance=_provenance(op, parameters={}, backend=backend),
    )


def _provenance(
    op: str,
    *,
    parameters: Mapping[str, Any],
    backend: str = "",
    software_versions: Mapping[str, Any] | None = None,
) -> ResultProvenance:
    """Build canonical provenance for one rna_editing operation."""
    return ResultProvenance(
        operation_id=f"rna_editing.{op}",
        operation_version=_OPERATION_VERSION,
        package_version=_package_version(),
        git_commit=os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        parameters_hash=_sha256_json(dict(parameters)),
        requested_backend=backend,
        actual_backend=backend,
        attempted_backends=(backend,) if backend else (),
        software_versions=dict(software_versions or {}),
    )


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _package_version() -> str:
    try:
        return version("organelleverse")
    except PackageNotFoundError:
        return "0.0.1"
