"""CMS evidence from canonical annotation, genomic context and edited CDSs."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from ..._bio import write_fasta
from ...annotation.genbank import _feature_label, _translation
from ...annotation.validation import validate_document
from ...annotation.writer import load_document_json
from ...core.artifacts import ArtifactRef
from ...core.errors import OrganelleInputError
from ...core.frozen import FrozenMap, thaw_json
from ...core.result import OrganelleResult
from .evidence import _unmatched, assess_cms_candidates


def _invalid(message):
    return OrganelleInputError(code="cms.invalid_annotation_evidence", message=message)


def _load_annotation(result):
    if result.operation_id not in {
        "annotation.annotate",
        "annotation.write",
    } or result.status not in {"ok", "warning"}:
        raise _invalid(
            "Expected a non-failed canonical annotation.annotate/annotation.write result"
        )
    if result.scope != "mitochondrion":
        raise _invalid("CMS assessment requires mitochondrial annotation")
    sources = [
        a
        for a in result.artifacts
        if a.kind in {"annotation", "annotation_json"} and a.format == "json"
    ]
    if len(sources) != 1:
        raise _invalid("Expected exactly one canonical annotation JSON artifact")
    source = sources[0]
    actual = ArtifactRef.from_path(
        source.resolve(), kind=source.kind, format=source.format, media_type=source.media_type
    )
    if actual.sha256 != source.sha256 or actual.size_bytes != source.size_bytes:
        raise _invalid("Canonical annotation artifact changed since its source result")
    try:
        document = load_document_json(source.resolve())
    except (OSError, ValueError) as error:
        raise _invalid("Cannot read a canonical annotation document") from error
    report = validate_document(document, document.requested_stages)
    if not report.valid:
        raise _invalid("Canonical annotation does not satisfy structural/translation validation")
    return document, source


def _location_key(seqid, parts):
    return seqid, tuple((p["start"], p["end"], p["strand"]) for p in parts)


def _context(feature, record, circular, limit):
    neighbors = []
    length = len(record.sequence)
    for other in record.features:
        if other.feature_id == feature.feature_id or other.type.casefold() not in {
            "cds",
            "trna",
            "rrna",
        }:
            continue
        intersections = [
            (max(a.start, b.start) + 1, min(a.end, b.end))
            for a in feature.parts
            for b in other.parts
            if max(a.start, b.start) < min(a.end, b.end)
        ]
        overlap = length - sum(p["end"] - p["start"] + 1 for p in _unmatched(length, intersections))
        shifts = (-length, 0, length) if circular else (0,)
        distance = min(
            max(0, b.start + shift - a.end, a.start - (b.end + shift))
            for a in feature.parts
            for b in other.parts
            for shift in shifts
        )
        neighbors.append(
            {
                "feature_id": other.feature_id,
                "gene": next(iter(other.qualifier_values("gene")), None),
                "locus_tag": next(iter(other.qualifier_values("locus_tag")), None),
                "type": other.type,
                "sequence_only_candidate": any(
                    value.startswith("ab initio prediction:orfipy:")
                    for value in other.qualifier_values("inference")
                ),
                "minimum_gap_bases": distance,
                "overlapping_bases": overlap,
                "parts": [p.model_dump(mode="json") for p in other.parts],
            }
        )
    neighbors.sort(key=lambda row: (row["minimum_gap_bases"], row["feature_id"]))
    return {
        "feature_id": feature.feature_id,
        "sequence_id": record.seqid,
        "gene": next(iter(feature.qualifier_values("gene")), None),
        "locus_tag": next(iter(feature.qualifier_values("locus_tag")), None),
        "product": next(iter(feature.qualifier_values("product")), None),
        "operator": feature.operator,
        "parts": [p.model_dump(mode="json") for p in feature.parts],
        "coordinate_system": "genomic 0-based half-open; parts in biological order",
        "topology": "unknown" if circular is None else "circular" if circular else "linear",
        "wraparound_neighbors_considered": circular is True,
        "sequence_only_candidate": any(
            value.startswith("ab initio prediction:orfipy:")
            for value in feature.qualifier_values("inference")
        ),
        "neighbors": neighbors[:limit],
        "annotated_neighbors": [row for row in neighbors if not row["sequence_only_candidate"]][
            :limit
        ],
        "candidate_orf_neighbors": [row for row in neighbors if row["sequence_only_candidate"]][
            :limit
        ],
        "neighbor_count": len(neighbors),
        "interpretation": "Genomic association/overlap, not promoter, cotranscription, fusion or causality proof",
    }


def _edited_rows(path, entries):
    try:
        edited = OrganelleResult.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as error:
        raise _invalid("Cannot read a canonical edited-CDS result") from error
    if (
        edited.operation_id != "annotation.translate_edited_cds"
        or edited.status != "ok"
        or edited.scope != "mitochondrion"
    ):
        raise _invalid("Expected successful annotation.translate_edited_cds JSON")
    index = {}
    for name, record, feature, _protein in entries:
        key = _location_key(record.seqid, [p.model_dump(mode="json") for p in feature.parts])
        if key in index:
            raise _invalid("Duplicate annotated CDS locations make edited-CDS mapping ambiguous")
        index[key] = (name, record, feature)
    rows = thaw_json(edited.metrics).get("cds")
    if not isinstance(rows, list):
        raise _invalid("Edited CDS result has no CDS table")
    mapped = {}
    for row in rows:
        required = {
            "seqid",
            "parts",
            "spliced_cds_before",
            "genetic_code",
            "codon_start",
            "protein_before",
            "protein_after",
        }
        if (
            not isinstance(row, dict)
            or not required <= row.keys()
            or not isinstance(row["parts"], list)
        ):
            raise _invalid("Edited CDS table has an invalid row")
        if row.get("pseudo"):
            continue
        if not row["parts"] or any(
            not isinstance(p, dict) or not {"start", "end", "strand"} <= p.keys()
            for p in row["parts"]
        ):
            raise _invalid("Edited CDS row lacks canonical location parts")
        from ...annotation.models import LocationPart

        try:
            parts = [
                LocationPart.model_validate(part).model_dump(mode="json") for part in row["parts"]
            ]
        except ValueError as error:
            raise _invalid("Edited CDS row has invalid canonical locations") from error
        if not all(
            isinstance(row[field], str)
            for field in ("seqid", "spliced_cds_before", "protein_before", "protein_after")
        ):
            raise _invalid("Edited CDS sequence fields must be strings")
        key = _location_key(row["seqid"], parts)
        if key not in index:
            raise _invalid("Edited CDS coordinates do not match the annotation")
        name, record, feature = index[key]
        if name in mapped:
            raise _invalid("Edited CDS result contains repeated annotation locations")
        code = int(next(iter(feature.qualifier_values("transl_table")), "1"))
        codon_start = int(next(iter(feature.qualifier_values("codon_start")), "1"))
        if (
            row["spliced_cds_before"] != feature.extract(record.sequence).upper()
            or row["genetic_code"] != code
            or row["codon_start"] != codon_start
        ):
            raise _invalid(
                "Edited CDS input sequence, genetic code or reading frame differs from annotation"
            )
        mapped[name] = row
    return mapped, edited


def assess_cms_annotation(
    annotation_result: OrganelleResult,
    *,
    conserved_proteins_fasta: str | Path,
    cms_proteins_fasta: str | Path | None = None,
    expression_tsv: str | Path | None = None,
    topology_predictions: str | Path | None = None,
    edited_cds_json: str | Path | None = None,
    protein_id_mode: Literal["export_id", "feature_id"] = "export_id",
    neighbor_limit: int = 5,
    min_identity: float = 80.0,
    min_homolog_coverage: float = 0.8,
    min_alignment_aa: int = 20,
    evalue: float = 1e-5,
    threads: int = 1,
    timeout: int = 120,
    losat_path: str = "LOSAT",
) -> OrganelleResult:
    """Combine sequence, genomic, topology, TPM and optional RNA-editing evidence.

    Default protein IDs match the canonical writer's proteins.fasta. Ambiguous
    export IDs fail explicitly; select feature_id mode for unique canonical IDs.
    All non-pseudo CDSs are assessed, including normal respiratory controls.
    No synthetic risk score or experimental CMS-causality decision is produced.
    """
    if (
        protein_id_mode not in {"export_id", "feature_id"}
        or isinstance(neighbor_limit, bool)
        or not isinstance(neighbor_limit, int)
        or neighbor_limit < 1
    ):
        raise _invalid("Invalid protein_id_mode or neighbor_limit")
    document, source = _load_annotation(annotation_result)
    entries = []
    ids = set()
    for record in document.records:
        for index, feature in enumerate(record.features):
            if feature.type.casefold() != "cds":
                continue
            if any(q.name in {"pseudo", "pseudogene"} for q in feature.qualifiers):
                continue
            protein = _translation(feature, record.sequence)
            if protein is not None:
                protein = protein.removesuffix("*")
            if not protein:
                raise _invalid(f"{feature.feature_id}: CDS has no assessable protein")
            name = (
                feature.feature_id
                if protein_id_mode == "feature_id"
                else _feature_label(feature, f"Protein_{index}")
            )
            if protein_id_mode == "export_id":
                tokens = name.split()
                if not tokens:
                    raise _invalid("Annotation export header contains no protein ID")
                name = tokens[0]
            elif len(name.split()) != 1:
                raise _invalid(
                    "Canonical feature IDs used as protein IDs must not contain whitespace"
                )
            if name in ids:
                raise _invalid(
                    "Annotation export protein IDs are ambiguous; use protein_id_mode='feature_id'"
                )
            ids.add(name)
            entries.append((name, record, feature, protein))
    if not entries:
        raise _invalid("Annotation contains no non-pseudo CDS proteins")
    params = dict(
        conserved_proteins_fasta=conserved_proteins_fasta,
        cms_proteins_fasta=cms_proteins_fasta,
        min_identity=min_identity,
        min_homolog_coverage=min_homolog_coverage,
        min_alignment_aa=min_alignment_aa,
        evalue=evalue,
        threads=threads,
        timeout=timeout,
        losat_path=losat_path,
    )
    edits, edited_source = (
        _edited_rows(edited_cds_json, entries) if edited_cds_json is not None else ({}, None)
    )
    with TemporaryDirectory(prefix="organelleverse-cms-annotation-") as scratch:
        query = write_fasta(
            Path(scratch) / "annotation-proteins.fasta",
            [(name, protein) for name, _, _, protein in entries],
        )
        result = assess_cms_candidates(
            query,
            expression_tsv=expression_tsv,
            topology_predictions=topology_predictions,
            **params,
        )
        changed = [
            (name, row["protein_after"])
            for name, row in edits.items()
            if row["protein_before"] != row["protein_after"] and row["protein_after"]
        ]
        edited_hits = {}
        if changed:
            edited_query = write_fasta(Path(scratch) / "edited-proteins.fasta", changed)
            compared = assess_cms_candidates(edited_query, **params)
            edited_hits = {
                row["candidate_id"]: row for row in thaw_json(compared.metrics)["candidate_table"]
            }
    metrics = thaw_json(result.metrics)
    by_id = {name: (record, feature) for name, record, feature, _ in entries}
    topology = document.source_metadata.get("topology")
    circular = True if topology == "circular" else False if topology == "linear" else None
    for row in metrics["candidate_table"]:
        name = row["candidate_id"]
        record, feature = by_id[name]
        row["genomic_context"] = _context(feature, record, circular, neighbor_limit)
        row["rna_editing"] = edits.get(name)
        if name in edited_hits:
            evidence = edited_hits[name]
            row["rna_editing"] = {
                **row["rna_editing"],
                "selected_edit_scenario_homology": {
                    "evidence_labels": evidence["evidence_labels"],
                    "cms_reference_hits": evidence["cms_reference_hits"],
                    "conserved_gene_hits": evidence["conserved_gene_hits"],
                },
            }
        row["evidence_availability"].update(genomic_context=True, rna_editing=name in edits)
    metrics.update(
        annotation_source=source.model_dump(mode="json"),
        protein_id_mode=protein_id_mode,
        source_annotation_status=annotation_result.status,
        source_annotation_flags=list(annotation_result.flags),
        source_annotation_metrics=thaw_json(annotation_result.metrics),
        neighbor_limit=neighbor_limit,
        edited_cds_source_flags=list(edited_source.flags) if edited_source else [],
        edited_cds_source_json=str(edited_cds_json) if edited_cds_json is not None else None,
    )
    flags = list(result.flags)
    if annotation_result.status == "warning":
        flags.append("source_annotation_needs_review")
    if edited_source is not None:
        flags.extend(edited_source.flags)
        flags.append("edited_protein_homology_is_selected_unphased_scenario")
    return result.model_copy(
        update={
            "operation_id": "phenotype.assess_cms_annotation",
            "status": annotation_result.status,
            "metrics": FrozenMap.from_json(metrics),
            "flags": tuple(dict.fromkeys(flags)),
            "summary_text": f"Assessed {len(entries)} annotated mitochondrial CDSs across CMS evidence axes.",
        }
    )
