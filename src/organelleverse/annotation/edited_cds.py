"""Translate annotated CDSs after applying genomic RNA editing evidence."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from Bio.Data import CodonTable
from Bio.Seq import Seq

from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleInputError, OrganelleParameterError
from ..core.frozen import thaw_json
from ..core.result import OrganelleResult
from .genbank import parse_genbank


def _invalid(message: str) -> OrganelleInputError:
    return OrganelleInputError(code="input.invalid_edited_cds", message=message)


def _table(number: int):
    if isinstance(number, bool) or not isinstance(number, int):
        raise OrganelleParameterError(
            code="parameters.invalid_translation_code", message="Genetic code must be an integer"
        )
    try:
        table = CodonTable.unambiguous_dna_by_id[number]
    except (KeyError, TypeError) as error:
        raise OrganelleParameterError(
            code="parameters.invalid_translation_code",
            message=f"Unsupported genetic code: {number}",
        ) from error
    if set(table.stop_codons) & set(table.forward_table):
        raise OrganelleParameterError(
            code="parameters.invalid_translation_code",
            message="Context-dependent dual-coding stop tables are not supported",
        )
    return table


def translate_edited_cds(
    editing_result: OrganelleResult,
    *,
    annotation_genbank: str | Path,
    genetic_code: int | None = None,
) -> OrganelleResult:
    """Reconstruct joined CDSs and compare unedited/edited translation.

    Consumes detect_editing_sites or quantify_efficiency results. Sites use
    1-based genomic plus-strand alleles, including G>A on minus transcripts.
    Listed sites with positive or unspecified evidence are applied as a selected-
    edit scenario. Zero edited fraction/depth is excluded and reported.
    Frequencies/depths are retained; joint editing is unphased.
    Annotation parts remain in stored biological order (0-based half-open).
    GenBank codon_start and transl_table are respected; an explicit genetic_code
    overrides the latter. Internal stops are retained, not silently truncated.
    This operation uses supplied exon boundaries and does not infer new ones.
    """
    if editing_result.status != "ok" or editing_result.operation_id not in {
        "rna_editing.detect_editing_sites",
        "rna_editing.quantify_efficiency",
    }:
        raise _invalid(
            "Expected a successful RNA editing detection or known-site efficiency result"
        )
    if genetic_code is not None:
        _table(genetic_code)
    document = parse_genbank(annotation_genbank)
    records = {record.seqid: record for record in document.records}
    raw_sites = thaw_json(editing_result.metrics).get("sites")
    if not isinstance(raw_sites, list):
        raise _invalid("Editing result must contain a sites list")
    sites = []
    excluded = []
    seen = set()
    for index, raw in enumerate(raw_sites):
        if not isinstance(raw, dict):
            raise _invalid(f"Site {index + 1}: expected an object")
        site = dict(raw)
        seqid, pos, strand = site.get("seqid"), site.get("position"), site.get("strand")
        ref, edited = site.get("ref"), site.get("edited")
        if (
            not isinstance(seqid, str)
            or seqid not in records
            or isinstance(pos, bool)
            or not isinstance(pos, int)
            or not 1 <= pos <= len(records[seqid].sequence)
            or not isinstance(strand, str)
            or strand not in {"+", "-"}
            or not isinstance(ref, str)
            or ref not in {"A", "C", "G", "T"}
            or not isinstance(edited, str)
            or edited not in {"A", "C", "G", "T"}
            or ref == edited
        ):
            raise _invalid(f"Site {index + 1}: invalid genomic coordinate, strand, or alleles")
        if records[seqid].sequence[pos - 1].upper() != ref:
            raise _invalid(f"{seqid}:{pos}: reference allele does not match GenBank")
        key = (seqid, pos, strand)
        if key in seen:
            raise _invalid(f"Duplicate editing coordinate/strand: {seqid}:{pos}:{strand}")
        seen.add(key)
        fraction = site.get("editing_fraction")
        if fraction is not None and (
            isinstance(fraction, bool)
            or not isinstance(fraction, (float, int))
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise _invalid(f"{seqid}:{pos}: editing_fraction must be null or in [0, 1]")
        if fraction == 0 or site.get("depth") == 0:
            excluded.append({**site, "reason": "no_edited_evidence"})
            continue
        sites.append(site)
    applied_to = [[] for _ in sites]
    rows = []
    for record in document.records:
        for feature in record.features:
            if feature.type.upper() != "CDS":
                continue
            if feature.operator == "order":
                raise _invalid(f"{feature.feature_id}: order() does not specify a joined CDS")
            if feature.qualifier_values("transl_except"):
                raise _invalid(
                    f"{feature.feature_id}: transl_except requires specialized translation"
                )
            if any("slippage" in value.lower() for value in feature.qualifier_values("exception")):
                raise _invalid(
                    f"{feature.feature_id}: ribosomal slippage requires specialized translation"
                )
            sequence = record.sequence.upper()
            if set(sequence) - set("ACGTRYSWKMBDHVN"):
                raise _invalid(f"{record.seqid}: expected IUPAC DNA sequence")
            mapping = []
            for part in feature.parts:
                if part.end > len(sequence):
                    raise _invalid(f"{feature.feature_id}: CDS part exceeds the record length")
                mapping.extend(
                    (pos, part.strand)
                    for pos in (
                        range(part.start, part.end)
                        if part.strand == 1
                        else range(part.end - 1, part.start - 1, -1)
                    )
                )
            if len({pos for pos, _ in mapping}) != len(mapping):
                raise _invalid(
                    f"{feature.feature_id}: overlapping/repeated CDS parts are unsupported"
                )
            for i, part in enumerate(feature.parts):
                first = part.start_status if part.strand == 1 else part.end_status
                last = part.end_status if part.strand == 1 else part.start_status
                if (i > 0 and first != "exact") or (i < len(feature.parts) - 1 and last != "exact"):
                    raise _invalid(f"{feature.feature_id}: uncertain internal splice boundary")
            raw = feature.extract(sequence)
            edited = list(raw)
            offsets = {
                position: (offset, direction)
                for offset, (position, direction) in enumerate(mapping)
            }
            applied = []
            for site_index, site in enumerate(sites):
                if site["seqid"] != record.seqid or site["position"] - 1 not in offsets:
                    continue
                offset, direction = offsets[site["position"] - 1]
                if site["strand"] != ("+" if direction == 1 else "-"):
                    continue
                allele = site["edited"] if direction == 1 else str(Seq(site["edited"]).complement())
                edited[offset] = allele
                applied.append({**site, "cds_position": offset + 1, "cds_edited_base": allele})
                applied_to[site_index].append(feature.feature_id)
            changed = "".join(edited)
            try:
                offset = int(next(iter(feature.qualifier_values("codon_start")), "1")) - 1
                code = (
                    genetic_code
                    if genetic_code is not None
                    else int(next(iter(feature.qualifier_values("transl_table")), "1"))
                )
            except ValueError as error:
                raise _invalid(f"{feature.feature_id}: invalid codon_start/transl_table") from error
            if offset not in {0, 1, 2} or offset >= len(raw):
                raise _invalid(f"{feature.feature_id}: invalid CDS translation offset")
            table = _table(code)
            stops = CodonTable.ambiguous_dna_by_id[code].stop_codons
            first_part = feature.parts[0]
            partial_5prime = (
                offset != 0
                or (first_part.start_status if first_part.strand == 1 else first_part.end_status)
                != "exact"
            )
            before = _translate(raw[offset:], code, table, partial_5prime)
            after = _translate(changed[offset:], code, table, partial_5prime)
            codons_before = [raw[i : i + 3] for i in range(offset, len(raw) - 2, 3)]
            codons_after = [changed[i : i + 3] for i in range(offset, len(changed) - 2, 3)]
            stop_gain = [
                i + 1
                for i, (a, b) in enumerate(zip(codons_before, codons_after, strict=True))
                if a not in stops and b in stops
            ]
            stop_loss = [
                i + 1
                for i, (a, b) in enumerate(zip(codons_before, codons_after, strict=True))
                if a in stops and b not in stops
            ]
            expected = next(iter(feature.qualifier_values("translation")), None)
            if expected is not None:
                expected = expected.removesuffix("*")
            # Compare raw translated codons, including stops, to keep changes
            # visible when a terminal stop is gained or lost.
            codon_protein_before = _translate(
                raw[offset:], code, table, partial_5prime, keep_stop=True
            )
            codon_protein_after = _translate(
                changed[offset:], code, table, partial_5prime, keep_stop=True
            )
            rows.append(
                {
                    "feature_id": feature.feature_id,
                    "seqid": record.seqid,
                    "gene": next(iter(feature.qualifier_values("gene")), None),
                    "parts": [part.model_dump(mode="json") for part in feature.parts],
                    "genetic_code": code,
                    "codon_start": offset + 1,
                    "partial_5prime": partial_5prime,
                    "partial_3prime": (
                        feature.parts[-1].end_status
                        if feature.parts[-1].strand == 1
                        else feature.parts[-1].start_status
                    )
                    != "exact",
                    "has_terminal_stop_before": bool(codons_before) and codons_before[-1] in stops,
                    "has_terminal_stop_after": bool(codons_after) and codons_after[-1] in stops,
                    "trailing_bases": (len(raw) - offset) % 3,
                    "spliced_cds_before": raw,
                    "spliced_cds_after": changed,
                    "protein_before": before,
                    "protein_after": after,
                    "start_gain": not partial_5prime
                    and codons_before[0] not in table.start_codons
                    and codons_after[0] in table.start_codons
                    if codons_before
                    else False,
                    "start_loss": not partial_5prime
                    and codons_before[0] in table.start_codons
                    and codons_after[0] not in table.start_codons
                    if codons_before
                    else False,
                    "stop_gain_positions": stop_gain,
                    "stop_loss_positions": stop_loss,
                    "amino_acid_changes": [
                        {"position": i + 1, "before": a, "after": b}
                        for i, (a, b) in enumerate(
                            zip(codon_protein_before, codon_protein_after, strict=True)
                        )
                        if a != b
                    ],
                    "internal_stop_positions_before": _internal_stops(codon_protein_before),
                    "internal_stop_positions_after": _internal_stops(codon_protein_after),
                    "annotated_protein_matches_before": before == expected
                    if expected is not None
                    else None,
                    "annotated_protein_matches_after": after == expected
                    if expected is not None
                    else None,
                    "applied_sites": applied,
                    "pseudo": bool(feature.qualifier_values("pseudogene"))
                    or any(qualifier.name == "pseudo" for qualifier in feature.qualifiers),
                }
            )
    if not rows:
        raise _invalid("GenBank contains no CDS features")
    unmapped = [site for site, targets in zip(sites, applied_to, strict=True) if not targets]
    return OrganelleResult(
        operation_id="annotation.translate_edited_cds",
        operation_version="1.0",
        scope=editing_result.scope,
        status="ok",
        summary_text=f"Reconstructed {len(rows)} annotated CDSs; {sum(bool(r['applied_sites']) for r in rows)} edited scenarios.",
        metrics={
            "cds_count": len(rows),
            "edited_cds_count": sum(bool(r["applied_sites"]) for r in rows),
            "site_count": len(raw_sites),
            "excluded_sites": excluded,
            "unmapped_sites": unmapped,
            "cds": rows,
            "coordinate_system": "parts: 0-based half-open; site/CDS/amino-acid positions: 1-based",
            "editing_policy": "selected positive/unquantified edits; exclude zero evidence; joint sites are unphased",
            "source_operation": editing_result.operation_id,
        },
        flags=(
            "supplied_splice_boundaries",
            "selected_edit_scenario_not_phased_haplotype",
            *editing_result.flags,
        ),
    )


def _translate(sequence, code, table, partial_5prime, *, keep_stop=False):
    complete = sequence[: len(sequence) // 3 * 3]
    protein = str(Seq(complete).translate(table=code))
    if not partial_5prime and complete[:3] in table.start_codons and protein:
        protein = "M" + protein[1:]
    return protein[:-1] if not keep_stop and protein.endswith("*") else protein


def _internal_stops(protein):
    return [i + 1 for i, aa in enumerate(protein[:-1]) if aa == "*"]


def write_edited_cds(result: OrganelleResult, *, output: str | Path) -> OrganelleResult:
    """Write paired CDS/protein FASTA and site mapping TSV for edited scenarios."""
    if result.operation_id != "annotation.translate_edited_cds" or result.status != "ok":
        raise _invalid("Expected a successful translate_edited_cds result")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    rows = thaw_json(result.metrics)["cds"]
    artifacts = []
    for field in ("spliced_cds_before", "spliced_cds_after", "protein_before", "protein_after"):
        path = directory / f"{field}.fasta"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(f">{row['feature_id']} gene={row['gene'] or '-'}\n{row[field]}\n")
        artifacts.append(
            ArtifactRef.from_path(path, kind="sequence", format="fasta", media_type="text/plain")
        )
    path = directory / "editing_cds_mapping.tsv"
    fields = [
        "feature_id",
        "seqid",
        "position",
        "strand",
        "ref",
        "edited",
        "cds_position",
        "editing_fraction",
        "depth",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for site in row["applied_sites"]:
                writer.writerow({**site, "feature_id": row["feature_id"]})
    artifacts.append(
        ArtifactRef.from_path(
            path, kind="table", format="tsv", media_type="text/tab-separated-values"
        )
    )
    return result.model_copy(
        update={"operation_id": "annotation.write_edited_cds", "artifacts": tuple(artifacts)}
    )
