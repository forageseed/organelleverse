"""Configurable, sequence-only ORF discovery using the orfipy engine."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from Bio.Data import CodonTable
from Bio.Seq import Seq

from .._bio import read_fasta, write_fasta
from ..core.artifacts import ArtifactRef
from ..core.errors import OrganelleDependencyError, OrganelleInputError, OrganelleParameterError
from ..core.frozen import thaw_json
from ..core.result import OrganelleResult


def find_orfs(
    genome_fasta: Path,
    *,
    organelle: Literal["none", "mitochondrion", "plastid"] = "none",
    genetic_code: int = 1,
    min_aa: int = 30,
    start_mode: Literal["atg", "table", "any"] = "atg",
    strand: Literal["both", "+", "-"] = "both",
    include_nested: bool = False,
    include_partial: bool = False,
    circular: bool = False,
) -> OrganelleResult:
    """Discover ORFs per FASTA record and retain original coordinates and proteins.

    Coordinates are 1-based inclusive; ``parts`` are in transcript order.
    Circular ORFs may cross the origin once, with total CDS length <= record
    length. Partial ORFs are linear-only. Nested mode retains alternative
    same-frame starts ending at the same stop. Any-start mode means stop-to-stop
    regions, without changing the first amino acid to methionine.
    """
    if organelle not in {"none", "mitochondrion", "plastid"}:
        raise _parameter_error("organelle must be none, mitochondrion, or plastid")
    if min_aa < 1:
        raise _parameter_error("min_aa must be >= 1")
    if start_mode not in {"atg", "table", "any"} or strand not in {"both", "+", "-"}:
        raise _parameter_error("unsupported start_mode or strand")
    if circular and include_partial:
        raise _parameter_error("circular records have no termini; partial ORFs are linear-only")
    try:
        table = CodonTable.unambiguous_dna_by_id[genetic_code]
    except KeyError as error:
        raise _parameter_error(f"unsupported genetic code: {genetic_code}") from error
    if set(table.stop_codons) & set(table.forward_table):
        raise _parameter_error("context-dependent dual-coding stop tables are not supported")
    starts = ["ATG"] if start_mode == "atg" else table.start_codons
    stops = CodonTable.ambiguous_dna_by_id[genetic_code].stop_codons
    try:
        import orfipy_core
    except ImportError as error:
        raise OrganelleDependencyError(
            code="dependency.orfipy_missing",
            message="ORF discovery requires orfipy; install organelleverse[orf]",
        ) from error
    try:
        records = read_fasta(genome_fasta)
    except (OSError, ValueError) as error:
        raise OrganelleInputError(code="input.invalid_orf_fasta", message=str(error)) from error
    rows = []
    for sequence_id, raw in records:
        sequence = raw.upper()
        if not sequence or set(sequence) - set("ACGTRYSWKMBDHVN"):
            raise OrganelleInputError(
                code="input.invalid_orf_sequence",
                message=f"{sequence_id}: expected a nonempty IUPAC DNA sequence",
            )
        n = len(sequence)
        hits = []
        for direction in ("+", "-") if strand == "both" else (strand,):
            oriented = sequence if direction == "+" else str(Seq(sequence).reverse_complement())
            search_sequence = oriented * 2 if circular else oriented
            # Retain broad intervals before selecting starts within one revolution.
            # Applying maxlen=n in the engine can discard valid nested circular starts.
            engine_hits = orfipy_core.orfs(
                search_sequence,
                minlen=3,
                maxlen=len(search_sequence),
                strand="f",
                starts=list(starts),
                stops=stops,
                include_stop=True,
                partial3=include_partial,
                partial5=include_partial,
                between_stops=start_mode == "any",
            )
            for start, end, _, description in engine_hits:
                kind = description.split("ORF_type=", 1)[1].split(";", 1)[0]
                has_stop = search_sequence[end - 3 : end] in stops
                coding_end = end - 3 if has_stop else end
                partial5 = kind == "5-prime-partial"
                if start_mode == "any":
                    upstream = (
                        "".join(oriented[(start - 3 + k) % n] for k in range(3))
                        if circular
                        else search_sequence[start - 3 : start]
                        if start >= 3
                        else ""
                    )
                    partial5 = upstream not in stops
                if circular:
                    if not has_stop or partial5:
                        continue
                elif not include_partial and (partial5 or not has_stop):
                    continue
                possible_starts = [start]
                if start_mode != "any" and (include_nested or circular):
                    possible_starts = [
                        pos
                        for pos in range(start, coding_end, 3)
                        if search_sequence[pos : pos + 3] in starts
                    ]
                    if partial5 and not circular:
                        possible_starts.insert(0, start)
                valid_starts = [
                    pos for pos in possible_starts if pos < n and (not circular or end - pos <= n)
                ]
                if not include_nested:
                    valid_starts = valid_starts[:1]
                for pos in valid_starts:
                    if (coding_end - pos) // 3 < min_aa:
                        continue
                    is_partial5 = partial5 and pos == start
                    dna = search_sequence[pos:end]
                    translated_dna = dna[:-3] if has_stop else dna
                    protein = str(Seq(translated_dna).translate(table=genetic_code))
                    has_start = start_mode != "any" and not is_partial5
                    if has_start:
                        protein = "M" + protein[1:]
                    spans = [(pos, min(end, n))]
                    if end > n:
                        spans.append((0, end - n))
                    parts = [
                        {"start": a + 1, "end": b}
                        if direction == "+"
                        else {"start": n - b + 1, "end": n - a}
                        for a, b in spans
                    ]
                    hits.append(
                        {
                            "sequence_id": sequence_id,
                            "start": min(part["start"] for part in parts),
                            "end": max(part["end"] for part in parts),
                            "parts": parts,
                            "strand": direction,
                            "frame": (pos % 3 + 1) * (1 if direction == "+" else -1),
                            "wraps_origin": end > n,
                            "partial_5prime": is_partial5,
                            "partial_3prime": not has_stop,
                            "start_codon": dna[:3] if has_start else None,
                            "stop_codon": dna[-3:] if has_stop else None,
                            "length_aa": len(protein),
                            "cds_sequence": dna,
                            "protein_sequence": protein,
                            "ambiguous_bases": sum(base not in "ACGT" for base in dna),
                        }
                    )
        hits.sort(key=lambda row: (row["start"], row["end"], row["strand"], row["frame"]))
        for row in hits:
            row["id"] = f"orf_{len(rows) + 1}"
            rows.append(row)
    parameters = {
        "organelle": organelle,
        "genetic_code": genetic_code,
        "min_aa": min_aa,
        "start_mode": start_mode,
        "strand": strand,
        "include_nested": include_nested,
        "include_partial": include_partial,
        "circular": circular,
    }
    return OrganelleResult(
        operation_id="annotation.find_orfs",
        scope=organelle,
        status="ok",
        summary_text=f"{len(rows)} sequence-only ORFs across {len(records)} records.",
        metrics={
            "orf_count": len(rows),
            "records_scanned": len(records),
            "sequence_lengths": {name: len(sequence) for name, sequence in records},
            "orfs": rows,
            "parameters": parameters,
            "backend": "orfipy",
            "backend_version": version("orfipy"),
            "coordinate_system": "1-based inclusive; parts in transcript order",
            "interpretation": "Genomic sequence candidates; RNA editing, splicing, expression, and CMS causality are not tested.",
        },
        flags=("sequence_only_orf_candidates",),
    )


def write_orfs(result: OrganelleResult, *, output: str | Path) -> OrganelleResult:
    """Write candidate CDS/protein FASTA and GFF3 from annotation.find_orfs results."""
    if result.operation_id != "annotation.find_orfs" or result.status != "ok":
        raise OrganelleInputError(
            code="input.invalid_orf_result",
            message="expected a successful annotation.find_orfs result",
        )
    rows = thaw_json(result.metrics)["orfs"]
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    cds = write_fasta(
        destination / "orfs.cds.fasta", [(row["id"], row["cds_sequence"]) for row in rows]
    )
    proteins = write_fasta(
        destination / "orfs.proteins.fasta", [(row["id"], row["protein_sequence"]) for row in rows]
    )
    metrics = thaw_json(result.metrics)
    lines = ["##gff-version 3"] + [
        f"##sequence-region {quote(name, safe='._-:')} 1 {length}"
        for name, length in metrics["sequence_lengths"].items()
    ]
    for row in rows:
        consumed = 0
        for part in row["parts"]:
            attributes = (
                f"ID={row['id']};Name={row['id']};"
                f"partial_5prime={str(row['partial_5prime']).lower()};"
                f"partial_3prime={str(row['partial_3prime']).lower()};"
                f"wraps_origin={str(row['wraps_origin']).lower()};"
                f"transl_table={metrics['parameters']['genetic_code']};Note=sequence_only_candidate"
            )
            lines.append(
                "\t".join(
                    [
                        quote(row["sequence_id"], safe="._-:"),
                        "orfipy",
                        "CDS",
                        str(part["start"]),
                        str(part["end"]),
                        ".",
                        row["strand"],
                        str((-consumed) % 3),
                        attributes,
                    ]
                )
            )
            consumed += part["end"] - part["start"] + 1
    gff = destination / "orfs.gff3"
    gff.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = tuple(
        ArtifactRef.from_path(path, kind="sequence" if fmt == "fasta" else "annotation", format=fmt)
        for path, fmt in ((cds, "fasta"), (proteins, "fasta"), (gff, "gff3"))
    )
    return OrganelleResult(
        operation_id="annotation.write_orfs",
        scope=result.scope,
        status="ok",
        summary_text=f"Wrote {len(rows)} ORFs as CDS/protein FASTA and GFF3.",
        metrics=result.metrics,
        artifacts=artifacts,
        flags=result.flags,
    )


def _parameter_error(message: str) -> OrganelleParameterError:
    return OrganelleParameterError(code="parameters.invalid_orf_options", message=message)
