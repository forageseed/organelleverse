"""Built-in desktop tool runners for OrganelleVerse Tools UI.

These are lightweight, dependency-cautious implementations that run locally
inside the desktop service.  They write outputs to the allocated run
``output_dir`` and return a summary dict compatible with
the host's tool run service.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_IUPAC_DNA = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "U": "T",
    "R": "AG",
    "Y": "CT",
    "S": "GC",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}


def _iupac_to_regex(pattern: str) -> str:
    """Convert an IUPAC DNA pattern to a case-insensitive regex."""
    parts = []
    for char in pattern.upper():
        if char in _IUPAC_DNA:
            choices = _IUPAC_DNA[char]
            if len(choices) == 1:
                parts.append(choices)
            else:
                parts.append(f"[{choices}]")
        else:
            parts.append(re.escape(char))
    return "".join(parts)


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    comp = {
        "A": "T",
        "T": "A",
        "U": "A",
        "G": "C",
        "C": "G",
        "R": "Y",
        "Y": "R",
        "S": "S",
        "W": "W",
        "K": "M",
        "M": "K",
        "B": "V",
        "D": "H",
        "H": "D",
        "V": "B",
        "N": "N",
    }
    return "".join(comp.get(b, b) for b in reversed(seq.upper()))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require_input(params: dict[str, Any]) -> Path:
    path = params.get("input")
    if not path:
        raise ValueError("'input' file path is required")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    return p


def run_motif(
    params: dict[str, Any],
    *,
    output_dir: Path,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Scan a FASTA for motif occurrences.

    Supports plain regex or IUPAC DNA patterns.  Results are written as a TSV
    plus a short consensus summary.
    """
    from Bio import SeqIO

    input_path = _require_input(params)
    raw_pattern = str(params.get("pattern", "")).strip()
    if not raw_pattern:
        raise ValueError("'pattern' is required")

    strand = str(params.get("strand", "both")).lower()
    if strand not in {"both", "forward", "reverse"}:
        strand = "both"

    use_iupac = bool(params.get("use_iupac", True))
    min_score = float(params.get("min_score", 1.0))
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("'min_score' must be between 0 and 1")

    try:
        regex_pattern = _iupac_to_regex(raw_pattern) if use_iupac else raw_pattern
        compiled = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid motif pattern: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    hits_path = output_dir / "motif_hits.tsv"
    summary_path = output_dir / "motif_summary.txt"

    hits: list[dict[str, Any]] = []
    total_sequences = 0

    with hits_path.open("w", newline="") as tsv, summary_path.open("w") as summary:
        writer = csv.DictWriter(
            tsv,
            fieldnames=["seq_id", "start", "end", "strand", "match", "score"],
            delimiter="\t",
        )
        writer.writeheader()

        for record in SeqIO.parse(str(input_path), "fasta"):
            total_sequences += 1
            seq = str(record.seq).upper()
            seq_id = record.id

            if strand in {"forward", "both"}:
                for m in compiled.finditer(seq):
                    match_seq = m.group(0).upper()
                    score = _motif_score(raw_pattern, match_seq, use_iupac)
                    if score + 1e-9 >= min_score:
                        hit = {
                            "seq_id": seq_id,
                            "start": m.start() + 1,
                            "end": m.end(),
                            "strand": "+",
                            "match": match_seq,
                            "score": round(score, 3),
                        }
                        hits.append(hit)
                        writer.writerow(hit)

            if strand in {"reverse", "both"}:
                rc = _reverse_complement(seq)
                seq_len = len(seq)
                for m in compiled.finditer(rc):
                    match_seq = m.group(0).upper()
                    score = _motif_score(raw_pattern, match_seq, use_iupac)
                    if score + 1e-9 >= min_score:
                        start = seq_len - m.end() + 1
                        end = seq_len - m.start()
                        hit = {
                            "seq_id": seq_id,
                            "start": start,
                            "end": end,
                            "strand": "-",
                            "match": match_seq,
                            "score": round(score, 3),
                        }
                        hits.append(hit)
                        writer.writerow(hit)

        summary.write(f"Motif scan: {raw_pattern}\n")
        summary.write(f"Pattern type: {'IUPAC' if use_iupac else 'regex'}\n")
        summary.write(f"Strand: {strand}\n")
        summary.write(f"Sequences scanned: {total_sequences}\n")
        summary.write(f"Total hits: {len(hits)}\n")

    log(f"Motif scan found {len(hits)} hits in {total_sequences} sequences")

    return {
        "module_id": "motif",
        "status": "ok",
        "summary_text": f"Found {len(hits)} motif hits across {total_sequences} sequences.",
        "observed_metrics": {
            "total_hits": len(hits),
            "sequences_scanned": total_sequences,
            "pattern": raw_pattern,
            "strand": strand,
        },
        "output_files": {
            "hits_tsv": str(hits_path),
            "summary": str(summary_path),
        },
        "completed_at": _now(),
    }


def _motif_score(pattern: str, match: str, use_iupac: bool) -> float:
    """Score a match against the pattern.

    For IUPAC patterns the score is the fraction of positions where the matched
    base is contained in the IUPAC code.  For regex patterns a perfect match
    always scores 1.0.
    """
    if not use_iupac:
        return 1.0
    pat = pattern.upper()
    seq = match.upper()
    if len(pat) != len(seq):
        return 1.0
    ok = 0
    for p, s in zip(pat, seq, strict=True):
        allowed = _IUPAC_DNA.get(p, p)
        if s in allowed:
            ok += 1
    return ok / len(pat)


def run_genecode_parse(
    params: dict[str, Any],
    *,
    output_dir: Path,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Parse a sequence file through GeneCode and emit a document report."""
    from organelleverse.operations.genecode_client import GeneCodeClient

    input_path = _require_input(params)
    client = GeneCodeClient()

    raw = input_path.read_text(encoding="utf-8", errors="replace")
    label = str(params.get("name") or input_path.stem or "Imported Sequence")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = client.parse_sequence(raw, name=label)
    document = result["document"]

    report_path = output_dir / "genecode_parse_report.txt"
    json_path = output_dir / "genecode_parse_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"GeneCode Sequence Parse: {document['name']}",
        f"Format: {document['format']}",
        f"Length: {document['length']} bp",
        f"Topology: {document['topology']}",
        f"Molecule type: {document['molecule_type']}",
        f"GC content: {result['biophysics']['gc_content']}%",
        "Messages:",
    ]
    lines.extend(f"  - {msg}" for msg in result["messages"])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    log(f"Parsed {document['format']} sequence ({document['length']} bp)")

    return {
        "module_id": "genecode_parse",
        "status": "ok",
        "summary_text": f"Parsed {document['format']} sequence ({document['length']} bp): {document['name']}.",
        "observed_metrics": {
            "length": document["length"],
            "gc_content": result["biophysics"]["gc_content"],
            "format": document["format"],
        },
        "output_files": {
            "report": str(report_path),
            "json": str(json_path),
        },
        "completed_at": _now(),
    }


def run_genecode_restriction_sites(
    params: dict[str, Any],
    *,
    output_dir: Path,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Scan insert and optional vector for restriction sites via GeneCode."""
    from organelleverse.operations.genecode_client import GeneCodeClient

    input_path = _require_input(params)
    client = GeneCodeClient()

    sequence = input_path.read_text(encoding="utf-8", errors="replace")
    vector_path = params.get("vector")
    vector_sequence = None
    if vector_path:
        vector_sequence = Path(vector_path).read_text(encoding="utf-8", errors="replace")

    enzymes_raw = str(params.get("enzymes", "")).strip()
    enzymes = [e.strip() for e in enzymes_raw.split(",") if e.strip()] or None

    topology = str(params.get("topology", "linear")).lower()
    vector_topology = str(params.get("vector_topology", "circular")).lower()

    output_dir.mkdir(parents=True, exist_ok=True)
    result = client.scan_restriction_sites(
        sequence,
        vector_sequence=vector_sequence,
        enzymes=enzymes,
        topology=topology,
        vector_topology=vector_topology,
    )
    analysis = result["meta"]["restrictionAnalysis"]

    report_path = output_dir / "genecode_restriction_report.txt"
    json_path = output_dir / "genecode_restriction_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["GeneCode Restriction Site Scan", f"Insert length: {analysis['insertLength']} bp"]
    if vector_sequence:
        lines.append(f"Vector length: {analysis.get('vectorLength')} bp")
    lines.extend(["", "Messages:"])
    lines.extend(f"  - {msg}" for msg in result["messages"])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    safe_cutters = analysis.get("summary", {}).get("insertSafeCutters", 0)
    log(f"Restriction scan: {safe_cutters} insert-safe enzymes found")

    return {
        "module_id": "genecode_restriction_sites",
        "status": "ok",
        "summary_text": f"Restriction scan complete. {safe_cutters} insert-safe enzymes found.",
        "observed_metrics": {
            "insert_length": analysis["insertLength"],
            "insert_safe_cutters": safe_cutters,
            "library_size": analysis.get("summary", {}).get("librarySize", 0),
        },
        "output_files": {
            "report": str(report_path),
            "json": str(json_path),
        },
        "completed_at": _now(),
    }


def run_genecode_rtqpcr(
    params: dict[str, Any],
    *,
    output_dir: Path,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Design RT-qPCR primers using GeneCode."""
    from organelleverse.operations.genecode_client import GeneCodeClient

    input_path = _require_input(params)
    client = GeneCodeClient()

    sequence = input_path.read_text(encoding="utf-8", errors="replace")
    label = str(params.get("name") or input_path.stem or "RT-qPCR")
    species = str(params.get("species") or "Homo sapiens")
    gene_symbol = params.get("gene_symbol") or None
    accession = params.get("accession") or None
    gdna_check = bool(params.get("gdna_check", True))
    include_probe = bool(params.get("include_probe", False))

    output_dir.mkdir(parents=True, exist_ok=True)
    result = client.design_rtqpcr(
        sequence,
        label=label,
        species=species,
        gene_symbol=gene_symbol,
        accession=accession,
        gdna_check=gdna_check,
        include_probe=include_probe,
    )
    candidates = result.get("results", [])

    report_path = output_dir / "genecode_rtqpcr_report.txt"
    json_path = output_dir / "genecode_rtqpcr_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"GeneCode RT-qPCR Design: {label}", ""]
    lines.extend(result["messages"])
    lines.append(f"\nTop candidates returned: {len(candidates)}")
    for idx, candidate in enumerate(candidates[:5], 1):
        lines.append(f"\n  Candidate {idx}:")
        for key, value in candidate.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value[:10])
            lines.append(f"    {key}: {value}")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    log(f"RT-qPCR design returned {len(candidates)} candidate primer sets")

    return {
        "module_id": "genecode_rtqpcr",
        "status": "ok",
        "summary_text": f"RT-qPCR design returned {len(candidates)} candidate primer sets.",
        "observed_metrics": {"candidates": len(candidates)},
        "output_files": {
            "report": str(report_path),
            "json": str(json_path),
        },
        "completed_at": _now(),
    }


def run_genecode_cloning(
    params: dict[str, Any],
    *,
    output_dir: Path,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Design molecular cloning primers/constructs using GeneCode."""
    from organelleverse.operations.genecode_client import GeneCodeClient

    input_path = _require_input(params)
    client = GeneCodeClient()

    raw = input_path.read_text(encoding="utf-8", errors="replace")
    # Accept either a single sequence file or a multi-FASTA of fragments.
    from Bio import SeqIO

    fragments = [str(record.seq) for record in SeqIO.parse(str(input_path), "fasta")]
    if not fragments:
        fragments = [raw]

    method = str(params.get("method", "gibson")).lower()
    label = str(params.get("name") or input_path.stem or "Cloning")
    vector_path = params.get("vector")
    vector_sequence = None
    if vector_path:
        vector_sequence = Path(vector_path).read_text(encoding="utf-8", errors="replace")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = client.design_cloning(
        fragments,
        method=method,
        label=label,
        vector_sequence=vector_sequence,
        left_homology=params.get("left_homology") or None,
        right_homology=params.get("right_homology") or None,
        homology_length=int(params.get("homology_length", 25)),
        type_iis_enzyme=str(params.get("type_iis_enzyme", "BsaI")),
        golden_gate_clamp_length=int(params.get("golden_gate_clamp_length", 4)),
    )
    candidates = result.get("results", [])

    report_path = output_dir / "genecode_cloning_report.txt"
    json_path = output_dir / "genecode_cloning_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"GeneCode Cloning Design: {label}", f"Method: {method}", ""]
    lines.extend(result["messages"])
    lines.append(f"\nTop candidates returned: {len(candidates)}")
    for idx, candidate in enumerate(candidates[:3], 1):
        lines.append(f"\n  Candidate {idx}:")
        for key, value in candidate.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value[:10])
            lines.append(f"    {key}: {value}")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    log(f"Cloning design returned {len(candidates)} candidate constructs")

    return {
        "module_id": "genecode_cloning",
        "status": "ok",
        "summary_text": f"Cloning design ({method}) returned {len(candidates)} candidate constructs.",
        "observed_metrics": {
            "candidates": len(candidates),
            "method": method,
            "fragments": len(fragments),
        },
        "output_files": {
            "report": str(report_path),
            "json": str(json_path),
        },
        "completed_at": _now(),
    }


def run_genecode_sgrna(
    params: dict[str, Any],
    *,
    output_dir: Path,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Design sgRNAs using GeneCode."""
    from organelleverse.operations.genecode_client import GeneCodeClient

    input_path = _require_input(params)
    client = GeneCodeClient()

    sequence = input_path.read_text(encoding="utf-8", errors="replace")
    label = str(params.get("name") or input_path.stem or "sgRNA")
    mode = str(params.get("mode", "ko")).lower()
    pam_set = str(params.get("pam_set", "spcas9_ngg"))

    output_dir.mkdir(parents=True, exist_ok=True)
    result = client.design_sgrna(sequence, mode=mode, pam_set=pam_set, label=label)
    candidates = result.get("results", [])

    report_path = output_dir / "genecode_sgrna_report.txt"
    json_path = output_dir / "genecode_sgrna_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"GeneCode sgRNA Design: {label}", f"Mode: {mode}", f"PAM set: {pam_set}", ""]
    lines.extend(result["messages"])
    lines.append(f"\nTop candidates returned: {len(candidates)}")
    for idx, candidate in enumerate(candidates[:10], 1):
        lines.append(f"\n  Candidate {idx}:")
        for key, value in candidate.items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value[:10])
            lines.append(f"    {key}: {value}")
    report_path.write_text("\n".join(lines), encoding="utf-8")

    log(f"sgRNA design returned {len(candidates)} candidates")

    return {
        "module_id": "genecode_sgrna",
        "status": "ok",
        "summary_text": f"sgRNA design returned {len(candidates)} candidates.",
        "observed_metrics": {
            "candidates": len(candidates),
            "mode": mode,
            "pam_set": pam_set,
        },
        "output_files": {
            "report": str(report_path),
            "json": str(json_path),
        },
        "completed_at": _now(),
    }


# Dispatch table consumed by ToolRunService (legacy path) and the tools.adapter
# plugin capability so both execution routes run the same local implementations.
BUILTIN_TOOL_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "motif": run_motif,
    "genecode_parse": run_genecode_parse,
    "genecode_restriction": run_genecode_restriction_sites,
    "genecode_rtqpcr": run_genecode_rtqpcr,
    "genecode_cloning": run_genecode_cloning,
    "genecode_sgrna": run_genecode_sgrna,
}

# The mito / chloro / general analysis tools run the organelleverse suites —
# real implementations, no simulated reports.
from organelleverse.operations.analysis_runners import ANALYSIS_TOOL_RUNNERS  # noqa: E402

BUILTIN_TOOL_RUNNERS.update(ANALYSIS_TOOL_RUNNERS)

# Image-workbench jobs: identical code on Windows, WSL and SSH targets.
from organelleverse.morphology.image_job import IMAGE_JOB_RUNNERS  # noqa: E402

BUILTIN_TOOL_RUNNERS.update(IMAGE_JOB_RUNNERS)
