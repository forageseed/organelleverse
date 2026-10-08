"""Closed, standard-library-only implementation of PPR code candidate matching."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SUPPORTED_PPR_CODES = {"TN": "A", "TD": "G", "NS": "C", "ND": "U"}
_OPERATION_ID = "ppr.predict_binding_sites"
_OPERATION_VERSION = "1.0"


def predict_binding_sites(
    repeat_tsv: Path,
    transcript_fasta: Path,
    organelle: str,
    extended_code_map_json: str = "{}",
) -> dict[str, object]:
    """Find exact candidate target strings for ordered, pre-annotated P-class repeats.

    Repeat TSV columns: protein_id, repeat_index (1-based N-to-C), and
    repeat_sequence (canonical 35-aa motifs). Transcript FASTA sequences are
    already oriented 5' to 3' as RNA; no strand inference or reverse-complement
    search is performed. The result is sequence-only evidence, not affinity.
    """
    repeat_path = Path(repeat_tsv)
    transcript_path = Path(transcript_fasta)
    parameters = {
        "repeat_tsv": str(repeat_path),
        "transcript_fasta": str(transcript_path),
        "organelle": organelle,
        "extended_code_map_json": extended_code_map_json,
    }
    if organelle not in {"mitochondrion", "plastid"}:
        return _failed(
            "organelle must be mitochondrion or plastid", "invalid_input", "none", parameters
        )
    if not repeat_path.is_file():
        return _failed(
            f"PPR repeat TSV does not exist: {repeat_path}",
            "repeat_tsv_missing",
            organelle,
            parameters,
        )
    if not transcript_path.is_file():
        return _failed(
            f"Transcript FASTA does not exist: {transcript_path}",
            "transcript_fasta_missing",
            organelle,
            parameters,
        )
    try:
        extensions = json.loads(extended_code_map_json)
        if not isinstance(extensions, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in extensions.items()
        ):
            raise ValueError("extended_code_map_json must encode a JSON object of string pairs")
        mapping = _mapping_extensions(extensions)
        proteins = _read_repeats(repeat_path)
        transcripts = _read_fasta(transcript_path)
        sites = []
        protein_targets = []
        for protein_id in sorted(proteins):
            repeats = proteins[protein_id]
            target = _code_target(repeats, mapping)
            protein_targets.append(
                {"protein_id": protein_id, "repeat_count": len(repeats), "target_sequence": target}
            )
            sites.extend(_find_exact_sites(protein_id, target, transcripts))
    except (ValueError, OSError, csv.Error) as error:
        return _failed(str(error), "invalid_input", organelle, parameters)

    return {
        "schema_version": "organelleverse.result.v1",
        "kind": "result",
        "operation_id": _OPERATION_ID,
        "operation_version": _OPERATION_VERSION,
        "scope": organelle,
        "status": "ok",
        "summary_text": (
            f"Found {len(sites)} exact candidate PPR-code sequence match(es) "
            f"across {len(proteins)} pre-annotated protein(s)."
        ),
        "metrics": {
            "proteins_scanned": len(proteins),
            "transcripts_scanned": len(transcripts),
            "candidate_sites": len(sites),
            "protein_targets": protein_targets,
            "sites": sites,
            "supported_default_mapping": dict(SUPPORTED_PPR_CODES),
            "mapping_used": mapping,
            "interpretation": (
                "Exact sequence candidate only; this does not estimate binding affinity, "
                "RNA accessibility, or validate protein-RNA binding."
            ),
        },
        "findings": [
            {
                "kind": "finding",
                "code": "candidate_sites",
                "metric": "candidate_sites",
                "value": len(sites),
            }
        ],
        "flags": ["candidate_sequence_matches"] if sites else [],
        "artifacts": [],
        "provenance": _provenance(parameters),
        "errors": [],
        "suggested_operations": [],
    }


def _read_repeats(path: Path) -> dict[str, tuple[str, ...]]:
    proteins: dict[str, dict[int, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"protein_id", "repeat_index", "repeat_sequence"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(
                "repeat TSV must contain protein_id, repeat_index, repeat_sequence columns"
            )
        for line_number, row in enumerate(reader, start=2):
            protein_id = (row["protein_id"] or "").strip()
            raw_index = (row["repeat_index"] or "").strip()
            repeat = "".join((row["repeat_sequence"] or "").split()).upper()
            if not protein_id:
                raise ValueError(f"repeat TSV line {line_number} has an empty protein_id")
            try:
                index = int(raw_index)
            except ValueError as error:
                raise ValueError(
                    f"repeat TSV line {line_number} has a non-integer repeat_index"
                ) from error
            if index < 1:
                raise ValueError(f"repeat TSV line {line_number} repeat_index must be >= 1")
            if len(repeat) != 35:
                raise ValueError(
                    f"repeat TSV line {line_number} repeat_sequence must be exactly 35 aa"
                )
            if any(residue not in "ACDEFGHIKLMNPQRSTVWY" for residue in repeat):
                raise ValueError(f"repeat TSV line {line_number} has invalid amino acids")
            indexed = proteins.setdefault(protein_id, {})
            if index in indexed:
                raise ValueError(f"protein {protein_id!r} has duplicate repeat_index {index}")
            indexed[index] = repeat
    if not proteins:
        raise ValueError("repeat TSV contains no PPR repeats")
    ordered = {}
    for protein_id, indexed in proteins.items():
        expected_indices = list(range(1, len(indexed) + 1))
        if sorted(indexed) != expected_indices:
            raise ValueError(
                f"protein {protein_id!r} repeat_index values must be contiguous from 1"
            )
        ordered[protein_id] = tuple(indexed[index] for index in expected_indices)
    return ordered


def _read_fasta(path: Path) -> tuple[tuple[str, str], ...]:
    records = []
    identifier = None
    chunks = []
    seen = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    records.append((identifier, "".join(chunks).upper()))
                identifier = line[1:].split(maxsplit=1)[0] if line[1:].strip() else ""
                if not identifier or identifier in seen:
                    raise ValueError(
                        f"transcript FASTA line {line_number} has an empty or duplicate ID"
                    )
                seen.add(identifier)
                chunks = []
            elif identifier is None:
                raise ValueError(
                    f"transcript FASTA sequence precedes first header at line {line_number}"
                )
            else:
                chunks.append("".join(line.split()))
    if identifier is not None:
        records.append((identifier, "".join(chunks).upper()))
    if not records:
        raise ValueError(f"Transcript FASTA contains no records: {path}")
    return tuple(records)


def _mapping_extensions(value: dict[str, str]) -> dict[str, str]:
    mapping = dict(SUPPORTED_PPR_CODES)
    for raw_code, raw_base in value.items():
        code = raw_code.strip().upper()
        base = raw_base.strip().upper().replace("T", "U")
        if len(code) != 2 or any(residue not in "ACDEFGHIKLMNPQRSTVWY" for residue in code):
            raise ValueError(f"invalid PPR code {raw_code!r}; expected two amino-acid letters")
        if base not in {"A", "C", "G", "U"}:
            raise ValueError(f"mapping for {code} must be one RNA base: A, C, G, or U")
        if code in SUPPORTED_PPR_CODES and SUPPORTED_PPR_CODES[code] != base:
            raise ValueError(f"extension cannot override supported mapping {code}")
        mapping[code] = base
    return mapping


def _code_target(repeats: tuple[str, ...], mapping: dict[str, str]) -> str:
    codes = tuple(repeat[4] + repeat[34] for repeat in repeats)
    unknown = sorted(set(codes) - set(mapping))
    if unknown:
        raise ValueError(
            "no experimentally supported or user-supplied target mapping for PPR code(s): "
            + ", ".join(unknown)
        )
    return "".join(mapping[code] for code in codes)


def _find_exact_sites(
    protein_id: str,
    target: str,
    transcripts: tuple[tuple[str, str], ...],
) -> list[dict[str, int | str]]:
    hits = []
    for transcript_id, raw_sequence in transcripts:
        sequence = raw_sequence.upper().replace("T", "U")
        start = 0
        while True:
            offset = sequence.find(target, start)
            if offset < 0:
                break
            hits.append(
                {
                    "protein_id": protein_id,
                    "transcript_id": transcript_id,
                    "start_1based": offset + 1,
                    "end_1based": offset + len(target),
                    "target_sequence": target,
                    "repeat_count": len(target),
                }
            )
            start = offset + 1
    return hits


def _failed(
    message: str, code: str, scope: str, parameters: dict[str, object]
) -> dict[str, object]:
    return {
        "schema_version": "organelleverse.result.v1",
        "kind": "result",
        "operation_id": _OPERATION_ID,
        "operation_version": _OPERATION_VERSION,
        "scope": scope,
        "status": "failed",
        "summary_text": message,
        "metrics": {},
        "findings": [],
        "flags": [],
        "artifacts": [],
        "provenance": _provenance(parameters),
        "errors": [{"kind": "error", "code": f"ppr.{code}", "message": message, "details": {}}],
        "suggested_operations": [],
    }


def _provenance(parameters: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(
        parameters, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    try:
        package_version = version("organelleverse")
    except PackageNotFoundError:
        package_version = "0.0.1"
    now = datetime.now(UTC).isoformat()
    return {
        "kind": "provenance",
        "operation_id": _OPERATION_ID,
        "operation_version": _OPERATION_VERSION,
        "package_version": package_version,
        "git_commit": os.environ.get("ORG_VERSE_GIT_COMMIT", ""),
        "parameters_hash": hashlib.sha256(encoded).hexdigest(),
        "callable_locator": "ppr_predict:predict_binding_sites",
        "actual_backend": "ppr_code_exact_match",
        "attempted_backends": ["ppr_code_exact_match"],
        "started_at": now,
        "finished_at": now,
        "duration_seconds": 0.0,
    }
