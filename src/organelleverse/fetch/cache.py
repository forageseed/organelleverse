"""Baseline, incremental merge, and resume — a 58k-record pull will be interrupted.

CLAUDE.md requires any job over ~10 samples to checkpoint and resume. A whole-
corpus organelle pull is 58,082 plastomes; it *will* be killed halfway at least
once. So metadata is written as it arrives (JSONL, one record per line, append-
only), and a rerun reads what is already there and asks NCBI only for the rest.

``baseline`` is the same idea across runs: point at last month's metadata and
only the new records are fetched.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..core.errors import OrganelleInputError

__all__ = [
    "append_records",
    "merge_with_baseline",
    "read_baseline",
    "read_jsonl",
    "write_jsonl",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL metadata file, skipping a truncated final line.

    A killed process often leaves a half-written last line. That is expected,
    not corruption — drop it and carry on.
    """
    file = Path(path)
    if not file.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue  # truncated tail from an interrupted run
        if isinstance(value, dict):
            records.append(value)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return file


def append_records(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Append as we go, so an interrupted run keeps everything it already had."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
    return file


def _accessions_in(records: Iterable[dict[str, Any]]) -> Iterator[str]:
    for record in records:
        accession = str(record.get("accession", "")).strip()
        if accession:
            yield accession


def read_baseline(path: str | Path) -> set[str]:
    """Accessions already held, from JSONL metadata or a plain accession list.

    Accepts either, because a baseline is often just a text file someone pasted.
    """
    file = Path(path)
    if not file.exists():
        raise OrganelleInputError(
            code="input.baseline_not_found",
            message=f"baseline does not exist: {file}",
            details={"path": str(file)},
            suggested_action={"check": "omit baseline= for a fresh pull"},
        )
    if file.suffix in {".jsonl", ".json"}:
        return set(_accessions_in(read_jsonl(file)))

    accessions: set[str] = set()
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        token = line.strip().split(",")[0].split("\t")[0].strip()
        if token and not token.startswith("#") and token.lower() != "accession":
            accessions.add(token)
    return accessions


def merge_with_baseline(
    baseline_path: str | Path,
    new_records: list[dict[str, Any]],
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Union of baseline and new records, de-duplicated on accession.

    The new record wins on collision: NCBI updates records in place, so the
    freshly fetched copy is the truthful one.
    """
    existing = read_jsonl(baseline_path) if Path(baseline_path).is_file() else []
    by_accession: dict[str, dict[str, Any]] = {
        str(r.get("accession", "")): r for r in existing if r.get("accession")
    }
    for record in new_records:
        accession = str(record.get("accession", ""))
        if accession:
            by_accession[accession] = record
    merged = [by_accession[key] for key in sorted(by_accession)]
    write_jsonl(merged, output_path)
    return merged
