#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

FIELDS = ["sample", "gene", "start", "end", "strand", "anticodon", "source", "sequence", "label"]
_DNA_COMPLEMENT = str.maketrans("ACGTUacgtuNn", "TGCAAtgcaaNn")


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text().splitlines():
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0]
            records.setdefault(current, [])
            continue
        if current is None:
            raise ValueError(f"FASTA sequence line found before header in {path}")
        records[current].append(line.strip())
    return {name: "".join(parts).upper().replace("U", "T") for name, parts in records.items()}


def parse_gff_attrs(text: str) -> dict[str, str]:
    attrs = {}
    for item in text.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            attrs[key] = value
    return attrs


def normalize_gene(name: str) -> str:
    value = name.strip()
    match = re.fullmatch(r"(trn[A-Za-z]{1,2})[-(]([ACGTUacgtuNn]{3})\)?", value)
    if match:
        return f"{match.group(1)}({match.group(2).upper().replace('U', 'T').lower()})"
    return value


def anticodon_from_gene(gene: str) -> str:
    match = re.search(r"\(([ACGTUacgtuNn]{3})\)", gene)
    if not match:
        return "nnn"
    return match.group(1).lower().replace("u", "t")


def rows_from_pmga_gff(path: Path, sample: str, sequences: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    parent_name: dict[str, str] = {}
    parsed = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        attrs = parse_gff_attrs(parts[8])
        name = attrs.get("gene") or attrs.get("Name") or attrs.get("Alias") or ""
        if attrs.get("ID") and name:
            parent_name[attrs["ID"]] = name
        parsed.append((parts, attrs))

    for parts, attrs in parsed:
        if parts[2] != "tRNA":
            continue
        name = attrs.get("gene") or attrs.get("Name") or ""
        if not name and attrs.get("Parent"):
            name = parent_name.get(attrs["Parent"], "")
        gene = normalize_gene(name)
        sequence = extract_sequence(sequences, parts[0], parts[3], parts[4], parts[6])
        if not sequence:
            continue
        rows.append(
            {
                "sample": sample,
                "gene": gene,
                "start": str(min(int(parts[3]), int(parts[4]))),
                "end": str(max(int(parts[3]), int(parts[4]))),
                "strand": parts[6],
                "anticodon": anticodon_from_gene(gene),
                "source": "PMGA",
                "sequence": sequence,
                "label": "positive",
            }
        )
    return rows


def read_benchmark_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def hard_negatives_from_benchmark(
    path: Path,
    sample: str,
    sequences: dict[str, str],
    positives: list[dict[str, str]],
    *,
    negative_method: str,
    overlap_threshold: float,
) -> list[dict[str, str]]:
    rows = []
    for row in read_benchmark_rows(path):
        if row.get("method") != negative_method:
            continue
        if row.get("feature_type") != "tRNA":
            continue
        if row.get("status", "").startswith("skipped"):
            continue
        if overlaps_any(row, positives, overlap_threshold):
            continue
        sequence = extract_sequence(
            sequences,
            row.get("sample", ""),
            row.get("start", ""),
            row.get("end", ""),
            row.get("strand", "+"),
        )
        if not sequence:
            sequence = extract_sequence(
                sequences, "", row.get("start", ""), row.get("end", ""), row.get("strand", "+")
            )
        if not sequence:
            continue
        gene = normalize_gene(row.get("gene", ""))
        rows.append(
            {
                "sample": sample,
                "gene": gene,
                "start": str(min(int(row["start"]), int(row["end"]))),
                "end": str(max(int(row["start"]), int(row["end"]))),
                "strand": row.get("strand", "+"),
                "anticodon": anticodon_from_gene(gene),
                "source": negative_method,
                "sequence": sequence,
                "label": "hard_negative",
            }
        )
    return rows


def extract_sequence(
    sequences: dict[str, str], seqid: str, start: str, end: str, strand: str
) -> str:
    try:
        start_i, end_i = sorted((int(start), int(end)))
    except (TypeError, ValueError):
        return ""
    sequence = sequences.get(seqid)
    if sequence is None and len(sequences) == 1:
        sequence = next(iter(sequences.values()))
    if sequence is None or start_i < 1 or end_i > len(sequence):
        return ""
    fragment = sequence[start_i - 1 : end_i]
    if strand == "-":
        return reverse_complement(fragment)
    return fragment


def overlaps_any(row: dict[str, str], positives: list[dict[str, str]], threshold: float) -> bool:
    return any(interval_overlap_fraction(row, positive) >= threshold for positive in positives)


def interval_overlap_fraction(a: dict[str, str], b: dict[str, str]) -> float:
    try:
        a0, a1 = sorted((int(a["start"]), int(a["end"])))
        b0, b1 = sorted((int(b["start"]), int(b["end"])))
    except (KeyError, TypeError, ValueError):
        return 0.0
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    shorter = min(a1 - a0 + 1, b1 - b0 + 1)
    return overlap / shorter if shorter else 0.0


def reverse_complement(seq: str) -> str:
    return seq.translate(_DNA_COMPLEMENT)[::-1].upper().replace("U", "T")


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--pmga-gff", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--positive-output", required=True, type=Path)
    parser.add_argument("--negative-output", required=True, type=Path)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--negative-method", default="OrganelleVerse-native")
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    sequences = read_fasta(args.fasta)
    positives = rows_from_pmga_gff(args.pmga_gff, args.sample, sequences)
    negatives = hard_negatives_from_benchmark(
        args.benchmark,
        args.sample,
        sequences,
        positives,
        negative_method=args.negative_method,
        overlap_threshold=args.overlap_threshold,
    )
    write_rows(args.positive_output, positives)
    write_rows(args.negative_output, negatives)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
