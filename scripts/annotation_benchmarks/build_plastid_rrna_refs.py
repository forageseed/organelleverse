#!/usr/bin/env python
"""Build the packaged plastid rRNA reference set for native rRNA detection.

Extracts chloroplast rRNA sequences (16S/23S/4.5S/5S) from plastid GenBank
genomes and writes a few length-diverse representatives per type into per-gene
FASTA files (rrn16/rrn23/rrn45/rrn5). The native rRNA detector runs pyhmmer
``nhmmer`` with these as queries (barrnap-equivalent, no external program, no
GPL data). Public sequence data; freely redistributable.
"""

from __future__ import annotations

import argparse
import glob
import statistics
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO

# Records far from their type's median length are annotation errors (Lactuca
# "4.5S" spanning 263 bp of spacer, Ginkgo 135 bp "5S"). Picking representatives
# by length always selected them as the longest, and nhmmer then stretched every
# hit in every genome over the same flanks. Lineage-specific lengths (grass 95 bp
# 4.5S, cycad/grass 2860-2888 bp 23S) stay within 10% of the median.
MAX_LENGTH_DEVIATION = 0.10


def _rrna_type(name: str) -> str | None:
    n = name.lower().replace("_", "").replace("-", "").replace(".", "").replace(" ", "")
    if "rrn16" in n or "16sr" in n or "16sribos" in n:
        return "rrn16"
    if "rrn23" in n or "23sr" in n:
        return "rrn23"
    if "rrn45" in n or "45sr" in n:
        return "rrn45"
    if "rrn5" in n or ("5sr" in n and "45" not in n):
        return "rrn5"
    return None


def build(genomes_dir: Path, out_dir: Path, exclude: tuple[str, ...], per_type: int = 5) -> dict:
    by_type: dict[str, set[str]] = defaultdict(set)
    for gb in sorted(glob.glob(str(genomes_dir / "*.gb"))):
        if any(x.lower() in gb.lower() for x in exclude):
            continue
        for rec in SeqIO.parse(gb, "genbank"):
            for f in rec.features:
                if f.type != "rRNA":
                    continue
                nm = f.qualifiers.get("gene", [""])[0] or f.qualifiers.get("product", [""])[0]
                t = _rrna_type(nm)
                if not t:
                    continue
                try:
                    s = str(f.extract(rec.seq)).upper().replace("U", "T")
                except Exception:
                    continue
                if s and not (set(s) - set("ACGTN")):
                    by_type[t].add(s)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for t, seqs in by_type.items():
        median = statistics.median(len(s) for s in seqs)
        typical = [s for s in seqs if abs(len(s) - median) <= MAX_LENGTH_DEVIATION * median]
        # pick length-diverse representatives among typical-length records
        ordered = sorted(typical, key=len)
        if len(ordered) > per_type:
            idx = [round(i * (len(ordered) - 1) / (per_type - 1)) for i in range(per_type)]
            ordered = [ordered[i] for i in sorted(set(idx))]
        with (out_dir / f"{t}.fasta").open("w") as fh:
            for i, s in enumerate(ordered):
                fh.write(f">{t}_ref{i}\n{s}\n")
        counts[t] = len(ordered)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genomes-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--per-type", type=int, default=5)
    args = ap.parse_args()
    counts = build(args.genomes_dir, args.out_dir, tuple(args.exclude), args.per_type)
    print(f"Wrote plastid rRNA refs to {args.out_dir}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
