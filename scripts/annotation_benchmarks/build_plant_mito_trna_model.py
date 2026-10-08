#!/usr/bin/env python
"""Build the plant-mitochondrial-specific tRNA CM + HMM.

The general Rfam RF00005 tRNA model cannot detect (or even HMM-filter-anchor)
several divergent plant-mitochondrial tRNAs (e.g. trnL-UAA, trnS-CGA). This
script builds a plant-mito-specific model:

1. Extract mature tRNA sequences (Biopython splices ``join()`` intron features)
   from public plant-mitochondrial GenBank genomes, excluding any benchmark
   species to avoid train/test leakage.
2. ``cmalign`` them to the RF00005 CM to get a structural Stockholm alignment.
3. ``cmbuild`` a plant-mito CM; build a p7 HMM filter from the same alignment.

Uses Infernal (``cmalign``/``cmbuild``) and pyhmmer at build time only; the
shipped artifacts derive from RF00005 (CC0) structure plus public NCBI sequence
data and are freely redistributable. GPL tRNAscan-SE models are never used.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
from pathlib import Path

import pyhmmer
from Bio import SeqIO

# Benchmark species excluded from training to avoid train/test leakage.
DEFAULT_EXCLUDE = ("Lactuca_sativa", "Nicotiana_tabacum", "Pinus_taeda", "Ranunculus")


def extract_trnas(
    genomes_dir: Path, exclude: tuple[str, ...], out_fa: Path, min_len: int = 60, max_len: int = 95
) -> int:
    gbs = sorted(glob.glob(str(genomes_dir / "*.gb")))
    gbs = [g for g in gbs if not any(x.lower() in g.lower() for x in exclude)]
    seen: set[str] = set()
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_fa.open("w") as fh:
        for gb in gbs:
            try:
                records = list(SeqIO.parse(gb, "genbank"))
            except Exception:
                continue
            for rec in records:
                for feat in rec.features:
                    if feat.type != "tRNA":
                        continue
                    try:
                        s = str(feat.extract(rec.seq)).upper().replace("U", "T")
                    except Exception:
                        continue
                    if not (min_len <= len(s) <= max_len) or (set(s) - set("ACGT")):
                        continue
                    if s in seen:
                        continue
                    seen.add(s)
                    fh.write(f">t{n}\n{s}\n")
                    n += 1
    return n


def build_hmm(sto: Path, out_hmm: Path, name: str) -> int:
    alphabet = pyhmmer.easel.Alphabet.dna()
    with pyhmmer.easel.MSAFile(str(sto), digital=True, alphabet=alphabet) as mf:
        msa = mf.read()
    msa.name = name.encode()
    builder = pyhmmer.plan7.Builder(alphabet)
    background = pyhmmer.plan7.Background(alphabet)
    hmm, _, _ = builder.build_msa(msa, background)
    with open(out_hmm, "wb") as fh:
        hmm.write(fh)
    return hmm.M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--genomes-dir", required=True, type=Path, help="directory of plant-mito *.gb genomes"
    )
    ap.add_argument("--base-cm", required=True, type=Path, help="RF00005 CM for cmalign")
    ap.add_argument("--out-cm", required=True, type=Path)
    ap.add_argument("--out-hmm", required=True, type=Path)
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    ap.add_argument("--work", type=Path, default=Path("plant_mito_build"))
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    fa = args.work / "trnas.fa"
    sto = args.work / "aligned.sto"

    n = extract_trnas(args.genomes_dir, tuple(args.exclude), fa)
    print(f"Extracted {n} unique mature tRNA sequences")

    subprocess.run(
        ["cmalign", "--noprob", "-o", str(sto), str(args.base_cm), str(fa)],
        check=True,
        capture_output=True,
    )
    args.out_cm.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmbuild", "-F", "--hand", str(args.out_cm), str(sto)], check=True, capture_output=True
    )
    m = build_hmm(sto, args.out_hmm, "plant_mito_tRNA")
    print(f"Wrote {args.out_cm} and {args.out_hmm} (HMM M={m})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
