#!/usr/bin/env python
"""Build the packaged CC0 tRNA profile HMM used as the genome-scale pre-filter.

Source: the Rfam RF00005 (tRNA) seed alignment (CC0, public domain). The seed
Stockholm alignment is downloaded once and an HMM is built from it with pyhmmer
(native; no external binary). The HMM is used by the native cmsearch engine to
find candidate tRNA windows genome-wide in ~0.1 s before the accurate CM CYK
runs on each window. Never uses GPL tRNAscan-SE data.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pyhmmer

RFAM_SEED_URL = "https://rfam.org/family/RF00005/alignment?acc=RF00005&format=stockholm&download=1"


def build_hmm(seed_sto: Path, out_hmm: Path) -> int:
    alphabet = pyhmmer.easel.Alphabet.dna()
    with pyhmmer.easel.MSAFile(str(seed_sto), digital=True, alphabet=alphabet) as mf:
        msa = mf.read()
    msa.name = b"tRNA"
    builder = pyhmmer.plan7.Builder(alphabet)
    background = pyhmmer.plan7.Background(alphabet)
    hmm, _, _ = builder.build_msa(msa, background)
    out_hmm.parent.mkdir(parents=True, exist_ok=True)
    with open(out_hmm, "wb") as fh:
        hmm.write(fh)
    return hmm.M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--seed",
        type=Path,
        default=None,
        help="local RF00005 seed Stockholm; downloaded if omitted",
    )
    ap.add_argument("--url", default=RFAM_SEED_URL)
    args = ap.parse_args()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = args.out.parent / "RF00005_seed.sto"
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.write_bytes(urllib.request.urlopen(args.url, timeout=120).read())

    m = build_hmm(seed, args.out)
    print(f"Wrote {args.out} (M={m})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
