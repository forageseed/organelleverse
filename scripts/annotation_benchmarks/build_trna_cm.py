#!/usr/bin/env python
"""Build the packaged CC0 tRNA covariance model.

Source: Rfam RF00005 (tRNA), released under CC0 (public domain) and freely
redistributable. Downloads the covariance model once at build time and writes
it into the package data directory. Never copies GPL tRNAscan-SE models or
barrnap data.

If the Rfam host is unreachable, build locally from a public tRNA Stockholm
alignment with Infernal ``cmbuild`` instead (pass ``--from-stockholm``).
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
import urllib.request
from pathlib import Path

RFAM_URL = "https://rfam.org/family/RF00005/cm"


def fetch_rfam(url: str, out: Path) -> None:
    data = urllib.request.urlopen(url, timeout=120).read()
    if url.endswith(".gz"):
        data = gzip.decompress(data)
    out.write_bytes(data)


def build_from_stockholm(sto: Path, out: Path) -> None:
    subprocess.run(["cmbuild", "-F", str(out), str(sto)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--url", default=RFAM_URL)
    ap.add_argument("--from-stockholm", type=Path, default=None)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.from_stockholm is not None:
        build_from_stockholm(args.from_stockholm, args.out)
    else:
        fetch_rfam(args.url, args.out)

    size = args.out.stat().st_size
    print(f"Wrote {args.out} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
