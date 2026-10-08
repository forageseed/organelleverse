#!/usr/bin/env python
"""Build the mitochondrial exon/intron boundary-flank database.

For every intron-containing PCG in a curated set of seed-plant mitochondrial
GenBank references, extract the conserved sequence flanking each internal
exon/intron junction: 20 bp of exon + 10 bp of intron (mRNA-sense orientation),
labelled ``R`` (donor, an exon's 3' end) or ``L`` (acceptor, an exon's 5' start).

The native boundary finder (see ``mitochondrion/boundary_db.py``) slides these
flanks over a target's approximate junction to pin the exact boundary — a
reference-anchored replacement for hardcoded per-gene offsets, following the
MGAVAS/PMGA approach but built from a much larger, license-clean species set.

Benchmark species are excluded so validation stays honest.
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

EXON_FLANK = 20
INTRON_FLANK = 10

# Genes whose plant-mito orthologs are cis/trans-spliced (have introns).
INTRON_GENES = {
    "ccmfc",
    "nad1",
    "nad2",
    "nad3",
    "nad4",
    "nad5",
    "nad7",
    "rps3",
    "rps10",
    "cox2",
    "cox1",
    "rpl2",
    "ccmfn",
    "rps14",
    "rpl16",
    "sdh3",
}

# Species used for validation/benchmarking — never contribute to the DB.
EXCLUDED_SPECIES = {
    "nicotiana tabacum",
    "pinus taeda",
    "lactuca sativa",
    "ranunculus",
    "oryza sativa",
    "arabidopsis thaliana",
}


def normalize_gene(raw: str) -> str:
    g = re.sub(r"[^a-z0-9]", "", (raw or "").split()[0].lower()) if raw else ""
    m = re.match(r"nd(\d.*)", g)  # nd5 -> nad5 naming variant
    if m:
        g = "nad" + m.group(1)
    if g == "cox21":  # observed typo variant
        g = "cox2"
    return g


def species_of(rec) -> str:
    for feat in rec.features:
        if feat.type == "source":
            return feat.qualifiers.get("organism", ["?"])[0]
    return "?"


def extract_flanks(rec, genome: str):
    """Yield (gene, exon_index, side, flank_seq) for every internal junction."""
    for feat in rec.features:
        if feat.type != "CDS":
            continue
        gene = normalize_gene(feat.qualifiers.get("gene", [""])[0])
        if gene not in INTRON_GENES:
            continue
        parts = feat.location.parts
        if len(parts) < 2:
            continue
        strand = feat.location.strand
        coords = [(int(p.start) + 1, int(p.end)) for p in parts]  # transcription order
        for idx, (s, e) in enumerate(coords):
            # Donor (R): 3' end of this exon (every exon except the last).
            if idx < len(coords) - 1:
                if strand == 1:
                    flank = genome[e - EXON_FLANK : e] + genome[e : e + INTRON_FLANK]
                else:
                    flank = str(Seq(genome[s - 1 : s - 1 + EXON_FLANK]).reverse_complement()) + str(
                        Seq(genome[s - 1 - INTRON_FLANK : s - 1]).reverse_complement()
                    )
                if len(flank) == EXON_FLANK + INTRON_FLANK and set(flank) <= set("ACGT"):
                    yield gene, idx + 1, "R", flank
            # Acceptor (L): 5' start of this exon (every exon except the first).
            if idx > 0:
                if strand == 1:
                    flank = (
                        genome[s - 1 - INTRON_FLANK : s - 1] + genome[s - 1 : s - 1 + EXON_FLANK]
                    )
                else:
                    flank = str(Seq(genome[e : e + INTRON_FLANK]).reverse_complement()) + str(
                        Seq(genome[e - EXON_FLANK : e]).reverse_complement()
                    )
                if len(flank) == EXON_FLANK + INTRON_FLANK and set(flank) <= set("ACGT"):
                    yield gene, idx + 1, "L", flank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gb-dir", default="/home/user/data16t/mito_genome/seed_plant_mitogenomes/genbank"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--max-per-key",
        type=int,
        default=60,
        help="cap distinct flanks per (gene, exon, side) for DB size",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.gb_dir}/**/*.gb", recursive=True))
    # key -> {flank_seq: species} (dedup identical flanks, keep first species)
    db: dict[tuple[str, int, str], dict[str, str]] = defaultdict(dict)
    n_species = 0
    for f in files:
        try:
            rec = next(SeqIO.parse(f, "genbank"))
        except Exception:
            continue
        sp = species_of(rec)
        if any(x in sp.lower() for x in EXCLUDED_SPECIES):
            continue
        genome = str(rec.seq).upper()
        used = False
        for gene, exon_idx, side, flank in extract_flanks(rec, genome):
            bucket = db[(gene, exon_idx, side)]
            if flank not in bucket:
                bucket[flank] = sp.replace(" ", "_")
                used = True
        n_species += used

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w") as fh:
        for (gene, exon_idx, side), bucket in sorted(db.items()):
            for _i, (flank, sp) in enumerate(list(bucket.items())[: args.max_per_key]):
                fh.write(f">{gene}|exon{exon_idx}|{side}|{sp}\n{flank}\n")
                written += 1

    genes = sorted({k[0] for k in db})
    print(f"species used: {n_species}, flanks written: {written}")
    print(f"genes: {', '.join(genes)}")
    for g in genes:
        keys = [k for k in db if k[0] == g]
        tot = sum(len(db[k]) for k in keys)
        print(f"  {g:8} junctions={len(keys):2}  flanks={tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
