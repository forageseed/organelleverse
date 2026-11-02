"""Deterministic toy population/pangenome data for the popgen figure tests and bundles.

Reference ``REF`` (2,000 bp, genes geneA/geneB/trnX) and eight haploid samples
in two groups (``wild`` W1-W4, ``crop`` C1-C4):

* SNPs: positions 51-451 (step 50) fixed between the groups (high F_ST);
  1001-1451 (step 50) polymorphic only within ``wild`` (π_wild / π_crop peak
  over geneB's neighbour); 1601-1901 (step 100) shared polymorphism.
* one 3-bp deletion record at 1700 carried by C2 (non-SNP ALT).
* graph: nodes n1 (1-600), n2 (601-900), n3 (901-1400), n5 (1401-2000) and a
  200-bp insertion n4; W4, C1-C3 lack n2; C1-C3 carry n4 between n3 and n5.

Everything is built from a fixed seed, so files and expected values are stable.
"""

from __future__ import annotations

import random
from pathlib import Path

LENGTH = 2000
WILD = ["W1", "W2", "W3", "W4"]
CROP = ["C1", "C2", "C3", "C4"]
SAMPLES = WILD + CROP
GROUPS = {**{s: "wild" for s in WILD}, **{s: "crop" for s in CROP}}
FIXED = list(range(51, 452, 50))
WILD_ONLY = list(range(1001, 1452, 50))
SHARED = list(range(1601, 1902, 100))
DELETION = (1700, "C2")
LACK_N2 = {"W4", "C1", "C2", "C3"}
CARRY_N4 = {"C1", "C2", "C3"}
NODE_SPANS = {"n1": (1, 600), "n2": (601, 900), "n3": (901, 1400), "n5": (1401, 2000)}
INSERT_LENGTH = 200


def _rng():
    return random.Random(20261011)


def reference() -> str:
    rng = _rng()
    return "".join(rng.choice("ACGT") for _ in range(LENGTH))


def insertion() -> str:
    rng = random.Random(7)
    return "".join(rng.choice("ACGT") for _ in range(INSERT_LENGTH))


def _alt(base: str) -> str:
    return {"A": "G", "C": "T", "G": "A", "T": "C"}[base]


def genotypes() -> dict[int, dict[str, str]]:
    """Haploid allele (``0``/``1``) per SNP position and sample."""
    calls: dict[int, dict[str, str]] = {}
    for pos in FIXED:
        calls[pos] = {s: ("1" if s in CROP else "0") for s in SAMPLES}
    for i, pos in enumerate(WILD_ONLY):
        carriers = {WILD[i % 4], WILD[(i + 1) % 4]}
        calls[pos] = {s: ("1" if s in carriers else "0") for s in SAMPLES}
    for i, pos in enumerate(SHARED):
        carriers = {WILD[i % 4], CROP[i % 4]}
        calls[pos] = {s: ("1" if s in carriers else "0") for s in SAMPLES}
    return calls


def sample_sequence(sample: str) -> str:
    seq = list(reference())
    for pos, calls in genotypes().items():
        if calls[sample] == "1":
            seq[pos - 1] = _alt(seq[pos - 1])
    return "".join(seq)


def vcf_text() -> str:
    ref = reference()
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=REF,length={LENGTH}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(SAMPLES),
    ]
    records = []
    for pos, calls in genotypes().items():
        base = ref[pos - 1]
        records.append((pos, base, _alt(base), [calls[s] for s in SAMPLES]))
    pos, carrier = DELETION
    records.append((pos, ref[pos - 1 : pos + 3], ref[pos - 1], ["1" if s == carrier else "0" for s in SAMPLES]))
    for pos, ref_allele, alt_allele, gts in sorted(records):
        lines.append(f"REF\t{pos}\t.\t{ref_allele}\t{alt_allele}\t60\tPASS\t.\tGT\t" + "\t".join(gts))
    return "\n".join(lines) + "\n"


def alignment_text() -> str:
    """REF plus every sample, SNP alleles only (no gaps) - the VCF as an alignment."""
    rows = [("REF", reference())] + [(s, sample_sequence(s)) for s in SAMPLES]
    return "".join(f">{name}\n{seq}\n" for name, seq in rows)


def reference_fasta_text() -> str:
    return f">REF\n{reference()}\n"


def genbank_text() -> str:
    seq = reference().lower()
    origin = []
    for i in range(0, len(seq), 60):
        chunk = " ".join(seq[j : j + 10] for j in range(i, min(i + 60, len(seq)), 10))
        origin.append(f"{i + 1:>9} {chunk}")
    return (
        f"LOCUS       REF                     {LENGTH} bp    DNA     circular PLN 11-OCT-2026\n"
        "DEFINITION  Toy plant mitochondrion, complete genome.\n"
        "ACCESSION   REF\n"
        "VERSION     REF\n"
        "SOURCE      mitochondrion Toy plant\n"
        "  ORGANISM  Toy plant\n"
        "FEATURES             Location/Qualifiers\n"
        f"     source          1..{LENGTH}\n"
        '                     /organism="Toy plant"\n'
        '                     /organelle="mitochondrion"\n'
        "     gene            101..400\n"
        '                     /gene="geneA"\n'
        "     CDS             101..400\n"
        '                     /gene="geneA"\n'
        "     gene            complement(1001..1300)\n"
        '                     /gene="geneB"\n'
        "     CDS             complement(1001..1300)\n"
        '                     /gene="geneB"\n'
        "     gene            1501..1572\n"
        '                     /gene="trnX"\n'
        "     tRNA            1501..1572\n"
        '                     /gene="trnX"\n'
        "ORIGIN\n" + "\n".join(origin) + "\n//\n"
    )


def gfa_text() -> str:
    """PGGB-style GFA (PanSN paths); node sequences from the reference / insertion."""
    ref = reference()
    nodes = {name: ref[a - 1 : b] for name, (a, b) in NODE_SPANS.items()}
    nodes["n4"] = insertion()
    lines = ["H\tVN:Z:1.0"]
    lines += [f"S\t{name}\t{seq}" for name, seq in nodes.items()]
    links = {("n1", "n2"), ("n2", "n3"), ("n1", "n3"), ("n3", "n5"), ("n3", "n4"), ("n4", "n5")}
    lines += [f"L\t{a}\t+\t{b}\t+\t0M" for a, b in sorted(links)]
    walks = {"REF": ["n1", "n2", "n3", "n5"]}
    for sample in SAMPLES:
        walk = ["n1"] + ([] if sample in LACK_N2 else ["n2"]) + ["n3"]
        walk += (["n4"] if sample in CARRY_N4 else []) + ["n5"]
        walks[sample] = walk
    for sample, walk in walks.items():
        lines.append(f"P\t{sample}#1#1\t" + ",".join(f"{n}+" for n in walk) + "\t*")
    return "\n".join(lines) + "\n"


def paf_text() -> str:
    """Sample-to-REF alignments matching the graph coverage (query = sample#1#1)."""
    lines = []
    for sample in SAMPLES:
        spans = [(0, 600), (900, 2000)] if sample in LACK_N2 else [(0, 2000)]
        for a, b in spans:
            lines.append(
                f"{sample}#1#1\t2200\t{a}\t{b}\t+\tREF\t{LENGTH}\t{a}\t{b}\t{b - a}\t{b - a}\t60"
            )
    return "\n".join(lines) + "\n"


FILES = {
    "toy.vcf": vcf_text,
    "toy.aln.fasta": alignment_text,
    "ref.fasta": reference_fasta_text,
    "ref.gb": genbank_text,
    "toy.gfa": gfa_text,
    "toy.paf": paf_text,
}


def write_all(directory: str | Path) -> dict[str, Path]:
    """Write every toy file into ``directory``; returns name -> path."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, build in FILES.items():
        path = target / name
        path.write_text(build())
        paths[name] = path
    return paths
