"""Native plastid rRNA detection via pyhmmer ``nhmmer``.

Detects the four chloroplast rRNAs (16S/23S/4.5S/5S, typically two copies each in
the inverted repeat) using packaged plastid rRNA reference sequences as
``nhmmer`` queries — the same barrnap-equivalent, fully-native method used for
mitochondrial rRNA. No external program and no GPL profile data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyhmmer

_REF_DIR = Path(__file__).resolve().parent.parent / "data" / "plastome" / "rrna_refs"

# rRNA sub-unit minimum bit-scores (16S/23S are long, 4.5S/5S short).
_MIN_SCORE = {"16S": 200.0, "23S": 300.0, "4.5S": 25.0, "5S": 25.0}


@dataclass
class PlastidRRNAHit:
    gene_name: str
    rrna_type: str
    start: int
    end: int
    strand: int
    score: float


def default_rrna_ref_dir() -> Path:
    return _REF_DIR


def _rrna_type_from_gene(gene: str) -> str:
    key = gene.lower()
    table = {"rrn5": "5S", "rrn45": "4.5S", "rrn16": "16S", "rrn23": "23S"}
    for prefix, label in table.items():
        if key.startswith(prefix):
            return label
    return gene.upper()


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    name, buf = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(buf)))
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def annotate_plastid_rrna(genome: str, ref_dir: Path | None = None) -> list[PlastidRRNAHit]:
    ref_dir = ref_dir or _REF_DIR
    if not ref_dir.is_dir():
        return []
    ref_files = sorted(ref_dir.glob("*.fasta"))
    if not ref_files:
        return []

    alphabet = pyhmmer.easel.Alphabet.dna()
    genome = genome.upper().replace("U", "T")
    target = pyhmmer.easel.TextSequence(name=b"genome", sequence=genome).digitize(alphabet)

    hits: list[PlastidRRNAHit] = []
    for ref_file in ref_files:
        gene = ref_file.stem  # rrn16 / rrn23 / rrn45 / rrn5
        rrna_type = _rrna_type_from_gene(gene)
        min_score = _MIN_SCORE.get(rrna_type, 30.0)
        queries = [
            pyhmmer.easel.TextSequence(
                name=n.encode(), sequence=s.upper().replace("U", "T")
            ).digitize(alphabet)
            for n, s in _read_fasta(ref_file)
            if s
        ]
        if not queries:
            continue
        builder = pyhmmer.plan7.Builder(alphabet)
        for top in pyhmmer.nhmmer(queries, [target], builder=builder):
            for hit in top:
                if hit.score < min_score:
                    continue
                for dom in hit.domains:
                    ali = dom.alignment
                    t_from, t_to = int(ali.target_from), int(ali.target_to)
                    hits.append(
                        PlastidRRNAHit(
                            gene_name=gene,
                            rrna_type=rrna_type,
                            start=min(t_from, t_to),
                            end=max(t_from, t_to),
                            strand=1 if t_from <= t_to else -1,
                            score=float(hit.score),
                        )
                    )
    return _dedupe(hits)


def _dedupe(hits: list[PlastidRRNAHit]) -> list[PlastidRRNAHit]:
    """Keep the highest-scoring hit among overlapping same-type intervals."""
    chosen: list[PlastidRRNAHit] = []
    for h in sorted(hits, key=lambda x: x.score, reverse=True):
        if any(
            h.rrna_type == c.rrna_type and not (h.end < c.start or h.start > c.end) for c in chosen
        ):
            continue
        chosen.append(h)
    return sorted(chosen, key=lambda x: (x.start, x.end))
