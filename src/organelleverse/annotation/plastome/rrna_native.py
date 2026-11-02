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
            pieces = []
            for index, hit in enumerate(top):
                for dom in hit.domains:
                    ali = dom.alignment
                    t_from, t_to = int(ali.target_from), int(ali.target_to)
                    pieces.append(
                        _Piece(
                            start=min(t_from, t_to),
                            end=max(t_from, t_to),
                            strand=1 if t_from <= t_to else -1,
                            q_from=int(ali.hmm_from),
                            q_to=int(ali.hmm_to),
                            q_length=int(ali.hmm_length),
                            scores={index: float(hit.score)},
                        )
                    )
            for piece in _join_split_pieces(pieces):
                if piece.score < min_score:
                    continue
                hits.append(
                    PlastidRRNAHit(
                        gene_name=gene,
                        rrna_type=rrna_type,
                        start=piece.start,
                        end=piece.end,
                        strand=piece.strand,
                        score=piece.score,
                    )
                )
    return _dedupe(hits)


# A fast-evolving lineage can carry an insertion that the packaged (mostly angiosperm) queries
# lack. Welwitschia 5S is 148 nt against 119-121 nt queries, and nhmmer reports its two halves as
# separate hits (query 2-60 and 35-121, adjacent on the genome), which would give two 5S fragments
# per copy. Pieces of one query that continue each other are joined before the score filter.
_MAX_PIECE_GAP = 30  # genome bp between consecutive pieces
_MAX_SPAN_FACTOR = 1.5  # joined span relative to the query length


@dataclass
class _Piece:
    start: int
    end: int
    strand: int
    q_from: int
    q_to: int
    q_length: int
    scores: dict[int, float]  # nhmmer hit index -> hit score (a hit can hold several domains)

    @property
    def score(self) -> float:
        return sum(self.scores.values())


def _continues(first: _Piece, second: _Piece) -> bool:
    """``second`` follows ``first`` in transcription order on the genome and on the query."""
    gap = second.start - first.end - 1 if first.strand == 1 else first.start - second.end - 1
    span = max(first.end, second.end) - min(first.start, second.start) + 1
    return (
        first.strand == second.strand
        and -5 <= gap <= _MAX_PIECE_GAP
        and second.q_from > first.q_from
        and second.q_to > first.q_to
        and span <= _MAX_SPAN_FACTOR * first.q_length
    )


def _join_split_pieces(pieces: list[_Piece]) -> list[_Piece]:
    joined: list[_Piece] = []
    for strand in (1, -1):
        ordered = sorted(
            (p for p in pieces if p.strand == strand),
            key=lambda p: p.start if strand == 1 else -p.end,
        )
        current: _Piece | None = None
        for piece in ordered:
            if current is not None and _continues(current, piece):
                current = _Piece(
                    start=min(current.start, piece.start),
                    end=max(current.end, piece.end),
                    strand=strand,
                    q_from=current.q_from,
                    q_to=piece.q_to,
                    q_length=current.q_length,
                    scores={**current.scores, **piece.scores},
                )
                continue
            if current is not None:
                joined.append(current)
            current = piece
        if current is not None:
            joined.append(current)
    return joined


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
