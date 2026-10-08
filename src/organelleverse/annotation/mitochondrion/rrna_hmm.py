"""Native rRNA detection via pyhmmer ``nhmmer``.

Barrnap-equivalent method (profile HMM search) with no external binary and no
GPL-licensed profile data: each packaged rRNA reference sequence is used as an
``nhmmer`` query, and pyhmmer builds the profile HMM on the fly. This keeps the
default rRNA path fully native (pyhmmer is a compiled wheel, not an external
program).
"""

from __future__ import annotations

from pathlib import Path

import pyhmmer
from Bio import SeqIO

from .db import DBManager
from .rrna import RawRRNA

# Minimum score to accept an nhmmer hit; filters spurious short matches.
_MIN_SCORE = 30.0


def _rrna_type_from_gene(gene: str) -> str:
    """Map a packaged rRNA gene name (rrn5/rrn18/rrn26/...) to an rRNA type label."""
    key = gene.lower()
    table = {
        "rrn5": "5S",
        "rrn45": "4.5S",
        "rrn4.5": "4.5S",
        "rrn16": "16S",
        "rrn18": "18S",
        "rrn23": "23S",
        "rrn26": "26S",
        "rrn28": "28S",
    }
    for prefix, label in table.items():
        if key.startswith(prefix):
            return label
    return gene.upper()


def _gene_from_reference(ref_file: Path) -> str:
    """Derive the gene name from a reference FASTA filename (e.g. rrn18.mito.fasta -> rrn18)."""
    return ref_file.name.split(".")[0]


def annotate_rrna_hmm(fasta_path: Path, output_dir: Path, db_manager: DBManager) -> list[RawRRNA]:
    ref_dir = db_manager.rrna_ref_dir
    if not ref_dir.is_dir():
        return []
    ref_files = sorted(ref_dir.glob("*.fasta"))
    if not ref_files:
        return []

    alphabet = pyhmmer.easel.Alphabet.dna()

    from .fasta import load_fasta

    # One target: the pipeline's merged genome (contigs joined by 200 N). Scanning the records
    # separately gave contig-local coordinates with no record identity, which land on the first
    # contig once read in the merged space every writer uses.
    try:
        genome = load_fasta(fasta_path)
    except ValueError:  # no records
        return []
    targets = [
        pyhmmer.easel.TextSequence(
            name=genome.seqid.encode(), sequence=genome.sequence.replace("U", "T")
        ).digitize(alphabet)
    ]

    hits: list[RawRRNA] = []
    for ref_file in ref_files:
        gene = _gene_from_reference(ref_file)
        rrna_type = _rrna_type_from_gene(gene)
        queries = []
        for rec in SeqIO.parse(str(ref_file), "fasta"):
            seq = str(rec.seq).upper().replace("U", "T")
            if not seq:
                continue
            queries.append(
                pyhmmer.easel.TextSequence(name=rec.id.encode(), sequence=seq).digitize(alphabet)
            )
        if not queries:
            continue

        builder = pyhmmer.plan7.Builder(alphabet)
        for top_hits in pyhmmer.nhmmer(queries, targets, builder=builder):
            for hit in top_hits:
                if hit.score < _MIN_SCORE:
                    continue
                for dom in hit.domains:
                    ali = dom.alignment
                    t_from = int(ali.target_from)
                    t_to = int(ali.target_to)
                    strand = 1 if t_from <= t_to else -1
                    start = min(t_from, t_to)
                    end = max(t_from, t_to)
                    hits.append(
                        RawRRNA(
                            gene_name=gene,
                            start=start,
                            end=end,
                            strand=strand,
                            score=float(hit.score),
                            rrna_type=rrna_type,
                            source_tool="pyhmmer-nhmmer",
                        )
                    )
    return hits
