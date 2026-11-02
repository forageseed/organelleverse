"""Gene-order synteny must not depend on which strand a genome was submitted on."""

from __future__ import annotations

from pathlib import Path

from organelleverse.comparative.compare import synteny
from organelleverse.comparative.compare_core import compute_synteny_score
from organelleverse.io_genome import read_genbank_genome

_GENES = ["psbA", "matK", "rbcL", "atpB", "petD"]


def _genbank(path: Path, genes: list[tuple[str, int, int, str]], length: int = 6000) -> Path:
    lines = [
        f"LOCUS       fake          {length} bp    DNA     circular     01-JAN-2026",
        "DEFINITION  synthetic.",
        "ACCESSION   FAKE",
        "VERSION     FAKE.1",
        "FEATURES             Location/Qualifiers",
    ]
    for name, start, end, strand in genes:
        loc = f"{start}..{end}" if strand == "+" else f"complement({start}..{end})"
        lines.append(f"     CDS             {loc}")
        lines.append(f'                     /gene="{name}"')
    lines.append("ORIGIN")
    seq = "ACGT" * (length // 4)
    lines += [f"{i + 1:>9} {seq[i:i + 60]}" for i in range(0, len(seq), 60)]
    lines.append("//")
    path.write_text("\n".join(lines) + "\n")
    return path


def _genomes(tmp_path: Path):
    forward = [(name, 100 + 1000 * i, 400 + 1000 * i, "+") for i, name in enumerate(_GENES)]
    # the same molecule written from the other strand: coordinates mirrored, order reversed
    reverse = [(name, 6001 - e, 6001 - s, "-") for name, s, e, _ in forward]
    return [
        read_genbank_genome(_genbank(tmp_path / "f.gb", forward), organelle="plastid", species="Forward"),
        read_genbank_genome(_genbank(tmp_path / "r.gb", reverse), organelle="plastid", species="Reverse"),
    ]


def test_reverse_strand_genome_keeps_every_adjacency(tmp_path: Path) -> None:
    genomes = _genomes(tmp_path)
    metrics = synteny(genomes).metrics
    assert metrics["order_jaccard"] == 1.0
    assert metrics["synteny_blocks"] == len(_GENES) - 1
    assert compute_synteny_score(genomes) == {"synteny_blocks": len(_GENES) - 1, "order_jaccard": 1.0}
