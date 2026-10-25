"""Plastid-derived (MTPT) pieces of a mitochondrial assembly graph, and ovasm run inputs.

MTPTs are found with the transfer suite's aligner and thresholds
(``transfer.transfer_core.alignment_hsps``: LOSAT, then NCBI blastn; word
size 11, e-value 1e-5, >= 80 % identity, >= 100 bp). HSPs are merged on each
mitochondrial segment exactly like ``detect_mtpt`` merges them on a genome;
each merged fragment keeps its donor blocks on the plastid reference so a
figure can connect the two. A segment is ``shared`` when MTPT covers at least
:data:`SHARED_FRACTION` of it (the same sequence is in both assemblies),
``mitochondrial`` otherwise; partial MTPTs are reported as fragments.

Whether an MTPT is really integrated is read evidence: a long read that
covers the fragment plus ``anchor`` bases of mitochondrial sequence on both
sides inside the segment. Plastid reads cannot do that.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .._bio import read_fasta
from ..core.errors import OrganelleDependencyError, OrganelleInputError
from .graph_conformation import AssemblyGraph, Probe, flip, node_label, parse_node_label

SHARED_FRACTION = 0.9
DONOR_GAP = 1000


def plastid_reference(path: str | Path) -> dict[str, Any]:
    """Representative plastid molecule (first FASTA record) with LSC/IRb/SSC/IRa if found."""
    from ..comparative.orientation import _quadripartite

    source = Path(path)
    if source.is_dir():
        source = source / "molecules.fasta"
    records = read_fasta(source)
    if not records:
        raise OrganelleInputError(
            code="structure.plastid_empty",
            message="The plastid FASTA has no records.",
            details={"path": str(source)},
        )
    name, sequence = records[0]
    sequence = sequence.upper()
    return {
        "path": str(source),
        "name": name.split()[0],
        "length": len(sequence),
        "regions": _quadripartite(sequence) or {},
        "sequence": sequence,
    }


def find_mtpts(
    graph: AssemblyGraph,
    plastid: dict[str, Any],
    *,
    min_length: int = 100,
    min_identity: float = 80.0,
) -> dict[str, Any]:
    """MTPT fragments per mitochondrial segment with their plastid donor blocks."""
    from ..transfer.transfer_core import _merge_hits, alignment_hsps

    with tempfile.TemporaryDirectory(prefix="ov_mtpt_") as tmp:
        target = Path(tmp) / "segments.fa"
        source = Path(tmp) / "plastid.fa"
        target.write_text(
            "".join(f">{s.name}\n{s.sequence}\n" for s in graph.segments.values() if s.sequence)
        )
        source.write_text(f">{plastid['name']}\n{plastid['sequence']}\n")
        found = alignment_hsps(target, source, min_len=min_length, min_identity=min_identity)
    if found is None:
        raise OrganelleDependencyError(
            code="structure.mtpt_aligner_missing",
            message="MTPT detection needs LOSAT or NCBI blastn.",
            details={"tools": ["losat", "blastn"]},
            retryable=False,
            suggested_action={"install": "mamba install -c bioconda blast"},
        )
    # _merge_hits carries each HSP (with donor coordinates) into its merged fragment.
    merged = _merge_hits(
        [
            (h["seqid"], h["start"], h["end"], h["identity"], h["strand"], {**h, "source_seqid": h["source_id"]})
            for h in found["hsps"]
        ]
    )
    fragments = []
    for index, fragment in enumerate(sorted(merged, key=lambda f: (f["seqid"], f["start"])), start=1):
        inside = fragment["hsps"]
        fragments.append(
            {
                "id": f"MTPT{index}",
                "segment": fragment["seqid"],
                "start": fragment["start"],
                "end": fragment["end"],
                "length": fragment["length"],
                "identity": fragment["identity"],
                "strand": fragment["strand"],
                "donor_blocks": _donor_blocks(inside),
            }
        )
    covered: dict[str, int] = {}
    for fragment in fragments:
        covered[fragment["segment"]] = covered.get(fragment["segment"], 0) + fragment["length"]
    return {
        "backend": found["backend"],
        "attempted": found["attempted"],
        "min_length": min_length,
        "min_identity": min_identity,
        "fragments": fragments,
        "plastid_fraction": {
            name: round(covered.get(name, 0) / s.length, 4) if s.length else 0.0
            for name, s in graph.segments.items()
        },
    }


def _donor_blocks(hsps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge a fragment's HSPs into contiguous donor intervals on the plastid."""
    blocks: list[dict[str, Any]] = []
    for hsp in sorted(hsps, key=lambda h: h["source_start"]):
        last = blocks[-1] if blocks else None
        if last is not None and hsp["source_start"] <= last["end"] + DONOR_GAP:
            last["end"] = max(last["end"], hsp["source_end"])
            last["_bases"] += hsp["length"]
            last["_ident"] += hsp["identity"] * hsp["length"]
        else:
            blocks.append(
                {
                    "start": hsp["source_start"],
                    "end": hsp["source_end"],
                    "_bases": hsp["length"],
                    "_ident": hsp["identity"] * hsp["length"],
                }
            )
    for block in blocks:
        block["identity"] = round(block.pop("_ident") / block.pop("_bases"), 2)
    return blocks


def segment_origin(fraction: float) -> str:
    return "shared" if fraction >= SHARED_FRACTION else "mitochondrial"


def mtpt_probes(graph: AssemblyGraph, fragments: list[dict[str, Any]], *, flank: int) -> list[Probe]:
    """One probe per MTPT: the fragment with up to ``flank`` segment bases on each side."""
    probes = []
    for fragment in fragments:
        sequence = graph.segments[fragment["segment"]].sequence
        lo = max(0, fragment["start"] - 1 - flank)
        hi = min(len(sequence), fragment["end"] + flank)
        probes.append(
            Probe(
                fragment["id"],
                ((fragment["segment"], "+"), (fragment["segment"], "+")),
                sequence[lo:hi],
                fragment["start"] - 1 - lo,
                fragment["length"],
                hi - fragment["end"],
            )
        )
    return probes


# ------------------------------------------------------------------- ovasm run


def load_ovasm_run(path: str | Path) -> dict[str, Any]:
    """Inputs from an ``ovasm run`` directory (one organelle, or ``organelle="both"``).

    Returns the mitochondrial ``organelle.gfa``, the plastid directory when the
    run assembled both, the repeat pairings of ``evidence.json`` translated to
    ``organelle.gfa`` names, and whether linearization was decisive.
    """
    root = Path(path)
    mito = root / "mitochondrion" if (root / "mitochondrion").is_dir() else root
    plastid = root / "plastid" if (root / "mitochondrion").is_dir() and (root / "plastid").is_dir() else None
    gfa = mito / "organelle.gfa"
    if not gfa.is_file():
        raise OrganelleInputError(
            code="structure.ovasm_run_incomplete",
            message="The ovasm run directory has no organelle.gfa.",
            details={"path": str(mito)},
        )
    out: dict[str, Any] = {"gfa": gfa, "plastid": plastid, "pairings": {}, "decisive": None}
    evidence_path = mito / "evidence.json"
    if evidence_path.is_file():
        out["pairings"] = _translate_pairings(mito, json.loads(evidence_path.read_text()))
    linearization = mito / "linearization.json"
    if linearization.is_file():
        components = json.loads(linearization.read_text()).get("components", [])
        out["decisive"] = all(c.get("decisive", False) for c in components) if components else None
    return out


def _translate_pairings(mito: Path, evidence: dict[str, Any]) -> dict[str, list]:
    """``evidence.json`` names assembly.gfa segments; organelle.gfa prefixes and may cut them."""
    from .graph_conformation import read_assembly_graph

    meta = json.loads((mito / "organelle.json").read_text())
    organelle = read_assembly_graph(mito / "organelle.gfa")
    assembly = read_assembly_graph(mito / "assembly.gfa")
    prefix = f"{meta['sample']}."
    pieces: dict[str, list[str]] = {}
    for name in organelle.segments:
        base = name[len(prefix) :]
        head, _, tail = base.rpartition(".")
        original = head if tail.isdigit() and head in assembly.segments else base
        pieces.setdefault(original, []).append(name)
    for original, names in pieces.items():
        names.sort(key=lambda n: [int(x) if x.isdigit() else x for x in n.split(".")])
        # Normalization trims link overlaps from piece ends, so pieces are
        # substrings; what matters is that they occur in this order.
        whole = assembly.segments[original].sequence
        first = whole.find(organelle.segments[names[0]].sequence)
        last = whole.rfind(organelle.segments[names[-1]].sequence)
        if first < 0 or last < 0 or (len(names) > 1 and last <= first):
            raise OrganelleInputError(
                code="structure.ovasm_pieces_unordered",
                message="organelle.gfa pieces do not occur in order in their assembly.gfa segment.",
                details={"segment": original, "pieces": names},
            )

    def end(label: str, *, leaving: bool) -> str:
        name, orient = parse_node_label(label)
        names = pieces[name]
        # Leaving a forward segment (or entering a reverse one) uses its right end.
        right = (orient == "+") == leaving
        return node_label((names[-1] if right else names[0], orient))

    translated: dict[str, list] = {}
    for row in evidence.get("repeats", []):
        repeat, orient = parse_node_label(row["repeat"])
        if len(pieces[repeat]) != 1:
            continue
        entries = [end(e, leaving=True) for e in row["entries"]]
        exits = [end(x, leaving=False) for x in row["exits"]]
        straight, crossed = row["pairing_reads"]
        matchings = [
            ([(entries[0], exits[0]), (entries[1], exits[1])], int(straight)),
            ([(entries[0], exits[1]), (entries[1], exits[0])], int(crossed)),
        ]
        name = pieces[repeat][0]
        if orient == "-":
            matchings = [
                ([(_flip_label(s), _flip_label(p)) for p, s in pairs], reads) for pairs, reads in matchings
            ]
        translated[name] = matchings
    return translated


def _flip_label(label: str) -> str:
    return node_label(flip(parse_node_label(label)))


__all__ = [
    "SHARED_FRACTION",
    "find_mtpts",
    "load_ovasm_run",
    "mtpt_probes",
    "plastid_reference",
    "segment_origin",
]
