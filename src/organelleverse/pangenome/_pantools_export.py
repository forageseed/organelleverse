"""Convert PanTools 4.3.5 property exports to exact overlapping GFA paths.

Independent interoperability implementation; no PanTools implementation is copied.
PanTools nucleotide names are sequences, and degenerate names may collide: their
source addresses plus edge localization distinguish the actual physical nodes.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from .graph import path_sequences

_SEQUENCE_LABELS = {"nucleotide", "degenerate"}


def export_to_gfa(
    nodes_path: Path, edges_path: Path, expected: dict[str, str], output: Path
) -> dict:
    """Write GFA only after every localized path exactly matches its FASTA input."""
    groups = []
    current = None
    with nodes_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 4:
                raise ValueError("Invalid PanTools node property row")
            name, label, prop, value = row
            if current is None or current[0:2] != (name, label) or prop in current[2]:
                current = (name, label, {})
                groups.append(current)
            current[2][prop] = value
    segments = {}
    lookup = defaultdict(list)
    molecules = {}
    metadata = {}
    for name, label, props in groups:
        if label == "pangenome":
            metadata.update(props)
        elif label == "sequence":
            key = (int(props["genome"]), int(props["number"]))
            title = props["title"]
            if (
                key in molecules
                or title not in expected
                or len(expected[title]) != int(props["length"])
            ):
                raise ValueError("PanTools sequence identity/length does not match input")
            molecules[key] = title
        elif label in _SEQUENCE_LABELS:
            address = tuple(map(int, props["address"].split(",")))
            if len(address) != 3 or int(props["length"]) != len(name):
                raise ValueError("Invalid PanTools node address/length")
            node = str(len(segments) + 1)
            segments[node] = {"sequence": name, "label": label, "address": address}
            lookup[(label, name)].append(node)
    if set(molecules.values()) != set(expected) or len(molecules) != len(expected):
        raise ValueError("PanTools export is missing input molecule paths")
    k = int(metadata["k_mer_size"])
    if len(segments) != int(metadata["num_nodes"]):
        raise ValueError("PanTools exported node count disagrees with database metadata")

    def resolve(label, sequence, molecule, start):
        candidates = lookup[(label, sequence)]
        if label == "degenerate":
            candidates = [n for n in candidates if segments[n]["address"] == (*molecule, start)]
        if len(candidates) != 1:
            raise ValueError(
                "PanTools node identity cannot be resolved unambiguously from localization"
            )
        return candidates[0]

    occurrences = defaultdict(dict)
    links = set()
    predecessors = defaultdict(dict)
    with edges_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 7:
                raise ValueError("Invalid PanTools relationship property row")
            kind, left_label, left, right_label, right, prop, value = row
            if kind == "has":
                continue
            if kind not in {"FF", "FR", "RF", "RR"}:
                raise ValueError(f"Unsupported PanTools relationship type: {kind}")
            if right_label == "sequence":
                continue  # terminal context node is not a sequence graph segment
            match = re.fullmatch(r"G(\d+)S(\d+)", prop)
            if right_label not in _SEQUENCE_LABELS or match is None:
                raise ValueError("PanTools graph edge lacks molecule localization")
            molecule = tuple(map(int, match.groups()))
            if molecule not in molecules:
                raise ValueError("PanTools edge refers to an unknown molecule")
            for position in map(int, value.split(",")):
                target = resolve(right_label, right, molecule, position)
                target_orientation = "+" if kind[1] == "F" else "-"
                occurrence = (target, target_orientation)
                previous = occurrences[molecule].setdefault(position, occurrence)
                if previous != occurrence:
                    raise ValueError("Multiple PanTools nodes claim the same molecule coordinate")
                if left_label == "sequence":
                    if left != f"{molecule[0]}_{molecule[1]}" or position != 0:
                        raise ValueError("Invalid PanTools initial molecule edge")
                elif left_label in _SEQUENCE_LABELS:
                    source_start = position - len(left) + k - 1
                    source = resolve(left_label, left, molecule, source_start)
                    source_orientation = "+" if kind[0] == "F" else "-"
                    predecessor = (source_start, source, source_orientation)
                    old = predecessors[molecule].setdefault(position, predecessor)
                    if old != predecessor:
                        raise ValueError("PanTools path has conflicting predecessors")
                    links.add((source, source_orientation, target, target_orientation))
                else:
                    raise ValueError("Unexpected PanTools edge endpoint")
    rows = ["H\tVN:Z:1.0", *(f"S\t{node}\t{data['sequence']}" for node, data in segments.items())]
    rows += [f"L\t{a}\t{ao}\t{b}\t{bo}\t{k - 1}M" for a, ao, b, bo in sorted(links)]
    path_rows = []
    for molecule, title in molecules.items():
        steps = sorted(occurrences[molecule].items())
        if not steps or steps[0][0] != 0:
            raise ValueError(f"PanTools did not represent molecule: {title}")
        for index, (start, (node, orientation)) in enumerate(steps):
            if index:
                previous_start, (previous_node, previous_orientation) = steps[index - 1]
                if predecessors[molecule].get(start) != (
                    previous_start,
                    previous_node,
                    previous_orientation,
                ):
                    raise ValueError("PanTools localized path adjacency is inconsistent")
            path_rows.append(
                {
                    "path": title,
                    "step": index,
                    "node": node,
                    "orientation": orientation,
                    "start": start,
                    "end": start + len(segments[node]["sequence"]),
                }
            )
        rows.append("P\t" + title + "\t" + ",".join(n + o for _, (n, o) in steps) + "\t*")
    output.write_text("\n".join(rows) + "\n")
    try:
        actual = path_sequences(output)
        if actual != {name: sequence.upper() for name, sequence in expected.items()}:
            raise ValueError(
                "PanTools reconstructed GFA paths do not exactly spell every input molecule"
            )
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {
        "k": k,
        "node_count": len(segments),
        "edge_count": len(links),
        "path_count": len(molecules),
        "node_identity": segments,
        "path_coordinates": path_rows,
        "path_sequence_verified": True,
        "overlap_policy": "exact k-1 bases",
        "source_format": "PanTools 4.3.5 export_pangenome properties",
    }
