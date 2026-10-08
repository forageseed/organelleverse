"""Numeric segment names for ODGI.

ODGI 0.9 parses GFA segment names as integers (``odgi build`` fails with ``Error parsing
segment 'S1.u0': stol``), while OV-GFA names segments ``<sample>.<name>[.<piece>]`` and other
backends use names such as ``s1`` or ``utg000001l``. Before a graph goes to ODGI its segments
are renamed 1..N in file order; everything else is kept, and ``node_names.tsv`` maps each
number back to its original name.
"""

from __future__ import annotations

import re
from pathlib import Path

_NUMERIC = re.compile(r"[1-9][0-9]*")
_WALK_STEP = re.compile(r"([<>])([^<>]+)")


def _is_numeric(name: str) -> bool:
    return _NUMERIC.fullmatch(name) is not None


def numeric_gfa(source: str | Path, destination: str | Path) -> dict[str, str] | None:
    """Write ``source`` to ``destination`` with segments renamed 1..N.

    Returns ``{number: original name}``, or ``None`` when every name is already a positive
    integer (``destination`` is then not written and ``source`` can be used as it is).
    """
    lines = Path(source).read_text().splitlines()
    names: list[str] = []
    for line in lines:
        if line.startswith("S\t"):
            names.append(line.split("\t", 2)[1])
    if all(_is_numeric(n) for n in names):
        return None
    number = {name: str(i + 1) for i, name in enumerate(names)}
    if len(number) != len(names):
        raise ValueError("GFA lists a segment name twice")

    def rename(name: str) -> str:
        try:
            return number[name]
        except KeyError:
            raise ValueError(f"GFA refers to segment {name!r}, which has no S line") from None

    out = []
    for line in lines:
        f = line.split("\t")
        kind = f[0]
        if kind == "S":
            f[1] = rename(f[1])
        elif kind in ("L", "C", "J"):
            f[1], f[3] = rename(f[1]), rename(f[3])
        elif kind == "P":
            f[2] = ",".join(rename(step[:-1]) + step[-1] for step in f[2].split(","))
        elif kind == "W":
            f[6] = "".join(o + rename(n) for o, n in _WALK_STEP.findall(f[6]))
        out.append("\t".join(f))
    Path(destination).write_text("\n".join(out) + "\n")
    return {v: k for k, v in number.items()}


def write_name_table(table: dict[str, str], path: str | Path) -> None:
    """``node_names.tsv``: ODGI node number and the segment name it stands for."""
    rows = sorted(table.items(), key=lambda kv: int(kv[0]))
    Path(path).write_text("node\tsegment\n" + "".join(f"{n}\t{s}\n" for n, s in rows))
