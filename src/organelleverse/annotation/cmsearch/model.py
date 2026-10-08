"""Parser for Infernal 1.1 covariance-model (``.cm``) files.

The state-line format (verified against real tRNA CMs) is::

    <type> <v> <plast> <pnum> <cfirst> <cnum> <b0> <b1> <b2> <b3>
        <t_0..t_{cnum-1}> <e_0..e_{K-1}>

where ``b0..b3`` are QDB band integers (ignored by full DP), the transition
block has ``cnum`` log-odds (none for bifurcation ``B`` states), and the
emission block has ``K`` log-odds: 16 for ``MP``, 4 for ``ML/MR/IL/IR``, 0 for
``S/D/B/E/EL``. Token invariant: ``len(tokens) == 10 + cnum + K``.
"""

from __future__ import annotations

from pathlib import Path

from .models import CMNode, CMState, CovarianceModel

_EMIT = {"MP": 16, "ML": 4, "MR": 4, "IL": 4, "IR": 4}
_STATE_TYPES = {"S", "D", "MP", "ML", "MR", "IL", "IR", "B", "E", "EL"}


def _f(token: str) -> float:
    """Parse an Infernal score token; ``*`` denotes -infinity (impossible)."""
    return float("-inf") if token == "*" else float(token)


def _col(token: str) -> int:
    """Parse a node consensus-column token; ``-`` means no column."""
    return -1 if token == "-" else int(token)


def parse_cm(path: str | Path) -> CovarianceModel:
    name, clen = "", 0
    null = [0.0, 0.0, 0.0, 0.0]
    states: list[CMState] = []
    nodes: list[CMNode] = []
    cur_node = -1
    in_cm = False

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not in_cm:
            if line == "CM":
                in_cm = True
            elif line.startswith("NAME"):
                name = line.split(None, 1)[1].strip()
            elif line.startswith("CLEN"):
                clen = int(line.split()[1])
            elif line.startswith("NULL"):
                null = [float(x) for x in line.split()[1:5]]
            continue

        if line.startswith("[") and "]" in line:
            # e.g. "[ MATP 2 ] 3 298 g c g c" -> type, idx, lcol, rcol, ...
            after = line.split("]", 1)
            head = after[0].strip("[ ").split()
            ntype, nidx = head[0], int(head[1])
            tail = after[1].split() if len(after) > 1 else []
            lcol = _col(tail[0]) if len(tail) > 0 else -1
            rcol = _col(tail[1]) if len(tail) > 1 else -1
            cur_node = len(nodes)
            nodes.append(CMNode(ntype, nidx, lcol, rcol))
            continue
        if line == "//":
            break

        tok = line.split()
        if not tok or tok[0] not in _STATE_TYPES:
            continue

        stype = tok[0]
        v = int(tok[1])
        plast, pnum, cfirst, cnum = int(tok[2]), int(tok[3]), int(tok[4]), int(tok[5])
        rest = tok[10:]  # skip 4 QDB band integers at tok[6:10]
        n_trans = 0 if stype == "B" else cnum
        transitions = [_f(x) for x in rest[:n_trans]]
        emissions = [_f(x) for x in rest[n_trans : n_trans + _EMIT.get(stype, 0)]]
        states.append(
            CMState(stype, v, plast, pnum, cfirst, cnum, transitions, emissions, node=cur_node)
        )

    return CovarianceModel(name=name, clen=clen, states=states, nodes=nodes, null=null)
