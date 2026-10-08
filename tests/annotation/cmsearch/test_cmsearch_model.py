from __future__ import annotations

import os
from pathlib import Path

import pytest

from organelleverse.annotation.cmsearch.model import parse_cm

# Benchmark model is external (GPL tRNAscan-SE CM), never shipped. Point the test
# at it with ORGANELLEVERSE_BENCHMARK_CM=/path/to/TRNAinf.cm; skipped otherwise.
_env = os.environ.get("ORGANELLEVERSE_BENCHMARK_CM")
CM_PATH = Path(_env) if _env else None
pytestmark = pytest.mark.skipif(
    CM_PATH is None or not CM_PATH.exists(),
    reason="set ORGANELLEVERSE_BENCHMARK_CM to a tRNAscan-SE TRNAinf.cm to run",
)

_EMIT = {"MP": 16, "ML": 4, "MR": 4, "IL": 4, "IR": 4, "S": 0, "D": 0, "B": 0, "E": 0, "EL": 0}


def test_parse_header_and_counts():
    cm = parse_cm(CM_PATH)
    assert cm.clen == 90
    assert len(cm.states) == 289
    assert len(cm.nodes) == 76


def test_token_accounting_holds_for_every_state():
    cm = parse_cm(CM_PATH)
    for s in cm.states:
        assert len(s.emissions) == _EMIT[s.type]
        assert len(s.transitions) == (0 if s.type == "B" else s.cnum)


def test_state_type_inventory():
    cm = parse_cm(CM_PATH)
    counts: dict[str, int] = {}
    for s in cm.states:
        counts[s.type] = counts.get(s.type, 0) + 1
    assert counts["MP"] == 28
    assert counts["B"] == 3
    assert counts["E"] == 4
    assert counts["S"] == 7
