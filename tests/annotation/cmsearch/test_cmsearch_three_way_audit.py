from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

from tests._paths import PROJECT_ROOT

_REAL = os.environ.get("ORGANELLEVERSE_REAL_DATA_DIR")
_FASTA = Path(_REAL) / "PMGA/output_Ranunculus/results/Ranunculus.fasta" if _REAL else None
_GFF = Path(_REAL) / "PMGA/output_Ranunculus/results/Ranunculus.gff" if _REAL else None
_RF_CM = PROJECT_ROOT / "src/annotation/data/models/trna/trna.cm"

pytestmark = pytest.mark.skipif(
    shutil.which("cmsearch") is None
    or _FASTA is None
    or not _FASTA.exists()
    or _GFF is None
    or not _GFF.exists()
    or not _RF_CM.exists(),
    reason="needs cmsearch, ORGANELLEVERSE_REAL_DATA_DIR, and packaged RF00005 CM",
)


def _load():
    path = PROJECT_ROOT / "scripts/annotation_benchmarks/three_way_trna_audit.py"
    spec = importlib.util.spec_from_file_location("three_way_trna_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_native_meets_gold_standard_and_pmga_errors_are_pmga_errors():
    stats = _load().audit(_FASTA, _GFF, _RF_CM)

    # 1) The native engine reproduces the independent gold standard (Infernal).
    assert stats["infernal_reproduced_by_native"] == stats["infernal"]
    # 2) It finds at least as many tRNAs as Infernal (intron/divergent recovery).
    assert stats["native"] >= stats["infernal"]
    # 3) No native regression: every mature PMGA tRNA that native misses is also
    #    missed by Infernal (so the disagreement is PMGA-vs-CM, not native error).
    assert stats["pmga_mature_missed_by_native"] == stats["pmga_mature_missed_by_both"]
    # 4) PMGA oversized entries are artifacts: Infernal recognises none of them.
    assert stats["pmga_oversized_in_infernal"] == 0
