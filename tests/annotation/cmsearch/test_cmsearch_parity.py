from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest

from tests._paths import PROJECT_ROOT


def _load_comparator():
    path = PROJECT_ROOT / "scripts/annotation_benchmarks/compare_cm_vs_cmsearch.py"
    spec = importlib.util.spec_from_file_location("compare_cm_vs_cmsearch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PACKAGED_CM = PROJECT_ROOT / "src/annotation/data/models/trna/trna.cm"
_REAL = os.environ.get("ORGANELLEVERSE_REAL_DATA_DIR")
_FASTA = Path(_REAL) / "PMGA/output_Ranunculus/results/Ranunculus.fasta" if _REAL else None

pytestmark = pytest.mark.skipif(
    shutil.which("cmsearch") is None
    or not _PACKAGED_CM.exists()
    or _FASTA is None
    or not _FASTA.exists(),
    reason="needs Infernal cmsearch, packaged CM, and ORGANELLEVERSE_REAL_DATA_DIR",
)


def test_native_cm_search_matches_cmsearch_coordinates_on_real_trnas():
    compare = _load_comparator().compare

    stats = compare(_PACKAGED_CM, _FASTA, evalue=1e-5, limit=15, pad=40, tol=3)
    # Native engine must reproduce Infernal coordinates on real tRNAs.
    assert stats["found"] == stats["loci"]
    assert stats["exact_frac"] >= 0.95
