"""Environment-controlled HiMT assembly QC release gate.

Consumes a real HiMT assembly ``Result`` (HiFi, CLR, or ONT fixture) and pipes
it through ``qc.assembly -> qc.write`` without re-running HiMT. Skips explicitly
when the real Result is not provided. Select with ``-m release_assembly_qc``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ._assembly_qc_gate import run_assembly_qc_release_gate

pytestmark = pytest.mark.release_assembly_qc


def test_himt_assembly_qc_release_gate(tmp_path: Path) -> None:
    run_assembly_qc_release_gate(
        tmp_path,
        backend="HiMT",
        expected_backend="himt",
        env_var="ORGANELLEVERSE_QC_HIMT_RESULT",
    )
