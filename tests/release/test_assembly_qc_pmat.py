"""Environment-controlled PMAT2 assembly QC release gate.

Consumes a real PMAT2 assembly ``Result`` (low-depth HiFi) and pipes it through
``qc.assembly -> qc.write`` without re-running PMAT2. Skips explicitly when the
real Result is not provided. In addition to the shared gate, this verifies that
policy mapping filters do not manufacture an artificial zero-coverage interval
in low-depth output (design §11, §13.3). Select with ``-m release_assembly_qc``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ._assembly_qc_gate import run_assembly_qc_release_gate

pytestmark = pytest.mark.release_assembly_qc


def test_pmat2_assembly_qc_release_gate(tmp_path: Path) -> None:
    run_assembly_qc_release_gate(
        tmp_path,
        backend="PMAT2",
        expected_backend="pmat",
        env_var="ORGANELLEVERSE_QC_PMAT_RESULT",
        require_zero_coverage_assessed=True,
    )
