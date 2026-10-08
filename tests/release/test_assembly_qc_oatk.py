"""Environment-controlled Oatk assembly QC release gate.

Consumes a real Oatk assembly ``Result`` and pipes it through
``qc.assembly -> qc.write`` without re-running Oatk. Skips explicitly when the
real Result is not provided. Select with ``-m release_assembly_qc``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ._assembly_qc_gate import run_assembly_qc_release_gate

pytestmark = pytest.mark.release_assembly_qc


def test_oatk_assembly_qc_release_gate(tmp_path: Path) -> None:
    run_assembly_qc_release_gate(
        tmp_path,
        backend="Oatk",
        expected_backend="oatk",
        env_var="ORGANELLEVERSE_QC_OATK_RESULT",
    )
