from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_legacy_qc_contract_is_absent() -> None:
    source = "\n".join(
        path.read_text(errors="ignore") for path in (ROOT / "src" / "organelleverse").rglob("*.py")
    )
    assert "quality_control.qc import _read_fasta" not in source
    assert "compute_qc_score" not in source
    assert "qc_score" not in source
    assert '("quality_control", "qc")' not in source
    assert '("qc", "qc")' not in source
    assert not (ROOT / "src/organelleverse/quality_control/qc.py").exists()
    assert not (ROOT / "src/organelleverse/quality_control/qc_core.py").exists()
