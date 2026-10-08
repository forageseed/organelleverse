from __future__ import annotations

from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleInputError
from organelleverse.quality_control.service import run_assembly_qc

from .test_service import _install_deterministic_collectors
from .test_static import _publish_assembly_evidence


def test_identical_run_reuses_verified_result(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    import organelleverse.quality_control.service as service

    calls = 0
    original = service.collect_mapping_evidence

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "collect_mapping_evidence", counted)
    first = run_assembly_qc(published.result)
    second = run_assembly_qc(published.result)

    assert calls == 1
    assert second == first


def test_tampered_reusable_report_fails_closed(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    published = _publish_assembly_evidence(source)
    _install_deterministic_collectors(monkeypatch, tmp_path)
    first = run_assembly_qc(published.result)
    report = next(item for item in first.artifacts if item.kind == "assembly_qc_report")
    Path(report.uri).write_text('{"tampered":true}\n')

    with pytest.raises(OrganelleInputError) as captured:
        run_assembly_qc(published.result)
    assert captured.value.code == "qc.destination_conflict"
