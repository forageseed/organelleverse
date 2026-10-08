"""Managed output paths are not scientific parameters."""

from __future__ import annotations

from organelleverse.coevolution import run_orthofinder


def test_orthofinder_plan_identity_ignores_generated_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    inputs = tmp_path / "proteomes"
    inputs.mkdir()
    (inputs / "a.fasta").write_text(">a\nMAAA\n")

    first = run_orthofinder(inputs)
    second = run_orthofinder(inputs)

    assert first.provenance is not None
    assert second.provenance is not None
    assert first.provenance.parameters_hash == second.provenance.parameters_hash
    assert first.metrics["argv"] != second.metrics["argv"]
    assert not (tmp_path / "cache" / "runs").exists()
